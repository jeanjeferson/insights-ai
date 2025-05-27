#!/usr/bin/env python
"""
🚀 INSIGHTS-AI AUTO SCALER
Sistema de auto-scaling inteligente para otimização de recursos

Características:
- Monitoramento contínuo de recursos
- Scaling baseado em métricas em tempo real
- Predições de carga futuras
- Otimização de custos
- Balanceamento automático
"""

import time
import logging
import threading
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import os

# Configurar logging
auto_scaler_logger = logging.getLogger(__name__)
auto_scaler_logger.setLevel(logging.INFO)

class ScalingAction(Enum):
    """Ações de scaling disponíveis"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    OPTIMIZE = "optimize"

class ResourceType(Enum):
    """Tipos de recursos monitorados"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    THREADS = "threads"
    CONNECTIONS = "connections"

@dataclass
class ScalingTrigger:
    """Trigger para ações de scaling"""
    resource_type: ResourceType
    threshold_up: float
    threshold_down: float
    duration_seconds: int
    cooldown_seconds: int
    priority: str
    
@dataclass
class ResourceMetrics:
    """Métricas de recursos do sistema"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_io_percent: float
    network_io_mbps: float
    active_threads: int
    active_connections: int
    
@dataclass
class ScalingDecision:
    """Decisão de scaling"""
    action: ScalingAction
    resource_type: ResourceType
    current_value: float
    target_value: float
    confidence: float
    reasoning: str
    estimated_impact: Dict[str, float]

class AutoScaler:
    """Sistema de auto-scaling inteligente"""
    
    def __init__(self, 
                 config_file: str = "data/optimization/auto_scaler_config.json",
                 enable_predictive: bool = True):
        
        self.config_file = config_file
        self.enable_predictive = enable_predictive
        
        # Estado do sistema
        self.is_active = False
        self.start_time = None
        self.last_scaling_action = None
        self.scaling_history: List[Dict[str, Any]] = []
        
        # Métricas e monitoramento
        self.metrics_history: List[ResourceMetrics] = []
        self.current_load: Dict[str, float] = {}
        
        # Configurações de scaling
        self.scaling_triggers = self._initialize_scaling_triggers()
        self.cooldown_periods: Dict[str, datetime] = {}
        
        # Threading
        self.monitoring_thread = None
        self.prediction_thread = None
        self.lock = threading.RLock()
        
        # Configurar sistema
        self._setup_auto_scaler()
        
        auto_scaler_logger.info("🎯 AutoScaler inicializado")
    
    def _initialize_scaling_triggers(self) -> List[ScalingTrigger]:
        """Inicializar triggers de scaling"""
        return [
            # CPU Triggers
            ScalingTrigger(
                resource_type=ResourceType.CPU,
                threshold_up=75.0,
                threshold_down=30.0,
                duration_seconds=120,
                cooldown_seconds=300,
                priority="high"
            ),
            
            # Memory Triggers
            ScalingTrigger(
                resource_type=ResourceType.MEMORY,
                threshold_up=80.0,
                threshold_down=40.0,
                duration_seconds=60,
                cooldown_seconds=180,
                priority="high"
            ),
            
            # Disk I/O Triggers
            ScalingTrigger(
                resource_type=ResourceType.DISK_IO,
                threshold_up=85.0,
                threshold_down=25.0,
                duration_seconds=180,
                cooldown_seconds=240,
                priority="medium"
            ),
            
            # Network I/O Triggers
            ScalingTrigger(
                resource_type=ResourceType.NETWORK_IO,
                threshold_up=70.0,
                threshold_down=20.0,
                duration_seconds=90,
                cooldown_seconds=150,
                priority="medium"
            )
        ]
    
    def _setup_auto_scaler(self):
        """Configurar sistema de auto-scaling"""
        try:
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            # Carregar configurações salvas
            self._load_configuration()
            
            auto_scaler_logger.info("✅ AutoScaler configurado com sucesso")
            
        except Exception as e:
            auto_scaler_logger.error(f"❌ Erro ao configurar AutoScaler: {e}")
    
    def start_monitoring(self):
        """Iniciar monitoramento de recursos"""
        if self.is_active:
            return
        
        self.is_active = True
        self.start_time = datetime.now()
        
        # Thread de monitoramento principal
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="AutoScalerMonitoring"
        )
        self.monitoring_thread.start()
        
        # Thread de predições (se habilitado)
        if self.enable_predictive:
            self.prediction_thread = threading.Thread(
                target=self._prediction_loop,
                daemon=True,
                name="AutoScalerPrediction"
            )
            self.prediction_thread.start()
        
        auto_scaler_logger.info("🚀 Monitoramento de auto-scaling iniciado")
    
    def stop_monitoring(self):
        """Parar monitoramento"""
        self.is_active = False
        auto_scaler_logger.info("⏹️ Monitoramento de auto-scaling parado")
    
    def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        while self.is_active:
            try:
                # Coletar métricas atuais
                current_metrics = self._collect_resource_metrics()
                
                # Analisar necessidade de scaling
                scaling_decisions = self._analyze_scaling_needs(current_metrics)
                
                # Executar ações de scaling
                for decision in scaling_decisions:
                    self._execute_scaling_decision(decision)
                
                # Salvar métricas
                with self.lock:
                    self.metrics_history.append(current_metrics)
                    self._cleanup_old_metrics()
                
            except Exception as e:
                auto_scaler_logger.error(f"❌ Erro no loop de monitoramento: {e}")
            
            time.sleep(10)  # Verificar a cada 10 segundos
    
    def _prediction_loop(self):
        """Loop de predições de carga"""
        while self.is_active:
            try:
                if len(self.metrics_history) >= 10:  # Precisamos de histórico
                    # Fazer predições de carga
                    predictions = self._predict_future_load()
                    
                    # Preparar scaling proativo
                    proactive_decisions = self._plan_proactive_scaling(predictions)
                    
                    # Executar scaling proativo (se necessário)
                    for decision in proactive_decisions:
                        if decision.confidence > 0.8:  # Alta confiança
                            self._execute_scaling_decision(decision)
                
            except Exception as e:
                auto_scaler_logger.error(f"❌ Erro no loop de predição: {e}")
            
            time.sleep(60)  # Predições a cada minuto
    
    def _collect_resource_metrics(self) -> ResourceMetrics:
        """Coletar métricas de recursos do sistema"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_percent = 0.0  # Calcular baseado em throughput
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_io_mbps = 0.0  # Calcular baseado em throughput
            
            # Threads e conexões
            active_threads = threading.active_count()
            active_connections = len(psutil.net_connections())
            
            return ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_mb=memory_available_mb,
                disk_io_percent=disk_io_percent,
                network_io_mbps=network_io_mbps,
                active_threads=active_threads,
                active_connections=active_connections
            )
            
        except Exception as e:
            auto_scaler_logger.error(f"❌ Erro ao coletar métricas: {e}")
            return ResourceMetrics(timestamp=datetime.now())
    
    def _analyze_scaling_needs(self, metrics: ResourceMetrics) -> List[ScalingDecision]:
        """Analisar necessidade de scaling"""
        decisions = []
        
        for trigger in self.scaling_triggers:
            # Verificar cooldown
            if self._is_in_cooldown(trigger.resource_type):
                continue
            
            # Obter valor atual do recurso
            current_value = self._get_resource_value(metrics, trigger.resource_type)
            
            # Determinar ação necessária
            action = self._determine_scaling_action(current_value, trigger)
            
            if action != ScalingAction.MAINTAIN:
                # Calcular confiança da decisão
                confidence = self._calculate_decision_confidence(
                    current_value, trigger, metrics
                )
                
                # Estimar impacto
                estimated_impact = self._estimate_scaling_impact(action, trigger)
                
                decision = ScalingDecision(
                    action=action,
                    resource_type=trigger.resource_type,
                    current_value=current_value,
                    target_value=self._calculate_target_value(action, trigger),
                    confidence=confidence,
                    reasoning=self._generate_scaling_reasoning(action, trigger, current_value),
                    estimated_impact=estimated_impact
                )
                
                decisions.append(decision)
        
        return decisions
    
    def _execute_scaling_decision(self, decision: ScalingDecision):
        """Executar decisão de scaling"""
        try:
            auto_scaler_logger.info(
                f"🎯 Executando scaling: {decision.action.value} para "
                f"{decision.resource_type.value} (confiança: {decision.confidence:.2f})"
            )
            
            # Registrar ação
            scaling_record = {
                "timestamp": datetime.now().isoformat(),
                "action": decision.action.value,
                "resource_type": decision.resource_type.value,
                "current_value": decision.current_value,
                "target_value": decision.target_value,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning
            }
            
            with self.lock:
                self.scaling_history.append(scaling_record)
                self.cooldown_periods[decision.resource_type.value] = datetime.now()
            
            # Executar ação específica
            if decision.action == ScalingAction.SCALE_UP:
                self._scale_up_resource(decision)
            elif decision.action == ScalingAction.SCALE_DOWN:
                self._scale_down_resource(decision)
            elif decision.action == ScalingAction.OPTIMIZE:
                self._optimize_resource(decision)
            
            auto_scaler_logger.info(f"✅ Scaling executado: {decision.reasoning}")
            
        except Exception as e:
            auto_scaler_logger.error(f"❌ Erro ao executar scaling: {e}")
    
    def _scale_up_resource(self, decision: ScalingDecision):
        """Fazer scale up de recurso"""
        resource_type = decision.resource_type
        
        if resource_type == ResourceType.CPU:
            # Otimizar uso de CPU
            self._optimize_cpu_usage()
        elif resource_type == ResourceType.MEMORY:
            # Otimizar uso de memória
            self._optimize_memory_usage()
        elif resource_type == ResourceType.THREADS:
            # Aumentar pool de threads se possível
            self._increase_thread_pool()
    
    def _scale_down_resource(self, decision: ScalingDecision):
        """Fazer scale down de recurso"""
        resource_type = decision.resource_type
        
        if resource_type == ResourceType.CPU:
            # Reduzir uso de CPU
            self._reduce_cpu_usage()
        elif resource_type == ResourceType.MEMORY:
            # Liberar memória
            self._free_memory()
        elif resource_type == ResourceType.THREADS:
            # Reduzir pool de threads
            self._reduce_thread_pool()
    
    def _optimize_resource(self, decision: ScalingDecision):
        """Otimizar uso de recurso"""
        # Implementar otimizações específicas
        auto_scaler_logger.info(f"🔧 Otimizando {decision.resource_type.value}")
    
    def _predict_future_load(self) -> Dict[str, float]:
        """Predizer carga futura baseada no histórico"""
        predictions = {}
        
        if len(self.metrics_history) < 10:
            return predictions
        
        # Análise simples de tendências (pode ser expandida com ML)
        recent_metrics = self.metrics_history[-10:]
        
        # Calcular tendências para cada recurso
        cpu_trend = self._calculate_trend([m.cpu_percent for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_percent for m in recent_metrics])
        
        # Predições para próximos 30 minutos
        current_cpu = recent_metrics[-1].cpu_percent
        current_memory = recent_metrics[-1].memory_percent
        
        predictions["cpu_30min"] = min(100.0, max(0.0, current_cpu + (cpu_trend * 30)))
        predictions["memory_30min"] = min(100.0, max(0.0, current_memory + (memory_trend * 30)))
        
        return predictions
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calcular tendência simples"""
        if len(values) < 2:
            return 0.0
        
        # Regressão linear simples
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i ** 2 for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum ** 2)
        return slope
    
    def get_current_status(self) -> Dict[str, Any]:
        """Obter status atual do auto-scaler"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        latest_metrics = self.metrics_history[-1]
        
        return {
            "is_active": self.is_active,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "current_metrics": {
                "cpu_percent": latest_metrics.cpu_percent,
                "memory_percent": latest_metrics.memory_percent,
                "memory_available_mb": latest_metrics.memory_available_mb,
                "active_threads": latest_metrics.active_threads,
                "active_connections": latest_metrics.active_connections
            },
            "scaling_actions_today": len([
                s for s in self.scaling_history 
                if datetime.fromisoformat(s["timestamp"]).date() == datetime.now().date()
            ]),
            "last_scaling_action": self.scaling_history[-1] if self.scaling_history else None
        }
    
    def get_scaling_history(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Obter histórico de scaling"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        return [
            s for s in self.scaling_history
            if datetime.fromisoformat(s["timestamp"]) >= cutoff_time
        ]
    
    # Métodos auxiliares
    def _is_in_cooldown(self, resource_type: ResourceType) -> bool:
        """Verificar se recurso está em cooldown"""
        cooldown_time = self.cooldown_periods.get(resource_type.value)
        if not cooldown_time:
            return False
        
        # Encontrar trigger para este recurso
        trigger = next((t for t in self.scaling_triggers if t.resource_type == resource_type), None)
        if not trigger:
            return False
        
        return (datetime.now() - cooldown_time).total_seconds() < trigger.cooldown_seconds
    
    def _get_resource_value(self, metrics: ResourceMetrics, resource_type: ResourceType) -> float:
        """Obter valor atual do recurso"""
        if resource_type == ResourceType.CPU:
            return metrics.cpu_percent
        elif resource_type == ResourceType.MEMORY:
            return metrics.memory_percent
        elif resource_type == ResourceType.DISK_IO:
            return metrics.disk_io_percent
        elif resource_type == ResourceType.NETWORK_IO:
            return metrics.network_io_mbps
        return 0.0
    
    def _determine_scaling_action(self, current_value: float, trigger: ScalingTrigger) -> ScalingAction:
        """Determinar ação de scaling necessária"""
        if current_value >= trigger.threshold_up:
            return ScalingAction.SCALE_UP
        elif current_value <= trigger.threshold_down:
            return ScalingAction.SCALE_DOWN
        return ScalingAction.MAINTAIN
    
    def _calculate_decision_confidence(self, current_value: float, trigger: ScalingTrigger, metrics: ResourceMetrics) -> float:
        """Calcular confiança da decisão"""
        # Lógica simples - pode ser expandida
        if current_value >= trigger.threshold_up * 1.2:  # 20% acima do threshold
            return 0.9
        elif current_value >= trigger.threshold_up:
            return 0.7
        elif current_value <= trigger.threshold_down * 0.8:  # 20% abaixo do threshold
            return 0.8
        return 0.5
    
    def _calculate_target_value(self, action: ScalingAction, trigger: ScalingTrigger) -> float:
        """Calcular valor alvo após scaling"""
        if action == ScalingAction.SCALE_UP:
            return trigger.threshold_down + 10  # Margem de segurança
        elif action == ScalingAction.SCALE_DOWN:
            return trigger.threshold_up - 10   # Margem de segurança
        return trigger.threshold_up
    
    def _generate_scaling_reasoning(self, action: ScalingAction, trigger: ScalingTrigger, current_value: float) -> str:
        """Gerar explicação para a decisão de scaling"""
        resource_name = trigger.resource_type.value.upper()
        
        if action == ScalingAction.SCALE_UP:
            return f"{resource_name} em {current_value:.1f}% (acima de {trigger.threshold_up}%) - scaling up necessário"
        elif action == ScalingAction.SCALE_DOWN:
            return f"{resource_name} em {current_value:.1f}% (abaixo de {trigger.threshold_down}%) - scaling down possível"
        return f"{resource_name} estável em {current_value:.1f}%"
    
    def _estimate_scaling_impact(self, action: ScalingAction, trigger: ScalingTrigger) -> Dict[str, float]:
        """Estimar impacto do scaling"""
        return {
            "performance_improvement": 0.15 if action == ScalingAction.SCALE_UP else -0.05,
            "resource_cost_change": 0.10 if action == ScalingAction.SCALE_UP else -0.10,
            "stability_improvement": 0.20 if action != ScalingAction.MAINTAIN else 0.0
        }
    
    def _plan_proactive_scaling(self, predictions: Dict[str, float]) -> List[ScalingDecision]:
        """Planejar scaling proativo baseado em predições"""
        decisions = []
        
        # Se CPU projetada > 80% nos próximos 30min, fazer scale up preventivo
        if predictions.get("cpu_30min", 0) > 80:
            decisions.append(ScalingDecision(
                action=ScalingAction.SCALE_UP,
                resource_type=ResourceType.CPU,
                current_value=predictions["cpu_30min"],
                target_value=65.0,
                confidence=0.75,
                reasoning="Scaling proativo - CPU alta prevista em 30min",
                estimated_impact={"performance_improvement": 0.20}
            ))
        
        return decisions
    
    def _optimize_cpu_usage(self):
        """Otimizar uso de CPU"""
        auto_scaler_logger.info("🚀 Otimizando uso de CPU")
    
    def _optimize_memory_usage(self):
        """Otimizar uso de memória"""
        auto_scaler_logger.info("🚀 Otimizando uso de memória")
        import gc
        gc.collect()
    
    def _increase_thread_pool(self):
        """Aumentar pool de threads"""
        auto_scaler_logger.info("🚀 Aumentando pool de threads")
    
    def _reduce_cpu_usage(self):
        """Reduzir uso de CPU"""
        auto_scaler_logger.info("📉 Reduzindo uso de CPU")
    
    def _free_memory(self):
        """Liberar memória"""
        auto_scaler_logger.info("📉 Liberando memória")
        import gc
        gc.collect()
    
    def _reduce_thread_pool(self):
        """Reduzir pool de threads"""
        auto_scaler_logger.info("📉 Reduzindo pool de threads")
    
    def _cleanup_old_metrics(self):
        """Limpar métricas antigas"""
        # Manter apenas 1000 métricas mais recentes
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-500:]
    
    def _load_configuration(self):
        """Carregar configuração salva"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    auto_scaler_logger.info("✅ Configuração carregada")
        except Exception as e:
            auto_scaler_logger.warning(f"⚠️ Não foi possível carregar configuração: {e}")
    
    def _save_configuration(self):
        """Salvar configuração atual"""
        try:
            config = {
                "scaling_history_count": len(self.scaling_history),
                "last_save": datetime.now().isoformat()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
        except Exception as e:
            auto_scaler_logger.error(f"❌ Erro ao salvar configuração: {e}") 