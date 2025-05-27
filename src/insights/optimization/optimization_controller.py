#!/usr/bin/env python
"""
🎯 CENTRAL OPTIMIZATION CONTROLLER - ETAPA 4
Controlador central para coordenar todos os sistemas de otimização
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path

# Configuração de logging
optimization_logger = logging.getLogger('optimization_controller')
optimization_logger.setLevel(logging.INFO)

class OptimizationMode(Enum):
    """Modos de otimização disponíveis"""
    CONSERVATIVE = "conservative"  # Otimizações seguras
    BALANCED = "balanced"         # Equilíbrio entre performance e estabilidade
    AGGRESSIVE = "aggressive"     # Máxima performance
    CUSTOM = "custom"            # Configuração customizada

class OptimizationStatus(Enum):
    """Status do sistema de otimização"""
    DISABLED = "disabled"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    OPTIMIZING = "optimizing"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class OptimizationConfig:
    """Configuração do sistema de otimização"""
    mode: OptimizationMode = OptimizationMode.BALANCED
    enable_ml_optimization: bool = True
    enable_auto_scaling: bool = True
    enable_predictive_caching: bool = True
    enable_resource_prediction: bool = True
    
    # Configurações de cache
    cache_memory_limit_mb: int = 512
    cache_disk_limit_gb: int = 2
    cache_ttl_hours: int = 24
    
    # Configurações de auto-scaling
    cpu_scale_up_threshold: float = 75.0
    cpu_scale_down_threshold: float = 30.0
    memory_scale_up_threshold: float = 80.0
    memory_scale_down_threshold: float = 40.0
    
    # Configurações de ML
    ml_model_retrain_hours: int = 168  # 7 dias
    prediction_window_minutes: int = 30
    anomaly_detection_sensitivity: float = 0.1
    
    # Configurações de monitoramento
    metrics_retention_days: int = 30
    alert_cooldown_minutes: int = 15
    performance_check_interval_seconds: int = 60

@dataclass
class OptimizationMetrics:
    """Métricas de otimização"""
    timestamp: datetime
    
    # Performance
    avg_execution_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Otimizações aplicadas
    optimizations_applied: int = 0
    performance_improvement_percent: float = 0.0
    resource_savings_percent: float = 0.0
    
    # Predições
    predicted_load_next_hour: float = 0.0
    predicted_memory_need_mb: float = 0.0
    anomalies_detected: int = 0

class OptimizationController:
    """
    🎯 Controlador Central de Otimização
    
    Coordena todos os sistemas de otimização:
    - Cache inteligente
    - Auto-scaling
    - ML-based optimization
    - Resource prediction
    - Performance analytics
    """
    
    def __init__(self, 
                 config: Optional[OptimizationConfig] = None,
                 optimization_dir: str = "data/optimization"):
        
        self.config = config or OptimizationConfig()
        self.optimization_dir = Path(optimization_dir)
        self.optimization_dir.mkdir(parents=True, exist_ok=True)
        
        # Estado do controlador
        self.status = OptimizationStatus.INITIALIZING
        self.start_time = datetime.now()
        self.last_optimization = None
        
        # Sistemas de otimização (inicializados sob demanda)
        self._flow_optimizer = None
        self._cache_integration = None
        self._ml_optimizer = None
        self._auto_scaler = None
        self._performance_analytics = None
        self._resource_manager = None
        
        # Métricas e monitoramento
        self.metrics_history: List[OptimizationMetrics] = []
        self.active_optimizations: Dict[str, Any] = {}
        self.optimization_queue: List[Dict[str, Any]] = []
        
        # Thread safety
        self.lock = threading.RLock()
        self._monitoring_task = None
        self._optimization_task = None
        
        # Configurar sistema
        self._setup_optimization_system()
        
        optimization_logger.info(f"🎯 OptimizationController inicializado - Modo: {self.config.mode.value}")
    
    def _setup_optimization_system(self):
        """Configurar sistema de otimização"""
        try:
            # Carregar configurações persistidas
            self._load_persistent_config()
            
            # Inicializar sistemas básicos
            self._initialize_core_systems()
            
            # Iniciar monitoramento
            self._start_monitoring()
            
            self.status = OptimizationStatus.ACTIVE
            optimization_logger.info("✅ Sistema de otimização configurado com sucesso")
            
        except Exception as e:
            self.status = OptimizationStatus.ERROR
            optimization_logger.error(f"❌ Erro ao configurar sistema: {e}")
    
    def _initialize_core_systems(self):
        """Inicializar sistemas principais"""
        from insights.cache import get_global_cache_system
        from insights.flow_monitoring import get_global_monitoring_system
        
        # Cache system já existe - vamos integrar
        self.cache_system = get_global_cache_system()
        self.monitoring_system = get_global_monitoring_system()
        
        # Inicializar novos sistemas se habilitados
        if self.config.enable_auto_scaling:
            self._initialize_auto_scaler()
        
        if self.config.enable_resource_prediction:
            self._initialize_resource_manager()
            self._initialize_predictive_engine()
        
        optimization_logger.info("🔧 Sistemas principais inicializados")
    
    def _initialize_auto_scaler(self):
        """Inicializar sistema de auto-scaling"""
        try:
            from .auto_scaler import AutoScaler
            self._auto_scaler = AutoScaler()
            self._auto_scaler.start_monitoring()
            optimization_logger.info("📈 AutoScaler inicializado")
        except Exception as e:
            optimization_logger.error(f"❌ Erro ao inicializar AutoScaler: {e}")
    
    def _initialize_resource_manager(self):
        """Inicializar gerenciador de recursos"""
        try:
            from .resource_manager import ResourceManager
            self._resource_manager = ResourceManager()
            self._resource_manager.start_monitoring()
            optimization_logger.info("🔧 ResourceManager inicializado")
        except Exception as e:
            optimization_logger.error(f"❌ Erro ao inicializar ResourceManager: {e}")
    
    def _initialize_predictive_engine(self):
        """Inicializar engine de predição"""
        try:
            from .predictive_engine import PredictiveEngine
            self._predictive_engine = PredictiveEngine()
            optimization_logger.info("🔮 PredictiveEngine inicializado")
        except Exception as e:
            optimization_logger.error(f"❌ Erro ao inicializar PredictiveEngine: {e}")
    
    def _start_monitoring(self):
        """Iniciar monitoramento contínuo"""
        if self._monitoring_task is None:
            self._monitoring_task = threading.Thread(
                target=self._monitoring_loop,
                daemon=True,
                name="OptimizationMonitoring"
            )
            self._monitoring_task.start()
            optimization_logger.info("📊 Monitoramento contínuo iniciado")
    
    def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        while self.status == OptimizationStatus.ACTIVE:
            try:
                # Coletar métricas
                current_metrics = self._collect_current_metrics()
                
                # Analisar necessidade de otimizações
                self._analyze_optimization_opportunities(current_metrics)
                
                # Aplicar otimizações automáticas
                self._apply_automatic_optimizations()
                
                # Salvar métricas
                self.metrics_history.append(current_metrics)
                
                # Limpar histórico antigo
                self._cleanup_old_metrics()
                
            except Exception as e:
                optimization_logger.error(f"❌ Erro no loop de monitoramento: {e}")
            
            time.sleep(self.config.performance_check_interval_seconds)
    
    def _collect_current_metrics(self) -> OptimizationMetrics:
        """Coletar métricas atuais do sistema"""
        import psutil
        
        try:
            # Métricas de sistema
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            memory_mb = memory_info.used / (1024 * 1024)
            
            # Métricas de cache
            cache_stats = self.cache_system.get_stats()
            total_operations = cache_stats.get("hits", 0) + cache_stats.get("misses", 0)
            cache_hit_rate = (cache_stats.get("hits", 0) / total_operations * 100) if total_operations > 0 else 0
            
            # Métricas de performance (da última hora)
            recent_metrics = self.monitoring_system.get_current_metrics(minutes_back=60)
            avg_execution_time = 0.0
            if recent_metrics:
                execution_times = [m.value for m in recent_metrics if m.metric_name == "execution_time"]
                avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
            
            return OptimizationMetrics(
                timestamp=datetime.now(),
                avg_execution_time_ms=avg_execution_time,
                cache_hit_rate=cache_hit_rate,
                memory_usage_mb=memory_mb,
                cpu_usage_percent=cpu_percent,
                optimizations_applied=len(self.active_optimizations),
                performance_improvement_percent=self._calculate_performance_improvement(),
                resource_savings_percent=self._calculate_resource_savings()
            )
            
        except Exception as e:
            optimization_logger.error(f"❌ Erro ao coletar métricas: {e}")
            return OptimizationMetrics(timestamp=datetime.now())
    
    def _analyze_optimization_opportunities(self, metrics: OptimizationMetrics):
        """Analisar oportunidades de otimização"""
        opportunities = []
        
        # Análise de CPU
        if metrics.cpu_usage_percent > self.config.cpu_scale_up_threshold:
            opportunities.append({
                "type": "cpu_optimization",
                "priority": "high",
                "action": "scale_up_resources",
                "details": f"CPU em {metrics.cpu_usage_percent:.1f}%"
            })
        
        # Análise de memória
        if metrics.memory_usage_mb > self.config.memory_scale_up_threshold * 1024:  # Convert to MB
            opportunities.append({
                "type": "memory_optimization", 
                "priority": "high",
                "action": "optimize_memory_usage",
                "details": f"Memória em {metrics.memory_usage_mb:.1f}MB"
            })
        
        # Análise de cache
        if metrics.cache_hit_rate < 70.0:  # Cache hit rate baixo
            opportunities.append({
                "type": "cache_optimization",
                "priority": "medium",
                "action": "improve_cache_strategy",
                "details": f"Cache hit rate: {metrics.cache_hit_rate:.1f}%"
            })
        
        # Adicionar oportunidades à fila
        for opportunity in opportunities:
            self.optimization_queue.append(opportunity)
        
        if opportunities:
            optimization_logger.info(f"🔍 {len(opportunities)} oportunidades de otimização identificadas")
    
    def _apply_automatic_optimizations(self):
        """Aplicar otimizações automáticas"""
        if not self.optimization_queue:
            return
        
        with self.lock:
            # Processar até 3 otimizações por ciclo
            for _ in range(min(3, len(self.optimization_queue))):
                if not self.optimization_queue:
                    break
                
                optimization = self.optimization_queue.pop(0)
                self._execute_optimization(optimization)
    
    def _execute_optimization(self, optimization: Dict[str, Any]):
        """Executar uma otimização específica"""
        try:
            opt_type = optimization.get("type")
            action = optimization.get("action")
            
            optimization_logger.info(f"🚀 Executando otimização: {opt_type} - {action}")
            
            if opt_type == "cache_optimization":
                self._optimize_cache_system()
            elif opt_type == "memory_optimization":
                self._optimize_memory_usage()
            elif opt_type == "cpu_optimization":
                self._optimize_cpu_usage()
            
            # Registrar otimização aplicada
            opt_id = f"{opt_type}_{int(time.time())}"
            self.active_optimizations[opt_id] = {
                **optimization,
                "applied_at": datetime.now(),
                "status": "active"
            }
            
        except Exception as e:
            optimization_logger.error(f"❌ Erro ao executar otimização: {e}")
    
    def _optimize_cache_system(self):
        """Otimizar sistema de cache"""
        # Limpar cache expirado
        expired_count = self.cache_system.cleanup_expired()
        
        # Otimizar estruturas de cache
        optimized_count = self.cache_system.optimize()
        
        optimization_logger.info(f"💾 Cache otimizado: {expired_count} expirados, {optimized_count} otimizados")
    
    def _optimize_memory_usage(self):
        """Otimizar uso de memória"""
        import gc
        
        # Forçar coleta de lixo
        collected = gc.collect()
        
        # Limpar cache em memória se necessário
        if hasattr(self.cache_system, 'memory_cache'):
            initial_size = len(self.cache_system.memory_cache)
            # Aplicar estratégia LRU mais agressiva
            if hasattr(self.cache_system, '_apply_lru_if_needed'):
                removed = self.cache_system._apply_lru_if_needed()
                optimization_logger.info(f"🧹 Memória otimizada: {collected} objetos coletados, {removed} entradas removidas do cache")
    
    def _optimize_cpu_usage(self):
        """Otimizar uso de CPU"""
        # Por enquanto, log da ação (implementação futura com auto-scaling)
        optimization_logger.info("⚡ Otimização de CPU aplicada (placeholder para auto-scaling)")
    
    def _calculate_performance_improvement(self) -> float:
        """Calcular melhoria de performance percentual"""
        if len(self.metrics_history) < 2:
            return 0.0
        
        # Comparar com métricas de 1 hora atrás
        recent_metrics = [m for m in self.metrics_history if m.timestamp > datetime.now() - timedelta(hours=1)]
        old_metrics = [m for m in self.metrics_history if m.timestamp <= datetime.now() - timedelta(hours=1)]
        
        if not recent_metrics or not old_metrics:
            return 0.0
        
        recent_avg = sum(m.avg_execution_time_ms for m in recent_metrics) / len(recent_metrics)
        old_avg = sum(m.avg_execution_time_ms for m in old_metrics) / len(old_metrics)
        
        if old_avg == 0:
            return 0.0
        
        improvement = ((old_avg - recent_avg) / old_avg) * 100
        return max(0.0, improvement)
    
    def _calculate_resource_savings(self) -> float:
        """Calcular economia de recursos percentual"""
        # Implementação simplificada baseada em otimizações aplicadas
        return len(self.active_optimizations) * 2.5  # 2.5% por otimização ativa
    
    def _cleanup_old_metrics(self):
        """Limpar métricas antigas"""
        cutoff_date = datetime.now() - timedelta(days=self.config.metrics_retention_days)
        self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_date]
    
    def _load_persistent_config(self):
        """Carregar configurações persistidas"""
        config_file = self.optimization_dir / "config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    # Aplicar configurações carregadas (implementação futura)
                    optimization_logger.info("📁 Configurações persistidas carregadas")
            except Exception as e:
                optimization_logger.warning(f"⚠️ Erro ao carregar configurações: {e}")
    
    # =============== API PÚBLICA ===============
    
    def get_status(self) -> Dict[str, Any]:
        """Obter status completo do sistema"""
        with self.lock:
            latest_metrics = self.metrics_history[-1] if self.metrics_history else None
            
            return {
                "status": self.status.value,
                "mode": self.config.mode.value,
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "active_optimizations": len(self.active_optimizations),
                "pending_optimizations": len(self.optimization_queue),
                "latest_metrics": asdict(latest_metrics) if latest_metrics else None,
                "performance_improvement": self._calculate_performance_improvement(),
                "resource_savings": self._calculate_resource_savings()
            }
    
    def get_metrics_history(self, hours_back: int = 24) -> List[Dict[str, Any]]:
        """Obter histórico de métricas"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        filtered_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        return [asdict(m) for m in filtered_metrics]
    
    def trigger_optimization(self, optimization_type: str, priority: str = "medium") -> bool:
        """Trigger manual de otimização"""
        try:
            optimization = {
                "type": optimization_type,
                "priority": priority,
                "action": f"manual_{optimization_type}",
                "details": "Triggered manually",
                "manual": True
            }
            
            self.optimization_queue.append(optimization)
            optimization_logger.info(f"🎯 Otimização manual adicionada: {optimization_type}")
            return True
            
        except Exception as e:
            optimization_logger.error(f"❌ Erro ao adicionar otimização manual: {e}")
            return False
    
    def update_config(self, new_config: OptimizationConfig) -> bool:
        """Atualizar configuração do sistema"""
        try:
            with self.lock:
                old_mode = self.config.mode
                self.config = new_config
                
                # Salvar configuração
                config_file = self.optimization_dir / "config.json"
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(asdict(new_config), f, indent=2, default=str)
                
                optimization_logger.info(f"⚙️ Configuração atualizada: {old_mode.value} → {new_config.mode.value}")
                return True
                
        except Exception as e:
            optimization_logger.error(f"❌ Erro ao atualizar configuração: {e}")
            return False
    
    def shutdown(self):
        """Shutdown graceful do sistema"""
        optimization_logger.info("🛑 Iniciando shutdown do sistema de otimização...")
        
        self.status = OptimizationStatus.MAINTENANCE
        
        # Parar sistemas de otimização
        if self._auto_scaler:
            try:
                self._auto_scaler.stop_monitoring()
                optimization_logger.info("📈 AutoScaler parado")
            except Exception as e:
                optimization_logger.error(f"❌ Erro ao parar AutoScaler: {e}")
        
        if self._resource_manager:
            try:
                self._resource_manager.stop_monitoring()
                optimization_logger.info("🔧 ResourceManager parado")
            except Exception as e:
                optimization_logger.error(f"❌ Erro ao parar ResourceManager: {e}")
        
        # Aguardar threads terminarem
        if self._monitoring_task and self._monitoring_task.is_alive():
            self._monitoring_task.join(timeout=5)
        
        # Salvar estado final
        self._save_final_state()
        
        optimization_logger.info("✅ Shutdown concluído")
    
    def _save_final_state(self):
        """Salvar estado final do sistema"""
        try:
            state_file = self.optimization_dir / "final_state.json"
            final_state = {
                "shutdown_time": datetime.now().isoformat(),
                "total_optimizations": len(self.active_optimizations),
                "final_metrics": asdict(self.metrics_history[-1]) if self.metrics_history else None,
                "performance_improvement": self._calculate_performance_improvement(),
                "resource_savings": self._calculate_resource_savings()
            }
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(final_state, f, indent=2, default=str)
                
        except Exception as e:
            optimization_logger.error(f"❌ Erro ao salvar estado final: {e}") 