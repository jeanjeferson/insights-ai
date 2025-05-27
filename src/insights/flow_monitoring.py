"""
Sistema de Monitoramento em Tempo Real para CrewAI Flow
Implementa monitoramento avançado, métricas de performance e alertas
"""

import json
import time
import threading
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import deque, defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configurar logger específico para monitoramento
monitoring_logger = logging.getLogger('flow_monitoring')
monitoring_logger.setLevel(logging.INFO)

class AlertLevel(Enum):
    """Níveis de alerta do sistema"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class MetricType(Enum):
    """Tipos de métricas coletadas"""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    BUSINESS = "business"
    HEALTH = "health"
    CUSTOM = "custom"

@dataclass
class MetricData:
    """Dados de uma métrica"""
    timestamp: str
    metric_name: str
    metric_type: MetricType
    value: float
    unit: str
    flow_id: str
    context: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['metric_type'] = self.metric_type.value
        return data

@dataclass
class Alert:
    """Alerta do sistema"""
    timestamp: str
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    flow_id: str
    metric_name: str = ""
    threshold_value: float = 0.0
    current_value: float = 0.0
    context: Dict[str, Any] = None
    resolved: bool = False
    resolved_timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['level'] = self.level.value
        return data

@dataclass
class HealthCheck:
    """Verificação de saúde do sistema"""
    timestamp: str
    flow_id: str
    component: str
    status: str  # "healthy", "warning", "error"
    response_time: float
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class FlowMonitoringSystem:
    """Sistema principal de monitoramento em tempo real"""
    
    def __init__(self, base_path: str = "logs/monitoring"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Configurações
        self.collection_interval = 5  # 5 segundos
        self.retention_hours = 24    # 24 horas de histórico
        self.alert_cooldown = 300    # 5 minutos entre alertas similares
        
        # Estado interno
        self.monitoring_active = False
        self.monitoring_thread = None
        self.current_flows: Dict[str, Any] = {}
        
        # Dados em memória
        self.metrics_buffer: deque = deque(maxlen=10000)  # Buffer de métricas
        self.alerts_history: List[Alert] = []
        self.health_checks: Dict[str, HealthCheck] = {}
        self.alert_cooldowns: Dict[str, float] = {}
        
        # Configurações de alertas
        self.alert_thresholds = {
            "cpu_usage": {"warning": 70.0, "critical": 90.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0},
            "disk_usage": {"warning": 85.0, "critical": 95.0},
            "execution_time": {"warning": 300.0, "critical": 600.0},  # segundos
            "error_rate": {"warning": 5.0, "critical": 10.0},  # percentage
            "response_time": {"warning": 30.0, "critical": 60.0}  # segundos
        }
        
        # Callbacks para alertas
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Executor para operações assíncronas
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        monitoring_logger.info("📊 FlowMonitoringSystem inicializado")
    
    def start_monitoring(self, flows: Dict[str, Any] = None):
        """Iniciar monitoramento em tempo real"""
        if flows:
            self.current_flows.update(flows)
        
        self.monitoring_active = True
        
        # Thread principal de monitoramento
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        monitoring_logger.info("🔍 Monitoramento em tempo real iniciado")
    
    def stop_monitoring(self):
        """Parar monitoramento"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=10)
        
        monitoring_logger.info("🛑 Monitoramento parado")
    
    def register_flow(self, flow_id: str, flow_instance: Any):
        """Registrar Flow para monitoramento"""
        self.current_flows[flow_id] = {
            "instance": flow_instance,
            "start_time": time.time(),
            "last_activity": time.time(),
            "metrics_count": 0
        }
        
        monitoring_logger.info(f"📝 Flow registrado para monitoramento: {flow_id}")
    
    def unregister_flow(self, flow_id: str):
        """Remover Flow do monitoramento"""
        if flow_id in self.current_flows:
            del self.current_flows[flow_id]
            monitoring_logger.info(f"❌ Flow removido do monitoramento: {flow_id}")
    
    def collect_metric(self, flow_id: str, metric_name: str, value: float, 
                      metric_type: MetricType = MetricType.CUSTOM, 
                      unit: str = "", context: Dict[str, Any] = None):
        """Coletar métrica específica"""
        metric = MetricData(
            timestamp=datetime.now().isoformat(),
            metric_name=metric_name,
            metric_type=metric_type,
            value=value,
            unit=unit,
            flow_id=flow_id,
            context=context or {}
        )
        
        self.metrics_buffer.append(metric)
        
        # Verificar se precisa gerar alerta
        self._check_metric_thresholds(metric)
        
        # Atualizar última atividade
        if flow_id in self.current_flows:
            try:
                self.current_flows[flow_id]["last_activity"] = time.time()
                self.current_flows[flow_id]["metrics_count"] += 1
            except TypeError:
                # Lidar com StateWithId que não permite assignment
                flow_info = dict(self.current_flows[flow_id])
                flow_info["last_activity"] = time.time()
                flow_info["metrics_count"] = flow_info.get("metrics_count", 0) + 1
                self.current_flows[flow_id] = flow_info
    
    def perform_health_check(self, flow_id: str, component: str = "main") -> HealthCheck:
        """Realizar verificação de saúde"""
        start_time = time.time()
        
        try:
            # Verificações específicas do componente
            details = {}
            status = "healthy"
            
            if component == "main":
                details = self._check_main_health(flow_id)
            elif component == "crews":
                details = self._check_crews_health(flow_id)
            elif component == "database":
                details = self._check_database_health(flow_id)
            elif component == "filesystem":
                details = self._check_filesystem_health(flow_id)
            else:
                details = {"error": f"Componente desconhecido: {component}"}
                status = "error"
            
            # Determinar status baseado nos resultados
            if details.get("errors", 0) > 0:
                status = "error"
            elif details.get("warnings", 0) > 0:
                status = "warning"
            
            response_time = time.time() - start_time
            
            health_check = HealthCheck(
                timestamp=datetime.now().isoformat(),
                flow_id=flow_id,
                component=component,
                status=status,
                response_time=response_time,
                details=details
            )
            
            self.health_checks[f"{flow_id}_{component}"] = health_check
            
            # Coletar métrica de response time
            self.collect_metric(
                flow_id=flow_id,
                metric_name="health_check_response_time",
                value=response_time,
                metric_type=MetricType.HEALTH,
                unit="seconds",
                context={"component": component}
            )
            
            return health_check
            
        except Exception as e:
            response_time = time.time() - start_time
            error_health = HealthCheck(
                timestamp=datetime.now().isoformat(),
                flow_id=flow_id,
                component=component,
                status="error",
                response_time=response_time,
                details={"error": str(e)}
            )
            
            self.health_checks[f"{flow_id}_{component}"] = error_health
            return error_health
    
    def create_alert(self, flow_id: str, level: AlertLevel, title: str, 
                    message: str, metric_name: str = "", 
                    threshold_value: float = 0.0, current_value: float = 0.0,
                    context: Dict[str, Any] = None) -> Alert:
        """Criar alerta do sistema"""
        alert_id = f"{flow_id}_{metric_name}_{int(time.time())}"
        
        alert = Alert(
            timestamp=datetime.now().isoformat(),
            alert_id=alert_id,
            level=level,
            title=title,
            message=message,
            flow_id=flow_id,
            metric_name=metric_name,
            threshold_value=threshold_value,
            current_value=current_value,
            context=context or {}
        )
        
        # Verificar cooldown
        cooldown_key = f"{flow_id}_{metric_name}_{level.value}"
        current_time = time.time()
        
        if cooldown_key in self.alert_cooldowns:
            if current_time - self.alert_cooldowns[cooldown_key] < self.alert_cooldown:
                return alert  # Não disparar alerta em cooldown
        
        self.alert_cooldowns[cooldown_key] = current_time
        self.alerts_history.append(alert)
        
        # Salvar alerta
        self._save_alert(alert)
        
        # Executar callbacks
        for callback in self.alert_callbacks:
            try:
                self.executor.submit(callback, alert)
            except Exception as e:
                monitoring_logger.error(f"❌ Erro no callback de alerta: {e}")
        
        monitoring_logger.warning(f"🚨 ALERTA {level.value.upper()}: {title}")
        
        return alert
    
    def resolve_alert(self, alert_id: str):
        """Resolver alerta"""
        for alert in self.alerts_history:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_timestamp = datetime.now().isoformat()
                monitoring_logger.info(f"✅ Alerta resolvido: {alert_id}")
                break
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Adicionar callback para alertas"""
        self.alert_callbacks.append(callback)
        monitoring_logger.info("📞 Callback de alerta adicionado")
    
    def get_current_metrics(self, flow_id: str = None, 
                           metric_type: MetricType = None,
                           minutes_back: int = 60) -> List[MetricData]:
        """Obter métricas atuais"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes_back)
        
        filtered_metrics = []
        for metric in self.metrics_buffer:
            metric_time = datetime.fromisoformat(metric.timestamp)
            
            if metric_time < cutoff_time:
                continue
            
            if flow_id and metric.flow_id != flow_id:
                continue
            
            if metric_type and metric.metric_type != metric_type:
                continue
            
            filtered_metrics.append(metric)
        
        return sorted(filtered_metrics, key=lambda x: x.timestamp, reverse=True)
    
    def get_active_alerts(self, flow_id: str = None, 
                         level: AlertLevel = None) -> List[Alert]:
        """Obter alertas ativos"""
        active_alerts = []
        
        for alert in self.alerts_history:
            if alert.resolved:
                continue
            
            if flow_id and alert.flow_id != flow_id:
                continue
            
            if level and alert.level != level:
                continue
            
            active_alerts.append(alert)
        
        return sorted(active_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_health_status(self, flow_id: str = None) -> Dict[str, Any]:
        """Obter status de saúde geral"""
        if flow_id:
            flow_health_checks = {k: v for k, v in self.health_checks.items() 
                                 if k.startswith(f"{flow_id}_")}
        else:
            flow_health_checks = self.health_checks
        
        status_summary = defaultdict(int)
        components = {}
        
        for key, health_check in flow_health_checks.items():
            status_summary[health_check.status] += 1
            components[key] = {
                "status": health_check.status,
                "response_time": health_check.response_time,
                "last_check": health_check.timestamp
            }
        
        overall_status = "healthy"
        if status_summary["error"] > 0:
            overall_status = "error"
        elif status_summary["warning"] > 0:
            overall_status = "warning"
        
        return {
            "overall_status": overall_status,
            "status_summary": dict(status_summary),
            "components": components,
            "total_checks": len(flow_health_checks),
            "last_update": datetime.now().isoformat()
        }
    
    def get_performance_summary(self, flow_id: str, minutes_back: int = 60) -> Dict[str, Any]:
        """Obter resumo de performance"""
        metrics = self.get_current_metrics(flow_id, minutes_back=minutes_back)
        
        if not metrics:
            return {"error": "Nenhuma métrica encontrada"}
        
        # Agrupar métricas por nome
        grouped_metrics = defaultdict(list)
        for metric in metrics:
            grouped_metrics[metric.metric_name].append(metric.value)
        
        # Calcular estatísticas
        summary = {}
        for metric_name, values in grouped_metrics.items():
            summary[metric_name] = {
                "count": len(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "current": values[0] if values else 0  # Mais recente
            }
        
        # Informações do Flow
        flow_info = self.current_flows.get(flow_id, {})
        if flow_info:
            uptime = time.time() - flow_info.get("start_time", time.time())
            summary["flow_info"] = {
                "uptime_seconds": uptime,
                "uptime_formatted": str(timedelta(seconds=int(uptime))),
                "metrics_collected": flow_info.get("metrics_count", 0),
                "last_activity": datetime.fromtimestamp(
                    flow_info.get("last_activity", time.time())
                ).isoformat()
            }
        
        return summary
    
    def export_metrics(self, flow_id: str = None, 
                      format: str = "json") -> str:
        """Exportar métricas para arquivo"""
        metrics = self.get_current_metrics(flow_id, minutes_back=24*60)  # 24 horas
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        flow_suffix = f"_{flow_id}" if flow_id else "_all"
        filename = f"metrics_export{flow_suffix}_{timestamp}.{format}"
        filepath = self.base_path / filename
        
        if format == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump([metric.to_dict() for metric in metrics], 
                         f, indent=2, ensure_ascii=False)
        elif format == "csv":
            import csv
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                if metrics:
                    writer = csv.DictWriter(f, fieldnames=metrics[0].to_dict().keys())
                    writer.writeheader()
                    for metric in metrics:
                        writer.writerow(metric.to_dict())
        
        monitoring_logger.info(f"📊 Métricas exportadas: {filepath}")
        return str(filepath)
    
    def cleanup_old_data(self):
        """Limpar dados antigos"""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        # Limpar buffer de métricas
        new_buffer = deque(maxlen=self.metrics_buffer.maxlen)
        for metric in self.metrics_buffer:
            metric_time = datetime.fromisoformat(metric.timestamp)
            if metric_time >= cutoff_time:
                new_buffer.append(metric)
        
        removed_metrics = len(self.metrics_buffer) - len(new_buffer)
        self.metrics_buffer = new_buffer
        
        # Limpar alertas antigos resolvidos
        active_alerts = []
        for alert in self.alerts_history:
            if not alert.resolved:
                active_alerts.append(alert)
            else:
                alert_time = datetime.fromisoformat(alert.timestamp)
                if alert_time >= cutoff_time:
                    active_alerts.append(alert)
        
        removed_alerts = len(self.alerts_history) - len(active_alerts)
        self.alerts_history = active_alerts
        
        monitoring_logger.info(f"🧹 Limpeza: {removed_metrics} métricas, {removed_alerts} alertas removidos")
    
    # =============== MÉTODOS INTERNOS ===============
    
    def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        while self.monitoring_active:
            try:
                # Coletar métricas de sistema
                self._collect_system_metrics()
                
                # Verificar saúde dos Flows
                for flow_id in list(self.current_flows.keys()):
                    self._monitor_flow_health(flow_id)
                
                # Limpar dados antigos periodicamente
                if int(time.time()) % 3600 == 0:  # A cada hora
                    self.cleanup_old_data()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                monitoring_logger.error(f"❌ Erro no loop de monitoramento: {e}")
                time.sleep(10)
    
    def _collect_system_metrics(self):
        """Coletar métricas do sistema"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self._record_system_metric("cpu_usage", cpu_percent, "percent")
            
            # Memória
            memory = psutil.virtual_memory()
            self._record_system_metric("memory_usage", memory.percent, "percent")
            self._record_system_metric("memory_available", memory.available / (1024**3), "GB")
            
            # Disco
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self._record_system_metric("disk_usage", disk_percent, "percent")
            self._record_system_metric("disk_free", disk.free / (1024**3), "GB")
            
            # Rede (se disponível)
            try:
                net_io = psutil.net_io_counters()
                self._record_system_metric("network_bytes_sent", net_io.bytes_sent, "bytes")
                self._record_system_metric("network_bytes_recv", net_io.bytes_recv, "bytes")
            except:
                pass
            
        except Exception as e:
            monitoring_logger.error(f"❌ Erro ao coletar métricas do sistema: {e}")
    
    def _record_system_metric(self, metric_name: str, value: float, unit: str):
        """Registrar métrica do sistema"""
        for flow_id in self.current_flows.keys():
            self.collect_metric(
                flow_id=flow_id,
                metric_name=metric_name,
                value=value,
                metric_type=MetricType.RESOURCE,
                unit=unit,
                context={"source": "system"}
            )
    
    def _monitor_flow_health(self, flow_id: str):
        """Monitorar saúde de um Flow específico"""
        try:
            flow_info = self.current_flows.get(flow_id)
            if not flow_info:
                return
            
            # Lidar com StateWithId que pode não ser subscriptable
            try:
                instance = flow_info["instance"]
                last_activity = flow_info.get("last_activity", time.time())
            except (TypeError, KeyError):
                # Se não conseguir acessar como dict, tratar como objeto
                instance = getattr(flow_info, 'instance', None)
                last_activity = getattr(flow_info, 'last_activity', time.time())
                if instance is None:
                    return
            
            current_time = time.time()
            inactive_time = current_time - last_activity
            
            self.collect_metric(
                flow_id=flow_id,
                metric_name="inactive_time",
                value=inactive_time,
                metric_type=MetricType.HEALTH,
                unit="seconds"
            )
            
            # Verificar se Flow está travado
            if inactive_time > 300:  # 5 minutos
                self.create_alert(
                    flow_id=flow_id,
                    level=AlertLevel.WARNING,
                    title="Flow Inativo",
                    message=f"Flow sem atividade há {inactive_time:.1f} segundos",
                    metric_name="inactive_time",
                    current_value=inactive_time,
                    threshold_value=300.0
                )
            
            # Realizar health check periódico
            if int(current_time) % 60 == 0:  # A cada minuto
                self.perform_health_check(flow_id, "main")
            
        except Exception as e:
            monitoring_logger.error(f"❌ Erro ao monitorar Flow {flow_id}: {e}")
    
    def _check_metric_thresholds(self, metric: MetricData):
        """Verificar se métrica excede thresholds"""
        thresholds = self.alert_thresholds.get(metric.metric_name)
        if not thresholds:
            return
        
        # Verificar threshold crítico
        if metric.value >= thresholds.get("critical", float('inf')):
            self.create_alert(
                flow_id=metric.flow_id,
                level=AlertLevel.CRITICAL,
                title=f"Threshold Crítico: {metric.metric_name}",
                message=f"{metric.metric_name} atingiu nível crítico: {metric.value}{metric.unit}",
                metric_name=metric.metric_name,
                threshold_value=thresholds["critical"],
                current_value=metric.value
            )
        
        # Verificar threshold de warning
        elif metric.value >= thresholds.get("warning", float('inf')):
            self.create_alert(
                flow_id=metric.flow_id,
                level=AlertLevel.WARNING,
                title=f"Threshold Warning: {metric.metric_name}",
                message=f"{metric.metric_name} atingiu nível de atenção: {metric.value}{metric.unit}",
                metric_name=metric.metric_name,
                threshold_value=thresholds["warning"],
                current_value=metric.value
            )
    
    def _check_main_health(self, flow_id: str) -> Dict[str, Any]:
        """Verificar saúde principal do Flow"""
        details = {"component": "main", "checks": [], "errors": 0, "warnings": 0}
        
        flow_info = self.current_flows.get(flow_id)
        if not flow_info:
            details["checks"].append({"check": "flow_registration", "status": "error", "message": "Flow não registrado"})
            details["errors"] += 1
            return details
        
        try:
            instance = flow_info["instance"]
        except TypeError:
            # Lidar com StateWithId
            instance = getattr(flow_info, 'instance', None)
            if instance is None:
                details["checks"].append({"check": "flow_instance", "status": "error", "message": "Instância de flow não encontrada"})
                details["errors"] += 1
                return details
        
        # Verificar se tem flow_id
        if not hasattr(instance, 'flow_id') or not instance.flow_id:
            details["checks"].append({"check": "flow_id", "status": "error", "message": "Flow sem ID válido"})
            details["errors"] += 1
        else:
            details["checks"].append({"check": "flow_id", "status": "ok", "message": f"Flow ID: {instance.flow_id}"})
        
        # Verificar se tem fase atual
        if hasattr(instance, 'fase_atual'):
            details["checks"].append({"check": "fase_atual", "status": "ok", "message": f"Fase: {instance.fase_atual}"})
        else:
            details["checks"].append({"check": "fase_atual", "status": "warning", "message": "Fase atual não definida"})
            details["warnings"] += 1
        
        return details
    
    def _check_crews_health(self, flow_id: str) -> Dict[str, Any]:
        """Verificar saúde dos Crews"""
        details = {"component": "crews", "checks": [], "errors": 0, "warnings": 0}
        
        flow_info = self.current_flows.get(flow_id)
        if not flow_info:
            details["errors"] += 1
            return details
        
        try:
            instance = flow_info["instance"]
        except TypeError:
            # Lidar com StateWithId
            instance = getattr(flow_info, 'instance', None)
            if instance is None:
                details["errors"] += 1
                return details
        
        # Verificar cache de crews
        if hasattr(instance, 'crews_cache'):
            crews_count = len(getattr(instance, 'crews_cache', {}))
            details["checks"].append({"check": "crews_cache", "status": "ok", "message": f"{crews_count} crews em cache"})
        else:
            details["checks"].append({"check": "crews_cache", "status": "warning", "message": "Cache de crews não encontrado"})
            details["warnings"] += 1
        
        return details
    
    def _check_database_health(self, flow_id: str) -> Dict[str, Any]:
        """Verificar saúde do banco de dados"""
        details = {"component": "database", "checks": [], "errors": 0, "warnings": 0}
        
        try:
            # Verificar se arquivo de dados existe
            data_file = Path("data/vendas.csv")
            if data_file.exists():
                size_mb = data_file.stat().st_size / (1024 * 1024)
                details["checks"].append({
                    "check": "data_file", 
                    "status": "ok", 
                    "message": f"Arquivo de dados: {size_mb:.2f} MB"
                })
            else:
                details["checks"].append({
                    "check": "data_file", 
                    "status": "error", 
                    "message": "Arquivo de dados não encontrado"
                })
                details["errors"] += 1
                
        except Exception as e:
            details["checks"].append({
                "check": "data_access", 
                "status": "error", 
                "message": f"Erro ao acessar dados: {e}"
            })
            details["errors"] += 1
        
        return details
    
    def _check_filesystem_health(self, flow_id: str) -> Dict[str, Any]:
        """Verificar saúde do sistema de arquivos"""
        details = {"component": "filesystem", "checks": [], "errors": 0, "warnings": 0}
        
        try:
            # Verificar diretórios principais
            dirs_to_check = ["output", "logs", "data"]
            for dir_name in dirs_to_check:
                dir_path = Path(dir_name)
                if dir_path.exists() and dir_path.is_dir():
                    details["checks"].append({
                        "check": f"directory_{dir_name}", 
                        "status": "ok", 
                        "message": f"Diretório {dir_name} OK"
                    })
                else:
                    details["checks"].append({
                        "check": f"directory_{dir_name}", 
                        "status": "error", 
                        "message": f"Diretório {dir_name} não encontrado"
                    })
                    details["errors"] += 1
            
            # Verificar permissões de escrita
            test_file = Path("logs/health_check_test.tmp")
            try:
                test_file.write_text("test")
                test_file.unlink()
                details["checks"].append({
                    "check": "write_permission", 
                    "status": "ok", 
                    "message": "Permissões de escrita OK"
                })
            except Exception as e:
                details["checks"].append({
                    "check": "write_permission", 
                    "status": "error", 
                    "message": f"Erro de permissão: {e}"
                })
                details["errors"] += 1
                
        except Exception as e:
            details["checks"].append({
                "check": "filesystem_access", 
                "status": "error", 
                "message": f"Erro ao verificar filesystem: {e}"
            })
            details["errors"] += 1
        
        return details
    
    def _save_alert(self, alert: Alert):
        """Salvar alerta em arquivo"""
        try:
            timestamp = alert.timestamp.replace(':', '-').replace('.', '-')
            filename = f"alert_{alert.flow_id}_{timestamp}.json"
            filepath = self.base_path / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(alert.to_dict(), f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            monitoring_logger.error(f"❌ Erro ao salvar alerta: {e}")


# =============== UTILITÁRIOS E FUNÇÕES AUXILIARES ===============

def create_monitoring_system(base_path: str = "logs/monitoring") -> FlowMonitoringSystem:
    """Factory function para criar sistema de monitoramento"""
    return FlowMonitoringSystem(base_path)

# Instância global para uso fácil
_global_monitoring_system = None

def get_global_monitoring_system() -> FlowMonitoringSystem:
    """Obter instância global do sistema de monitoramento"""
    global _global_monitoring_system
    if _global_monitoring_system is None:
        _global_monitoring_system = create_monitoring_system()
    return _global_monitoring_system

# Callbacks pré-definidos para alertas
def console_alert_callback(alert: Alert):
    """Callback para exibir alertas no console"""
    icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}.get(alert.level.value, "📢")
    print(f"{icon} [{alert.level.value.upper()}] {alert.title}: {alert.message}")

def email_alert_callback(alert: Alert):
    """Callback para enviar alertas por email (implementação básica)"""
    # Implementação básica - pode ser expandida com SMTP real
    monitoring_logger.info(f"📧 Email alert: {alert.title} - {alert.message}")

def slack_alert_callback(alert: Alert):
    """Callback para enviar alertas para Slack (implementação básica)"""
    # Implementação básica - pode ser expandida com webhook real
    monitoring_logger.info(f"💬 Slack alert: {alert.title} - {alert.message}") 