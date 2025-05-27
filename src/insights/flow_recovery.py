"""
Sistema de Recovery e Restart Automático para CrewAI Flow
Implementa recuperação automática de falhas e restart inteligente
"""

import json
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Configurar logger específico para recovery
recovery_logger = logging.getLogger('flow_recovery')
recovery_logger.setLevel(logging.INFO)

class RecoveryLevel(Enum):
    """Níveis de recovery do sistema"""
    MINIMAL = "minimal"        # Apenas estado básico
    PARTIAL = "partial"        # Estado + resultados parciais  
    COMPLETE = "complete"      # Estado completo + caches
    DEEP = "deep"             # Tudo + análise de dependências

class FailureType(Enum):
    """Tipos de falhas que podem ocorrer"""
    NETWORK_ERROR = "network_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    DATA_ERROR = "data_error"
    CREW_ERROR = "crew_error"
    DEPENDENCY_ERROR = "dependency_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class RecoveryPoint:
    """Ponto de recuperação do Flow"""
    timestamp: str
    flow_id: str
    estado_atual: str
    analises_concluidas: List[str]
    analises_em_execucao: List[str]
    dados_estado: Dict[str, Any]
    checksum: str
    versao_recovery: str = "3.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecoveryPoint':
        return cls(**data)

@dataclass
class FailureRecord:
    """Registro de falha do sistema"""
    timestamp: str
    flow_id: str
    failure_type: FailureType
    error_message: str
    stack_trace: str
    recovery_attempts: int
    context: Dict[str, Any]

class FlowRecoverySystem:
    """Sistema principal de recovery e restart automático"""
    
    def __init__(self, base_path: str = "logs/recovery"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Configurações
        self.max_recovery_attempts = 3
        self.recovery_timeout = 300  # 5 minutos
        self.checkpoint_interval = 30  # 30 segundos
        self.auto_recovery_enabled = True
        
        # Estado interno
        self.current_flow_id = None
        self.recovery_thread = None
        self.monitoring_active = False
        self.failure_history: List[FailureRecord] = []
        
        # Executor para operações assíncronas
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        recovery_logger.info("🛡️ FlowRecoverySystem inicializado")
    
    def create_checkpoint(self, flow_state: Any, level: RecoveryLevel = RecoveryLevel.COMPLETE) -> str:
        """Criar ponto de recuperação do Flow"""
        try:
            if not hasattr(flow_state, 'flow_id') or not flow_state.flow_id:
                recovery_logger.warning("❌ Flow sem ID válido para checkpoint")
                return ""
            
            # Extrair dados do estado
            dados_estado = self._extract_state_data(flow_state, level)
            
            # Criar recovery point
            recovery_point = RecoveryPoint(
                timestamp=datetime.now().isoformat(),
                flow_id=flow_state.flow_id,
                estado_atual=getattr(flow_state, 'fase_atual', 'desconhecido'),
                analises_concluidas=self._get_completed_analyses(flow_state),
                analises_em_execucao=self._get_running_analyses(flow_state),
                dados_estado=dados_estado,
                checksum=self._calculate_checksum(dados_estado)
            )
            
            # Salvar checkpoint
            checkpoint_path = self._save_checkpoint(recovery_point)
            
            recovery_logger.info(f"✅ Checkpoint criado: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            recovery_logger.error(f"❌ Erro ao criar checkpoint: {e}")
            return ""
    
    def recover_flow(self, flow_id: str, target_class=None) -> Optional[Any]:
        """Recuperar Flow a partir do último checkpoint"""
        try:
            recovery_logger.info(f"🔄 Iniciando recovery do Flow: {flow_id}")
            
            # Encontrar último checkpoint válido
            checkpoint_path = self._find_latest_checkpoint(flow_id)
            if not checkpoint_path:
                recovery_logger.error(f"❌ Nenhum checkpoint encontrado para Flow: {flow_id}")
                return None
            
            # Carregar recovery point
            recovery_point = self._load_checkpoint(checkpoint_path)
            if not recovery_point:
                recovery_logger.error(f"❌ Erro ao carregar checkpoint: {checkpoint_path}")
                return None
            
            # Validar integridade
            if not self._validate_checkpoint(recovery_point):
                recovery_logger.error(f"❌ Checkpoint corrompido: {checkpoint_path}")
                return None
            
            # Reconstruir Flow
            recovered_flow = self._reconstruct_flow(recovery_point, target_class)
            if recovered_flow:
                recovery_logger.info(f"✅ Flow recuperado com sucesso: {flow_id}")
                return recovered_flow
            else:
                recovery_logger.error(f"❌ Falha na reconstrução do Flow: {flow_id}")
                return None
                
        except Exception as e:
            recovery_logger.error(f"❌ Erro no recovery: {e}")
            return None
    
    def start_auto_recovery(self, flow_instance: Any):
        """Iniciar monitoramento automático de recovery"""
        self.current_flow_id = getattr(flow_instance, 'flow_id', None)
        self.monitoring_active = True
        
        # Thread de monitoramento
        self.recovery_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(flow_instance,),
            daemon=True
        )
        self.recovery_thread.start()
        
        recovery_logger.info(f"🔍 Auto-recovery iniciado para Flow: {self.current_flow_id}")
    
    def stop_auto_recovery(self):
        """Parar monitoramento automático"""
        self.monitoring_active = False
        if self.recovery_thread and self.recovery_thread.is_alive():
            self.recovery_thread.join(timeout=5)
        
        recovery_logger.info("🛑 Auto-recovery parado")
    
    def register_failure(self, flow_id: str, failure_type: FailureType, 
                        error_message: str, stack_trace: str = "", 
                        context: Dict[str, Any] = None):
        """Registrar falha do sistema"""
        failure_record = FailureRecord(
            timestamp=datetime.now().isoformat(),
            flow_id=flow_id,
            failure_type=failure_type,
            error_message=error_message,
            stack_trace=stack_trace,
            recovery_attempts=0,
            context=context or {}
        )
        
        self.failure_history.append(failure_record)
        
        # Salvar registro de falha
        self._save_failure_record(failure_record)
        
        recovery_logger.error(f"💥 Falha registrada: {failure_type.value} - {error_message}")
        
        # Tentar recovery automático se habilitado
        if self.auto_recovery_enabled:
            self._attempt_auto_recovery(failure_record)
    
    def get_recovery_status(self, flow_id: str) -> Dict[str, Any]:
        """Obter status de recovery de um Flow"""
        checkpoints = self._list_checkpoints(flow_id)
        failures = [f for f in self.failure_history if f.flow_id == flow_id]
        
        return {
            "flow_id": flow_id,
            "total_checkpoints": len(checkpoints),
            "latest_checkpoint": checkpoints[0] if checkpoints else None,
            "total_failures": len(failures),
            "last_failure": failures[-1] if failures else None,
            "recovery_possible": len(checkpoints) > 0,
            "monitoring_active": self.monitoring_active and self.current_flow_id == flow_id
        }
    
    def cleanup_old_checkpoints(self, days_to_keep: int = 7):
        """Limpar checkpoints antigos"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleaned_count = 0
        
        for checkpoint_file in self.base_path.glob("checkpoint_*.json"):
            try:
                file_time = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
                if file_time < cutoff_date:
                    checkpoint_file.unlink()
                    cleaned_count += 1
            except Exception as e:
                recovery_logger.warning(f"⚠️ Erro ao limpar checkpoint {checkpoint_file}: {e}")
        
        recovery_logger.info(f"🧹 Limpeza concluída: {cleaned_count} checkpoints removidos")
    
    # =============== MÉTODOS INTERNOS ===============
    
    def _extract_state_data(self, flow_state: Any, level: RecoveryLevel) -> Dict[str, Any]:
        """Extrair dados do estado baseado no nível de recovery"""
        dados = {
            "flow_id": getattr(flow_state, 'flow_id', ''),
            "fase_atual": getattr(flow_state, 'fase_atual', ''),
            "inicio_execucao": getattr(flow_state, 'inicio_execucao', ''),
            "nivel_recovery": level.value
        }
        
        if level in [RecoveryLevel.PARTIAL, RecoveryLevel.COMPLETE, RecoveryLevel.DEEP]:
            # Adicionar resultados de análises
            for attr in ['analise_tendencias', 'analise_sazonalidade', 
                        'analise_segmentos', 'analise_projecoes']:
                if hasattr(flow_state, attr):
                    dados[attr] = getattr(flow_state, attr, {})
        
        if level in [RecoveryLevel.COMPLETE, RecoveryLevel.DEEP]:
            # Adicionar dados completos do estado
            dados['crews_cache'] = getattr(flow_state, 'crews_cache', {})
            dados['arquivos_gerados'] = getattr(flow_state, 'arquivos_gerados', [])
            dados['metricas_performance'] = getattr(flow_state, 'metricas_performance', {})
        
        if level == RecoveryLevel.DEEP:
            # Análise profunda de dependências
            dados['dependency_graph'] = self._analyze_dependencies(flow_state)
            dados['resource_usage'] = self._get_resource_usage()
        
        return dados
    
    def _get_completed_analyses(self, flow_state: Any) -> List[str]:
        """Obter lista de análises concluídas"""
        completed = []
        for analysis in ['analise_tendencias', 'analise_sazonalidade', 
                        'analise_segmentos', 'analise_projecoes']:
            if hasattr(flow_state, analysis):
                analysis_data = getattr(flow_state, analysis, {})
                if analysis_data.get('status') == 'concluido':
                    completed.append(analysis)
        return completed
    
    def _get_running_analyses(self, flow_state: Any) -> List[str]:
        """Obter lista de análises em execução"""
        running = []
        for analysis in ['analise_tendencias', 'analise_sazonalidade', 
                        'analise_segmentos', 'analise_projecoes']:
            if hasattr(flow_state, analysis):
                analysis_data = getattr(flow_state, analysis, {})
                if analysis_data.get('status') == 'executando':
                    running.append(analysis)
        return running
    
    def _calculate_checksum(self, dados: Dict[str, Any]) -> str:
        """Calcular checksum dos dados para validação"""
        dados_str = json.dumps(dados, sort_keys=True, default=str)
        return hashlib.sha256(dados_str.encode()).hexdigest()
    
    def _save_checkpoint(self, recovery_point: RecoveryPoint) -> str:
        """Salvar checkpoint no disco"""
        timestamp = recovery_point.timestamp.replace(':', '-').replace('.', '-')
        filename = f"checkpoint_{recovery_point.flow_id}_{timestamp}.json"
        filepath = self.base_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(recovery_point.to_dict(), f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def _find_latest_checkpoint(self, flow_id: str) -> Optional[str]:
        """Encontrar último checkpoint válido"""
        checkpoints = list(self.base_path.glob(f"checkpoint_{flow_id}_*.json"))
        if not checkpoints:
            return None
        
        # Ordenar por data de modificação (mais recente primeiro)
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return str(checkpoints[0])
    
    def _load_checkpoint(self, checkpoint_path: str) -> Optional[RecoveryPoint]:
        """Carregar checkpoint do disco"""
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return RecoveryPoint.from_dict(data)
        except Exception as e:
            recovery_logger.error(f"❌ Erro ao carregar checkpoint {checkpoint_path}: {e}")
            return None
    
    def _validate_checkpoint(self, recovery_point: RecoveryPoint) -> bool:
        """Validar integridade do checkpoint"""
        try:
            # Verificar checksum
            calculated_checksum = self._calculate_checksum(recovery_point.dados_estado)
            if calculated_checksum != recovery_point.checksum:
                recovery_logger.error("❌ Checksum inválido")
                return False
            
            # Verificar campos obrigatórios
            required_fields = ['flow_id', 'timestamp', 'estado_atual']
            for field in required_fields:
                if not getattr(recovery_point, field, None):
                    recovery_logger.error(f"❌ Campo obrigatório ausente: {field}")
                    return False
            
            return True
            
        except Exception as e:
            recovery_logger.error(f"❌ Erro na validação: {e}")
            return False
    
    def _reconstruct_flow(self, recovery_point: RecoveryPoint, target_class=None) -> Optional[Any]:
        """Reconstruir Flow a partir do recovery point"""
        try:
            # Se não temos a classe alvo, tentar importar
            if target_class is None:
                from insights.flow_main import InsightsFlowState
                target_class = InsightsFlowState
            
            # Criar nova instância
            recovered_state = target_class()
            
            # Restaurar dados
            dados = recovery_point.dados_estado
            for key, value in dados.items():
                if hasattr(recovered_state, key):
                    setattr(recovered_state, key, value)
            
            recovery_logger.info(f"✅ Estado reconstruído para Flow: {recovery_point.flow_id}")
            return recovered_state
            
        except Exception as e:
            recovery_logger.error(f"❌ Erro na reconstrução: {e}")
            return None
    
    def _monitoring_loop(self, flow_instance: Any):
        """Loop principal de monitoramento"""
        last_checkpoint = time.time()
        
        while self.monitoring_active:
            try:
                current_time = time.time()
                
                # Criar checkpoint periódico
                if current_time - last_checkpoint > self.checkpoint_interval:
                    self.create_checkpoint(flow_instance, RecoveryLevel.PARTIAL)
                    last_checkpoint = current_time
                
                # Verificar saúde do Flow
                if not self._check_flow_health(flow_instance):
                    recovery_logger.warning("⚠️ Flow não está respondendo adequadamente")
                
                time.sleep(5)  # Verificar a cada 5 segundos
                
            except Exception as e:
                recovery_logger.error(f"❌ Erro no monitoramento: {e}")
                time.sleep(10)  # Esperar mais em caso de erro
    
    def _check_flow_health(self, flow_instance: Any) -> bool:
        """Verificar saúde do Flow"""
        try:
            # Verificações básicas
            if not hasattr(flow_instance, 'flow_id'):
                return False
            
            # Verificar se não está travado
            if hasattr(flow_instance, 'ultima_atividade'):
                ultima_atividade = getattr(flow_instance, 'ultima_atividade', 0)
                if time.time() - ultima_atividade > 300:  # 5 minutos sem atividade
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _attempt_auto_recovery(self, failure_record: FailureRecord):
        """Tentar recovery automático"""
        if failure_record.recovery_attempts >= self.max_recovery_attempts:
            recovery_logger.error(f"❌ Máximo de tentativas de recovery atingido para {failure_record.flow_id}")
            return
        
        recovery_logger.info(f"🔄 Tentando auto-recovery para {failure_record.flow_id}")
        
        # Atualizar contador de tentativas
        failure_record.recovery_attempts += 1
        
        # Executar recovery em thread separada
        self.executor.submit(self._execute_auto_recovery, failure_record)
    
    def _execute_auto_recovery(self, failure_record: FailureRecord):
        """Executar recovery automático"""
        try:
            # Tentar recuperar Flow
            recovered_flow = self.recover_flow(failure_record.flow_id)
            
            if recovered_flow:
                recovery_logger.info(f"✅ Auto-recovery bem-sucedido para {failure_record.flow_id}")
                # Restartar monitoramento
                self.start_auto_recovery(recovered_flow)
            else:
                recovery_logger.error(f"❌ Auto-recovery falhado para {failure_record.flow_id}")
                
        except Exception as e:
            recovery_logger.error(f"❌ Erro no auto-recovery: {e}")
    
    def _save_failure_record(self, failure_record: FailureRecord):
        """Salvar registro de falha"""
        timestamp = failure_record.timestamp.replace(':', '-').replace('.', '-')
        filename = f"failure_{failure_record.flow_id}_{timestamp}.json"
        filepath = self.base_path / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(failure_record), f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            recovery_logger.error(f"❌ Erro ao salvar registro de falha: {e}")
    
    def _list_checkpoints(self, flow_id: str) -> List[str]:
        """Listar checkpoints de um Flow"""
        checkpoints = list(self.base_path.glob(f"checkpoint_{flow_id}_*.json"))
        return [str(cp) for cp in sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)]
    
    def _analyze_dependencies(self, flow_state: Any) -> Dict[str, Any]:
        """Analisar dependências do Flow"""
        # Implementação básica - pode ser expandida
        dependencies = {
            "analises_dependentes": [],
            "recursos_necessarios": [],
            "estado_dependencias": {}
        }
        
        # Verificar dependências entre análises
        if hasattr(flow_state, 'analise_tendencias') and hasattr(flow_state, 'analise_sazonalidade'):
            if (flow_state.analise_tendencias.get('status') == 'concluido' and 
                flow_state.analise_sazonalidade.get('status') == 'concluido'):
                dependencies["analises_dependentes"].append("projecoes_ready")
        
        return dependencies
    
    def _get_resource_usage(self) -> Dict[str, Any]:
        """Obter uso de recursos do sistema"""
        try:
            import psutil
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent if psutil.disk_usage('/') else 0,
                "timestamp": datetime.now().isoformat()
            }
        except ImportError:
            return {"error": "psutil não disponível"}
        except Exception as e:
            return {"error": str(e)}


# =============== UTILITÁRIOS E FUNÇÕES AUXILIARES ===============

def create_recovery_system(base_path: str = "logs/recovery") -> FlowRecoverySystem:
    """Factory function para criar sistema de recovery"""
    return FlowRecoverySystem(base_path)

def emergency_recovery(flow_id: str, recovery_system: FlowRecoverySystem = None) -> Optional[Any]:
    """Função de emergência para recovery rápido"""
    if recovery_system is None:
        recovery_system = create_recovery_system()
    
    recovery_logger.info(f"🚨 RECOVERY DE EMERGÊNCIA: {flow_id}")
    return recovery_system.recover_flow(flow_id)

# Instância global para uso fácil
_global_recovery_system = None

def get_global_recovery_system() -> FlowRecoverySystem:
    """Obter instância global do sistema de recovery"""
    global _global_recovery_system
    if _global_recovery_system is None:
        _global_recovery_system = create_recovery_system()
    return _global_recovery_system 