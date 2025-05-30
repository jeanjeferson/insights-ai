"""
🔍 ENHANCED LOGGING SYSTEM - FASE 1
===================================

Sistema de logging melhorado focado em:
- Context-aware error handling
- Progress indicators básicos  
- Correção de erros de reasoning
- Manter simplicidade e performance
"""

import time
import logging
import traceback
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from functools import wraps
import psutil
import os

# =============== ESTRUTURAS DE DADOS ===============

@dataclass
class LogContext:
    """Contexto para logs mais informativos"""
    agent: Optional[str] = None
    operation: Optional[str] = None
    operation_id: Optional[str] = None
    start_time: Optional[float] = None
    data_size: Optional[int] = None
    memory_usage: Optional[float] = None

@dataclass
class OperationProgress:
    """Tracking de progresso de operações"""
    name: str
    start_time: float
    expected_duration: Optional[float] = None
    current_step: int = 0
    total_steps: int = 1
    last_update: float = 0

# =============== CONTEXT MANAGER ===============

class LogContextManager:
    """Gerenciador de contexto thread-safe para logs"""
    
    def __init__(self):
        self._local = threading.local()
        self._operation_counter = 0
        self._lock = threading.Lock()
    
    def get_context(self) -> LogContext:
        """Obter contexto atual"""
        return getattr(self._local, 'context', LogContext())
    
    def set_context(self, **kwargs):
        """Definir contexto atual"""
        current = self.get_context()
        for key, value in kwargs.items():
            if hasattr(current, key):
                setattr(current, key, value)
        self._local.context = current
    
    def generate_operation_id(self) -> str:
        """Gerar ID único para operação"""
        with self._lock:
            self._operation_counter += 1
            return f"OP_{self._operation_counter:04d}"
    
    def clear_context(self):
        """Limpar contexto atual"""
        self._local.context = LogContext()

# =============== ENHANCED LOGGER ===============

class EnhancedLogger:
    """Logger melhorado com context-aware e progress tracking"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context_manager = LogContextManager()
        self.active_operations: Dict[str, OperationProgress] = {}
        self.error_count = 0
        self.warning_count = 0
        self._lock = threading.Lock()
        
        # Configurar formato básico se não configurado
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _get_memory_usage(self) -> float:
        """Obter uso de memória atual de forma eficiente"""
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024  # MB
        except:
            return 0.0
    
    def _format_context_prefix(self) -> str:
        """Criar prefixo com contexto atual"""
        context = self.context_manager.get_context()
        parts = []
        
        if context.operation_id:
            parts.append(f"[{context.operation_id}]")
        if context.agent:
            parts.append(f"{context.agent}")
        if context.operation:
            parts.append(f"{context.operation}")
            
        return " ".join(parts) + " " if parts else ""
    
    # =============== CONTEXT OPERATIONS ===============
    
    def start_operation(self, operation: str, agent: str = None, 
                       expected_duration: float = None, total_steps: int = 1) -> str:
        """Iniciar operação com tracking automático"""
        op_id = self.context_manager.generate_operation_id()
        
        # Configurar contexto
        self.context_manager.set_context(
            agent=agent,
            operation=operation,
            operation_id=op_id,
            start_time=time.time(),
            data_size=0,
            memory_usage=self._get_memory_usage()
        )
        
        # Registrar operação
        with self._lock:
            self.active_operations[op_id] = OperationProgress(
                name=operation,
                start_time=time.time(),
                expected_duration=expected_duration,
                total_steps=total_steps
            )
        
        # Log inicial
        prefix = f"🚀 [{op_id}]"
        if agent:
            prefix += f" {agent} -"
        
        self.info(f"{prefix} {operation}")
        
        if expected_duration:
            self.info(f"   ⏱️ ETA: {expected_duration:.1f}s")
        
        return op_id
    
    def update_progress(self, step: int = None, message: str = None):
        """Atualizar progresso da operação atual"""
        context = self.context_manager.get_context()
        if not context.operation_id or context.operation_id not in self.active_operations:
            return
        
        op = self.active_operations[context.operation_id]
        
        if step is not None:
            op.current_step = step
        else:
            op.current_step += 1
        
        op.last_update = time.time()
        
        # Calcular progresso
        progress_pct = (op.current_step / op.total_steps) * 100
        elapsed = time.time() - op.start_time
        
        # Calcular ETA
        if op.current_step > 0:
            eta = (elapsed / op.current_step) * (op.total_steps - op.current_step)
        else:
            eta = op.expected_duration or 0
        
        # Log de progresso
        progress_msg = f"📊 [{context.operation_id}] {progress_pct:.1f}%"
        if eta > 0:
            progress_msg += f" | ETA: {eta:.1f}s"
        if message:
            progress_msg += f" | {message}"
            
        self.info(progress_msg)
    
    def finish_operation(self, success: bool = True, message: str = None):
        """Finalizar operação atual"""
        context = self.context_manager.get_context()
        if not context.operation_id:
            return
        
        elapsed = time.time() - (context.start_time or time.time())
        memory = self._get_memory_usage()
        
        # Status
        status = "✅" if success else "❌"
        final_msg = f"{status} [{context.operation_id}] Concluído em {elapsed:.2f}s"
        
        if memory > 0:
            final_msg += f" | Mem: {memory:.1f}MB"
        
        if message:
            final_msg += f" | {message}"
        
        self.info(final_msg)
        
        # Limpar operação
        with self._lock:
            if context.operation_id in self.active_operations:
                del self.active_operations[context.operation_id]
        
        self.context_manager.clear_context()
    
    # =============== ENHANCED LOGGING METHODS ===============
    
    def info(self, message: str):
        """Log info com contexto"""
        prefix = self._format_context_prefix()
        self.logger.info(f"{prefix}{message}")
    
    def warning(self, message: str):
        """Log warning com contexto e contador"""
        with self._lock:
            self.warning_count += 1
        
        prefix = self._format_context_prefix()
        self.logger.warning(f"{prefix}⚠️ {message}")
    
    def error_with_context(self, error: Exception, operation: str = None, 
                          include_stack: bool = True):
        """Log erro com contexto completo e recovery suggestions"""
        with self._lock:
            self.error_count += 1
        
        context = self.context_manager.get_context()
        
        # Informações básicas
        error_msg = f"❌ ERRO #{self.error_count}"
        
        if context.operation_id:
            error_msg += f" [{context.operation_id}]"
        if context.agent:
            error_msg += f" {context.agent}"
        if operation or context.operation:
            error_msg += f" - {operation or context.operation}"
        
        self.logger.error(error_msg)
        
        # Contexto detalhado
        if context.start_time:
            elapsed = time.time() - context.start_time
            self.logger.error(f"   ⏱️ Tempo decorrido: {elapsed:.2f}s")
        
        if context.data_size and context.data_size > 0:
            self.logger.error(f"   📊 Dados processados: {context.data_size:,} registros")
        
        memory = self._get_memory_usage()
        if memory > 0:
            self.logger.error(f"   💾 Memória atual: {memory:.1f}MB")
        
        # Erro específico
        self.logger.error(f"   🔍 Tipo: {type(error).__name__}")
        self.logger.error(f"   📝 Mensagem: {str(error)}")
        
        # Stack trace se solicitado
        if include_stack and os.getenv('INSIGHTS_DEBUG', 'false').lower() == 'true':
            self.logger.error("   📚 Stack trace:")
            for line in traceback.format_exception(type(error), error, error.__traceback__):
                self.logger.error(f"     {line.rstrip()}")
        
        # Sugestões de recovery para erros conhecidos
        self._suggest_recovery(error)
    
    def _suggest_recovery(self, error: Exception):
        """Sugerir ações de recovery para erros conhecidos"""
        error_str = str(error).lower()
        
        if "create_reasoning_plan" in error_str:
            self.logger.error("   💡 SUGESTÃO: Erro conhecido do CrewAI - tentando retry automático")
            self.logger.error("   💡 AÇÃO: Sistema irá prosseguir com plano atual")
        
        elif "too large" in error_str or "token" in error_str:
            self.logger.error("   💡 SUGESTÃO: Problema de contexto/memória")
            self.logger.error("   💡 AÇÃO: Reduzir tamanho dos dados ou usar amostragem")
        
        elif "connection" in error_str or "timeout" in error_str:
            self.logger.error("   💡 SUGESTÃO: Problema de conectividade")
            self.logger.error("   💡 AÇÃO: Verificar conexão de rede/banco")
        
        elif "memory" in error_str or "out of memory" in error_str:
            self.logger.error("   💡 SUGESTÃO: Problema de memória")
            self.logger.error("   💡 AÇÃO: Reduzir dados ou aumentar recursos")
    
    def debug(self, message: str):
        """Log debug com contexto"""
        if os.getenv('INSIGHTS_DEBUG', 'false').lower() == 'true':
            prefix = self._format_context_prefix()
            self.logger.debug(f"{prefix}🔍 {message}")
    
    # =============== UTILITY METHODS ===============
    
    def get_stats(self) -> Dict[str, Any]:
        """Obter estatísticas do logger"""
        with self._lock:
            active_ops = len(self.active_operations)
            return {
                'active_operations': active_ops,
                'total_errors': self.error_count,
                'total_warnings': self.warning_count,
                'memory_usage_mb': self._get_memory_usage()
            }
    
    def log_milestone(self, milestone: str, metrics: Dict[str, Any] = None):
        """Log marco importante com métricas"""
        self.info("🎯 " + "="*40)
        self.info(f"🎯 MARCO: {milestone}")
        
        if metrics:
            for key, value in metrics.items():
                self.info(f"   📊 {key}: {value}")
        
        stats = self.get_stats()
        self.info(f"   📈 Operações ativas: {stats['active_operations']}")
        self.info(f"   ⚠️ Total warnings: {stats['total_warnings']}")
        self.info(f"   ❌ Total errors: {stats['total_errors']}")
        
        self.info("🎯 " + "="*40)

# =============== DECORATORS ===============

def with_enhanced_logging(operation_name: str = None, expected_duration: float = None):
    """Decorator para logging automático de operações"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Descobrir nome da operação
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # Tentar descobrir agente do self
            agent_name = None
            if args and hasattr(args[0], '__class__'):
                class_name = args[0].__class__.__name__
                if 'Tool' in class_name or 'Engine' in class_name:
                    agent_name = class_name.replace('Tool', '').replace('Engine', '')
            
            # Obter logger global
            logger = get_enhanced_logger()
            
            # Iniciar operação
            op_id = logger.start_operation(
                operation=op_name,
                agent=agent_name,
                expected_duration=expected_duration
            )
            
            try:
                result = func(*args, **kwargs)
                logger.finish_operation(success=True)
                return result
                
            except Exception as e:
                logger.error_with_context(e, op_name)
                logger.finish_operation(success=False)
                raise
        
        return wrapper
    return decorator

# =============== GLOBAL INSTANCE ===============

_enhanced_logger = None

def get_enhanced_logger(name: str = "insights_enhanced") -> EnhancedLogger:
    """Obter instância global do enhanced logger"""
    global _enhanced_logger
    if _enhanced_logger is None:
        _enhanced_logger = EnhancedLogger(name)
    return _enhanced_logger

def reset_enhanced_logger():
    """Reset do logger global (útil para testes)"""
    global _enhanced_logger
    _enhanced_logger = None 