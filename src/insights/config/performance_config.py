"""
⚡ CONFIGURAÇÃO DE PERFORMANCE E LOGGING OTIMIZADO
================================================

Sistema de configuração para otimização de performance e logging estruturado.
Reduz verbosidade e melhora performance de inicialização.
"""

import os
import logging
import time
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

# =============== NÍVEIS DE LOGGING ESTRUTURADOS ===============

class LogLevel(Enum):
    """Níveis de logging otimizados"""
    SILENT = 0      # Apenas erros críticos
    MINIMAL = 1     # Resumos essenciais
    NORMAL = 2      # Logs importantes
    VERBOSE = 3     # Todos os logs (desenvolvimento)
    DEBUG = 4       # Logs de debug (apenas desenvolvimento)

# =============== CONFIGURAÇÃO DE PERFORMANCE ===============

@dataclass
class PerformanceSettings:
    """Configurações de performance"""
    
    # Logging
    log_level: LogLevel = LogLevel.NORMAL
    enable_file_logging: bool = True
    log_flush_frequency: int = 10  # Flush a cada N logs
    log_buffer_size: int = 100
    
    # Inicialização
    lazy_tool_loading: bool = True
    parallel_agent_init: bool = True
    cache_validations: bool = True
    skip_non_critical_validations: bool = True
    
    # Cache
    enable_tool_cache: bool = True
    cache_timeout_seconds: int = 3600
    cache_max_size: int = 100
    
    # Performance
    max_concurrent_agents: int = 4
    tool_initialization_timeout: int = 30
    validation_batch_size: int = 5

# =============== CONFIGURAÇÃO BASEADA EM ENVIRONMENT ===============

def get_performance_config() -> PerformanceSettings:
    """Obter configurações baseadas no ambiente"""
    
    # Detectar ambiente
    is_production = os.getenv("ENVIRONMENT", "development") == "production"
    is_testing = os.getenv("PYTEST_CURRENT_TEST") is not None
    is_debug = os.getenv("INSIGHTS_DEBUG", "false").lower() == "true"
    
    if is_testing:
        # Configuração para testes - máxima performance
        return PerformanceSettings(
            log_level=LogLevel.MINIMAL,
            enable_file_logging=False,
            lazy_tool_loading=True,
            parallel_agent_init=True,
            cache_validations=True,
            skip_non_critical_validations=True,
            max_concurrent_agents=2,
            log_flush_frequency=50
        )
    elif is_production:
        # Configuração para produção - balanceada
        return PerformanceSettings(
            log_level=LogLevel.NORMAL,
            enable_file_logging=True,
            lazy_tool_loading=True,
            parallel_agent_init=True,
            cache_validations=True,
            skip_non_critical_validations=False,
            max_concurrent_agents=4,
            log_flush_frequency=20
        )
    elif is_debug:
        # Configuração para debug - máxima verbosidade
        return PerformanceSettings(
            log_level=LogLevel.DEBUG,
            enable_file_logging=True,
            lazy_tool_loading=False,
            parallel_agent_init=False,
            cache_validations=False,
            skip_non_critical_validations=False,
            max_concurrent_agents=1,
            log_flush_frequency=1
        )
    else:
        # Configuração padrão - desenvolvimento
        return PerformanceSettings(
            log_level=LogLevel.NORMAL,
            enable_file_logging=True,
            lazy_tool_loading=True,
            parallel_agent_init=True,
            cache_validations=True,
            skip_non_critical_validations=True,
            max_concurrent_agents=4,
            log_flush_frequency=10
        )

# =============== LOGGER OTIMIZADO ===============

class OptimizedLogger:
    """Logger otimizado com buffer e níveis contextuais"""
    
    def __init__(self, name: str, config: PerformanceSettings):
        self.config = config
        self.logger = logging.getLogger(name)
        self.buffer = []
        self.last_flush = time.time()
        self.setup_logger()
    
    def setup_logger(self):
        """Configurar logger baseado na configuração"""
        
        # Limpar handlers existentes
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Definir nível baseado na configuração
        level_mapping = {
            LogLevel.SILENT: logging.CRITICAL,
            LogLevel.MINIMAL: logging.WARNING,
            LogLevel.NORMAL: logging.INFO,
            LogLevel.VERBOSE: logging.INFO,
            LogLevel.DEBUG: logging.DEBUG
        }
        
        self.logger.setLevel(level_mapping[self.config.log_level])
        
        # Console handler sempre presente (para erros críticos)
        console_handler = logging.StreamHandler()
        
        # Formato otimizado baseado no nível
        if self.config.log_level in [LogLevel.SILENT, LogLevel.MINIMAL]:
            formatter = logging.Formatter('%(levelname)s: %(message)s')
        elif self.config.log_level == LogLevel.NORMAL:
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', 
                                        datefmt='%H:%M:%S')
        else:  # VERBOSE, DEBUG
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s',
                datefmt='%H:%M:%S'
            )
        
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (se habilitado)
        if self.config.enable_file_logging:
            log_file = Path("logs/optimized_execution.log")
            log_file.parent.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Prevenir propagação
        self.logger.propagate = False
    
    def _should_log(self, level: str) -> bool:
        """Verificar se deve fazer log baseado na configuração"""
        
        level_priorities = {
            LogLevel.SILENT: ['CRITICAL'],
            LogLevel.MINIMAL: ['CRITICAL', 'ERROR', 'WARNING'],
            LogLevel.NORMAL: ['CRITICAL', 'ERROR', 'WARNING', 'INFO'],
            LogLevel.VERBOSE: ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'],
            LogLevel.DEBUG: ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
        }
        
        return level.upper() in level_priorities[self.config.log_level]
    
    def _log_with_buffer(self, level: str, message: str):
        """Log com buffer para performance"""
        
        if not self._should_log(level):
            return
        
        # Adicionar ao buffer
        log_entry = {
            'level': level,
            'message': message,
            'timestamp': time.time()
        }
        self.buffer.append(log_entry)
        
        # Flush baseado na configuração
        should_flush = (
            len(self.buffer) >= self.config.log_flush_frequency or
            time.time() - self.last_flush > 5.0 or  # Flush a cada 5s mínimo
            level in ['CRITICAL', 'ERROR']  # Flush imediato para erros
        )
        
        if should_flush:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush do buffer de logs"""
        if not self.buffer:
            return
        
        for entry in self.buffer:
            getattr(self.logger, entry['level'].lower())(entry['message'])
        
        self.buffer.clear()
        self.last_flush = time.time()
        
        # Flush dos handlers apenas se necessário
        if self.config.log_level in [LogLevel.DEBUG, LogLevel.VERBOSE]:
            for handler in self.logger.handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()
    
    # Métodos de logging
    def debug(self, msg: str):
        self._log_with_buffer('DEBUG', msg)
    
    def info(self, msg: str):
        self._log_with_buffer('INFO', msg)
    
    def warning(self, msg: str):
        self._log_with_buffer('WARNING', msg)
    
    def error(self, msg: str):
        self._log_with_buffer('ERROR', msg)
    
    def critical(self, msg: str):
        self._log_with_buffer('CRITICAL', msg)
    
    def performance(self, operation: str, duration: float, **metrics):
        """Log específico para performance"""
        if self.config.log_level in [LogLevel.VERBOSE, LogLevel.DEBUG]:
            details = " | ".join([f"{k}={v}" for k, v in metrics.items()])
            self.info(f"⚡ {operation}: {duration:.3f}s | {details}")
    
    def startup_summary(self, components: List[str], total_time: float):
        """Log resumo de inicialização"""
        if self.config.log_level != LogLevel.SILENT:
            self.info(f"🚀 Inicialização concluída: {len(components)} componentes em {total_time:.2f}s")
    
    def finalize(self):
        """Finalizar logger e fazer flush final"""
        self._flush_buffer()

# =============== CACHE DE PERFORMANCE ===============

class PerformanceCache:
    """Cache otimizado para performance"""
    
    def __init__(self, config: PerformanceSettings):
        self.config = config
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, float] = {}
        self.enabled = config.enable_tool_cache
    
    def get(self, key: str) -> Optional[Any]:
        """Obter valor do cache"""
        if not self.enabled or key not in self.cache:
            return None
        
        # Verificar se expirou
        if time.time() - self.timestamps[key] > self.config.cache_timeout_seconds:
            self.remove(key)
            return None
        
        return self.cache[key]
    
    def set(self, key: str, value: Any):
        """Definir valor no cache"""
        if not self.enabled:
            return
        
        # Limpar cache se necessário
        if len(self.cache) >= self.config.cache_max_size:
            self._cleanup_old_entries()
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def remove(self, key: str):
        """Remover entrada do cache"""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
    
    def _cleanup_old_entries(self):
        """Limpar entradas antigas"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.config.cache_timeout_seconds
        ]
        
        for key in expired_keys:
            self.remove(key)
        
        # Se ainda estiver cheio, remover as mais antigas
        if len(self.cache) >= self.config.cache_max_size:
            sorted_keys = sorted(self.timestamps.items(), key=lambda x: x[1])
            keys_to_remove = [key for key, _ in sorted_keys[:10]]  # Remove 10 mais antigas
            
            for key in keys_to_remove:
                self.remove(key)

# =============== INSTÂNCIAS GLOBAIS ===============

# Configuração padrão
PERFORMANCE_CONFIG = get_performance_config()

# Logger otimizado global
optimized_logger = OptimizedLogger('insights_optimized', PERFORMANCE_CONFIG)

# Cache global
performance_cache = PerformanceCache(PERFORMANCE_CONFIG)

# =============== DECORATORS DE PERFORMANCE ===============

def performance_tracked(operation_name: str = None):
    """Decorator para rastrear performance de funções"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                optimized_logger.performance(op_name, duration, status="success")
                return result
            except Exception as e:
                duration = time.time() - start_time
                optimized_logger.performance(op_name, duration, status="error", error=str(e))
                raise
        
        return wrapper
    return decorator

def cached_result(cache_key_func=None):
    """Decorator para cache de resultados"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not PERFORMANCE_CONFIG.enable_tool_cache:
                return func(*args, **kwargs)
            
            # Gerar chave do cache
            if cache_key_func:
                cache_key = cache_key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # Tentar obter do cache
            cached_result = performance_cache.get(cache_key)
            if cached_result is not None:
                optimized_logger.debug(f"Cache hit: {cache_key}")
                return cached_result
            
            # Executar função e cachear resultado
            result = func(*args, **kwargs)
            performance_cache.set(cache_key, result)
            optimized_logger.debug(f"Cache miss: {cache_key}")
            
            return result
        
        return wrapper
    return decorator

# =============== UTILITÁRIOS DE PERFORMANCE ===============

def log_startup_metrics(components: List[str], start_time: float):
    """Log otimizado de métricas de inicialização"""
    total_time = time.time() - start_time
    
    if PERFORMANCE_CONFIG.log_level == LogLevel.MINIMAL:
        optimized_logger.info(f"✅ Sistema iniciado ({total_time:.1f}s)")
    elif PERFORMANCE_CONFIG.log_level == LogLevel.NORMAL:
        optimized_logger.startup_summary(components, total_time)
    else:
        optimized_logger.info(f"🚀 Componentes inicializados: {', '.join(components)}")
        optimized_logger.info(f"⏱️ Tempo total: {total_time:.3f}s")

def should_skip_validation(validation_type: str) -> bool:
    """Verificar se deve pular validação baseada na configuração"""
    if not PERFORMANCE_CONFIG.skip_non_critical_validations:
        return False
    
    # Validações que podem ser puladas em produção
    non_critical_validations = [
        'tool_method_validation',
        'detailed_compatibility_check',
        'advanced_logging_setup',
        'system_info_collection'
    ]
    
    return validation_type in non_critical_validations

def get_optimized_tool_list(agent_role: str, all_tools: List) -> List:
    """Obter lista otimizada de ferramentas para um agente"""
    
    if not PERFORMANCE_CONFIG.lazy_tool_loading:
        return all_tools
    
    # Mapeamento otimizado de ferramentas essenciais por agente
    essential_tools_map = {
        'engenheiro_dados': 3,          # SQL, File, BI
        'analista_financeiro': 7,       # Financial analysis pack
        'especialista_clientes': 6,     # Customer analysis pack
        'analista_vendas_tendencias': 8, # Full statistical pack
        'especialista_produtos': 6,     # Product analysis pack
        'analista_estoque': 6,          # Inventory pack
        'analista_performance': 4,      # Performance pack
        'diretor_insights': 5           # Executive pack
    }
    
    max_tools = essential_tools_map.get(agent_role, len(all_tools))
    return all_tools[:max_tools]  # Retorna apenas as ferramentas essenciais

# =============== CLEANUP ===============

def cleanup_performance_resources():
    """Limpar recursos de performance"""
    optimized_logger.finalize()
    performance_cache.cache.clear()
    performance_cache.timestamps.clear() 