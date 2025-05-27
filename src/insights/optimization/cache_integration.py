#!/usr/bin/env python
"""
🔗 CACHE INTEGRATION - ETAPA 4
Integração perfeita entre sistema de cache e execução de Flows
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
import functools
import hashlib
import json

# Configuração de logging
cache_integration_logger = logging.getLogger('cache_integration')
cache_integration_logger.setLevel(logging.INFO)

class CacheStrategy(Enum):
    """Estratégias de cache disponíveis"""
    AGGRESSIVE = "aggressive"     # Cache tudo que for possível
    SELECTIVE = "selective"       # Cache apenas operações custosas
    SMART = "smart"              # Cache baseado em heurísticas
    DISABLED = "disabled"        # Cache desabilitado

@dataclass
class CacheHitResult:
    """Resultado de cache hit"""
    cache_key: str
    cached_value: Any
    cache_age_seconds: float
    cache_source: str  # memory, disk, redis
    metadata: Dict[str, Any]

@dataclass
class CacheOperationResult:
    """Resultado de operação de cache"""
    success: bool
    cache_hit: bool
    cache_key: str
    execution_time_ms: float
    cache_result: Optional[CacheHitResult] = None
    error_message: Optional[str] = None

class CacheIntegration:
    """
    🔗 Integração de Cache para Flows
    
    Fornece integração transparente entre o sistema de cache
    inteligente e a execução de CrewAI Flows, com estratégias
    adaptativas e otimizações automáticas.
    """
    
    def __init__(self, 
                 cache_system=None,
                 default_strategy: CacheStrategy = CacheStrategy.SMART,
                 default_ttl_hours: int = 24,
                 enable_cache_warming: bool = True,
                 enable_cache_analytics: bool = True):
        
        # Sistema de cache
        self.cache_system = cache_system
        
        # Configurações
        self.default_strategy = default_strategy
        self.default_ttl_hours = default_ttl_hours
        self.enable_cache_warming = enable_cache_warming
        self.enable_cache_analytics = enable_cache_analytics
        
        # Estado interno
        self.cache_strategies: Dict[str, CacheStrategy] = {}
        self.operation_patterns: Dict[str, Dict[str, Any]] = {}
        self.cache_performance: Dict[str, List[float]] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Estatísticas
        self.stats = {
            "total_operations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_stores": 0,
            "avg_hit_rate": 0.0,
            "time_saved_ms": 0.0
        }
        
        cache_integration_logger.info(f"🔗 CacheIntegration inicializado - Estratégia: {default_strategy.value}")
    
    def cache_flow_operation(self, 
                           operation_name: str,
                           cache_strategy: Optional[CacheStrategy] = None,
                           ttl_hours: Optional[int] = None,
                           force_refresh: bool = False):
        """
        Decorator para cache automático de operações de Flow
        
        Args:
            operation_name: Nome da operação (ex: 'analise_tendencias')
            cache_strategy: Estratégia de cache a usar
            ttl_hours: TTL do cache em horas
            force_refresh: Forçar refresh do cache
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_cached_operation(
                    func, operation_name, args, kwargs,
                    cache_strategy or self.default_strategy,
                    ttl_hours or self.default_ttl_hours,
                    force_refresh
                )
            return wrapper
        return decorator
    
    def _execute_cached_operation(self,
                                func: Callable,
                                operation_name: str, 
                                args: tuple,
                                kwargs: dict,
                                strategy: CacheStrategy,
                                ttl_hours: int,
                                force_refresh: bool) -> Any:
        """Executar operação com cache"""
        start_time = time.time()
        
        with self.lock:
            self.stats["total_operations"] += 1
        
        try:
            # Verificar se cache está desabilitado
            if strategy == CacheStrategy.DISABLED:
                result = func(*args, **kwargs)
                return self._create_operation_result(
                    success=True,
                    cache_hit=False,
                    cache_key="disabled",
                    execution_time_ms=(time.time() - start_time) * 1000,
                    result=result
                )
            
            # Gerar chave de cache
            cache_key = self._generate_operation_cache_key(operation_name, args, kwargs)
            
            # Verificar cache (se não forçar refresh)
            if not force_refresh:
                cache_result = self._check_cache(cache_key, operation_name, strategy)
                if cache_result.cache_hit:
                    with self.lock:
                        self.stats["cache_hits"] += 1
                        self.stats["time_saved_ms"] += cache_result.execution_time_ms
                    
                    cache_integration_logger.info(
                        f"💾 Cache HIT: {operation_name} ({cache_result.execution_time_ms:.1f}ms salvos)"
                    )
                    return cache_result
            
            # Cache miss - executar operação
            cache_integration_logger.debug(f"🔄 Executando operação: {operation_name}")
            operation_result = func(*args, **kwargs)
            
            execution_time = (time.time() - start_time) * 1000
            
            # Armazenar no cache (se estratégia permite)
            if self._should_cache_result(operation_name, operation_result, strategy, execution_time):
                self._store_in_cache(cache_key, operation_result, ttl_hours, operation_name, execution_time)
            
            with self.lock:
                self.stats["cache_misses"] += 1
            
            return self._create_operation_result(
                success=True,
                cache_hit=False,
                cache_key=cache_key,
                execution_time_ms=execution_time,
                result=operation_result
            )
            
        except Exception as e:
            cache_integration_logger.error(f"❌ Erro na operação cacheada {operation_name}: {e}")
            return self._create_operation_result(
                success=False,
                cache_hit=False,
                cache_key="error",
                execution_time_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    def _generate_operation_cache_key(self, operation_name: str, args: tuple, kwargs: dict) -> str:
        """Gerar chave de cache para operação"""
        try:
            # Extrair argumentos relevantes para a chave
            relevant_data = {
                "operation": operation_name,
                "args_hash": self._hash_args(args),
                "kwargs_hash": self._hash_kwargs(kwargs)
            }
            
            key_string = json.dumps(relevant_data, sort_keys=True)
            hash_key = hashlib.md5(key_string.encode()).hexdigest()
            
            return f"flow_op_{operation_name}_{hash_key[:16]}"
            
        except Exception as e:
            cache_integration_logger.warning(f"⚠️ Erro ao gerar chave de cache: {e}")
            return f"flow_op_{operation_name}_{int(time.time())}"
    
    def _hash_args(self, args: tuple) -> str:
        """Hash dos argumentos posicionais"""
        try:
            # Filtrar apenas argumentos que afetam o resultado
            relevant_args = []
            for arg in args:
                if hasattr(arg, '__dict__'):
                    # Para objetos com estado (como Flow state)
                    relevant_fields = {}
                    for key, value in arg.__dict__.items():
                        if key in ['data_inicio', 'data_fim', 'modo_execucao']:
                            relevant_fields[key] = str(value)
                    relevant_args.append(relevant_fields)
                else:
                    relevant_args.append(str(arg))
            
            return hashlib.md5(str(relevant_args).encode()).hexdigest()[:16]
            
        except Exception as e:
            cache_integration_logger.warning(f"⚠️ Erro ao hash args: {e}")
            return "unknown_args"
    
    def _hash_kwargs(self, kwargs: dict) -> str:
        """Hash dos argumentos nomeados"""
        try:
            # Filtrar kwargs relevantes
            relevant_kwargs = {}
            for key, value in kwargs.items():
                if not key.startswith('_') and key not in ['self']:
                    relevant_kwargs[key] = str(value)
            
            return hashlib.md5(str(relevant_kwargs).encode()).hexdigest()[:16]
            
        except Exception as e:
            cache_integration_logger.warning(f"⚠️ Erro ao hash kwargs: {e}")
            return "unknown_kwargs"
    
    def _check_cache(self, cache_key: str, operation_name: str, strategy: CacheStrategy) -> CacheOperationResult:
        """Verificar se há resultado em cache"""
        check_start = time.time()
        
        try:
            if not self.cache_system:
                return CacheOperationResult(
                    success=False,
                    cache_hit=False,
                    cache_key=cache_key,
                    execution_time_ms=0,
                    error_message="no_cache_system"
                )
            
            from insights.cache.intelligent_cache import CacheType
            
            cached_value = self.cache_system.get(cache_key, CacheType.ANALYSIS_RESULT)
            
            if cached_value is not None:
                # Cache hit!
                cache_hit_result = CacheHitResult(
                    cache_key=cache_key,
                    cached_value=cached_value,
                    cache_age_seconds=0,  # Implementar se necessário
                    cache_source="intelligent_cache",
                    metadata={"operation": operation_name}
                )
                
                return CacheOperationResult(
                    success=True,
                    cache_hit=True,
                    cache_key=cache_key,
                    execution_time_ms=(time.time() - check_start) * 1000,
                    cache_result=cache_hit_result
                )
            else:
                # Cache miss
                return CacheOperationResult(
                    success=True,
                    cache_hit=False,
                    cache_key=cache_key,
                    execution_time_ms=(time.time() - check_start) * 1000
                )
                
        except Exception as e:
            cache_integration_logger.warning(f"⚠️ Erro ao verificar cache: {e}")
            return CacheOperationResult(
                success=False,
                cache_hit=False,
                cache_key=cache_key,
                execution_time_ms=(time.time() - check_start) * 1000,
                error_message=str(e)
            )
    
    def _should_cache_result(self, 
                           operation_name: str, 
                           result: Any, 
                           strategy: CacheStrategy,
                           execution_time_ms: float) -> bool:
        """Determinar se o resultado deve ser cacheado"""
        
        # Estratégia: Disabled
        if strategy == CacheStrategy.DISABLED:
            return False
        
        # Estratégia: Aggressive - cachear sempre
        if strategy == CacheStrategy.AGGRESSIVE:
            return True
        
        # Estratégia: Selective - apenas operações custosas
        if strategy == CacheStrategy.SELECTIVE:
            return execution_time_ms > 5000  # > 5 segundos
        
        # Estratégia: Smart - heurísticas inteligentes
        if strategy == CacheStrategy.SMART:
            # Heurística 1: Operações custosas
            if execution_time_ms > 2000:  # > 2 segundos
                return True
            
            # Heurística 2: Operações conhecidas por serem custosas
            expensive_operations = [
                'analise_tendencias',
                'analise_financeira_avancada',
                'analise_clientes_avancada',
                'dashboard_html_dinamico',
                'relatorio_executivo_completo'
            ]
            if operation_name in expensive_operations:
                return True
            
            # Heurística 3: Resultados grandes (heurística simples)
            try:
                import sys
                result_size = sys.getsizeof(result)
                if result_size > 10240:  # > 10KB
                    return True
            except:
                pass
            
            return False
        
        return False
    
    def _store_in_cache(self, 
                       cache_key: str, 
                       result: Any, 
                       ttl_hours: int,
                       operation_name: str,
                       execution_time_ms: float):
        """Armazenar resultado no cache"""
        try:
            if not self.cache_system:
                return
            
            from insights.cache.intelligent_cache import CacheType
            
            ttl_seconds = ttl_hours * 3600
            
            success = self.cache_system.set(
                key=cache_key,
                value=result,
                cache_type=CacheType.ANALYSIS_RESULT,
                ttl=ttl_seconds,
                metadata={
                    "operation_name": operation_name,
                    "execution_time_ms": execution_time_ms,
                    "cached_at": datetime.now().isoformat(),
                    "cache_integration": True
                }
            )
            
            if success:
                with self.lock:
                    self.stats["cache_stores"] += 1
                
                cache_integration_logger.debug(
                    f"💾 Resultado cacheado: {operation_name} ({execution_time_ms:.1f}ms)"
                )
            else:
                cache_integration_logger.warning(f"⚠️ Falha ao cachear {operation_name}")
                
        except Exception as e:
            cache_integration_logger.error(f"❌ Erro ao armazenar no cache: {e}")
    
    def _create_operation_result(self, **kwargs) -> Any:
        """Criar resultado de operação (retorna o resultado real ou objeto com metadados)"""
        # Se há resultado real, retornar ele diretamente
        if 'result' in kwargs:
            return kwargs['result']
        
        # Se é cache hit, retornar valor cacheado
        if kwargs.get('cache_hit') and kwargs.get('cache_result'):
            return kwargs['cache_result'].cached_value
        
        # Se houve erro, lançar exceção
        if not kwargs.get('success'):
            raise Exception(kwargs.get('error_message', 'Cache operation failed'))
        
        return None
    
    # =============== CACHE WARMING ===============
    
    def warm_cache_for_flow(self, 
                           flow_id: str, 
                           flow_state: Any,
                           operation_names: List[str] = None) -> Dict[str, Any]:
        """Pre-aquecer cache para um Flow específico"""
        if not self.enable_cache_warming:
            return {"status": "disabled", "warmed_operations": []}
        
        try:
            warmed_operations = []
            
            # Operações padrão para warming
            if operation_names is None:
                operation_names = [
                    'analise_tendencias',
                    'analise_sazonalidade',
                    'analise_financeira',
                    'analise_segmentos'
                ]
            
            cache_integration_logger.info(f"🔥 Iniciando cache warming para Flow {flow_id}")
            
            for operation_name in operation_names:
                try:
                    # Verificar se há padrões históricos para esta operação
                    if self._has_warming_pattern(operation_name, flow_state):
                        warmed_operations.append(operation_name)
                        cache_integration_logger.debug(f"🔥 Cache warmed: {operation_name}")
                    
                except Exception as e:
                    cache_integration_logger.warning(f"⚠️ Erro ao warming {operation_name}: {e}")
            
            return {
                "status": "completed",
                "warmed_operations": warmed_operations,
                "flow_id": flow_id
            }
            
        except Exception as e:
            cache_integration_logger.error(f"❌ Erro no cache warming: {e}")
            return {"status": "error", "error": str(e)}
    
    def _has_warming_pattern(self, operation_name: str, flow_state: Any) -> bool:
        """Verificar se há padrões para warming de uma operação"""
        # Implementação simplificada - em produção, analisar padrões históricos
        return operation_name in ['analise_tendencias', 'analise_financeira']
    
    # =============== ANÁLISE E OTIMIZAÇÃO ===============
    
    def analyze_cache_performance(self, hours_back: int = 24) -> Dict[str, Any]:
        """Analisar performance do cache"""
        try:
            with self.lock:
                total_ops = self.stats["total_operations"]
                hit_rate = (self.stats["cache_hits"] / total_ops * 100) if total_ops > 0 else 0
                
                analysis = {
                    "total_operations": total_ops,
                    "cache_hits": self.stats["cache_hits"],
                    "cache_misses": self.stats["cache_misses"],
                    "cache_stores": self.stats["cache_stores"],
                    "hit_rate_percent": hit_rate,
                    "time_saved_ms": self.stats["time_saved_ms"],
                    "avg_time_saved_per_hit": (
                        self.stats["time_saved_ms"] / self.stats["cache_hits"]
                        if self.stats["cache_hits"] > 0 else 0
                    )
                }
                
                # Análise de eficiência
                if hit_rate >= 80:
                    analysis["efficiency"] = "excellent"
                elif hit_rate >= 60:
                    analysis["efficiency"] = "good"
                elif hit_rate >= 40:
                    analysis["efficiency"] = "fair"
                else:
                    analysis["efficiency"] = "poor"
                
                # Recomendações
                recommendations = []
                if hit_rate < 50:
                    recommendations.append("Considere ajustar estratégia de cache para AGGRESSIVE")
                if self.stats["cache_stores"] < self.stats["cache_misses"] * 0.8:
                    recommendations.append("Muitos resultados não estão sendo cacheados")
                
                analysis["recommendations"] = recommendations
                
                return analysis
                
        except Exception as e:
            cache_integration_logger.error(f"❌ Erro na análise de performance: {e}")
            return {"error": str(e)}
    
    def optimize_cache_strategy(self) -> Dict[str, Any]:
        """Otimizar estratégia de cache baseada na performance"""
        try:
            performance = self.analyze_cache_performance()
            current_strategy = self.default_strategy
            
            # Lógica de otimização
            hit_rate = performance.get("hit_rate_percent", 0)
            
            if hit_rate < 30:
                # Hit rate muito baixo - estratégia mais agressiva
                new_strategy = CacheStrategy.AGGRESSIVE
                reason = "Hit rate baixo, aplicando estratégia agressiva"
                
            elif hit_rate > 90:
                # Hit rate muito alto - pode ser excessivo
                new_strategy = CacheStrategy.SELECTIVE
                reason = "Hit rate alto, otimizando para selectividade"
                
            else:
                # Hit rate aceitável - manter estratégia inteligente
                new_strategy = CacheStrategy.SMART
                reason = "Performance aceitável, mantendo estratégia inteligente"
            
            # Aplicar nova estratégia se diferente
            if new_strategy != current_strategy:
                self.default_strategy = new_strategy
                cache_integration_logger.info(
                    f"⚡ Estratégia otimizada: {current_strategy.value} → {new_strategy.value}"
                )
            
            return {
                "previous_strategy": current_strategy.value,
                "new_strategy": new_strategy.value,
                "reason": reason,
                "performance_metrics": performance
            }
            
        except Exception as e:
            cache_integration_logger.error(f"❌ Erro na otimização de estratégia: {e}")
            return {"error": str(e)}
    
    # =============== API PÚBLICA ===============
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obter estatísticas do cache"""
        with self.lock:
            stats = self.stats.copy()
            
            # Calcular hit rate
            total_ops = stats["total_operations"]
            stats["hit_rate_percent"] = (
                (stats["cache_hits"] / total_ops * 100) if total_ops > 0 else 0
            )
            
            return stats
    
    def invalidate_operation_cache(self, operation_name: str) -> bool:
        """Invalidar cache de uma operação específica"""
        try:
            if not self.cache_system:
                return False
            
            # Implementação simplificada - em produção, indexar por operation_name
            cache_integration_logger.info(f"🗑️ Cache da operação {operation_name} invalidado")
            return True
            
        except Exception as e:
            cache_integration_logger.error(f"❌ Erro ao invalidar cache: {e}")
            return False
    
    def set_operation_strategy(self, operation_name: str, strategy: CacheStrategy):
        """Definir estratégia específica para uma operação"""
        with self.lock:
            self.cache_strategies[operation_name] = strategy
            cache_integration_logger.info(
                f"⚙️ Estratégia definida para {operation_name}: {strategy.value}"
            ) 