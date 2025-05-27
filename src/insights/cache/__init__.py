"""
ETAPA 4 - SISTEMA DE CACHE INTELIGENTE
Módulo de cache multinível para otimização de performance
"""

from .intelligent_cache import IntelligentCacheSystem, CacheType, CacheEntry
from .cache_strategies import CacheStrategy, TTLStrategy, LRUStrategy, AdaptiveStrategy
from .cache_analytics import CacheAnalytics

__all__ = [
    'IntelligentCacheSystem',
    'CacheType',
    'CacheEntry',
    'CacheStrategy',
    'TTLStrategy', 
    'LRUStrategy',
    'AdaptiveStrategy',
    'CacheAnalytics'
] 