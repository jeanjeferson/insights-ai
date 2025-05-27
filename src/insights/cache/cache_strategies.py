#!/usr/bin/env python
"""
ETAPA 4 - ESTRATÉGIAS DE CACHE
Diferentes estratégias de cache para otimização
"""

import time
import math
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from .intelligent_cache import CacheEntry, CacheType

# Configuração de logging
strategy_logger = logging.getLogger('cache_strategy')
strategy_logger.setLevel(logging.INFO)

class CacheStrategy(ABC):
    """Estratégia abstrata de cache"""
    
    def __init__(self, name: str):
        self.name = name
        self.created_at = datetime.now()
        self.statistics = {
            "applications": 0,
            "evictions": 0,
            "promotions": 0,
            "demotions": 0
        }
    
    @abstractmethod
    def should_cache(self, key: str, value: Any, cache_type: CacheType, metadata: Dict[str, Any]) -> bool:
        """Decidir se deve cachear este item"""
        pass
    
    @abstractmethod
    def calculate_ttl(self, key: str, value: Any, cache_type: CacheType, metadata: Dict[str, Any]) -> Optional[float]:
        """Calcular TTL para este item"""
        pass
    
    @abstractmethod
    def should_evict(self, entry: CacheEntry, current_memory_usage: int, max_memory: int) -> bool:
        """Decidir se deve remover entrada da memória"""
        pass
    
    @abstractmethod
    def get_priority_score(self, entry: CacheEntry) -> float:
        """Calcular score de prioridade para ordenação"""
        pass
    
    def apply_strategy(self, entries: List[CacheEntry], **kwargs) -> List[str]:
        """Aplicar estratégia e retornar chaves para eviction"""
        self.statistics["applications"] += 1
        return []

class TTLStrategy(CacheStrategy):
    """Estratégia baseada em Time To Live"""
    
    def __init__(self, base_ttl: float = 3600, ttl_multipliers: Dict[CacheType, float] = None):
        super().__init__("TTL")
        self.base_ttl = base_ttl
        self.ttl_multipliers = ttl_multipliers or {
            CacheType.ANALYSIS_RESULT: 2.0,    # Análises duram mais
            CacheType.CREW_INSTANCE: 0.5,      # Crews são mais voláteis
            CacheType.DATA_QUERY: 1.0,         # Queries padrão
            CacheType.PROCESSED_DATA: 1.5,     # Dados processados duram mais
            CacheType.MODEL_OUTPUT: 3.0,       # Outputs de modelo duram muito
            CacheType.DASHBOARD_DATA: 0.3      # Dashboard precisa ser atualizado frequentemente
        }
    
    def should_cache(self, key: str, value: Any, cache_type: CacheType, metadata: Dict[str, Any]) -> bool:
        """TTL sempre cacheia, mas com TTLs diferentes"""
        return True
    
    def calculate_ttl(self, key: str, value: Any, cache_type: CacheType, metadata: Dict[str, Any]) -> Optional[float]:
        """Calcular TTL baseado no tipo"""
        multiplier = self.ttl_multipliers.get(cache_type, 1.0)
        
        # Ajustar baseado em metadados
        if metadata:
            # Se tem confidence score, ajustar TTL
            confidence = metadata.get('confidence_score', 100)
            confidence_multiplier = confidence / 100.0
            
            # Se teve muito acesso recente, aumentar TTL
            recent_access = metadata.get('recent_access_count', 0)
            if recent_access > 10:
                multiplier *= 1.5
            
            # Se é análise crítica, aumentar TTL
            if metadata.get('is_critical', False):
                multiplier *= 2.0
                
            multiplier *= confidence_multiplier
        
        return self.base_ttl * multiplier
    
    def should_evict(self, entry: CacheEntry, current_memory_usage: int, max_memory: int) -> bool:
        """Evict se expirado ou se memória está > 90% cheia"""
        if entry.is_expired():
            return True
        
        memory_pressure = current_memory_usage / max_memory
        if memory_pressure > 0.9:
            # Sob pressão, evict itens com TTL baixo restante
            remaining_ttl = entry.ttl - (time.time() - entry.created_at.timestamp())
            return remaining_ttl < (entry.ttl * 0.1)  # Menos de 10% do TTL restante
        
        return False
    
    def get_priority_score(self, entry: CacheEntry) -> float:
        """Score baseado no tempo restante de TTL"""
        if entry.ttl is None:
            return float('inf')
        
        remaining_ttl = entry.ttl - (time.time() - entry.created_at.timestamp())
        return remaining_ttl

class LRUStrategy(CacheStrategy):
    """Estratégia Least Recently Used"""
    
    def __init__(self, access_weight: float = 0.7, frequency_weight: float = 0.3):
        super().__init__("LRU")
        self.access_weight = access_weight
        self.frequency_weight = frequency_weight
    
    def should_cache(self, key: str, value: Any, cache_type: CacheType, metadata: Dict[str, Any]) -> bool:
        """LRU cacheia tudo"""
        return True
    
    def calculate_ttl(self, key: str, value: Any, cache_type: CacheType, metadata: Dict[str, Any]) -> Optional[float]:
        """LRU não usa TTL fixo"""
        return None
    
    def should_evict(self, entry: CacheEntry, current_memory_usage: int, max_memory: int) -> bool:
        """Evict baseado em uso recente"""
        memory_pressure = current_memory_usage / max_memory
        
        if memory_pressure < 0.8:
            return False
        
        # Calcular score LRU
        now = time.time()
        last_access_age = now - entry.last_accessed.timestamp()
        
        # Itens não acessados há mais de 1 hora são candidatos
        if last_access_age > 3600:  # 1 hora
            return True
            
        # Sob alta pressão (>95%), ser mais agressivo
        if memory_pressure > 0.95:
            return last_access_age > 1800  # 30 minutos
            
        return False
    
    def get_priority_score(self, entry: CacheEntry) -> float:
        """Score baseado em recência e frequência de acesso"""
        now = time.time()
        
        # Componente de recência (quanto menor, melhor)
        last_access_age = now - entry.last_accessed.timestamp()
        recency_score = last_access_age
        
        # Componente de frequência (quanto maior, melhor)
        entry_age = now - entry.created_at.timestamp()
        access_frequency = entry.access_count / max(entry_age / 3600, 0.1)  # acessos por hora
        frequency_score = 1.0 / max(access_frequency, 0.1)  # Inverter para que menor seja melhor
        
        # Combinar scores
        combined_score = (self.access_weight * recency_score + 
                         self.frequency_weight * frequency_score)
        
        return combined_score

class AdaptiveStrategy(CacheStrategy):
    """Estratégia adaptativa que aprende com padrões de uso"""
    
    def __init__(self, learning_rate: float = 0.1, min_observations: int = 10):
        super().__init__("Adaptive")
        self.learning_rate = learning_rate
        self.min_observations = min_observations
        
        # Estatísticas de aprendizado por tipo de cache
        self.type_stats = {}
        for cache_type in CacheType:
            self.type_stats[cache_type] = {
                "hit_rate": 0.5,        # Taxa de hit inicial
                "avg_access_count": 1.0, # Média de acessos
                "avg_lifetime": 3600.0,  # Tempo de vida médio
                "observations": 0
            }
        
        # Padrões temporais aprendidos
        self.temporal_patterns = {
            "hourly_multipliers": [1.0] * 24,  # Multiplicadores por hora do dia
            "daily_multipliers": [1.0] * 7,    # Multiplicadores por dia da semana
        }
    
    def should_cache(self, key: str, value: Any, cache_type: CacheType, metadata: Dict[str, Any]) -> bool:
        """Decidir baseado na taxa de hit aprendida"""
        stats = self.type_stats[cache_type]
        
        # Se ainda não temos observações suficientes, cachear
        if stats["observations"] < self.min_observations:
            return True
        
        # Cachear se a taxa de hit for boa
        return stats["hit_rate"] > 0.3
    
    def calculate_ttl(self, key: str, value: Any, cache_type: CacheType, metadata: Dict[str, Any]) -> Optional[float]:
        """TTL adaptativo baseado em padrões aprendidos"""
        stats = self.type_stats[cache_type]
        base_ttl = stats["avg_lifetime"]
        
        # Ajustar baseado na hora atual
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        
        temporal_multiplier = (
            self.temporal_patterns["hourly_multipliers"][current_hour] *
            self.temporal_patterns["daily_multipliers"][current_day]
        )
        
        # Ajustar baseado na frequência de acesso esperada
        if stats["avg_access_count"] > 5:
            access_multiplier = 1.5
        elif stats["avg_access_count"] < 2:
            access_multiplier = 0.7
        else:
            access_multiplier = 1.0
        
        # Combinar multiplicadores
        final_ttl = base_ttl * temporal_multiplier * access_multiplier
        
        # Limites mínimo e máximo
        return max(300, min(86400, final_ttl))  # Entre 5 minutos e 24 horas
    
    def should_evict(self, entry: CacheEntry, current_memory_usage: int, max_memory: int) -> bool:
        """Eviction inteligente baseada em padrões"""
        memory_pressure = current_memory_usage / max_memory
        
        if memory_pressure < 0.7:
            return False
        
        # Calcular score de valor
        value_score = self._calculate_value_score(entry)
        
        # Thresholds adaptativos baseados na pressão de memória
        if memory_pressure > 0.95:
            threshold = 0.2  # Muito agressivo
        elif memory_pressure > 0.85:
            threshold = 0.4  # Moderado
        else:
            threshold = 0.6  # Conservador
        
        return value_score < threshold
    
    def get_priority_score(self, entry: CacheEntry) -> float:
        """Score adaptativo considerando múltiplos fatores"""
        return 1.0 / max(self._calculate_value_score(entry), 0.01)
    
    def learn_from_access(self, entry: CacheEntry, was_hit: bool):
        """Aprender com padrões de acesso"""
        cache_type = entry.cache_type
        stats = self.type_stats[cache_type]
        
        # Atualizar taxa de hit
        old_hit_rate = stats["hit_rate"]
        if was_hit:
            new_hit_rate = old_hit_rate + self.learning_rate * (1.0 - old_hit_rate)
        else:
            new_hit_rate = old_hit_rate + self.learning_rate * (0.0 - old_hit_rate)
        
        stats["hit_rate"] = new_hit_rate
        
        # Atualizar média de acessos
        old_avg_access = stats["avg_access_count"]
        new_avg_access = old_avg_access + self.learning_rate * (entry.access_count - old_avg_access)
        stats["avg_access_count"] = new_avg_access
        
        # Atualizar tempo de vida médio
        entry_lifetime = time.time() - entry.created_at.timestamp()
        old_avg_lifetime = stats["avg_lifetime"]
        new_avg_lifetime = old_avg_lifetime + self.learning_rate * (entry_lifetime - old_avg_lifetime)
        stats["avg_lifetime"] = new_avg_lifetime
        
        stats["observations"] += 1
        
        # Aprender padrões temporais
        self._learn_temporal_patterns(entry)
    
    def _calculate_value_score(self, entry: CacheEntry) -> float:
        """Calcular score de valor da entrada"""
        now = time.time()
        
        # Fator de recência (0-1, maior é melhor)
        last_access_age = now - entry.last_accessed.timestamp()
        recency_factor = 1.0 / (1.0 + last_access_age / 3600)  # Decai em 1 hora
        
        # Fator de frequência (0-1, maior é melhor)
        entry_age = now - entry.created_at.timestamp()
        access_rate = entry.access_count / max(entry_age / 3600, 0.1)
        frequency_factor = min(access_rate / 10.0, 1.0)  # Normalizar para max 10 acessos/hora
        
        # Fator de tipo (baseado na taxa de hit aprendida)
        type_factor = self.type_stats[entry.cache_type]["hit_rate"]
        
        # Fator de tamanho (menores têm vantagem)
        size_factor = 1.0 / (1.0 + entry.size_bytes / (1024 * 1024))  # Normalizar por MB
        
        # Combinar fatores
        value_score = (
            0.3 * recency_factor +
            0.3 * frequency_factor +
            0.2 * type_factor +
            0.2 * size_factor
        )
        
        return value_score
    
    def _learn_temporal_patterns(self, entry: CacheEntry):
        """Aprender padrões temporais de acesso"""
        access_hour = entry.last_accessed.hour
        access_day = entry.last_accessed.weekday()
        
        # Incrementar multiplicadores baseado no sucesso do acesso
        access_success = min(entry.access_count / 5.0, 1.0)  # Normalizar
        
        # Atualizar multiplicador horário
        old_hourly = self.temporal_patterns["hourly_multipliers"][access_hour]
        new_hourly = old_hourly + self.learning_rate * (access_success - old_hourly)
        self.temporal_patterns["hourly_multipliers"][access_hour] = new_hourly
        
        # Atualizar multiplicador diário
        old_daily = self.temporal_patterns["daily_multipliers"][access_day]
        new_daily = old_daily + self.learning_rate * (access_success - old_daily)
        self.temporal_patterns["daily_multipliers"][access_day] = new_daily

class HybridStrategy(CacheStrategy):
    """Estratégia híbrida que combina múltiplas estratégias"""
    
    def __init__(self, strategies: List[CacheStrategy], weights: List[float] = None):
        super().__init__("Hybrid")
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
        
        if len(self.weights) != len(self.strategies):
            raise ValueError("Número de pesos deve ser igual ao número de estratégias")
    
    def should_cache(self, key: str, value: Any, cache_type: CacheType, metadata: Dict[str, Any]) -> bool:
        """Decisão baseada na votação ponderada das estratégias"""
        votes = []
        for strategy in self.strategies:
            votes.append(1.0 if strategy.should_cache(key, value, cache_type, metadata) else 0.0)
        
        weighted_sum = sum(vote * weight for vote, weight in zip(votes, self.weights))
        return weighted_sum > 0.5
    
    def calculate_ttl(self, key: str, value: Any, cache_type: CacheType, metadata: Dict[str, Any]) -> Optional[float]:
        """TTL baseado na média ponderada das estratégias"""
        ttls = []
        valid_weights = []
        
        for strategy, weight in zip(self.strategies, self.weights):
            ttl = strategy.calculate_ttl(key, value, cache_type, metadata)
            if ttl is not None:
                ttls.append(ttl)
                valid_weights.append(weight)
        
        if not ttls:
            return None
        
        # Normalizar pesos
        total_weight = sum(valid_weights)
        if total_weight == 0:
            return sum(ttls) / len(ttls)
        
        weighted_sum = sum(ttl * weight for ttl, weight in zip(ttls, valid_weights))
        return weighted_sum / total_weight
    
    def should_evict(self, entry: CacheEntry, current_memory_usage: int, max_memory: int) -> bool:
        """Eviction baseada na votação das estratégias"""
        votes = []
        for strategy in self.strategies:
            votes.append(1.0 if strategy.should_evict(entry, current_memory_usage, max_memory) else 0.0)
        
        weighted_sum = sum(vote * weight for vote, weight in zip(votes, self.weights))
        return weighted_sum > 0.5
    
    def get_priority_score(self, entry: CacheEntry) -> float:
        """Score baseado na média ponderada das estratégias"""
        scores = []
        for strategy in self.strategies:
            scores.append(strategy.get_priority_score(entry))
        
        weighted_sum = sum(score * weight for score, weight in zip(scores, self.weights))
        return weighted_sum

# ========== FACTORY DE ESTRATÉGIAS ==========

class StrategyFactory:
    """Factory para criar estratégias de cache"""
    
    @staticmethod
    def create_default_strategy() -> CacheStrategy:
        """Criar estratégia padrão (híbrida)"""
        ttl_strategy = TTLStrategy()
        lru_strategy = LRUStrategy()
        adaptive_strategy = AdaptiveStrategy()
        
        return HybridStrategy(
            strategies=[ttl_strategy, lru_strategy, adaptive_strategy],
            weights=[0.4, 0.3, 0.3]
        )
    
    @staticmethod
    def create_performance_strategy() -> CacheStrategy:
        """Estratégia focada em performance (mais agressiva)"""
        return TTLStrategy(
            base_ttl=7200,  # 2 horas
            ttl_multipliers={
                CacheType.ANALYSIS_RESULT: 4.0,
                CacheType.CREW_INSTANCE: 1.0,
                CacheType.DATA_QUERY: 2.0,
                CacheType.PROCESSED_DATA: 3.0,
                CacheType.MODEL_OUTPUT: 6.0,
                CacheType.DASHBOARD_DATA: 0.5
            }
        )
    
    @staticmethod
    def create_memory_conservative_strategy() -> CacheStrategy:
        """Estratégia conservadora de memória"""
        return LRUStrategy(access_weight=0.8, frequency_weight=0.2)
    
    @staticmethod
    def create_learning_strategy() -> CacheStrategy:
        """Estratégia de aprendizado agressivo"""
        return AdaptiveStrategy(learning_rate=0.2, min_observations=5) 