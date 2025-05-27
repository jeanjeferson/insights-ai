#!/usr/bin/env python
"""
ETAPA 4 - SISTEMA DE CACHE INTELIGENTE
Sistema de cache multinível com estratégias adaptativas
"""

import json
import pickle
import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum

# Configuração de logging
cache_logger = logging.getLogger('intelligent_cache')
cache_logger.setLevel(logging.INFO)

class CacheLevel(Enum):
    """Níveis de cache disponíveis"""
    MEMORY = "memory"       # Cache em memória (mais rápido)
    DISK = "disk"          # Cache em disco (persistente)
    REDIS = "redis"        # Cache Redis (distribuído)
    HYBRID = "hybrid"      # Cache híbrido (memória + disco)

class CacheType(Enum):
    """Tipos de dados que podem ser cached"""
    ANALYSIS_RESULT = "analysis_result"
    CREW_INSTANCE = "crew_instance"
    DATA_QUERY = "data_query"
    PROCESSED_DATA = "processed_data"
    MODEL_OUTPUT = "model_output"
    DASHBOARD_DATA = "dashboard_data"

@dataclass
class CacheEntry:
    """Entrada do cache com metadados"""
    key: str
    value: Any
    cache_type: CacheType
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl: Optional[float]  # Time to live em segundos
    size_bytes: int
    metadata: Dict[str, Any]
    
    def is_expired(self) -> bool:
        """Verificar se a entrada expirou"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at.timestamp() > self.ttl
    
    def update_access(self):
        """Atualizar informações de acesso"""
        self.last_accessed = datetime.now()
        self.access_count += 1

class IntelligentCacheSystem:
    """Sistema de cache inteligente multinível"""
    
    def __init__(self, 
                 cache_dir: str = "data/cache",
                 max_memory_size: int = 100 * 1024 * 1024,  # 100MB
                 max_disk_size: int = 1024 * 1024 * 1024,   # 1GB
                 default_ttl: float = 3600,  # 1 hora
                 enable_analytics: bool = True):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_size = max_memory_size
        self.max_disk_size = max_disk_size
        self.default_ttl = default_ttl
        self.enable_analytics = enable_analytics
        
        # Caches por nível
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.disk_cache_index: Dict[str, str] = {}  # key -> filepath
        
        # Estatísticas
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size_memory": 0,
            "size_disk": 0
        }
        
        # Thread lock para operações thread-safe
        self.lock = threading.RLock()
        
        # Carregar índice do disco
        self._load_disk_index()
        
        cache_logger.info(f"🧠 IntelligentCacheSystem inicializado - {cache_dir}")
    
    def get(self, key: str, cache_type: CacheType = None) -> Optional[Any]:
        """Buscar valor no cache (todos os níveis)"""
        with self.lock:
            # 1. Verificar cache em memória primeiro
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if not entry.is_expired():
                    entry.update_access()
                    self.stats["hits"] += 1
                    cache_logger.debug(f"💨 Cache HIT (memory): {key}")
                    return entry.value
                else:
                    # Remover entrada expirada
                    self._remove_from_memory(key)
            
            # 2. Verificar cache em disco
            if key in self.disk_cache_index:
                try:
                    entry = self._load_from_disk(key)
                    if entry and not entry.is_expired():
                        # Promover para memória se houver espaço
                        if self._can_fit_in_memory(entry):
                            self.memory_cache[key] = entry
                            self.stats["size_memory"] += entry.size_bytes
                        
                        entry.update_access()
                        self.stats["hits"] += 1
                        cache_logger.debug(f"💿 Cache HIT (disk): {key}")
                        return entry.value
                    else:
                        # Remover entrada expirada do disco
                        self._remove_from_disk(key)
                except Exception as e:
                    cache_logger.warning(f"⚠️ Erro ao carregar do disco: {e}")
                    self._remove_from_disk(key)
            
            # Cache miss
            self.stats["misses"] += 1
            cache_logger.debug(f"❌ Cache MISS: {key}")
            return None
    
    def set(self, 
            key: str, 
            value: Any, 
            cache_type: CacheType,
            ttl: Optional[float] = None,
            metadata: Dict[str, Any] = None) -> bool:
        """Armazenar valor no cache"""
        with self.lock:
            try:
                # Calcular tamanho
                serialized = pickle.dumps(value)
                size_bytes = len(serialized)
                
                # Criar entrada
                entry = CacheEntry(
                    key=key,
                    value=value,
                    cache_type=cache_type,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    ttl=ttl or self.default_ttl,
                    size_bytes=size_bytes,
                    metadata=metadata or {}
                )
                
                # Decidir onde armazenar baseado no tamanho e estratégia
                stored = False
                
                # Tentar armazenar em memória primeiro (para itens pequenos)
                if size_bytes <= 1024 * 1024 and self._can_fit_in_memory(entry):  # 1MB
                    self._ensure_memory_space(entry.size_bytes)
                    self.memory_cache[key] = entry
                    self.stats["size_memory"] += entry.size_bytes
                    stored = True
                    cache_logger.debug(f"💾 Armazenado em memória: {key} ({size_bytes} bytes)")
                
                # Armazenar em disco (sempre como backup ou para itens grandes)
                if self._can_fit_on_disk(entry):
                    self._ensure_disk_space(entry.size_bytes)
                    if self._save_to_disk(entry):
                        stored = True
                        cache_logger.debug(f"💿 Armazenado em disco: {key} ({size_bytes} bytes)")
                
                if not stored:
                    cache_logger.warning(f"⚠️ Não foi possível armazenar: {key}")
                    return False
                
                return True
                
            except Exception as e:
                cache_logger.error(f"❌ Erro ao armazenar no cache: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Remover entrada do cache"""
        with self.lock:
            removed = False
            
            # Remover da memória
            if key in self.memory_cache:
                self._remove_from_memory(key)
                removed = True
            
            # Remover do disco
            if key in self.disk_cache_index:
                self._remove_from_disk(key)
                removed = True
            
            if removed:
                cache_logger.debug(f"🗑️ Removido do cache: {key}")
            
            return removed
    
    def clear(self, cache_type: Optional[CacheType] = None) -> int:
        """Limpar cache (tudo ou por tipo)"""
        with self.lock:
            removed_count = 0
            
            if cache_type is None:
                # Limpar tudo
                removed_count = len(self.memory_cache) + len(self.disk_cache_index)
                self.memory_cache.clear()
                self.stats["size_memory"] = 0
                
                # Limpar arquivos do disco
                for filepath in self.disk_cache_index.values():
                    try:
                        Path(filepath).unlink(missing_ok=True)
                    except Exception as e:
                        cache_logger.warning(f"⚠️ Erro ao remover arquivo: {e}")
                
                self.disk_cache_index.clear()
                self.stats["size_disk"] = 0
                self._save_disk_index()
                
            else:
                # Limpar por tipo
                keys_to_remove = []
                
                # Verificar memória
                for key, entry in self.memory_cache.items():
                    if entry.cache_type == cache_type:
                        keys_to_remove.append(key)
                
                # Verificar disco (precisa carregar para verificar tipo)
                for key in self.disk_cache_index.keys():
                    try:
                        entry = self._load_from_disk(key)
                        if entry and entry.cache_type == cache_type:
                            keys_to_remove.append(key)
                    except Exception:
                        pass
                
                # Remover chaves identificadas
                for key in set(keys_to_remove):
                    if self.delete(key):
                        removed_count += 1
            
            cache_logger.info(f"🧹 Cache limpo: {removed_count} entradas removidas")
            return removed_count
    
    def cleanup_expired(self) -> int:
        """Limpar entradas expiradas"""
        with self.lock:
            expired_keys = []
            
            # Verificar memória
            for key, entry in self.memory_cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            # Verificar disco
            for key in list(self.disk_cache_index.keys()):
                try:
                    entry = self._load_from_disk(key)
                    if entry and entry.is_expired():
                        expired_keys.append(key)
                except Exception:
                    # Se não conseguir carregar, considerar como expirado
                    expired_keys.append(key)
            
            # Remover expirados
            removed_count = 0
            for key in set(expired_keys):
                if self.delete(key):
                    removed_count += 1
            
            if removed_count > 0:
                cache_logger.info(f"🧹 Entradas expiradas removidas: {removed_count}")
            
            return removed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Obter estatísticas do cache"""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                **self.stats,
                "memory_entries": len(self.memory_cache),
                "disk_entries": len(self.disk_cache_index),
                "total_entries": len(self.memory_cache) + len(self.disk_cache_index),
                "hit_rate_percent": round(hit_rate, 2),
                "memory_usage_percent": round(self.stats["size_memory"] / self.max_memory_size * 100, 2),
                "disk_usage_percent": round(self.stats["size_disk"] / self.max_disk_size * 100, 2)
            }
    
    def optimize(self) -> Dict[str, int]:
        """Otimizar cache (limpeza + reorganização)"""
        cache_logger.info("🔧 Iniciando otimização do cache...")
        
        # 1. Limpar expirados
        expired_count = self.cleanup_expired()
        
        # 2. Aplicar algoritmo LRU se necessário
        lru_count = self._apply_lru_if_needed()
        
        # 3. Compactar arquivos do disco
        compact_count = self._compact_disk_cache()
        
        # 4. Atualizar estatísticas
        self._recalculate_sizes()
        
        result = {
            "expired_removed": expired_count,
            "lru_evicted": lru_count,
            "files_compacted": compact_count
        }
        
        cache_logger.info(f"✅ Otimização concluída: {result}")
        return result
    
    # ========== MÉTODOS PRIVADOS ==========
    
    def _can_fit_in_memory(self, entry: CacheEntry) -> bool:
        """Verificar se entrada cabe na memória"""
        return (self.stats["size_memory"] + entry.size_bytes) <= self.max_memory_size
    
    def _can_fit_on_disk(self, entry: CacheEntry) -> bool:
        """Verificar se entrada cabe no disco"""
        return (self.stats["size_disk"] + entry.size_bytes) <= self.max_disk_size
    
    def _ensure_memory_space(self, needed_bytes: int):
        """Garantir espaço em memória aplicando LRU"""
        while (self.stats["size_memory"] + needed_bytes) > self.max_memory_size:
            if not self.memory_cache:
                break
            
            # Encontrar entrada menos recentemente usada
            lru_key = min(self.memory_cache.keys(), 
                         key=lambda k: self.memory_cache[k].last_accessed)
            self._remove_from_memory(lru_key)
            self.stats["evictions"] += 1
    
    def _ensure_disk_space(self, needed_bytes: int):
        """Garantir espaço em disco aplicando LRU"""
        while (self.stats["size_disk"] + needed_bytes) > self.max_disk_size:
            if not self.disk_cache_index:
                break
            
            # Carregar metadados e encontrar LRU
            oldest_key = None
            oldest_time = datetime.now()
            
            for key in list(self.disk_cache_index.keys()):
                try:
                    entry = self._load_from_disk(key)
                    if entry and entry.last_accessed < oldest_time:
                        oldest_time = entry.last_accessed
                        oldest_key = key
                except Exception:
                    # Se não conseguir carregar, remover
                    oldest_key = key
                    break
            
            if oldest_key:
                self._remove_from_disk(oldest_key)
                self.stats["evictions"] += 1
            else:
                break
    
    def _remove_from_memory(self, key: str):
        """Remover entrada da memória"""
        if key in self.memory_cache:
            entry = self.memory_cache.pop(key)
            self.stats["size_memory"] -= entry.size_bytes
    
    def _remove_from_disk(self, key: str):
        """Remover entrada do disco"""
        if key in self.disk_cache_index:
            filepath = self.disk_cache_index.pop(key)
            try:
                file_path = Path(filepath)
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    self.stats["size_disk"] -= file_size
            except Exception as e:
                cache_logger.warning(f"⚠️ Erro ao remover arquivo do disco: {e}")
            
            self._save_disk_index()
    
    def _save_to_disk(self, entry: CacheEntry) -> bool:
        """Salvar entrada no disco"""
        try:
            # Gerar nome do arquivo baseado no hash da chave
            key_hash = hashlib.md5(entry.key.encode()).hexdigest()
            filepath = self.cache_dir / f"{key_hash}.cache"
            
            # Salvar dados serializados
            with open(filepath, 'wb') as f:
                pickle.dump(entry, f)
            
            # Atualizar índice
            self.disk_cache_index[entry.key] = str(filepath)
            self.stats["size_disk"] += entry.size_bytes
            self._save_disk_index()
            
            return True
            
        except Exception as e:
            cache_logger.error(f"❌ Erro ao salvar no disco: {e}")
            return False
    
    def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        """Carregar entrada do disco"""
        if key not in self.disk_cache_index:
            return None
        
        try:
            filepath = self.disk_cache_index[key]
            with open(filepath, 'rb') as f:
                entry = pickle.load(f)
            return entry
        except Exception as e:
            cache_logger.warning(f"⚠️ Erro ao carregar do disco: {e}")
            return None
    
    def _load_disk_index(self):
        """Carregar índice do disco"""
        index_file = self.cache_dir / "cache_index.json"
        try:
            if index_file.exists():
                with open(index_file, 'r') as f:
                    self.disk_cache_index = json.load(f)
                
                # Recalcular tamanho do disco
                total_size = 0
                invalid_keys = []
                
                for key, filepath in self.disk_cache_index.items():
                    try:
                        file_path = Path(filepath)
                        if file_path.exists():
                            total_size += file_path.stat().st_size
                        else:
                            invalid_keys.append(key)
                    except Exception:
                        invalid_keys.append(key)
                
                # Remover chaves inválidas
                for key in invalid_keys:
                    del self.disk_cache_index[key]
                
                self.stats["size_disk"] = total_size
                
                if invalid_keys:
                    self._save_disk_index()
                    
        except Exception as e:
            cache_logger.warning(f"⚠️ Erro ao carregar índice do disco: {e}")
            self.disk_cache_index = {}
    
    def _save_disk_index(self):
        """Salvar índice do disco"""
        index_file = self.cache_dir / "cache_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.disk_cache_index, f, indent=2)
        except Exception as e:
            cache_logger.error(f"❌ Erro ao salvar índice do disco: {e}")
    
    def _apply_lru_if_needed(self) -> int:
        """Aplicar LRU se necessário"""
        evicted = 0
        
        # Verificar se memória está muito cheia (>90%)
        if self.stats["size_memory"] > (self.max_memory_size * 0.9):
            target_size = self.max_memory_size * 0.7  # Reduzir para 70%
            
            while self.stats["size_memory"] > target_size and self.memory_cache:
                lru_key = min(self.memory_cache.keys(), 
                             key=lambda k: self.memory_cache[k].last_accessed)
                self._remove_from_memory(lru_key)
                evicted += 1
        
        return evicted
    
    def _compact_disk_cache(self) -> int:
        """Compactar cache do disco removendo fragmentação"""
        # Por agora, apenas validar arquivos existentes
        compacted = 0
        invalid_keys = []
        
        for key, filepath in self.disk_cache_index.items():
            if not Path(filepath).exists():
                invalid_keys.append(key)
                compacted += 1
        
        for key in invalid_keys:
            del self.disk_cache_index[key]
        
        if invalid_keys:
            self._save_disk_index()
        
        return compacted
    
    def _recalculate_sizes(self):
        """Recalcular tamanhos dos caches"""
        # Memória
        memory_size = sum(entry.size_bytes for entry in self.memory_cache.values())
        self.stats["size_memory"] = memory_size
        
        # Disco
        disk_size = 0
        for filepath in self.disk_cache_index.values():
            try:
                disk_size += Path(filepath).stat().st_size
            except Exception:
                pass
        self.stats["size_disk"] = disk_size

# ========== FUNÇÕES UTILITÁRIAS ==========

def create_cache_key(*args, **kwargs) -> str:
    """Criar chave de cache baseada em argumentos"""
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items())
    }
    key_string = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.md5(key_string.encode()).hexdigest()

def cache_decorator(cache_system: IntelligentCacheSystem, 
                   cache_type: CacheType,
                   ttl: Optional[float] = None):
    """Decorator para cache automático de funções"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Gerar chave baseada na função e argumentos
            func_key = f"{func.__module__}.{func.__name__}"
            cache_key = create_cache_key(func_key, *args, **kwargs)
            
            # Tentar buscar no cache
            cached_result = cache_system.get(cache_key, cache_type)
            if cached_result is not None:
                return cached_result
            
            # Executar função e cachear resultado
            result = func(*args, **kwargs)
            cache_system.set(cache_key, result, cache_type, ttl)
            
            return result
        return wrapper
    return decorator

# ========== SISTEMA GLOBAL ==========

_global_cache_system: Optional[IntelligentCacheSystem] = None

def get_global_cache_system() -> IntelligentCacheSystem:
    """Obter instância global do sistema de cache"""
    global _global_cache_system
    if _global_cache_system is None:
        _global_cache_system = IntelligentCacheSystem()
    return _global_cache_system

def init_global_cache_system(**kwargs) -> IntelligentCacheSystem:
    """Inicializar sistema global de cache com configurações específicas"""
    global _global_cache_system
    _global_cache_system = IntelligentCacheSystem(**kwargs)
    return _global_cache_system 