#!/usr/bin/env python
"""
🚀 INSIGHTS-AI RESOURCE MANAGER
Gerenciador inteligente de recursos do sistema

Características:
- Gestão de memória otimizada
- Pool de conexões dinâmico
- Controle de threads inteligente
- Monitoramento de recursos
- Limpeza automática
"""

import gc
import os
import time
import logging
import threading
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import weakref
import queue
import concurrent.futures
from contextlib import contextmanager

# Configurar logging
resource_logger = logging.getLogger(__name__)
resource_logger.setLevel(logging.INFO)

class ResourceType(Enum):
    """Tipos de recursos gerenciados"""
    MEMORY = "memory"
    THREADS = "threads"
    CONNECTIONS = "connections"
    FILE_HANDLES = "file_handles"
    CACHE = "cache"
    TEMP_FILES = "temp_files"

class ResourceStatus(Enum):
    """Status dos recursos"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    OPTIMIZING = "optimizing"

@dataclass
class ResourceLimits:
    """Limites de recursos"""
    max_memory_mb: int = 1024
    max_threads: int = 50
    max_connections: int = 100
    max_file_handles: int = 500
    max_cache_size_mb: int = 256
    max_temp_files: int = 100

@dataclass
class ResourceUsage:
    """Uso atual de recursos"""
    timestamp: datetime
    memory_mb: float
    memory_percent: float
    active_threads: int
    active_connections: int
    open_file_handles: int
    cache_size_mb: float
    temp_files_count: int

class ResourceManager:
    """Gerenciador de recursos do sistema"""
    
    def __init__(self, 
                 limits: Optional[ResourceLimits] = None,
                 auto_cleanup: bool = True,
                 cleanup_interval_seconds: int = 300):
        
        self.limits = limits or ResourceLimits()
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval_seconds = cleanup_interval_seconds
        
        # Estado do sistema
        self.is_active = False
        self.start_time = None
        
        # Pools e managers
        self._thread_pool = None
        self._connection_pool: Dict[str, queue.Queue] = {}
        self._file_handle_registry = weakref.WeakSet()
        self._temp_files_registry: List[str] = []
        
        # Monitoramento
        self.usage_history: List[ResourceUsage] = []
        self.cleanup_history: List[Dict[str, Any]] = []
        
        # Threading
        self.cleanup_thread = None
        self.monitoring_thread = None
        self.lock = threading.RLock()
        
        # Callbacks para limpeza customizada
        self.cleanup_callbacks: Dict[str, Callable] = {}
        
        # Configurar sistema
        self._setup_resource_manager()
        
        resource_logger.info("🎯 ResourceManager inicializado")
    
    def _setup_resource_manager(self):
        """Configurar gerenciador de recursos"""
        try:
            # Inicializar pools
            self._initialize_thread_pool()
            self._initialize_connection_pools()
            
            # Registrar callbacks de limpeza padrão
            self._register_default_cleanup_callbacks()
            
            resource_logger.info("✅ ResourceManager configurado")
            
        except Exception as e:
            resource_logger.error(f"❌ Erro ao configurar ResourceManager: {e}")
    
    def start_monitoring(self):
        """Iniciar monitoramento de recursos"""
        if self.is_active:
            return
        
        self.is_active = True
        self.start_time = datetime.now()
        
        # Thread de monitoramento
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ResourceMonitoring"
        )
        self.monitoring_thread.start()
        
        # Thread de limpeza automática
        if self.auto_cleanup:
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True,
                name="ResourceCleanup"
            )
            self.cleanup_thread.start()
        
        resource_logger.info("🚀 Monitoramento de recursos iniciado")
    
    def stop_monitoring(self):
        """Parar monitoramento"""
        self.is_active = False
        
        # Limpeza final
        self.cleanup_all_resources()
        
        resource_logger.info("⏹️ Monitoramento de recursos parado")
    
    def _monitoring_loop(self):
        """Loop de monitoramento de recursos"""
        while self.is_active:
            try:
                # Coletar uso atual
                current_usage = self._collect_resource_usage()
                
                # Analisar status
                status = self._analyze_resource_status(current_usage)
                
                # Tomar ações baseadas no status
                self._handle_resource_status(status, current_usage)
                
                # Salvar histórico
                with self.lock:
                    self.usage_history.append(current_usage)
                    self._cleanup_old_usage_history()
                
            except Exception as e:
                resource_logger.error(f"❌ Erro no monitoramento: {e}")
            
            time.sleep(30)  # Verificar a cada 30 segundos
    
    def _cleanup_loop(self):
        """Loop de limpeza automática"""
        while self.is_active:
            try:
                self._perform_automatic_cleanup()
                
            except Exception as e:
                resource_logger.error(f"❌ Erro na limpeza automática: {e}")
            
            time.sleep(self.cleanup_interval_seconds)
    
    def _collect_resource_usage(self) -> ResourceUsage:
        """Coletar uso atual de recursos"""
        try:
            # Memória
            memory = psutil.virtual_memory()
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_percent = memory.percent
            
            # Threads
            active_threads = threading.active_count()
            
            # Conexões
            active_connections = len(psutil.net_connections(kind='inet'))
            
            # File handles
            try:
                open_files = len(process.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = 0
            
            # Cache e arquivos temporários
            cache_size_mb = self._estimate_cache_size()
            temp_files_count = len(self._temp_files_registry)
            
            return ResourceUsage(
                timestamp=datetime.now(),
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                active_threads=active_threads,
                active_connections=active_connections,
                open_file_handles=open_files,
                cache_size_mb=cache_size_mb,
                temp_files_count=temp_files_count
            )
            
        except Exception as e:
            resource_logger.error(f"❌ Erro ao coletar recursos: {e}")
            return ResourceUsage(timestamp=datetime.now())
    
    def _analyze_resource_status(self, usage: ResourceUsage) -> Dict[ResourceType, ResourceStatus]:
        """Analisar status dos recursos"""
        status = {}
        
        # Análise de memória
        if usage.memory_mb > self.limits.max_memory_mb * 0.9:
            status[ResourceType.MEMORY] = ResourceStatus.CRITICAL
        elif usage.memory_mb > self.limits.max_memory_mb * 0.7:
            status[ResourceType.MEMORY] = ResourceStatus.WARNING
        else:
            status[ResourceType.MEMORY] = ResourceStatus.NORMAL
        
        # Análise de threads
        if usage.active_threads > self.limits.max_threads * 0.9:
            status[ResourceType.THREADS] = ResourceStatus.CRITICAL
        elif usage.active_threads > self.limits.max_threads * 0.7:
            status[ResourceType.THREADS] = ResourceStatus.WARNING
        else:
            status[ResourceType.THREADS] = ResourceStatus.NORMAL
        
        # Análise de conexões
        if usage.active_connections > self.limits.max_connections * 0.9:
            status[ResourceType.CONNECTIONS] = ResourceStatus.CRITICAL
        elif usage.active_connections > self.limits.max_connections * 0.7:
            status[ResourceType.CONNECTIONS] = ResourceStatus.WARNING
        else:
            status[ResourceType.CONNECTIONS] = ResourceStatus.NORMAL
        
        # Análise de file handles
        if usage.open_file_handles > self.limits.max_file_handles * 0.9:
            status[ResourceType.FILE_HANDLES] = ResourceStatus.CRITICAL
        elif usage.open_file_handles > self.limits.max_file_handles * 0.7:
            status[ResourceType.FILE_HANDLES] = ResourceStatus.WARNING
        else:
            status[ResourceType.FILE_HANDLES] = ResourceStatus.NORMAL
        
        return status
    
    def _handle_resource_status(self, status: Dict[ResourceType, ResourceStatus], usage: ResourceUsage):
        """Lidar com status dos recursos"""
        for resource_type, resource_status in status.items():
            if resource_status == ResourceStatus.CRITICAL:
                resource_logger.warning(f"🚨 CRÍTICO: {resource_type.value} em uso excessivo")
                self._emergency_cleanup(resource_type)
            elif resource_status == ResourceStatus.WARNING:
                resource_logger.info(f"⚠️ ATENÇÃO: {resource_type.value} em uso alto")
                self._preventive_cleanup(resource_type)
    
    def _perform_automatic_cleanup(self):
        """Realizar limpeza automática"""
        cleanup_summary = {
            "timestamp": datetime.now().isoformat(),
            "actions": [],
            "resources_freed": {}
        }
        
        try:
            # Limpeza de memória
            memory_freed = self._cleanup_memory()
            if memory_freed > 0:
                cleanup_summary["actions"].append("memory_cleanup")
                cleanup_summary["resources_freed"]["memory_mb"] = memory_freed
            
            # Limpeza de arquivos temporários
            temp_files_removed = self._cleanup_temp_files()
            if temp_files_removed > 0:
                cleanup_summary["actions"].append("temp_files_cleanup")
                cleanup_summary["resources_freed"]["temp_files"] = temp_files_removed
            
            # Limpeza de conexões inativas
            connections_closed = self._cleanup_idle_connections()
            if connections_closed > 0:
                cleanup_summary["actions"].append("connections_cleanup")
                cleanup_summary["resources_freed"]["connections"] = connections_closed
            
            # Executar callbacks customizados
            for callback_name, callback in self.cleanup_callbacks.items():
                try:
                    result = callback()
                    if result:
                        cleanup_summary["actions"].append(f"custom_{callback_name}")
                except Exception as e:
                    resource_logger.error(f"❌ Erro no callback {callback_name}: {e}")
            
            # Salvar histórico se houve limpeza
            if cleanup_summary["actions"]:
                with self.lock:
                    self.cleanup_history.append(cleanup_summary)
                    self._cleanup_old_cleanup_history()
                
                resource_logger.info(f"🧹 Limpeza automática: {len(cleanup_summary['actions'])} ações")
        
        except Exception as e:
            resource_logger.error(f"❌ Erro na limpeza automática: {e}")
    
    def _emergency_cleanup(self, resource_type: ResourceType):
        """Limpeza de emergência para recurso crítico"""
        resource_logger.warning(f"🚨 Limpeza de emergência: {resource_type.value}")
        
        if resource_type == ResourceType.MEMORY:
            self._aggressive_memory_cleanup()
        elif resource_type == ResourceType.THREADS:
            self._cleanup_idle_threads()
        elif resource_type == ResourceType.CONNECTIONS:
            self._cleanup_idle_connections()
        elif resource_type == ResourceType.FILE_HANDLES:
            self._cleanup_file_handles()
    
    def _preventive_cleanup(self, resource_type: ResourceType):
        """Limpeza preventiva para recurso em warning"""
        resource_logger.info(f"🧹 Limpeza preventiva: {resource_type.value}")
        
        if resource_type == ResourceType.MEMORY:
            self._cleanup_memory()
        elif resource_type == ResourceType.CONNECTIONS:
            self._cleanup_idle_connections()
    
    def _cleanup_memory(self) -> float:
        """Limpeza de memória"""
        try:
            # Memória antes
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024 * 1024)
            
            # Forçar garbage collection
            gc.collect()
            
            # Memória depois
            memory_after = process.memory_info().rss / (1024 * 1024)
            memory_freed = max(0, memory_before - memory_after)
            
            if memory_freed > 0:
                resource_logger.info(f"🧹 Memória liberada: {memory_freed:.1f}MB")
            
            return memory_freed
            
        except Exception as e:
            resource_logger.error(f"❌ Erro na limpeza de memória: {e}")
            return 0.0
    
    def _aggressive_memory_cleanup(self):
        """Limpeza agressiva de memória"""
        try:
            # Múltiplas rodadas de GC
            for _ in range(3):
                gc.collect()
            
            # Limpar caches internos se possível
            self._clear_internal_caches()
            
            resource_logger.info("🚨 Limpeza agressiva de memória executada")
            
        except Exception as e:
            resource_logger.error(f"❌ Erro na limpeza agressiva: {e}")
    
    def _cleanup_temp_files(self) -> int:
        """Limpeza de arquivos temporários"""
        files_removed = 0
        
        try:
            files_to_remove = []
            
            for temp_file in self._temp_files_registry.copy():
                try:
                    if os.path.exists(temp_file):
                        # Verificar se arquivo é antigo (>1 hora)
                        file_age = time.time() - os.path.getmtime(temp_file)
                        if file_age > 3600:  # 1 hora
                            os.remove(temp_file)
                            files_to_remove.append(temp_file)
                            files_removed += 1
                    else:
                        files_to_remove.append(temp_file)
                except Exception as e:
                    resource_logger.warning(f"⚠️ Erro ao remover {temp_file}: {e}")
            
            # Remover da registry
            for temp_file in files_to_remove:
                if temp_file in self._temp_files_registry:
                    self._temp_files_registry.remove(temp_file)
            
            if files_removed > 0:
                resource_logger.info(f"🧹 Arquivos temporários removidos: {files_removed}")
            
        except Exception as e:
            resource_logger.error(f"❌ Erro na limpeza de arquivos temporários: {e}")
        
        return files_removed
    
    def _cleanup_idle_connections(self) -> int:
        """Limpeza de conexões inativas"""
        connections_closed = 0
        
        try:
            for pool_name, pool in self._connection_pool.items():
                # Limpar conexões antigas da pool
                while not pool.empty():
                    try:
                        conn = pool.get_nowait()
                        # Verificar se conexão ainda é válida
                        # Fechar se for muito antiga
                        connections_closed += 1
                    except queue.Empty:
                        break
                    except Exception:
                        continue
            
            if connections_closed > 0:
                resource_logger.info(f"🧹 Conexões inativas fechadas: {connections_closed}")
        
        except Exception as e:
            resource_logger.error(f"❌ Erro na limpeza de conexões: {e}")
        
        return connections_closed
    
    # Métodos públicos para gerenciamento de recursos
    
    @contextmanager
    def get_thread_from_pool(self):
        """Context manager para obter thread do pool"""
        if not self._thread_pool:
            self._initialize_thread_pool()
        
        try:
            yield self._thread_pool
        finally:
            pass  # Pool gerencia automaticamente
    
    def register_temp_file(self, filepath: str):
        """Registrar arquivo temporário para limpeza"""
        with self.lock:
            if filepath not in self._temp_files_registry:
                self._temp_files_registry.append(filepath)
    
    def register_cleanup_callback(self, name: str, callback: Callable):
        """Registrar callback de limpeza customizado"""
        self.cleanup_callbacks[name] = callback
        resource_logger.info(f"✅ Callback de limpeza registrado: {name}")
    
    def force_cleanup(self, resource_types: Optional[List[ResourceType]] = None):
        """Forçar limpeza de tipos específicos de recursos"""
        if not resource_types:
            resource_types = list(ResourceType)
        
        for resource_type in resource_types:
            self._emergency_cleanup(resource_type)
    
    def cleanup_all_resources(self):
        """Limpeza completa de todos os recursos"""
        resource_logger.info("🧹 Iniciando limpeza completa de recursos")
        
        try:
            # Limpeza de memória
            self._aggressive_memory_cleanup()
            
            # Limpeza de arquivos temporários
            self._cleanup_temp_files()
            
            # Fechar todas as conexões
            self._cleanup_idle_connections()
            
            # Shutdown do thread pool
            if self._thread_pool:
                self._thread_pool.shutdown(wait=True)
                self._thread_pool = None
            
            resource_logger.info("✅ Limpeza completa finalizada")
            
        except Exception as e:
            resource_logger.error(f"❌ Erro na limpeza completa: {e}")
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Obter uso atual de recursos"""
        if not self.usage_history:
            return {"status": "no_data"}
        
        latest_usage = self.usage_history[-1]
        
        return {
            "timestamp": latest_usage.timestamp.isoformat(),
            "memory": {
                "current_mb": latest_usage.memory_mb,
                "limit_mb": self.limits.max_memory_mb,
                "utilization_percent": (latest_usage.memory_mb / self.limits.max_memory_mb) * 100
            },
            "threads": {
                "current": latest_usage.active_threads,
                "limit": self.limits.max_threads,
                "utilization_percent": (latest_usage.active_threads / self.limits.max_threads) * 100
            },
            "connections": {
                "current": latest_usage.active_connections,
                "limit": self.limits.max_connections,
                "utilization_percent": (latest_usage.active_connections / self.limits.max_connections) * 100
            },
            "file_handles": {
                "current": latest_usage.open_file_handles,
                "limit": self.limits.max_file_handles,
                "utilization_percent": (latest_usage.open_file_handles / self.limits.max_file_handles) * 100
            }
        }
    
    def get_cleanup_stats(self, hours_back: int = 24) -> Dict[str, Any]:
        """Obter estatísticas de limpeza"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        recent_cleanups = [
            cleanup for cleanup in self.cleanup_history
            if datetime.fromisoformat(cleanup["timestamp"]) >= cutoff_time
        ]
        
        total_actions = sum(len(cleanup["actions"]) for cleanup in recent_cleanups)
        
        return {
            "total_cleanups": len(recent_cleanups),
            "total_actions": total_actions,
            "avg_actions_per_cleanup": total_actions / len(recent_cleanups) if recent_cleanups else 0,
            "last_cleanup": recent_cleanups[-1] if recent_cleanups else None
        }
    
    # Métodos auxiliares privados
    
    def _initialize_thread_pool(self):
        """Inicializar pool de threads"""
        max_workers = min(self.limits.max_threads // 2, 10)  # Conservador
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ResourceManager"
        )
    
    def _initialize_connection_pools(self):
        """Inicializar pools de conexão"""
        # Pools básicos
        self._connection_pool["database"] = queue.Queue(maxsize=20)
        self._connection_pool["http"] = queue.Queue(maxsize=10)
        self._connection_pool["cache"] = queue.Queue(maxsize=5)
    
    def _register_default_cleanup_callbacks(self):
        """Registrar callbacks de limpeza padrão"""
        def cleanup_logs():
            # Limpar logs antigos se necessário
            return False
        
        def cleanup_caches():
            # Limpar caches internos
            return False
        
        self.register_cleanup_callback("logs", cleanup_logs)
        self.register_cleanup_callback("caches", cleanup_caches)
    
    def _estimate_cache_size(self) -> float:
        """Estimar tamanho do cache em MB"""
        # Implementação básica - pode ser expandida
        return 0.0
    
    def _clear_internal_caches(self):
        """Limpar caches internos"""
        # Implementar limpeza de caches específicos
        pass
    
    def _cleanup_idle_threads(self):
        """Limpar threads inativas"""
        # Implementar limpeza de threads se possível
        pass
    
    def _cleanup_file_handles(self):
        """Limpar file handles não utilizados"""
        # Implementar limpeza de file handles
        pass
    
    def _cleanup_old_usage_history(self):
        """Limpar histórico antigo de uso"""
        # Manter apenas 1000 registros mais recentes
        if len(self.usage_history) > 1000:
            self.usage_history = self.usage_history[-500:]
    
    def _cleanup_old_cleanup_history(self):
        """Limpar histórico antigo de limpeza"""
        # Manter apenas 100 registros mais recentes
        if len(self.cleanup_history) > 100:
            self.cleanup_history = self.cleanup_history[-50:] 