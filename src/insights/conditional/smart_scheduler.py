#!/usr/bin/env python
"""
ETAPA 4 - SCHEDULER INTELIGENTE
Sistema de agendamento inteligente baseado em condições
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .condition_engine import ConditionEngine, ConditionEngineResult

# Configuração de logging
scheduler_logger = logging.getLogger('smart_scheduler')
scheduler_logger.setLevel(logging.INFO)

class ScheduleType(Enum):
    """Tipos de agendamento"""
    IMMEDIATE = "immediate"
    INTERVAL = "interval"
    CONDITIONAL = "conditional"
    HYBRID = "hybrid"

@dataclass
class ScheduleConfig:
    """Configuração de agendamento"""
    name: str
    schedule_type: ScheduleType
    interval_seconds: Optional[float] = None
    max_retries: int = 3
    retry_delay_seconds: float = 60.0
    timeout_seconds: float = 300.0
    enabled: bool = True
    metadata: Dict[str, Any] = None

@dataclass
class ScheduledTask:
    """Tarefa agendada"""
    id: str
    name: str
    function: Callable
    config: ScheduleConfig
    condition_engine: Optional[ConditionEngine]
    next_run: datetime
    last_run: Optional[datetime]
    last_result: Optional[Any]
    retry_count: int
    execution_count: int
    is_running: bool

class SmartScheduler:
    """Scheduler inteligente com condições"""
    
    def __init__(self, max_concurrent_tasks: int = 5):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.task_threads: List[threading.Thread] = []
        self.stop_event = threading.Event()
        
        scheduler_logger.info("🗓️ SmartScheduler inicializado")
    
    def add_task(self, 
                 task_id: str,
                 task_name: str,
                 function: Callable,
                 config: ScheduleConfig,
                 condition_engine: Optional[ConditionEngine] = None) -> bool:
        """Adicionar tarefa ao scheduler"""
        try:
            if task_id in self.tasks:
                scheduler_logger.warning(f"⚠️ Tarefa {task_id} já existe - substituindo")
            
            # Calcular próxima execução
            next_run = self._calculate_next_run(config)
            
            task = ScheduledTask(
                id=task_id,
                name=task_name,
                function=function,
                config=config,
                condition_engine=condition_engine,
                next_run=next_run,
                last_run=None,
                last_result=None,
                retry_count=0,
                execution_count=0,
                is_running=False
            )
            
            self.tasks[task_id] = task
            scheduler_logger.info(f"➕ Tarefa adicionada: {task_name} ({task_id})")
            
            return True
            
        except Exception as e:
            scheduler_logger.error(f"❌ Erro ao adicionar tarefa {task_id}: {e}")
            return False
    
    def remove_task(self, task_id: str) -> bool:
        """Remover tarefa do scheduler"""
        try:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.is_running:
                    scheduler_logger.warning(f"⚠️ Tarefa {task_id} está executando - será removida após conclusão")
                
                del self.tasks[task_id]
                scheduler_logger.info(f"➖ Tarefa removida: {task_id}")
                return True
            else:
                scheduler_logger.warning(f"⚠️ Tarefa {task_id} não encontrada")
                return False
                
        except Exception as e:
            scheduler_logger.error(f"❌ Erro ao remover tarefa {task_id}: {e}")
            return False
    
    def start(self) -> bool:
        """Iniciar scheduler"""
        try:
            if self.running:
                scheduler_logger.warning("⚠️ Scheduler já está rodando")
                return False
            
            self.running = True
            self.stop_event.clear()
            
            # Iniciar thread principal do scheduler
            self.scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                name="SmartScheduler-Main",
                daemon=True
            )
            self.scheduler_thread.start()
            
            scheduler_logger.info("🚀 SmartScheduler iniciado")
            return True
            
        except Exception as e:
            scheduler_logger.error(f"❌ Erro ao iniciar scheduler: {e}")
            self.running = False
            return False
    
    def stop(self, wait_for_completion: bool = True) -> bool:
        """Parar scheduler"""
        try:
            if not self.running:
                scheduler_logger.warning("⚠️ Scheduler não está rodando")
                return False
            
            scheduler_logger.info("🛑 Parando SmartScheduler...")
            
            # Sinalizar parada
            self.stop_event.set()
            self.running = False
            
            # Aguardar conclusão se solicitado
            if wait_for_completion:
                # Aguardar thread principal
                if self.scheduler_thread and self.scheduler_thread.is_alive():
                    self.scheduler_thread.join(timeout=10)
                
                # Aguardar threads de tarefas
                for thread in self.task_threads[:]:
                    if thread.is_alive():
                        thread.join(timeout=5)
                        if thread.is_alive():
                            scheduler_logger.warning(f"⚠️ Thread {thread.name} não finalizou")
            
            scheduler_logger.info("✅ SmartScheduler parado")
            return True
            
        except Exception as e:
            scheduler_logger.error(f"❌ Erro ao parar scheduler: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Obter status do scheduler"""
        try:
            running_tasks = [t for t in self.tasks.values() if t.is_running]
            enabled_tasks = [t for t in self.tasks.values() if t.config.enabled]
            
            # Próxima tarefa a executar
            next_task = None
            next_run_time = None
            
            if enabled_tasks:
                upcoming_tasks = sorted(enabled_tasks, key=lambda t: t.next_run)
                if upcoming_tasks:
                    next_task = upcoming_tasks[0]
                    next_run_time = next_task.next_run
            
            status = {
                "running": self.running,
                "total_tasks": len(self.tasks),
                "enabled_tasks": len(enabled_tasks),
                "running_tasks": len(running_tasks),
                "max_concurrent": self.max_concurrent_tasks,
                "next_run": next_run_time.isoformat() if next_run_time else None,
                "next_task": next_task.name if next_task else None,
                "active_threads": len([t for t in self.task_threads if t.is_alive()]),
                "task_summary": []
            }
            
            # Resumo das tarefas
            for task in self.tasks.values():
                status["task_summary"].append({
                    "id": task.id,
                    "name": task.name,
                    "enabled": task.config.enabled,
                    "is_running": task.is_running,
                    "execution_count": task.execution_count,
                    "retry_count": task.retry_count,
                    "next_run": task.next_run.isoformat(),
                    "last_run": task.last_run.isoformat() if task.last_run else None
                })
            
            return status
            
        except Exception as e:
            scheduler_logger.error(f"❌ Erro ao obter status: {e}")
            return {"error": str(e)}
    
    def force_run_task(self, task_id: str) -> bool:
        """Forçar execução imediata de uma tarefa"""
        try:
            if task_id not in self.tasks:
                scheduler_logger.error(f"❌ Tarefa {task_id} não encontrada")
                return False
            
            task = self.tasks[task_id]
            
            if task.is_running:
                scheduler_logger.warning(f"⚠️ Tarefa {task_id} já está executando")
                return False
            
            # Executar em thread separada
            thread = threading.Thread(
                target=self._execute_task,
                args=(task, True),  # True = execução forçada
                name=f"Task-{task.name}-Forced",
                daemon=True
            )
            thread.start()
            self.task_threads.append(thread)
            
            scheduler_logger.info(f"🚀 Execução forçada da tarefa: {task.name}")
            return True
            
        except Exception as e:
            scheduler_logger.error(f"❌ Erro ao forçar execução de {task_id}: {e}")
            return False
    
    def _scheduler_loop(self):
        """Loop principal do scheduler"""
        try:
            scheduler_logger.info("🔄 Loop do scheduler iniciado")
            
            while self.running and not self.stop_event.is_set():
                try:
                    # Verificar tarefas para execução
                    self._check_and_execute_tasks()
                    
                    # Limpar threads finalizadas
                    self._cleanup_threads()
                    
                    # Aguardar próxima verificação (1 segundo)
                    if self.stop_event.wait(1.0):
                        break
                        
                except Exception as e:
                    scheduler_logger.error(f"❌ Erro no loop do scheduler: {e}")
                    time.sleep(5)  # Aguardar antes de tentar novamente
            
            scheduler_logger.info("🏁 Loop do scheduler finalizado")
            
        except Exception as e:
            scheduler_logger.error(f"❌ Erro fatal no scheduler loop: {e}")
        finally:
            self.running = False
    
    def _check_and_execute_tasks(self):
        """Verificar e executar tarefas que devem rodar"""
        try:
            now = datetime.now()
            running_count = len([t for t in self.tasks.values() if t.is_running])
            
            # Não exceder limite de tarefas concorrentes
            if running_count >= self.max_concurrent_tasks:
                return
            
            # Procurar tarefas prontas para execução
            ready_tasks = []
            for task in self.tasks.values():
                if (task.config.enabled and 
                    not task.is_running and 
                    task.next_run <= now):
                    ready_tasks.append(task)
            
            # Ordenar por prioridade (próxima execução primeiro)
            ready_tasks.sort(key=lambda t: t.next_run)
            
            # Executar tarefas até o limite
            for task in ready_tasks:
                if running_count >= self.max_concurrent_tasks:
                    break
                
                # Verificar condições se configurado
                should_execute = True
                if task.condition_engine:
                    try:
                        context = self._build_execution_context(task)
                        evaluation = task.condition_engine.evaluate_all(context)
                        should_execute = evaluation.should_execute
                        
                        if not should_execute:
                            scheduler_logger.info(
                                f"⏸️ Execução de {task.name} adiada: {evaluation.primary_reason}"
                            )
                            # Reagendar para próxima verificação
                            task.next_run = now + timedelta(seconds=60)
                            continue
                            
                    except Exception as e:
                        scheduler_logger.error(f"❌ Erro ao avaliar condições para {task.name}: {e}")
                        should_execute = False
                
                if should_execute:
                    # Executar tarefa
                    thread = threading.Thread(
                        target=self._execute_task,
                        args=(task, False),  # False = execução normal
                        name=f"Task-{task.name}",
                        daemon=True
                    )
                    thread.start()
                    self.task_threads.append(thread)
                    running_count += 1
                    
                    scheduler_logger.info(f"🚀 Executando tarefa: {task.name}")
                
        except Exception as e:
            scheduler_logger.error(f"❌ Erro ao verificar tarefas: {e}")
    
    def _execute_task(self, task: ScheduledTask, forced: bool = False):
        """Executar uma tarefa específica"""
        try:
            task.is_running = True
            start_time = time.time()
            
            scheduler_logger.info(f"▶️ Iniciando execução: {task.name}")
            
            try:
                # Executar função da tarefa
                result = task.function()
                
                # Atualizar estatísticas
                task.last_result = result
                task.last_run = datetime.now()
                task.execution_count += 1
                task.retry_count = 0  # Reset retry count on success
                
                execution_time = time.time() - start_time
                scheduler_logger.info(
                    f"✅ Tarefa concluída: {task.name} ({execution_time:.1f}s)"
                )
                
            except Exception as e:
                # Tratar erro na execução
                task.retry_count += 1
                execution_time = time.time() - start_time
                
                scheduler_logger.error(
                    f"❌ Erro na execução de {task.name}: {e} (tentativa {task.retry_count}/{task.config.max_retries})"
                )
                
                # Verificar se deve tentar novamente
                if task.retry_count < task.config.max_retries:
                    # Reagendar para retry
                    retry_delay = task.config.retry_delay_seconds * task.retry_count
                    task.next_run = datetime.now() + timedelta(seconds=retry_delay)
                    scheduler_logger.info(f"🔄 Reagendando {task.name} para retry em {retry_delay}s")
                else:
                    scheduler_logger.error(f"💥 Máximo de tentativas excedido para {task.name}")
            
            finally:
                # Reagendar próxima execução se não foi forçada
                if not forced and task.config.enabled:
                    task.next_run = self._calculate_next_run(task.config, task.last_run)
                
                task.is_running = False
                
        except Exception as e:
            scheduler_logger.error(f"❌ Erro fatal na execução de {task.name}: {e}")
            task.is_running = False
    
    def _build_execution_context(self, task: ScheduledTask) -> Dict[str, Any]:
        """Construir contexto para avaliação de condições"""
        try:
            context = {
                "task_id": task.id,
                "task_name": task.name,
                "last_execution_time": task.last_run,
                "execution_count": task.execution_count,
                "retry_count": task.retry_count,
                "current_time": datetime.now(),
                "scheduler_stats": {
                    "running_tasks": len([t for t in self.tasks.values() if t.is_running]),
                    "total_tasks": len(self.tasks)
                }
            }
            
            # Adicionar metadados da configuração
            if task.config.metadata:
                context.update(task.config.metadata)
            
            return context
            
        except Exception as e:
            scheduler_logger.error(f"❌ Erro ao construir contexto para {task.name}: {e}")
            return {}
    
    def _calculate_next_run(self, 
                           config: ScheduleConfig, 
                           last_run: Optional[datetime] = None) -> datetime:
        """Calcular próxima execução baseado na configuração"""
        try:
            now = datetime.now()
            
            if config.schedule_type == ScheduleType.IMMEDIATE:
                return now
            
            elif config.schedule_type == ScheduleType.INTERVAL:
                if config.interval_seconds is None:
                    raise ValueError("interval_seconds requerido para ScheduleType.INTERVAL")
                
                if last_run:
                    return last_run + timedelta(seconds=config.interval_seconds)
                else:
                    return now + timedelta(seconds=config.interval_seconds)
            
            elif config.schedule_type == ScheduleType.CONDITIONAL:
                # Para execução condicional, agendar verificação em 1 minuto
                return now + timedelta(minutes=1)
            
            elif config.schedule_type == ScheduleType.HYBRID:
                # Combinar intervalo com condições
                if config.interval_seconds is None:
                    config.interval_seconds = 3600  # Default 1 hora
                
                if last_run:
                    return last_run + timedelta(seconds=config.interval_seconds)
                else:
                    return now
            
            else:
                raise ValueError(f"ScheduleType desconhecido: {config.schedule_type}")
                
        except Exception as e:
            scheduler_logger.error(f"❌ Erro ao calcular próxima execução: {e}")
            return datetime.now() + timedelta(minutes=5)  # Fallback
    
    def _cleanup_threads(self):
        """Limpar threads finalizadas"""
        try:
            active_threads = []
            for thread in self.task_threads:
                if thread.is_alive():
                    active_threads.append(thread)
            
            cleaned_count = len(self.task_threads) - len(active_threads)
            self.task_threads = active_threads
            
            if cleaned_count > 0:
                scheduler_logger.debug(f"🧹 {cleaned_count} threads finalizadas removidas")
                
        except Exception as e:
            scheduler_logger.error(f"❌ Erro ao limpar threads: {e}")

# ========== FUNÇÕES UTILITÁRIAS ==========

def create_simple_scheduler() -> SmartScheduler:
    """Criar scheduler simples para desenvolvimento"""
    return SmartScheduler(max_concurrent_tasks=3)

def create_production_scheduler() -> SmartScheduler:
    """Criar scheduler para produção"""
    return SmartScheduler(max_concurrent_tasks=10) 