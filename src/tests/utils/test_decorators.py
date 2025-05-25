"""
🎯 TEST DECORATORS - Decoradores para Testes Avançados
======================================================

Decoradores para automatizar logging, medição de performance e validações
durante a execução dos testes.
"""

import time
import psutil
import traceback
from functools import wraps
from typing import Callable, Any, Dict, Optional
from datetime import datetime


def log_test_execution(include_performance: bool = True, log_level: str = "INFO"):
    """
    Decorador para logging automático de execução de testes.
    
    Args:
        include_performance: Se deve incluir métricas de performance
        log_level: Nível de log a usar
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extrair instância da classe de teste se disponível
            test_instance = None
            if args and hasattr(args[0], '__class__'):
                test_instance = args[0]
            
            # Obter logger se disponível na instância
            logger = getattr(test_instance, 'logger', None) if test_instance else None
            
            # Configurar métricas iniciais
            start_time = time.time()
            start_memory = 0
            process = None
            
            if include_performance:
                try:
                    process = psutil.Process()
                    start_memory = process.memory_info().rss / 1024 / 1024
                except:
                    pass
            
            # Log início
            if logger:
                logger.log(log_level, f"🚀 Iniciando {func.__name__}")
            else:
                print(f"🚀 [{datetime.now().strftime('%H:%M:%S')}] Iniciando {func.__name__}")
            
            try:
                # Executar função original
                result = func(*args, **kwargs)
                success = True
                error_message = None
                
            except Exception as e:
                success = False
                error_message = str(e)
                result = None
                
                if logger:
                    logger.log_error(e, f"Erro em {func.__name__}")
                else:
                    print(f"❌ Erro em {func.__name__}: {str(e)}")
                
                # Re-raise a exceção para pytest
                raise
                
            finally:
                # Calcular métricas finais
                duration = time.time() - start_time
                memory_used = 0
                
                if include_performance and process:
                    try:
                        end_memory = process.memory_info().rss / 1024 / 1024
                        memory_used = end_memory - start_memory
                    except:
                        pass
                
                # Log final
                status = "✅ SUCESSO" if success else "❌ FALHA"
                
                if logger:
                    logger.log(log_level, f"{status} - {func.__name__} concluído",
                             duration=f"{duration:.2f}s",
                             memory_used=f"{memory_used:.1f}MB" if include_performance else "N/A")
                else:
                    perf_info = f" | {duration:.2f}s | {memory_used:.1f}MB" if include_performance else f" | {duration:.2f}s"
                    print(f"{status} [{datetime.now().strftime('%H:%M:%S')}] {func.__name__}{perf_info}")
            
            return result
        
        return wrapper
    return decorator


def measure_performance(threshold_seconds: Optional[float] = None, 
                       memory_threshold_mb: Optional[float] = None):
    """
    Decorador para medição detalhada de performance.
    
    Args:
        threshold_seconds: Limite de tempo em segundos (alerta se exceder)
        memory_threshold_mb: Limite de memória em MB (alerta se exceder)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extrair instância da classe de teste
            test_instance = None
            if args and hasattr(args[0], '__class__'):
                test_instance = args[0]
            
            # Obter profiler se disponível
            profiler = getattr(test_instance, 'profiler', None) if test_instance else None
            logger = getattr(test_instance, 'logger', None) if test_instance else None
            
            # Iniciar medição
            start_time = time.time()
            start_metrics = {}
            
            try:
                process = psutil.Process()
                start_metrics = {
                    'memory_rss': process.memory_info().rss / 1024 / 1024,
                    'memory_vms': process.memory_info().vms / 1024 / 1024,
                    'cpu_percent': process.cpu_percent(),
                    'num_threads': process.num_threads(),
                    'open_files': len(process.open_files()) if hasattr(process, 'open_files') else 0
                }
            except Exception as e:
                if logger:
                    logger.log("WARNING", f"Erro ao obter métricas iniciais: {str(e)}")
                start_metrics = {'error': str(e)}
            
            try:
                # Executar função
                result = func(*args, **kwargs)
                
                # Medir métricas finais
                end_time = time.time()
                duration = end_time - start_time
                
                end_metrics = {}
                try:
                    process = psutil.Process()
                    end_metrics = {
                        'memory_rss': process.memory_info().rss / 1024 / 1024,
                        'memory_vms': process.memory_info().vms / 1024 / 1024,
                        'cpu_percent': process.cpu_percent(),
                        'num_threads': process.num_threads(),
                        'open_files': len(process.open_files()) if hasattr(process, 'open_files') else 0
                    }
                except Exception as e:
                    end_metrics = {'error': str(e)}
                
                # Calcular diferenças
                performance_metrics = {
                    'duration_seconds': round(duration, 3),
                    'memory_used_mb': round(end_metrics.get('memory_rss', 0) - start_metrics.get('memory_rss', 0), 2),
                    'memory_peak_mb': round(end_metrics.get('memory_rss', 0), 2),
                    'cpu_avg_percent': round((start_metrics.get('cpu_percent', 0) + end_metrics.get('cpu_percent', 0)) / 2, 1),
                    'threads_change': end_metrics.get('num_threads', 0) - start_metrics.get('num_threads', 0),
                    'open_files_change': end_metrics.get('open_files', 0) - start_metrics.get('open_files', 0)
                }
                
                # Verificar thresholds e gerar alertas
                alerts = []
                if threshold_seconds and duration > threshold_seconds:
                    alerts.append(f"⚠️ TIMEOUT: Execução demorou {duration:.2f}s (limite: {threshold_seconds}s)")
                
                if memory_threshold_mb and performance_metrics['memory_used_mb'] > memory_threshold_mb:
                    alerts.append(f"⚠️ MEMÓRIA: Uso de {performance_metrics['memory_used_mb']:.1f}MB (limite: {memory_threshold_mb}MB)")
                
                # Log métricas
                if profiler:
                    profiler.add_measurement(func.__name__, performance_metrics)
                
                if logger:
                    logger.log_performance(func.__name__, duration, **performance_metrics)
                    
                    if alerts:
                        for alert in alerts:
                            logger.log("WARNING", alert)
                
                # Adicionar métricas ao resultado se for um dict
                if isinstance(result, dict):
                    result['_performance_metrics'] = performance_metrics
                
                return result
                
            except Exception as e:
                # Log erro com contexto de performance
                if logger:
                    logger.log_error(e, f"Erro de performance em {func.__name__}")
                raise
        
        return wrapper
    return decorator


def require_data_quality(min_score: float = 70):
    """
    Decorador para validar qualidade mínima dos dados antes do teste.
    
    Args:
        min_score: Score mínimo de qualidade necessário (0-100)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Extrair instância da classe de teste
            test_instance = None
            if args and hasattr(args[0], '__class__'):
                test_instance = args[0]
            
            # Verificar se há validador de qualidade
            data_validator = getattr(test_instance, 'data_validator', None) if test_instance else None
            logger = getattr(test_instance, 'logger', None) if test_instance else None
            
            if data_validator:
                quality_data = getattr(test_instance, 'data_quality', {}) if test_instance else {}
                overall_score = quality_data.get('overall_score', 0)
                
                if overall_score < min_score:
                    error_msg = f"Qualidade dos dados insuficiente: {overall_score}/100 (mínimo: {min_score})"
                    
                    if logger:
                        logger.log("ERROR", error_msg)
                    
                    import pytest
                    pytest.skip(error_msg)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def timeout_test(seconds: float):
    """
    Decorador para aplicar timeout em testes.
    
    Args:
        seconds: Tempo limite em segundos
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Teste {func.__name__} excedeu timeout de {seconds}s")
            
            # Configurar timeout (apenas em sistemas Unix)
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(seconds))
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    signal.alarm(0)  # Cancelar timeout
                    
            except AttributeError:
                # Sistema Windows - usar implementação alternativa
                import threading
                
                result_container = {'result': None, 'exception': None, 'completed': False}
                
                def run_test():
                    try:
                        result_container['result'] = func(*args, **kwargs)
                        result_container['completed'] = True
                    except Exception as e:
                        result_container['exception'] = e
                        result_container['completed'] = True
                
                thread = threading.Thread(target=run_test)
                thread.daemon = True
                thread.start()
                thread.join(seconds)
                
                if not result_container['completed']:
                    raise TimeoutError(f"Teste {func.__name__} excedeu timeout de {seconds}s")
                
                if result_container['exception']:
                    raise result_container['exception']
                
                return result_container['result']
        
        return wrapper
    return decorator


def retry_on_failure(max_attempts: int = 3, delay_seconds: float = 1.0):
    """
    Decorador para retry automático em caso de falha.
    
    Args:
        max_attempts: Número máximo de tentativas
        delay_seconds: Delay entre tentativas
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Log tentativa falhada
                    test_instance = args[0] if args and hasattr(args[0], '__class__') else None
                    logger = getattr(test_instance, 'logger', None) if test_instance else None
                    
                    if logger:
                        logger.log("WARNING", f"Tentativa {attempt + 1}/{max_attempts} falhou: {str(e)}")
                    
                    # Delay antes da próxima tentativa
                    if attempt < max_attempts - 1:
                        time.sleep(delay_seconds)
            
            # Se chegou aqui, todas as tentativas falharam
            raise last_exception
        
        return wrapper
    return decorator


class TestMetricsCollector:
    """
    Coletor de métricas para múltiplos testes.
    """
    
    def __init__(self):
        self.metrics = []
        self.start_time = time.time()
    
    def add_test_metrics(self, test_name: str, duration: float, success: bool, **kwargs):
        """Adicionar métricas de um teste."""
        self.metrics.append({
            'test_name': test_name,
            'duration': duration,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Obter resumo das métricas coletadas."""
        if not self.metrics:
            return {'error': 'Nenhuma métrica coletada'}
        
        total_duration = sum(m['duration'] for m in self.metrics)
        success_count = sum(1 for m in self.metrics if m['success'])
        
        return {
            'total_tests': len(self.metrics),
            'successful_tests': success_count,
            'failed_tests': len(self.metrics) - success_count,
            'success_rate': round(success_count / len(self.metrics) * 100, 2),
            'total_duration': round(total_duration, 3),
            'average_duration': round(total_duration / len(self.metrics), 3),
            'collection_duration': round(time.time() - self.start_time, 3)
        } 