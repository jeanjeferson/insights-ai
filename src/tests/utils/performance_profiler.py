"""
⚡ PERFORMANCE PROFILER - Sistema de Profiling Avançado para Testes
==================================================================

Sistema completo de profiling com métricas detalhadas de CPU, memória,
I/O e performance de execução para otimização dos testes.
"""

import time
import psutil
import tracemalloc
import gc
import threading
from functools import wraps
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """Classe para armazenar métricas de performance."""
    
    function_name: str
    start_time: float
    end_time: float
    duration: float
    cpu_percent: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    memory_used_mb: float
    thread_count: int
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converter para dicionário."""
        return {
            'function_name': self.function_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_s': round(self.duration, 3),
            'cpu_percent': round(self.cpu_percent, 1),
            'memory_before_mb': round(self.memory_before_mb, 2),
            'memory_after_mb': round(self.memory_after_mb, 2),
            'memory_peak_mb': round(self.memory_peak_mb, 2),
            'memory_used_mb': round(self.memory_used_mb, 2),
            'thread_count': self.thread_count,
            'timestamp': datetime.fromtimestamp(self.start_time).isoformat(),
            'context': self.context
        }


class PerformanceProfiler:
    """
    Profiler avançado de performance para testes.
    
    Features:
    - Medição de tempo de execução
    - Monitoramento de uso de memória
    - Tracking de CPU usage
    - Detecção de memory leaks
    - Profiling de I/O operations
    - Análise de concorrência
    """
    
    def __init__(self, name: str = "default_profiler"):
        self.name = name
        self.metrics_history: List[PerformanceMetrics] = []
        self.active_sessions = {}
        self.baseline_memory = None
        self.process = None
        
        # Inicializar processo psutil
        try:
            self.process = psutil.Process()
            self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
        except Exception as e:
            print(f"⚠️ Aviso: psutil não disponível: {e}")
    
    def start_profiling(self, session_name: str, **context) -> str:
        """
        Iniciar sessão de profiling.
        
        Args:
            session_name: Nome da sessão
            **context: Contexto adicional para logging
            
        Returns:
            session_id: ID único da sessão
        """
        session_id = f"{session_name}_{int(time.time())}"
        
        # Iniciar tracemalloc para memory profiling
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        # Métricas iniciais
        start_metrics = {
            'start_time': time.time(),
            'memory_before': self._get_memory_usage(),
            'cpu_start': self._get_cpu_percent(),
            'threads_start': threading.active_count(),
            'context': context
        }
        
        self.active_sessions[session_id] = start_metrics
        
        print(f"⚡ Profiling iniciado: {session_name} (ID: {session_id})")
        return session_id
    
    def stop_profiling(self, session_id: str, **context) -> PerformanceMetrics:
        """
        Finalizar sessão de profiling e retornar métricas.
        
        Args:
            session_id: ID da sessão ativa
            **context: Contexto adicional
            
        Returns:
            PerformanceMetrics: Métricas coletadas
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Sessão não encontrada: {session_id}")
        
        start_data = self.active_sessions[session_id]
        end_time = time.time()
        
        # Métricas finais
        memory_after = self._get_memory_usage()
        memory_peak = self._get_peak_memory()
        cpu_percent = self._get_cpu_percent()
        thread_count = threading.active_count()
        
        # Criar objeto de métricas
        metrics = PerformanceMetrics(
            function_name=session_id.split('_')[0],
            start_time=start_data['start_time'],
            end_time=end_time,
            duration=end_time - start_data['start_time'],
            cpu_percent=cpu_percent,
            memory_before_mb=start_data['memory_before'],
            memory_after_mb=memory_after,
            memory_peak_mb=memory_peak,
            memory_used_mb=memory_after - start_data['memory_before'],
            thread_count=thread_count,
            context={**start_data['context'], **context}
        )
        
        # Armazenar métricas
        self.metrics_history.append(metrics)
        
        # Limpar sessão
        del self.active_sessions[session_id]
        
        # Log das métricas
        self._log_metrics(metrics)
        
        return metrics
    
    def profile_function(self, func: Callable, *args, **kwargs) -> tuple:
        """
        Profile uma função específica.
        
        Returns:
            tuple: (resultado_funcao, metricas_performance)
        """
        session_id = self.start_profiling(func.__name__, 
                                        module=func.__module__,
                                        args_count=len(args),
                                        kwargs_count=len(kwargs))
        
        try:
            result = func(*args, **kwargs)
            metrics = self.stop_profiling(session_id, success=True)
            return result, metrics
        
        except Exception as e:
            metrics = self.stop_profiling(session_id, success=False, error=str(e))
            raise e
    
    def _get_memory_usage(self) -> float:
        """Obter uso atual de memória em MB."""
        if self.process:
            return self.process.memory_info().rss / 1024 / 1024
        return 0.0
    
    def _get_peak_memory(self) -> float:
        """Obter pico de uso de memória em MB."""
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            return peak / 1024 / 1024
        return self._get_memory_usage()
    
    def _get_cpu_percent(self) -> float:
        """Obter percentual de uso de CPU."""
        if self.process:
            try:
                return self.process.cpu_percent(interval=0.1)
            except:
                return 0.0
        return 0.0
    
    def _log_metrics(self, metrics: PerformanceMetrics):
        """Log das métricas coletadas."""
        print(f"📊 Performance {metrics.function_name}:")
        print(f"   ⏱️  Duração: {metrics.duration:.3f}s")
        print(f"   🧠 Memória: {metrics.memory_used_mb:+.1f}MB (pico: {metrics.memory_peak_mb:.1f}MB)")
        print(f"   💻 CPU: {metrics.cpu_percent:.1f}%")
        print(f"   🧵 Threads: {metrics.thread_count}")
    
    def detect_memory_leaks(self, threshold_mb: float = 10.0) -> List[Dict[str, Any]]:
        """
        Detectar possíveis memory leaks analisando histórico.
        
        Args:
            threshold_mb: Threshold para consideração de leak
            
        Returns:
            Lista de possíveis leaks detectados
        """
        leaks = []
        
        if len(self.metrics_history) < 2:
            return leaks
        
        total_memory_growth = 0
        for i in range(1, len(self.metrics_history)):
            current = self.metrics_history[i]
            previous = self.metrics_history[i-1]
            
            memory_growth = current.memory_after_mb - previous.memory_after_mb
            total_memory_growth += memory_growth
            
            if memory_growth > threshold_mb:
                leaks.append({
                    'function': current.function_name,
                    'memory_growth_mb': round(memory_growth, 2),
                    'timestamp': datetime.fromtimestamp(current.start_time).isoformat(),
                    'context': current.context
                })
        
        # Leak geral se crescimento total > threshold
        if total_memory_growth > threshold_mb * 2:
            leaks.append({
                'type': 'general_leak',
                'total_growth_mb': round(total_memory_growth, 2),
                'functions_analyzed': len(self.metrics_history)
            })
        
        return leaks
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Obter resumo de performance."""
        if not self.metrics_history:
            return {'error': 'Nenhuma métrica coletada'}
        
        durations = [m.duration for m in self.metrics_history]
        memory_usage = [m.memory_used_mb for m in self.metrics_history]
        cpu_usage = [m.cpu_percent for m in self.metrics_history]
        
        return {
            'total_functions_profiled': len(self.metrics_history),
            'total_execution_time': round(sum(durations), 3),
            'average_execution_time': round(sum(durations) / len(durations), 3),
            'fastest_function': min(self.metrics_history, key=lambda x: x.duration).function_name,
            'slowest_function': max(self.metrics_history, key=lambda x: x.duration).function_name,
            'average_memory_usage': round(sum(memory_usage) / len(memory_usage), 2),
            'peak_memory_usage': round(max(memory_usage), 2),
            'average_cpu_usage': round(sum(cpu_usage) / len(cpu_usage), 1),
            'memory_leaks_detected': len(self.detect_memory_leaks()),
            'baseline_memory_mb': self.baseline_memory
        }
    
    def save_performance_report(self, filename: str = None) -> Path:
        """Salvar relatório detalhado de performance."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{self.name}_{timestamp}.json"
        
        # Criar diretório
        report_dir = Path("performance_reports")
        report_dir.mkdir(exist_ok=True)
        
        # Dados do relatório
        report_data = {
            'profiler_name': self.name,
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_performance_summary(),
            'memory_leaks': self.detect_memory_leaks(),
            'detailed_metrics': [m.to_dict() for m in self.metrics_history]
        }
        
        # Salvar arquivo
        file_path = report_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Relatório de performance salvo: {file_path}")
        return file_path
    
    def clear_history(self):
        """Limpar histórico de métricas."""
        self.metrics_history.clear()
        print(f"🧹 Histórico de performance limpo para {self.name}")


def measure_performance(include_memory=True, include_cpu=True, 
                       save_report=False, profiler_name=None):
    """
    Decorator para medição automática de performance.
    
    Args:
        include_memory: Incluir métricas de memória
        include_cpu: Incluir métricas de CPU
        save_report: Salvar relatório automático
        profiler_name: Nome do profiler (usa nome da função se None)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Criar profiler
            name = profiler_name or func.__name__
            profiler = PerformanceProfiler(name)
            
            # Context do profiler
            context = {
                'function': func.__name__,
                'module': func.__module__,
                'include_memory': include_memory,
                'include_cpu': include_cpu
            }
            
            # Profile a função
            try:
                result, metrics = profiler.profile_function(func, *args, **kwargs)
                
                # Salvar relatório se solicitado
                if save_report:
                    profiler.save_performance_report()
                
                # Adicionar métricas ao resultado se for um dict
                if isinstance(result, dict):
                    result['_performance_metrics'] = metrics.to_dict()
                
                return result
                
            except Exception as e:
                print(f"❌ Erro durante profiling de {func.__name__}: {str(e)}")
                raise
        
        return wrapper
    return decorator


class PerformanceBenchmark:
    """
    Sistema de benchmarks para comparação de performance.
    """
    
    def __init__(self):
        self.benchmarks = {}
        self.load_benchmarks()
    
    def load_benchmarks(self):
        """Carregar benchmarks padrão."""
        self.benchmarks = {
            'kpi_calculation': {
                'max_duration_s': 30,
                'max_memory_mb': 200,
                'max_cpu_percent': 80
            },
            'data_loading': {
                'max_duration_s': 10,
                'max_memory_mb': 500,
                'max_cpu_percent': 60
            },
            'report_generation': {
                'max_duration_s': 45,
                'max_memory_mb': 300,
                'max_cpu_percent': 70
            },
            'statistical_analysis': {
                'max_duration_s': 60,
                'max_memory_mb': 400,
                'max_cpu_percent': 90
            }
        }
    
    def check_performance(self, metrics: PerformanceMetrics, 
                         benchmark_type: str) -> Dict[str, Any]:
        """
        Verificar se métricas atendem benchmarks.
        
        Args:
            metrics: Métricas de performance
            benchmark_type: Tipo de benchmark a verificar
            
        Returns:
            Resultado da verificação
        """
        if benchmark_type not in self.benchmarks:
            return {'error': f'Benchmark {benchmark_type} não encontrado'}
        
        benchmark = self.benchmarks[benchmark_type]
        results = {
            'benchmark_type': benchmark_type,
            'passed': True,
            'issues': []
        }
        
        # Verificar duração
        if metrics.duration > benchmark['max_duration_s']:
            results['passed'] = False
            results['issues'].append({
                'metric': 'duration',
                'actual': round(metrics.duration, 2),
                'expected_max': benchmark['max_duration_s'],
                'severity': 'high' if metrics.duration > benchmark['max_duration_s'] * 2 else 'medium'
            })
        
        # Verificar memória
        if metrics.memory_used_mb > benchmark['max_memory_mb']:
            results['passed'] = False
            results['issues'].append({
                'metric': 'memory',
                'actual': round(metrics.memory_used_mb, 1),
                'expected_max': benchmark['max_memory_mb'],
                'severity': 'high' if metrics.memory_used_mb > benchmark['max_memory_mb'] * 2 else 'medium'
            })
        
        # Verificar CPU
        if metrics.cpu_percent > benchmark['max_cpu_percent']:
            results['passed'] = False
            results['issues'].append({
                'metric': 'cpu',
                'actual': round(metrics.cpu_percent, 1),
                'expected_max': benchmark['max_cpu_percent'],
                'severity': 'medium'  # CPU usage é menos crítico
            })
        
        return results
    
    def add_custom_benchmark(self, name: str, max_duration_s: float, 
                           max_memory_mb: float, max_cpu_percent: float):
        """Adicionar benchmark customizado."""
        self.benchmarks[name] = {
            'max_duration_s': max_duration_s,
            'max_memory_mb': max_memory_mb,
            'max_cpu_percent': max_cpu_percent
        }
        print(f"📊 Benchmark customizado adicionado: {name}")


# Instância global de benchmark
global_benchmark = PerformanceBenchmark()


def benchmark_performance(benchmark_type: str):
    """
    Decorator para verificar performance contra benchmarks.
    
    Args:
        benchmark_type: Tipo de benchmark a usar
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = PerformanceProfiler(f"benchmark_{func.__name__}")
            
            # Executar com profiling
            result, metrics = profiler.profile_function(func, *args, **kwargs)
            
            # Verificar contra benchmark
            benchmark_result = global_benchmark.check_performance(metrics, benchmark_type)
            
            # Log do resultado
            if benchmark_result['passed']:
                print(f"✅ Benchmark PASSOU: {func.__name__} ({benchmark_type})")
            else:
                print(f"❌ Benchmark FALHOU: {func.__name__} ({benchmark_type})")
                for issue in benchmark_result['issues']:
                    severity_emoji = "🚨" if issue['severity'] == 'high' else "⚠️"
                    print(f"   {severity_emoji} {issue['metric']}: {issue['actual']} > {issue['expected_max']}")
            
            # Adicionar resultado do benchmark ao retorno
            if isinstance(result, dict):
                result['_benchmark_result'] = benchmark_result
            
            return result
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Exemplo de uso do profiler
    profiler = PerformanceProfiler("teste_exemplo")
    
    # Teste com profiling manual
    session_id = profiler.start_profiling("operacao_teste", contexto="exemplo")
    
    # Simular processamento
    time.sleep(1)
    data = [i**2 for i in range(100000)]  # Operação que usa CPU e memória
    
    metrics = profiler.stop_profiling(session_id, result_size=len(data))
    
    # Verificar benchmark
    benchmark_result = global_benchmark.check_performance(metrics, 'data_loading')
    print(f"\n📊 Resultado do benchmark: {benchmark_result}")
    
    # Salvar relatório
    profiler.save_performance_report()
