"""
🚀 SIMPLE PERFORMANCE PROFILER - Profiler sem dependências externas
===================================================================

Profiler simplificado que não depende do psutil para medição básica de performance.
"""

import time
import tracemalloc
from typing import Dict, Any, List, Optional
from datetime import datetime


class SimplePerformanceProfiler:
    """
    Profiler de performance simplificado sem dependências externas.
    """
    
    def __init__(self):
        self.measurements = []
        self.start_time = None
        self.start_memory = None
        
    def start_measurement(self, operation: str = "default") -> Dict[str, Any]:
        """Iniciar medição de performance."""
        # Iniciar rastreamento de memória
        tracemalloc.start()
        
        self.start_time = time.time()
        
        # Obter uso inicial de memória
        current, peak = tracemalloc.get_traced_memory()
        self.start_memory = current / 1024 / 1024  # MB
        
        return {
            'operation': operation,
            'start_time': self.start_time,
            'start_memory_mb': self.start_memory
        }
    
    def end_measurement(self, operation: str = "default") -> Dict[str, Any]:
        """Finalizar medição de performance."""
        end_time = time.time()
        
        # Obter uso final de memória
        current, peak = tracemalloc.get_traced_memory()
        end_memory = current / 1024 / 1024  # MB
        peak_memory = peak / 1024 / 1024  # MB
        
        tracemalloc.stop()
        
        # Calcular métricas
        duration = end_time - (self.start_time or end_time)
        memory_used = end_memory - (self.start_memory or 0)
        
        measurement = {
            'operation': operation,
            'duration_seconds': round(duration, 3),
            'memory_used_mb': round(memory_used, 2),
            'memory_peak_mb': round(peak_memory, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        self.measurements.append(measurement)
        return measurement
    
    def add_measurement(self, operation: str, metrics: Dict[str, Any]):
        """Adicionar medição manual."""
        measurement = {
            'operation': operation,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.measurements.append(measurement)
    
    def get_summary(self) -> Dict[str, Any]:
        """Obter resumo das medições."""
        if not self.measurements:
            return {'error': 'Nenhuma medição disponível'}
        
        durations = [m.get('duration_seconds', 0) for m in self.measurements if 'duration_seconds' in m]
        memories = [m.get('memory_used_mb', 0) for m in self.measurements if 'memory_used_mb' in m]
        
        return {
            'total_measurements': len(self.measurements),
            'total_duration': round(sum(durations), 3),
            'avg_duration': round(sum(durations) / len(durations), 3) if durations else 0,
            'total_memory_used': round(sum(memories), 2),
            'avg_memory_used': round(sum(memories) / len(memories), 2) if memories else 0,
            'measurements': self.measurements
        }
    
    def clear(self):
        """Limpar todas as medições."""
        self.measurements.clear()
        self.start_time = None
        self.start_memory = None


def measure_function_performance(func, *args, **kwargs) -> Dict[str, Any]:
    """
    Medir performance de uma função específica.
    
    Args:
        func: Função a ser medida
        *args: Argumentos da função
        **kwargs: Argumentos nomeados da função
        
    Returns:
        Dict com métricas de performance e resultado da função
    """
    profiler = SimplePerformanceProfiler()
    
    # Iniciar medição
    profiler.start_measurement(func.__name__)
    
    try:
        # Executar função
        result = func(*args, **kwargs)
        success = True
        error = None
    except Exception as e:
        result = None
        success = False
        error = str(e)
    
    # Finalizar medição
    metrics = profiler.end_measurement(func.__name__)
    
    return {
        'success': success,
        'result': result,
        'error': error,
        'performance': metrics
    } 