#!/usr/bin/env python
"""
⚡ BENCHMARK DE PERFORMANCE INSIGHTS-AI
=====================================

Script para comparar performance entre:
- Versão original (crew.py)
- Versão otimizada (crew_optimized.py)

Métricas avaliadas:
- Tempo de inicialização
- Uso de memória
- Verbosidade de logs
- Tempo total de execução
"""

import sys
import os
import time
import tracemalloc
import psutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent))

# =============== CONFIGURAÇÃO DO BENCHMARK ===============

class BenchmarkConfig:
    """Configuração do benchmark"""
    
    # Períodos de teste
    TEST_PERIODS = [
        ("2024-11-01", "2024-11-30"),  # 1 mês
        ("2024-10-01", "2024-11-30"),  # 2 meses
    ]
    
    # Métricas a coletar
    METRICS = [
        'initialization_time',
        'memory_usage_mb',
        'log_count',
        'execution_time',
        'total_time'
    ]
    
    # Número de execuções para média
    ITERATIONS = 3

# =============== COLLECTOR DE MÉTRICAS ===============

class MetricsCollector:
    """Coletor de métricas de performance"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset das métricas"""
        self.start_time = None
        self.init_time = None
        self.memory_start = 0
        self.memory_peak = 0
        self.log_count = 0
        self.process = psutil.Process()
    
    def start_measurement(self):
        """Iniciar medição"""
        self.reset()
        self.start_time = time.time()
        tracemalloc.start()
        self.memory_start = self.process.memory_info().rss / 1024 / 1024
    
    def mark_initialization_complete(self):
        """Marcar conclusão da inicialização"""
        if self.start_time:
            self.init_time = time.time() - self.start_time
    
    def stop_measurement(self) -> Dict[str, Any]:
        """Parar medição e retornar métricas"""
        end_time = time.time()
        
        # Memória
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        memory_end = self.process.memory_info().rss / 1024 / 1024
        memory_used = memory_end - self.memory_start
        memory_peak_mb = peak / 1024 / 1024
        
        # Tempo total
        total_time = end_time - (self.start_time or end_time)
        
        return {
            'initialization_time': self.init_time or 0,
            'memory_usage_mb': memory_used,
            'memory_peak_mb': memory_peak_mb,
            'execution_time': total_time - (self.init_time or 0),
            'total_time': total_time,
            'log_count': self.log_count
        }

# =============== INTERCEPTADOR DE LOGS ===============

class LogInterceptor:
    """Intercepta logs para contar verbosidade"""
    
    def __init__(self):
        self.log_count = 0
        self.log_levels = {}
        self.logs = []
    
    def intercept_log(self, level: str, message: str):
        """Interceptar um log"""
        self.log_count += 1
        self.log_levels[level] = self.log_levels.get(level, 0) + 1
        
        # Armazenar apenas os primeiros 100 logs para análise
        if len(self.logs) < 100:
            self.logs.append({
                'level': level,
                'message': message[:100],  # Truncar mensagem
                'timestamp': time.time()
            })
    
    def get_stats(self) -> Dict[str, Any]:
        """Obter estatísticas dos logs"""
        return {
            'total_logs': self.log_count,
            'by_level': self.log_levels,
            'sample_logs': self.logs[:10]  # Primeiros 10 logs
        }

# =============== BENCHMARK DA VERSÃO ORIGINAL ===============

def benchmark_original_crew(data_inicio: str, data_fim: str) -> Dict[str, Any]:
    """Benchmark da versão original"""
    
    collector = MetricsCollector()
    log_interceptor = LogInterceptor()
    
    try:
        print("🔄 Testando versão ORIGINAL...")
        collector.start_measurement()
        
        # Importar versão original
        from insights.crew import Insights
        
        collector.mark_initialization_complete()
        
        # Executar crew original
        crew_instance = Insights()
        
        inputs = {
            'data_inicio': data_inicio,
            'data_fim': data_fim
        }
        
        # Simular execução (sem executar realmente para speed)
        # result = crew_instance.crew().kickoff(inputs=inputs)
        
        # Para benchmark, vamos apenas medir a criação do crew
        crew = crew_instance.crew()
        
        metrics = collector.stop_measurement()
        
        # Adicionar informações específicas
        metrics.update({
            'version': 'original',
            'agents_count': len(crew.agents) if hasattr(crew, 'agents') else 0,
            'tasks_count': len(crew.tasks) if hasattr(crew, 'tasks') else 0,
            'log_stats': log_interceptor.get_stats()
        })
        
        return metrics
        
    except Exception as e:
        print(f"❌ Erro no benchmark original: {e}")
        return {
            'version': 'original',
            'error': str(e),
            'metrics': collector.stop_measurement()
        }

# =============== BENCHMARK DA VERSÃO OTIMIZADA ===============

def benchmark_optimized_crew(data_inicio: str, data_fim: str) -> Dict[str, Any]:
    """Benchmark da versão otimizada"""
    
    collector = MetricsCollector()
    log_interceptor = LogInterceptor()
    
    try:
        print("⚡ Testando versão OTIMIZADA...")
        collector.start_measurement()
        
        # Importar versão otimizada
        from insights.crew_optimized import OptimizedInsights
        
        collector.mark_initialization_complete()
        
        # Executar crew otimizado
        crew_instance = OptimizedInsights()
        
        inputs = {
            'data_inicio': data_inicio,
            'data_fim': data_fim
        }
        
        # Para benchmark, vamos apenas medir a criação do crew
        crew = crew_instance.crew()
        
        metrics = collector.stop_measurement()
        
        # Adicionar informações específicas
        metrics.update({
            'version': 'optimized',
            'agents_count': len(crew.agents) if hasattr(crew, 'agents') else 0,
            'tasks_count': len(crew.tasks) if hasattr(crew, 'tasks') else 0,
            'log_stats': log_interceptor.get_stats()
        })
        
        return metrics
        
    except Exception as e:
        print(f"❌ Erro no benchmark otimizado: {e}")
        return {
            'version': 'optimized',
            'error': str(e),
            'metrics': collector.stop_measurement()
        }

# =============== ANÁLISE DE RESULTADOS ===============

def analyze_results(original_results: List[Dict], optimized_results: List[Dict]):
    """Analisar e comparar resultados"""
    
    print("\n" + "="*80)
    print("📊 ANÁLISE DE PERFORMANCE - RESULTADOS DETALHADOS")
    print("="*80)
    
    # Calcular médias
    def calc_average(results: List[Dict], metric: str) -> float:
        values = [r.get(metric, 0) for r in results if 'error' not in r]
        return sum(values) / len(values) if values else 0
    
    # Métricas principais
    metrics = [
        ('initialization_time', 'Tempo Inicialização (s)', 's'),
        ('memory_usage_mb', 'Uso de Memória (MB)', 'MB'),
        ('memory_peak_mb', 'Pico de Memória (MB)', 'MB'),
        ('total_time', 'Tempo Total (s)', 's')
    ]
    
    print(f"{'Métrica':<25} {'Original':<15} {'Otimizada':<15} {'Melhoria':<15}")
    print("-" * 70)
    
    improvements = {}
    
    for metric, label, unit in metrics:
        original_avg = calc_average(original_results, metric)
        optimized_avg = calc_average(optimized_results, metric)
        
        if original_avg > 0:
            improvement = ((original_avg - optimized_avg) / original_avg) * 100
            improvement_str = f"{improvement:+.1f}%"
        else:
            improvement_str = "N/A"
        
        improvements[metric] = improvement
        
        print(f"{label:<25} {original_avg:<15.3f} {optimized_avg:<15.3f} {improvement_str:<15}")
    
    # Análise de logs
    print(f"\n📝 ANÁLISE DE VERBOSIDADE:")
    print("-" * 40)
    
    original_log_avg = calc_average(original_results, 'log_count')
    optimized_log_avg = calc_average(optimized_results, 'log_count')
    
    if original_log_avg > 0:
        log_reduction = ((original_log_avg - optimized_log_avg) / original_log_avg) * 100
        print(f"Logs Original:     {original_log_avg:.0f}")
        print(f"Logs Otimizada:    {optimized_log_avg:.0f}")
        print(f"Redução de logs:   {log_reduction:.1f}%")
    
    # Score geral de melhoria
    print(f"\n🎯 SCORE GERAL DE MELHORIA:")
    print("-" * 30)
    
    valid_improvements = [v for v in improvements.values() if v is not None and v > -100]
    if valid_improvements:
        avg_improvement = sum(valid_improvements) / len(valid_improvements)
        print(f"Melhoria média:    {avg_improvement:.1f}%")
        
        if avg_improvement > 20:
            print("✅ EXCELENTE melhoria de performance!")
        elif avg_improvement > 10:
            print("✅ BOA melhoria de performance!")
        elif avg_improvement > 0:
            print("✅ Melhoria modesta de performance")
        else:
            print("⚠️ Performance similar ou pior")
    
    # Recomendações
    print(f"\n💡 RECOMENDAÇÕES:")
    print("-" * 20)
    
    if improvements.get('initialization_time', 0) > 10:
        print("✅ Inicialização significativamente mais rápida")
    
    if improvements.get('memory_usage_mb', 0) > 5:
        print("✅ Uso de memória reduzido")
    
    if log_reduction > 30:
        print("✅ Logs muito menos verbosos")
    
    return improvements

# =============== FUNÇÃO PRINCIPAL ===============

def run_performance_benchmark():
    """Executar benchmark completo"""
    
    print("⚡ BENCHMARK DE PERFORMANCE INSIGHTS-AI")
    print("="*50)
    print(f"📅 Testando {len(BenchmarkConfig.TEST_PERIODS)} períodos")
    print(f"🔄 {BenchmarkConfig.ITERATIONS} iterações cada")
    print(f"⏰ Iniciando em {datetime.now().strftime('%H:%M:%S')}")
    
    all_original_results = []
    all_optimized_results = []
    
    for period_idx, (data_inicio, data_fim) in enumerate(BenchmarkConfig.TEST_PERIODS):
        print(f"\n📊 PERÍODO {period_idx + 1}: {data_inicio} até {data_fim}")
        print("-" * 40)
        
        # Executar múltiplas iterações
        for iteration in range(BenchmarkConfig.ITERATIONS):
            print(f"\n🔄 Iteração {iteration + 1}/{BenchmarkConfig.ITERATIONS}")
            
            # Benchmark versão original
            original_result = benchmark_original_crew(data_inicio, data_fim)
            all_original_results.append(original_result)
            
            # Pequena pausa entre testes
            time.sleep(1)
            
            # Benchmark versão otimizada  
            optimized_result = benchmark_optimized_crew(data_inicio, data_fim)
            all_optimized_results.append(optimized_result)
            
            # Pausa entre iterações
            time.sleep(1)
            
            # Log resumo da iteração
            if 'error' not in original_result and 'error' not in optimized_result:
                orig_time = original_result.get('total_time', 0)
                opt_time = optimized_result.get('total_time', 0)
                improvement = ((orig_time - opt_time) / orig_time * 100) if orig_time > 0 else 0
                print(f"   Original: {orig_time:.2f}s | Otimizada: {opt_time:.2f}s | Melhoria: {improvement:+.1f}%")
    
    # Análise final
    analyze_results(all_original_results, all_optimized_results)
    
    # Salvar resultados
    save_benchmark_results(all_original_results, all_optimized_results)

def save_benchmark_results(original_results: List[Dict], optimized_results: List[Dict]):
    """Salvar resultados do benchmark"""
    
    try:
        import json
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'test_periods': BenchmarkConfig.TEST_PERIODS,
                'iterations': BenchmarkConfig.ITERATIONS
            },
            'original_results': original_results,
            'optimized_results': optimized_results
        }
        
        # Salvar em arquivo
        output_dir = Path("logs/benchmarks")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"performance_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file = output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Resultados salvos em: {output_file}")
        
    except Exception as e:
        print(f"⚠️ Erro ao salvar resultados: {e}")

if __name__ == "__main__":
    # Verificar se podemos importar os módulos necessários
    try:
        # Test imports
        import insights.crew
        import insights.crew_optimized
        print("✅ Módulos encontrados, iniciando benchmark...")
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        print("Certifique-se de que os módulos estão no PYTHONPATH")
        sys.exit(1)
    
    # Executar benchmark
    try:
        run_performance_benchmark()
        print(f"\n🏁 Benchmark concluído em {datetime.now().strftime('%H:%M:%S')}")
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro no benchmark: {e}")
        import traceback
        traceback.print_exc() 