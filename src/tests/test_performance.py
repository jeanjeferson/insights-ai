"""
⚡ TESTE: PERFORMANCE E STRESS
=============================

Testa a performance e limites das ferramentas do Insights-AI.
Inclui testes de stress, memory usage, tempo de execução e escalabilidade.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import gc
import tempfile

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent))

# Importar ferramentas para teste de performance
try:
    from insights.tools.kpi_calculator_tool import KPICalculatorTool
    from insights.tools.prophet_tool import ProphetForecastTool
    from insights.tools.statistical_analysis_tool import StatisticalAnalysisTool
    from insights.tools.advanced_visualization_tool import AdvancedVisualizationTool
except ImportError as e:
    print(f"⚠️ Erro ao importar ferramentas: {e}")

def create_stress_test_data(size='medium'):
    """Criar dados de diferentes tamanhos para testes de stress"""
    np.random.seed(42)
    
    # Definir tamanho do dataset
    sizes = {
        'small': 1000,      # 1k registros
        'medium': 10000,    # 10k registros
        'large': 50000,     # 50k registros
        'xlarge': 100000    # 100k registros
    }
    
    n_records = sizes.get(size, sizes['medium'])
    
    if n_records > 50000:
        print(f"⚠️ Gerando dataset grande ({n_records:,} registros) - pode demorar...")
    
    # Gerar dados otimizados para performance
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # Pre-gerar arrays para performance
    dates = np.random.choice(pd.date_range(start_date, end_date), n_records)
    customers = np.random.choice([f"CLI_{i:06d}" for i in range(1, min(n_records//10, 10000))], n_records)
    products = np.random.choice([f"PROD_{i:04d}" for i in range(1, min(n_records//50, 1000))], n_records)
    categories = np.random.choice(['Anéis', 'Brincos', 'Colares', 'Pulseiras', 'Alianças'], n_records)
    metals = np.random.choice(['Ouro', 'Prata', 'Ouro Branco', 'Ouro Rosé'], n_records)
    
    # Gerar valores numéricos
    base_prices = np.random.normal(1500, 500, n_records)
    quantities = np.random.choice([1, 1, 1, 2, 2, 3], n_records, p=[0.5, 0.2, 0.15, 0.1, 0.03, 0.02])
    total_values = base_prices * quantities * np.random.uniform(0.8, 1.2, n_records)
    
    # Criar DataFrame diretamente com arrays (mais eficiente)
    data = {
        'Data': dates.strftime('%Y-%m-%d'),
        'Ano': dates.year,
        'Mes': dates.month,
        'Codigo_Cliente': customers,
        'Codigo_Produto': products,
        'Categoria': categories,
        'Metal': metals,
        'Quantidade': quantities,
        'Total_Liquido': np.round(total_values, 2),
        'Preco_Unitario': np.round(total_values / quantities, 2),
        'Custo_Produto': np.round(total_values * 0.4, 2),
        'Margem_Bruta': np.round(total_values * 0.6, 2)
    }
    
    return pd.DataFrame(data)

def test_performance_scalability(verbose=False, quick=False):
    """Teste de escalabilidade com diferentes tamanhos de dados"""
    result = {'success': False, 'details': {}, 'warnings': [], 'errors': []}
    
    try:
        if verbose:
            print("📈 Testando escalabilidade das ferramentas...")
        
        # Definir tamanhos para teste
        if quick:
            test_sizes = ['small', 'medium']
        else:
            test_sizes = ['small', 'medium', 'large']
        
        scalability_results = {}
        
        for size in test_sizes:
            if verbose:
                print(f"🔍 Testando com dataset {size}...")
            
            try:
                # Gerar dados
                start_time = time.time()
                test_data = create_stress_test_data(size)
                data_generation_time = time.time() - start_time
                
                # Salvar temporariamente
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp_file:
                    test_data.to_csv(tmp_file.name, sep=';', index=False)
                    test_csv_path = tmp_file.name
                
                # Teste KPI Calculator
                kpi_start = time.time()
                kpi_tool = KPICalculatorTool()
                kpi_result = kpi_tool._run(data_csv=test_csv_path, categoria="revenue", periodo="monthly")
                kpi_time = time.time() - kpi_start
                kpi_success = isinstance(kpi_result, str) and len(kpi_result) > 0
                
                # Teste Statistical Analysis
                stats_start = time.time()
                stats_tool = StatisticalAnalysisTool()
                stats_result = stats_tool._run(analysis_type="correlation", data=test_csv_path, target_column="Total_Liquido")
                stats_time = time.time() - stats_start
                stats_success = isinstance(stats_result, str) and len(stats_result) > 0
                
                # Limpeza
                os.unlink(test_csv_path)
                
                scalability_results[size] = {
                    'records': len(test_data),
                    'data_generation_time': round(data_generation_time, 2),
                    'kpi_time': round(kpi_time, 2),
                    'kpi_success': kpi_success,
                    'kpi_records_per_second': round(len(test_data) / kpi_time, 0) if kpi_time > 0 else 0,
                    'stats_time': round(stats_time, 2),
                    'stats_success': stats_success,
                    'stats_records_per_second': round(len(test_data) / stats_time, 0) if stats_time > 0 else 0,
                    'total_processing_time': round(kpi_time + stats_time, 2)
                }
                
                if verbose:
                    print(f"  ✅ {size}: {len(test_data):,} registros em {scalability_results[size]['total_processing_time']}s")
                
            except Exception as e:
                scalability_results[size] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                result['warnings'].append(f"Erro no teste {size}: {str(e)}")
        
        result['details'] = {
            'scalability_results': scalability_results,
            'test_sizes': test_sizes
        }
        
        # Verificar se pelo menos um tamanho funcionou
        successful_sizes = len([r for r in scalability_results.values() if r.get('kpi_success', False)])
        result['success'] = successful_sizes > 0
        
        if verbose:
            print(f"✅ Escalabilidade: {successful_sizes}/{len(test_sizes)} tamanhos funcionaram")
        
    except Exception as e:
        result['errors'].append(f"Erro no teste de escalabilidade: {str(e)}")
    
    return result

def test_memory_usage(verbose=False, quick=False):
    """Teste de uso de memória"""
    result = {'success': False, 'details': {}, 'warnings': [], 'errors': []}
    
    try:
        if verbose:
            print("💾 Testando uso de memória...")
        
        # Tentar importar psutil para monitoramento de memória
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_monitoring = True
        except ImportError:
            memory_monitoring = False
            result['warnings'].append("psutil não disponível - monitoramento de memória limitado")
        
        memory_tests = {}
        
        # Baseline de memória
        if memory_monitoring:
            gc.collect()  # Força garbage collection
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        else:
            baseline_memory = 0
        
        # Teste com dataset medium
        test_data = create_stress_test_data('medium')
        
        if memory_monitoring:
            after_data_memory = process.memory_info().rss / 1024 / 1024
            data_memory_usage = after_data_memory - baseline_memory
        else:
            data_memory_usage = 0
        
        # Teste de múltiplas operações
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp_file:
            test_data.to_csv(tmp_file.name, sep=';', index=False)
            test_csv_path = tmp_file.name
        
        # Múltiplas execuções para testar vazamentos de memória
        memory_usage_per_iteration = []
        
        for i in range(3 if quick else 5):
            if memory_monitoring:
                before_iteration = process.memory_info().rss / 1024 / 1024
            
            # Executar operações
            kpi_tool = KPICalculatorTool()
            kpi_result = kpi_tool._run(data_csv=test_csv_path, categoria="revenue")
            
            stats_tool = StatisticalAnalysisTool()
            stats_result = stats_tool._run(analysis_type="correlation", data=test_csv_path)
            
            # Força limpeza
            del kpi_tool, stats_tool
            gc.collect()
            
            if memory_monitoring:
                after_iteration = process.memory_info().rss / 1024 / 1024
                iteration_usage = after_iteration - before_iteration
                memory_usage_per_iteration.append(iteration_usage)
        
        # Limpeza
        os.unlink(test_csv_path)
        
        # Análise de vazamentos
        if memory_usage_per_iteration:
            avg_usage = np.mean(memory_usage_per_iteration)
            max_usage = np.max(memory_usage_per_iteration)
            memory_trend = np.polyfit(range(len(memory_usage_per_iteration)), memory_usage_per_iteration, 1)[0]
            
            memory_tests = {
                'baseline_memory_mb': round(baseline_memory, 2),
                'data_memory_usage_mb': round(data_memory_usage, 2),
                'avg_iteration_usage_mb': round(avg_usage, 2),
                'max_iteration_usage_mb': round(max_usage, 2),
                'memory_trend': round(memory_trend, 3),  # MB por iteração
                'potential_leak': memory_trend > 5,  # Mais de 5MB por iteração indica possível vazamento
                'total_iterations': len(memory_usage_per_iteration),
                'usage_per_iteration': [round(x, 2) for x in memory_usage_per_iteration]
            }
            
            if memory_trend > 5:
                result['warnings'].append(f"Possível vazamento de memória detectado: {memory_trend:.2f}MB por iteração")
        else:
            memory_tests = {
                'monitoring_available': False,
                'message': 'Monitoramento de memória não disponível'
            }
        
        result['details'] = {
            'memory_monitoring_available': memory_monitoring,
            'test_data_size': len(test_data),
            'memory_tests': memory_tests
        }
        
        result['success'] = not memory_tests.get('potential_leak', False)
        
        if verbose:
            if memory_monitoring:
                print(f"✅ Memória: uso médio {memory_tests.get('avg_iteration_usage_mb', 0)}MB por operação")
            else:
                print("⚠️ Memória: monitoramento não disponível")
        
    except Exception as e:
        result['errors'].append(f"Erro no teste de memória: {str(e)}")
    
    return result

def test_concurrent_execution(verbose=False, quick=False):
    """Teste de execução concorrente (simulada)"""
    result = {'success': False, 'details': {}, 'warnings': [], 'errors': []}
    
    try:
        if verbose:
            print("🔄 Testando execução concorrente...")
        
        # Criar múltiplos datasets pequenos
        datasets = []
        csv_paths = []
        
        num_datasets = 3 if quick else 5
        
        for i in range(num_datasets):
            test_data = create_stress_test_data('small')
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp_file:
                test_data.to_csv(tmp_file.name, sep=';', index=False)
                csv_paths.append(tmp_file.name)
                datasets.append(test_data)
        
        # Teste sequencial vs "concorrente" (simulado)
        concurrent_tests = {}
        
        # Execução sequencial
        sequential_start = time.time()
        sequential_results = []
        
        for csv_path in csv_paths:
            kpi_tool = KPICalculatorTool()
            result_seq = kpi_tool._run(data_csv=csv_path, categoria="revenue")
            sequential_results.append(isinstance(result_seq, str) and len(result_seq) > 0)
        
        sequential_time = time.time() - sequential_start
        
        # Simulação de execução "concorrente" (na verdade sequencial rápida)
        concurrent_start = time.time()
        concurrent_results = []
        
        # Reutilizar a mesma instância da ferramenta (simula shared resources)
        shared_kpi_tool = KPICalculatorTool()
        
        for csv_path in csv_paths:
            result_conc = shared_kpi_tool._run(data_csv=csv_path, categoria="revenue")
            concurrent_results.append(isinstance(result_conc, str) and len(result_conc) > 0)
        
        concurrent_time = time.time() - concurrent_start
        
        # Limpeza
        for csv_path in csv_paths:
            try:
                os.unlink(csv_path)
            except:
                pass
        
        concurrent_tests = {
            'num_datasets': num_datasets,
            'sequential_time': round(sequential_time, 2),
            'sequential_success_rate': round(sum(sequential_results) / len(sequential_results) * 100, 1),
            'concurrent_time': round(concurrent_time, 2),
            'concurrent_success_rate': round(sum(concurrent_results) / len(concurrent_results) * 100, 1),
            'time_difference': round(sequential_time - concurrent_time, 2),
            'efficiency_gain': round((sequential_time - concurrent_time) / sequential_time * 100, 1) if sequential_time > 0 else 0
        }
        
        result['details'] = {
            'concurrent_tests': concurrent_tests,
            'total_records_processed': sum(len(df) for df in datasets)
        }
        
        # Sucesso se ambas execuções tiveram pelo menos 80% de sucesso
        result['success'] = (concurrent_tests['sequential_success_rate'] >= 80 and 
                           concurrent_tests['concurrent_success_rate'] >= 80)
        
        if verbose:
            print(f"✅ Concorrência: {concurrent_tests['concurrent_success_rate']}% sucesso, {concurrent_tests['efficiency_gain']}% ganho")
        
    except Exception as e:
        result['errors'].append(f"Erro no teste de concorrência: {str(e)}")
    
    return result

def test_performance_stress(verbose=False, quick=False):
    """
    Teste consolidado de performance e stress
    """
    result = {
        'success': False,
        'details': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        if verbose:
            print("⚡ Iniciando testes de performance e stress...")
        
        # Executar todos os testes de performance
        performance_tests = {}
        
        # 1. Teste de escalabilidade
        scalability_result = test_performance_scalability(verbose=verbose, quick=quick)
        performance_tests['scalability'] = scalability_result
        
        # 2. Teste de uso de memória
        memory_result = test_memory_usage(verbose=verbose, quick=quick)
        performance_tests['memory'] = memory_result
        
        # 3. Teste de execução concorrente
        concurrent_result = test_concurrent_execution(verbose=verbose, quick=quick)
        performance_tests['concurrent'] = concurrent_result
        
        # Estatísticas consolidadas
        total_tests = len(performance_tests)
        successful_tests = len([t for t in performance_tests.values() if t.get('success', False)])
        total_warnings = sum(len(t.get('warnings', [])) for t in performance_tests.values())
        total_errors = sum(len(t.get('errors', [])) for t in performance_tests.values())
        
        result['details'] = {
            'total_performance_tests': total_tests,
            'successful_tests': successful_tests,
            'total_warnings': total_warnings,
            'total_errors': total_errors,
            'success_rate': round(successful_tests / total_tests * 100, 1) if total_tests > 0 else 0,
            'individual_results': performance_tests
        }
        
        # Consolidar warnings e errors
        for test_result in performance_tests.values():
            result['warnings'].extend(test_result.get('warnings', []))
            result['errors'].extend(test_result.get('errors', []))
        
        # Determinar sucesso geral
        result['success'] = successful_tests >= 2  # Pelo menos 2 testes devem passar
        
        if verbose:
            print(f"⚡ Performance: {successful_tests}/{total_tests} testes passaram")
            print(f"📊 Taxa de sucesso: {result['details']['success_rate']}%")
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Erro inesperado nos testes de performance: {str(e)}")
        result['success'] = False
        return result

if __name__ == "__main__":
    # Teste standalone
    result = test_performance_stress(verbose=True, quick=False)
    print("\n📊 RESULTADO DOS TESTES DE PERFORMANCE:")
    print(f"✅ Sucesso: {result['success']}")
    print(f"📈 Taxa de Sucesso: {result['details'].get('success_rate', 0)}%")
    print(f"⚠️ Warnings: {len(result['warnings'])}")
    print(f"❌ Erros: {len(result['errors'])}")
    
    if result['warnings']:
        print("\nWarnings:")
        for warning in result['warnings'][:3]:
            print(f"  - {warning}")
        if len(result['warnings']) > 3:
            print(f"  ... e mais {len(result['warnings']) - 3} warnings")
    
    if result['errors']:
        print("\nErros:")
        for error in result['errors'][:3]:
            print(f"  - {error}")
        if len(result['errors']) > 3:
            print(f"  ... e mais {len(result['errors']) - 3} erros")
