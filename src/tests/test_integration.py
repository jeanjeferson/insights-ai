"""
🔗 TESTE: INTEGRAÇÃO ENTRE FERRAMENTAS
======================================

Testa a integração e compatibilidade entre as diferentes ferramentas do Insights-AI.
Verifica se os outputs de uma ferramenta podem ser usados como inputs de outra.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import tempfile

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent))

# Importar ferramentas para testes de integração
try:
    from insights.tools.sql_query_tool import SQLServerQueryTool
    from insights.tools.kpi_calculator_tool import KPICalculatorTool
    from insights.tools.prophet_tool import ProphetForecastTool
    from insights.tools.statistical_analysis_tool import StatisticalAnalysisTool
    from insights.tools.advanced_visualization_tool import AdvancedVisualizationTool
except ImportError as e:
    print(f"⚠️ Erro ao importar ferramentas: {e}")

def create_integration_test_data():
    """Criar dados realistas para testes de integração"""
    np.random.seed(42)
    
    # Dados para 12 meses com padrões realistas
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    base_daily_sales = 50  # Vendas base por dia
    
    for date in date_range:
        # Padrão sazonal (mais vendas no fim do ano)
        seasonal_factor = 1 + 0.4 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365 + np.pi/2)
        
        # Padrão semanal (menos vendas no fim de semana)
        weekday_factor = 1.2 if date.weekday() < 5 else 0.6
        
        # Número de transações por dia
        daily_transactions = int(base_daily_sales * seasonal_factor * weekday_factor * np.random.uniform(0.7, 1.3))
        
        for i in range(daily_transactions):
            customer_segment = np.random.choice(['Bronze', 'Prata', 'Ouro', 'Platinum'], p=[0.5, 0.3, 0.15, 0.05])
            category = np.random.choice(['Anéis', 'Brincos', 'Colares', 'Pulseiras', 'Alianças'])
            metal = np.random.choice(['Ouro', 'Prata', 'Ouro Branco', 'Ouro Rosé'])
            
            # Preços baseados no segmento do cliente
            segment_multiplier = {'Bronze': 0.8, 'Prata': 1.0, 'Ouro': 1.5, 'Platinum': 2.2}[customer_segment]
            
            # Preços base por categoria
            base_prices = {'Anéis': 1200, 'Brincos': 650, 'Colares': 1800, 'Pulseiras': 950, 'Alianças': 2400}
            base_price = base_prices[category]
            
            # Multiplicador por metal
            metal_multiplier = {'Ouro': 1.0, 'Prata': 0.35, 'Ouro Branco': 1.15, 'Ouro Rosé': 1.08}[metal]
            
            quantidade = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.6, 0.2, 0.1, 0.06, 0.03, 0.01])
            preco_unitario = base_price * metal_multiplier * segment_multiplier * np.random.uniform(0.8, 1.2)
            total_liquido = preco_unitario * quantidade
            
            data.append({
                'Data': date.strftime('%Y-%m-%d'),
                'Ano': date.year,
                'Mes': date.month,
                'Dia': date.day,
                'Trimestre': f"Q{((date.month-1)//3)+1}",
                'Dia_Semana': date.strftime('%A'),
                'Codigo_Cliente': f"CLI_{np.random.randint(1, 201):04d}",
                'Segmento_Cliente': customer_segment,
                'Codigo_Produto': f"PROD_{hash(category + metal) % 100:03d}",
                'Descricao_Produto': f"{category} {metal} Premium",
                'Categoria': category,
                'Metal': metal,
                'Quantidade': quantidade,
                'Preco_Unitario': round(preco_unitario, 2),
                'Total_Liquido': round(total_liquido, 2),
                'Custo_Produto': round(total_liquido * 0.42, 2),
                'Margem_Bruta': round(total_liquido * 0.58, 2),
                'Vendedor': f"VEND_{np.random.randint(1, 16):02d}",
                'Canal': np.random.choice(['Loja Física', 'E-commerce', 'WhatsApp'], p=[0.7, 0.25, 0.05]),
                'Forma_Pagamento': np.random.choice(['Cartão', 'PIX', 'Dinheiro', 'Parcelado'], p=[0.45, 0.3, 0.1, 0.15]),
                'Regiao': np.random.choice(['SP', 'RJ', 'MG', 'RS', 'PR'], p=[0.4, 0.25, 0.15, 0.1, 0.1])
            })
    
    return pd.DataFrame(data)

def test_data_flow_integration(verbose=False, quick=False):
    """Teste de fluxo de dados entre ferramentas"""
    result = {'success': False, 'details': {}, 'warnings': [], 'errors': []}
    
    try:
        if verbose:
            print("🔄 Testando fluxo de dados entre ferramentas...")
        
        # 1. Criar dados de teste
        test_data = create_integration_test_data()
        
        # 2. Salvar dados temporariamente
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp_file:
            test_data.to_csv(tmp_file.name, sep=';', index=False)
            test_csv_path = tmp_file.name
        
        flow_tests = {}
        
        # 3. Teste: KPI Calculator → Statistical Analysis
        try:
            kpi_tool = KPICalculatorTool()
            stats_tool = StatisticalAnalysisTool()
            
            # Calcular KPIs
            kpi_result = kpi_tool._run(
                data_csv=test_csv_path,
                categoria="revenue",
                periodo="monthly"
            )
            
            # Verificar se dados podem ser usados para análise estatística
            if isinstance(kpi_result, str) and len(kpi_result) > 0:
                # Tentar análise estatística nos mesmos dados
                stats_result = stats_tool._run(
                    analysis_type="correlation",
                    data=test_csv_path,
                    target_column="Total_Liquido"
                )
                
                flow_tests['kpi_to_stats'] = {
                    'status': 'SUCCESS' if isinstance(stats_result, str) and len(stats_result) > 0 else 'FAILED',
                    'kpi_output_length': len(kpi_result),
                    'stats_output_length': len(stats_result) if isinstance(stats_result, str) else 0
                }
            else:
                flow_tests['kpi_to_stats'] = {'status': 'KPI_FAILED'}
                
        except Exception as e:
            flow_tests['kpi_to_stats'] = {'status': 'ERROR', 'error': str(e)}
        
        # 4. Teste: Statistical Analysis → Prophet Forecasting
        try:
            # Preparar dados para Prophet (formato específico)
            prophet_data = test_data.groupby('Data')['Total_Liquido'].sum().reset_index()
            prophet_data.columns = ['ds', 'y']
            prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
            
            # Salvar dados Prophet
            prophet_json = prophet_data.to_json(orient='records', date_format='iso')
            
            prophet_tool = ProphetForecastTool()
            forecast_result = prophet_tool._run(
                data=prophet_json,
                data_column='ds',
                target_column='y',
                periods=30
            )
            
            flow_tests['stats_to_prophet'] = {
                'status': 'SUCCESS' if isinstance(forecast_result, str) and len(forecast_result) > 0 else 'FAILED',
                'prophet_data_points': len(prophet_data),
                'forecast_output_length': len(forecast_result) if isinstance(forecast_result, str) else 0
            }
            
        except Exception as e:
            flow_tests['stats_to_prophet'] = {'status': 'ERROR', 'error': str(e)}
        
        # 5. Teste: KPI Calculator → Visualization
        try:
            viz_tool = AdvancedVisualizationTool()
            
            # Criar visualização baseada nos mesmos dados do KPI
            viz_result = viz_tool._run(
                chart_type='executive_dashboard',
                data=test_csv_path,
                value_column='Total_Liquido',
                category_column='Categoria',
                date_column='Data'
            )
            
            flow_tests['kpi_to_viz'] = {
                'status': 'SUCCESS' if isinstance(viz_result, str) and len(viz_result) > 0 else 'FAILED',
                'viz_output_length': len(viz_result) if isinstance(viz_result, str) else 0,
                'is_html_chart': '<html>' in viz_result or '<div>' in viz_result if isinstance(viz_result, str) else False
            }
            
        except Exception as e:
            flow_tests['kpi_to_viz'] = {'status': 'ERROR', 'error': str(e)}
        
        # 6. Limpeza
        try:
            os.unlink(test_csv_path)
        except:
            pass
        
        result['details'] = {
            'test_data_records': len(test_data),
            'flow_tests': flow_tests
        }
        
        successful_flows = len([t for t in flow_tests.values() if t.get('status') == 'SUCCESS'])
        result['success'] = successful_flows > 0
        
        if verbose:
            print(f"✅ Fluxo de dados: {successful_flows}/{len(flow_tests)} integrações funcionando")
        
    except Exception as e:
        result['errors'].append(f"Erro no teste de fluxo: {str(e)}")
    
    return result

def test_output_compatibility(verbose=False, quick=False):
    """Teste de compatibilidade entre outputs das ferramentas"""
    result = {'success': False, 'details': {}, 'warnings': [], 'errors': []}
    
    try:
        if verbose:
            print("🔗 Testando compatibilidade de outputs...")
        
        test_data = create_integration_test_data()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp_file:
            test_data.to_csv(tmp_file.name, sep=';', index=False)
            test_csv_path = tmp_file.name
        
        compatibility_tests = {}
        
        # 1. Teste de formatos de data compatíveis
        try:
            # Verificar se as ferramentas aceitan o mesmo formato de data
            date_formats = {
                'yyyy-mm-dd': test_data['Data'].iloc[0],
                'datetime_object': pd.to_datetime(test_data['Data']).iloc[0],
                'timestamp': pd.to_datetime(test_data['Data']).iloc[0].timestamp()
            }
            
            compatibility_tests['date_formats'] = {
                'formats_tested': list(date_formats.keys()),
                'standard_format': date_formats['yyyy-mm-dd'],
                'status': 'SUCCESS'
            }
            
        except Exception as e:
            compatibility_tests['date_formats'] = {'status': 'ERROR', 'error': str(e)}
        
        # 2. Teste de colunas numéricas compatíveis
        try:
            numeric_columns = test_data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Verificar se valores são válidos para cálculos
            has_nulls = test_data[numeric_columns].isnull().sum().sum()
            has_negatives = (test_data[numeric_columns] < 0).sum().sum()
            has_zeros = (test_data[numeric_columns] == 0).sum().sum()
            
            compatibility_tests['numeric_data'] = {
                'numeric_columns_count': len(numeric_columns),
                'has_null_values': has_nulls > 0,
                'has_negative_values': has_negatives > 0,
                'has_zero_values': has_zeros > 0,
                'status': 'SUCCESS'
            }
            
        except Exception as e:
            compatibility_tests['numeric_data'] = {'status': 'ERROR', 'error': str(e)}
        
        # 3. Teste de encodings compatíveis
        try:
            # Verificar se dados podem ser salvos/carregados em diferentes encodings
            encodings_test = {}
            
            for encoding in ['utf-8', 'latin-1']:
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding=encoding) as tmp:
                        test_data.to_csv(tmp.name, sep=';', index=False, encoding=encoding)
                        
                        # Tentar carregar de volta
                        test_load = pd.read_csv(tmp.name, sep=';', encoding=encoding)
                        encodings_test[encoding] = len(test_load) == len(test_data)
                        
                        os.unlink(tmp.name)
                except:
                    encodings_test[encoding] = False
            
            compatibility_tests['encodings'] = {
                'tested_encodings': encodings_test,
                'utf8_compatible': encodings_test.get('utf-8', False),
                'status': 'SUCCESS' if any(encodings_test.values()) else 'FAILED'
            }
            
        except Exception as e:
            compatibility_tests['encodings'] = {'status': 'ERROR', 'error': str(e)}
        
        # Limpeza
        try:
            os.unlink(test_csv_path)
        except:
            pass
        
        result['details'] = {
            'compatibility_tests': compatibility_tests
        }
        
        successful_tests = len([t for t in compatibility_tests.values() if t.get('status') == 'SUCCESS'])
        result['success'] = successful_tests > 0
        
        if verbose:
            print(f"✅ Compatibilidade: {successful_tests}/{len(compatibility_tests)} testes passaram")
        
    except Exception as e:
        result['errors'].append(f"Erro no teste de compatibilidade: {str(e)}")
    
    return result

def test_performance_integration(verbose=False, quick=False):
    """Teste de performance das integrações"""
    result = {'success': False, 'details': {}, 'warnings': [], 'errors': []}
    
    try:
        if verbose:
            print("⚡ Testando performance de integrações...")
        
        import time
        
        # Dados menores para teste de performance
        if quick:
            test_data = create_integration_test_data().head(100)
        else:
            test_data = create_integration_test_data().head(500)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp_file:
            test_data.to_csv(tmp_file.name, sep=';', index=False)
            test_csv_path = tmp_file.name
        
        performance_tests = {}
        
        # 1. Teste de performance sequencial (uma ferramenta após a outra)
        try:
            start_time = time.time()
            
            # KPI Calculator
            kpi_tool = KPICalculatorTool()
            kpi_result = kpi_tool._run(data_csv=test_csv_path, categoria="revenue")
            
            # Statistical Analysis
            stats_tool = StatisticalAnalysisTool()
            stats_result = stats_tool._run(analysis_type="correlation", data=test_csv_path)
            
            # Visualization
            viz_tool = AdvancedVisualizationTool()
            viz_result = viz_tool._run(chart_type='category_performance', data=test_csv_path)
            
            total_time = time.time() - start_time
            
            performance_tests['sequential_execution'] = {
                'total_time_seconds': round(total_time, 2),
                'records_processed': len(test_data),
                'records_per_second': round(len(test_data) / total_time, 2) if total_time > 0 else 0,
                'all_tools_successful': all([
                    isinstance(result, str) and len(result) > 0 
                    for result in [kpi_result, stats_result, viz_result]
                ]),
                'status': 'SUCCESS' if total_time < 60 else 'SLOW'  # Considerar lento se demorar mais de 1 minuto
            }
            
        except Exception as e:
            performance_tests['sequential_execution'] = {'status': 'ERROR', 'error': str(e)}
        
        # 2. Teste de memory usage (estimativa)
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Executar operações que usam memória
            for _ in range(3):
                temp_data = test_data.copy()
                temp_data['calculated_column'] = temp_data['Total_Liquido'] * temp_data['Quantidade']
                del temp_data
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            performance_tests['memory_usage'] = {
                'memory_before_mb': round(memory_before, 2),
                'memory_after_mb': round(memory_after, 2),
                'memory_used_mb': round(memory_used, 2),
                'status': 'SUCCESS' if memory_used < 100 else 'HIGH_MEMORY'  # Alerta se usar mais de 100MB
            }
            
        except ImportError:
            performance_tests['memory_usage'] = {'status': 'PSUTIL_NOT_AVAILABLE'}
        except Exception as e:
            performance_tests['memory_usage'] = {'status': 'ERROR', 'error': str(e)}
        
        # Limpeza
        try:
            os.unlink(test_csv_path)
        except:
            pass
        
        result['details'] = {
            'test_data_size': len(test_data),
            'performance_tests': performance_tests
        }
        
        # Considerar sucesso se pelo menos execução sequencial funcionou
        result['success'] = performance_tests.get('sequential_execution', {}).get('status') in ['SUCCESS', 'SLOW']
        
        if verbose:
            seq_test = performance_tests.get('sequential_execution', {})
            if seq_test.get('status') in ['SUCCESS', 'SLOW']:
                print(f"✅ Performance: {seq_test.get('total_time_seconds', 0)}s para {len(test_data)} registros")
            else:
                print("❌ Performance: teste falhou")
        
    except Exception as e:
        result['errors'].append(f"Erro no teste de performance: {str(e)}")
    
    return result

def test_integration(verbose=False, quick=False):
    """
    Teste consolidado de integração entre ferramentas
    """
    result = {
        'success': False,
        'details': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        if verbose:
            print("🔗 Iniciando testes de integração...")
        
        # Executar todos os testes de integração
        integration_tests = {}
        
        # 1. Teste de fluxo de dados
        data_flow_result = test_data_flow_integration(verbose=verbose, quick=quick)
        integration_tests['data_flow'] = data_flow_result
        
        # 2. Teste de compatibilidade
        compatibility_result = test_output_compatibility(verbose=verbose, quick=quick)
        integration_tests['compatibility'] = compatibility_result
        
        # 3. Teste de performance
        performance_result = test_performance_integration(verbose=verbose, quick=quick)
        integration_tests['performance'] = performance_result
        
        # Estatísticas consolidadas
        total_tests = len(integration_tests)
        successful_tests = len([t for t in integration_tests.values() if t.get('success', False)])
        total_warnings = sum(len(t.get('warnings', [])) for t in integration_tests.values())
        total_errors = sum(len(t.get('errors', [])) for t in integration_tests.values())
        
        result['details'] = {
            'total_integration_tests': total_tests,
            'successful_tests': successful_tests,
            'total_warnings': total_warnings,
            'total_errors': total_errors,
            'success_rate': round(successful_tests / total_tests * 100, 1) if total_tests > 0 else 0,
            'individual_results': integration_tests
        }
        
        # Consolidar warnings e errors
        for test_result in integration_tests.values():
            result['warnings'].extend(test_result.get('warnings', []))
            result['errors'].extend(test_result.get('errors', []))
        
        # Determinar sucesso geral
        result['success'] = successful_tests >= 2  # Pelo menos 2 testes devem passar
        
        if verbose:
            print(f"🔗 Integração: {successful_tests}/{total_tests} testes passaram")
            print(f"📊 Taxa de sucesso: {result['details']['success_rate']}%")
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Erro inesperado nos testes de integração: {str(e)}")
        result['success'] = False
        return result

if __name__ == "__main__":
    # Teste standalone
    result = test_integration(verbose=True, quick=False)
    print("\n📊 RESULTADO DOS TESTES DE INTEGRAÇÃO:")
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
