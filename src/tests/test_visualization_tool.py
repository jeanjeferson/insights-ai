"""
📊 TESTE: VISUALIZATION TOOL
============================

Testa a ferramenta de visualização avançada do projeto Insights-AI.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import base64

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent))

try:
    from insights.tools.advanced_visualization_tool import AdvancedVisualizationTool
except ImportError as e:
    print(f"⚠️ Erro ao importar AdvancedVisualizationTool: {e}")
    AdvancedVisualizationTool = None

def create_visualization_test_data():
    """Criar dados para testes de visualização"""
    np.random.seed(42)
    
    # Dados de vendas com múltiplas dimensões
    n_samples = 150
    start_date = datetime(2024, 1, 1)
    
    data = []
    for i in range(n_samples):
        date = start_date + timedelta(days=i % 365)
        grupo = np.random.choice(['Anéis', 'Brincos', 'Colares', 'Pulseiras'])
        metal = np.random.choice(['Ouro', 'Prata', 'Ouro Branco'])
        
        # Valor baseado em sazonalidade e tipo
        base_value = 1000
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
        group_multiplier = {'Anéis': 1.5, 'Brincos': 0.8, 'Colares': 2.0, 'Pulseiras': 1.2}[grupo]
        metal_multiplier = {'Ouro': 1.0, 'Prata': 0.4, 'Ouro Branco': 1.2}[metal]
        
        value = base_value * seasonal_factor * group_multiplier * metal_multiplier * np.random.uniform(0.7, 1.3)
        
        data.append({
            'Data': date.strftime('%Y-%m-%d'),
            'Ano': date.year,
            'Mes': date.month,
            'Codigo_Produto': f"PROD_{i % 30:03d}",
            'Descricao_Produto': f"{grupo} {metal}",
            'Grupo_Produto': grupo,
            'Metal': metal,
            'Quantidade': np.random.randint(1, 4),
            'Total_Liquido': value,
            'Preco_Unitario': value / np.random.randint(1, 4),
            'Customer_ID': f"CLI_{i % 50:03d}"
        })
    
    return pd.DataFrame(data)

def test_visualization_tool(verbose=False, quick=False):
    """
    Teste da ferramenta Advanced Visualization Tool
    """
    result = {
        'success': False,
        'details': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        if verbose:
            print("📊 Testando Advanced Visualization Tool...")
        
        # 1. Verificar se a classe foi importada
        if AdvancedVisualizationTool is None:
            result['errors'].append("Não foi possível importar AdvancedVisualizationTool")
            return result
        
        # 2. Verificar dependências de visualização
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            import matplotlib.pyplot as plt
            import seaborn as sns
            dependencies_ok = True
            if verbose:
                print("✅ Dependências de visualização disponíveis")
        except ImportError as e:
            dependencies_ok = False
            result['errors'].append(f"Dependências de visualização ausentes: {str(e)}")
            return result
        
        # 3. Instanciar a ferramenta
        try:
            viz_tool = AdvancedVisualizationTool()
            if verbose:
                print("✅ AdvancedVisualizationTool instanciada com sucesso")
        except Exception as e:
            result['errors'].append(f"Erro ao instanciar AdvancedVisualizationTool: {str(e)}")
            return result
        
        # 4. Verificar atributos da ferramenta
        tool_info = {
            'name': getattr(viz_tool, 'name', 'N/A'),
            'description': getattr(viz_tool, 'description', 'N/A')[:200] + "..." if len(getattr(viz_tool, 'description', '')) > 200 else getattr(viz_tool, 'description', 'N/A')
        }
        
        # 5. Criar dados de teste
        try:
            test_data = create_visualization_test_data()
            if verbose:
                print(f"✅ Dados de teste criados: {len(test_data)} registros")
        except Exception as e:
            result['errors'].append(f"Erro ao criar dados de teste: {str(e)}")
            return result
        
        # 6. Salvar dados de teste temporariamente
        test_csv_path = "temp_viz_test_data.csv"
        try:
            test_data.to_csv(test_csv_path, sep=';', index=False, encoding='utf-8')
            test_file_created = True
        except Exception as e:
            result['warnings'].append(f"Não foi possível criar arquivo de teste: {str(e)}")
            test_file_created = False
        
        # 7. Testar diferentes tipos de visualização
        chart_types = [
            'executive_dashboard',
            'sales_trends', 
            'product_analysis',
            'seasonal_heatmap',
            'category_performance'
        ]
        
        if quick:
            chart_types = ['sales_trends', 'category_performance']  # Testes mais rápidos
        
        visualization_results = {}
        
        for chart_type in chart_types:
            try:
                if verbose:
                    print(f"🎨 Testando visualização: {chart_type}")
                
                # Usar dados de teste se arquivo foi criado
                data_source = test_csv_path if test_file_created else "data/vendas.csv"
                
                chart_result = viz_tool._run(
                    chart_type=chart_type,
                    data=data_source,
                    value_column='Total_Liquido',
                    category_column='Grupo_Produto',
                    date_column='Data'
                )
                
                # Verificar resultado
                if isinstance(chart_result, str) and len(chart_result) > 0:
                    # Verificar se é HTML válido (para Plotly) ou base64 (para matplotlib)
                    is_html = '<html>' in chart_result or '<div>' in chart_result
                    is_base64 = chart_result.startswith('iVBORw0KGgo') or chart_result.startswith('/9j/')
                    is_json = chart_result.startswith('{') and chart_result.endswith('}')
                    
                    visualization_results[chart_type] = {
                        'status': 'SUCCESS',
                        'output_length': len(chart_result),
                        'format': 'HTML' if is_html else 'BASE64' if is_base64 else 'JSON' if is_json else 'OTHER',
                        'has_plotly_elements': 'plotly' in chart_result.lower(),
                        'sample_output': chart_result[:100] + "..." if len(chart_result) > 100 else chart_result
                    }
                else:
                    visualization_results[chart_type] = {
                        'status': 'EMPTY_RESULT',
                        'output': str(chart_result)
                    }
                    result['warnings'].append(f"Visualização {chart_type} retornou resultado vazio")
                
            except Exception as e:
                visualization_results[chart_type] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                result['warnings'].append(f"Erro na visualização {chart_type}: {str(e)}")
        
        # 8. Testar métodos auxiliares de visualização
        auxiliary_methods = {}
        
        # Testar _create_executive_dashboard
        if hasattr(viz_tool, '_create_executive_dashboard'):
            try:
                dashboard_result = viz_tool._create_executive_dashboard(test_data)
                auxiliary_methods['create_executive_dashboard'] = {
                    'status': 'OK' if dashboard_result else 'NULL_RESULT',
                    'is_plotly': 'plotly' in str(type(dashboard_result)).lower() if dashboard_result else False,
                    'is_html': isinstance(dashboard_result, str) and '<html>' in dashboard_result
                }
            except Exception as e:
                auxiliary_methods['create_executive_dashboard'] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        # Testar _create_sales_trends
        if hasattr(viz_tool, '_create_sales_trends'):
            try:
                trends_result = viz_tool._create_sales_trends(test_data, date_column='Data', value_column='Total_Liquido')
                auxiliary_methods['create_sales_trends'] = {
                    'status': 'OK' if trends_result else 'NULL_RESULT',
                    'has_temporal_data': 'Data' in test_data.columns
                }
            except Exception as e:
                auxiliary_methods['create_sales_trends'] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        # Testar _create_category_performance
        if hasattr(viz_tool, '_create_category_performance'):
            try:
                category_result = viz_tool._create_category_performance(test_data, category_column='Grupo_Produto', value_column='Total_Liquido')
                auxiliary_methods['create_category_performance'] = {
                    'status': 'OK' if category_result else 'NULL_RESULT',
                    'has_categories': 'Grupo_Produto' in test_data.columns
                }
            except Exception as e:
                auxiliary_methods['create_category_performance'] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        # 9. Testar geração de outputs específicos
        output_tests = {}
        
        # Teste de geração de HTML
        try:
            html_test = viz_tool._run(
                chart_type='sales_trends',
                data=test_csv_path if test_file_created else "data/vendas.csv",
                output_format='html'
            )
            output_tests['html_generation'] = {
                'status': 'SUCCESS' if isinstance(html_test, str) and len(html_test) > 0 else 'FAILED',
                'is_valid_html': '<html>' in html_test or '<div>' in html_test if isinstance(html_test, str) else False
            }
        except Exception as e:
            output_tests['html_generation'] = {
                'status': 'ERROR',
                'error': str(e)
            }
        
        # Teste de geração de base64
        try:
            if hasattr(viz_tool, '_save_chart'):
                # Simular criação de gráfico matplotlib
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
                plt.title("Teste")
                
                base64_result = viz_tool._save_chart()
                plt.close()
                
                output_tests['base64_generation'] = {
                    'status': 'SUCCESS' if isinstance(base64_result, str) and len(base64_result) > 0 else 'FAILED',
                    'is_valid_base64': base64_result.replace('+', '').replace('/', '').replace('=', '').isalnum() if isinstance(base64_result, str) else False
                }
            else:
                output_tests['base64_generation'] = {'status': 'METHOD_NOT_FOUND'}
        except Exception as e:
            output_tests['base64_generation'] = {
                'status': 'ERROR',
                'error': str(e)
            }
        
        # 10. Testar tratamento de erros
        error_handling = {}
        
        # Teste com tipo de gráfico inválido
        try:
            if test_file_created:
                invalid_chart = viz_tool._run(
                    chart_type="grafico_inexistente",
                    data=test_csv_path
                )
                error_handling['invalid_chart_type'] = 'ERROR_HANDLED' if 'não suportado' in invalid_chart else 'NO_ERROR'
        except Exception as e:
            error_handling['invalid_chart_type'] = 'EXCEPTION'
        
        # Teste com dados inválidos
        try:
            invalid_data = viz_tool._run(
                chart_type="sales_trends",
                data="arquivo_inexistente.csv"
            )
            error_handling['invalid_data'] = 'ERROR_HANDLED' if 'erro' in invalid_data.lower() else 'NO_ERROR'
        except Exception as e:
            error_handling['invalid_data'] = 'EXCEPTION'
        
        # 11. Testar performance de visualização
        performance_metrics = {}
        
        try:
            import time
            start_time = time.time()
            
            # Visualização simples para medir performance
            if test_file_created:
                perf_result = viz_tool._run(
                    chart_type='category_performance',
                    data=test_csv_path
                )
            
            end_time = time.time()
            performance_metrics = {
                'execution_time': round(end_time - start_time, 2),
                'success': isinstance(perf_result, str) and len(perf_result) > 0
            }
            
        except Exception as e:
            performance_metrics = {
                'error': str(e)
            }
        
        # 12. Verificar formatos de saída suportados
        supported_formats = {}
        
        format_indicators = {
            'html': ['<html>', '<div>', 'plotly'],
            'base64': ['iVBORw0KGgo', '/9j/', 'data:image'],
            'json': ['{', '}', 'data'],
            'svg': ['<svg', '</svg>']
        }
        
        # Analisar resultados para identificar formatos
        for chart_type, result_data in visualization_results.items():
            if result_data.get('status') == 'SUCCESS':
                sample_output = result_data.get('sample_output', '')
                for format_name, indicators in format_indicators.items():
                    if any(indicator in sample_output for indicator in indicators):
                        if format_name not in supported_formats:
                            supported_formats[format_name] = []
                        supported_formats[format_name].append(chart_type)
        
        # 13. Limpeza
        if test_file_created:
            try:
                os.remove(test_csv_path)
            except:
                pass
        
        # 14. Compilar resultados
        result['details'] = {
            'tool_info': tool_info,
            'dependencies_ok': dependencies_ok,
            'test_data_stats': {
                'rows': len(test_data),
                'date_range': f"{test_data['Data'].min()} até {test_data['Data'].max()}",
                'categories': test_data['Grupo_Produto'].nunique(),
                'metals': test_data['Metal'].nunique()
            },
            'visualization_results': visualization_results,
            'auxiliary_methods': auxiliary_methods,
            'output_tests': output_tests,
            'error_handling': error_handling,
            'performance_metrics': performance_metrics,
            'supported_formats': supported_formats
        }
        
        # 15. Determinar sucesso
        successful_charts = len([r for r in visualization_results.values() if r.get('status') == 'SUCCESS'])
        total_charts = len(visualization_results)
        
        if successful_charts > 0 and dependencies_ok:
            result['success'] = True
            if verbose:
                print(f"✅ Visualization Tool: {successful_charts}/{total_charts} visualizações funcionando")
        else:
            if verbose:
                print("❌ Visualization Tool com problemas")
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Erro inesperado no teste Visualization: {str(e)}")
        result['success'] = False
        return result

def test_chart_generation():
    """Teste específico de geração de gráficos"""
    try:
        # Teste básico com matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title("Teste Básico")
        
        # Verificar se o gráfico foi criado
        chart_created = len(fig.axes) > 0
        plt.close(fig)
        
        # Teste básico com plotly se disponível
        plotly_test = False
        try:
            import plotly.graph_objects as go
            fig_plotly = go.Figure()
            fig_plotly.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13]))
            plotly_test = True
        except:
            pass
        
        tests = {
            'matplotlib': chart_created,
            'plotly': plotly_test
        }
        
        return all(tests.values()), f"Testes: {tests}"
        
    except Exception as e:
        return False, f"Erro: {str(e)}"

if __name__ == "__main__":
    # Teste standalone
    result = test_visualization_tool(verbose=True, quick=False)
    print("\n📊 RESULTADO DO TESTE VISUALIZATION:")
    print(f"✅ Sucesso: {result['success']}")
    print(f"⚠️ Warnings: {len(result['warnings'])}")
    print(f"❌ Erros: {len(result['errors'])}")
    
    if result['warnings']:
        print("\nWarnings:")
        for warning in result['warnings']:
            print(f"  - {warning}")
    
    if result['errors']:
        print("\nErros:")
        for error in result['errors']:
            print(f"  - {error}")
    
    # Teste adicional de geração de gráficos
    print("\n📈 TESTE DE GERAÇÃO DE GRÁFICOS:")
    success, message = test_chart_generation()
    print(f"{'✅' if success else '❌'} {message}")
