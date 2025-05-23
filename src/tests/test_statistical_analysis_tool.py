"""
📈 TESTE: STATISTICAL ANALYSIS TOOL
===================================

Testa a ferramenta de análise estatística do projeto Insights-AI.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent))

try:
    from insights.tools.statistical_analysis_tool import StatisticalAnalysisTool
except ImportError as e:
    print(f"⚠️ Erro ao importar StatisticalAnalysisTool: {e}")
    StatisticalAnalysisTool = None

def create_statistical_test_data():
    """Criar dados para testes estatísticos"""
    np.random.seed(42)
    
    # Gerar dados com correlações conhecidas
    n_samples = 200
    
    # Variável base
    x1 = np.random.normal(100, 15, n_samples)
    
    # Variável correlacionada positivamente
    x2 = x1 * 0.8 + np.random.normal(0, 5, n_samples)
    
    # Variável correlacionada negativamente
    x3 = -x1 * 0.6 + np.random.normal(200, 10, n_samples)
    
    # Variável independente
    x4 = np.random.normal(50, 12, n_samples)
    
    # Criar DataFrame
    data = []
    for i in range(n_samples):
        data.append({
            'Data': (datetime(2024, 1, 1) + timedelta(days=i//5)).strftime('%Y-%m-%d'),
            'Codigo_Produto': f"PROD_{(i % 20):03d}",
            'Descricao_Produto': f"Produto {(i % 20):03d}",
            'Grupo_Produto': np.random.choice(['Anéis', 'Brincos', 'Colares', 'Pulseiras']),
            'Total_Liquido': max(0, x1[i]),
            'Quantidade': max(1, int(x2[i] / 20)),
            'Preco_Unitario': max(0, x3[i]),
            'Custo_Produto': max(0, x4[i]),
            'Customer_ID': f"CLI_{(i % 50):03d}"
        })
    
    return pd.DataFrame(data)

def test_statistical_analysis_tool(verbose=False, quick=False):
    """
    Teste da ferramenta Statistical Analysis Tool
    """
    result = {
        'success': False,
        'details': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        if verbose:
            print("📈 Testando Statistical Analysis Tool...")
        
        # 1. Verificar se a classe foi importada
        if StatisticalAnalysisTool is None:
            result['errors'].append("Não foi possível importar StatisticalAnalysisTool")
            return result
        
        # 2. Verificar dependências estatísticas
        try:
            from scipy import stats
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            dependencies_ok = True
            if verbose:
                print("✅ Dependências estatísticas disponíveis")
        except ImportError as e:
            dependencies_ok = False
            result['errors'].append(f"Dependências estatísticas ausentes: {str(e)}")
            return result
        
        # 3. Instanciar a ferramenta
        try:
            stats_tool = StatisticalAnalysisTool()
            if verbose:
                print("✅ StatisticalAnalysisTool instanciada com sucesso")
        except Exception as e:
            result['errors'].append(f"Erro ao instanciar StatisticalAnalysisTool: {str(e)}")
            return result
        
        # 4. Verificar atributos da ferramenta
        tool_info = {
            'name': getattr(stats_tool, 'name', 'N/A'),
            'description': getattr(stats_tool, 'description', 'N/A')[:200] + "..." if len(getattr(stats_tool, 'description', '')) > 200 else getattr(stats_tool, 'description', 'N/A')
        }
        
        # 5. Criar dados de teste
        try:
            test_data = create_statistical_test_data()
            if verbose:
                print(f"✅ Dados de teste criados: {len(test_data)} registros")
        except Exception as e:
            result['errors'].append(f"Erro ao criar dados de teste: {str(e)}")
            return result
        
        # 6. Salvar dados de teste temporariamente
        test_csv_path = "temp_stats_test_data.csv"
        try:
            test_data.to_csv(test_csv_path, sep=';', index=False, encoding='utf-8')
            test_file_created = True
        except Exception as e:
            result['warnings'].append(f"Não foi possível criar arquivo de teste: {str(e)}")
            test_file_created = False
        
        # 7. Testar diferentes tipos de análise
        analysis_types = ['correlation', 'clustering', 'outliers', 'trend_detection', 'distribution']
        if quick:
            analysis_types = ['correlation', 'outliers']  # Testes mais rápidos
        
        analysis_results = {}
        
        for analysis_type in analysis_types:
            try:
                if verbose:
                    print(f"🔍 Testando análise: {analysis_type}")
                
                # Usar dados de teste se arquivo foi criado
                if test_file_created:
                    analysis_result = stats_tool._run(
                        analysis_type=analysis_type,
                        data=test_csv_path,
                        target_column='Total_Liquido',
                        group_column='Grupo_Produto'
                    )
                else:
                    # Tentar com dados reais
                    analysis_result = stats_tool._run(
                        analysis_type=analysis_type,
                        data="data/vendas.csv",
                        target_column='Total_Liquido'
                    )
                
                # Verificar resultado
                if isinstance(analysis_result, str) and len(analysis_result) > 0:
                    analysis_results[analysis_type] = {
                        'status': 'SUCCESS',
                        'output_length': len(analysis_result),
                        'has_statistics': any(term in analysis_result.lower() for term in ['correlation', 'mean', 'std', 'test']),
                        'has_insights': 'insight' in analysis_result.lower(),
                        'sample_output': analysis_result[:200] + "..." if len(analysis_result) > 200 else analysis_result
                    }
                else:
                    analysis_results[analysis_type] = {
                        'status': 'EMPTY_RESULT',
                        'output': str(analysis_result)
                    }
                    result['warnings'].append(f"Análise {analysis_type} retornou resultado vazio")
                
            except Exception as e:
                analysis_results[analysis_type] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                result['warnings'].append(f"Erro na análise {analysis_type}: {str(e)}")
        
        # 8. Testar métodos auxiliares específicos
        auxiliary_methods = {}
        
        # Testar _correlation_analysis
        if hasattr(stats_tool, '_correlation_analysis'):
            try:
                corr_result = stats_tool._correlation_analysis(test_data, 'Total_Liquido', 'Grupo_Produto')
                auxiliary_methods['correlation_analysis'] = {
                    'status': 'OK' if isinstance(corr_result, dict) else 'INVALID_TYPE',
                    'has_matrix': 'correlation_matrix' in corr_result if isinstance(corr_result, dict) else False,
                    'has_insights': 'insights' in corr_result if isinstance(corr_result, dict) else False
                }
            except Exception as e:
                auxiliary_methods['correlation_analysis'] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        # Testar _outlier_detection
        if hasattr(stats_tool, '_outlier_detection'):
            try:
                outlier_result = stats_tool._outlier_detection(test_data, 'Total_Liquido', 'Grupo_Produto')
                auxiliary_methods['outlier_detection'] = {
                    'status': 'OK' if isinstance(outlier_result, dict) else 'INVALID_TYPE',
                    'has_methods': 'iqr_method' in outlier_result if isinstance(outlier_result, dict) else False,
                    'detected_outliers': outlier_result.get('iqr_method', {}).get('outliers_count', 0) if isinstance(outlier_result, dict) else 0
                }
            except Exception as e:
                auxiliary_methods['outlier_detection'] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        # Testar _trend_detection
        if hasattr(stats_tool, '_trend_detection'):
            try:
                trend_result = stats_tool._trend_detection(test_data, 'Total_Liquido', 'Grupo_Produto')
                auxiliary_methods['trend_detection'] = {
                    'status': 'OK' if isinstance(trend_result, dict) else 'INVALID_TYPE',
                    'has_trend_analysis': 'trend_analysis' in trend_result if isinstance(trend_result, dict) else False,
                    'trend_direction': trend_result.get('forecast_direction', 'N/A') if isinstance(trend_result, dict) else 'N/A'
                }
            except Exception as e:
                auxiliary_methods['trend_detection'] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        # 9. Testar clustering se disponível
        clustering_test = {}
        if hasattr(stats_tool, '_product_clustering'):
            try:
                # Dados com mais variáveis numéricas para clustering
                cluster_data = test_data[['Total_Liquido', 'Quantidade', 'Preco_Unitario', 'Custo_Produto']].dropna()
                if len(cluster_data) > 10:  # Mínimo para clustering
                    cluster_result = stats_tool._product_clustering(test_data, 'Total_Liquido', 'Grupo_Produto')
                    clustering_test = {
                        'status': 'OK' if isinstance(cluster_result, dict) else 'INVALID_TYPE',
                        'has_clusters': 'optimal_clusters' in cluster_result if isinstance(cluster_result, dict) else False,
                        'cluster_count': cluster_result.get('optimal_clusters', 0) if isinstance(cluster_result, dict) else 0
                    }
                else:
                    clustering_test = {'status': 'INSUFFICIENT_DATA'}
            except Exception as e:
                clustering_test = {
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        # 10. Testar tratamento de erros
        error_handling = {}
        
        # Teste com tipo de análise inválido
        try:
            if test_file_created:
                invalid_analysis = stats_tool._run(
                    analysis_type="analise_inexistente",
                    data=test_csv_path
                )
                error_handling['invalid_analysis_type'] = 'ERROR_HANDLED' if 'não suportado' in invalid_analysis else 'NO_ERROR'
        except Exception as e:
            error_handling['invalid_analysis_type'] = 'EXCEPTION'
        
        # Teste com coluna inexistente
        try:
            if test_file_created:
                invalid_column = stats_tool._run(
                    analysis_type="correlation",
                    data=test_csv_path,
                    target_column="coluna_inexistente"
                )
                error_handling['invalid_column'] = 'ERROR_HANDLED' if 'erro' in invalid_column.lower() else 'NO_ERROR'
        except Exception as e:
            error_handling['invalid_column'] = 'EXCEPTION'
        
        # 11. Verificar cálculos estatísticos básicos
        basic_stats = {}
        
        try:
            # Testar correlação simples
            from scipy.stats import pearsonr
            numeric_cols = test_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                corr, p_value = pearsonr(test_data[numeric_cols[0]], test_data[numeric_cols[1]])
                basic_stats['pearson_correlation'] = {
                    'correlation': round(corr, 3),
                    'p_value': round(p_value, 3),
                    'significant': p_value < 0.05
                }
            
            # Testar detecção de outliers IQR
            if 'Total_Liquido' in test_data.columns:
                Q1 = test_data['Total_Liquido'].quantile(0.25)
                Q3 = test_data['Total_Liquido'].quantile(0.75)
                IQR = Q3 - Q1
                outliers = len(test_data[(test_data['Total_Liquido'] < Q1 - 1.5*IQR) | 
                                       (test_data['Total_Liquido'] > Q3 + 1.5*IQR)])
                basic_stats['iqr_outliers'] = {
                    'count': outliers,
                    'percentage': round(outliers / len(test_data) * 100, 2)
                }
                
        except Exception as e:
            basic_stats['error'] = str(e)
        
        # 12. Limpeza
        if test_file_created:
            try:
                os.remove(test_csv_path)
            except:
                pass
        
        # 13. Compilar resultados
        result['details'] = {
            'tool_info': tool_info,
            'dependencies_ok': dependencies_ok,
            'test_data_stats': {
                'rows': len(test_data),
                'numeric_columns': len(test_data.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(test_data.select_dtypes(include=[object]).columns)
            },
            'analysis_results': analysis_results,
            'auxiliary_methods': auxiliary_methods,
            'clustering_test': clustering_test,
            'error_handling': error_handling,
            'basic_stats': basic_stats
        }
        
        # 14. Determinar sucesso
        successful_analyses = len([r for r in analysis_results.values() if r.get('status') == 'SUCCESS'])
        total_analyses = len(analysis_results)
        
        if successful_analyses > 0 and dependencies_ok:
            result['success'] = True
            if verbose:
                print(f"✅ Statistical Analysis Tool: {successful_analyses}/{total_analyses} análises funcionando")
        else:
            if verbose:
                print("❌ Statistical Analysis Tool com problemas")
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Erro inesperado no teste Statistical: {str(e)}")
        result['success'] = False
        return result

def test_statistical_calculations():
    """Teste específico de cálculos estatísticos"""
    try:
        # Dados de teste simples
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Testes básicos
        tests = {
            'mean': np.mean(data) == 5.5,
            'std': abs(np.std(data) - 2.872) < 0.01,  # Aproximadamente
            'correlation': True  # Placeholder para teste de correlação
        }
        
        # Teste de correlação
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])  # Correlação perfeita
        from scipy.stats import pearsonr
        corr, _ = pearsonr(x, y)
        tests['correlation'] = abs(corr - 1.0) < 0.01
        
        return all(tests.values()), f"Testes: {tests}"
        
    except Exception as e:
        return False, f"Erro: {str(e)}"

if __name__ == "__main__":
    # Teste standalone
    result = test_statistical_analysis_tool(verbose=True, quick=False)
    print("\n📊 RESULTADO DO TESTE STATISTICAL:")
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
    
    # Teste adicional de cálculos
    print("\n📊 TESTE DE CÁLCULOS ESTATÍSTICOS:")
    success, message = test_statistical_calculations()
    print(f"{'✅' if success else '❌'} {message}")
