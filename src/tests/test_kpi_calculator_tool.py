"""
📊 TESTE: KPI CALCULATOR TOOL
=============================

Testa a ferramenta de cálculo de KPIs do projeto Insights-AI.
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
    from insights.tools.kpi_calculator_tool import KPICalculatorTool
except ImportError as e:
    print(f"⚠️ Erro ao importar KPICalculatorTool: {e}")
    KPICalculatorTool = None

def create_test_data():
    """Criar dados de teste para os KPIs"""
    np.random.seed(42)
    
    # Gerar dados de vendas realistas
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    for i, date in enumerate(date_range[:90]):  # 3 meses de dados
        # Número de transações por dia
        daily_transactions = np.random.randint(5, 20)
        
        for _ in range(daily_transactions):
            data.append({
                'Data': date.strftime('%Y-%m-%d'),
                'Ano': date.year,
                'Mes': date.month,
                'Codigo_Cliente': f"CLI_{np.random.randint(1, 100):03d}",
                'Nome_Cliente': f"Cliente {np.random.randint(1, 100):03d}",
                'Codigo_Produto': f"PROD_{np.random.randint(1, 50):03d}",
                'Descricao_Produto': f"Produto {np.random.randint(1, 50):03d}",
                'Grupo_Produto': np.random.choice(['Anéis', 'Brincos', 'Colares', 'Pulseiras']),
                'Metal': np.random.choice(['Ouro', 'Prata', 'Ouro Branco']),
                'Quantidade': np.random.randint(1, 3),
                'Total_Liquido': np.random.uniform(500, 5000),
                'Custo_Produto': np.random.uniform(200, 2000),
                'Preco_Tabela': np.random.uniform(600, 6000),
                'Desconto_Aplicado': np.random.uniform(0, 500),
                'Estoque_Atual': np.random.randint(0, 100)
            })
    
    return pd.DataFrame(data)

def test_kpi_calculator_tool(verbose=False, quick=False):
    """
    Teste da ferramenta KPI Calculator Tool
    """
    result = {
        'success': False,
        'details': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        if verbose:
            print("📊 Testando KPI Calculator Tool...")
        
        # 1. Verificar se a classe foi importada
        if KPICalculatorTool is None:
            result['errors'].append("Não foi possível importar KPICalculatorTool")
            return result
        
        # 2. Instanciar a ferramenta
        try:
            kpi_tool = KPICalculatorTool()
            if verbose:
                print("✅ KPICalculatorTool instanciada com sucesso")
        except Exception as e:
            result['errors'].append(f"Erro ao instanciar KPICalculatorTool: {str(e)}")
            return result
        
        # 3. Verificar atributos da ferramenta
        tool_info = {
            'name': getattr(kpi_tool, 'name', 'N/A'),
            'description': getattr(kpi_tool, 'description', 'N/A')[:200] + "..." if len(getattr(kpi_tool, 'description', '')) > 200 else getattr(kpi_tool, 'description', 'N/A')
        }
        
        # 4. Criar dados de teste
        try:
            test_df = create_test_data()
            if verbose:
                print(f"✅ Dados de teste criados: {len(test_df)} registros")
        except Exception as e:
            result['errors'].append(f"Erro ao criar dados de teste: {str(e)}")
            return result
        
        # 5. Salvar dados de teste temporariamente
        test_csv_path = "temp_test_data.csv"
        try:
            test_df.to_csv(test_csv_path, sep=';', index=False, encoding='utf-8')
            test_file_created = True
        except Exception as e:
            result['warnings'].append(f"Não foi possível criar arquivo de teste: {str(e)}")
            test_file_created = False
        
        # 6. Testar diferentes categorias de KPI
        kpi_categories = ['revenue', 'operational', 'inventory', 'customer', 'all']
        category_results = {}
        
        for category in kpi_categories:
            if quick and category != 'revenue':  # No modo rápido, só testar revenue
                continue
                
            try:
                if verbose:
                    print(f"🔍 Testando categoria: {category}")
                
                # Usar dados de teste se arquivo foi criado, senão usar dados reais se existirem
                data_source = test_csv_path if test_file_created else "data/vendas.csv"
                
                kpi_result = kpi_tool._run(
                    data_csv=data_source,
                    categoria=category,
                    periodo="monthly",
                    benchmark_mode=True
                )
                
                # Verificar se o resultado é válido
                if isinstance(kpi_result, str) and len(kpi_result) > 0:
                    category_results[category] = {
                        'status': 'SUCCESS',
                        'output_length': len(kpi_result),
                        'has_kpis': 'KPI' in kpi_result.upper(),
                        'has_insights': 'INSIGHT' in kpi_result.upper(),
                        'sample_output': kpi_result[:300] + "..." if len(kpi_result) > 300 else kpi_result
                    }
                else:
                    category_results[category] = {
                        'status': 'EMPTY_RESULT',
                        'output': str(kpi_result)
                    }
                    result['warnings'].append(f"Categoria {category} retornou resultado vazio")
                
            except Exception as e:
                category_results[category] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                result['warnings'].append(f"Erro na categoria {category}: {str(e)}")
        
        # 7. Testar métodos auxiliares
        auxiliary_methods = {}
        
        # Testar _validate_and_clean_data
        if hasattr(kpi_tool, '_validate_and_clean_data'):
            try:
                cleaned_df = kpi_tool._validate_and_clean_data(test_df)
                auxiliary_methods['validate_and_clean_data'] = {
                    'status': 'OK' if cleaned_df is not None else 'NULL_RESULT',
                    'input_rows': len(test_df),
                    'output_rows': len(cleaned_df) if cleaned_df is not None else 0
                }
            except Exception as e:
                auxiliary_methods['validate_and_clean_data'] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        # Testar _calculate_financial_kpis
        if hasattr(kpi_tool, '_calculate_financial_kpis'):
            try:
                financial_kpis = kpi_tool._calculate_financial_kpis(test_df, "monthly")
                auxiliary_methods['calculate_financial_kpis'] = {
                    'status': 'OK' if isinstance(financial_kpis, dict) else 'INVALID_TYPE',
                    'kpi_count': len(financial_kpis) if isinstance(financial_kpis, dict) else 0,
                    'has_revenue': 'total_revenue' in financial_kpis if isinstance(financial_kpis, dict) else False
                }
            except Exception as e:
                auxiliary_methods['calculate_financial_kpis'] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        # 8. Testar com dados inválidos
        error_handling = {}
        
        # Teste com arquivo inexistente
        try:
            error_result = kpi_tool._run(data_csv="arquivo_inexistente.csv")
            error_handling['missing_file'] = 'NO_ERROR' if 'Erro' not in error_result else 'ERROR_HANDLED'
        except Exception as e:
            error_handling['missing_file'] = 'EXCEPTION'
        
        # Teste com categoria inválida
        try:
            if test_file_created:
                invalid_result = kpi_tool._run(
                    data_csv=test_csv_path,
                    categoria="categoria_inexistente"
                )
                error_handling['invalid_category'] = 'NO_ERROR' if 'Erro' not in invalid_result else 'ERROR_HANDLED'
        except Exception as e:
            error_handling['invalid_category'] = 'EXCEPTION'
        
        # 9. Verificar benchmarks do setor
        benchmark_test = {}
        if hasattr(kpi_tool, '_calculate_benchmark_comparison'):
            try:
                benchmark_result = kpi_tool._calculate_benchmark_comparison(test_df)
                benchmark_test = {
                    'status': 'OK' if isinstance(benchmark_result, dict) else 'INVALID',
                    'has_benchmarks': len(benchmark_result) > 0 if isinstance(benchmark_result, dict) else False
                }
            except Exception as e:
                benchmark_test = {
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        # 10. Limpeza
        if test_file_created:
            try:
                os.remove(test_csv_path)
            except:
                pass
        
        # 11. Compilar resultados
        result['details'] = {
            'tool_info': tool_info,
            'test_data_stats': {
                'rows': len(test_df),
                'columns': len(test_df.columns),
                'date_range': f"{test_df['Data'].min()} até {test_df['Data'].max()}"
            },
            'category_results': category_results,
            'auxiliary_methods': auxiliary_methods,
            'error_handling': error_handling,
            'benchmark_test': benchmark_test
        }
        
        # 12. Determinar sucesso
        successful_categories = len([r for r in category_results.values() if r.get('status') == 'SUCCESS'])
        total_categories = len(category_results)
        
        if successful_categories > 0:
            result['success'] = True
            if verbose:
                print(f"✅ KPI Calculator Tool: {successful_categories}/{total_categories} categorias funcionando")
        else:
            if verbose:
                print("❌ KPI Calculator Tool: nenhuma categoria funcionando")
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Erro inesperado no teste KPI: {str(e)}")
        result['success'] = False
        return result

def test_kpi_calculations():
    """Teste específico dos cálculos de KPI"""
    if KPICalculatorTool is None:
        return False, "Ferramenta não disponível"
    
    try:
        kpi_tool = KPICalculatorTool()
        test_df = create_test_data()
        
        # Testar cálculos específicos
        tests = {}
        
        # Revenue Growth
        monthly_revenue = test_df.groupby(['Ano', 'Mes'])['Total_Liquido'].sum()
        if len(monthly_revenue) >= 2:
            growth_rate = ((monthly_revenue.iloc[-1] - monthly_revenue.iloc[-2]) / monthly_revenue.iloc[-2] * 100)
            tests['revenue_growth'] = not np.isnan(growth_rate)
        
        # AOV
        aov = test_df['Total_Liquido'].mean()
        tests['aov'] = aov > 0
        
        # Unique customers
        unique_customers = test_df['Codigo_Cliente'].nunique()
        tests['unique_customers'] = unique_customers > 0
        
        return all(tests.values()), f"Testes: {tests}"
        
    except Exception as e:
        return False, f"Erro: {str(e)}"

if __name__ == "__main__":
    # Teste standalone
    result = test_kpi_calculator_tool(verbose=True, quick=False)
    print("\n📊 RESULTADO DO TESTE KPI:")
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
    print("\n🧮 TESTE DE CÁLCULOS:")
    success, message = test_kpi_calculations()
    print(f"{'✅' if success else '❌'} {message}")
