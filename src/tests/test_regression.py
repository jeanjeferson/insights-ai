"""
🔄 TESTE: REGRESSÃO E COMPATIBILIDADE
===================================

Testa regressões entre versões do projeto Insights-AI.
Garante que atualizações não quebrem funcionalidades existentes.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
import tempfile
import hashlib
from datetime import datetime, timedelta
import pickle

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent))

# Importar todas as ferramentas para teste de regressão
try:
    from insights.tools.kpi_calculator_tool import KPICalculatorTool
    from insights.tools.prophet_tool import ProphetForecastTool
    from insights.tools.statistical_analysis_tool import StatisticalAnalysisTool
    from insights.tools.advanced_visualization_tool import AdvancedVisualizationTool
    from insights.tools.sql_query_tool import SQLServerQueryTool
except ImportError as e:
    print(f"⚠️ Erro ao importar ferramentas: {e}")

def create_regression_baseline_data():
    """Criar dados consistentes para testes de regressão"""
    # Usar seed fixo para reprodutibilidade
    np.random.seed(123456)
    
    # Dataset de referência para comparações
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)  # 6 meses
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    for date in date_range:
        # Padrão determinístico para reprodutibilidade
        day_of_year = date.timetuple().tm_yday
        
        # Vendas diárias com padrão previsível
        base_sales = 50 + 20 * np.sin(2 * np.pi * day_of_year / 365)
        daily_transactions = int(base_sales * (1 + 0.1 * np.sin(2 * np.pi * day_of_year / 7)))
        
        for i in range(daily_transactions):
            # IDs determinísticos
            customer_id = f"CLI_{(day_of_year + i) % 100:04d}"
            product_id = f"PROD_{(day_of_year * 3 + i) % 50:04d}"
            
            # Categorias cíclicas
            categories = ['Anéis', 'Brincos', 'Colares', 'Pulseiras', 'Alianças']
            category = categories[(day_of_year + i) % len(categories)]
            
            # Metais cíclicos
            metals = ['Ouro', 'Prata', 'Ouro Branco', 'Ouro Rosé']
            metal = metals[(day_of_year + i * 2) % len(metals)]
            
            # Valores determinísticos baseados em fórmulas
            base_price = 1000 + 500 * np.sin(2 * np.pi * (day_of_year + i) / 100)
            quantity = 1 + ((day_of_year + i) % 3)
            total_liquid = base_price * quantity
            
            data.append({
                'Data': date.strftime('%Y-%m-%d'),
                'Ano': date.year,
                'Mes': date.month,
                'Codigo_Cliente': customer_id,
                'Nome_Cliente': f"Cliente {customer_id.split('_')[1]}",
                'Codigo_Produto': product_id,
                'Descricao_Produto': f"{category} {metal} Premium",
                'Categoria': category,
                'Metal': metal,
                'Quantidade': quantity,
                'Preco_Unitario': round(base_price, 2),
                'Total_Liquido': round(total_liquid, 2),
                'Custo_Produto': round(total_liquid * 0.4, 2),
                'Margem_Bruta': round(total_liquid * 0.6, 2),
                'Vendedor': f"VEND_{(day_of_year + i) % 10:02d}",
                'Canal': ['Loja Física', 'E-commerce', 'WhatsApp'][(day_of_year + i) % 3],
                'Regiao': ['SP', 'RJ', 'MG', 'RS'][(day_of_year + i) % 4]
            })
    
    return pd.DataFrame(data)

def calculate_output_signature(output_text):
    """Calcular assinatura do output para comparação"""
    if not isinstance(output_text, str):
        output_text = str(output_text)
    
    # Remover timestamps e valores que podem variar
    cleaned_output = output_text.lower()
    
    # Remover timestamps comuns
    import re
    cleaned_output = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', '[TIMESTAMP]', cleaned_output)
    cleaned_output = re.sub(r'\d{2}/\d{2}/\d{4}', '[DATE]', cleaned_output)
    cleaned_output = re.sub(r'\d+\.\d+s', '[DURATION]', cleaned_output)
    
    # Gerar hash
    return hashlib.md5(cleaned_output.encode()).hexdigest()

def save_regression_baseline(test_results, baseline_file="regression_baseline.json"):
    """Salvar baseline para comparações futuras"""
    try:
        baseline_data = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',  # Versão base
            'test_results': test_results
        }
        
        baseline_path = Path(__file__).parent / baseline_file
        with open(baseline_path, 'w', encoding='utf-8') as f:
            json.dump(baseline_data, f, indent=2, ensure_ascii=False)
        
        return True, str(baseline_path)
        
    except Exception as e:
        return False, str(e)

def load_regression_baseline(baseline_file="regression_baseline.json"):
    """Carregar baseline para comparação"""
    try:
        baseline_path = Path(__file__).parent / baseline_file
        if not baseline_path.exists():
            return None, "Baseline file não encontrado"
        
        with open(baseline_path, 'r', encoding='utf-8') as f:
            baseline_data = json.load(f)
        
        return baseline_data, None
        
    except Exception as e:
        return None, str(e)

def test_kpi_regression(verbose=False, quick=False):
    """Teste de regressão para KPI Calculator"""
    result = {'success': False, 'details': {}, 'warnings': [], 'errors': []}
    
    try:
        if verbose:
            print("📊 Testando regressão KPI Calculator...")
        
        if 'KPICalculatorTool' not in globals():
            result['errors'].append("KPICalculatorTool não disponível")
            return result
        
        # Dados de referência
        baseline_data = create_regression_baseline_data()
        
        # Salvar temporariamente
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp_file:
            baseline_data.to_csv(tmp_file.name, sep=';', index=False)
            test_csv = tmp_file.name
        
        kpi_regression_tests = {}
        
        # Teste de KPIs específicos com dados determinísticos
        kpi_categories = ['revenue'] if quick else ['revenue', 'operational', 'customer']
        
        for category in kpi_categories:
            try:
                kpi_tool = KPICalculatorTool()
                kpi_result = kpi_tool._run(
                    data_csv=test_csv,
                    categoria=category,
                    periodo="monthly"
                )
                
                if isinstance(kpi_result, str) and len(kip_result) > 0:
                    # Calcular assinatura do resultado
                    result_signature = calculate_output_signature(kpi_result)
                    
                    # Extrair métricas numéricas se possível
                    numeric_values = []
                    import re
                    numbers = re.findall(r'-?\d+\.?\d*', kpi_result)
                    if numbers:
                        numeric_values = [float(x) for x in numbers[:10]]  # Primeiros 10 números
                    
                    kpi_regression_tests[category] = {
                        'status': 'SUCCESS',
                        'output_length': len(kpi_result),
                        'output_signature': result_signature,
                        'numeric_values': numeric_values,
                        'has_insights': 'insight' in kpi_result.lower(),
                        'has_recommendations': 'recomenda' in kpi_result.lower()
                    }
                else:
                    kpi_regression_tests[category] = {
                        'status': 'EMPTY_OUTPUT',
                        'output': str(kpi_result)
                    }
                    
            except Exception as e:
                kpi_regression_tests[category] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        # Limpeza
        try:
            os.unlink(test_csv)
        except:
            pass
        
        result['details'] = {
            'kpi_regression_tests': kpi_regression_tests,
            'baseline_data_rows': len(baseline_data)
        }
        
        # Determinar sucesso
        successful_categories = len([t for t in kpi_regression_tests.values() if t.get('status') == 'SUCCESS'])
        result['success'] = successful_categories > 0
        
        if verbose:
            print(f"📊 KPI Regressão: {successful_categories}/{len(kpi_categories)} categorias funcionando")
        
    except Exception as e:
        result['errors'].append(f"Erro no teste KPI regressão: {str(e)}")
    
    return result

def test_prophet_regression(verbose=False, quick=False):
    """Teste de regressão para Prophet Forecasting"""
    result = {'success': False, 'details': {}, 'warnings': [], 'errors': []}
    
    try:
        if verbose:
            print("🔮 Testando regressão Prophet Forecasting...")
        
        if 'ProphetForecastTool' not in globals():
            result['errors'].append("ProphetForecastTool não disponível")
            return result
        
        # Criar dados de série temporal determinística
        np.random.seed(123456)  # Seed fixo
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        
        # Valores determinísticos
        values = []
        for i, date in enumerate(dates):
            base_value = 1000 + 200 * np.sin(2 * np.pi * i / 30)  # Ciclo de 30 dias
            values.append(base_value)
        
        prophet_data = pd.DataFrame({
            'ds': dates,
            'y': values
        })
        
        prophet_json = prophet_data.to_json(orient='records', date_format='iso')
        
        prophet_regression_tests = {}
        
        # Teste de forecast determinístico
        try:
            prophet_tool = ProphetForecastTool()
            
            periods = 7 if quick else 15
            forecast_result = prophet_tool._run(
                data=prophet_json,
                data_column='ds',
                target_column='y',
                periods=periods,
                seasonality_mode='additive'
            )
            
            if isinstance(forecast_result, str) and len(forecast_result) > 0:
                result_signature = calculate_output_signature(forecast_result)
                
                # Tentar extrair valores de forecast
                forecast_values = []
                import re
                numbers = re.findall(r'-?\d+\.?\d*', forecast_result)
                if numbers:
                    forecast_values = [float(x) for x in numbers[:5]]  # Primeiros 5 valores
                
                prophet_regression_tests['basic_forecast'] = {
                    'status': 'SUCCESS',
                    'output_length': len(forecast_result),
                    'output_signature': result_signature,
                    'forecast_values_sample': forecast_values,
                    'periods_requested': periods,
                    'has_confidence_intervals': 'confidence' in forecast_result.lower() or 'interval' in forecast_result.lower()
                }
            else:
                prophet_regression_tests['basic_forecast'] = {
                    'status': 'EMPTY_OUTPUT',
                    'output': str(forecast_result)
                }
                
        except Exception as e:
            prophet_regression_tests['basic_forecast'] = {
                'status': 'ERROR',
                'error': str(e)
            }
        
        result['details'] = {
            'prophet_regression_tests': prophet_regression_tests,
            'input_data_points': len(prophet_data)
        }
        
        # Determinar sucesso
        result['success'] = prophet_regression_tests.get('basic_forecast', {}).get('status') == 'SUCCESS'
        
        if verbose:
            status = prophet_regression_tests.get('basic_forecast', {}).get('status', 'UNKNOWN')
            print(f"🔮 Prophet Regressão: {status}")
        
    except Exception as e:
        result['errors'].append(f"Erro no teste Prophet regressão: {str(e)}")
    
    return result

def test_output_consistency(verbose=False, quick=False):
    """Teste de consistência de outputs entre execuções"""
    result = {'success': False, 'details': {}, 'warnings': [], 'errors': []}
    
    try:
        if verbose:
            print("🔄 Testando consistência de outputs...")
        
        baseline_data = create_regression_baseline_data()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp_file:
            baseline_data.to_csv(tmp_file.name, sep=';', index=False)
            test_csv = tmp_file.name
        
        consistency_tests = {}
        
        # Executar a mesma operação múltiplas vezes
        iterations = 3 if quick else 5
        
        if 'KPICalculatorTool' in globals():
            kpi_outputs = []
            
            for i in range(iterations):
                try:
                    kpi_tool = KPICalculatorTool()
                    kpi_result = kpi_tool._run(
                        data_csv=test_csv,
                        categoria="revenue",
                        periodo="monthly"
                    )
                    
                    if isinstance(kpi_result, str):
                        # Calcular assinatura normalizada
                        signature = calculate_output_signature(kpi_result)
                        kpi_outputs.append({
                            'iteration': i + 1,
                            'signature': signature,
                            'length': len(kpi_result)
                        })
                        
                except Exception as e:
                    kpi_outputs.append({
                        'iteration': i + 1,
                        'error': str(e)
                    })
            
            # Verificar consistência
            signatures = [o.get('signature') for o in kpi_outputs if 'signature' in o]
            if signatures:
                unique_signatures = set(signatures)
                consistency_tests['kpi_calculator'] = {
                    'total_iterations': iterations,
                    'successful_iterations': len(signatures),
                    'unique_signatures': len(unique_signatures),
                    'is_consistent': len(unique_signatures) == 1,
                    'outputs': kpi_outputs
                }
            else:
                consistency_tests['kpi_calculator'] = {
                    'total_iterations': iterations,
                    'successful_iterations': 0,
                    'error': 'Nenhuma execução bem-sucedida'
                }
        
        # Limpeza
        try:
            os.unlink(test_csv)
        except:
            pass
        
        result['details'] = {
            'consistency_tests': consistency_tests,
            'iterations_per_tool': iterations
        }
        
        # Determinar sucesso (deve haver consistência entre execuções)
        consistent_tools = 0
        total_tools = 0
        
        for tool_test in consistency_tests.values():
            if 'is_consistent' in tool_test:
                total_tools += 1
                if tool_test['is_consistent']:
                    consistent_tools += 1
        
        result['success'] = consistent_tools > 0 and (consistent_tools / max(total_tools, 1)) >= 0.8
        
        if verbose:
            print(f"🔄 Consistência: {consistent_tools}/{total_tools} ferramentas consistentes")
        
    except Exception as e:
        result['errors'].append(f"Erro no teste de consistência: {str(e)}")
    
    return result

def test_backward_compatibility(verbose=False, quick=False):
    """Teste de compatibilidade com versões anteriores"""
    result = {'success': False, 'details': {}, 'warnings': [], 'errors': []}
    
    try:
        if verbose:
            print("⏪ Testando compatibilidade com versões anteriores...")
        
        compatibility_tests = {}
        
        # Carregar baseline anterior se existir
        baseline_data, baseline_error = load_regression_baseline()
        
        if baseline_data:
            # Comparar com baseline anterior
            if verbose:
                print(f"📋 Baseline encontrado: versão {baseline_data.get('version', 'unknown')}")
            
            compatibility_tests['baseline_comparison'] = {
                'baseline_found': True,
                'baseline_version': baseline_data.get('version', 'unknown'),
                'baseline_timestamp': baseline_data.get('timestamp', 'unknown')
            }
            
            # Executar mesmos testes e comparar resultados
            current_results = {}
            
            # Teste KPI com dados baseline
            baseline_kpi_test = test_kpi_regression(verbose=False, quick=True)
            current_results['kpi'] = baseline_kpi_test
            
            # Comparar assinaturas se disponível
            baseline_kpi_data = baseline_data.get('test_results', {}).get('kpi', {})
            current_kpi_data = baseline_kpi_test.get('details', {}).get('kpi_regression_tests', {})
            
            signature_comparison = {}
            for category in current_kpi_data.keys():
                if category in baseline_kpi_data:
                    current_sig = current_kpi_data[category].get('output_signature')
                    baseline_sig = baseline_kpi_data.get(category, {}).get('output_signature')
                    
                    signature_comparison[category] = {
                        'signatures_match': current_sig == baseline_sig,
                        'current_signature': current_sig,
                        'baseline_signature': baseline_sig
                    }
            
            compatibility_tests['signature_comparison'] = signature_comparison
            
        else:
            # Criar novo baseline
            if verbose:
                print("📋 Criando novo baseline...")
            
            baseline_results = {
                'kpi': test_kpi_regression(verbose=False, quick=quick),
                'prophet': test_prophet_regression(verbose=False, quick=quick)
            }
            
            success, baseline_path = save_regression_baseline(baseline_results)
            
            compatibility_tests['baseline_creation'] = {
                'baseline_found': False,
                'new_baseline_created': success,
                'baseline_path': baseline_path if success else None,
                'error': baseline_path if not success else None
            }
        
        result['details'] = {
            'compatibility_tests': compatibility_tests
        }
        
        # Determinar sucesso
        if baseline_data:
            # Se há baseline, verificar compatibilidade
            signatures_match = 0
            total_signatures = 0
            
            for comparison in compatibility_tests.get('signature_comparison', {}).values():
                total_signatures += 1
                if comparison.get('signatures_match', False):
                    signatures_match += 1
            
            # Permitir alguma divergência (80% de compatibilidade)
            if total_signatures > 0:
                compatibility_rate = signatures_match / total_signatures
                result['success'] = compatibility_rate >= 0.8
            else:
                result['success'] = True  # Se não há assinaturas para comparar
        else:
            # Se não há baseline, sucesso ao criar um novo
            result['success'] = compatibility_tests.get('baseline_creation', {}).get('new_baseline_created', False)
        
        if verbose:
            if baseline_data:
                sig_comparison = compatibility_tests.get('signature_comparison', {})
                matches = len([c for c in sig_comparison.values() if c.get('signatures_match', False)])
                total = len(sig_comparison)
                print(f"⏪ Compatibilidade: {matches}/{total} assinaturas compatíveis")
            else:
                print("⏪ Novo baseline criado para futuras comparações")
        
    except Exception as e:
        result['errors'].append(f"Erro no teste de compatibilidade: {str(e)}")
    
    return result

def test_regression(verbose=False, quick=False):
    """
    Teste consolidado de regressão
    """
    result = {
        'success': False,
        'details': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        if verbose:
            print("🔄 Iniciando testes de regressão...")
        
        # Executar todos os testes de regressão
        regression_tests = {}
        
        # 1. KPI Regression
        kpi_regression_result = test_kpi_regression(verbose=verbose, quick=quick)
        regression_tests['kpi_regression'] = kpi_regression_result
        
        # 2. Prophet Regression
        prophet_regression_result = test_prophet_regression(verbose=verbose, quick=quick)
        regression_tests['prophet_regression'] = prophet_regression_result
        
        # 3. Output Consistency
        consistency_result = test_output_consistency(verbose=verbose, quick=quick)
        regression_tests['output_consistency'] = consistency_result
        
        # 4. Backward Compatibility
        compatibility_result = test_backward_compatibility(verbose=verbose, quick=quick)
        regression_tests['backward_compatibility'] = compatibility_result
        
        # Estatísticas consolidadas
        total_tests = len(regression_tests)
        successful_tests = len([t for t in regression_tests.values() if t.get('success', False)])
        total_warnings = sum(len(t.get('warnings', [])) for t in regression_tests.values())
        total_errors = sum(len(t.get('errors', [])) for t in regression_tests.values())
        
        result['details'] = {
            'total_regression_tests': total_tests,
            'successful_tests': successful_tests,
            'total_warnings': total_warnings,
            'total_errors': total_errors,
            'regression_score': round(successful_tests / total_tests * 100, 1) if total_tests > 0 else 0,
            'individual_results': regression_tests
        }
        
        # Consolidar warnings e errors
        for test_result in regression_tests.values():
            result['warnings'].extend(test_result.get('warnings', []))
            result['errors'].extend(test_result.get('errors', []))
        
        # Determinar sucesso geral (pelo menos 75% dos testes devem passar)
        result['success'] = (successful_tests / total_tests) >= 0.75 if total_tests > 0 else True
        
        if verbose:
            print(f"🔄 Regressão: {successful_tests}/{total_tests} testes passaram")
            print(f"📊 Score de regressão: {result['details']['regression_score']}%")
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Erro inesperado nos testes de regressão: {str(e)}")
        result['success'] = False
        return result

if __name__ == "__main__":
    # Teste standalone
    result = test_regression(verbose=True, quick=False)
    print("\n📊 RESULTADO DOS TESTES DE REGRESSÃO:")
    print(f"🔄 Sucesso: {result['success']}")
    print(f"📈 Score de Regressão: {result['details'].get('regression_score', 0)}%")
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
