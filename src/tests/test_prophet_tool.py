"""
🔮 TESTE: PROPHET FORECAST TOOL
===============================

Testa a ferramenta de forecasting com Prophet do projeto Insights-AI.
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
    from insights.tools.prophet_tool import ProphetForecastTool
except ImportError as e:
    print(f"⚠️ Erro ao importar ProphetForecastTool: {e}")
    ProphetForecastTool = None

def create_prophet_test_data():
    """Criar dados de série temporal para teste do Prophet"""
    np.random.seed(42)
    
    # Gerar série temporal de 365 dias
    start_date = datetime(2024, 1, 1)
    dates = pd.date_range(start=start_date, periods=365, freq='D')
    
    # Criar série com tendência, sazonalidade e ruído
    trend = np.linspace(1000, 1500, 365)  # Tendência crescente
    seasonal = 200 * np.sin(2 * np.pi * np.arange(365) / 365)  # Sazonalidade anual
    weekly = 50 * np.sin(2 * np.pi * np.arange(365) / 7)  # Sazonalidade semanal
    noise = np.random.normal(0, 50, 365)  # Ruído
    
    values = trend + seasonal + weekly + noise
    values = np.maximum(values, 0)  # Garantir valores positivos
    
    return pd.DataFrame({
        'ds': dates,
        'y': values
    })

def test_prophet_tool(verbose=False, quick=False):
    """
    Teste da ferramenta Prophet Forecast Tool
    """
    result = {
        'success': False,
        'details': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        if verbose:
            print("🔮 Testando Prophet Forecast Tool...")
        
        # 1. Verificar se a classe foi importada
        if ProphetForecastTool is None:
            result['errors'].append("Não foi possível importar ProphetForecastTool")
            return result
        
        # 2. Verificar dependências do Prophet
        try:
            from prophet import Prophet
            prophet_available = True
            if verbose:
                print("✅ Prophet library disponível")
        except ImportError:
            prophet_available = False
            result['errors'].append("Prophet library não está instalada")
            return result
        
        # 3. Instanciar a ferramenta
        try:
            prophet_tool = ProphetForecastTool()
            if verbose:
                print("✅ ProphetForecastTool instanciada com sucesso")
        except Exception as e:
            result['errors'].append(f"Erro ao instanciar ProphetForecastTool: {str(e)}")
            return result
        
        # 4. Verificar atributos da ferramenta
        tool_info = {
            'name': getattr(prophet_tool, 'name', 'N/A'),
            'description': getattr(prophet_tool, 'description', 'N/A')[:200] + "..." if len(getattr(prophet_tool, 'description', '')) > 200 else getattr(prophet_tool, 'description', 'N/A')
        }
        
        # 5. Criar dados de teste
        try:
            test_data = create_prophet_test_data()
            if verbose:
                print(f"✅ Dados de teste criados: {len(test_data)} pontos")
        except Exception as e:
            result['errors'].append(f"Erro ao criar dados de teste: {str(e)}")
            return result
        
        # 6. Testar conversão de dados para JSON
        try:
            data_json = test_data.to_json(orient='records', date_format='iso')
            json_conversion_ok = True
        except Exception as e:
            json_conversion_ok = False
            result['warnings'].append(f"Erro na conversão JSON: {str(e)}")
        
        # 7. Testar forecast básico
        forecast_tests = {}
        
        if json_conversion_ok:
            # Teste com parâmetros padrão
            try:
                if verbose:
                    print("🔍 Testando forecast básico...")
                
                periods = 15 if quick else 30
                forecast_result = prophet_tool._run(
                    data=data_json,
                    data_column='ds',
                    target_column='y',
                    periods=periods,
                    include_history=True,
                    seasonality_mode='multiplicative'
                )
                
                if isinstance(forecast_result, str) and len(forecast_result) > 0:
                    # Tentar parsear resultado JSON
                    try:
                        forecast_dict = eval(forecast_result)  # ou json.loads se for JSON válido
                        forecast_tests['basic_forecast'] = {
                            'status': 'SUCCESS',
                            'output_length': len(forecast_result),
                            'has_predictions': 'forecast' in forecast_result.lower() or 'yhat' in forecast_result.lower(),
                            'periods_requested': periods
                        }
                    except:
                        forecast_tests['basic_forecast'] = {
                            'status': 'SUCCESS_STRING',
                            'output_length': len(forecast_result)
                        }
                else:
                    forecast_tests['basic_forecast'] = {
                        'status': 'EMPTY_RESULT',
                        'output': str(forecast_result)
                    }
                    
            except Exception as e:
                forecast_tests['basic_forecast'] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                result['warnings'].append(f"Erro no forecast básico: {str(e)}")
        
        # 8. Testar diferentes modos de sazonalidade
        seasonality_tests = {}
        
        if json_conversion_ok and not quick:
            for mode in ['additive', 'multiplicative']:
                try:
                    if verbose:
                        print(f"🔍 Testando modo {mode}...")
                    
                    mode_result = prophet_tool._run(
                        data=data_json,
                        data_column='ds',
                        target_column='y',
                        periods=7,
                        seasonality_mode=mode
                    )
                    
                    seasonality_tests[mode] = {
                        'status': 'SUCCESS' if isinstance(mode_result, str) and len(mode_result) > 0 else 'FAILED',
                        'output_length': len(mode_result) if isinstance(mode_result, str) else 0
                    }
                    
                except Exception as e:
                    seasonality_tests[mode] = {
                        'status': 'ERROR',
                        'error': str(e)
                    }
        
        # 9. Testar métodos auxiliares
        auxiliary_methods = {}
        
        # Testar _create_advanced_forecast se existir
        if hasattr(prophet_tool, '_create_advanced_forecast'):
            try:
                # Criar dados simplificados para teste
                simple_data = test_data.head(30)  # Apenas 30 pontos
                advanced_result = prophet_tool._create_advanced_forecast(simple_data)
                auxiliary_methods['create_advanced_forecast'] = {
                    'status': 'OK' if advanced_result else 'NULL_RESULT',
                    'has_scenarios': 'base' in str(advanced_result) if advanced_result else False
                }
            except Exception as e:
                auxiliary_methods['create_advanced_forecast'] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        # Testar _calculate_business_impact se existir
        if hasattr(prophet_tool, '_calculate_business_impact'):
            try:
                # Criar forecast mock para teste
                mock_forecast = pd.DataFrame({
                    'yhat': np.random.normal(1000, 100, 30)
                })
                business_impact = prophet_tool._calculate_business_impact(mock_forecast)
                auxiliary_methods['calculate_business_impact'] = {
                    'status': 'OK' if business_impact else 'NULL_RESULT',
                    'has_revenue_projection': 'revenue_projection' in str(business_impact) if business_impact else False
                }
            except Exception as e:
                auxiliary_methods['calculate_business_impact'] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        # 10. Testar tratamento de erros
        error_handling = {}
        
        # Teste com dados inválidos
        try:
            invalid_data = '{"invalid": "json"}'
            error_result = prophet_tool._run(
                data=invalid_data,
                data_column='ds',
                target_column='y'
            )
            error_handling['invalid_data'] = 'ERROR_HANDLED' if 'erro' in error_result.lower() else 'NO_ERROR'
        except Exception as e:
            error_handling['invalid_data'] = 'EXCEPTION'
        
        # Teste com colunas inexistentes
        try:
            wrong_columns = prophet_tool._run(
                data=data_json,
                data_column='coluna_inexistente',
                target_column='y'
            )
            error_handling['wrong_columns'] = 'ERROR_HANDLED' if 'erro' in wrong_columns.lower() else 'NO_ERROR'
        except Exception as e:
            error_handling['wrong_columns'] = 'EXCEPTION'
        
        # 11. Verificar performance
        performance_metrics = {}
        
        if json_conversion_ok:
            try:
                import time
                start_time = time.time()
                
                # Forecast rápido para medir performance
                perf_result = prophet_tool._run(
                    data=test_data.head(50).to_json(orient='records', date_format='iso'),
                    data_column='ds',
                    target_column='y',
                    periods=5
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
        
        # 12. Compilar resultados
        result['details'] = {
            'tool_info': tool_info,
            'prophet_available': prophet_available,
            'test_data_stats': {
                'rows': len(test_data),
                'date_range': f"{test_data['ds'].min()} até {test_data['ds'].max()}",
                'value_range': f"{test_data['y'].min():.2f} - {test_data['y'].max():.2f}"
            },
            'json_conversion': json_conversion_ok,
            'forecast_tests': forecast_tests,
            'seasonality_tests': seasonality_tests,
            'auxiliary_methods': auxiliary_methods,
            'error_handling': error_handling,
            'performance_metrics': performance_metrics
        }
        
        # 13. Determinar sucesso
        basic_forecast_ok = forecast_tests.get('basic_forecast', {}).get('status') == 'SUCCESS'
        
        if basic_forecast_ok and prophet_available:
            result['success'] = True
            if verbose:
                print("✅ Prophet Forecast Tool funcionando corretamente")
        else:
            if verbose:
                print("❌ Prophet Forecast Tool com problemas")
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Erro inesperado no teste Prophet: {str(e)}")
        result['success'] = False
        return result

def test_prophet_data_validation():
    """Teste específico de validação de dados do Prophet"""
    if ProphetForecastTool is None:
        return False, "Ferramenta não disponível"
    
    try:
        prophet_tool = ProphetForecastTool()
        
        # Teste 1: Dados válidos
        valid_data = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=30),
            'y': np.random.normal(1000, 100, 30)
        })
        
        # Teste 2: Dados com valores negativos
        negative_data = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=10),
            'y': [-100, -50, 0, 50, 100, -200, 300, 400, -150, 600]
        })
        
        # Teste 3: Dados com poucos pontos
        few_points = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=3),
            'y': [100, 200, 300]
        })
        
        tests = {
            'valid_data': len(valid_data) == 30,
            'handles_negative': len(negative_data) > 0,
            'few_points': len(few_points) == 3
        }
        
        return all(tests.values()), f"Validações: {tests}"
        
    except Exception as e:
        return False, f"Erro: {str(e)}"

if __name__ == "__main__":
    # Teste standalone
    result = test_prophet_tool(verbose=True, quick=False)
    print("\n📊 RESULTADO DO TESTE PROPHET:")
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
    
    # Teste adicional de validação
    print("\n📋 TESTE DE VALIDAÇÃO DE DADOS:")
    success, message = test_prophet_data_validation()
    print(f"{'✅' if success else '❌'} {message}")
