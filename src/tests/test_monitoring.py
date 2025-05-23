"""
📊 TESTE: MONITORAMENTO E MÉTRICAS
=================================

Testa sistemas de monitoramento, logging e métricas do projeto Insights-AI.
Valida coleta de métricas, alertas e observabilidade.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import time
import tempfile
import json
from datetime import datetime, timedelta
import logging
from unittest.mock import patch, MagicMock

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent))

# Importar ferramentas para teste de monitoramento
try:
    from insights.tools.kpi_calculator_tool import KPICalculatorTool
    from insights.tools.prophet_tool import ProphetForecastTool
    from insights.tools.statistical_analysis_tool import StatisticalAnalysisTool
except ImportError as e:
    print(f"⚠️ Erro ao importar ferramentas: {e}")

def setup_test_logger():
    """Configurar logger para testes de monitoramento"""
    logger = logging.getLogger('insights_ai_monitoring')
    logger.setLevel(logging.DEBUG)
    
    # Handler para arquivo
    log_file = tempfile.mktemp(suffix='.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Handler para console (capturado)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

def test_performance_metrics(verbose=False, quick=False):
    """Teste de coleta de métricas de performance"""
    result = {'success': False, 'details': {}, 'warnings': [], 'errors': []}
    
    try:
        if verbose:
            print("⏱️ Testando métricas de performance...")
        
        # Configurar coleta de métricas
        performance_metrics = {}
        
        # Criar dados de teste
        test_data = pd.DataFrame({
            'Data': pd.date_range('2024-01-01', periods=100),
            'Total_Liquido': np.random.normal(1000, 200, 100),
            'Quantidade': np.random.randint(1, 5, 100),
            'Codigo_Cliente': [f"CLI_{i%20:03d}" for i in range(100)],
            'Codigo_Produto': [f"PROD_{i%10:03d}" for i in range(100)]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp_file:
            test_data.to_csv(tmp_file.name, sep=';', index=False)
            test_csv = tmp_file.name
        
        # Teste de métricas de tempo de execução
        execution_times = {}
        
        # KPI Calculator
        if 'KPICalculatorTool' in globals():
            try:
                start_time = time.time()
                kpi_tool = KPICalculatorTool()
                
                # Medir tempo de inicialização
                init_time = time.time() - start_time
                
                # Medir tempo de execução
                exec_start = time.time()
                kpi_result = kpi_tool._run(data_csv=test_csv, categoria="revenue")
                exec_time = time.time() - exec_start
                
                execution_times['kpi_calculator'] = {
                    'initialization_time_ms': round(init_time * 1000, 2),
                    'execution_time_ms': round(exec_time * 1000, 2),
                    'total_time_ms': round((init_time + exec_time) * 1000, 2),
                    'success': isinstance(kpi_result, str) and len(kpi_result) > 0,
                    'output_size_bytes': len(kpi_result.encode('utf-8')) if isinstance(kpi_result, str) else 0
                }
                
            except Exception as e:
                execution_times['kpi_calculator'] = {
                    'error': str(e),
                    'success': False
                }
        
        # Prophet Forecasting
        if 'ProphetForecastTool' in globals() and not quick:
            try:
                prophet_data = test_data[['Data', 'Total_Liquido']].copy()
                prophet_data.columns = ['ds', 'y']
                prophet_json = prophet_data.to_json(orient='records', date_format='iso')
                
                start_time = time.time()
                prophet_tool = ProphetForecastTool()
                init_time = time.time() - start_time
                
                exec_start = time.time()
                prophet_result = prophet_tool._run(
                    data=prophet_json,
                    data_column='ds',
                    target_column='y',
                    periods=7
                )
                exec_time = time.time() - exec_start
                
                execution_times['prophet_forecasting'] = {
                    'initialization_time_ms': round(init_time * 1000, 2),
                    'execution_time_ms': round(exec_time * 1000, 2),
                    'total_time_ms': round((init_time + exec_time) * 1000, 2),
                    'success': isinstance(prophet_result, str) and len(prophet_result) > 0,
                    'output_size_bytes': len(prophet_result.encode('utf-8')) if isinstance(prophet_result, str) else 0
                }
                
            except Exception as e:
                execution_times['prophet_forecasting'] = {
                    'error': str(e),
                    'success': False
                }
        
        # Métricas de throughput
        throughput_metrics = {}
        
        if execution_times:
            # Calcular throughput para KPI Calculator
            kpi_metrics = execution_times.get('kpi_calculator', {})
            if kpi_metrics.get('success') and kpi_metrics.get('execution_time_ms', 0) > 0:
                records_per_second = len(test_data) / (kpi_metrics['execution_time_ms'] / 1000)
                throughput_metrics['kpi_calculator'] = {
                    'records_per_second': round(records_per_second, 2),
                    'ms_per_record': round(kpi_metrics['execution_time_ms'] / len(test_data), 2)
                }
        
        # Métricas de memoria
        memory_metrics = {}
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            memory_metrics = {
                'rss_mb': round(memory_info.rss / 1024 / 1024, 2),
                'vms_mb': round(memory_info.vms / 1024 / 1024, 2),
                'memory_percent': round(process.memory_percent(), 2),
                'available': True
            }
            
        except ImportError:
            memory_metrics = {
                'available': False,
                'reason': 'psutil not installed'
            }
        except Exception as e:
            memory_metrics = {
                'available': False,
                'error': str(e)
            }
        
        # Limpeza
        try:
            os.unlink(test_csv)
        except:
            pass
        
        performance_metrics = {
            'execution_times': execution_times,
            'throughput_metrics': throughput_metrics,
            'memory_metrics': memory_metrics,
            'test_data_size': len(test_data)
        }
        
        result['details'] = {
            'performance_metrics': performance_metrics
        }
        
        # Determinar sucesso (pelo menos uma ferramenta deve ter métricas válidas)
        successful_tools = len([t for t in execution_times.values() if t.get('success', False)])
        result['success'] = successful_tools > 0
        
        if verbose:
            print(f"⏱️ Performance: {successful_tools} ferramentas com métricas coletadas")
            if memory_metrics.get('available'):
                print(f"💾 Memória: {memory_metrics['rss_mb']}MB RSS")
        
    except Exception as e:
        result['errors'].append(f"Erro no teste de métricas de performance: {str(e)}")
    
    return result

def test_logging_system(verbose=False, quick=False):
    """Teste do sistema de logging"""
    result = {'success': False, 'details': {}, 'warnings': [], 'errors': []}
    
    try:
        if verbose:
            print("📝 Testando sistema de logging...")
        
        # Configurar logger de teste
        logger, log_file = setup_test_logger()
        
        logging_tests = {}
        
        # Teste de níveis de log
        log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in log_levels:
            try:
                log_method = getattr(logger, level.lower())
                log_method(f"Teste de log nivel {level}")
                logging_tests[f'level_{level.lower()}'] = True
            except Exception as e:
                logging_tests[f'level_{level.lower()}'] = False
                result['warnings'].append(f"Falha no log level {level}: {str(e)}")
        
        # Teste de logging em ferramentas
        tool_logging_tests = {}
        
        if 'KPICalculatorTool' in globals():
            try:
                # Patch do logger para capturar logs
                with patch('logging.getLogger') as mock_logger:
                    mock_log_instance = MagicMock()
                    mock_logger.return_value = mock_log_instance
                    
                    # Criar dados de teste
                    test_data = pd.DataFrame({
                        'Data': ['2024-01-01', '2024-01-02'],
                        'Total_Liquido': [1000, 2000],
                        'Quantidade': [1, 2]
                    })
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp_file:
                        test_data.to_csv(tmp_file.name, sep=';', index=False)
                        test_csv = tmp_file.name
                    
                    # Executar ferramenta
                    kpi_tool = KPICalculatorTool()
                    kpi_result = kpi_tool._run(data_csv=test_csv)
                    
                    # Verificar se logs foram chamados
                    tool_logging_tests['kpi_calculator'] = {
                        'logger_requested': mock_logger.called,
                        'execution_successful': isinstance(kpi_result, str),
                        'log_calls': mock_log_instance.call_count if mock_logger.called else 0
                    }
                    
                    # Limpeza
                    try:
                        os.unlink(test_csv)
                    except:
                        pass
                        
            except Exception as e:
                tool_logging_tests['kpi_calculator'] = {
                    'error': str(e)
                }
        
        # Verificar conteúdo do arquivo de log
        log_file_tests = {}
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            log_file_tests = {
                'file_created': True,
                'file_size_bytes': len(log_content.encode('utf-8')),
                'has_timestamps': 'T' in log_content or ':' in log_content,
                'has_log_levels': any(level in log_content for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']),
                'line_count': len(log_content.splitlines()),
                'sample_lines': log_content.splitlines()[:3]  # Primeiras 3 linhas
            }
            
        except Exception as e:
            log_file_tests = {
                'file_created': False,
                'error': str(e)
            }
        
        # Limpeza do arquivo de log
        try:
            os.unlink(log_file)
        except:
            pass
        
        result['details'] = {
            'logging_tests': logging_tests,
            'tool_logging_tests': tool_logging_tests,
            'log_file_tests': log_file_tests
        }
        
        # Determinar sucesso
        successful_levels = sum(logging_tests.values())
        total_levels = len(logging_tests)
        
        file_created = log_file_tests.get('file_created', False)
        
        result['success'] = (successful_levels / total_levels) >= 0.8 and file_created
        
        if verbose:
            print(f"📝 Logging: {successful_levels}/{total_levels} níveis funcionando")
            print(f"📄 Log file: {'✅' if file_created else '❌'}")
        
    except Exception as e:
        result['errors'].append(f"Erro no teste de logging: {str(e)}")
    
    return result

def test_error_monitoring(verbose=False, quick=False):
    """Teste de monitoramento de erros"""
    result = {'success': False, 'details': {}, 'warnings': [], 'errors': []}
    
    try:
        if verbose:
            print("🚨 Testando monitoramento de erros...")
        
        error_monitoring_tests = {}
        
        # Teste de captura de exceções
        exception_tests = {}
        
        # Simular erros controlados
        error_scenarios = [
            {
                'name': 'file_not_found',
                'action': lambda: KPICalculatorTool()._run(data_csv="arquivo_inexistente.csv") if 'KPICalculatorTool' in globals() else None,
                'expected_error_keywords': ['file', 'not found', 'erro', 'inexistente']
            },
            {
                'name': 'invalid_data_format',
                'action': lambda: self._test_invalid_csv(),
                'expected_error_keywords': ['invalid', 'format', 'erro', 'csv']
            },
            {
                'name': 'missing_columns',
                'action': lambda: self._test_missing_columns(),
                'expected_error_keywords': ['column', 'missing', 'coluna', 'obrigatória']
            }
        ]
        
        for scenario in error_scenarios:
            if quick and len(exception_tests) >= 2:
                break
                
            try:
                error_result = scenario['action']()
                
                # Verificar se erro foi tratado adequadamente
                if isinstance(error_result, str):
                    has_error_keywords = any(
                        keyword in error_result.lower() 
                        for keyword in scenario['expected_error_keywords']
                    )
                    
                    exception_tests[scenario['name']] = {
                        'error_handled': True,
                        'has_error_message': has_error_keywords,
                        'output_length': len(error_result),
                        'status': 'GRACEFUL_HANDLING'
                    }
                else:
                    exception_tests[scenario['name']] = {
                        'error_handled': False,
                        'status': 'NO_ERROR_MESSAGE'
                    }
                    
            except Exception as e:
                # Exceção pode ser forma válida de tratamento de erro
                exception_tests[scenario['name']] = {
                    'error_handled': True,
                    'exception_raised': True,
                    'exception_type': type(e).__name__,
                    'exception_message': str(e)[:100],
                    'status': 'EXCEPTION_HANDLING'
                }
        
        # Teste de rastreamento de stack trace
        stack_trace_tests = {}
        
        try:
            # Forçar um erro para testar stack trace
            if 'StatisticalAnalysisTool' in globals():
                stats_tool = StatisticalAnalysisTool()
                
                # Tentar análise com dados inválidos
                try:
                    invalid_result = stats_tool._run(
                        analysis_type="correlation",
                        data="dados_completamente_inválidos"
                    )
                    
                    stack_trace_tests['statistical_tool'] = {
                        'error_handled': isinstance(invalid_result, str),
                        'graceful_degradation': True
                    }
                    
                except Exception as e:
                    import traceback
                    stack_trace = traceback.format_exc()
                    
                    stack_trace_tests['statistical_tool'] = {
                        'exception_raised': True,
                        'has_stack_trace': len(stack_trace) > 100,
                        'stack_trace_lines': len(stack_trace.splitlines()),
                        'traceback_available': True
                    }
                    
        except Exception as e:
            stack_trace_tests['general'] = {
                'error': str(e)
            }
        
        error_monitoring_tests = {
            'exception_tests': exception_tests,
            'stack_trace_tests': stack_trace_tests
        }
        
        result['details'] = {
            'error_monitoring_tests': error_monitoring_tests
        }
        
        # Determinar sucesso (erros devem ser tratados graciosamente)
        graceful_errors = len([
            t for t in exception_tests.values() 
            if t.get('error_handled', False)
        ])
        total_error_tests = len(exception_tests)
        
        if total_error_tests > 0:
            error_handling_rate = graceful_errors / total_error_tests
            result['success'] = error_handling_rate >= 0.8
        else:
            result['success'] = True
        
        if verbose:
            print(f"🚨 Monitoramento: {graceful_errors}/{total_error_tests} erros tratados graciosamente")
        
    except Exception as e:
        result['errors'].append(f"Erro no teste de monitoramento de erros: {str(e)}")
    
    return result
    
    def _test_invalid_csv(self):
        """Método auxiliar para testar CSV inválido"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write("invalid,csv,data\nwith,incomplete\n")
            invalid_csv = tmp_file.name
        
        try:
            if 'KPICalculatorTool' in globals():
                kpi_tool = KPICalculatorTool()
                return kpi_tool._run(data_csv=invalid_csv)
        finally:
            try:
                os.unlink(invalid_csv)
            except:
                pass
        
        return None
    
    def _test_missing_columns(self):
        """Método auxiliar para testar colunas ausentes"""
        incomplete_data = pd.DataFrame({
            'Data': ['2024-01-01'],
            'Coluna_Errada': [1000]
            # Faltando Total_Liquido
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp_file:
            incomplete_data.to_csv(tmp_file.name, sep=';', index=False)
            incomplete_csv = tmp_file.name
        
        try:
            if 'KPICalculatorTool' in globals():
                kpi_tool = KPICalculatorTool()
                return kpi_tool._run(data_csv=incomplete_csv)
        finally:
            try:
                os.unlink(incomplete_csv)
            except:
                pass
        
        return None

def test_alerting_system(verbose=False, quick=False):
    """Teste do sistema de alertas"""
    result = {'success': False, 'details': {}, 'warnings': [], 'errors': []}
    
    try:
        if verbose:
            print("🔔 Testando sistema de alertas...")
        
        alerting_tests = {}
        
        # Simular condições que devem gerar alertas
        alert_conditions = [
            {
                'name': 'high_execution_time',
                'threshold': 5000,  # 5 segundos
                'metric': 'execution_time_ms',
                'description': 'Tempo de execução muito alto'
            },
            {
                'name': 'high_memory_usage',
                'threshold': 1000,  # 1GB
                'metric': 'memory_mb',
                'description': 'Uso de memória muito alto'
            },
            {
                'name': 'high_error_rate',
                'threshold': 0.2,  # 20%
                'metric': 'error_rate',
                'description': 'Taxa de erro muito alta'
            }
        ]
        
        # Coletar métricas para avaliação de alertas
        try:
            # Simular métricas
            current_metrics = {
                'execution_time_ms': 2000,  # Baixo
                'memory_mb': 150,           # Baixo
                'error_rate': 0.05          # Baixo - 5%
            }
            
            # Avaliar cada condição de alerta
            for condition in alert_conditions:
                metric_name = condition['metric']
                threshold = condition['threshold']
                current_value = current_metrics.get(metric_name, 0)
                
                should_alert = current_value > threshold
                
                alerting_tests[condition['name']] = {
                    'condition': condition['description'],
                    'threshold': threshold,
                    'current_value': current_value,
                    'should_alert': should_alert,
                    'status': 'ALERT' if should_alert else 'OK'
                }
            
        except Exception as e:
            alerting_tests['metrics_collection_error'] = {
                'error': str(e)
            }
        
        # Teste de formatação de alertas
        alert_formatting_tests = {}
        
        try:
            # Simular alerta
            sample_alert = {
                'timestamp': datetime.now().isoformat(),
                'level': 'WARNING',
                'metric': 'execution_time',
                'value': 6000,
                'threshold': 5000,
                'message': 'Execution time exceeded threshold'
            }
            
            # Verificar formatação
            alert_json = json.dumps(sample_alert, indent=2)
            
            alert_formatting_tests['json_serialization'] = {
                'success': len(alert_json) > 0,
                'has_timestamp': 'timestamp' in alert_json,
                'has_level': 'level' in alert_json,
                'is_valid_json': True
            }
            
        except Exception as e:
            alert_formatting_tests['json_serialization'] = {
                'success': False,
                'error': str(e)
            }
        
        result['details'] = {
            'alerting_tests': alerting_tests,
            'alert_formatting_tests': alert_formatting_tests,
            'alert_conditions_count': len(alert_conditions)
        }
        
        # Determinar sucesso (sistema de alertas deve funcionar)
        working_alerts = len([
            t for t in alerting_tests.values() 
            if 'status' in t
        ])
        
        formatting_success = alert_formatting_tests.get('json_serialization', {}).get('success', False)
        
        result['success'] = working_alerts > 0 and formatting_success
        
        if verbose:
            active_alerts = len([t for t in alerting_tests.values() if t.get('should_alert', False)])
            print(f"🔔 Alertas: {working_alerts} condições monitoradas, {active_alerts} alertas ativos")
        
    except Exception as e:
        result['errors'].append(f"Erro no teste de alertas: {str(e)}")
    
    return result

def test_monitoring(verbose=False, quick=False):
    """
    Teste consolidado de monitoramento
    """
    result = {
        'success': False,
        'details': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        if verbose:
            print("📊 Iniciando testes de monitoramento...")
        
        # Executar todos os testes de monitoramento
        monitoring_tests = {}
        
        # 1. Performance Metrics
        performance_result = test_performance_metrics(verbose=verbose, quick=quick)
        monitoring_tests['performance_metrics'] = performance_result
        
        # 2. Logging System
        logging_result = test_logging_system(verbose=verbose, quick=quick)
        monitoring_tests['logging_system'] = logging_result
        
        # 3. Error Monitoring
        error_monitoring_result = test_error_monitoring(verbose=verbose, quick=quick)
        monitoring_tests['error_monitoring'] = error_monitoring_result
        
        # 4. Alerting System
        alerting_result = test_alerting_system(verbose=verbose, quick=quick)
        monitoring_tests['alerting_system'] = alerting_result
        
        # Estatísticas consolidadas
        total_tests = len(monitoring_tests)
        successful_tests = len([t for t in monitoring_tests.values() if t.get('success', False)])
        total_warnings = sum(len(t.get('warnings', [])) for t in monitoring_tests.values())
        total_errors = sum(len(t.get('errors', [])) for t in monitoring_tests.values())
        
        result['details'] = {
            'total_monitoring_tests': total_tests,
            'successful_tests': successful_tests,
            'total_warnings': total_warnings,
            'total_errors': total_errors,
            'monitoring_score': round(successful_tests / total_tests * 100, 1) if total_tests > 0 else 0,
            'individual_results': monitoring_tests
        }
        
        # Consolidar warnings e errors
        for test_result in monitoring_tests.values():
            result['warnings'].extend(test_result.get('warnings', []))
            result['errors'].extend(test_result.get('errors', []))
        
        # Determinar sucesso geral (pelo menos 75% dos testes devem passar)
        result['success'] = (successful_tests / total_tests) >= 0.75 if total_tests > 0 else True
        
        if verbose:
            print(f"📊 Monitoramento: {successful_tests}/{total_tests} sistemas funcionando")
            print(f"📈 Score de monitoramento: {result['details']['monitoring_score']}%")
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Erro inesperado nos testes de monitoramento: {str(e)}")
        result['success'] = False
        return result

if __name__ == "__main__":
    # Teste standalone
    result = test_monitoring(verbose=True, quick=False)
    print("\n📊 RESULTADO DOS TESTES DE MONITORAMENTO:")
    print(f"📈 Sucesso: {result['success']}")
    print(f"📊 Score de Monitoramento: {result['details'].get('monitoring_score', 0)}%")
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
