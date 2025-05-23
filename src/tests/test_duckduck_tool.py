"""
🔧 TESTE: DUCKDUCKGO SEARCH TOOL
================================

Testa a ferramenta de busca DuckDuckGo do projeto Insights-AI.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import time
import threading
from unittest.mock import patch, MagicMock

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent))

try:
    from insights.tools.duckduck_tool import DuckDuckGoSearchTool, DuckSearchInput
except ImportError as e:
    print(f"⚠️ Erro ao importar DuckDuckGoSearchTool: {e}")
    DuckDuckGoSearchTool = None
    DuckSearchInput = None

def test_duckduck_tool(verbose=False, quick=False):
    """
    Teste da ferramenta DuckDuckGo Search Tool
    """
    result = {
        'success': False,
        'details': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        if verbose:
            print("🔧 Testando DuckDuckGo Search Tool...")
        
        # 1. Verificar se as classes foram importadas
        if DuckDuckGoSearchTool is None:
            result['errors'].append("Não foi possível importar DuckDuckGoSearchTool")
            return result
            
        if DuckSearchInput is None:
            result['errors'].append("Não foi possível importar DuckSearchInput")
            return result
        
        # 2. Instanciar a ferramenta
        try:
            search_tool = DuckDuckGoSearchTool()
            if verbose:
                print("✅ DuckDuckGoSearchTool instanciada com sucesso")
        except Exception as e:
            result['errors'].append(f"Erro ao instanciar DuckDuckGoSearchTool: {str(e)}")
            return result
        
        # 3. Verificar se tem os atributos necessários
        required_attributes = ['name', 'description', '_run', 'args_schema']
        missing_attributes = [attr for attr in required_attributes if not hasattr(search_tool, attr)]
        if missing_attributes:
            result['warnings'].append(f"Atributos ausentes: {missing_attributes}")
        
        # 4. Testar informações básicas da ferramenta
        tool_info = {
            'name': getattr(search_tool, 'name', 'N/A'),
            'description': getattr(search_tool, 'description', 'N/A')[:100] + "..." if len(getattr(search_tool, 'description', '')) > 100 else getattr(search_tool, 'description', 'N/A'),
            'has_args_schema': hasattr(search_tool, 'args_schema'),
            'schema_type': str(type(getattr(search_tool, 'args_schema', None)))
        }
        
        # 5. Testar schema de entrada
        schema_validation = {}
        try:
            # Testar criação do schema com dados válidos
            valid_input = DuckSearchInput(query="test query")
            schema_validation['valid_input_creation'] = 'OK'
            schema_validation['query_field'] = hasattr(valid_input, 'query')
            
            # Testar validação de campos obrigatórios
            try:
                invalid_input = DuckSearchInput()
                schema_validation['required_validation'] = 'FALHOU - Query deveria ser obrigatório'
            except Exception:
                schema_validation['required_validation'] = 'OK - Query é obrigatório'
                
        except Exception as e:
            schema_validation['schema_error'] = f'ERRO: {str(e)}'
        
        # 6. Verificar controle de rate limit
        rate_limit_test = {}
        try:
            # Verificar se tem os atributos de controle
            rate_limit_test['has_last_request_time'] = hasattr(search_tool, '_last_request_time')
            rate_limit_test['has_lock'] = hasattr(search_tool, '_lock')
            rate_limit_test['has_min_interval'] = hasattr(search_tool, '_MIN_INTERVAL')
            
            # Verificar valores iniciais
            if hasattr(search_tool, '_MIN_INTERVAL'):
                rate_limit_test['min_interval_value'] = search_tool._MIN_INTERVAL
                rate_limit_test['interval_reasonable'] = 0.5 <= search_tool._MIN_INTERVAL <= 5.0
            
            if hasattr(search_tool, '_lock'):
                rate_limit_test['lock_is_threading_lock'] = isinstance(search_tool._lock, type(threading.Lock()))
                
        except Exception as e:
            rate_limit_test['rate_limit_error'] = f'ERRO: {str(e)}'
        
        # 7. Testar funcionalidade básica (mock)
        mock_search_test = {}
        try:
            with patch.object(search_tool, 'search_tool') as mock_search:
                mock_search.run.return_value = "Resultado de teste da busca"
                
                # Teste básico
                result_basic = search_tool._run("teste")
                mock_search_test['basic_search'] = 'OK' if result_basic else 'Vazio'
                mock_search_test['basic_result'] = result_basic[:50] + "..." if len(str(result_basic)) > 50 else str(result_basic)
                
                # Teste com domínio específico
                result_domain = search_tool._run("teste", domain="com")
                mock_search_test['domain_search'] = 'OK' if result_domain else 'Vazio'
                
                # Verificar se o método foi chamado corretamente
                mock_search_test['method_calls'] = mock_search.run.call_count
                mock_search_test['last_call_args'] = str(mock_search.run.call_args) if mock_search.run.call_args else 'N/A'
                
        except Exception as e:
            mock_search_test['mock_error'] = f'ERRO: {str(e)}'
        
        # 8. Testar rate limiting (simulado)
        rate_limit_functional_test = {}
        if not quick:  # Só executar se não for quick test
            try:
                with patch.object(search_tool, 'search_tool') as mock_search:
                    mock_search.run.return_value = "Resultado teste"
                    
                    # Executar duas buscas consecutivas e medir tempo
                    start_time = time.time()
                    search_tool._run("query1")
                    mid_time = time.time()
                    search_tool._run("query2")
                    end_time = time.time()
                    
                    total_time = end_time - start_time
                    min_expected_time = search_tool._MIN_INTERVAL if hasattr(search_tool, '_MIN_INTERVAL') else 1.0
                    
                    rate_limit_functional_test['total_time'] = round(total_time, 3)
                    rate_limit_functional_test['min_expected'] = min_expected_time
                    rate_limit_functional_test['rate_limit_working'] = total_time >= min_expected_time * 0.8  # 80% de tolerância
                    
            except Exception as e:
                rate_limit_functional_test['functional_error'] = f'ERRO: {str(e)}'
        else:
            rate_limit_functional_test['skipped'] = 'Pulado no modo quick'
        
        # 9. Testar parâmetros de entrada
        parameter_test = {}
        try:
            with patch.object(search_tool, 'search_tool') as mock_search:
                mock_search.run.return_value = "Resultado teste"
                
                # Teste com query vazia
                empty_result = search_tool._run("")
                parameter_test['empty_query'] = 'OK' if empty_result else 'Falhou'
                
                # Teste com query longa
                long_query = "a" * 200
                long_result = search_tool._run(long_query)
                parameter_test['long_query'] = 'OK' if long_result else 'Falhou'
                
                # Teste com caracteres especiais
                special_query = "joias & anéis 'premium' \"diamantes\""
                special_result = search_tool._run(special_query)
                parameter_test['special_chars'] = 'OK' if special_result else 'Falhou'
                
                # Teste com diferentes domínios
                domains = ['br', 'com', 'org']
                domain_results = {}
                for domain in domains:
                    try:
                        domain_result = search_tool._run("test", domain=domain)
                        domain_results[domain] = 'OK' if domain_result else 'Vazio'
                    except Exception as e:
                        domain_results[domain] = f'ERRO: {str(e)}'
                
                parameter_test['domain_tests'] = domain_results
                
        except Exception as e:
            parameter_test['parameter_error'] = f'ERRO: {str(e)}'
        
        # 10. Testar dependency de langchain
        dependency_test = {}
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            dependency_test['langchain_import'] = 'OK'
            
            # Verificar se search_tool tem o tipo correto
            if hasattr(search_tool, 'search_tool'):
                dependency_test['search_tool_type'] = str(type(search_tool.search_tool))
                dependency_test['correct_type'] = isinstance(search_tool.search_tool, DuckDuckGoSearchRun)
            else:
                dependency_test['search_tool_missing'] = 'search_tool attribute não encontrado'
                
        except ImportError as e:
            dependency_test['langchain_error'] = f'ERRO: {str(e)}'
            result['warnings'].append("Dependência langchain_community não está disponível")
        
        # 11. Compilar resultados
        result['details'] = {
            'tool_info': tool_info,
            'schema_validation': schema_validation,
            'rate_limit_test': rate_limit_test,
            'mock_search_test': mock_search_test,
            'rate_limit_functional_test': rate_limit_functional_test,
            'parameter_test': parameter_test,
            'dependency_test': dependency_test
        }
        
        # 12. Determinar sucesso
        critical_errors = [
            error for error in result['errors'] 
            if 'instanciar' in error or 'importar' in error
        ]
        
        # Verificar se passou nos testes essenciais
        essential_tests_passed = (
            tool_info.get('name') != 'N/A' and
            'OK' in str(schema_validation) and
            'OK' in str(mock_search_test)
        )
        
        if not critical_errors and essential_tests_passed:
            result['success'] = True
            if verbose:
                print("✅ DuckDuckGo Search Tool passou nos testes básicos")
        else:
            if verbose:
                print("❌ DuckDuckGo Search Tool falhou em testes críticos")
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Erro inesperado no teste DuckDuckGo: {str(e)}")
        result['success'] = False
        return result

def test_duckduck_real_search():
    """Teste real de busca (usar com cuidado - rate limit)"""
    if DuckDuckGoSearchTool is None:
        return False, "Ferramenta não disponível"
    
    try:
        search_tool = DuckDuckGoSearchTool()
        
        # Busca simples e rápida
        result = search_tool._run("joalherias brasil", domain="br")
        
        # Verificações básicas
        success = (
            isinstance(result, str) and
            len(result) > 0 and
            'joalherias' in result.lower() or 'brasil' in result.lower()
        )
        
        return success, result[:100] + "..." if len(result) > 100 else result
        
    except Exception as e:
        return False, f"Erro no teste real: {str(e)}"

def test_rate_limit_timing():
    """Teste específico do rate limiting"""
    if DuckDuckGoSearchTool is None:
        return False, "Ferramenta não disponível"
    
    try:
        search_tool = DuckDuckGoSearchTool()
        
        with patch.object(search_tool, 'search_tool') as mock_search:
            mock_search.run.return_value = "Resultado teste"
            
            times = []
            for i in range(3):
                start = time.time()
                search_tool._run(f"query{i}")
                end = time.time()
                times.append(end - start)
            
            # O primeiro deve ser rápido, os outros devem respeitar o rate limit
            first_fast = times[0] < 0.5
            others_respect_limit = all(t >= search_tool._MIN_INTERVAL * 0.8 for t in times[1:])
            
            return first_fast and others_respect_limit, {
                'times': [round(t, 3) for t in times],
                'first_fast': first_fast,
                'others_respect_limit': others_respect_limit
            }
            
    except Exception as e:
        return False, f"Erro no teste de rate limit: {str(e)}"

if __name__ == "__main__":
    # Executar teste se chamado diretamente
    print("🧪 Executando teste do DuckDuckGo Search Tool...")
    result = test_duckduck_tool(verbose=True, quick=False)
    
    if result['success']:
        print("✅ Todos os testes passaram!")
    else:
        print("❌ Alguns testes falharam:")
        for error in result['errors']:
            print(f"   • {error}")
    
    print("\n📊 Detalhes:")
    for section, details in result['details'].items():
        print(f"   {section}: {details}") 