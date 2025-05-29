#!/usr/bin/env python3
"""
🧪 TESTE DE ARQUITETURA ETL - SEPARAÇÃO SQL/CSV
===============================================

Este script testa a nova arquitetura onde:
- APENAS o engenheiro_dados tem acesso ao SQL Server
- Outros agentes trabalham com vendas.csv exportado
- Separação clara de responsabilidades e segurança

Funcionalidades testadas:
- Isolamento de acesso SQL ao engenheiro_dados
- Validação de ferramentas por agente
- Configuração correta do fluxo ETL
- Instruções específicas por tipo de agente
"""

import os
import sys
from datetime import datetime, timedelta

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from insights.crew_optimized import (
    OptimizedInsights, 
    IntelligentContextManager,
    get_performance_metrics,
    log_performance_summary
)

from insights.config.tools_config_v3 import (
    validate_agent_data_access,
    get_data_flow_architecture,
    get_tools_for_agent,
    sql_query_tool,
    file_read_tool
)

def test_sql_access_isolation():
    """Testar se apenas o engenheiro_dados tem acesso SQL"""
    
    print("🧪 TESTE: Isolamento de Acesso SQL")
    print("=" * 50)
    
    all_agents = [
        'engenheiro_dados', 'analista_vendas_tendencias', 'especialista_produtos',
        'analista_estoque', 'analista_financeiro', 'especialista_clientes',
        'analista_performance', 'diretor_insights'
    ]
    
    sql_access_results = {}
    
    for agent in all_agents:
        tools = get_tools_for_agent(agent)
        has_sql = sql_query_tool in tools
        has_file_read = file_read_tool in tools
        
        sql_access_results[agent] = {
            'has_sql': has_sql,
            'has_file_read': has_file_read,
            'tools_count': len(tools)
        }
        
        if agent == 'engenheiro_dados':
            if has_sql and has_file_read:
                print(f"✅ {agent}: SQL ✅ + FileRead ✅ (correto)")
            else:
                print(f"❌ {agent}: SQL {'❌' if not has_sql else '✅'} + FileRead {'❌' if not has_file_read else '✅'} (erro)")
        else:
            if not has_sql and has_file_read:
                print(f"✅ {agent}: SQL ❌ + FileRead ✅ (correto)")
            else:
                print(f"❌ {agent}: SQL {'✅' if has_sql else '❌'} + FileRead {'✅' if has_file_read else '❌'} (erro)")
    
    # Verificar se isolamento está correto
    engineer_has_sql = sql_access_results['engenheiro_dados']['has_sql']
    others_without_sql = all(not sql_access_results[agent]['has_sql'] 
                           for agent in all_agents if agent != 'engenheiro_dados')
    all_have_file_read = all(sql_access_results[agent]['has_file_read'] for agent in all_agents)
    
    success = engineer_has_sql and others_without_sql and all_have_file_read
    
    print(f"\n📊 Resultado do isolamento:")
    print(f"   🔐 Engenheiro tem SQL: {'✅' if engineer_has_sql else '❌'}")
    print(f"   🚫 Outros sem SQL: {'✅' if others_without_sql else '❌'}")
    print(f"   📁 Todos têm FileRead: {'✅' if all_have_file_read else '❌'}")
    
    return success

def test_access_validation():
    """Testar a validação de acesso aos dados"""
    
    print("\n🧪 TESTE: Validação de Acesso aos Dados")
    print("=" * 50)
    
    all_agents = [
        'engenheiro_dados', 'analista_vendas_tendencias', 'especialista_produtos',
        'analista_estoque', 'analista_financeiro', 'especialista_clientes',
        'analista_performance', 'diretor_insights'
    ]
    
    validation_success = 0
    
    for agent in all_agents:
        validation = validate_agent_data_access(agent)
        
        if validation['valid']:
            print(f"✅ {agent}: {validation['expected_access']}")
            validation_success += 1
        else:
            print(f"❌ {agent}: {validation.get('error', 'Erro desconhecido')}")
    
    success_rate = (validation_success / len(all_agents)) * 100
    print(f"\n📊 Taxa de sucesso na validação: {success_rate:.1f}% ({validation_success}/{len(all_agents)})")
    
    return success_rate == 100.0

def test_etl_architecture():
    """Testar a arquitetura ETL configurada"""
    
    print("\n🧪 TESTE: Arquitetura ETL")
    print("=" * 50)
    
    # Verificar configuração da arquitetura
    data_flow = get_data_flow_architecture()
    
    # Validações da arquitetura
    checks = [
        ('Fonte de dados', data_flow['data_source'] == 'SQL Server Database'),
        ('Extrator de dados', data_flow['data_extractor'] == 'engenheiro_dados'),
        ('Arquivo gerado', data_flow['extracted_file'] == 'data/vendas.csv'),
        ('Tipo de arquitetura', data_flow['architecture_type'] == 'ETL + File-based Analysis'),
        ('SQL restrito a', data_flow['sql_access_restricted_to'] == ['engenheiro_dados']),
        ('Workflow definido', len(data_flow['workflow']) == 5)
    ]
    
    passed_checks = 0
    for check_name, check_result in checks:
        if check_result:
            print(f"✅ {check_name}: OK")
            passed_checks += 1
        else:
            print(f"❌ {check_name}: Falhou")
    
    print(f"\n📊 Arquitetura ETL: {passed_checks}/{len(checks)} verificações passaram")
    
    # Verificar métricas
    metrics = get_performance_metrics()
    etl_metrics_ok = (
        metrics['architecture'] == 'ETL_CSV_based' and
        metrics['data_access_model'] == 'separated_responsibilities' and
        metrics['sql_access_restricted_to'] == 'engenheiro_dados' and
        metrics['csv_based_analysis'] == 'enabled'
    )
    
    if etl_metrics_ok:
        print("✅ Métricas ETL: Configuradas corretamente")
    else:
        print("❌ Métricas ETL: Problemas na configuração")
    
    return passed_checks == len(checks) and etl_metrics_ok

def test_task_instructions():
    """Testar se as instruções das tasks estão corretas para a arquitetura ETL"""
    
    print("\n🧪 TESTE: Instruções das Tasks")
    print("=" * 50)
    
    try:
        crew_instance = OptimizedInsights()
        
        # Simular inputs
        test_inputs = {
            'data_inicio': '2024-01-01',
            'data_fim': '2024-01-31'
        }
        
        # Executar before_kickoff
        processed_inputs = crew_instance.before_kickoff(test_inputs)
        
        # Verificar configurações ETL
        etl_checks = [
            ('Arquitetura ETL', 'data_flow_architecture' in processed_inputs),
            ('Arquivo ETL definido', processed_inputs.get('etl_file') == 'data/vendas.csv'),
            ('Extrator definido', processed_inputs.get('data_extractor') == 'engenheiro_dados'),
            ('Instruções de fluxo', 'data_flow_instructions' in processed_inputs),
            ('Config engenheiro', 'engenheiro_dados_config' in processed_inputs),
            ('Validação de acesso', 'access_validation' in processed_inputs)
        ]
        
        passed_etl = 0
        for check_name, check_result in etl_checks:
            if check_result:
                print(f"✅ {check_name}: Configurado")
                passed_etl += 1
            else:
                print(f"❌ {check_name}: Ausente")
        
        # Verificar instruções específicas
        instructions = processed_inputs.get('data_flow_instructions', '')
        
        instruction_checks = [
            ('SQL apenas para engenheiro', 'ÚNICO agente com acesso direto ao SQL Server' in instructions),
            ('CSV para outros', 'NÃO fazem consultas SQL diretamente' in instructions),
            ('FileReadTool mencionado', 'FileReadTool' in instructions),
            ('Fluxo definido', 'FLUXO DE ARQUIVOS' in instructions)
        ]
        
        passed_instructions = 0
        for check_name, check_result in instruction_checks:
            if check_result:
                print(f"✅ {check_name}: OK")
                passed_instructions += 1
            else:
                print(f"❌ {check_name}: Ausente")
        
        total_passed = passed_etl + passed_instructions
        total_checks = len(etl_checks) + len(instruction_checks)
        
        print(f"\n📊 Configuração das tasks: {total_passed}/{total_checks} verificações")
        
        return total_passed == total_checks
        
    except Exception as e:
        print(f"❌ ERRO NO TESTE: {e}")
        return False

def main():
    """Executar todos os testes da arquitetura ETL"""
    
    print("🚀 TESTE COMPLETO: Arquitetura ETL - Separação SQL/CSV")
    print("=" * 60)
    
    # Configurar ambiente de teste
    os.environ['INSIGHTS_LOG_LEVEL'] = 'DEBUG'
    
    tests_passed = 0
    total_tests = 4
    
    # Executar testes
    try:
        if test_sql_access_isolation():
            tests_passed += 1
            
        if test_access_validation():
            tests_passed += 1
            
        if test_etl_architecture():
            tests_passed += 1
            
        if test_task_instructions():
            tests_passed += 1
            
    except Exception as e:
        print(f"❌ ERRO GERAL: {e}")
    
    # Resultado final
    print("\n" + "=" * 60)
    print("📊 RESULTADO DOS TESTES:")
    print(f"✅ Testes passaram: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n🎉 ARQUITETURA ETL PERFEITA!")
        print("✅ Acesso SQL isolado ao engenheiro_dados")
        print("✅ Outros agentes trabalham apenas com CSV")
        print("✅ Separação de responsabilidades implementada")
        print("✅ Validações de acesso funcionando")
        print("✅ Fluxo ETL corretamente configurado")
        print("\n💡 Sistema pronto para análises seguras e eficientes!")
    else:
        print("\n⚠️ PROBLEMAS DETECTADOS NA ARQUITETURA")
        print("❌ Verifique as configurações antes de executar")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 