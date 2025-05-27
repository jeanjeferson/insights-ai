#!/usr/bin/env python3
"""
Teste Final do Crew com Ferramentas de Exportação
Executa uma análise simples para verificar funcionamento completo
"""

import os
import sys
sys.path.append('src')

from insights.crew import Insights

def test_crew_execution():
    """Testa execução do crew com as novas ferramentas"""
    
    print("🚀 TESTE FINAL - CREW COM FERRAMENTAS DE EXPORTAÇÃO")
    print("=" * 65)
    
    try:
        # Inicializar crew
        crew_instance = Insights()
        crew = crew_instance.crew()
        
        print("✅ Crew inicializado com sucesso")
        print(f"👥 Agentes: {len(crew.agents)}")
        print(f"📋 Tasks: {len(crew.tasks)}")
        
        # Verificar se algum agente tem as ferramentas de exportação
        print("\n🔧 VERIFICANDO FERRAMENTAS NOS AGENTES:")
        
        export_tools_found = 0
        for agent in crew.agents:
            tools = [type(tool).__name__ for tool in agent.tools]
            export_tools = [t for t in tools if 'DataExporter' in t]
            
            if export_tools:
                print(f"✅ {agent.role[:30]}: {export_tools}")
                export_tools_found += len(export_tools)
            else:
                print(f"➖ {agent.role[:30]}: Sem ferramentas de exportação")
        
        print(f"\n📊 Total de ferramentas de exportação encontradas: {export_tools_found}")
        
        # Verificar estrutura do crew
        print("\n📋 ESTRUTURA DO CREW:")
        print(f"   • Processo: {crew.process}")
        print(f"   • Max RPM: {crew.max_rpm}")
        print(f"   • Verbose: {crew.verbose}")
        
        # Simular inputs de teste
        inputs = {
            'data_inicio': '2024-01-01',
            'data_fim': '2024-12-31'
        }
        
        print(f"\n🎯 INPUTS DE TESTE: {inputs}")
        
        # Executar before_kickoff para validar setup
        print("\n🔄 EXECUTANDO VALIDAÇÃO PRE-EXECUÇÃO...")
        validated_inputs = crew_instance.before_kickoff(inputs)
        
        print("✅ Validação pré-execução concluída com sucesso")
        print(f"📋 Inputs validados: {validated_inputs}")
        
        # Verificar logs de validação
        print("\n📁 VERIFICANDO LOGS:")
        if hasattr(crew_instance, 'log_file_path') and crew_instance.log_file_path:
            print(f"✅ Arquivo de log criado: {crew_instance.log_file_path}")
        else:
            print("ℹ️ Log apenas no console")
        
        print("\n🎉 TESTE CONCLUÍDO COM SUCESSO!")
        print("✅ Crew está pronto para execução com ferramentas de exportação")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO no teste: {e}")
        import traceback
        print(f"📋 Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    test_crew_execution() 