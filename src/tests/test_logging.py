#!/usr/bin/env python3
"""
Teste do Sistema de Logging do Insights-AI Crew
Executa uma simulação básica para validar o logging em arquivo
"""

import os
import sys
from pathlib import Path

# Adicionar o src ao path para importar os módulos
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_logging_system():
    """Testa o sistema de logging sem executar o crew completo"""
    try:
        print("🧪 TESTANDO SISTEMA DE LOGGING...")
        print("=" * 50)
        
        # Importar e testar a função de logging
        from insights.crew import setup_crew_file_logging
        
        # Configurar logging
        crew_logger, log_file_path = setup_crew_file_logging()
        
        print(f"✅ Logger configurado com sucesso!")
        print(f"📁 Arquivo de log: {log_file_path}")
        
        # Testar diferentes níveis de log
        crew_logger.debug("🔍 Teste de log DEBUG")
        crew_logger.info("ℹ️ Teste de log INFO")
        crew_logger.warning("⚠️ Teste de log WARNING")
        crew_logger.error("❌ Teste de log ERROR")
        
        # Forçar flush
        for handler in crew_logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
        
        # Verificar se o arquivo foi criado
        if os.path.exists(log_file_path):
            file_size = os.path.getsize(log_file_path)
            print(f"✅ Arquivo de log criado com sucesso!")
            print(f"📊 Tamanho do arquivo: {file_size} bytes")
            
            # Ler e exibir o conteúdo
            with open(log_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"📄 Conteúdo do arquivo ({len(content)} chars):")
                print("-" * 30)
                print(content)
                print("-" * 30)
        else:
            print("❌ Arquivo de log não foi criado!")
            return False
        
        print("✅ TESTE DE LOGGING CONCLUÍDO COM SUCESSO!")
        return True
        
    except Exception as e:
        print(f"❌ ERRO no teste de logging: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_logging_system()
    if success:
        print("\n🎉 Sistema de logging está funcionando corretamente!")
    else:
        print("\n💥 Falha no sistema de logging!")
        sys.exit(1) 