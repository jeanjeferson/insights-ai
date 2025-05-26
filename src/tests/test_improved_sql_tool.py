#!/usr/bin/env python
"""
Script de teste para a SQL Query Tool melhorada
"""

import sys
import time
import logging
from datetime import datetime, timedelta
from src.insights.tools.sql_query_tool_improved import SQLServerQueryToolImproved

# Configurar logging detalhado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sql_tool_improved_debug.log')
    ]
)

logger = logging.getLogger(__name__)

def test_improved_tool():
    """Testar a tool melhorada"""
    logger.info("🚀 TESTANDO SQL QUERY TOOL MELHORADA")
    logger.info("=" * 60)
    
    try:
        tool = SQLServerQueryToolImproved()
        
        # Teste 1: Conectividade
        logger.info("🔍 TESTE 1: Conectividade")
        if tool.test_connection():
            logger.info("✅ Conectividade OK")
        else:
            logger.error("❌ Falha na conectividade")
            return False
        
        # Teste 2: Query com período pequeno
        logger.info("\n🔍 TESTE 2: Query período pequeno")
        result = tool._run(
            date_start="2024-01-01", 
            date_end="2024-01-02",
            output_format="summary"
        )
        logger.info(f"✅ Resultado: {len(str(result))} caracteres")
        
        # Teste 3: Query com período do crew (pode demorar)
        logger.info("\n🔍 TESTE 3: Query período crew (com progress logs)")
        data_fim = datetime.now().strftime('%Y-%m-%d')
        data_inicio = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        start_time = time.time()
        result = tool._run(
            date_start=data_inicio,
            date_end=data_fim,
            output_format="summary"
        )
        execution_time = time.time() - start_time
        
        logger.info(f"✅ Query crew executada em {execution_time:.2f}s")
        logger.info(f"📊 Resultado: {len(str(result))} caracteres")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no teste: {e}")
        logger.exception("Stack trace:")
        return False

if __name__ == "__main__":
    try:
        success = test_improved_tool()
        if success:
            logger.info("🎉 TOOL MELHORADA FUNCIONANDO PERFEITAMENTE!")
        else:
            logger.error("❌ TOOL MELHORADA COM PROBLEMAS")
    except KeyboardInterrupt:
        logger.info("⚠️ Teste interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro geral: {e}")
        logger.exception("Stack trace completo:") 