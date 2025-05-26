#!/usr/bin/env python
"""
Script de teste isolado para diagnosticar problemas na SQL Query Tool
"""

import sys
import time
import logging
from datetime import datetime, timedelta
from src.insights.tools.sql_query_tool import SQLServerQueryTool

# Configurar logging detalhado
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('sql_tool_debug.log')
    ]
)

logger = logging.getLogger(__name__)

def test_sql_tool_connection():
    """Testar só a conectividade com o SQL Server"""
    logger.info("🔍 TESTE 1: Conectividade com SQL Server")
    
    try:
        import pyodbc
        from src.insights.tools.sql_query_tool import SQLServerQueryTool
        
        tool = SQLServerQueryTool()
        
        # Construir string de conexão
        conn_str = (
            f"DRIVER={{{tool.DB_DRIVER}}};"
            f"SERVER={tool.DB_SERVER},{tool.DB_PORT};"
            f"DATABASE={tool.DB_DATABASE};"
            f"UID={tool.DB_UID};"
            f"PWD={tool.DB_PWD};"
        )
        
        logger.info(f"🔌 Tentando conectar: SERVER={tool.DB_SERVER}:{tool.DB_PORT}, DB={tool.DB_DATABASE}")
        
        # Teste de conexão com timeout
        start_time = time.time()
        conn = pyodbc.connect(conn_str, timeout=30)
        
        connection_time = time.time() - start_time
        logger.info(f"✅ Conexão estabelecida em {connection_time:.2f}s")
        
        # Testar query simples
        cursor = conn.cursor()
        cursor.execute("SELECT GETDATE() as current_time")
        result = cursor.fetchone()
        logger.info(f"✅ Query teste executada: {result[0]}")
        
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro de conectividade: {e}")
        return False

def test_sql_tool_simple_query():
    """Testar query simples na SQL Tool"""
    logger.info("🔍 TESTE 2: Query Simples SQL Tool")
    
    try:
        tool = SQLServerQueryTool()
        
        # Parâmetros simples
        date_start = "2024-01-01"
        date_end = "2024-01-02" 
        
        logger.info(f"📅 Testando período: {date_start} a {date_end}")
        
        start_time = time.time()
        result = tool._run(date_start=date_start, date_end=date_end, output_format="summary")
        execution_time = time.time() - start_time
        
        logger.info(f"✅ Query executada em {execution_time:.2f}s")
        logger.info(f"📊 Resultado: {len(str(result))} caracteres")
        logger.info(f"📋 Preview: {str(result)[:200]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro na query simples: {e}")
        logger.exception("Stack trace:")
        return False

def test_sql_tool_with_current_inputs():
    """Testar com os inputs atuais do crew"""
    logger.info("🔍 TESTE 3: Query com Inputs do Crew")
    
    try:
        tool = SQLServerQueryTool()
        
        # Inputs atuais (últimos 2 anos)
        data_fim = datetime.now().strftime('%Y-%m-%d')
        data_inicio = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        logger.info(f"📅 Período do crew: {data_inicio} a {data_fim}")
        
        start_time = time.time()
        result = tool._run(date_start=data_inicio, date_end=data_fim, output_format="summary")
        execution_time = time.time() - start_time
        
        logger.info(f"✅ Query crew executada em {execution_time:.2f}s")
        logger.info(f"📊 Resultado: {len(str(result))} caracteres")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro na query do crew: {e}")
        logger.exception("Stack trace:")
        return False

def test_sql_tool_validation():
    """Testar validações da tool"""
    logger.info("🔍 TESTE 4: Validações da Tool")
    
    try:
        from src.insights.tools.sql_query_tool import SQLServerQueryInput
        
        # Teste 1: Parâmetros válidos
        logger.info("📋 Testando parâmetros válidos...")
        valid_input = SQLServerQueryInput(
            date_start="2024-01-01",
            date_end="2024-01-02",
            output_format="csv"
        )
        logger.info(f"✅ Validação passou: {valid_input}")
        
        # Teste 2: Data inválida
        logger.info("📋 Testando data inválida...")
        try:
            invalid_input = SQLServerQueryInput(
                date_start="invalid-date",
                date_end="2024-01-02"
            )
            logger.warning("⚠️ Validação deveria ter falhado mas passou")
        except Exception as e:
            logger.info(f"✅ Validação corretamente rejeitou data inválida: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro nos testes de validação: {e}")
        return False

def main():
    """Executar todos os testes"""
    logger.info("🚀 INICIANDO DIAGNÓSTICO SQL QUERY TOOL")
    logger.info("=" * 60)
    
    tests = [
        ("Conectividade", test_sql_tool_connection),
        ("Query Simples", test_sql_tool_simple_query),
        ("Inputs do Crew", test_sql_tool_with_current_inputs),
        ("Validações", test_sql_tool_validation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} falhou com exceção: {e}")
            results[test_name] = False
        
        time.sleep(2)  # Pausa entre testes
    
    # Resumo final
    logger.info("\n" + "="*60)
    logger.info("📊 RESUMO DOS TESTES:")
    for test_name, success in results.items():
        status = "✅ PASSOU" if success else "❌ FALHOU"
        logger.info(f"   {test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("🎉 TODOS OS TESTES PASSARAM - SQL Tool está funcionando")
    else:
        logger.error("⚠️ ALGUNS TESTES FALHARAM - Verificar logs para detalhes")
    
    return all_passed

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("⚠️ Teste interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro geral no teste: {e}")
        logger.exception("Stack trace completo:") 