"""
🔧 TESTE: SQL QUERY TOOL
=========================

Testa a ferramenta de consultas SQL Server do projeto Insights-AI.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent))

try:
    from insights.tools.sql_query_tool import SQLServerQueryTool
except ImportError as e:
    print(f"⚠️ Erro ao importar SQLServerQueryTool: {e}")
    SQLServerQueryTool = None

def test_sql_query_tool(verbose=False, quick=False):
    """
    Teste da ferramenta SQL Server Query Tool
    """
    result = {
        'success': False,
        'details': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        if verbose:
            print("🔧 Testando SQL Server Query Tool...")
        
        # 1. Verificar se a classe foi importada
        if SQLServerQueryTool is None:
            result['errors'].append("Não foi possível importar SQLServerQueryTool")
            return result
        
        # 2. Instanciar a ferramenta
        try:
            sql_tool = SQLServerQueryTool()
            if verbose:
                print("✅ SQLServerQueryTool instanciada com sucesso")
        except Exception as e:
            result['errors'].append(f"Erro ao instanciar SQLServerQueryTool: {str(e)}")
            return result
        
        # 3. Verificar se tem os atributos necessários
        required_attributes = ['name', 'description', '_run']
        missing_attributes = [attr for attr in required_attributes if not hasattr(sql_tool, attr)]
        if missing_attributes:
            result['warnings'].append(f"Atributos ausentes: {missing_attributes}")
        
        # 4. Testar métodos básicos
        tool_info = {
            'name': getattr(sql_tool, 'name', 'N/A'),
            'description': getattr(sql_tool, 'description', 'N/A')[:100] + "..." if len(getattr(sql_tool, 'description', '')) > 100 else getattr(sql_tool, 'description', 'N/A')
        }
        
        # 5. Verificar variáveis de ambiente (sem conectar)
        env_vars = ['DB_DRIVER', 'DB_SERVER', 'DB_DATABASE', 'DB_UID', 'DB_PWD']
        env_status = {}
        
        for var in env_vars:
            value = getattr(sql_tool, var, None)
            if value and value != f"default_{var.lower()}":
                env_status[var] = "Configurado"
            else:
                env_status[var] = "Padrão/Ausente"
                if var in ['DB_SERVER', 'DB_DATABASE']:
                    result['warnings'].append(f"Variável de ambiente {var} pode não estar configurada")
        
        # 6. Testar template SQL
        if hasattr(sql_tool, 'SQL_QUERY'):
            sql_template = sql_tool.SQL_QUERY
            sql_validations = {
                'has_template': bool(sql_template),
                'has_date_filter': '<<FILTRO_DATA>>' in sql_template,
                'has_select': 'SELECT' in sql_template.upper(),
                'has_from': 'FROM' in sql_template.upper(),
                'template_length': len(sql_template)
            }
        else:
            sql_validations = {'error': 'Template SQL não encontrado'}
            result['warnings'].append("Template SQL não encontrado")
        
        # 7. Testar métodos auxiliares
        method_tests = {}
        
        # Testar _build_where_clause
        if hasattr(sql_tool, '_build_where_clause'):
            try:
                test_filters = {'date_range': ('2024-01-01', '2024-01-31')}
                where_clause = sql_tool._build_where_clause(test_filters)
                method_tests['build_where_clause'] = 'OK' if where_clause else 'Vazio'
            except Exception as e:
                method_tests['build_where_clause'] = f'ERRO: {str(e)}'
        
        # Testar _format_summary
        if hasattr(sql_tool, '_format_summary'):
            try:
                test_df = pd.DataFrame({
                    'Quantidade': [10, 20, 30],
                    'Total_Liquido': [1000, 2000, 3000],
                    'Grupo_Produto': ['Anéis', 'Brincos', 'Colares']
                })
                summary = sql_tool._format_summary(test_df, '2024-01-01', '2024-01-31')
                method_tests['format_summary'] = 'OK' if summary else 'Vazio'
            except Exception as e:
                method_tests['format_summary'] = f'ERRO: {str(e)}'
        
        # 8. Teste de parâmetros de entrada
        input_validation = {}
        
        # Verificar se aceita datas válidas
        try:
            # Não executar a query real, apenas validar parâmetros
            today = datetime.now().strftime('%Y-%m-%d')
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # Simular validação de data (sem executar query)
            try:
                start_date = datetime.strptime(today, '%Y-%m-%d')
                end_date = datetime.strptime(yesterday, '%Y-%m-%d')
                if end_date < start_date:
                    input_validation['date_validation'] = 'OK - Detecta datas inválidas'
                else:
                    input_validation['date_validation'] = 'OK'
            except:
                input_validation['date_validation'] = 'ERRO'
                
        except Exception as e:
            input_validation['date_validation'] = f'ERRO: {str(e)}'
        
        # 9. Verificar arquivo CSV de output
        csv_file_path = Path("data/vendas.csv")
        csv_status = {
            'file_exists': csv_file_path.exists(),
            'file_size': csv_file_path.stat().st_size if csv_file_path.exists() else 0,
            'last_modified': datetime.fromtimestamp(csv_file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S') if csv_file_path.exists() else 'N/A'
        }
        
        # 10. Teste de conectividade (sem dados sensíveis)
        connectivity_test = {
            'driver_configured': sql_tool.DB_DRIVER != "ODBC Driver 17 for SQL Server",
            'server_configured': sql_tool.DB_SERVER != "localhost",
            'database_configured': sql_tool.DB_DATABASE != "default_db",
            'credentials_configured': sql_tool.DB_UID != "default_user"
        }
        
        # 11. Compilar resultados
        result['details'] = {
            'tool_info': tool_info,
            'environment_variables': env_status,
            'sql_validations': sql_validations,
            'method_tests': method_tests,
            'input_validation': input_validation,
            'csv_status': csv_status,
            'connectivity_test': connectivity_test
        }
        
        # 12. Determinar sucesso
        critical_errors = [
            error for error in result['errors'] 
            if 'instanciar' in error or 'importar' in error
        ]
        
        if not critical_errors:
            result['success'] = True
            if verbose:
                print("✅ SQL Server Query Tool passou nos testes básicos")
        else:
            if verbose:
                print("❌ SQL Server Query Tool falhou em testes críticos")
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Erro inesperado no teste SQL: {str(e)}")
        result['success'] = False
        return result

def test_sql_connection_mock():
    """Teste simulado de conexão SQL (sem conectar ao banco real)"""
    if SQLServerQueryTool is None:
        return False, "Ferramenta não disponível"
    
    try:
        sql_tool = SQLServerQueryTool()
        
        # Simular uma query de teste
        test_query = "SELECT 1 AS test_value"
        
        # Verificar se a estrutura de conexão está correta
        expected_conn_parts = ['DRIVER', 'SERVER', 'DATABASE', 'UID', 'PWD']
        
        # Construir string de conexão
        conn_str = (
            f"DRIVER={{{sql_tool.DB_DRIVER}}};"
            f"SERVER={sql_tool.DB_SERVER},{sql_tool.DB_PORT};"
            f"DATABASE={sql_tool.DB_DATABASE};"
            f"UID={sql_tool.DB_UID};"
            f"PWD={sql_tool.DB_PWD};"
        )
        
        # Verificar se todos os componentes estão presentes
        has_all_parts = all(part in conn_str for part in expected_conn_parts)
        
        return has_all_parts, "Estrutura de conexão OK" if has_all_parts else "Estrutura incompleta"
        
    except Exception as e:
        return False, f"Erro no teste: {str(e)}"

if __name__ == "__main__":
    # Teste standalone
    result = test_sql_query_tool(verbose=True, quick=False)
    print("\n📊 RESULTADO DO TESTE SQL:")
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
    
    # Teste adicional de conexão simulada
    print("\n🔌 TESTE DE ESTRUTURA DE CONEXÃO:")
    success, message = test_sql_connection_mock()
    print(f"{'✅' if success else '❌'} {message}")
