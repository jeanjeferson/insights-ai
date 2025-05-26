"""
🚀 SCRIPT: EXECUTAR EXPORT CSV REAL
==================================

Script para executar consulta real ao banco SQL Server
e salvar os dados em data/vendas.csv
"""

import sys
from pathlib import Path

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from insights.tools.sql_query_tool import SQLServerQueryTool
    print("✅ SQL Query Tool importado com sucesso")
except ImportError as e:
    print(f"❌ Erro ao importar SQL Query Tool: {e}")
    print("💡 Verifique se o caminho src/insights/tools/sql_query_tool.py está correto")
    sys.exit(1)

def executar_export_csv_completo():
    """Executar export completo de dados para CSV"""
    print("🚀 INICIANDO EXPORT CSV COMPLETO")
    print("=" * 40)
    
    # Instanciar ferramenta
    sql_tool = SQLServerQueryTool()
    
    # Mostrar configurações do banco
    print("🔧 CONFIGURAÇÕES DO BANCO:")
    print(f"   🖥️  Servidor: {sql_tool.DB_SERVER}:{sql_tool.DB_PORT}")
    print(f"   🗄️  Database: {sql_tool.DB_DATABASE}")
    print(f"   👤 Usuário: {sql_tool.DB_UID}")
    print(f"   🚗 Driver: {sql_tool.DB_DRIVER}")
    
    # Verificar se diretório data existe
    data_dir = Path("data")
    if not data_dir.exists():
        print("📁 Criando diretório 'data'...")
        data_dir.mkdir(exist_ok=True)
    
    try:
        print("\n🔗 Conectando ao banco e executando consulta...")
        print("⏳ Isto pode demorar alguns minutos...")
        
        # Executar o método que faz consulta real
        sql_tool._execute_query_and_save_to_csv()
        
        # Verificar se arquivo foi criado
        csv_file = Path("data/vendas.csv")
        
        if csv_file.exists():
            file_size = csv_file.stat().st_size
            print(f"\n✅ SUCESSO! Arquivo CSV criado:")
            print(f"   📁 Local: {csv_file.absolute()}")
            print(f"   📊 Tamanho: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
            
            # Tentar ler primeiras linhas para verificar
            try:
                import pandas as pd
                df_sample = pd.read_csv(csv_file, sep=';', encoding='utf-8', nrows=5)
                print(f"   📋 Colunas: {len(df_sample.columns)}")
                print(f"   📈 Amostra carregada: {len(df_sample)} linhas")
                
                # Mostrar nome das colunas
                print(f"   🏷️  Colunas: {', '.join(df_sample.columns[:5])}...")
                
            except Exception as read_error:
                print(f"   ⚠️ Arquivo criado mas erro ao ler amostra: {read_error}")
            
            return True
            
        else:
            print("❌ ERRO: Arquivo CSV não foi criado")
            return False
            
    except Exception as e:
        error_msg = str(e).lower()
        
        if any(keyword in error_msg for keyword in ["fonte de dados", "driver", "odbc"]):
            print(f"\n❌ ERRO DE CONEXÃO/DRIVER:")
            print(f"   {e}")
            print(f"\n💡 SOLUÇÕES:")
            print(f"   1. Verifique se o SQL Server está rodando")
            print(f"   2. Confirme as credenciais no arquivo .env")
            print(f"   3. Instale o driver ODBC: {sql_tool.DB_DRIVER}")
            print(f"   4. Verifique conectividade de rede com o servidor")
            
        elif "login" in error_msg or "senha" in error_msg:
            print(f"\n❌ ERRO DE AUTENTICAÇÃO:")
            print(f"   {e}")
            print(f"\n💡 VERIFICAR:")
            print(f"   - Usuário: {sql_tool.DB_UID}")
            print(f"   - Senha configurada no .env")
            print(f"   - Permissões do usuário no banco")
            
        else:
            print(f"\n❌ ERRO INESPERADO:")
            print(f"   {e}")
            
        return False

def executar_export_periodo_especifico():
    """Executar export com período específico"""
    print("\n🗓️ EXPORT COM PERÍODO ESPECÍFICO")
    print("=" * 35)
    
    from datetime import datetime, timedelta
    
    # Configurar período (últimos 90 dias)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    date_start = start_date.strftime('%Y-%m-%d')
    date_end = end_date.strftime('%Y-%m-%d')
    
    print(f"📅 Período: {date_start} até {date_end}")
    
    sql_tool = SQLServerQueryTool()
    
    try:
        print("🔍 Executando consulta com filtro de data...")
        
        # Usar método _run para obter dados
        result = sql_tool._run(
            date_start=date_start,
            date_end=date_end,
            output_format="csv"
        )
        
        if "Erro ao executar consulta SQL" in result:
            print(f"❌ Erro na consulta: {result}")
            return False
        
        # Extrair dados CSV e salvar
        if "Recuperados" in result:
            # Separar informações do CSV
            lines = result.split('\n')
            csv_start = -1
            
            for i, line in enumerate(lines):
                if line.strip() and ',' in line or ';' in line:
                    csv_start = i
                    break
            
            if csv_start >= 0:
                csv_data = '\n'.join(lines[csv_start:])
                
                # Salvar arquivo
                csv_path = "data/vendas.csv"
                with open(csv_path, 'w', encoding='utf-8') as f:
                    f.write(csv_data)
                
                print(f"✅ Dados salvos em {csv_path}")
                print(f"📊 {len(csv_data)} caracteres salvos")
                return True
            
        print("⚠️ Nenhum dado CSV encontrado no resultado")
        return False
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    print("🚀 EXPORT CSV - SQL SERVER")
    print("=" * 30)
    
    # Verificar argumentos
    if len(sys.argv) > 1:
        if sys.argv[1] == "--periodo":
            success = executar_export_periodo_especifico()
        else:
            print("❓ Argumentos disponíveis:")
            print("   --periodo : Export com período específico (90 dias)")
            print("   (sem args): Export completo (2 anos)")
            sys.exit(0)
    else:
        success = executar_export_csv_completo()
    
    if success:
        print("\n🎉 EXPORT CONCLUÍDO COM SUCESSO!")
        print("📁 Verifique o arquivo data/vendas.csv")
    else:
        print("\n❌ EXPORT FALHOU")
        print("💡 Verifique a configuração do banco de dados") 