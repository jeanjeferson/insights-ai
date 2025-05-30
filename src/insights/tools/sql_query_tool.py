from crewai.tools import BaseTool
from typing import Type, Optional, ClassVar
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timedelta
import pyodbc
import pandas as pd
import os
import time

class SQLServerQueryInput(BaseModel):
    """Schema otimizado para consultas SQL Server com validações robustas."""
    
    date_start: str = Field(
        ..., 
        description="Data inicial para filtro no formato 'YYYY-MM-DD'. Use data_inicio fornecida pelo sistema.",
        json_schema_extra={
            "example": "2024-01-01",
            "pattern": r"^\d{4}-\d{2}-\d{2}$"
        }
    )
    
    date_end: Optional[str] = Field(
        None, 
        description="Data final para filtro no formato 'YYYY-MM-DD'. Se não fornecida, usa apenas data inicial. Use data_fim fornecida pelo sistema.",
        json_schema_extra={
            "example": "2024-12-31",
            "pattern": r"^\d{4}-\d{2}-\d{2}$"
        }
    )
    
    output_format: Optional[str] = Field(
        "csv", 
        description="Formato de saída: 'csv' (dados estruturados), 'summary' (resumo), 'json' (JSON), 'raw' (dados brutos).",
        json_schema_extra={
            "pattern": "^(csv|summary|json|raw)$"
        }
    )
    
    max_records: Optional[int] = Field(
        None,
        description="Limite máximo de registros para retornar. Para datasets grandes, use um limite menor.",
        json_schema_extra={
            "example": 100000
        }
    )
    
    @field_validator('date_start')
    @classmethod
    def validate_date_start(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError("date_start deve estar no formato YYYY-MM-DD")
    
    @field_validator('date_end')
    @classmethod
    def validate_date_end(cls, v, info):
        if v is not None:
            try:
                end_date = datetime.strptime(v, '%Y-%m-%d')
                # Em Pydantic v2, usar info.data para acessar outros campos
                if hasattr(info, 'data') and 'date_start' in info.data:
                    start_date = datetime.strptime(info.data['date_start'], '%Y-%m-%d')
                    if end_date < start_date:
                        raise ValueError("date_end deve ser posterior a date_start")
                return v
            except ValueError as e:
                if "date_end deve ser posterior" in str(e):
                    raise e
                raise ValueError("date_end deve estar no formato YYYY-MM-DD")
        return v

class SQLServerQueryTool(BaseTool):
    """
    🗄️ FERRAMENTA DE CONSULTA SQL SERVER PARA DADOS DE VENDAS
    
    QUANDO USAR:
    - Extrair dados de vendas do banco SQL Server
    - Aplicar filtros de data específicos nos dados
    - Obter dados estruturados para análises posteriores
    - Gerar relatórios baseados em períodos específicos
    - Alimentar outras ferramentas com dados filtrados
    
    CASOS DE USO ESPECÍFICOS:
    - Extrair vendas de um período específico para análise
    - Obter dados para alimentar ferramentas de KPI/BI
    - Gerar datasets para análises estatísticas
    - Criar extratos de vendas para relatórios executivos
    - Filtrar dados por data para análises temporais
    
    RESULTADOS ENTREGUES:
    - Dados de vendas estruturados em CSV/JSON
    - Informações completas de clientes, produtos e vendedores
    - Dados de margem, estoque e performance
    - Métricas agregadas por período
    - Dados prontos para análises avançadas
    
    IMPORTANTE:
    - SEMPRE use os inputs data_inicio e data_fim fornecidos pelo sistema
    - O filtro de data é aplicado automaticamente na consulta SQL
    - Dados incluem informações completas de vendas, clientes e produtos
    """
    
    name: str = "SQL Server Query Tool"
    description: str = (
        "Ferramenta especializada para extrair dados de vendas do SQL Server com filtros de data dinâmicos. "
        "Executa consultas otimizadas e retorna dados estruturados prontos para análises. "
        "OBRIGATÓRIO: Use sempre os inputs data_inicio e data_fim fornecidos pelo sistema."
    )
    args_schema: Type[BaseModel] = SQLServerQueryInput
    
    # Parâmetros de conexão com o banco de dados lidos do .env
    DB_DRIVER: str = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
    DB_SERVER: str = os.getenv("DB_SERVER", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "1433")
    DB_DATABASE: str = os.getenv("DB_DATABASE", "default_db")
    DB_UID: str = os.getenv("DB_UID", "default_user")
    DB_PWD: str = os.getenv("DB_PWD", "default_password")
    
    # Template da consulta SQL
    SQL_QUERY: ClassVar[str] = """
    SELECT
        CAST(vendas.datas AS DATE) AS Data,
        YEAR(vendas.datas) AS Ano,
        MONTH(vendas.datas) AS Mes,
        RTRIM(vendas.iclis) AS Codigo_Cliente,
        RTRIM(vendas.rclis) AS Nome_Cliente,
        RTRIM(clientes.sexos) AS Sexo,
        RTRIM(clientes.estcivils) AS Estado_Civil,
        CAST(clientes.nascs AS DATE) AS Data_Nascimento,
        DATEDIFF(YEAR, ISNULL(clientes.nascs, GETDATE()), GETDATE()) AS Idade,
        RTRIM(clientes.cidas) AS Cidade,
        RTRIM(clientes.estas) AS Estado,
        RTRIM(vendas.vends) AS Codigo_Vendedor,
        RTRIM(consultora.rclis) AS Nome_Vendedor,
        RTRIM(vendas.cpros) AS Codigo_Produto,
        RTRIM(prod.dpros) AS Descricao_Produto,
        ISNULL(estoque_atual.Estoque_Total, 0) AS Estoque_Atual,
        RTRIM(prod.colecoes) AS Colecao,
        RTRIM(grp.dgrus) AS Grupo_Produto,
        RTRIM(subgrp.descricaos) AS Subgrupo_Produto,
        RTRIM(prod.metals) AS Metal,
        SUM(CAST(vendas.qtds AS BIGINT)) AS Quantidade,
        CAST(SUM(vendas.custos) AS DECIMAL(10, 3)) AS Custo_Produto,
        CAST(SUM(vendas.totbrts) AS DECIMAL(10, 3)) AS Preco_Tabela,
        SUM(vendas.VALDESCS - vendas.VALRATS) AS Desconto_Aplicado,
        SUM(CAST(vendas.totas AS DECIMAL(10,3))) AS Total_Liquido
    FROM sljgdmi AS vendas WITH (NOLOCK)
        LEFT JOIN SLJCLI AS clientes ON clientes.ICLIS = vendas.ICLIS
        LEFT JOIN SLJCLI AS consultora ON consultora.ICLIS = vendas.VENDS
        LEFT JOIN sljpro AS prod WITH (NOLOCK) ON vendas.cpros = prod.cpros
        LEFT JOIN (
            SELECT cpros, SUM(CAST(sqtds AS INTEGER)) AS Estoque_Total
            FROM sljest WITH (NOLOCK)
            GROUP BY cpros
        ) AS estoque_atual ON estoque_atual.cpros = vendas.cpros
        LEFT JOIN sljgru AS grp WITH (NOLOCK) ON prod.cgrus = grp.cgrus
        LEFT JOIN sljsgru AS subgrp WITH (NOLOCK) ON prod.cgrus + prod.sgrus = subgrp.cgrucods
    WHERE vendas.ggrus IN (
        SELECT DISTINCT A.ggrus
        FROM SLJGDMI A
        JOIN SLJGGRP B ON A.ggrus = B.codigos AND B.relgers <> 2
    )
    -- <<FILTRO_DATA>>
    GROUP BY 
        CAST(vendas.datas AS DATE),
        YEAR(vendas.datas),
        MONTH(vendas.datas),
        RTRIM(vendas.iclis),
        RTRIM(vendas.rclis),
        RTRIM(clientes.sexos),
        RTRIM(clientes.estcivils),
        CAST(clientes.nascs AS DATE),
        RTRIM(clientes.cidas),
        RTRIM(clientes.estas),
        RTRIM(vendas.cpros),
        RTRIM(prod.dpros),
        RTRIM(prod.colecoes),
        RTRIM(grp.dgrus),
        RTRIM(subgrp.descricaos),
        RTRIM(prod.metals),
        RTRIM(vendas.vends),
        RTRIM(consultora.rclis),
        estoque_atual.Estoque_Total,
        clientes.nascs
    ORDER BY RTRIM(grp.dgrus), YEAR(vendas.datas) DESC, MONTH(vendas.datas) DESC, CAST(vendas.datas AS DATE) DESC 
    """

    def _run(self, date_start: str, date_end: Optional[str] = None, output_format: str = "csv", max_records: Optional[int] = None) -> str:
        print(f"🔍 SQL Query Tool executando com parâmetros:")
        print(f"   📅 Data início: {date_start}")
        print(f"   📅 Data fim: {date_end}")
        print(f"   📋 Formato: {output_format}")
        if max_records:
            print(f"   🔢 Limite: {max_records:,} registros")
        
        # Validar datas
        try:
            start_date = datetime.strptime(date_start, '%Y-%m-%d')
            if date_end:
                end_date = datetime.strptime(date_end, '%Y-%m-%d')
                if end_date < start_date:
                    return "❌ Erro: Data final deve ser posterior à data inicial."
            print("✅ Formato de datas validado")
        except ValueError:
            return "❌ Erro: As datas devem estar no formato 'YYYY-MM-DD'."
            
        # Construir a cláusula de filtro de data
        if date_end:
            date_filter = f"AND vendas.datas BETWEEN '{date_start}' AND '{date_end}'"
            print(f"🔍 Filtro SQL criado: BETWEEN {date_start} AND {date_end}")
        else:
            date_filter = f"AND vendas.datas = '{date_start}'"
            print(f"🔍 Filtro SQL criado: = {date_start}")
            
        # Detalhes da conexão com o SQL Server
        conn_str = (
            f"DRIVER={{{self.DB_DRIVER}}};"
            f"SERVER={self.DB_SERVER},{self.DB_PORT};"
            f"DATABASE={self.DB_DATABASE};"
            f"UID={self.DB_UID};"
            f"PWD={self.DB_PWD};"
        )
        
        # Substituir o placeholder pelo filtro de data real
        sql_query = self.SQL_QUERY.replace('-- <<FILTRO_DATA>>', date_filter)
        
        # Aplicar limite de registros se especificado
        if max_records:
            # Adicionar TOP clause após SELECT
            sql_query = sql_query.replace('SELECT', f'SELECT TOP {max_records}', 1)
            print(f"✅ Limite de {max_records:,} registros aplicado na query")
        
        print(f"✅ Placeholder -- <<FILTRO_DATA>> substituído por: {date_filter}")
        
        try:
            print("🔌 Conectando ao SQL Server...")
            start_time = time.time()
            
            # Conectar ao SQL Server com timeout maior
            conn = pyodbc.connect(conn_str, timeout=120)
            
            connection_time = time.time() - start_time
            print(f"✅ Conexão estabelecida em {connection_time:.2f} segundos")
            
            # Executar a consulta diretamente com pandas (mais simples e estável)
            print("⏰ Executando consulta SQL (pode demorar alguns segundos)...")
            
            query_start_time = time.time()
            df = pd.read_sql(sql_query, conn)
            
            query_exec_time = time.time() - query_start_time
            print(f"✅ Query executada e processada em {query_exec_time:.2f} segundos")
            
            conn.close()
            
            total_time = time.time() - start_time
            print(f"🎉 Operação completa em {total_time:.2f} segundos")
            
            # Verificar se obtivemos resultados
            if not df.empty:
                print(f"📊 Total de {len(df)} registros extraídos")
                
                # Formatar a saída com base no formato solicitado
                if output_format == "summary":
                    return self._format_summary(df, date_start, date_end)
                elif output_format == "raw":
                    return f"Recuperados {len(df)} registros.\n\n{df.head(20).to_string()}"
                elif output_format == "json":
                    return f"Recuperados {len(df)} registros.\n\n{df.to_json(orient='records')}"
                elif output_format == "csv":
                    # Para datasets grandes, salvar em arquivo ao invés de retornar string
                    if len(df) > 1000:  # Se mais de 50k registros
                        # Sempre salvar em data/vendas.csv para padronização
                        os.makedirs('data', exist_ok=True)
                        filename = 'data/vendas.csv'
                        df.to_csv(filename, index=False, sep=';', encoding='utf-8')
                        
                        # Retornar apenas confirmação sem amostra para datasets grandes
                        return (
                            f"✅ DADOS EXTRAÍDOS COM SUCESSO!\n"
                            f"📊 Total: {len(df):,} registros de vendas ({date_start} a {date_end})\n"
                            f"💾 Arquivo salvo em: {filename}\n"
                            f"📁 Tamanho do dataset: {len(df):,} registros\n"
                            f"📋 Colunas disponíveis: {len(df.columns)} colunas\n"
                            f"📋 Principais colunas: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}\n\n"
                            f"🎯 Dataset pronto para análise pelos próximos agentes!\n"
                            f"💡 ATENÇÃO: Arquivo padronizado salvo em 'data/vendas.csv'\n"
                            f"💡 Todos os agentes devem ler deste arquivo padrão."
                        )
                    else:
                        # Para datasets pequenos, também salvar no arquivo padrão
                        os.makedirs('data', exist_ok=True)
                        filename = 'data/vendas.csv'
                        df.to_csv(filename, index=False, sep=';', encoding='utf-8')
                        
                        csv_result = df.to_csv(index=False, sep=';', encoding='utf-8')
                        print(f"✅ CSV gerado com {len(csv_result)} caracteres e salvo em {filename}")
                        return (
                            f"✅ DADOS EXTRAÍDOS COM SUCESSO!\n"
                            f"📊 Total: {len(df):,} registros de vendas ({date_start} a {date_end})\n"
                            f"💾 Arquivo salvo em: {filename}\n"
                            f"🎯 Dataset pronto para análise pelos próximos agentes!\n"
                            f"💡 ATENÇÃO: Arquivo padronizado salvo em 'data/vendas.csv'\n"
                            f"💡 Todos os agentes devem ler deste arquivo padrão.\n\n"
                            f"Dados em formato CSV:\n\n{csv_result[:1000]}{'...' if len(csv_result) > 1000 else ''}"
                        )
                else:
                    return f"Formato de saída não suportado: '{output_format}'"
            else:
                return "A consulta foi executada com sucesso mas não retornou resultados."
            
        except pyodbc.OperationalError as e:
            if "timeout" in str(e).lower():
                print("⏰ TIMEOUT detectado na consulta SQL")
                return f"❌ Timeout na consulta SQL: A query demorou mais que o esperado. Tente reduzir o período de análise."
            else:
                print(f"❌ Erro operacional SQL: {e}")
                return f"❌ Erro operacional SQL: {str(e)}"
                
        except Exception as e:
            print(f"❌ Erro geral: {e}")
            return f"❌ Erro ao executar consulta SQL: {str(e)}"
    
    def _format_summary(self, df: pd.DataFrame, date_start: str, date_end: Optional[str] = None) -> str:
        """Formatar um resumo dos resultados."""
        records_count = len(df)
        date_range = f"para {date_start}" if not date_end else f"de {date_start} até {date_end}"
        
        # Obter estatísticas gerais
        total_quantity = df['Quantidade'].sum()
        total_value = df['Total_Liquido'].sum()
        
        # Agrupar por grupo de produto para obter um resumo
        if 'Grupo_Produto' in df.columns:
            group_summary = df.groupby('Grupo_Produto').agg({
                'Quantidade': 'sum',
                'Total_Liquido': 'sum'
            }).reset_index()
            
            group_summary_str = "\n".join([
                f"- {row['Grupo_Produto']}: {row['Quantidade']} unidades, R$ {row['Total_Liquido']:.2f}"
                for _, row in group_summary.iterrows()
            ])
        else:
            group_summary_str = "Informações de grupo de produto não disponíveis."
        
        # Retornar um resumo formatado
        return (
            f"Recuperados {records_count} registros de vendas {date_range}.\n\n"
            f"Quantidade Total: {total_quantity}\n"
            f"Valor Total: R$ {total_value:.2f}\n\n"
            f"Resumo por Grupo de Produto:\n{group_summary_str}\n\n"
            f"Amostra de dados (primeiras 5 linhas):\n{df.head(5).to_string()}"
        )

    def _save_to_csv(self, df: pd.DataFrame, filename: str):
        # Change to CSV
        df.to_csv(filename, index=False, sep=';', encoding='utf-8')
    
    # crie uma função que execute a query e salve o resultado em um arquivo csv
    def _execute_query_and_save_to_csv(self):
        
        print("Executando consulta SQL e salvando em arquivo CSV...")
        date_end = datetime.now().strftime('%Y-%m-%d')
        date_start = (datetime.now() - timedelta(days=1460)).strftime('%Y-%m-%d')
        date_filter = f"AND vendas.datas BETWEEN '{date_start}' AND '{date_end}'"
        
        print("Data inicial:", date_start, "Data final:", date_end)
        
        filename = 'data/vendas.csv'
        sql_query = self.SQL_QUERY.replace('-- <<FILTRO_DATA>>', date_filter)
              
        # Detalhes da conexão com o SQL Server
        conn_str = (
            f"DRIVER={{{self.DB_DRIVER}}};"
            f"SERVER={self.DB_SERVER},{self.DB_PORT};"
            f"DATABASE={self.DB_DATABASE};"
            f"UID={self.DB_UID};"
            f"PWD={self.DB_PWD};"
        )
        
        print("Conectando ao SQL Server...", conn_str)
        
        # Conectar ao SQL Server
        conn = pyodbc.connect(conn_str)
        
        print("Executando consulta SQL...")
        
        # Executar a consulta e buscar resultados
        df = pd.read_sql(sql_query, conn)
        
        print("Consulta executada com sucesso!")
        
        # Salvar o resultado em um arquivo csv
        print("Salvando resultado em arquivo CSV...")
        df.to_csv(filename, index=False, sep=';', encoding='utf-8')
        
        print(f"Resultado salvo em {filename}")
        
        # Fechar a conexão
        conn.close()
        print("Conexão fechada com sucesso!")

if __name__ == "__main__":
    sql_tool = SQLServerQueryTool()
    sql_tool._execute_query_and_save_to_csv()