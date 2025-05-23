from crewai.tools import BaseTool
from typing import Type, Optional, ClassVar
from pydantic import BaseModel, Field
import pyodbc
import pandas as pd
from datetime import datetime, timedelta

class SQLServerQueryInput(BaseModel):
    """Schema de entrada para a ferramenta SQL Server Query."""
    date_start: str = Field(..., description="Data inicial para o filtro no formato 'YYYY-MM-DD'.")
    date_end: Optional[str] = Field(None, description="Data final para o filtro no formato 'YYYY-MM-DD'. Se não for fornecida, apenas a data inicial será usada.")
    output_format: Optional[str] = Field("summary", description="Formato da saída: 'summary' para um resumo, 'raw' para os dados brutos, 'json' ou 'csv' para exportação."
    )

class SQLServerQueryTool(BaseTool):
    name: str = "SQL Server Query Tool"
    description: str = (
        "Executa uma consulta SQL no SQL Server com um filtro de data dinâmico. "
        "Forneça uma data inicial e, opcionalmente, uma data final para obter vendas nesse período."
    )
    args_schema: Type[BaseModel] = SQLServerQueryInput
    
    # Parâmetros de conexão com o banco de dados
    DB_DRIVER: ClassVar[str] = "ODBC Driver 17 for SQL Server"
    DB_SERVER: ClassVar[str] = "192.168.28.196"
    DB_PORT: ClassVar[str] = "1433"
    DB_DATABASE: ClassVar[str] = "005RG_ERP_BI"
    DB_UID: ClassVar[str] = "captabi"
    DB_PWD: ClassVar[str] = "Sox@25bi"
    
    # Template da consulta SQL
    SQL_QUERY: ClassVar[str] = """
    SELECT
        CAST(vendas.datas AS DATE) AS Data,
        YEAR(vendas.datas) AS Ano,
        MONTH(vendas.datas) AS Mes,
        RTRIM(vendas.cpros) AS Codigo_Produto,
        RTRIM(prod.dpros) AS Descricao_Produto,
        RTRIM(prod.colecoes) AS Colecao,
        RTRIM(grp.dgrus) AS Grupo_Produto,
        RTRIM(subgrp.descricaos) AS Subgrupo_Produto,
        RTRIM(prod.metals) AS Metal,
        SUM(CAST(vendas.qtds AS BIGINT)) AS Quantidade,
        SUM(CAST(vendas.totas AS DECIMAL(10,3))) AS Total_Liquido
    FROM sljgdmi AS vendas WITH (NOLOCK)
        LEFT JOIN sljpro AS prod WITH (NOLOCK) ON vendas.cpros = prod.cpros
        LEFT JOIN sljgru AS grp WITH (NOLOCK) ON prod.cgrus = grp.cgrus
        LEFT JOIN sljsgru AS subgrp WITH (NOLOCK) ON prod.cgrus + prod.sgrus = subgrp.cgrucods
    WHERE vendas.ggrus IN (
        SELECT DISTINCT A.ggrus
        FROM SLJGDMI A
        JOIN SLJGGRP B
        ON A.ggrus = B.codigos AND B.relgers <> 2
    )
    -- <<FILTRO_DATA>>
    GROUP BY 
        CAST(vendas.datas AS DATE),
        YEAR(vendas.datas),
        MONTH(vendas.datas),
        RTRIM(vendas.cpros),
        RTRIM(prod.dpros),
        RTRIM(prod.colecoes),
        RTRIM(grp.dgrus),
        RTRIM(subgrp.descricaos),
        RTRIM(prod.metals)
    ORDER BY RTRIM(grp.dgrus), YEAR(vendas.datas) DESC, MONTH(vendas.datas) DESC, CAST(vendas.datas AS DATE) DESC 
    """

    def _run(self, date_start: str, date_end: Optional[str] = None, output_format: str = "csv") -> str:
        # Validar datas
        try:
            start_date = datetime.strptime(date_start, '%Y-%m-%d')
            if date_end:
                end_date = datetime.strptime(date_end, '%Y-%m-%d')
                if end_date < start_date:
                    return "Erro: Data final deve ser posterior à data inicial."
        except ValueError:
            return "Erro: As datas devem estar no formato 'YYYY-MM-DD'."
            
        # Construir a cláusula de filtro de data
        if date_end:
            date_filter = f"AND vendas.datas BETWEEN '{date_start}' AND '{date_end}'"
        else:
            date_filter = f"AND vendas.datas = '{date_start}'"
            
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
        
        try:
            # Conectar ao SQL Server
            conn = pyodbc.connect(conn_str)
            
            # Executar a consulta e buscar resultados
            df = pd.read_sql(sql_query, conn)
            
            # Fechar a conexão
            conn.close()
            
            # Verificar se obtivemos resultados
            if df.empty:
                return "A consulta foi executada com sucesso mas não retornou resultados."
            
            # Formatar a saída com base no formato solicitado
            if output_format == "summary":
                return self._format_summary(df, date_start, date_end)
            elif output_format == "raw":
                return f"Recuperados {len(df)} registros.\n\n{df.head(20).to_string()}"
            elif output_format == "json":
                return f"Recuperados {len(df)} registros.\n\n{df.to_json(orient='records')}"
            elif output_format == "csv":
                return f"Recuperados {len(df)} registros.\n\n{df.to_csv(index=False, sep=';', encoding='utf-8')}"
            else:
                return f"Formato de saída não suportado: '{output_format}'"
            
        except Exception as e:
            return f"Erro ao executar consulta SQL: {str(e)}"
    
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
        date_start = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
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
        
        # Salvar o resultado em um arquivo csv
        df.to_csv(filename, index=False, sep=';', encoding='utf-8')
        
        # Fechar a conexão
        conn.close()

if __name__ == "__main__":
    sql_tool = SQLServerQueryTool()
    sql_tool._execute_query_and_save_to_csv()