from crewai.tools import BaseTool
from typing import Type, Optional, ClassVar
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timedelta
import pyodbc
import pandas as pd
import os
import logging
import time
import signal
import threading
from contextlib import contextmanager

# Configurar logger específico para a tool
tool_logger = logging.getLogger('sql_query_tool')
tool_logger.setLevel(logging.DEBUG)

class TimeoutError(Exception):
    """Exceção customizada para timeout"""
    pass

@contextmanager
def timeout_context(seconds):
    """Context manager para timeout de operações"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operação excedeu timeout de {seconds} segundos")
    
    # Configurar handler de timeout (funciona apenas no Unix/Linux)
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

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

class SQLServerQueryToolImproved(BaseTool):
    """
    🗄️ FERRAMENTA DE CONSULTA SQL SERVER MELHORADA COM TIMEOUTS E LOGS
    
    MELHORIAS:
    - Timeouts configuráveis para conexão e query
    - Logs detalhados de progresso
    - Tratamento de erro granular
    - Fallback para dados existentes
    - Validação de conectividade antes da query
    """
    
    name: str = "SQL Server Query Tool Improved"
    description: str = (
        "Ferramenta especializada para extrair dados de vendas do SQL Server com timeouts e logs detalhados. "
        "Versão melhorada com tratamento robusto de erros e fallbacks automáticos."
    )
    args_schema: Type[BaseModel] = SQLServerQueryInput
    
    # Parâmetros de conexão
    DB_DRIVER: str = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
    DB_SERVER: str = os.getenv("DB_SERVER", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "1433")
    DB_DATABASE: str = os.getenv("DB_DATABASE", "default_db")
    DB_UID: str = os.getenv("DB_UID", "default_user")
    DB_PWD: str = os.getenv("DB_PWD", "default_password")
    
    # Timeouts
    CONNECTION_TIMEOUT: int = 30  # segundos
    QUERY_TIMEOUT: int = 60      # 1 minuto (query demora ~23s)
    
    # Template da consulta SQL (mesmo da versão original)
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
    
    def test_connection(self) -> bool:
        """Testar conectividade antes de executar query"""
        tool_logger.info("🔌 Testando conectividade com SQL Server...")
        
        try:
            conn_str = self._build_connection_string()
            tool_logger.debug(f"🔗 String de conexão: SERVER={self.DB_SERVER}:{self.DB_PORT}, DB={self.DB_DATABASE}")
            
            start_time = time.time()
            conn = pyodbc.connect(conn_str, timeout=self.CONNECTION_TIMEOUT)
            
            connection_time = time.time() - start_time
            tool_logger.info(f"✅ Conexão estabelecida em {connection_time:.2f}s")
            
            # Testar query simples - corrigir sintaxe SQL Server
            cursor = conn.cursor()
            cursor.execute("SELECT GETDATE() AS CurrentTime")
            result = cursor.fetchone()
            tool_logger.info(f"✅ Query teste executada: {result[0]}")
            
            conn.close()
            return True
            
        except Exception as e:
            tool_logger.error(f"❌ Falha na conectividade: {e}")
            return False
    
    def _build_connection_string(self) -> str:
        """Construir string de conexão"""
        return (
            f"DRIVER={{{self.DB_DRIVER}}};"
            f"SERVER={self.DB_SERVER},{self.DB_PORT};"
            f"DATABASE={self.DB_DATABASE};"
            f"UID={self.DB_UID};"
            f"PWD={self.DB_PWD};"
        )
    
    def _run(self, date_start: str, date_end: Optional[str] = None, output_format: str = "csv") -> str:
        tool_logger.info("🚀 SQL Query Tool Improved executando...")
        tool_logger.info(f"📅 Parâmetros: {date_start} até {date_end or date_start}, formato: {output_format}")
        
        # Etapa 1: Validar datas
        tool_logger.info("📋 ETAPA 1: Validando datas...")
        try:
            start_date = datetime.strptime(date_start, '%Y-%m-%d')
            if date_end:
                end_date = datetime.strptime(date_end, '%Y-%m-%d')
                if end_date < start_date:
                    return "❌ Erro: Data final deve ser posterior à data inicial."
            tool_logger.info("✅ Formato de datas validado")
        except ValueError as e:
            tool_logger.error(f"❌ Erro na validação de datas: {e}")
            return "❌ Erro: As datas devem estar no formato 'YYYY-MM-DD'."
        
        # Etapa 2: Testar conectividade
        tool_logger.info("🔌 ETAPA 2: Testando conectividade...")
        if not self.test_connection():
            tool_logger.error("❌ Falha na conectividade - tentando fallback")
            return self._try_fallback(date_start, date_end, output_format)
        
        # Etapa 3: Preparar query
        tool_logger.info("📝 ETAPA 3: Preparando query SQL...")
        if date_end:
            date_filter = f"AND vendas.datas BETWEEN '{date_start}' AND '{date_end}'"
        else:
            date_filter = f"AND vendas.datas = '{date_start}'"
        
        sql_query = self.SQL_QUERY.replace('-- <<FILTRO_DATA>>', date_filter)
        tool_logger.info(f"✅ Filtro de data aplicado: {date_filter}")
        
        # Etapa 4: Executar query com timeout
        tool_logger.info("🔄 ETAPA 4: Executando query SQL...")
        try:
            return self._execute_query_with_timeout(sql_query, output_format, date_start, date_end)
        except Exception as e:
            tool_logger.error(f"❌ Erro na execução da query: {e}")
            return self._try_fallback(date_start, date_end, output_format)
    
    def _progress_monitor(self, start_time: float, stop_event: threading.Event):
        """Monitor de progresso para queries longas"""
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            tool_logger.info(f"⏱️ Query executando há {elapsed:.0f}s...")
            stop_event.wait(5)  # Log a cada 5 segundos
    
    def _execute_query_with_timeout(self, sql_query: str, output_format: str, date_start: str, date_end: Optional[str]) -> str:
        """Executar query com timeout e logs detalhados"""
        conn_str = self._build_connection_string()
        
        try:
            # Conectar com timeout
            tool_logger.info("🔗 Conectando ao SQL Server...")
            start_time = time.time()
            conn = pyodbc.connect(conn_str, timeout=self.CONNECTION_TIMEOUT)
            
            connection_time = time.time() - start_time
            tool_logger.info(f"✅ Conectado em {connection_time:.2f}s")
            
            # Executar query com timeout e progress monitoring
            tool_logger.info("📊 Executando query principal (pode demorar até 1 minuto)...")
            query_start = time.time()
            
            # Iniciar monitor de progresso
            stop_event = threading.Event()
            progress_thread = threading.Thread(
                target=self._progress_monitor, 
                args=(query_start, stop_event),
                daemon=True
            )
            progress_thread.start()
            
            try:
                # Configurar timeout da query
                conn.timeout = self.QUERY_TIMEOUT
                df = pd.read_sql(sql_query, conn)
            finally:
                # Parar monitor de progresso
                stop_event.set()
            
            query_time = time.time() - query_start
            tool_logger.info(f"✅ Query executada em {query_time:.2f}s")
            tool_logger.info(f"📊 Registros retornados: {len(df)}")
            
            # Fechar conexão
            conn.close()
            tool_logger.info("🔒 Conexão fechada")
            
            # Verificar se obtivemos resultados
            if df.empty:
                tool_logger.warning("⚠️ Query executada mas sem resultados")
                return "A consulta foi executada com sucesso mas não retornou resultados."
            
            # Formatar saída
            tool_logger.info(f"📋 Formatando saída como {output_format}...")
            return self._format_output(df, output_format, date_start, date_end)
            
        except Exception as e:
            tool_logger.error(f"❌ Erro na execução: {e}")
            raise
    
    def _format_output(self, df: pd.DataFrame, output_format: str, date_start: str, date_end: Optional[str]) -> str:
        """Formatar saída baseado no formato solicitado"""
        records_count = len(df)
        
        if output_format == "summary":
            return self._format_summary(df, date_start, date_end)
        elif output_format == "raw":
            return f"Recuperados {records_count} registros.\n\n{df.head(20).to_string()}"
        elif output_format == "json":
            return f"Recuperados {records_count} registros.\n\n{df.to_json(orient='records')}"
        elif output_format == "csv":
            return f"Recuperados {records_count} registros.\n\n{df.to_csv(index=False, sep=';', encoding='utf-8')}"
        else:
            return f"Formato de saída não suportado: '{output_format}'"
    
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
        
        return (
            f"Recuperados {records_count} registros de vendas {date_range}.\n\n"
            f"Quantidade Total: {total_quantity}\n"
            f"Valor Total: R$ {total_value:.2f}\n\n"
            f"Resumo por Grupo de Produto:\n{group_summary_str}\n\n"
            f"Amostra de dados (primeiras 5 linhas):\n{df.head(5).to_string()}"
        )
    
    def _try_fallback(self, date_start: str, date_end: Optional[str], output_format: str) -> str:
        """Tentar fallback para dados existentes"""
        tool_logger.info("🔄 Tentando fallback para dados existentes...")
        
        fallback_file = "data/vendas.csv"
        if os.path.exists(fallback_file):
            try:
                tool_logger.info(f"📂 Carregando dados de {fallback_file}...")
                df = pd.read_csv(fallback_file, sep=';', encoding='utf-8')
                
                # Filtrar por data se possível
                if 'Data' in df.columns:
                    df['Data'] = pd.to_datetime(df['Data'])
                    start_date = datetime.strptime(date_start, '%Y-%m-%d')
                    
                    if date_end:
                        end_date = datetime.strptime(date_end, '%Y-%m-%d')
                        df_filtered = df[(df['Data'] >= start_date) & (df['Data'] <= end_date)]
                    else:
                        df_filtered = df[df['Data'] == start_date]
                    
                    tool_logger.info(f"✅ Dados filtrados: {len(df_filtered)} registros")
                    return f"📂 DADOS DE FALLBACK (filtrados para {date_start} até {date_end or date_start}):\n\n" + \
                           self._format_output(df_filtered, output_format, date_start, date_end)
                else:
                    tool_logger.info(f"✅ Dados carregados: {len(df)} registros (sem filtro de data)")
                    return f"📂 DADOS DE FALLBACK (sem filtro de data):\n\n" + \
                           self._format_output(df, output_format, date_start, date_end)
                           
            except Exception as e:
                tool_logger.error(f"❌ Erro no fallback: {e}")
        
        # Último recurso
        tool_logger.error("❌ Todos os fallbacks falharam")
        return (
            f"❌ ERRO: Não foi possível conectar ao SQL Server e não há dados de fallback disponíveis.\n"
            f"Tentativa de período: {date_start} até {date_end or date_start}\n"
            f"Verifique a conectividade com o banco de dados e tente novamente."
        ) 