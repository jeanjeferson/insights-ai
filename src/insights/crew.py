from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import FileReadTool
from typing import List
from dotenv import load_dotenv
import os
import logging
import time
from datetime import datetime
from pathlib import Path

# =============== IMPORTAÇÕES OTIMIZADAS DE FERRAMENTAS ===============

# Ferramentas básicas
from insights.tools.sql_query_tool import SQLServerQueryTool
from insights.tools.kpi_calculator_tool import KPICalculatorTool
from insights.tools.prophet_tool import ProphetForecastTool
from insights.tools.statistical_analysis_tool import StatisticalAnalysisTool
from insights.tools.business_intelligence_tool import BusinessIntelligenceTool
from insights.tools.duckduck_tool import DuckDuckGoSearchTool

# Ferramentas avançadas
from insights.tools.advanced.customer_insights_engine import CustomerInsightsEngine
from insights.tools.advanced.recommendation_engine import RecommendationEngine
from insights.tools.advanced.advanced_analytics_engine_tool import AdvancedAnalyticsEngineTool
from insights.tools.advanced.risk_assessment_tool import RiskAssessmentTool
from insights.tools.advanced.competitive_intelligence_tool import CompetitiveIntelligenceTool

load_dotenv()

# =============== CONFIGURAÇÃO AVANÇADA DE LOGGING ===============

def setup_crew_file_logging():
    """
    Configura logging em tempo real para arquivo com rotação por execução
    """
    # Criar timestamp para nome único do arquivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Garantir que a pasta logs existe
    logs_dir = Path("logs/crew_executions")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Caminho do arquivo de log
    log_file = logs_dir / f"crew_execution_{timestamp}.log"
    
    # Configurar logger específico para o crew
    crew_logger = logging.getLogger('crew_insights')
    crew_logger.setLevel(logging.DEBUG)
    
    # Remover handlers existentes para evitar duplicação
    for handler in crew_logger.handlers[:]:
        crew_logger.removeHandler(handler)
    
    # Handler para arquivo com flush imediato
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Handler para console (manter visualização)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatação rica para arquivo
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Formatação simples para console
    console_formatter = logging.Formatter(
        '%(asctime)s - %(process)d - %(filename)s-%(funcName)s:%(lineno)d - %(levelname)s: %(message)s'
    )
    
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Adicionar handlers
    crew_logger.addHandler(file_handler)
    crew_logger.addHandler(console_handler)
    
    # Garantir que não propague para o logger raiz
    crew_logger.propagate = False
    
    # Log inicial de teste
    crew_logger.info("=" * 80)
    crew_logger.info("🚀 INSIGHTS-AI CREW - LOGGING INICIADO")
    crew_logger.info(f"📁 Arquivo de log: {log_file}")
    crew_logger.info(f"🕒 Execução iniciada em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    crew_logger.info("=" * 80)
    
    return crew_logger, str(log_file)

# Configurar logging específico para o crew
crew_logger = logging.getLogger('crew_insights')
crew_logger.setLevel(logging.DEBUG)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

llm = LLM(
    model="openrouter/deepseek/deepseek-r1",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# =============== INSTANCIAÇÃO COMPLETA DE FERRAMENTAS ===============

# Ferramentas básicas
file_tool = FileReadTool()
sql_tool = SQLServerQueryTool()
search_tool = DuckDuckGoSearchTool()

# Ferramentas de análise
kpi_tool = KPICalculatorTool()
prophet_tool = ProphetForecastTool()
stats_tool = StatisticalAnalysisTool()
bi_tool = BusinessIntelligenceTool()

# Ferramentas avançadas de IA/ML
customer_engine = CustomerInsightsEngine()
recommendation_engine = RecommendationEngine()
analytics_engine = AdvancedAnalyticsEngineTool()
risk_tool = RiskAssessmentTool()
competitive_tool = CompetitiveIntelligenceTool()

@CrewBase
class Insights():
    """
    Crew otimizada com distribuição inteligente de ferramentas:
    - Validação automática de ferramentas
    - Rate limiting inteligente  
    - Monitoramento de performance
    - Fallbacks para ferramentas indisponíveis
    - Distribuição especializada por agente
    """

    def __init__(self):
        super().__init__()
        
        # =============== CONFIGURAR LOGGING EM ARQUIVO ===============
        try:
            global crew_logger
            crew_logger, self.log_file_path = setup_crew_file_logging()
            crew_logger.info("✅ Sistema de logging em arquivo configurado com sucesso")
        except Exception as e:
            print(f"⚠️ WARNING: Erro ao configurar logging em arquivo: {e}")
            # Continuar com logging apenas no console
            self.log_file_path = None
        
        # Só validar ferramentas se em modo debug
        if os.getenv("INSIGHTS_DEBUG", "false").lower() == "true":
            self.tools_status = validate_tools_setup()
        else:
            self.tools_status = validate_tools_setup_quiet()
        self.task_start_times = {}
        self.setup_logging_callbacks()
        
        # Log de informações do sistema
        self._log_system_info()

    agents: List[BaseAgent]
    tasks: List[Task]
    
    @before_kickoff
    def before_kickoff(self, inputs):
        """Before kickoff otimizado com validações e inputs de data"""
        start_time = time.time()
        
        # =============== LOG METADADOS DA EXECUÇÃO ===============
        crew_logger.info("🚀 INICIANDO INSIGHTS-AI OTIMIZADO")
        crew_logger.info("=" * 60)
        crew_logger.info(f"📅 Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        crew_logger.info(f"📋 Inputs recebidos: {inputs}")
        crew_logger.info(f"📁 Log sendo salvo em: {getattr(self, 'log_file_path', 'APENAS CONSOLE')}")
        crew_logger.info("=" * 60)
        
        # Validar e exibir inputs de data
        data_inicio = inputs.get('data_inicio')
        data_fim = inputs.get('data_fim')
        
        if data_inicio and data_fim:
            crew_logger.info(f"📅 Período de análise: {data_inicio} até {data_fim}")
            
            # Validar formato das datas
            try:
                datetime.strptime(data_inicio, '%Y-%m-%d')
                datetime.strptime(data_fim, '%Y-%m-%d')
                crew_logger.info("✅ Formato de datas validado com sucesso")
            except ValueError as e:
                crew_logger.warning(f"⚠️ WARNING: Formato de data inválido! Use YYYY-MM-DD. Erro: {e}")
        else:
            crew_logger.warning("⚠️ WARNING: Inputs de data não fornecidos! Usando dados padrão.")
        
        # Validar ferramentas críticas com mais detalhes
        crew_logger.info("🔧 VALIDANDO FERRAMENTAS CRÍTICAS:")
        critical_tools = ['sql_tool', 'prophet_tool', 'stats_tool', 'bi_tool']
        tool_status = {}
        
        for tool_name in critical_tools:
            tool_obj = globals().get(tool_name)
            if tool_obj is None:
                crew_logger.error(f"   ❌ {tool_name} NÃO ENCONTRADA!")
                tool_status[tool_name] = "❌ FALHA"
            else:
                crew_logger.info(f"   ✅ {tool_name} carregada - Tipo: {type(tool_obj).__name__}")
                tool_status[tool_name] = "✅ OK"
        
        # Resumo do status das ferramentas
        working_tools = sum(1 for status in tool_status.values() if "✅" in status)
        crew_logger.info(f"📊 Status geral: {working_tools}/{len(critical_tools)} ferramentas críticas funcionando")
        
        # Executar SQL extraction com as datas fornecidas (se disponíveis)
        crew_logger.info("📊 PREPARANDO EXTRAÇÃO DE DADOS:")
        try:
            if data_inicio and data_fim:
                crew_logger.info(f"   🔄 Modo: Extração com filtro temporal {data_inicio} a {data_fim}")
                crew_logger.info("   📋 Dados serão extraídos pelo agente usando os inputs fornecidos")
            else:
                crew_logger.info("   🔄 Modo: Extração padrão (sem filtro temporal)")
                crew_logger.info("   ⚠️ Executando extração padrão...")
                sql_tool._execute_query_and_save_to_csv()
                crew_logger.info("   ✅ Dados extraídos com sucesso do SQL Server (período padrão)")
        except Exception as e:
            crew_logger.error(f"   ❌ ERRO na extração SQL: {e}")
            crew_logger.info("   🔄 Tentando usar dados existentes como fallback...")
        
        setup_time = time.time() - start_time
        crew_logger.info("=" * 60)
        crew_logger.info(f"⏱️ SETUP CONCLUÍDO em {setup_time:.2f} segundos")
        crew_logger.info("🚀 INICIANDO EXECUÇÃO DO CREW...")
        crew_logger.info("=" * 60)
        
        # Forçar flush para garantir que tudo seja escrito
        self._flush_logs()
        
        return inputs
    
    def _log_system_info(self):
        """Log informações do sistema e ambiente para debug"""
        try:
            import platform
            import sys
            
            crew_logger.info("🖥️ INFORMAÇÕES DO SISTEMA:")
            crew_logger.info(f"   • OS: {platform.system()} {platform.release()}")
            crew_logger.info(f"   • Python: {sys.version.split()[0]}")
            crew_logger.info(f"   • Working Directory: {os.getcwd()}")
            crew_logger.info(f"   • Process ID: {os.getpid()}")
            
            # Log variáveis de ambiente relevantes (sem expor segredos)
            env_vars = ['INSIGHTS_DEBUG', 'OPENROUTER_API_KEY']
            crew_logger.info("🔑 VARIÁVEIS DE AMBIENTE:")
            for var in env_vars:
                value = os.getenv(var)
                if value:
                    if 'API_KEY' in var or 'TOKEN' in var:
                        masked_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
                        crew_logger.info(f"   • {var}: {masked_value}")
                    else:
                        crew_logger.info(f"   • {var}: {value}")
                else:
                    crew_logger.warning(f"   • {var}: NÃO DEFINIDA")
            
            crew_logger.info("-" * 50)
            
        except Exception as e:
            crew_logger.error(f"❌ Erro ao capturar informações do sistema: {e}")

    def setup_logging_callbacks(self):
        """Configurar callbacks de logging para monitoramento detalhado"""
        crew_logger.info("🔧 Configurando callbacks de monitoramento...")
        
        # Log da configuração dos handlers ativos
        active_handlers = [type(h).__name__ for h in crew_logger.handlers]
        crew_logger.info(f"   • Handlers ativos: {active_handlers}")
        
        if hasattr(self, 'log_file_path') and self.log_file_path:
            crew_logger.info(f"   • Arquivo de log: {self.log_file_path}")
        
        # Forçar flush dos handlers de arquivo
        for handler in crew_logger.handlers:
            if hasattr(handler, 'flush'):
                handler.flush()
    
    def log_task_start(self, task_name: str):
        """Log início de task com timestamp"""
        self.task_start_times[task_name] = time.time()
        crew_logger.info(f"🚀 INICIANDO TASK: {task_name}")
        crew_logger.info(f"⏰ Timestamp: {datetime.now().strftime('%H:%M:%S')}")
        self._flush_logs()
    
    def log_task_progress(self, task_name: str, message: str):
        """Log progresso da task"""
        elapsed = time.time() - self.task_start_times.get(task_name, time.time())
        crew_logger.info(f"🔄 [{task_name}] {message} (elapsed: {elapsed:.1f}s)")
        self._flush_logs()
    
    def log_task_end(self, task_name: str):
        """Log fim de task com tempo total"""
        elapsed = time.time() - self.task_start_times.get(task_name, time.time())
        crew_logger.info(f"✅ CONCLUÍDA: {task_name} em {elapsed:.2f} segundos")
        self._flush_logs()
        
    def log_agent_action(self, agent_name: str, action: str, details: str = ""):
        """Log ações dos agentes"""
        crew_logger.info(f"🧠 [{agent_name}] {action} {details}")
        self._flush_logs()
    
    def log_tool_usage(self, agent_name: str, tool_name: str, status: str = "chamada"):
        """Log uso de ferramentas"""
        crew_logger.info(f"🔧 [{agent_name}] Ferramenta {tool_name} - {status}")
        self._flush_logs()

    def _flush_logs(self):
        """Força o flush de todos os handlers de log para escrita imediata"""
        try:
            for handler in crew_logger.handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()
        except Exception as e:
            print(f"⚠️ Erro ao fazer flush dos logs: {e}")

    def _log_execution_summary(self):
        """Log resumo final da execução"""
        try:
            crew_logger.info("=" * 80)
            crew_logger.info("📊 RESUMO FINAL DA EXECUÇÃO")
            crew_logger.info("=" * 80)
            
            # Tempo total de execução
            if hasattr(self, 'execution_start_time'):
                total_time = time.time() - self.execution_start_time
                crew_logger.info(f"⏱️ Tempo total de execução: {total_time:.2f} segundos")
            
            # Status das tasks
            if hasattr(self, 'task_start_times') and self.task_start_times:
                crew_logger.info(f"📋 Tasks executadas: {len(self.task_start_times)}")
                for task_name, start_time in self.task_start_times.items():
                    elapsed = time.time() - start_time
                    crew_logger.info(f"   • {task_name}: {elapsed:.2f}s")
            
            crew_logger.info(f"📁 Log completo salvo em: {getattr(self, 'log_file_path', 'APENAS CONSOLE')}")
            crew_logger.info("✅ EXECUÇÃO FINALIZADA COM SUCESSO")
            crew_logger.info("=" * 80)
            
            self._flush_logs()
            
        except Exception as e:
            crew_logger.error(f"❌ Erro ao gerar resumo final: {e}")
    
    # =============== AGENTES OTIMIZADOS COM DISTRIBUIÇÃO ESPECIALIZADA ===============
    
    @agent
    def engenheiro_dados(self) -> Agent:
        """
        🔧 ESPECIALISTA EM DADOS E ETL
        Ferramentas: SQL + Analytics Engine + File Tool
        Foco: Extração, transformação e validação de dados
        """
        crew_logger.info("🔧 Inicializando Engenheiro de Dados...")
        return Agent(
            config=self.agents_config['engenheiro_dados'],
            verbose=True,
            llm=llm,
            tools=[
                # file_tool,                # ✅ Leitura de arquivos
                sql_tool,                 # ✅ Acesso direto SQL Server
                # analytics_engine          # ✅ ETL avançado e preparação
            ]
        )

    @agent
    def analista_tendencias(self) -> Agent:
        """
        📈 ESPECIALISTA EM PADRÕES E PESQUISA
        Ferramentas: Statistical Analysis + DuckDuckGo + BI Dashboard
        Foco: Análise de correlações, tendências e contexto externo
        """
        return Agent(
            config=self.agents_config['analista_tendencias'],
            verbose=True,
            llm=llm,
            tools=[
                file_tool,                # ✅ Leitura de dados
                stats_tool,               # ✅ Análise estatística avançada
                search_tool,              # ✅ NOVO: Pesquisa de contexto externo
                bi_tool                   # ✅ Visualizações e dashboards
            ],
            respect_context_window=True
        )

    @agent
    def especialista_sazonalidade(self) -> Agent:
        """
        🌊 EXPERT EM SAZONALIDADE E CICLOS
        Ferramentas: Statistical Analysis + Analytics Engine + BI Dashboard
        Foco: Decomposição sazonal, modelagem temporal e padrões cíclicos
        """
        return Agent(
            config=self.agents_config['especialista_sazonalidade'],
            verbose=True,
            llm=llm,
            tools=[
                file_tool,                # ✅ Leitura de dados
                stats_tool,               # ✅ Decomposição sazonal STL
                analytics_engine,         # ✅ Modelagem temporal avançada
                bi_tool                   # ✅ Visualizações sazonais
            ],
            respect_context_window=True
        )
        
    @agent
    def especialista_projecoes(self) -> Agent:
        """
        🔮 FORECASTER PROFISSIONAL
        Ferramentas: Prophet + Statistical Analysis + BI Dashboard
        Foco: Previsões precisas, validação de modelos e cenários
        """
        return Agent(
            config=self.agents_config['especialista_projecoes'],
            verbose=True,
            llm=llm,
            tools=[
                file_tool,                # ✅ Leitura de dados
                prophet_tool,             # ✅ CRÍTICO: Prophet forecasting
                stats_tool,               # ✅ Validação estatística de modelos
                bi_tool                   # ✅ Gráficos de projeção profissionais
            ],
            respect_context_window=True
        )
        
    @agent
    def analista_segmentos(self) -> Agent:
        """
        👥 ESPECIALISTA EM CATEGORIAS E CLIENTES
        Ferramentas: KPI Calculator + Customer Insights + BI Dashboard
        Foco: Segmentação, análise por categoria e comportamento do cliente
        """
        return Agent(
            config=self.agents_config['analista_segmentos'],
            verbose=True,
            llm=llm,
            tools=[
                file_tool,                # ✅ Leitura de dados
                kpi_tool,                 # ✅ KPIs especializados por categoria
                customer_engine,          # ✅ Segmentação automática IA
                bi_tool                   # ✅ Dashboards comparativos
            ],
            respect_context_window=True
        )

    @agent
    def analista_inventario(self) -> Agent:
        """
        📦 OTIMIZADOR DE ESTOQUE INTELIGENTE
        Ferramentas: KPI Calculator + Recommendation Engine + Risk Assessment + BI Dashboard
        Foco: Otimização de inventário, gestão de riscos e recomendações automáticas
        """
        return Agent(
            config=self.agents_config['analista_inventario'],
            verbose=True,
            llm=llm,
            tools=[
                file_tool,                # ✅ Leitura de dados
                kpi_tool,                 # ✅ KPIs de estoque especializados
                recommendation_engine,    # ✅ Recomendações ML para estoque
                risk_tool,                # ✅ Avaliação de riscos de inventário
                bi_tool                   # ✅ Dashboards operacionais
            ],
            respect_context_window=True
        )

    @agent  
    def diretor_insights(self) -> Agent:
        """
        🎯 EXECUTIVO C-LEVEL COM ARSENAL ESTRATÉGICO
        Ferramentas: BI Dashboard + Recommendation Engine + Competitive Intelligence + KPI Calculator
        Foco: Síntese estratégica, benchmarking competitivo e decisões executivas
        """
        return Agent(
            config=self.agents_config['diretor_insights'],
            verbose=True,
            llm=llm,
            tools=[
                file_tool,                      # ✅ Leitura de dados
                kpi_tool,                       # ✅ KPIs executivos
                bi_tool,                        # ✅ Dashboards executivos
                recommendation_engine,          # ✅ Recomendações estratégicas
                competitive_tool,               # ✅ Inteligência competitiva
            ],
            respect_context_window=True
        )

    # =============== TASKS ===============
    
    @task
    def engenheiro_dados_task(self) -> Task:
        crew_logger.info("📋 Configurando Task: Engenheiro de Dados")
        return Task(
            config=self.tasks_config['engenheiro_dados_task'],
            # Garantir que os inputs de data sejam passados para a task
            # context_variables=['data_inicio', 'data_fim'],
            callback=lambda output: crew_logger.info(f"✅ [Engenheiro] Task concluída: {len(str(output))} chars")
        )
    
    @task
    def analista_tendencias_task(self) -> Task:
        return Task(
            config=self.tasks_config['analista_tendencias_task'],
            context=[self.engenheiro_dados_task()]
        )
    
    @task
    def especialista_sazonalidade_task(self) -> Task:
        return Task(
            config=self.tasks_config['especialista_sazonalidade_task'],
            context=[self.engenheiro_dados_task()]
        )
    
    @task
    def especialista_projecoes_task(self) -> Task:
        return Task(
            config=self.tasks_config['especialista_projecoes_task'],
            context=[self.engenheiro_dados_task(), self.especialista_sazonalidade_task()]
        )
    
    @task
    def analista_segmentos_task(self) -> Task:
        return Task(
            config=self.tasks_config['analista_segmentos_task'],
            context=[self.engenheiro_dados_task()]
        )

    @task
    def analise_inventario_task(self) -> Task:
        return Task(
            config=self.tasks_config['analise_inventario_task'],
            context=[self.engenheiro_dados_task()]
        )
        
    @task
    def relatorio_executivo_completo_task(self) -> Task:
        """
        TASK FINAL OTIMIZADA - Síntese estratégica com todas as ferramentas
        """
        return Task(
            config=self.tasks_config['relatorio_executivo_completo_task'],
            context=[
                self.engenheiro_dados_task(), 
                self.analista_tendencias_task(), 
                self.especialista_sazonalidade_task(), 
                self.especialista_projecoes_task(), 
                self.analista_segmentos_task(), 
                self.analise_inventario_task()
            ],
            output_file='reports/relatorio_executivo_completo.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the OPTIMIZED Insights crew with enhanced configuration"""
        crew_logger.info("🚀 Configurando Crew com logging avançado...")
        
        def task_callback(task_output):
            """Callback chamado quando cada task termina"""
            try:
                crew_logger.info(f"✅ Task concluída: {task_output.description[:100]}...")
                crew_logger.info(f"📊 Output type: {type(task_output)}")
                crew_logger.info(f"📏 Output length: {len(str(task_output))} chars")
                self._flush_logs()
            except Exception as e:
                crew_logger.error(f"❌ Erro no callback de task: {e}")
            
        def agent_callback(agent_output):
            """Callback chamado para cada ação do agente"""  
            try:
                crew_logger.info(f"🧠 Agente executando: {str(agent_output)[:200]}...")
                self._flush_logs()
            except Exception as e:
                crew_logger.error(f"❌ Erro no callback de agente: {e}")
            
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,             # ✅ Boolean, não inteiro
            memory=False,
            # planning=True,
            max_rpm=20,              # ✅ Reduzido para mais estabilidade 
            task_callback=task_callback,
            # embedder={               # ✅ Embedding para memória otimizada
            #     "provider": "ollama",
            #     "config": {
            #         "model": "nomic-embed-text",
            #         "base_url": "https://ollama.capta.com.br"
            #     }
            # }
        )


# =============== VALIDAÇÃO AVANÇADA DE FERRAMENTAS ===============

def validate_tools_setup_quiet():
    """Validação silenciosa das ferramentas - só reporta problemas críticos"""
    
    tools_status = {
        "Básicas": {
            "FileReadTool": _validate_tool(file_tool, "file_tool"),
            "SQLServerQueryTool": _validate_tool(sql_tool, "sql_tool"),
            "DuckDuckGoSearchTool": _validate_tool(search_tool, "search_tool"),
            "KPICalculatorTool": _validate_tool(kpi_tool, "kpi_tool"),
            "ProphetForecastTool": _validate_tool(prophet_tool, "prophet_tool"),
            "StatisticalAnalysisTool": _validate_tool(stats_tool, "stats_tool"),
            "BusinessIntelligenceTool": _validate_tool(bi_tool, "bi_tool"),
        },
        "Avançadas": {
            "CustomerInsightsEngine": _validate_tool(customer_engine, "customer_engine"),
            "RecommendationEngine": _validate_tool(recommendation_engine, "recommendation_engine"),
            "AdvancedAnalyticsEngine": _validate_tool(analytics_engine, "analytics_engine"),
            "RiskAssessmentTool": _validate_tool(risk_tool, "risk_tool"),
            "CompetitiveIntelligenceTool": _validate_tool(competitive_tool, "competitive_tool")
        }
    }
    
    # Contar ferramentas e identificar problemas críticos
    total_tools = 0
    working_tools = 0
    critical_errors = []
    
    for category, tools in tools_status.items():
        for tool_name, status in tools.items():
            total_tools += 1
            if status['available']:
                working_tools += 1
            else:
                critical_errors.append(f"{tool_name}: {status.get('error', 'Não disponível')}")
    
    success_rate = (working_tools / total_tools) * 100
    
    # Só reportar se houver problemas críticos ou taxa muito baixa
    if success_rate < 75:
        crew_logger.warning(f"⚠️ Taxa de sucesso das ferramentas baixa: {success_rate:.1f}%")
        for error in critical_errors[:3]:  # Só os 3 primeiros erros
            crew_logger.warning(f"❌ {error}")
    else:
        crew_logger.info(f"✅ Ferramentas validadas: {working_tools}/{total_tools} funcionando")
    
    return tools_status

def validate_tools_setup():
    """Validação completa e detalhada das ferramentas (modo debug)"""
    
    tools_status = {
        "Básicas": {
            "FileReadTool": _validate_tool(file_tool, "file_tool"),
            "SQLServerQueryTool": _validate_tool(sql_tool, "sql_tool"),
            "DuckDuckGoSearchTool": _validate_tool(search_tool, "search_tool"),
            "KPICalculatorTool": _validate_tool(kpi_tool, "kpi_tool"),
            "ProphetForecastTool": _validate_tool(prophet_tool, "prophet_tool"),
            "StatisticalAnalysisTool": _validate_tool(stats_tool, "stats_tool"),
            "BusinessIntelligenceTool": _validate_tool(bi_tool, "bi_tool"),
        },
        "Avançadas": {
            "CustomerInsightsEngine": _validate_tool(customer_engine, "customer_engine"),
            "RecommendationEngine": _validate_tool(recommendation_engine, "recommendation_engine"),
            "AdvancedAnalyticsEngine": _validate_tool(analytics_engine, "analytics_engine"),
            "RiskAssessmentTool": _validate_tool(risk_tool, "risk_tool"),
            "CompetitiveIntelligenceTool": _validate_tool(competitive_tool, "competitive_tool")
        }
    }
    
    crew_logger.info("🔧 VALIDAÇÃO COMPLETA DE FERRAMENTAS:")
    crew_logger.info("=" * 50)
    
    total_tools = 0
    working_tools = 0
    
    for category, tools in tools_status.items():
        crew_logger.info(f"\n📂 {category}:")
        for tool_name, status in tools.items():
            total_tools += 1
            status_icon = "✅" if status['available'] else "❌"
            crew_logger.info(f"  {status_icon} {tool_name}")
            
            if status['available']:
                working_tools += 1
                if status.get('methods'):
                    crew_logger.debug(f"      Métodos: {', '.join(status['methods'][:3])}...")
            else:
                crew_logger.warning(f"      Erro: {status.get('error', 'Não disponível')}")
    
    success_rate = (working_tools / total_tools) * 100
    crew_logger.info(f"\n📊 RESUMO:")
    crew_logger.info(f"  ✅ Ferramentas funcionando: {working_tools}/{total_tools} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        crew_logger.info(f"  🎉 EXCELENTE! Sistema totalmente operacional")
    elif success_rate >= 75:
        crew_logger.info(f"  ✅ BOM! Maioria das ferramentas funcionando")
    else:
        crew_logger.warning(f"  ⚠️ ATENÇÃO! Muitas ferramentas com problemas")
    
    return tools_status

def _validate_tool(tool_instance, tool_name: str) -> dict:
    """Validar uma ferramenta específica"""
    try:
        if tool_instance is None:
            return {'available': False, 'error': 'Instância não criada'}
        
        # Verificar se tem método _run (padrão CrewAI)
        has_run = hasattr(tool_instance, '_run')
        
        # Listar métodos disponíveis
        methods = [method for method in dir(tool_instance) 
                  if not method.startswith('_') and callable(getattr(tool_instance, method))]
        
        return {
            'available': True,
            'has_run_method': has_run,
            'methods': methods[:5],  # Primeiros 5 métodos
            'class_name': tool_instance.__class__.__name__
        }
        
    except Exception as e:
        return {'available': False, 'error': str(e)}

def get_tools_by_agent():
    """Retornar mapeamento de ferramentas por agente para debugging"""
    return {
        "engenheiro_dados": ["FileReadTool", "SQLServerQueryTool", "AdvancedAnalyticsEngine"],
        "analista_tendencias": ["FileReadTool", "StatisticalAnalysisTool", "DuckDuckGoSearchTool", "BusinessIntelligenceTool"],
        "especialista_sazonalidade": ["FileReadTool", "StatisticalAnalysisTool", "AdvancedAnalyticsEngine", "BusinessIntelligenceTool"],
        "especialista_projecoes": ["FileReadTool", "ProphetForecastTool", "StatisticalAnalysisTool", "BusinessIntelligenceTool"],
        "analista_segmentos": ["FileReadTool", "KPICalculatorTool", "CustomerInsightsEngine", "BusinessIntelligenceTool"],
        "analista_inventario": ["FileReadTool", "KPICalculatorTool", "RecommendationEngine", "RiskAssessmentTool", "BusinessIntelligenceTool"],
        "diretor_insights": ["FileReadTool", "KPICalculatorTool", "BusinessIntelligenceTool", "RecommendationEngine", "CompetitiveIntelligenceTool"]
    }


if __name__ == "__main__":
    # Configurar logging para execução direta
    logging.basicConfig(level=logging.INFO)
    
    # Validar setup ao executar diretamente
    crew_logger.info("🚀 INSIGHTS-AI CREW OTIMIZADA")
    crew_logger.info("=" * 50)
    
    tools_status = validate_tools_setup()
    
    crew_logger.info(f"\n🎯 DISTRIBUIÇÃO DE FERRAMENTAS POR AGENTE:")
    crew_logger.info("=" * 50)
    
    agent_tools = get_tools_by_agent()
    for agent, tools in agent_tools.items():
        crew_logger.info(f"\n👤 {agent.replace('_', ' ').title()}:")
        for tool in tools:
            crew_logger.info(f"  🔧 {tool}")
    
    crew_logger.info(f"\n🚀 Crew otimizada pronta para uso!")
    crew_logger.info(f"📊 Ferramentas distribuídas por especialização")
    crew_logger.info(f"🎯 Capacidade analítica maximizada")
    crew_logger.info(f"⚡ Performance e rate limiting otimizados")
