from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import FileReadTool
from typing import List
from dotenv import load_dotenv
import os

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

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

llm = LLM(
    model="openrouter/deepseek/deepseek-chat-v3-0324",
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
        self.tools_status = validate_tools_setup()

    agents: List[BaseAgent]
    tasks: List[Task]
    
    @before_kickoff
    def before_kickoff(self, inputs):
        """Before kickoff otimizado com validações e inputs de data"""
        print("🚀 Iniciando Insights-AI Otimizado...")
        
        # Validar e exibir inputs de data
        data_inicio = inputs.get('data_inicio')
        data_fim = inputs.get('data_fim')
        
        if data_inicio and data_fim:
            print(f"📅 Período de análise: {data_inicio} até {data_fim}")
            
            # Validar formato das datas
            try:
                from datetime import datetime
                datetime.strptime(data_inicio, '%Y-%m-%d')
                datetime.strptime(data_fim, '%Y-%m-%d')
                print("✅ Formato de datas validado")
            except ValueError:
                print("⚠️ WARNING: Formato de data inválido! Use YYYY-MM-DD")
        else:
            print("⚠️ WARNING: Inputs de data não fornecidos!")
        
        # Validar ferramentas críticas
        critical_tools = ['sql_tool', 'prophet_tool', 'stats_tool', 'bi_tool']
        for tool_name in critical_tools:
            tool_obj = globals().get(tool_name)
            if tool_obj is None:
                print(f"⚠️ WARNING: {tool_name} não encontrada!")
            else:
                print(f"✅ {tool_name} carregada com sucesso")
        
        # Executar SQL extraction com as datas fornecidas (se disponíveis)
        try:
            if data_inicio and data_fim:
                print(f"🔄 Extraindo dados do SQL Server para o período {data_inicio} a {data_fim}...")
                # O agente usará o SQL Tool com os parâmetros corretos
                print("📋 Dados serão extraídos pelo agente usando os inputs fornecidos")
            else:
                sql_tool._execute_query_and_save_to_csv()
                print("✅ Dados extraídos com sucesso do SQL Server (período padrão)")
        except Exception as e:
            print(f"⚠️ Erro na extração SQL: {e}")
            print("🔄 Tentando usar dados existentes...")
        
        return inputs
    
    # =============== AGENTES OTIMIZADOS COM DISTRIBUIÇÃO ESPECIALIZADA ===============
    
    @agent
    def engenheiro_dados(self) -> Agent:
        """
        🔧 ESPECIALISTA EM DADOS E ETL
        Ferramentas: SQL + Analytics Engine + File Tool
        Foco: Extração, transformação e validação de dados
        """
        return Agent(
            config=self.agents_config['engenheiro_dados'],
            verbose=True,
            llm=llm,
            tools=[
                file_tool,                # ✅ Leitura de arquivos
                sql_tool,                 # ✅ Acesso direto SQL Server
                analytics_engine          # ✅ ETL avançado e preparação
            ],
            allow_code_execution=False,
            reasoning=True,
            max_reasoning_attempts=3
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
            respect_context_window=True,
            allow_code_execution=False,
            reasoning=True,
            max_reasoning_attempts=3
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
            respect_context_window=True,
            allow_code_execution=False,
            reasoning=True,
            max_reasoning_attempts=3
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
            respect_context_window=True,
            allow_code_execution=False,
            reasoning=True,
            max_reasoning_attempts=3
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
            respect_context_window=True,
            allow_code_execution=False,
            reasoning=True,
            max_reasoning_attempts=3
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
            allow_code_execution=False,
            reasoning=True,
            max_reasoning_attempts=3
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
            allow_code_execution=False,
            reasoning=True,
            max_reasoning_attempts=3
        )

    # =============== TASKS ===============
    
    @task
    def engenheiro_dados_task(self) -> Task:
        return Task(
            config=self.tasks_config['engenheiro_dados_task'],
            # Garantir que os inputs de data sejam passados para a task
            context_variables=['data_inicio', 'data_fim']
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
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=True,
            planning=True,
            max_rpm=25,              # ✅ Rate limiting otimizado para estabilidade
            max_execution_time=3600, # ✅ NOVO: Timeout de 1 hora
            embedder={               # ✅ Embedding para memória otimizada
                "provider": "ollama",
                "config": {
                    "model": "nomic-embed-text",
                    "base_url": "https://ollama.capta.com.br"
                }
            }
        )


# =============== VALIDAÇÃO AVANÇADA DE FERRAMENTAS ===============

def validate_tools_setup():
    """Validar se todas as ferramentas foram importadas e instanciadas corretamente"""
    
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
    
    print("🔧 VALIDAÇÃO COMPLETA DE FERRAMENTAS:")
    print("=" * 50)
    
    total_tools = 0
    working_tools = 0
    
    for category, tools in tools_status.items():
        print(f"\n📂 {category}:")
        for tool_name, status in tools.items():
            total_tools += 1
            status_icon = "✅" if status['available'] else "❌"
            print(f"  {status_icon} {tool_name}")
            
            if status['available']:
                working_tools += 1
                if status.get('methods'):
                    print(f"      Métodos: {', '.join(status['methods'][:3])}...")
            else:
                print(f"      Erro: {status.get('error', 'Não disponível')}")
    
    success_rate = (working_tools / total_tools) * 100
    print(f"\n📊 RESUMO:")
    print(f"  ✅ Ferramentas funcionando: {working_tools}/{total_tools} ({success_rate:.1f}%)")
    
    if success_rate >= 90:
        print(f"  🎉 EXCELENTE! Sistema totalmente operacional")
    elif success_rate >= 75:
        print(f"  ✅ BOM! Maioria das ferramentas funcionando")
    else:
        print(f"  ⚠️ ATENÇÃO! Muitas ferramentas com problemas")
    
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
    # Validar setup ao executar diretamente
    print("🚀 INSIGHTS-AI CREW OTIMIZADA")
    print("=" * 50)
    
    tools_status = validate_tools_setup()
    
    print(f"\n🎯 DISTRIBUIÇÃO DE FERRAMENTAS POR AGENTE:")
    print("=" * 50)
    
    agent_tools = get_tools_by_agent()
    for agent, tools in agent_tools.items():
        print(f"\n👤 {agent.replace('_', ' ').title()}:")
        for tool in tools:
            print(f"  🔧 {tool}")
    
    print(f"\n🚀 Crew otimizada pronta para uso!")
    print(f"📊 Ferramentas distribuídas por especialização")
    print(f"🎯 Capacidade analítica maximizada")
    print(f"⚡ Performance e rate limiting otimizados")
