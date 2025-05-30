from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from dotenv import load_dotenv
import os
import logging
import time
from datetime import datetime
from pathlib import Path

# =============== IMPORTAÇÕES OTIMIZADAS DE FERRAMENTAS ===============

# Importação centralizada das ferramentas configuradas
from insights.config.tools_config_v3 import (
    TOOLS, 
    TOOLS_BY_CATEGORY, 
    get_tools_for_analysis,
    kpi_calculator_tool,
    statistical_analysis_tool,
    INTEGRATION_CONFIG,
    MIGRATION_MAP,
    AGENT_TOOL_MAPPING,
    get_tools_for_agent,
    get_integration_status,
    get_tools_statistics,
    validate_tool_compatibility,
    file_read_tool as file_tool,
    sql_query_tool as sql_tool,
    search_tool,
    prophet_forecast_tool as prophet_tool,
    business_intelligence_tool as bi_tool,
    kpi_calculator_tool as kpi_tool,
    statistical_analysis_tool as stats_tool,
    customer_insights_engine as customer_engine,
    recommendation_engine,
    advanced_analytics_engine as analytics_engine,
    risk_assessment_tool as risk_tool,
    competitive_intelligence_tool as competitive_tool,
    file_generation_tool,
    product_data_exporter,
    inventory_data_exporter,
    customer_data_exporter,
    financial_data_exporter
)

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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = LLM(
    model="gpt-4.1-mini",
    api_key=OPENAI_API_KEY
)

# =============== SISTEMA 100% CENTRALIZADO - SEM IMPORTS DUPLICADOS ===============

# ✅ TODAS AS FERRAMENTAS AGORA VÊM DO TOOLS_CONFIG_V3 - SISTEMA 100% CENTRALIZADO!
# ✅ REMOVIDOS TODOS OS IMPORTS DUPLICADOS - PONTO ÚNICO DE CONFIGURAÇÃO
# ✅ 17 FERRAMENTAS GERENCIADAS CENTRALMENTE VIA TOOLS_CONFIG_V3

# Validar sistema centralizado na inicialização
def validate_centralized_system():
    """Valida o sistema centralizado de ferramentas"""
    try:
        stats = get_tools_statistics()
        integration_status = get_integration_status()
        compatibility = validate_tool_compatibility()
        
        working_tools = sum(1 for status in compatibility.values() if status.get('status') == 'OK')
        total_tools = len(compatibility)
        
        crew_logger.info("🔧 SISTEMA CENTRALIZADO VALIDADO:")
        crew_logger.info(f"   • Total de ferramentas: {stats['total_tools']}")
        crew_logger.info(f"   • Categorias disponíveis: {stats['categories']}")
        crew_logger.info(f"   • Integrações configuradas: {stats['integrations']}")
        crew_logger.info(f"   • Agentes mapeados: {stats['agent_mappings']}")
        crew_logger.info(f"   • Migrações documentadas: {stats['migrations_documented']}")
        crew_logger.info(f"   • Ferramentas funcionando: {working_tools}/{total_tools} ({working_tools/total_tools*100:.1f}%)")
        
        # Status das integrações
        enabled_integrations = sum(1 for enabled in integration_status.values() if enabled)
        crew_logger.info(f"   • Integrações habilitadas: {enabled_integrations}/{len(integration_status)}")
        
        return True
    except Exception as e:
        crew_logger.error(f"❌ Erro na validação do sistema centralizado: {e}")
        return False

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
        
        # =============== VALIDAÇÃO COMPLETA DO SISTEMA CENTRALIZADO ===============
        crew_logger.info("🔧 Integrando com sistema COMPLETAMENTE centralizado de ferramentas...")
        
        # Validar sistema tools_config_v3 expandido
        system_ok = validate_centralized_system()
        if not system_ok:
            crew_logger.warning("⚠️ Sistema centralizado com problemas - usando fallback")
        
        # Log informações avançadas da configuração
        self._log_advanced_tools_config_info()
        
        # Validação de compatibilidade (modo inteligente baseado em debug)
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
    
    def _log_advanced_tools_config_info(self):
        """Log informações avançadas sobre a configuração centralizada completa"""
        try:
            # Estatísticas do sistema
            stats = get_tools_statistics()
            crew_logger.info("📊 SISTEMA TOOLS_CONFIG_V3 EXPANDIDO:")
            crew_logger.info(f"   • Total de ferramentas: {stats['total_tools']}")
            crew_logger.info(f"   • Categorias disponíveis: {stats['categories']}")
            crew_logger.info(f"   • Integrações configuradas: {stats['integrations']}")
            crew_logger.info(f"   • Agentes mapeados: {stats['agent_mappings']}")
            crew_logger.info(f"   • Migrações documentadas: {stats['migrations_documented']}")
            
            # Status das integrações
            integration_status = get_integration_status()
            enabled_count = sum(1 for enabled in integration_status.values() if enabled)
            crew_logger.info(f"   • Integrações habilitadas: {enabled_count}/{len(integration_status)}")
            
            # Principais categorias
            major_categories = ['basic_tools', 'advanced_ai', 'data_export', 'financial_analysis', 'customer_analysis']
            for category in major_categories:
                if category in stats['tools_by_category']:
                    crew_logger.info(f"     - {category}: {stats['tools_by_category'][category]} ferramentas")
            
        except Exception as e:
            crew_logger.error(f"❌ Erro ao capturar info avançada do tools_config: {e}")

    def get_tools_for_agent(self, agent_role: str):
        """
        Retorna ferramentas apropriadas para um agente baseado em seu papel
        Integra a lógica do tools_config_v3 com as ferramentas específicas
        """
        try:
            # Mapeamento dos 8 AGENTES VALIDADOS para tipos de análise
            agent_analysis_map = {
                'engenheiro_dados': 'data_engineering',
                'analista_vendas_tendencias': 'statistical',  # Consolidado com sazonalidade/projeções
                'especialista_produtos': 'comprehensive',     # Consolidado com segmentação
                'analista_estoque': 'business',               # Consolidado com inventário
                'analista_financeiro': 'business',
                'especialista_clientes': 'comprehensive',
                'analista_performance': 'statistical',       # Consolidado com vendedores
                'diretor_insights': 'comprehensive'
            }
            
            analysis_type = agent_analysis_map.get(agent_role, 'comprehensive')
            
            # Usar get_tools_for_analysis do tools_config_v3 como base
            base_tools = get_tools_for_analysis(analysis_type)
            
            # Adicionar ferramentas específicas para cada agente consolidado
            if agent_role == 'engenheiro_dados':
                return [sql_tool]
            elif agent_role == 'analista_vendas_tendencias':
                # Consolidado: tendências + sazonalidade + projeções
                return base_tools + [file_tool, prophet_tool, search_tool, analytics_engine, bi_tool, file_generation_tool]
            elif agent_role == 'especialista_produtos':
                # Consolidado: produtos + segmentação + categorias
                return base_tools + [file_tool, analytics_engine, recommendation_engine, bi_tool, file_generation_tool]
            elif agent_role == 'analista_estoque':
                # Consolidado: estoque + inventário
                return base_tools + [file_tool, analytics_engine, recommendation_engine, risk_tool, bi_tool, file_generation_tool]
            elif agent_role == 'analista_financeiro':
                return base_tools + [file_tool, analytics_engine, prophet_tool, competitive_tool, risk_tool, bi_tool, file_generation_tool]
            elif agent_role == 'especialista_clientes':
                return base_tools + [file_tool, customer_engine, analytics_engine, recommendation_engine, bi_tool, file_generation_tool]
            elif agent_role == 'analista_performance':
                # Consolidado: performance + vendedores
                return base_tools + [file_tool, analytics_engine, bi_tool, file_generation_tool]
            elif agent_role == 'diretor_insights':
                return base_tools + [file_tool, bi_tool, recommendation_engine, competitive_tool, file_generation_tool]
            else:
                return base_tools + [file_tool, bi_tool]
                
        except Exception as e:
            crew_logger.error(f"❌ Erro ao obter ferramentas para {agent_role}: {e}")
            # Fallback: retornar ferramentas básicas
            return [file_tool, stats_tool, kpi_tool]

    def get_tools_for_agent_legacy(self, agent_role: str):
        """
        MÉTODO LEGACY - mantido para compatibilidade
        Nova implementação usa get_tools_for_agent do tools_config_v3
        """
        try:
            # Usar a nova função centralizada do tools_config_v3
            return get_tools_for_agent(agent_role)
        except Exception as e:
            crew_logger.error(f"❌ Erro ao obter ferramentas para {agent_role}: {e}")
            # Fallback: retornar ferramentas básicas
            return [file_tool, stats_tool, kpi_tool]

    # =============== AGENTES OTIMIZADOS COM DISTRIBUIÇÃO ESPECIALIZADA ===============
    
    @agent
    def engenheiro_dados(self) -> Agent:
        """
        🔧 ESPECIALISTA EM DADOS E ETL
        Ferramentas: Via tools_config_v3 (data_engineering)
        Foco: Extração, transformação e validação de dados
        """
        crew_logger.info("🔧 Inicializando Engenheiro de Dados via sistema centralizado...")
        
        # Usar sistema completamente centralizado
        agent_tools = get_tools_for_agent('engenheiro_dados')
        crew_logger.info(f"   • Ferramentas configuradas: {[tool.__class__.__name__ for tool in agent_tools]}")
        
        return Agent(
            config=self.agents_config['engenheiro_dados'],
            verbose=True,
            llm=llm,
            max_rpm=30,
            tools=agent_tools
        )

    @agent
    def analista_vendas_tendencias(self) -> Agent:
        """
        📈 ESPECIALISTA EM VENDAS, TENDÊNCIAS, SAZONALIDADE E PROJEÇÕES (CONSOLIDADO)
        Ferramentas: Via tools_config_v3 (statistical + forecasting)
        Foco: Análise temporal completa, sazonalidade, previsões e contexto externo
        """
        crew_logger.info("📈 Inicializando Analista de Vendas e Tendências via sistema centralizado...")
        
        # Usar sistema centralizado completo
        agent_tools = get_tools_for_agent('analista_vendas_tendencias')
        crew_logger.info(f"   • Total de ferramentas: {len(agent_tools)}")
        crew_logger.info(f"   • Principais: {[tool.__class__.__name__ for tool in agent_tools[:4]]}")
        
        return Agent(
            config=self.agents_config['analista_vendas_tendencias'],
            verbose=True,
            llm=llm,
            max_rpm=30,
            reasoning=True,
            max_reasoning_attempts=3,
            tools=agent_tools,
            respect_context_window=True
        )

    @agent
    def especialista_produtos(self) -> Agent:
        """
        🎯 ESPECIALISTA EM PRODUTOS, CATEGORIAS E SEGMENTAÇÃO (CONSOLIDADO)
        Ferramentas: Advanced Analytics + Statistical + Recommendation + Risk + BI + File Generation + Product Export
        Foco: Análise ABC, market basket, cross-sell, ciclo de vida, segmentação e exportação
        """
        crew_logger.info("🎯 Inicializando Especialista em Produtos via sistema centralizado...")
        
        agent_tools = get_tools_for_agent('especialista_produtos')
        crew_logger.info(f"   • Total de ferramentas: {len(agent_tools)}")
        
        return Agent(
            config=self.agents_config['especialista_produtos'],
            verbose=True,
            llm=llm,
            max_rpm=30,
            reasoning=True,
            max_reasoning_attempts=3,
            tools=agent_tools,
            respect_context_window=True
        )

    @agent
    def analista_estoque(self) -> Agent:
        """
        📦 ESPECIALISTA EM GESTÃO DE ESTOQUE E INVENTÁRIO (CONSOLIDADO)
        Ferramentas: KPI + Recommendation + Risk + Analytics + BI + File Generation + Inventory Export
        Foco: Otimização de inventário, gestão de riscos, recomendações ML e exportação
        """
        crew_logger.info("📦 Inicializando Analista de Estoque via sistema centralizado...")
        
        agent_tools = get_tools_for_agent('analista_estoque')
        crew_logger.info(f"   • Total de ferramentas: {len(agent_tools)}")
        
        return Agent(
            config=self.agents_config['analista_estoque'],
            verbose=True,
            llm=llm,
            max_rpm=30,
            reasoning=True,
            max_reasoning_attempts=3,
            tools=agent_tools,
            respect_context_window=True
        )

    @agent
    def analista_financeiro(self) -> Agent:
        """
        💰 ANALISTA FINANCEIRO SÊNIOR COM IA + EXPORTAÇÃO CSV
        Ferramentas: Via tools_config_v3 (financial_analysis completa)
        Foco: KPIs críticos, análise de margens/custos, elasticidade de preços, projeções financeiras
        """
        crew_logger.info("💰 Inicializando Analista Financeiro via sistema centralizado...")
        
        # Usar categoria financial_analysis do tools_config_v3
        agent_tools = get_tools_for_agent('analista_financeiro')
        
        crew_logger.info(f"   • Categoria: financial_analysis")
        crew_logger.info(f"   • Total de ferramentas: {len(agent_tools)}")
        crew_logger.info(f"   • Ferramentas financeiras especializadas disponíveis")
        
        return Agent(
            config=self.agents_config['analista_financeiro'],
            verbose=True,
            llm=llm,
            max_rpm=30,
            reasoning=True,
            max_reasoning_attempts=3,
            tools=agent_tools,
            respect_context_window=True
        )

    @agent
    def especialista_clientes(self) -> Agent:
        """
        👥 ESPECIALISTA EM INTELIGÊNCIA DE CLIENTES RFV + EXPORTAÇÃO CSV
        Ferramentas: Via tools_config_v3 (customer_analysis completa)
        Foco: Segmentação RFV com ML, CLV preditivo, análise demográfica/geográfica
        """
        crew_logger.info("👥 Inicializando Especialista em Clientes via sistema centralizado...")
        
        # Usar categoria customer_analysis do tools_config_v3
        agent_tools = get_tools_for_agent('especialista_clientes')
        
        crew_logger.info(f"   • Categoria: customer_analysis")
        crew_logger.info(f"   • Total de ferramentas: {len(agent_tools)}")
        crew_logger.info(f"   • ML e análise comportamental incluídos")
        
        return Agent(
            config=self.agents_config['especialista_clientes'],
            verbose=True,
            llm=llm,
            max_rpm=30,
            reasoning=True,
            max_reasoning_attempts=3,
            tools=agent_tools,
            respect_context_window=True
        )

    @agent
    def analista_performance(self) -> Agent:
        """
        👤 ANALISTA DE PERFORMANCE DE VENDEDORES (CONSOLIDADO)
        Ferramentas: Statistical Analysis + KPI Calculator + BI Tool + File Generation Tool
        Foco: Avaliação de performance individual, desenvolvimento da equipe e best practices
        """
        crew_logger.info("👤 Inicializando Analista de Performance via sistema centralizado...")
        
        agent_tools = get_tools_for_agent('analista_performance')
        crew_logger.info(f"   • Total de ferramentas: {len(agent_tools)}")
        
        return Agent(
            config=self.agents_config['analista_performance'],
            verbose=True,
            llm=llm,
            max_rpm=30,
            reasoning=True,
            max_reasoning_attempts=3,
            tools=agent_tools,
            respect_context_window=True
        )

    @agent  
    def diretor_insights(self) -> Agent:
        """
        🎯 EXECUTIVO C-LEVEL COM ARSENAL ESTRATÉGICO
        Ferramentas: BI Dashboard + Recommendation Engine + Competitive Intelligence + KPI Calculator
        Foco: Síntese estratégica, benchmarking competitivo e decisões executivas
        """
        crew_logger.info("🎯 Inicializando Diretor de Insights via sistema centralizado...")
        
        agent_tools = get_tools_for_agent('diretor_insights')
        crew_logger.info(f"   • Total de ferramentas: {len(agent_tools)}")
        
        return Agent(
            config=self.agents_config['diretor_insights'],
            verbose=True,
            llm=llm,
            max_rpm=30,
            reasoning=True,
            max_reasoning_attempts=3,
            tools=agent_tools,
            respect_context_window=True
        )

    # =============== TASKS ALINHADAS COM OS 8 AGENTES ===============
    
    @task
    def engenheiro_dados_task(self) -> Task:
        crew_logger.info("📋 Configurando Task: Engenheiro de Dados")
        return Task(
            config=self.tasks_config['engenheiro_dados_task'],
            callback=lambda output: crew_logger.info(f"✅ [Engenheiro] Task concluída: {len(str(output))} chars")
        )
    
    @task
    def analista_vendas_tendencias_task(self) -> Task:
        crew_logger.info("📋 Configurando Task: Analista Vendas e Tendências (Consolidada)")
        return Task(
            config=self.tasks_config['analista_vendas_tendencias_task'],
            context=[self.engenheiro_dados_task()],
            callback=lambda output: crew_logger.info(f"✅ [Vendas/Tendências] Task concluída: {len(str(output))} chars")
        )
    
    @task
    def especialista_produtos_task(self) -> Task:
        crew_logger.info("📋 Configurando Task: Especialista Produtos (Consolidada)")
        return Task(
            config=self.tasks_config['especialista_produtos_task'],
            context=[self.engenheiro_dados_task()],
            callback=lambda output: crew_logger.info(f"✅ [Produtos] Task concluída: {len(str(output))} chars")
        )

    @task
    def analista_estoque_task(self) -> Task:
        crew_logger.info("📋 Configurando Task: Analista Estoque (Consolidada)")
        return Task(
            config=self.tasks_config['analista_estoque_task'],
            context=[self.engenheiro_dados_task()],
            callback=lambda output: crew_logger.info(f"✅ [Estoque] Task concluída: {len(str(output))} chars")
        )

    @task
    def analista_financeiro_task(self) -> Task:
        crew_logger.info("📋 Configurando Task: Analista Financeiro")
        return Task(
            config=self.tasks_config['analista_financeiro_task'],
            context=[self.engenheiro_dados_task()],
            callback=lambda output: crew_logger.info(f"✅ [Financeiro] Task concluída: {len(str(output))} chars")
        )

    @task
    def especialista_clientes_task(self) -> Task:
        crew_logger.info("📋 Configurando Task: Especialista Clientes")
        return Task(
            config=self.tasks_config['especialista_clientes_task'],
            context=[self.engenheiro_dados_task()],
            callback=lambda output: crew_logger.info(f"✅ [Clientes] Task concluída: {len(str(output))} chars")
        )

    @task
    def analista_performance_task(self) -> Task:
        crew_logger.info("📋 Configurando Task: Analista Performance (Consolidada)")
        return Task(
            config=self.tasks_config['analista_performance_task'],
            context=[self.engenheiro_dados_task()],
            callback=lambda output: crew_logger.info(f"✅ [Performance] Task concluída: {len(str(output))} chars")
        )

    @task
    def diretor_insights_task(self) -> Task:
        crew_logger.info("📋 Configurando Task: Diretor Insights (Síntese Final)")
        return Task(
            config=self.tasks_config['diretor_insights_task'],
            context=[
                self.engenheiro_dados_task(),
                self.analista_vendas_tendencias_task(),
                self.especialista_produtos_task(),
                self.analista_estoque_task(),
                self.analista_financeiro_task(),
                self.especialista_clientes_task(),
                self.analista_performance_task()
            ],
            callback=lambda output: crew_logger.info(f"✅ [Diretor] Task final concluída: {len(str(output))} chars")
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
    """Validação silenciosa integrada com tools_config_v3 expandido"""
    
    try:
        # Usar função de validação do tools_config_v3
        compatibility = validate_tool_compatibility()
        stats = get_tools_statistics()
        
        working_tools = sum(1 for status in compatibility.values() if status.get('status') == 'OK')
        total_tools = len(compatibility)
        success_rate = (working_tools / total_tools) * 100
        
        # Log do sistema expandido
        crew_logger.info(f"🔧 Sistema Tools Config V3 Expandido:")
        crew_logger.info(f"   • Ferramentas carregadas: {stats['total_tools']}")
        crew_logger.info(f"   • Categorias disponíveis: {stats['categories']}")
        crew_logger.info(f"   • Integrações configuradas: {stats['integrations']}")
        crew_logger.info(f"   • Agentes mapeados: {stats['agent_mappings']}")
        
        # Status geral
        if success_rate < 75:
            crew_logger.warning(f"⚠️ Taxa de sucesso baixa: {success_rate:.1f}%")
        else:
            crew_logger.info(f"✅ Sistema expandido validado: {working_tools}/{total_tools} funcionando ({success_rate:.1f}%)")
        
        # Preparar resultado para compatibilidade
        tools_status = {
            "Tools_Config_V3_Expandido": {
                tool_name: {
                    'available': status.get('status') == 'OK',
                    'status': status.get('status', 'UNKNOWN'),
                    'has_run_method': status.get('has_run_method', False)
                }
                for tool_name, status in compatibility.items()
            }
        }
        
        return tools_status
        
    except Exception as e:
        crew_logger.error(f"❌ Erro na validação expandida: {e}")
        # Fallback para validação padrão
        crew_logger.info("🔄 Executando fallback para validação padrão...")
        return {
            "Fallback_Validation": {
                "basic_tools": {
                    'available': True,
                    'status': 'FALLBACK',
                    'has_run_method': True
                }
            }
        }

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
            "CompetitiveIntelligenceTool": _validate_tool(competitive_tool, "competitive_tool"),
            "FileGenerationTool": _validate_tool(file_generation_tool, "file_generation_tool")
        },
        "Exportação": {
            "ProductDataExporter": _validate_tool(product_data_exporter, "product_data_exporter"),
            "InventoryDataExporter": _validate_tool(inventory_data_exporter, "inventory_data_exporter"),
            "CustomerDataExporter": _validate_tool(customer_data_exporter, "customer_data_exporter"),
            "FinancialDataExporter": _validate_tool(financial_data_exporter, "financial_data_exporter")
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
    """Retornar mapeamento de ferramentas por agente - integrado com tools_config_v3 - APENAS 8 AGENTES VALIDADOS"""
    
    # Informações sobre ferramentas do tools_config_v3
    config_tools_info = f"Tools Config V3: {len(TOOLS)} ferramentas base"
    
    return {
        "tools_config_v3_info": config_tools_info,
        "integration_enabled": INTEGRATION_CONFIG.get('kpi_statistical_integration', {}).get('enabled', False),
        "total_validated_agents": 8,
        "agents_mapping": {
            "engenheiro_dados": {
                "base_analysis_type": "data_engineering",
                "tools": ["SQLServerQueryTool"],
                "tools_config_v3": "Não usa diretamente (ferramentas específicas)"
            },
            "analista_vendas_tendencias": {
                "base_analysis_type": "statistical", 
                "tools_config_v3": "StatisticalAnalysisTool",
                "additional_tools": ["FileReadTool", "ProphetForecastTool", "DuckDuckGoSearchTool", "AdvancedAnalyticsEngine", "BusinessIntelligenceTool", "FileGenerationTool"],
                "consolidation_note": "Unifica: vendas + tendências + sazonalidade + projeções"
            },
            "especialista_produtos": {
                "base_analysis_type": "comprehensive",
                "tools_config_v3": "KPICalculatorTool + StatisticalAnalysisTool",
                "additional_tools": ["FileReadTool", "AdvancedAnalyticsEngine", "RecommendationEngine", "BusinessIntelligenceTool", "FileGenerationTool"],
                "consolidation_note": "Unifica: produtos + categorias + segmentação + market basket"
            },
            "analista_estoque": {
                "base_analysis_type": "business",
                "tools_config_v3": "KPICalculatorTool",
                "additional_tools": ["FileReadTool", "AdvancedAnalyticsEngine", "RecommendationEngine", "RiskAssessmentTool", "BusinessIntelligenceTool", "FileGenerationTool"],
                "consolidation_note": "Unifica: estoque + inventário + gestão de riscos"
            },
            "analista_financeiro": {
                "base_analysis_type": "business",
                "tools_config_v3": "KPICalculatorTool",
                "additional_tools": ["FileReadTool", "AdvancedAnalyticsEngine", "ProphetForecastTool", "CompetitiveIntelligenceTool", "RiskAssessmentTool", "BusinessIntelligenceTool", "FileGenerationTool"]
            },
            "especialista_clientes": {
                "base_analysis_type": "comprehensive",
                "tools_config_v3": "KPICalculatorTool + StatisticalAnalysisTool",
                "additional_tools": ["FileReadTool", "CustomerInsightsEngine", "AdvancedAnalyticsEngine", "RecommendationEngine", "BusinessIntelligenceTool", "FileGenerationTool"]
            },
            "analista_performance": {
                "base_analysis_type": "statistical",
                "tools_config_v3": "StatisticalAnalysisTool + KPICalculatorTool",
                "additional_tools": ["FileReadTool", "AdvancedAnalyticsEngine", "BusinessIntelligenceTool", "FileGenerationTool"],
                "consolidation_note": "Unifica: performance + vendedores + desenvolvimento de equipe"
            },
            "diretor_insights": {
                "base_analysis_type": "comprehensive",
                "tools_config_v3": "KPICalculatorTool + StatisticalAnalysisTool",
                "additional_tools": ["FileReadTool", "BusinessIntelligenceTool", "RecommendationEngine", "CompetitiveIntelligenceTool", "FileGenerationTool"]
            }
        },
        "removed_orphan_agents": [
            "analista_tendencias",      # → consolidado em analista_vendas_tendencias
            "especialista_sazonalidade", # → consolidado em analista_vendas_tendencias  
            "especialista_projecoes",   # → consolidado em analista_vendas_tendencias
            "analista_segmentos",       # → consolidado em especialista_produtos
            "analista_inventario",      # → consolidado em analista_estoque
            "analista_vendedores"       # → consolidado em analista_performance
        ]
    }


if __name__ == "__main__":
    # Configurar logging para execução direta
    logging.basicConfig(level=logging.INFO)
    
    # Validar setup ao executar diretamente
    crew_logger.info("🚀 INSIGHTS-AI CREW EXPANDIDA E OTIMIZADA")
    crew_logger.info("🔧 INTEGRADO COM TOOLS_CONFIG_V3")
    crew_logger.info("=" * 60)
    
    # Log informações do tools_config_v3
    crew_logger.info("📊 INTEGRAÇÃO TOOLS_CONFIG_V3:")
    crew_logger.info(f"   • Ferramentas base configuradas: {len(TOOLS)}")
    crew_logger.info(f"   • Categorias disponíveis: {list(TOOLS_BY_CATEGORY.keys())}")
    crew_logger.info(f"   • Integração KPI-Statistical: {INTEGRATION_CONFIG.get('kpi_statistical_integration', {}).get('enabled', False)}")
    crew_logger.info(f"   • Migrações documentadas: {sum(len(items) for items in MIGRATION_MAP.values())} funcionalidades")
    
    tools_status = validate_tools_setup()
    
    crew_logger.info(f"\n🎯 DISTRIBUIÇÃO DE FERRAMENTAS POR AGENTE (NOVO SISTEMA):")
    crew_logger.info("=" * 60)
    
    agent_tools = get_tools_by_agent()
    
    # Mostrar informações da integração
    crew_logger.info(f"📋 {agent_tools['tools_config_v3_info']}")
    crew_logger.info(f"🔗 Integração habilitada: {agent_tools['integration_enabled']}")
    crew_logger.info("")
    
    # Estatísticas por tipo de análise
    analysis_types = {}
    for agent, info in agent_tools['agents_mapping'].items():
        analysis_type = info['base_analysis_type']
        if analysis_type not in analysis_types:
            analysis_types[analysis_type] = []
        analysis_types[analysis_type].append(agent)
    
    crew_logger.info("📊 DISTRIBUIÇÃO POR TIPO DE ANÁLISE:")
    for analysis_type, agents in analysis_types.items():
        crew_logger.info(f"   • {analysis_type}: {len(agents)} agentes")
        
    crew_logger.info(f"\n📊 ESTATÍSTICAS DO SISTEMA INTEGRADO:")
    crew_logger.info("=" * 40)
    crew_logger.info(f"🤖 Total de agentes especializados: {len(agent_tools['agents_mapping'])}")
    crew_logger.info(f"🔧 Ferramentas do tools_config_v3: {len(TOOLS)}")
    crew_logger.info(f"📋 Categorias de ferramentas: {len(TOOLS_BY_CATEGORY)}")
    crew_logger.info(f"⚙️ Configurações de integração: {len(INTEGRATION_CONFIG)}")
    
    crew_logger.info(f"\n🎯 CAPACIDADES EXPANDIDAS COM INTEGRAÇÃO:")
    crew_logger.info("=" * 40)
    crew_logger.info(f"✅ Sistema centralizado de ferramentas")
    crew_logger.info(f"✅ Seleção automática por tipo de análise")
    crew_logger.info(f"✅ Configuração de integração entre ferramentas")
    crew_logger.info(f"✅ Documentação automática de migrações")
    crew_logger.info(f"✅ Validação integrada de ferramentas")
    crew_logger.info(f"✅ Mapeamento inteligente de agentes")
    
    crew_logger.info(f"\n🔧 FERRAMENTAS DISPONÍVEIS NO TOOLS_CONFIG_V3:")
    crew_logger.info("=" * 40)
    for category, tools in TOOLS_BY_CATEGORY.items():
        if tools:  # Só mostrar categorias com ferramentas
            crew_logger.info(f"📂 {category}: {len(tools)} ferramenta(s)")
    
    crew_logger.info(f"\n🚀 Sistema integrado pronto para análises avançadas!")
    crew_logger.info(f"📊 Utilização otimizada do tools_config_v3")
    crew_logger.info(f"🎯 Distribuição inteligente de ferramentas")
    crew_logger.info(f"🔗 Integração automática entre KPI e Statistical Analysis")
    crew_logger.info(f"⚡ Performance e manutenibilidade melhoradas")
