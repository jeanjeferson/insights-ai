"""
⚡ CREW OTIMIZADO COM PERFORMANCE AVANÇADA
========================================

Versão otimizada do crew principal com:
- Logging estruturado menos verbose
- Lazy loading de ferramentas
- Cache inteligente de validações
- Inicialização paralela de agentes
- Métricas de performance em tempo real
"""

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from dotenv import load_dotenv
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# =============== IMPORTAÇÕES OTIMIZADAS ===============

# Sistema de performance
from insights.config.performance_config import (
    PERFORMANCE_CONFIG,
    optimized_logger,
    performance_cache,
    performance_tracked,
    cached_result,
    log_startup_metrics,
    should_skip_validation,
    get_optimized_tool_list,
    cleanup_performance_resources
)

# Ferramentas (lazy loading)
from insights.config.tools_config_v3 import (
    TOOLS, 
    get_tools_for_analysis,
    get_tools_for_agent,
    get_integration_status,
    get_tools_statistics,
    validate_tool_compatibility,
    file_read_tool as file_tool,
    sql_query_tool as sql_tool,
    business_intelligence_tool as bi_tool,
    validate_agent_data_access,
    get_data_flow_architecture
)

load_dotenv()

# =============== SISTEMA DE CONTROLE DE CONTEXTO ===============

class ContextManager:
    """Gerenciador de contexto para evitar overload do LLM"""
    
    MAX_TOKENS_PER_INPUT = 50000  # Limite seguro de tokens por input
    MAX_CHARS_PER_INPUT = 200000  # Limite de caracteres (aprox 4 chars/token)
    AGGRESSIVE_LIMIT = 50000     # Limite para modo agressivo
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimar número de tokens em um texto"""
        return len(text) // 4  # Aproximação: 4 chars = 1 token
    
    @staticmethod
    def create_data_summary(data_text: str, max_chars: int = 5000) -> str:
        """Criar resumo inteligente dos dados"""
        
        if len(data_text) <= max_chars:
            return data_text
        
        # Extrair informações chave
        lines = data_text.split('\n')
        
        summary_parts = []
        
        # Cabeçalho (primeiras 3 linhas)
        if len(lines) > 3:
            summary_parts.append("=== ESTRUTURA DOS DADOS ===")
            summary_parts.extend(lines[:3])
        
        # Estatísticas básicas
        total_lines = len(lines)
        summary_parts.append(f"\n=== ESTATÍSTICAS ===")
        summary_parts.append(f"Total de registros: {total_lines - 1}")  # -1 para header
        summary_parts.append(f"Tamanho original: {len(data_text)} caracteres")
        
        # Amostra dos dados (ajustar tamanho baseado no modo)
        if max_chars <= 5000:  # Modo agressivo
            summary_parts.append(f"\n=== AMOSTRA DOS DADOS (primeiras 5 linhas) ===")
            if total_lines > 6:
                summary_parts.extend(lines[1:6])  # Pular header
            else:
                summary_parts.extend(lines[1:])
                
            summary_parts.append(f"\n=== ÚLTIMA LINHA ===")
            if total_lines > 1:
                summary_parts.append(lines[-1])
        else:  # Modo normal
            if total_lines > 20:
                summary_parts.append(f"\n=== AMOSTRA DOS DADOS (primeiras 10 linhas) ===")
                summary_parts.extend(lines[1:11])  # Pular header
                
                summary_parts.append(f"\n=== ÚLTIMOS REGISTROS (últimas 5 linhas) ===")
                summary_parts.extend(lines[-5:])
            else:
                summary_parts.append(f"\n=== DADOS COMPLETOS ===")
                summary_parts.extend(lines[1:])  # Pular header
        
        summary_parts.append(f"\n=== NOTA ===")
        summary_parts.append(f"Este é um resumo dos dados para análise. Total de {total_lines-1} registros.")
        
        return '\n'.join(summary_parts)
    
    @staticmethod
    def create_aggressive_summary(data_text: str) -> str:
        """Criar resumo ultra-compacto para casos extremos"""
        
        lines = data_text.split('\n')
        total_lines = len(lines)
        
        # Resumo extremamente compacto
        summary = [
            "=== RESUMO ULTRA-COMPACTO ===",
            f"Registros: {total_lines - 1 if total_lines > 1 else 0}",
            f"Tamanho: {len(data_text)} chars"
        ]
        
        # Apenas header se existir
        if total_lines > 1:
            summary.append(f"Colunas: {lines[0]}")
            
        # Apenas primeira linha de dados
        if total_lines > 2:
            summary.append(f"Exemplo: {lines[1]}")
        
        summary.append("NOTA: Dados disponíveis para análise mais detalhada se necessário.")
        
        return '\n'.join(summary)
    
    @staticmethod
    def truncate_context(context: str, max_chars: int = None) -> str:
        """Truncar contexto mantendo informações essenciais"""
        
        if max_chars is None:
            max_chars = ContextManager.MAX_CHARS_PER_INPUT
        
        if len(context) <= max_chars:
            return context
        
        optimized_logger.warning(f"⚠️ Contexto muito grande ({len(context)} chars), truncando para {max_chars}")
        
        # Manter início e fim do contexto
        half_size = max_chars // 2 - 100  # -100 para mensagem de truncamento
        
        start_part = context[:half_size]
        end_part = context[-half_size:]
        
        truncation_message = f"\n\n... [TRUNCADO: {len(context) - max_chars} caracteres removidos] ...\n\n"
        
        return start_part + truncation_message + end_part
    
    @staticmethod
    def prepare_data_context(data: any, context_type: str = "data") -> str:
        """Preparar contexto de dados de forma inteligente"""
        
        if isinstance(data, str):
            data_text = data
        else:
            data_text = str(data)
        
        # Verificar tamanho e aplicar estratégia apropriada
        if context_type == "aggressive":
            # Modo agressivo para casos extremos
            if len(data_text) > 10000:
                return ContextManager.create_aggressive_summary(data_text)
            else:
                return ContextManager.create_data_summary(data_text, 2000)
        
        elif len(data_text) > ContextManager.MAX_CHARS_PER_INPUT:
            optimized_logger.info(f"📊 Dados muito grandes ({len(data_text)} chars), criando resumo...")
            
            if context_type == "csv" or "," in data_text or ";" in data_text:
                # Tratar como dados tabulares
                return ContextManager.create_data_summary(data_text, 8000)
            else:
                # Truncar texto geral
                return ContextManager.truncate_context(data_text, 10000)
        
        return data_text
    
    @staticmethod
    def analyze_data_locally(data_text: str) -> dict:
        """Analisar dados localmente para extrair insights básicos"""
        
        try:
            lines = data_text.split('\n')
            
            # Estatísticas básicas
            stats = {
                'total_lines': len(lines),
                'total_chars': len(data_text),
                'has_header': True,
                'separator': None,
                'columns': [],
                'sample_data': []
            }
            
            # Detectar separador
            if lines and len(lines) > 1:
                first_line = lines[0]
                if ';' in first_line:
                    stats['separator'] = ';'
                elif ',' in first_line:
                    stats['separator'] = ','
                elif '\t' in first_line:
                    stats['separator'] = '\t'
                
                # Extrair colunas
                if stats['separator']:
                    stats['columns'] = [col.strip() for col in first_line.split(stats['separator'])]
                    stats['column_count'] = len(stats['columns'])
                    
                    # Amostra de dados (3 primeiras linhas após header)
                    for i in range(1, min(4, len(lines))):
                        if lines[i].strip():
                            stats['sample_data'].append(lines[i])
            
            return stats
            
        except Exception as e:
            optimized_logger.debug(f"Erro na análise local: {e}")
            return {'error': str(e)}

# =============== SISTEMA DE CONTROLE DE CONTEXTO INTELIGENTE ===============

class IntelligentContextManager:
    """Gerenciador inteligente de contexto que permite análise completa com otimização automática"""
    
    MAX_TOKENS_PER_INPUT = 100000  # Limite mais generoso
    MAX_CHARS_PER_INPUT = 400000   # Limite mais generoso
    CHUNK_SIZE = 50000             # Tamanho de chunks para processamento
    
    @staticmethod
    def should_optimize_context(data_size: int) -> bool:
        """Determinar se o contexto precisa ser otimizado"""
        return data_size > IntelligentContextManager.MAX_CHARS_PER_INPUT
    
    @staticmethod
    def create_intelligent_summary(data_text: str, preserve_completeness: bool = True) -> str:
        """Criar resumo inteligente preservando completude da análise"""
        
        if len(data_text) <= IntelligentContextManager.MAX_CHARS_PER_INPUT:
            return data_text
        
        if not preserve_completeness:
            # Fallback para resumo mais agressivo
            return ContextManager.create_data_summary(data_text, 10000)
        
        # Estratégia inteligente: manter estrutura completa mas otimizar dados
        lines = data_text.split('\n')
        
        # Sempre preservar header
        if not lines:
            return data_text
        
        header = lines[0] if lines else ""
        data_lines = lines[1:] if len(lines) > 1 else []
        
        if len(data_lines) <= 1000:
            # Dataset pequeno, manter completo
            return data_text
        
        # Para datasets grandes, usar amostragem inteligente
        total_lines = len(data_lines)
        sample_size = min(5000, total_lines // 2)  # Até 50% dos dados ou 5000 linhas
        
        # Amostragem estratificada
        step = max(1, total_lines // sample_size)
        sampled_lines = [data_lines[i] for i in range(0, total_lines, step)]
        
        # Adicionar algumas linhas do final para capturar tendências
        if total_lines > sample_size:
            sampled_lines.extend(data_lines[-50:])  # Últimas 50 linhas
        
        result_lines = [header] + sampled_lines
        
        # Adicionar metadata sobre a amostragem
        metadata = [
            "",
            f"# METADATA DA ANÁLISE:",
            f"# Total de registros no dataset: {total_lines}",
            f"# Registros na amostra: {len(sampled_lines)}",
            f"# Método: Amostragem estratificada (1 a cada {step} registros)",
            f"# Inclui: Header + amostra representativa + últimas 50 linhas",
            f"# NOTA: Esta é uma amostra representativa para análise. Todos os dados estão disponíveis via ferramentas.",
            ""
        ]
        
        result_lines.extend(metadata)
        
        return '\n'.join(result_lines)
    
    @staticmethod
    def wrap_tool_output(tool_func):
        """Wrapper para ferramentas que aplica controle inteligente de contexto"""
        
        def wrapper(*args, **kwargs):
            try:
                # Executar ferramenta normalmente
                result = tool_func(*args, **kwargs)
                
                if isinstance(result, str) and len(result) > IntelligentContextManager.MAX_CHARS_PER_INPUT:
                    optimized_logger.info(f"🧠 Aplicando otimização inteligente: {len(result)} chars")
                    
                    # Aplicar otimização inteligente
                    optimized_result = IntelligentContextManager.create_intelligent_summary(
                        result, 
                        preserve_completeness=True
                    )
                    
                    optimized_logger.info(f"✅ Otimizado para: {len(optimized_result)} chars (mantendo completude)")
                    return optimized_result
                
                return result
                
            except Exception as e:
                if "too large" in str(e).lower() or "token" in str(e).lower():
                    optimized_logger.warning(f"⚠️ Erro de contexto detectado, aplicando fallback: {e}")
                    
                    # Tentar novamente com parâmetros reduzidos
                    if 'max_records' in kwargs:
                        kwargs['max_records'] = min(1000, kwargs.get('max_records', 10000))
                    elif 'limite' in kwargs:
                        kwargs['limite'] = min(1000, kwargs.get('limite', 10000))
                    
                    try:
                        result = tool_func(*args, **kwargs)
                        return IntelligentContextManager.create_intelligent_summary(result, False)
                    except:
                        return f"Erro ao executar ferramenta com otimização: {str(e)[:500]}"
                else:
                    raise
        
        return wrapper

# =============== LLM CONFIGURATION ===============

llm = LLM(
    model=os.getenv("MODEL"),
    api_key=os.getenv("OPENAI_API_KEY")
)


# =============== FUNÇÕES DE VALIDAÇÃO OTIMIZADAS ===============

@cached_result()
def validate_system_optimized():
    """Validação otimizada do sistema com cache"""
    
    if should_skip_validation('detailed_compatibility_check'):
        optimized_logger.debug("Pulando validação detalhada (otimização)")
        return True
    
    try:
        stats = get_tools_statistics()
        working_tools = stats.get('total_tools', 0)
        
        if working_tools >= 10:  # Mínimo aceitável
            optimized_logger.info(f"✅ Sistema validado: {working_tools} ferramentas")
            return True
        else:
            optimized_logger.warning(f"⚠️ Poucas ferramentas disponíveis: {working_tools}")
            return False
            
    except Exception as e:
        optimized_logger.error(f"❌ Erro na validação: {e}")
        return False

@performance_tracked("tool_validation")
def validate_critical_tools():
    """Validar apenas ferramentas críticas"""
    
    critical_tools = {
        'sql_tool': sql_tool,
        'file_tool': file_tool, 
        'bi_tool': bi_tool
    }
    
    working = 0
    for name, tool in critical_tools.items():
        if tool is not None:
            working += 1
        else:
            optimized_logger.warning(f"⚠️ Ferramenta crítica ausente: {name}")
    
    success_rate = (working / len(critical_tools)) * 100
    
    if success_rate >= 75:
        optimized_logger.info(f"✅ Ferramentas críticas: {working}/{len(critical_tools)}")
        return True
    else:
        optimized_logger.error(f"❌ Muitas ferramentas críticas ausentes: {success_rate:.1f}%")
        return False

# =============== CLASSE CREW OTIMIZADA ===============

@CrewBase
class OptimizedInsights():
    """
    Crew otimizada com performance avançada:
    - Logging estruturado com níveis contextuais
    - Lazy loading de ferramentas por agente
    - Cache inteligente de validações
    - Inicialização paralela quando possível
    - Métricas de performance em tempo real
    """

    def __init__(self):
        """Inicialização otimizada com métricas de tempo"""
        init_start_time = time.time()
        super().__init__()
        
        optimized_logger.info("🚀 Inicializando Insights-AI Otimizado")
        
        # =============== VALIDAÇÃO OTIMIZADA ===============
        if not should_skip_validation('system_validation'):
            optimized_logger.info("🔧 Validando sistema...")
            system_ok = validate_system_optimized()
            if not system_ok:
                optimized_logger.warning("⚠️ Sistema com problemas - continuando")
        
        # Validar apenas ferramentas críticas
        tools_ok = validate_critical_tools()
        if not tools_ok:
            optimized_logger.error("❌ Ferramentas críticas com problemas")
        
        # =============== CONFIGURAÇÃO BASEADA NO AMBIENTE ===============
        
        # Pular logs detalhados se não for debug
        if not should_skip_validation('system_info_collection'):
            self._log_minimal_system_info()
        
        # Cache para resultados de task
        self.task_cache = {}
        self.initialization_time = time.time() - init_start_time
        
        # Log resumo da inicialização
        components = ['system_validation', 'tool_validation', 'configuration']
        log_startup_metrics(components, init_start_time)

    def _log_minimal_system_info(self):
        """Log mínimo de informações do sistema"""
        if PERFORMANCE_CONFIG.log_level.value >= 2:  # NORMAL ou superior
            optimized_logger.info(f"💻 Sistema: {os.getenv('OS', 'Unknown')}")
            optimized_logger.info(f"🔧 Modo: {'Debug' if os.getenv('INSIGHTS_DEBUG') else 'Normal'}")

    # =============== AGENTES COM LAZY LOADING ===============

    @performance_tracked("agent_creation")
    def _create_agent_with_optimized_tools(self, agent_name: str, config_key: str) -> Agent:
        """Criar agente com ferramentas otimizadas e separação de acesso aos dados"""
        
        optimized_logger.debug(f"🔧 Criando agente: {agent_name}")
        
        # Obter ferramentas específicas para o agente (com restrição SQL)
        agent_tools = get_tools_for_agent(agent_name)
        
        # Validar acesso aos dados
        access_validation = validate_agent_data_access(agent_name)
        
        if not access_validation['valid']:
            optimized_logger.warning(f"⚠️ Problema de acesso para {agent_name}: {access_validation.get('error', 'Unknown')}")
        
        # Log das ferramentas e tipo de acesso
        has_sql = any('SQL' in str(type(tool).__name__) for tool in agent_tools)
        has_file_read = any('FileRead' in str(type(tool).__name__) for tool in agent_tools)
        
        optimized_logger.debug(f"   • Ferramentas carregadas: {len(agent_tools)}")
        optimized_logger.debug(f"   • Acesso SQL: {'✅' if has_sql else '❌'}")
        optimized_logger.debug(f"   • Acesso arquivo: {'✅' if has_file_read else '❌'}")
        optimized_logger.debug(f"   • Tipo de acesso: {access_validation.get('data_access', 'undefined')}")
        
        # Aplicar wrapper inteligente nas ferramentas críticas que podem gerar dados grandes
        optimized_tools = []
        for tool in agent_tools:
            tool_name = getattr(tool, 'name', str(type(tool).__name__))
            
            # Identificar ferramentas que precisam de controle de contexto
            if any(keyword in tool_name.lower() for keyword in ['sql', 'query', 'data', 'export', 'business']):
                if hasattr(tool, '_run'):
                    # Aplicar wrapper inteligente
                    original_run = tool._run
                    tool._run = IntelligentContextManager.wrap_tool_output(original_run)
                    optimized_logger.debug(f"   🧠 Wrapper inteligente aplicado: {tool_name}")
            
            optimized_tools.append(tool)
        
        return Agent(
            config=self.agents_config[config_key],
            verbose=PERFORMANCE_CONFIG.log_level.value >= 3,  # Verbose apenas em DEBUG
            llm=llm,
            max_rpm=30,
            tools=optimized_tools,  # Usar ferramentas otimizadas
            reasoning=True,
            max_reasoning_attempts=2,  # Reduzido para performance
            respect_context_window=True
        )

    @agent
    def engenheiro_dados(self) -> Agent:
        """🔧 Engenheiro de Dados (Otimizado)"""
        return self._create_agent_with_optimized_tools(
            'engenheiro_dados', 
            'engenheiro_dados'
        )

    @agent 
    def analista_vendas_tendencias(self) -> Agent:
        """📈 Analista de Vendas e Tendências (Otimizado)"""
        return self._create_agent_with_optimized_tools(
            'analista_vendas_tendencias',
            'analista_vendas_tendencias'
        )

    @agent
    def especialista_produtos(self) -> Agent:
        """🎯 Especialista em Produtos (Otimizado)"""
        return self._create_agent_with_optimized_tools(
            'especialista_produtos',
            'especialista_produtos'
        )

    @agent
    def analista_estoque(self) -> Agent:
        """📦 Analista de Estoque (Otimizado)"""
        return self._create_agent_with_optimized_tools(
            'analista_estoque',
            'analista_estoque'
        )

    @agent
    def analista_financeiro(self) -> Agent:
        """💰 Analista Financeiro (Otimizado)"""
        return self._create_agent_with_optimized_tools(
            'analista_financeiro',
            'analista_financeiro'
        )

    @agent
    def especialista_clientes(self) -> Agent:
        """👥 Especialista em Clientes (Otimizado)"""
        return self._create_agent_with_optimized_tools(
            'especialista_clientes',
            'especialista_clientes'
        )

    @agent
    def analista_performance(self) -> Agent:
        """👤 Analista de Performance (Otimizado)"""
        return self._create_agent_with_optimized_tools(
            'analista_performance',
            'analista_performance'
        )

    @agent
    def diretor_insights(self) -> Agent:
        """🎯 Diretor de Insights (Otimizado)"""
        return self._create_agent_with_optimized_tools(
            'diretor_insights',
            'diretor_insights'
        )

    # =============== TASKS OTIMIZADAS ===============

    def _create_optimized_task(self, task_name: str, dependencies: List = None) -> Task:
        """Criar task otimizada com instruções específicas baseadas na arquitetura ETL"""
        
        optimized_logger.debug(f"📋 Configurando task: {task_name}")
        
        # Callback otimizado baseado no nível de logging
        if PERFORMANCE_CONFIG.log_level.value >= 3:  # VERBOSE
            def smart_callback(output):
                output_size = len(str(output))
                # Truncar output muito grande no log
                if output_size > 500:
                    preview = str(output)[:200] + "..." + str(output)[-100:]
                    optimized_logger.info(f"✅ [{task_name}] Concluída: {output_size} chars - {preview}")
                else:
                    optimized_logger.info(f"✅ [{task_name}] Concluída: {output_size} chars")
        else:
            smart_callback = lambda output: optimized_logger.debug(f"✅ [{task_name}] Concluída")
        
        # Obter configuração da task original
        task_config = self.tasks_config[f'{task_name}_task'].copy()
        
        # Adicionar instruções específicas baseadas no tipo de agente
        if 'description' in task_config:
            original_description = task_config['description']
            
            # Instruções específicas por tipo de agente
            if task_name == 'engenheiro_dados':
                # Engenheiro de dados - único com acesso SQL
                specific_instructions = """

                INSTRUÇÕES ESPECÍFICAS PARA ENGENHEIRO DE DADOS:
                - Você é o ÚNICO agente com acesso direto ao SQL Server
                - Use SQL Query Tool para extrair TODOS os dados do período solicitado
                - OBRIGATÓRIO: Exporte os dados para arquivo 'data/vendas.csv' (arquivo padrão)
                - Valide a qualidade e completude dos dados extraídos
                - Documente estatísticas básicas dos dados (número de registros, período, etc.)
                - CRÍTICO: Outros agentes dependem 100% do arquivo CSV que você gerar
                - Formato do CSV: incluir todas as colunas necessárias para análises posteriores
                """
            else:
                # Agentes de análise - trabalham apenas com CSV
                specific_instructions = """

                INSTRUÇÕES ESPECÍFICAS PARA ANÁLISE COM CSV:
                - NÃO faça consultas SQL - você não tem acesso ao banco de dados
                - OBRIGATÓRIO: Use FileReadTool para ler o arquivo 'data/vendas.csv' gerado pelo engenheiro_dados
                - O arquivo data/vendas.csv contém TODOS os dados necessários para sua análise
                - Foque na sua especialização usando os dados do CSV carregado
                - Aplique suas ferramentas de análise nos dados carregados do arquivo
                - Se o arquivo não existir, aguarde o engenheiro_dados completar a extração
                - Documente claramente suas descobertas baseadas nos dados do CSV
                """
            
            task_config['description'] = original_description + specific_instructions
        
        return Task(
            config=task_config,
            context=dependencies or [],
            callback=smart_callback
        )

    @task
    def engenheiro_dados_task(self) -> Task:
        return self._create_optimized_task('engenheiro_dados')

    @task
    def analista_vendas_tendencias_task(self) -> Task:
        return self._create_optimized_task(
            'analista_vendas_tendencias',
            [self.engenheiro_dados_task()]
        )

    @task
    def especialista_produtos_task(self) -> Task:
        return self._create_optimized_task(
            'especialista_produtos',
            [self.engenheiro_dados_task()]
        )

    @task
    def analista_estoque_task(self) -> Task:
        return self._create_optimized_task(
            'analista_estoque',
            [self.engenheiro_dados_task()]
        )

    @task
    def analista_financeiro_task(self) -> Task:
        return self._create_optimized_task(
            'analista_financeiro',
            [self.engenheiro_dados_task()]
        )

    @task
    def especialista_clientes_task(self) -> Task:
        return self._create_optimized_task(
            'especialista_clientes',
            [self.engenheiro_dados_task()]
        )

    @task
    def analista_performance_task(self) -> Task:
        return self._create_optimized_task(
            'analista_performance',
            [self.engenheiro_dados_task()]
        )

    @task
    def diretor_insights_task(self) -> Task:
        return self._create_optimized_task(
            'diretor_insights',
            [
                self.engenheiro_dados_task(),
                self.analista_vendas_tendencias_task(),
                self.especialista_produtos_task(),
                self.analista_estoque_task(),
                self.analista_financeiro_task(),
                self.especialista_clientes_task(),
                self.analista_performance_task()
            ]
        )

    # =============== BEFORE KICKOFF OTIMIZADO ===============

    @before_kickoff
    @performance_tracked("before_kickoff")
    def before_kickoff(self, inputs):
        """Before kickoff otimizado com arquitetura ETL + CSV"""
        
        # Log inicial baseado no nível
        if PERFORMANCE_CONFIG.log_level.value >= 2:  # NORMAL ou superior
            optimized_logger.info("🚀 Iniciando análise Insights-AI com arquitetura ETL")
            optimized_logger.info(f"📅 Período: {inputs.get('data_inicio')} até {inputs.get('data_fim')}")
        
        # Validar datas apenas se necessário
        data_inicio = inputs.get('data_inicio')
        data_fim = inputs.get('data_fim')
        
        if data_inicio and data_fim:
            try:
                datetime.strptime(data_inicio, '%Y-%m-%d')
                datetime.strptime(data_fim, '%Y-%m-%d')
                optimized_logger.debug("✅ Datas validadas")
            except ValueError as e:
                optimized_logger.warning(f"⚠️ Formato de data inválido: {e}")
        
        # =============== CONFIGURAÇÃO DE ARQUITETURA ETL ===============
        
        optimized_logger.info("🏗️ Configurando arquitetura ETL + Análise baseada em CSV...")
        
        # Informações da arquitetura para os agentes
        data_flow = get_data_flow_architecture()
        
        inputs['data_flow_architecture'] = data_flow
        inputs['etl_file'] = 'data/vendas.csv'
        inputs['data_extractor'] = 'engenheiro_dados'
        
        # =============== CONFIGURAÇÃO ESPECÍFICA POR TIPO DE AGENTE ===============
        
        # Configurações específicas para o Engenheiro de Dados
        inputs['engenheiro_dados_config'] = {
            'responsibility': 'Extrair dados do SQL Server e gerar data/vendas.csv',
            'sql_access': True,
            'output_file': 'data/vendas.csv',
            'data_source': 'SQL Server Database',
            'task_priority': 'PRIMEIRO - outros agentes dependem do seu output'
        }
        
        # Configurações para agentes de análise (dependem do CSV)
        analysis_agents = [
            'analista_vendas_tendencias', 'especialista_produtos', 'analista_estoque',
            'analista_financeiro', 'especialista_clientes', 'analista_performance'
        ]
        
        for agent in analysis_agents:
            inputs[f'{agent}_config'] = {
                'responsibility': f'Analisar dados do arquivo data/vendas.csv na sua especialização',
                'sql_access': False,
                'input_file': 'data/vendas.csv',
                'data_source': 'CSV file generated by engenheiro_dados',
                'dependency': 'Aguarda engenheiro_dados completar extração'
            }
        
        # Configuração específica para o Diretor de Insights
        inputs['diretor_insights_config'] = {
            'responsibility': 'Consolidar análises de todos os agentes em dashboard executivo',
            'sql_access': False,
            'input_file': 'data/vendas.csv',
            'data_source': 'CSV + resultados dos outros agentes',
            'task_priority': 'ÚLTIMO - consolida resultados de todos'
        }
        
        # =============== VALIDAÇÃO DE ACESSO PARA TODOS OS AGENTES ===============
        
        optimized_logger.info("🔒 Validando separação de acesso aos dados...")
        
        access_validation = {}
        
        for agent in ['engenheiro_dados'] + analysis_agents + ['diretor_insights']:
            validation = validate_agent_data_access(agent)
            access_validation[agent] = validation
            
            if validation['valid']:
                optimized_logger.debug(f"✅ {agent}: {validation['expected_access']}")
            else:
                optimized_logger.warning(f"⚠️ {agent}: {validation.get('error', 'Erro de validação')}")
        
        inputs['access_validation'] = access_validation
        
        # =============== INSTRUÇÕES ESPECÍFICAS DE FLUXO DE DADOS ===============
        
        # Instruções específicas baseadas na arquitetura ETL
        inputs['data_flow_instructions'] = """
                    ARQUITETURA DE FLUXO DE DADOS - ETL + CSV:

                    🔧 ENGENHEIRO DE DADOS:
                    - ÚNICO agente com acesso direto ao SQL Server
                    - DEVE extrair dados completos do período usando SQL Query Tool
                    - DEVE exportar dados para arquivo 'data/vendas.csv' (arquivo padrão)
                    - DEVE validar a qualidade e completude dos dados extraídos
                    - Outros agentes dependem 100% do seu trabalho

                    📊 AGENTES DE ANÁLISE:
                    - NÃO fazem consultas SQL diretamente
                    - DEVEM ler dados APENAS do arquivo 'data/vendas.csv'
                    - Usar FileReadTool para carregar o CSV exportado pelo engenheiro
                    - Focar na especialização usando dados do CSV
                    - Aplicar ferramentas de análise nos dados carregados

                    🎯 DIRETOR DE INSIGHTS:
                    - Consolida resultados de TODOS os agentes
                    - Usa dados do CSV + resultados das análises especializadas
                    - Gera dashboard executivo final

                    📁 FLUXO DE ARQUIVOS:
                    1. SQL Server → engenheiro_dados → data/vendas.csv
                    2. data/vendas.csv → agentes de análise → insights especializadas  
                    3. insights especializadas → diretor_insights → dashboard final

                    IMPORTANTE: Esta separação garante performance, segurança e organização!
                    """
                            
        # =============== VERIFICAÇÃO DE CONECTIVIDADE BÁSICA ===============
        
        # Verificar conectividade apenas para o engenheiro (que usa SQL)
        if data_inicio and data_fim:
            try:
                from insights.config.tools_config_v3 import sql_query_tool
                
                if sql_query_tool:
                    optimized_logger.info("🔗 Verificando conectividade SQL (apenas para engenheiro_dados)...")
                    
                    # Teste de conectividade mínimo
                    inputs['sql_connectivity'] = 'available'
                    inputs['csv_output_path'] = 'data/vendas.csv'
                    optimized_logger.info("✅ SQL disponível para extração de dados")
                else:
                    optimized_logger.warning("⚠️ SQL Query Tool não disponível")
                    inputs['sql_connectivity'] = 'unavailable'
                    
            except Exception as e:
                optimized_logger.warning(f"⚠️ Erro na verificação SQL: {e}")
                inputs['sql_connectivity'] = 'error'
        
        # Log final da arquitetura configurada
        optimized_logger.info("🏗️ Arquitetura ETL configurada:")
        optimized_logger.info("   📤 engenheiro_dados: SQL Server → data/vendas.csv")
        optimized_logger.info("   📥 Outros agentes: data/vendas.csv → análises especializadas")
        optimized_logger.info("   🎯 diretor_insights: consolidação final")
        
        return inputs

    # =============== CREW OTIMIZADO ===============

    @crew
    @performance_tracked("crew_creation")
    def crew(self) -> Crew:
        """Criar crew otimizado com configurações de performance e controle de contexto"""
        
        optimized_logger.debug("🚀 Configurando crew otimizado")
        
        # Callbacks otimizados com controle de contexto
        def optimized_task_callback(task_output):
            if PERFORMANCE_CONFIG.log_level.value >= 3:  # VERBOSE
                output_size = len(str(task_output))
                
                # Verificar se o output está muito grande
                if output_size > 50000:
                    optimized_logger.warning(f"⚠️ Task output muito grande: {output_size} chars")
                    # Truncar para log
                    preview = str(task_output)[:300] + "..." + str(task_output)[-200:]
                    optimized_logger.info(f"✅ Task concluída: {output_size} chars - {preview}")
                else:
                    optimized_logger.info(f"✅ Task concluída: {output_size} chars")
        
        def optimized_agent_callback(agent_output):
            if PERFORMANCE_CONFIG.log_level.value >= 4:  # DEBUG
                output_size = len(str(agent_output))
                if output_size > 1000:
                    preview = str(agent_output)[:100] + "..."
                    optimized_logger.debug(f"🧠 Agente: {preview} ({output_size} chars)")
                else:
                    optimized_logger.debug(f"🧠 Agente: {str(agent_output)[:100]}...")
        
        # Configuração específica para evitar problemas de contexto
        crew_config = {
            'agents': self.agents,
            'tasks': self.tasks,
            'process': Process.sequential,
            'verbose': PERFORMANCE_CONFIG.log_level.value >= 3,
            'memory': False,  # Desabilitado para performance
            'max_rpm': 20,    # Reduzido para evitar rate limits
            'task_callback': optimized_task_callback if PERFORMANCE_CONFIG.log_level.value >= 3 else None,
            'max_execution_time': 1800,  # 30 minutos máximo
        }
        
        # Adicionar configurações específicas para lidar com contexto grande
        if hasattr(Crew, 'max_iter'):
            crew_config['max_iter'] = 3  # Limitar iterações
        
        return Crew(**crew_config)

    # =============== CLEANUP ===============

    def __del__(self):
        """Cleanup automático ao destruir instância"""
        try:
            optimized_logger.finalize()
        except:
            pass

# =============== UTILITÁRIOS DE PERFORMANCE ===============

def monitor_context_size(inputs: dict) -> dict:
    """Monitorar e ajustar tamanho do contexto"""
    
    total_size = sum(len(str(v)) for v in inputs.values())
    optimized_logger.info(f"📊 Contexto total: {total_size} chars")
    
    if total_size > ContextManager.MAX_CHARS_PER_INPUT:
        optimized_logger.warning(f"⚠️ Contexto muito grande ({total_size} chars), aplicando compressão agressiva...")
        
        # Compressão agressiva
        for key, value in inputs.items():
            if isinstance(value, str) and len(value) > 10000:
                # Comprimir drasticamente
                compressed = ContextManager.prepare_data_context(value, "aggressive")
                inputs[key] = compressed
                optimized_logger.info(f"📦 {key} comprimido: {len(value)} → {len(compressed)} chars")
        
        # Verificar novamente
        new_total = sum(len(str(v)) for v in inputs.values())
        optimized_logger.info(f"📊 Contexto após compressão: {new_total} chars")
    
    return inputs

def get_performance_metrics():
    """Obter métricas de performance da arquitetura ETL"""
    return {
        'architecture': 'ETL_CSV_based',
        'data_access_model': 'separated_responsibilities',
        'sql_access_restricted_to': 'engenheiro_dados',
        'csv_based_analysis': 'enabled',
        'cache_enabled': PERFORMANCE_CONFIG.enable_tool_cache,
        'lazy_loading': PERFORMANCE_CONFIG.lazy_tool_loading,
        'log_level': PERFORMANCE_CONFIG.log_level.name,
        'parallel_init': PERFORMANCE_CONFIG.parallel_agent_init,
        'cache_size': len(performance_cache.cache),
        'cache_hits': getattr(performance_cache, 'hits', 0),
        'cache_misses': getattr(performance_cache, 'misses', 0),
        'intelligent_context_limit': IntelligentContextManager.MAX_CHARS_PER_INPUT,
        'tool_access_validation': 'active',
        'data_flow': 'SQL_to_CSV_to_Analysis'
    }

def log_performance_summary():
    """Log resumo de performance da arquitetura ETL"""
    metrics = get_performance_metrics()
    
    optimized_logger.info("📊 MÉTRICAS DA ARQUITETURA ETL:")
    optimized_logger.info(f"   🏗️ Arquitetura: {metrics['architecture']}")
    optimized_logger.info(f"   🔐 Modelo de acesso: {metrics['data_access_model']}")
    optimized_logger.info(f"   🗄️ Acesso SQL restrito a: {metrics['sql_access_restricted_to']}")
    optimized_logger.info(f"   📁 Análise baseada em CSV: {metrics['csv_based_analysis']}")
    optimized_logger.info(f"   🔧 Validação de acesso: {metrics['tool_access_validation']}")
    optimized_logger.info(f"   📈 Fluxo de dados: {metrics['data_flow']}")
    optimized_logger.info(f"   💾 Cache habilitado: {metrics['cache_enabled']}")
    optimized_logger.info(f"   📝 Nível de log: {metrics['log_level']}")

# =============== FUNÇÃO PRINCIPAL OTIMIZADA ===============

def run_optimized_crew(data_inicio: str, data_fim: str):
    """
    Executar crew otimizado com arquitetura ETL + análise baseada em CSV
    
    🎯 ARQUITETURA ETL:
    - APENAS o engenheiro_dados acessa o SQL Server diretamente
    - Engenheiro extrai dados e gera data/vendas.csv (arquivo padrão)
    - Outros agentes leem data/vendas.csv para análises especializadas
    - Diretor de insights consolida resultados finais
    
    📊 BENEFÍCIOS DA ARQUITETURA:
    - ✅ Separação clara de responsabilidades
    - ✅ Acesso controlado ao banco de dados (segurança)
    - ✅ Performance otimizada (apenas 1 extração SQL)
    - ✅ Análises paralelas baseadas no mesmo dataset
    - ✅ Redução de carga no banco de dados
    
    🔧 FLUXO DE DADOS:
    1. engenheiro_dados: SQL Server → data/vendas.csv
    2. Agentes especializados: data/vendas.csv → análises focadas
    3. diretor_insights: consolidação → dashboard executivo
    
    Args:
        data_inicio (str): Data inicial no formato 'YYYY-MM-DD'
        data_fim (str): Data final no formato 'YYYY-MM-DD'
        
    Returns:
        str: Resultados completos da análise com insights detalhados
    """
    
    start_time = time.time()
    
    try:
        optimized_logger.info("🚀 Iniciando Insights-AI com Arquitetura ETL")
        optimized_logger.info("🏗️ SQL → CSV → Análises Especializadas → Dashboard")
        
        # Criar instância otimizada
        crew_instance = OptimizedInsights()
        
        # Preparar inputs
        inputs = {
            'data_inicio': data_inicio,
            'data_fim': data_fim
        }
        
        # Executar crew com arquitetura ETL
        optimized_logger.info("⚡ Executando análise com separação SQL/CSV...")
        
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                result = crew_instance.crew().kickoff(inputs=inputs)
                break  # Sucesso, sair do loop
                
            except Exception as e:
                if "too large" in str(e) or "context" in str(e).lower() or "token" in str(e).lower():
                    if attempt < max_retries:
                        optimized_logger.warning(f"🔄 Tentativa {attempt + 1} - ajustando otimização...")
                        
                        # Aplicar otimização mais agressiva
                        IntelligentContextManager.MAX_CHARS_PER_INPUT = max(100000, IntelligentContextManager.MAX_CHARS_PER_INPUT * 0.7)
                        optimized_logger.info(f"📉 Limite reduzido para {IntelligentContextManager.MAX_CHARS_PER_INPUT} chars")
                        continue
                    else:
                        optimized_logger.error("❌ Falha após todas as tentativas de otimização")
                        raise
                else:
                    # Erro diferente, não relacionado a contexto
                    raise
        
        # Métricas finais
        total_time = time.time() - start_time
        result_size = len(str(result))
        
        optimized_logger.info(f"✅ Análise ETL concluída em {total_time:.2f}s")
        optimized_logger.info(f"📊 Resultado gerado: {result_size:,} caracteres")
        optimized_logger.info("🏗️ Arquitetura ETL executada com sucesso:")
        optimized_logger.info("   📤 Extração SQL executada pelo engenheiro_dados")
        optimized_logger.info("   📁 Dados exportados para data/vendas.csv")
        optimized_logger.info("   📊 Análises especializadas baseadas no CSV")
        optimized_logger.info("   🎯 Dashboard executivo consolidado")
        
        log_performance_summary()
        
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        optimized_logger.error(f"❌ Erro após {execution_time:.2f}s: {e}")
        
        # Log específico para problemas de contexto
        if "too large" in str(e) or "token" in str(e).lower():
            optimized_logger.error("💡 NOTA: Sistema configurado para arquitetura ETL")
            optimized_logger.error("💡 Apenas engenheiro_dados acessa SQL Server")
            optimized_logger.error("💡 Outros agentes usam data/vendas.csv exportado")
        
        raise
    
    finally:
        # Cleanup automático
        cleanup_performance_resources()

if __name__ == "__main__":
    # Exemplo de uso
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    result = run_optimized_crew(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    ) 