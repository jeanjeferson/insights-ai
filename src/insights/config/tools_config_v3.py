"""
🔧 CONFIGURAÇÃO COMPLETA DAS FERRAMENTAS V3.0 CONSOLIDADAS
==========================================================

Este arquivo configura TODAS as ferramentas do sistema CrewAI de forma centralizada.
Incluindo ferramentas básicas, avançadas, de exportação e geração.
"""

# =============== IMPORTAÇÕES DE TODAS AS FERRAMENTAS ===============

# Ferramentas básicas
# from crewai_tools import FileReadTool  # Comentada para usar versão customizada
from ..tools.sql_query_tool import SQLServerQueryTool
from ..tools.duckduck_tool import DuckDuckGoSearchTool
from ..tools.prophet_tool import ProphetForecastTool
from ..tools.business_intelligence_tool import BusinessIntelligenceTool

# Ferramentas de análise core
from ..tools.kpi_calculator_tool import KPICalculatorTool
from ..tools.statistical_analysis_tool import StatisticalAnalysisTool

# Ferramentas avançadas de IA/ML
from ..tools.advanced.customer_insights_engine import CustomerInsightsEngine
from ..tools.advanced.recommendation_engine import RecommendationEngine
from ..tools.advanced.advanced_analytics_engine_tool import AdvancedAnalyticsEngineTool
from ..tools.advanced.risk_assessment_tool import RiskAssessmentTool
from ..tools.advanced.competitive_intelligence_tool import CompetitiveIntelligenceTool

# Ferramenta de geração de arquivos
from ..tools.file_generation_tool import FileGenerationTool

# Ferramentas especializadas de exportação de dados
from ..tools.product_data_exporter import ProductDataExporter
from ..tools.inventory_data_exporter import InventoryDataExporter
from ..tools.customer_data_exporter import CustomerDataExporter
from ..tools.financial_data_exporter import FinancialDataExporter

# =============== FERRAMENTA FILE READ CUSTOMIZADA ===============

from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field
import os

class CustomFileReadToolInput(BaseModel):
    """Schema para ferramenta de leitura de arquivos customizada."""
    
    file_path: str = Field(
        ..., 
        description="Caminho completo para o arquivo a ser lido",
        json_schema_extra={"example": "data/vendas.csv"}
    )
    
    max_lines: Optional[int] = Field(
        default=3, 
        description="Número máximo de linhas a serem lidas (padrão: 3)",
        json_schema_extra={"example": 3}
    )

class CustomFileReadTool(BaseTool):
    """
    📁 FERRAMENTA CUSTOMIZADA PARA LEITURA DE ARQUIVOS
    
    Esta ferramenta substitui a FileReadTool padrão com controle de linhas:
    - Limite padrão: 3 linhas (ao invés de 50)
    - Limite configurável por chamada
    - Otimizada para evitar sobrecarga de contexto
    - Mantém compatibilidade com o sistema ETL
    
    QUANDO USAR:
    - Ler arquivo data/vendas.csv exportado pelo engenheiro_dados
    - Verificar estrutura de dados rapidamente
    - Obter amostra dos dados para análise
    - Mantém compatibilidade com o sistema ETL
    """
    
    name: str = "Custom File Read Tool"
    description: str = (
        "Ferramenta customizada para leitura de arquivos com limite configurável de linhas. "
        "Padrão: 3 linhas. Use para ler data/vendas.csv e outros arquivos de dados."
    )
    args_schema: Type[BaseModel] = CustomFileReadToolInput
    
    def _run(self, file_path: str, max_lines: int = 3) -> str:
        try:
            # Verificar se arquivo existe
            if not os.path.exists(file_path):
                return f"❌ Arquivo não encontrado: {file_path}"
            
            # Ler arquivo com limite de linhas
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = []
                for i, line in enumerate(file):
                    if i >= max_lines:
                        break
                    lines.append(line.rstrip('\n'))
            
            if not lines:
                return f"📄 Arquivo está vazio: {file_path}"
            
            # Contar total de linhas no arquivo
            with open(file_path, 'r', encoding='utf-8') as file:
                total_lines = sum(1 for _ in file)
            
            # Formatar resultado
            result = f"📁 Arquivo: {file_path}\n"
            result += f"📊 Mostrando {len(lines)} de {total_lines} linhas totais\n"
            result += f"{'='*50}\n"
            
            for i, line in enumerate(lines, 1):
                result += f"Linha {i}: {line}\n"
            
            if total_lines > max_lines:
                result += f"{'='*50}\n"
                result += f"📝 NOTA: Arquivo contém {total_lines - max_lines} linhas adicionais.\n"
                result += f"💡 Para análise completa, use outras ferramentas especializadas.\n"
            
            return result
            
        except UnicodeDecodeError:
            try:
                # Tentar com encoding alternativo
                with open(file_path, 'r', encoding='latin-1') as file:
                    lines = [file.readline().rstrip('\n') for _ in range(max_lines)]
                return f"📁 {file_path} (encoding: latin-1)\n" + '\n'.join(f"Linha {i+1}: {line}" for i, line in enumerate(lines))
            except Exception as e:
                return f"❌ Erro de encoding ao ler {file_path}: {str(e)}"
        except Exception as e:
            return f"❌ Erro ao ler arquivo {file_path}: {str(e)}"

# =============== INSTANCIAÇÃO CENTRALIZADA DE TODAS AS FERRAMENTAS ===============

# Ferramentas básicas
file_read_tool = CustomFileReadTool()  # Usando versão customizada
sql_query_tool = SQLServerQueryTool()
search_tool = DuckDuckGoSearchTool()
prophet_forecast_tool = ProphetForecastTool()
business_intelligence_tool = BusinessIntelligenceTool()

# Ferramentas de análise core (originais do v3)
kpi_calculator_tool = KPICalculatorTool()
statistical_analysis_tool = StatisticalAnalysisTool()

# Ferramentas avançadas de IA/ML
customer_insights_engine = CustomerInsightsEngine()
recommendation_engine = RecommendationEngine()
advanced_analytics_engine = AdvancedAnalyticsEngineTool()
risk_assessment_tool = RiskAssessmentTool()
competitive_intelligence_tool = CompetitiveIntelligenceTool()

# Ferramenta de geração
file_generation_tool = FileGenerationTool()

# Ferramentas de exportação
product_data_exporter = ProductDataExporter()
inventory_data_exporter = InventoryDataExporter()
customer_data_exporter = CustomerDataExporter()
financial_data_exporter = FinancialDataExporter()

# =============== ORGANIZAÇÃO COMPLETA POR CATEGORIAS ===============

# Lista completa de todas as ferramentas
TOOLS = [
    # Básicas
    file_read_tool,
    sql_query_tool,
    search_tool,
    prophet_forecast_tool,
    business_intelligence_tool,
    
    # Análise core
    kpi_calculator_tool,
    statistical_analysis_tool,
    
    # Avançadas IA/ML
    customer_insights_engine,
    recommendation_engine,
    advanced_analytics_engine,
    risk_assessment_tool,
    competitive_intelligence_tool,
    
    # Geração
    file_generation_tool,
    
    # Exportação
    product_data_exporter,
    inventory_data_exporter,
    customer_data_exporter,
    financial_data_exporter
]

# Mapeamento expandido de ferramentas por categoria
TOOLS_BY_CATEGORY = {
    # Categorias básicas
    'basic_tools': [
        file_read_tool,
        sql_query_tool,
        search_tool,
        prophet_forecast_tool,
        business_intelligence_tool
    ],
    
    # Análise de negócios
    'business_kpis': [kpi_calculator_tool],
    'statistical_analysis': [statistical_analysis_tool],
    
    # Ferramentas avançadas de IA
    'advanced_ai': [
        customer_insights_engine,
        recommendation_engine,
        advanced_analytics_engine,
        risk_assessment_tool,
        competitive_intelligence_tool
    ],
    
    # Geração de arquivos e dashboards
    'file_generation': [file_generation_tool],
    
    # Exportação especializada
    'data_export': [
        product_data_exporter,
        inventory_data_exporter,
        customer_data_exporter,
        financial_data_exporter
    ],
    
    # Categorias combinadas para tipos de análise
    'data_engineering': [sql_query_tool, file_read_tool],
    'financial_analysis': [kpi_calculator_tool, advanced_analytics_engine, prophet_forecast_tool, competitive_intelligence_tool, risk_assessment_tool],
    'customer_analysis': [customer_insights_engine, statistical_analysis_tool, advanced_analytics_engine, recommendation_engine],
    'product_analysis': [advanced_analytics_engine, statistical_analysis_tool, recommendation_engine, risk_assessment_tool],
    'forecasting': [prophet_forecast_tool, statistical_analysis_tool, advanced_analytics_engine],
    'comprehensive': TOOLS,  # Todas as ferramentas
    'all': TOOLS  # Alias para comprehensive
}

# =============== CONFIGURAÇÃO AVANÇADA DE INTEGRAÇÃO ===============

INTEGRATION_CONFIG = {
    # Integração KPI-Statistical (original)
    'kpi_statistical_integration': {
        'enabled': True,
        'statistical_insights_for_kpi': True,
        'shared_cache': True,
        'cross_validation': True
    },
    
    # Nova: Integração Advanced Analytics
    'advanced_analytics_integration': {
        'enabled': True,
        'ml_insights_for_kpi': True,
        'clustering_for_segments': True,
        'risk_assessment_integration': True
    },
    
    # Nova: Integração Customer Intelligence
    'customer_intelligence_integration': {
        'enabled': True,
        'rfm_with_ml_clustering': True,
        'recommendation_engine_integration': True,
        'geographic_analysis': True
    },
    
    # Nova: Integração de Exportação
    'export_integration': {
        'enabled': True,
        'auto_csv_generation': True,
        'specialized_formats': True,
        'cross_tool_data_sharing': True
    },
    
    # Nova: Integração Business Intelligence
    'bi_integration': {
        'enabled': True,
        'dashboard_auto_generation': True,
        'interactive_charts': True,
        'real_time_updates': True
    },
    
    # Configuração de cache compartilhado
    'shared_cache': {
        'enabled': True,
        'cache_duration': 3600,  # 1 hora
        'shared_between_tools': [
            'kpi_calculator_tool',
            'statistical_analysis_tool',
            'advanced_analytics_engine'
        ]
    },
    
    # Módulos compartilhados expandidos
    'shared_modules': {
        'data_preparation': 'src.insights.tools.shared.data_preparation',
        'report_formatter': 'src.insights.tools.shared.report_formatter', 
        'business_mixins': 'src.insights.tools.shared.business_mixins',
        'ml_utilities': 'src.insights.tools.shared.ml_utilities',
        'export_utilities': 'src.insights.tools.shared.export_utilities'
    }
}

# =============== MAPEAMENTO EXPANDIDO DE MIGRAÇÕES ===============

MIGRATION_MAP = {
    'removed_from_kpi_tool': [
        'Análises demográficas completas -> StatisticalAnalysisTool',
        'Análises geográficas completas -> StatisticalAnalysisTool',
        'Clustering básico -> AdvancedAnalyticsEngineTool',
        'Correlações simples -> StatisticalAnalysisTool',
        'Previsões básicas -> ProphetForecastTool'
    ],
    'removed_from_statistical_tool': [
        'KPIs básicos de negócio -> KPICalculatorTool',
        'Cálculos de margem e ROI -> KPICalculatorTool', 
        'Benchmarks do setor -> CompetitiveIntelligenceTool',
        'Alertas automáticos -> RiskAssessmentTool',
        'Recomendações ML -> RecommendationEngine'
    ],
    'consolidated_in_advanced_tools': [
        'Machine Learning clustering -> AdvancedAnalyticsEngineTool',
        'Segmentação RFM avançada -> CustomerInsightsEngine',
        'Market basket analysis -> RecommendationEngine',
        'Análise de riscos -> RiskAssessmentTool',
        'Inteligência competitiva -> CompetitiveIntelligenceTool'
    ],
    'new_specialized_exports': [
        'Exportação de produtos ABC/BCG -> ProductDataExporter',
        'Exportação de estoque otimizado -> InventoryDataExporter',
        'Exportação de clientes RFM/CLV -> CustomerDataExporter',
        'Exportação financeira KPIs -> FinancialDataExporter'
    ],
    'consolidated_in_shared': [
        'Preparação de dados -> DataPreparationMixin',
        'Formatação de relatórios -> ReportFormatterMixin',
        'Análises RFM -> JewelryRFMAnalysisMixin',
        'Análises BCG/ABC -> JewelryBusinessAnalysisMixin',
        'Benchmarks do setor -> JewelryBenchmarkMixin',
        'Utilidades ML -> MLUtilitiesMixin',
        'Utilidades de exportação -> ExportUtilitiesMixin'
    ]
}

# =============== CONFIGURAÇÃO DE AGENTES ESPECIALIZADA ===============

AGENT_TOOL_MAPPING = {
    'engenheiro_dados': {
        'primary_tools': ['sql_query_tool', 'file_generation_tool'],  # Único com acesso SQL + geração de arquivos
        'analysis_type': 'data_engineering',
        'specialization': 'ETL, consultas SQL e exportação de dados',
        'data_access': 'database',  # Acesso direto ao banco
        'output_files': ['data/vendas.csv']  # Arquivo que deve gerar
    },
    'analista_vendas_tendencias': {
        'primary_tools': ['statistical_analysis_tool', 'file_read_tool', 'business_intelligence_tool'],
        'analysis_type': 'statistical_analysis',
        'specialization': 'Análise de tendências e padrões de vendas',
        'data_access': 'csv_files',  # Trabalha com arquivos CSV
        'input_files': ['data/vendas.csv']  # Arquivo que precisa ler
    },
    'especialista_produtos': {
        'primary_tools': ['advanced_analytics_engine', 'statistical_analysis_tool', 'file_read_tool'],
        'analysis_type': 'product_analysis',
        'specialization': 'Análise de produtos, ABC e cross-sell',
        'data_access': 'csv_files',
        'input_files': ['data/vendas.csv']
    },
    'analista_estoque': {
        'primary_tools': ['kpi_calculator_tool', 'risk_assessment_tool', 'file_read_tool'],
        'analysis_type': 'business_kpis',
        'specialization': 'Otimização e gestão de estoque',
        'data_access': 'csv_files',
        'input_files': ['data/vendas.csv']
    },
    'analista_financeiro': {
        'primary_tools': ['kpi_calculator_tool', 'advanced_analytics_engine', 'file_read_tool'],
        'analysis_type': 'financial_analysis',
        'specialization': 'KPIs financeiros e análise de margens',
        'data_access': 'csv_files',
        'input_files': ['data/vendas.csv']
    },
    'especialista_clientes': {
        'primary_tools': ['customer_insights_engine', 'advanced_analytics_engine', 'file_read_tool'],
        'analysis_type': 'customer_analysis',
        'specialization': 'Análise RFM, CLV e segmentação',
        'data_access': 'csv_files',
        'input_files': ['data/vendas.csv']
    },
    'analista_performance': {
        'primary_tools': ['statistical_analysis_tool', 'kpi_calculator_tool', 'file_read_tool'],
        'analysis_type': 'statistical_analysis',
        'specialization': 'Performance de vendedores e métricas',
        'data_access': 'csv_files',
        'input_files': ['data/vendas.csv']
    },
    'diretor_insights': {
        'primary_tools': ['business_intelligence_tool', 'competitive_intelligence_tool', 'file_read_tool'],
        'analysis_type': 'comprehensive',
        'specialization': 'Visão estratégica e dashboards executivos',
        'data_access': 'csv_files',
        'input_files': ['data/vendas.csv']
    }
}

# =============== CATEGORIAS REORGANIZADAS POR ACESSO A DADOS ===============

# Ferramentas exclusivas do Engenheiro de Dados (acesso direto ao banco)
DATA_ENGINEERING_TOOLS = [
    sql_query_tool,
    file_generation_tool,
    file_read_tool  # Para validar exports
]

# Ferramentas para análise de arquivos CSV (todos os outros agentes)
CSV_ANALYSIS_TOOLS = [
    file_read_tool,  # Essencial para ler vendas.csv
    business_intelligence_tool,  # Para dashboards e visualizações
    statistical_analysis_tool,  # Para análises estatísticas
    kpi_calculator_tool,  # Para cálculos de KPIs
    advanced_analytics_engine,  # Para ML e analytics avançados
    customer_insights_engine,  # Para análise de clientes
    recommendation_engine,  # Para recomendações
    risk_assessment_tool,  # Para análise de riscos
    competitive_intelligence_tool,  # Para inteligência competitiva
    prophet_forecast_tool,  # Para previsões
]

# Ferramentas de exportação especializada (opcionais para análises específicas)
EXPORT_TOOLS = [
    product_data_exporter,
    inventory_data_exporter,
    customer_data_exporter,
    financial_data_exporter
]

# Atualizar mapeamento por categoria
TOOLS_BY_CATEGORY.update({
    # Categoria exclusiva do engenheiro
    'data_engineering': DATA_ENGINEERING_TOOLS,
    
    # Categoria para análise de CSV
    'csv_analysis': CSV_ANALYSIS_TOOLS,
    
    # Ferramentas específicas por tipo de análise (sem SQL)
    'financial_analysis': [
        file_read_tool, kpi_calculator_tool, advanced_analytics_engine, 
        prophet_forecast_tool, competitive_intelligence_tool, risk_assessment_tool
    ],
    'customer_analysis': [
        file_read_tool, customer_insights_engine, statistical_analysis_tool, 
        advanced_analytics_engine, recommendation_engine
    ],
    'product_analysis': [
        file_read_tool, advanced_analytics_engine, statistical_analysis_tool, 
        recommendation_engine, risk_assessment_tool
    ],
    'statistical_analysis': [
        file_read_tool, statistical_analysis_tool, business_intelligence_tool, 
        prophet_forecast_tool
    ],
    'business_kpis': [
        file_read_tool, kpi_calculator_tool, business_intelligence_tool, 
        risk_assessment_tool
    ]
})

# =============== FUNÇÕES EXPANDIDAS DE SELEÇÃO ===============

def get_tools_for_analysis(analysis_type: str):
    """
    Retorna as ferramentas apropriadas para um tipo de análise.
    Versão expandida com suporte a todos os tipos.
    
    Args:
        analysis_type: Tipo de análise desejada
        
    Returns:
        Lista de ferramentas apropriadas
    """
    # Mapeamento direto das categorias
    if analysis_type in TOOLS_BY_CATEGORY:
        return TOOLS_BY_CATEGORY[analysis_type]
    
    # Mapeamento de aliases e variações
    analysis_mapping = {
        'kpi': 'business_kpis',
        'business': 'business_kpis',
        'financial': 'financial_analysis',
        'operational': 'business_kpis',
        'statistical': 'statistical_analysis',
        'demographic': 'customer_analysis',
        'geographic': 'customer_analysis',
        'clustering': 'advanced_ai',
        'ml': 'advanced_ai',
        'ai': 'advanced_ai',
        'export': 'data_export',
        'dashboard': 'file_generation',
        'comprehensive': 'comprehensive',
        'full': 'comprehensive',
        'complete': 'comprehensive',
        'all': 'all'
    }
    
    mapped_type = analysis_mapping.get(analysis_type.lower(), 'comprehensive')
    return TOOLS_BY_CATEGORY.get(mapped_type, TOOLS)

def get_tools_for_agent(agent_name: str):
    """
    Retorna ferramentas específicas para um agente baseado em sua configuração de acesso.
    IMPORTANTE: Apenas o engenheiro_dados tem acesso ao SQL Query Tool.
    
    Args:
        agent_name: Nome do agente
        
    Returns:
        Lista de ferramentas configuradas para o agente
    """
    if agent_name not in AGENT_TOOL_MAPPING:
        # Fallback: ferramentas CSV (sem SQL)
        return CSV_ANALYSIS_TOOLS
    
    agent_config = AGENT_TOOL_MAPPING[agent_name]
    data_access = agent_config.get('data_access', 'csv_files')
    
    # REGRA CRÍTICA: Apenas engenheiro_dados pode acessar banco SQL
    if agent_name == 'engenheiro_dados' and data_access == 'database':
        # Engenheiro de dados: SQL + ferramentas de exportação
        return DATA_ENGINEERING_TOOLS + [business_intelligence_tool]
    else:
        # Todos os outros agentes: apenas ferramentas CSV
        analysis_type = agent_config['analysis_type']
        
        # Obter ferramentas base (sem SQL)
        if analysis_type in TOOLS_BY_CATEGORY:
            base_tools = TOOLS_BY_CATEGORY[analysis_type]
        else:
            base_tools = CSV_ANALYSIS_TOOLS
        
        # GARANTIR que SQL Query Tool nunca seja incluído para outros agentes
        filtered_tools = [tool for tool in base_tools if tool != sql_query_tool]
        
        # Sempre garantir que file_read_tool está incluído (para ler vendas.csv)
        if file_read_tool not in filtered_tools:
            filtered_tools.append(file_read_tool)
        
        return filtered_tools

def validate_agent_data_access(agent_name: str):
    """
    Valida se um agente tem o acesso correto aos dados.
    
    Args:
        agent_name: Nome do agente
        
    Returns:
        Dicionário com informações de validação
    """
    if agent_name not in AGENT_TOOL_MAPPING:
        return {
            'valid': False,
            'error': f'Agente {agent_name} não encontrado no mapeamento',
            'recommended_access': 'csv_files'
        }
    
    agent_config = AGENT_TOOL_MAPPING[agent_name]
    agent_tools = get_tools_for_agent(agent_name)
    
    has_sql = sql_query_tool in agent_tools
    has_file_read = file_read_tool in agent_tools
    is_engineer = agent_name == 'engenheiro_dados'
    
    validation = {
        'agent_name': agent_name,
        'data_access': agent_config.get('data_access'),
        'has_sql_access': has_sql,
        'has_file_read': has_file_read,
        'is_data_engineer': is_engineer,
        'tools_count': len(agent_tools),
        'specialization': agent_config.get('specialization')
    }
    
    # Validar regras de acesso
    if is_engineer:
        validation['valid'] = has_sql and has_file_read
        validation['expected_access'] = 'database + file generation'
        if not validation['valid']:
            validation['error'] = 'Engenheiro deve ter acesso SQL e file_read'
    else:
        validation['valid'] = not has_sql and has_file_read
        validation['expected_access'] = 'csv files only'
        if has_sql:
            validation['error'] = f'Agente {agent_name} não deve ter acesso SQL'
        elif not has_file_read:
            validation['error'] = f'Agente {agent_name} deve ter file_read para CSV'
    
    return validation

def get_data_flow_architecture():
    """
    Retorna a arquitetura de fluxo de dados do sistema.
    
    Returns:
        Dicionário descrevendo o fluxo de dados
    """
    return {
        'data_source': 'SQL Server Database',
        'data_extractor': 'engenheiro_dados',
        'extracted_file': 'data/vendas.csv',
        'file_consumers': [agent for agent in AGENT_TOOL_MAPPING.keys() if agent != 'engenheiro_dados'],
        'architecture_type': 'ETL + File-based Analysis',
        'sql_access_restricted_to': ['engenheiro_dados'],
        'file_access_required_for': [agent for agent in AGENT_TOOL_MAPPING.keys() if agent != 'engenheiro_dados'],
        'workflow': [
            '1. engenheiro_dados extrai dados do SQL Server',
            '2. engenheiro_dados gera data/vendas.csv',
            '3. Outros agentes leem data/vendas.csv para análises',
            '4. Cada agente executa sua especialização',
            '5. diretor_insights consolida resultados'
        ]
    }

def get_integration_status():
    """
    Retorna status das integrações habilitadas.
    
    Returns:
        Dicionário com status das integrações
    """
    status = {}
    for integration_name, config in INTEGRATION_CONFIG.items():
        if isinstance(config, dict) and 'enabled' in config:
            status[integration_name] = config['enabled']
    
    return status

def get_tools_statistics():
    """
    Retorna estatísticas do sistema de ferramentas.
    
    Returns:
        Dicionário com estatísticas
    """
    return {
        'total_tools': len(TOOLS),
        'categories': len(TOOLS_BY_CATEGORY),
        'integrations': len(INTEGRATION_CONFIG),
        'agent_mappings': len(AGENT_TOOL_MAPPING),
        'migrations_documented': sum(len(items) for items in MIGRATION_MAP.values()),
        'tools_by_category': {cat: len(tools) for cat, tools in TOOLS_BY_CATEGORY.items()}
    }

def validate_tool_compatibility():
    """
    Valida se todas as ferramentas estão funcionando corretamente.
    
    Returns:
        Dicionário com status de cada ferramenta
    """
    validation_results = {}
    
    for tool in TOOLS:
        tool_name = tool.__class__.__name__
        try:
            # Verificar se tem método _run (padrão CrewAI)
            has_run = hasattr(tool, '_run')
            
            # Verificar se a instância é válida
            is_valid = tool is not None
            
            validation_results[tool_name] = {
                'valid': is_valid,
                'has_run_method': has_run,
                'class_name': tool.__class__.__name__,
                'status': 'OK' if (is_valid and has_run) else 'WARNING'
            }
        except Exception as e:
            validation_results[tool_name] = {
                'valid': False,
                'error': str(e),
                'status': 'ERROR'
            }
    
    return validation_results 