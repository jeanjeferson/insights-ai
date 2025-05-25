"""
🔧 CONFIGURAÇÃO DAS FERRAMENTAS V3.0 CONSOLIDADAS
=================================================

Este arquivo configura as ferramentas refatoradas para uso no crew.py
"""

from ..tools.kpi_calculator_tool import KPICalculatorTool
from ..tools.statistical_analysis_tool import StatisticalAnalysisTool

# Instanciar as ferramentas
kpi_calculator_tool = KPICalculatorTool()
statistical_analysis_tool = StatisticalAnalysisTool()

# Lista das ferramentas disponíveis
TOOLS = [
    kpi_calculator_tool,
    statistical_analysis_tool
]

# Mapeamento de ferramentas por categoria
TOOLS_BY_CATEGORY = {
    'business_kpis': [kpi_calculator_tool],
    'statistical_analysis': [statistical_analysis_tool],
    'all': TOOLS
}

# Configuração de integração entre ferramentas
INTEGRATION_CONFIG = {
    'kpi_statistical_integration': {
        'enabled': True,
        'statistical_insights_for_kpi': True,
        'shared_cache': True,
        'cross_validation': True
    },
    'shared_modules': {
        'data_preparation': 'src.insights.tools.shared.data_preparation',
        'report_formatter': 'src.insights.tools.shared.report_formatter', 
        'business_mixins': 'src.insights.tools.shared.business_mixins'
    }
}

# Funcionalidades removidas/movidas
MIGRATION_MAP = {
    'removed_from_kpi_tool': [
        'Análises demográficas completas -> StatisticalAnalysisTool',
        'Análises geográficas completas -> StatisticalAnalysisTool',
        'Clustering básico -> StatisticalAnalysisTool',
        'Correlações simples -> StatisticalAnalysisTool'
    ],
    'removed_from_statistical_tool': [
        'KPIs básicos de negócio -> KPICalculatorTool',
        'Cálculos de margem e ROI -> KPICalculatorTool', 
        'Benchmarks do setor -> KPICalculatorTool',
        'Alertas automáticos -> KPICalculatorTool'
    ],
    'consolidated_in_shared': [
        'Preparação de dados -> DataPreparationMixin',
        'Formatação de relatórios -> ReportFormatterMixin',
        'Análises RFM -> JewelryRFMAnalysisMixin',
        'Análises BCG/ABC -> JewelryBusinessAnalysisMixin',
        'Benchmarks do setor -> JewelryBenchmarkMixin'
    ]
}

def get_tools_for_analysis(analysis_type: str):
    """
    Retorna as ferramentas apropriadas para um tipo de análise.
    
    Args:
        analysis_type: Tipo de análise desejada
        
    Returns:
        Lista de ferramentas apropriadas
    """
    if analysis_type in ['kpi', 'business', 'financial', 'operational']:
        return [kpi_calculator_tool]
    elif analysis_type in ['statistical', 'demographic', 'geographic', 'clustering']:
        return [statistical_analysis_tool]
    elif analysis_type in ['comprehensive', 'full', 'complete']:
        return TOOLS
    else:
        return TOOLS  # Default: todas as ferramentas 