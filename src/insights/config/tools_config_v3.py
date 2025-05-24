"""
🔧 CONFIGURAÇÃO DAS FERRAMENTAS V3.0 CONSOLIDADAS
=================================================

Este arquivo configura as ferramentas refatoradas para uso no crew.py
"""

from ..tools.kpi_calculator_tool_v3 import KPICalculatorToolV3
from ..tools.statistical_analysis_tool_v3 import StatisticalAnalysisToolV3

# Instanciar as ferramentas v3.0
kpi_calculator_v3 = KPICalculatorToolV3()
statistical_analysis_v3 = StatisticalAnalysisToolV3()

# Lista das ferramentas disponíveis v3.0
TOOLS_V3 = [
    kpi_calculator_v3,
    statistical_analysis_v3
]

# Mapeamento de ferramentas por categoria
TOOLS_BY_CATEGORY = {
    'business_kpis': [kpi_calculator_v3],
    'statistical_analysis': [statistical_analysis_v3],
    'all': TOOLS_V3
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
        'Análises demográficas completas -> StatisticalAnalysisToolV3',
        'Análises geográficas completas -> StatisticalAnalysisToolV3',
        'Clustering básico -> StatisticalAnalysisToolV3',
        'Correlações simples -> StatisticalAnalysisToolV3'
    ],
    'removed_from_statistical_tool': [
        'KPIs básicos de negócio -> KPICalculatorToolV3',
        'Cálculos de margem e ROI -> KPICalculatorToolV3', 
        'Benchmarks do setor -> KPICalculatorToolV3',
        'Alertas automáticos -> KPICalculatorToolV3'
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
        return [kpi_calculator_v3]
    elif analysis_type in ['statistical', 'demographic', 'geographic', 'clustering']:
        return [statistical_analysis_v3]
    elif analysis_type in ['comprehensive', 'full', 'complete']:
        return TOOLS_V3
    else:
        return TOOLS_V3  # Default: todas as ferramentas 