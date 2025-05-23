"""
🔬 TESTE: FERRAMENTAS AVANÇADAS
===============================

Testa as ferramentas avançadas do projeto Insights-AI.
Inclui: Customer Insights, Business Intelligence, Risk Assessment, etc.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent))

# Importar ferramentas avançadas
try:
    from insights.tools.advanced.customer_insights_engine import CustomerInsightsEngine
except ImportError as e:
    print(f"⚠️ Erro ao importar CustomerInsightsEngine: {e}")
    CustomerInsightsEngine = None

try:
    from insights.tools.advanced.business_intelligence_dashboard import BusinessIntelligenceDashboard
except ImportError as e:
    print(f"⚠️ Erro ao importar BusinessIntelligenceDashboard: {e}")
    BusinessIntelligenceDashboard = None

try:
    from insights.tools.advanced.risk_assessment_tool import RiskAssessmentTool
except ImportError as e:
    print(f"⚠️ Erro ao importar RiskAssessmentTool: {e}")
    RiskAssessmentTool = None

try:
    from insights.tools.advanced.recommendation_engine import RecommendationEngine
except ImportError as e:
    print(f"⚠️ Erro ao importar RecommendationEngine: {e}")
    RecommendationEngine = None

try:
    from insights.tools.advanced.advanced_analytics_engine import AdvancedAnalyticsEngine
except ImportError as e:
    print(f"⚠️ Erro ao importar AdvancedAnalyticsEngine: {e}")
    AdvancedAnalyticsEngine = None

try:
    from insights.tools.advanced.competitive_intelligence_tool import CompetitiveIntelligenceTool
except ImportError as e:
    print(f"⚠️ Erro ao importar CompetitiveIntelligenceTool: {e}")
    CompetitiveIntelligenceTool = None

def create_advanced_test_data():
    """Criar dados robustos para testes de ferramentas avançadas"""
    np.random.seed(42)
    
    n_samples = 500
    start_date = datetime(2024, 1, 1)
    
    # Dados mais complexos para ferramentas avançadas
    customers = [f"CLI_{i:04d}" for i in range(1, 101)]  # 100 clientes únicos
    products = [f"PROD_{i:04d}" for i in range(1, 51)]   # 50 produtos únicos
    categories = ['Anéis', 'Brincos', 'Colares', 'Pulseiras', 'Alianças']
    metals = ['Ouro', 'Prata', 'Ouro Branco', 'Ouro Rosé', 'Platina']
    sellers = [f"VEND_{i:02d}" for i in range(1, 11)]     # 10 vendedores
    
    data = []
    for i in range(n_samples):
        date = start_date + timedelta(days=np.random.randint(0, 365))
        customer = np.random.choice(customers)
        product = np.random.choice(products)
        category = np.random.choice(categories)
        metal = np.random.choice(metals)
        seller = np.random.choice(sellers)
        
        # Valores com correlações e padrões realistas
        base_price = {
            'Anéis': 1500, 'Brincos': 800, 'Colares': 2200,
            'Pulseiras': 1300, 'Alianças': 2800
        }[category]
        
        metal_multiplier = {
            'Ouro': 1.0, 'Prata': 0.35, 'Ouro Branco': 1.15,
            'Ouro Rosé': 1.08, 'Platina': 1.75
        }[metal]
        
        # Adicionar variação sazonal
        month = date.month
        seasonal_factor = 1 + 0.4 * np.sin(2 * np.pi * month / 12 + np.pi/2)  # Pico em dezembro
        
        # Variação por vendedor (alguns são melhores)
        seller_multiplier = 1 + np.random.normal(0, 0.2)
        
        quantidade = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.5, 0.2, 0.15, 0.1, 0.03, 0.02])
        preco_unitario = base_price * metal_multiplier * seasonal_factor * seller_multiplier
        total_liquido = preco_unitario * quantidade
        
        # Adicionar mais campos para análises avançadas
        data.append({
            'Data': date.strftime('%Y-%m-%d'),
            'Ano': date.year,
            'Mes': date.month,
            'Trimestre': f"Q{((date.month-1)//3)+1}",
            'Dia_Semana': date.weekday(),
            'Codigo_Cliente': customer,
            'Nome_Cliente': f"Cliente {customer.split('_')[1]}",
            'Codigo_Produto': product,
            'Descricao_Produto': f"{category} {metal} Premium",
            'Categoria': category,
            'Metal': metal,
            'Codigo_Vendedor': seller,
            'Nome_Vendedor': f"Vendedor {seller.split('_')[1]}",
            'Quantidade': quantidade,
            'Preco_Unitario': round(preco_unitario, 2),
            'Total_Liquido': round(total_liquido, 2),
            'Custo_Produto': round(total_liquido * 0.4, 2),  # 40% de custo
            'Margem_Liquida': round(total_liquido * 0.6, 2),  # 60% de margem
            'Desconto_Aplicado': round(total_liquido * np.random.uniform(0, 0.15), 2),
            'Canal_Venda': np.random.choice(['Loja Física', 'E-commerce', 'WhatsApp'], p=[0.6, 0.3, 0.1]),
            'Forma_Pagamento': np.random.choice(['Cartão', 'PIX', 'Dinheiro', 'Parcelado'], p=[0.4, 0.3, 0.15, 0.15]),
            'Regiao_Cliente': np.random.choice(['Sudeste', 'Sul', 'Nordeste', 'Centro-Oeste'], p=[0.5, 0.25, 0.15, 0.1]),
            'Tipo_Cliente': np.random.choice(['Novo', 'Recorrente', 'VIP'], p=[0.3, 0.6, 0.1]),
            'Satisfacao_Cliente': np.random.randint(1, 6),  # 1-5 escala
            'Lead_Source': np.random.choice(['Orgânico', 'Redes Sociais', 'Indicação', 'Campanha'], p=[0.4, 0.3, 0.2, 0.1])
        })
    
    return pd.DataFrame(data)

def test_customer_insights_engine(verbose=False, quick=False):
    """Teste da Customer Insights Engine"""
    result = {'success': False, 'details': {}, 'warnings': [], 'errors': []}
    
    try:
        if verbose:
            print("👥 Testando Customer Insights Engine...")
        
        if CustomerInsightsEngine is None:
            result['errors'].append("CustomerInsightsEngine não disponível")
            return result
        
        # Instanciar ferramenta
        insights_engine = CustomerInsightsEngine()
        test_data = create_advanced_test_data()
        
        # Criar arquivo temporário
        test_csv = "temp_customer_test.csv"
        test_data.to_csv(test_csv, sep=';', index=False, encoding='utf-8')
        
        # Testes específicos
        tests_results = {}
        
        # Teste de segmentação RFM
        try:
            rfm_result = insights_engine._run(
                data_csv=test_csv,
                analysis_type="rfm_segmentation",
                customer_id_column="Codigo_Cliente",
                date_column="Data",
                value_column="Total_Liquido"
            )
            tests_results['rfm_segmentation'] = {
                'status': 'SUCCESS' if isinstance(rfm_result, str) and len(rfm_result) > 0 else 'FAILED',
                'has_segments': 'Champions' in rfm_result or 'VIP' in rfm_result if isinstance(rfm_result, str) else False
            }
        except Exception as e:
            tests_results['rfm_segmentation'] = {'status': 'ERROR', 'error': str(e)}
        
        # Teste de análise de comportamento
        try:
            behavior_result = insights_engine._run(
                data_csv=test_csv,
                analysis_type="behavioral_analysis",
                customer_id_column="Codigo_Cliente"
            )
            tests_results['behavioral_analysis'] = {
                'status': 'SUCCESS' if isinstance(behavior_result, str) and len(behavior_result) > 0 else 'FAILED',
                'has_patterns': 'padrão' in behavior_result.lower() or 'comportamento' in behavior_result.lower() if isinstance(behavior_result, str) else False
            }
        except Exception as e:
            tests_results['behavioral_analysis'] = {'status': 'ERROR', 'error': str(e)}
        
        # Teste de CLV (Customer Lifetime Value)
        try:
            clv_result = insights_engine._run(
                data_csv=test_csv,
                analysis_type="clv_analysis",
                customer_id_column="Codigo_Cliente",
                value_column="Total_Liquido"
            )
            tests_results['clv_analysis'] = {
                'status': 'SUCCESS' if isinstance(clv_result, str) and len(clv_result) > 0 else 'FAILED',
                'has_clv_metrics': 'lifetime' in clv_result.lower() or 'clv' in clv_result.lower() if isinstance(clv_result, str) else False
            }
        except Exception as e:
            tests_results['clv_analysis'] = {'status': 'ERROR', 'error': str(e)}
        
        # Limpeza
        if os.path.exists(test_csv):
            os.remove(test_csv)
        
        result['details'] = {
            'tool_instantiated': True,
            'test_data_rows': len(test_data),
            'tests_results': tests_results
        }
        
        successful_tests = len([t for t in tests_results.values() if t.get('status') == 'SUCCESS'])
        result['success'] = successful_tests > 0
        
        if verbose:
            print(f"✅ Customer Insights: {successful_tests}/{len(tests_results)} testes passaram")
        
    except Exception as e:
        result['errors'].append(f"Erro no teste Customer Insights: {str(e)}")
    
    return result

def test_business_intelligence_dashboard(verbose=False, quick=False):
    """Teste do Business Intelligence Dashboard"""
    result = {'success': False, 'details': {}, 'warnings': [], 'errors': []}
    
    try:
        if verbose:
            print("📊 Testando Business Intelligence Dashboard...")
        
        if BusinessIntelligenceDashboard is None:
            result['errors'].append("BusinessIntelligenceDashboard não disponível")
            return result
        
        bi_dashboard = BusinessIntelligenceDashboard()
        test_data = create_advanced_test_data()
        
        # Criar arquivo temporário
        test_csv = "temp_bi_test.csv"
        test_data.to_csv(test_csv, sep=';', index=False, encoding='utf-8')
        
        dashboard_tests = {}
        
        # Teste de dashboard executivo
        try:
            exec_dashboard = bi_dashboard._run(
                data_csv=test_csv,
                dashboard_type="executive",
                date_column="Data",
                value_column="Total_Liquido"
            )
            dashboard_tests['executive_dashboard'] = {
                'status': 'SUCCESS' if isinstance(exec_dashboard, str) and len(exec_dashboard) > 0 else 'FAILED',
                'is_interactive': 'plotly' in exec_dashboard.lower() or 'html' in exec_dashboard.lower() if isinstance(exec_dashboard, str) else False
            }
        except Exception as e:
            dashboard_tests['executive_dashboard'] = {'status': 'ERROR', 'error': str(e)}
        
        # Teste de dashboard operacional
        try:
            oper_dashboard = bi_dashboard._run(
                data_csv=test_csv,
                dashboard_type="operational",
                category_column="Categoria"
            )
            dashboard_tests['operational_dashboard'] = {
                'status': 'SUCCESS' if isinstance(oper_dashboard, str) and len(oper_dashboard) > 0 else 'FAILED',
                'has_kpis': 'kpi' in oper_dashboard.lower() or 'métrica' in oper_dashboard.lower() if isinstance(oper_dashboard, str) else False
            }
        except Exception as e:
            dashboard_tests['operational_dashboard'] = {'status': 'ERROR', 'error': str(e)}
        
        # Limpeza
        if os.path.exists(test_csv):
            os.remove(test_csv)
        
        result['details'] = {
            'tool_instantiated': True,
            'dashboard_tests': dashboard_tests
        }
        
        successful_tests = len([t for t in dashboard_tests.values() if t.get('status') == 'SUCCESS'])
        result['success'] = successful_tests > 0
        
        if verbose:
            print(f"✅ BI Dashboard: {successful_tests}/{len(dashboard_tests)} testes passaram")
        
    except Exception as e:
        result['errors'].append(f"Erro no teste BI Dashboard: {str(e)}")
    
    return result

def test_risk_assessment_tool(verbose=False, quick=False):
    """Teste da Risk Assessment Tool"""
    result = {'success': False, 'details': {}, 'warnings': [], 'errors': []}
    
    try:
        if verbose:
            print("⚠️ Testando Risk Assessment Tool...")
        
        if RiskAssessmentTool is None:
            result['errors'].append("RiskAssessmentTool não disponível")
            return result
        
        risk_tool = RiskAssessmentTool()
        test_data = create_advanced_test_data()
        
        test_csv = "temp_risk_test.csv"
        test_data.to_csv(test_csv, sep=';', index=False, encoding='utf-8')
        
        risk_tests = {}
        
        # Teste de análise de risco de estoque
        try:
            inventory_risk = risk_tool._run(
                data_csv=test_csv,
                risk_type="inventory_risk",
                product_column="Codigo_Produto",
                value_column="Total_Liquido"
            )
            risk_tests['inventory_risk'] = {
                'status': 'SUCCESS' if isinstance(inventory_risk, str) and len(inventory_risk) > 0 else 'FAILED',
                'has_risk_score': 'risco' in inventory_risk.lower() or 'score' in inventory_risk.lower() if isinstance(inventory_risk, str) else False
            }
        except Exception as e:
            risk_tests['inventory_risk'] = {'status': 'ERROR', 'error': str(e)}
        
        # Teste de análise de risco de cliente
        try:
            customer_risk = risk_tool._run(
                data_csv=test_csv,
                risk_type="customer_risk",
                customer_column="Codigo_Cliente"
            )
            risk_tests['customer_risk'] = {
                'status': 'SUCCESS' if isinstance(customer_risk, str) and len(customer_risk) > 0 else 'FAILED',
                'has_churn_analysis': 'churn' in customer_risk.lower() or 'abandono' in customer_risk.lower() if isinstance(customer_risk, str) else False
            }
        except Exception as e:
            risk_tests['customer_risk'] = {'status': 'ERROR', 'error': str(e)}
        
        # Limpeza
        if os.path.exists(test_csv):
            os.remove(test_csv)
        
        result['details'] = {
            'tool_instantiated': True,
            'risk_tests': risk_tests
        }
        
        successful_tests = len([t for t in risk_tests.values() if t.get('status') == 'SUCCESS'])
        result['success'] = successful_tests > 0
        
        if verbose:
            print(f"✅ Risk Assessment: {successful_tests}/{len(risk_tests)} testes passaram")
        
    except Exception as e:
        result['errors'].append(f"Erro no teste Risk Assessment: {str(e)}")
    
    return result

def test_recommendation_engine(verbose=False, quick=False):
    """Teste da Recommendation Engine"""
    result = {'success': False, 'details': {}, 'warnings': [], 'errors': []}
    
    try:
        if verbose:
            print("💡 Testando Recommendation Engine...")
        
        if RecommendationEngine is None:
            result['errors'].append("RecommendationEngine não disponível")
            return result
        
        rec_engine = RecommendationEngine()
        test_data = create_advanced_test_data()
        
        test_csv = "temp_rec_test.csv"
        test_data.to_csv(test_csv, sep=';', index=False, encoding='utf-8')
        
        recommendation_tests = {}
        
        # Teste de recomendações de produto
        try:
            product_rec = rec_engine._run(
                data_csv=test_csv,
                recommendation_type="product_recommendations",
                customer_column="Codigo_Cliente",
                product_column="Codigo_Produto"
            )
            recommendation_tests['product_recommendations'] = {
                'status': 'SUCCESS' if isinstance(product_rec, str) and len(product_rec) > 0 else 'FAILED',
                'has_recommendations': 'recomendação' in product_rec.lower() or 'sugestão' in product_rec.lower() if isinstance(product_rec, str) else False
            }
        except Exception as e:
            recommendation_tests['product_recommendations'] = {'status': 'ERROR', 'error': str(e)}
        
        # Teste de cross-selling
        try:
            cross_sell = rec_engine._run(
                data_csv=test_csv,
                recommendation_type="cross_selling",
                category_column="Categoria"
            )
            recommendation_tests['cross_selling'] = {
                'status': 'SUCCESS' if isinstance(cross_sell, str) and len(cross_sell) > 0 else 'FAILED',
                'has_cross_sell': 'cross' in cross_sell.lower() or 'cruzada' in cross_sell.lower() if isinstance(cross_sell, str) else False
            }
        except Exception as e:
            recommendation_tests['cross_selling'] = {'status': 'ERROR', 'error': str(e)}
        
        # Limpeza
        if os.path.exists(test_csv):
            os.remove(test_csv)
        
        result['details'] = {
            'tool_instantiated': True,
            'recommendation_tests': recommendation_tests
        }
        
        successful_tests = len([t for t in recommendation_tests.values() if t.get('status') == 'SUCCESS'])
        result['success'] = successful_tests > 0
        
        if verbose:
            print(f"✅ Recommendation Engine: {successful_tests}/{len(recommendation_tests)} testes passaram")
        
    except Exception as e:
        result['errors'].append(f"Erro no teste Recommendation Engine: {str(e)}")
    
    return result

def test_advanced_tools(verbose=False, quick=False):
    """
    Teste consolidado de todas as ferramentas avançadas
    """
    result = {
        'success': False,
        'details': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        if verbose:
            print("🔬 Iniciando testes das ferramentas avançadas...")
        
        # Executar todos os testes de ferramentas avançadas
        advanced_tests = {}
        
        # Customer Insights Engine
        customer_result = test_customer_insights_engine(verbose=verbose, quick=quick)
        advanced_tests['customer_insights'] = customer_result
        
        # Business Intelligence Dashboard
        bi_result = test_business_intelligence_dashboard(verbose=verbose, quick=quick)
        advanced_tests['bi_dashboard'] = bi_result
        
        # Risk Assessment Tool
        risk_result = test_risk_assessment_tool(verbose=verbose, quick=quick)
        advanced_tests['risk_assessment'] = risk_result
        
        # Recommendation Engine
        rec_result = test_recommendation_engine(verbose=verbose, quick=quick)
        advanced_tests['recommendation_engine'] = rec_result
        
        # Estatísticas consolidadas
        total_tools = len(advanced_tests)
        successful_tools = len([t for t in advanced_tests.values() if t.get('success', False)])
        total_warnings = sum(len(t.get('warnings', [])) for t in advanced_tests.values())
        total_errors = sum(len(t.get('errors', [])) for t in advanced_tests.values())
        
        result['details'] = {
            'total_advanced_tools': total_tools,
            'successful_tools': successful_tools,
            'total_warnings': total_warnings,
            'total_errors': total_errors,
            'success_rate': round(successful_tools / total_tools * 100, 1) if total_tools > 0 else 0,
            'individual_results': advanced_tests
        }
        
        # Consolidar warnings e errors
        for tool_result in advanced_tests.values():
            result['warnings'].extend(tool_result.get('warnings', []))
            result['errors'].extend(tool_result.get('errors', []))
        
        # Determinar sucesso geral
        result['success'] = successful_tools > 0  # Pelo menos uma ferramenta deve funcionar
        
        if verbose:
            print(f"🔬 Ferramentas Avançadas: {successful_tools}/{total_tools} funcionando")
            print(f"⚠️ Total de warnings: {total_warnings}")
            print(f"❌ Total de erros: {total_errors}")
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Erro inesperado nos testes avançados: {str(e)}")
        result['success'] = False
        return result

if __name__ == "__main__":
    # Teste standalone
    result = test_advanced_tools(verbose=True, quick=False)
    print("\n📊 RESULTADO DOS TESTES AVANÇADOS:")
    print(f"✅ Sucesso: {result['success']}")
    print(f"📈 Taxa de Sucesso: {result['details'].get('success_rate', 0)}%")
    print(f"⚠️ Warnings: {len(result['warnings'])}")
    print(f"❌ Erros: {len(result['errors'])}")
    
    if result['warnings']:
        print("\nWarnings:")
        for warning in result['warnings'][:5]:  # Mostrar apenas os primeiros 5
            print(f"  - {warning}")
        if len(result['warnings']) > 5:
            print(f"  ... e mais {len(result['warnings']) - 5} warnings")
    
    if result['errors']:
        print("\nErros:")
        for error in result['errors'][:5]:  # Mostrar apenas os primeiros 5
            print(f"  - {error}")
        if len(result['errors']) > 5:
            print(f"  ... e mais {len(result['errors']) - 5} erros")
