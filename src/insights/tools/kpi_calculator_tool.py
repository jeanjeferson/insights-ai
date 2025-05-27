from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings

# Importar módulos compartilhados consolidados
from .shared.data_preparation import DataPreparationMixin
from .shared.report_formatter import ReportFormatterMixin
from .shared.business_mixins import JewelryRFMAnalysisMixin, JewelryBusinessAnalysisMixin, JewelryBenchmarkMixin

warnings.filterwarnings('ignore')

class KPICalculatorToolInput(BaseModel):
    """Schema otimizado para cálculo de KPIs com validações robustas."""
    
    data_csv: str = Field(
        default="data/vendas.csv", 
        description="Caminho para arquivo CSV de vendas. Use 'data/vendas.csv' para dados principais.",
        json_schema_extra={"example": "data/vendas.csv"}
    )
    
    categoria: str = Field(
        default="all", 
        description="Categoria de KPIs: 'all' (completo), 'revenue' (financeiros), 'operational' (operacionais), 'inventory' (estoque), 'customer' (clientes), 'products' (produtos).",
        json_schema_extra={
            "pattern": "^(all|revenue|operational|inventory|customer|products)$"
        }
    )
    
    periodo: str = Field(
        default="monthly", 
        description="Período de análise: 'daily' (diário), 'weekly' (semanal), 'monthly' (mensal).",
        json_schema_extra={
            "pattern": "^(daily|weekly|monthly)$"
        }
    )
    
    benchmark_mode: bool = Field(
        default=True, 
        description="Incluir benchmarks do setor de joalherias. Recomendado: True para comparação com mercado."
    )
    
    include_statistical_insights: bool = Field(
        default=True, 
        description="Incluir insights de análises estatísticas avançadas. Use True para análises completas."
    )
    
    cache_data: bool = Field(
        default=True, 
        description="Usar cache para otimizar performance. Recomendado: True para datasets grandes."
    )
    
    alert_threshold: Optional[float] = Field(
        default=0.05,
        description="Limite para alertas automáticos (0.01-0.10). Menor valor = mais alertas sensíveis.",
        ge=0.01,
        le=0.10
    )
    
    @field_validator('categoria')
    @classmethod
    def validate_categoria(cls, v):
        valid_categories = ['all', 'revenue', 'operational', 'inventory', 'customer', 'products']
        if v not in valid_categories:
            raise ValueError(f"categoria deve ser um de: {valid_categories}")
        return v

class KPICalculatorTool(BaseTool, 
                         DataPreparationMixin, 
                         ReportFormatterMixin,
                         JewelryRFMAnalysisMixin, 
                         JewelryBusinessAnalysisMixin,
                         JewelryBenchmarkMixin):
    """
    📊 CALCULADORA AVANÇADA DE KPIs PARA JOALHERIAS
    
    QUANDO USAR:
    - Calcular KPIs essenciais de negócio
    - Monitorar performance financeira e operacional
    - Gerar alertas automáticos de problemas críticos
    - Comparar com benchmarks do setor
    - Avaliar saúde geral do negócio
    
    CASOS DE USO ESPECÍFICOS:
    - categoria='revenue': KPIs financeiros (margem, ROI, crescimento)
    - categoria='operational': KPIs operacionais (giro, velocidade, concentração)
    - categoria='inventory': KPIs de estoque (ABC, turnover, alertas)
    - categoria='customer': KPIs de clientes (segmentação, CLV, retenção)
    - categoria='products': KPIs de produtos (BCG matrix, performance)
    - categoria='all': Relatório executivo completo com todos os KPIs
    
    RESULTADOS ENTREGUES:
    - KPIs calculados com precisão e contexto
    - Alertas automáticos para problemas críticos
    - Comparação com benchmarks do setor
    - Insights acionáveis para tomada de decisão
    - Scores de saúde do negócio
    - Recomendações estratégicas automatizadas
    """
    
    name: str = "KPI Calculator Tool"
    description: str = (
        "Calculadora avançada de KPIs para joalherias com alertas automáticos e benchmarks. "
        "Calcula métricas essenciais de negócio, compara com padrões do setor e gera insights acionáveis. "
        "Ideal para monitoramento contínuo de performance e identificação de oportunidades de melhoria."
    )
    args_schema: Type[BaseModel] = KPICalculatorToolInput
    
    def __init__(self):
        super().__init__()
        self._data_cache = {}  # Cache para dados preparados
    
    def _run(self, data_csv: str = "data/vendas.csv", categoria: str = "all", 
             periodo: str = "monthly", benchmark_mode: bool = True,
             include_statistical_insights: bool = True, cache_data: bool = True,
             alert_threshold: float = 0.05) -> str:
        try:
            print(f"📊 Iniciando KPI Calculator v3.0 - Categoria: {categoria}")
            print(f"⚙️ Configurações: período={periodo}, benchmarks={benchmark_mode}, cache={cache_data}")
            
            # 1. Carregar e preparar dados usando módulo consolidado
            df = self._load_and_prepare_data(data_csv, cache_data)
            if df is None:
                return json.dumps({
                    "error": "Não foi possível carregar os dados ou estrutura inválida",
                    "troubleshooting": {
                        "check_file_exists": f"Verifique se {data_csv} existe",
                        "check_file_format": "Confirme que o arquivo está em formato CSV com separador ';'",
                        "check_required_columns": "Verifique se as colunas obrigatórias estão presentes"
                    },
                    "metadata": {
                        "tool": "KPI Calculator",
                        "status": "error",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }, ensure_ascii=False, indent=2)
            
            print(f"✅ Dados preparados: {len(df)} registros com {len(df.columns)} campos")
            
            # 2. Calcular KPIs por categoria com responsabilidades redefinidas
            kpis = {
                "metadata": {
                    "tool": "KPI Calculator v3.0",
                    "categoria": categoria,
                    "periodo": periodo,
                    "total_records": len(df),
                    "date_range": {
                        "start": df['Data'].min().strftime("%Y-%m-%d") if 'Data' in df.columns else "N/A",
                        "end": df['Data'].max().strftime("%Y-%m-%d") if 'Data' in df.columns else "N/A"
                    },
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
            if categoria == "all" or categoria == "revenue":
                print("💰 Calculando KPIs financeiros...")
                kpis['financeiros'] = self._calculate_financial_kpis_v3(df, periodo)
            
            if categoria == "all" or categoria == "operational":
                print("⚙️ Calculando KPIs operacionais...")
                kpis['operacionais'] = self._calculate_operational_kpis_v3(df, periodo)
            
            if categoria == "all" or categoria == "inventory":
                print("📦 Calculando KPIs de inventário...")
                kpis['inventario'] = self._calculate_inventory_kpis_v3(df, periodo)
            
            if categoria == "all" or categoria == "customer":
                print("👥 Calculando KPIs de clientes...")
                kpis['clientes'] = self._calculate_customer_kpis_v3(df, periodo)
            
            if categoria == "all" or categoria == "products":
                print("💎 Calculando KPIs de produtos...")
                kpis['produtos'] = self._calculate_product_kpis_v3(df, periodo)
            
            # 3. Análises consolidadas (sempre incluídas quando categoria = "all")
            if categoria == "all":
                if benchmark_mode:
                    print("📈 Comparando com benchmarks...")
                    kpis['benchmarks'] = self._calculate_benchmark_comparison_v3(df)
                
                print("🚨 Gerando alertas inteligentes...")
                kpis['alertas'] = self._generate_intelligent_alerts(df, kpis, alert_threshold)
                
                print("💡 Gerando insights de negócio...")
                kpis['insights'] = self._generate_business_insights_v3(df, kpis)
                
                # 4. Integração com Statistical Tool (se solicitado)
                if include_statistical_insights:
                    print("🔬 Integrando insights estatísticos...")
                    kpis['statistical_insights'] = self._integrate_statistical_insights(df)
                
                # 5. Score de saúde do negócio
                print("🎯 Calculando score de saúde...")
                kpis['health_score'] = self._calculate_business_health_score(kpis)
            
            # 6. Formatar resultado final
            print("✅ KPIs calculados com sucesso!")
            return json.dumps(kpis, ensure_ascii=False, indent=2, default=str)
            
        except Exception as e:
            error_response = {
                "error": f"Erro no KPI Calculator v3.0: {str(e)}",
                "categoria": categoria,
                "data_csv": data_csv,
                "troubleshooting": {
                    "check_data_format": "Verifique se os dados estão no formato correto",
                    "check_memory": "Dados muito grandes podem causar problemas de memória",
                    "try_smaller_dataset": "Tente com um subset menor dos dados"
                },
                "metadata": {
                    "tool": "KPI Calculator",
                    "status": "error",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            return json.dumps(error_response, ensure_ascii=False, indent=2)
    
    def _load_and_prepare_data(self, data_csv: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Carregar e preparar dados usando módulo consolidado com cache."""
        cache_key = f"kpi_data_{hash(data_csv)}"
        
        # Verificar cache
        if use_cache and cache_key in self._data_cache:
            print("📋 Usando dados do cache")
            return self._data_cache[cache_key]
        
        try:
            # Carregar dados brutos
            df = pd.read_csv(data_csv, sep=';', encoding='utf-8')
            print(f"📁 Arquivo carregado: {len(df)} registros")
            
            # Preparar dados usando mixin consolidado
            df_prepared = self.prepare_jewelry_data(df, validation_level="standard")
            
            if df_prepared is None:
                print("❌ Falha na preparação dos dados")
                return None
            
            # Armazenar no cache
            if use_cache and df_prepared is not None:
                self._data_cache[cache_key] = df_prepared
                print("💾 Dados salvos no cache")
            
            return df_prepared
            
        except Exception as e:
            print(f"❌ Erro no carregamento de dados: {str(e)}")
            return None
    
    def _calculate_business_health_score(self, kpis: Dict[str, Any]) -> Dict[str, Any]:
        """Calcular score de saúde do negócio baseado nos KPIs."""
        try:
            scores = {}
            total_score = 0
            weight_sum = 0
            
            # Score financeiro (peso 30%)
            if 'financeiros' in kpis:
                fin_score = self._calculate_financial_health_score(kpis['financeiros'])
                scores['financial_health'] = fin_score
                total_score += fin_score * 0.3
                weight_sum += 0.3
            
            # Score operacional (peso 25%)
            if 'operacionais' in kpis:
                op_score = self._calculate_operational_health_score(kpis['operacionais'])
                scores['operational_health'] = op_score
                total_score += op_score * 0.25
                weight_sum += 0.25
            
            # Score de inventário (peso 20%)
            if 'inventario' in kpis:
                inv_score = self._calculate_inventory_health_score(kpis['inventario'])
                scores['inventory_health'] = inv_score
                total_score += inv_score * 0.2
                weight_sum += 0.2
            
            # Score de clientes (peso 15%)
            if 'clientes' in kpis:
                cust_score = self._calculate_customer_health_score(kpis['clientes'])
                scores['customer_health'] = cust_score
                total_score += cust_score * 0.15
                weight_sum += 0.15
            
            # Score de produtos (peso 10%)
            if 'produtos' in kpis:
                prod_score = self._calculate_product_health_score(kpis['produtos'])
                scores['product_health'] = prod_score
                total_score += prod_score * 0.1
                weight_sum += 0.1
            
            # Score geral
            overall_score = total_score / weight_sum if weight_sum > 0 else 0
            
            # Classificação
            if overall_score >= 80:
                classification = "Excelente"
                status = "🟢"
            elif overall_score >= 70:
                classification = "Bom"
                status = "🟡"
            elif overall_score >= 60:
                classification = "Regular"
                status = "🟠"
            else:
                classification = "Crítico"
                status = "🔴"
            
            return {
                "overall_score": round(overall_score, 1),
                "classification": classification,
                "status": status,
                "component_scores": scores,
                "recommendations": self._generate_health_recommendations(overall_score, scores)
            }
            
        except Exception as e:
            return {"error": f"Erro no cálculo de saúde: {str(e)}"}
    
    def _calculate_financial_health_score(self, financial_kpis: Dict[str, Any]) -> float:
        """Calcular score de saúde financeira."""
        score = 70  # Base score
        
        # Margem
        if 'margem_analysis' in financial_kpis:
            margem = financial_kpis['margem_analysis'].get('margem_percentual_media', 0)
            if margem > 50:
                score += 15
            elif margem > 40:
                score += 10
            elif margem > 30:
                score += 5
            else:
                score -= 10
        
        # Crescimento
        if 'growth_analysis' in financial_kpis:
            growth = financial_kpis['growth_analysis'].get('mom_growth_rate', 0)
            if growth > 10:
                score += 15
            elif growth > 5:
                score += 10
            elif growth > 0:
                score += 5
            else:
                score -= 10
        
        return min(max(score, 0), 100)
    
    def _calculate_operational_health_score(self, operational_kpis: Dict[str, Any]) -> float:
        """Calcular score de saúde operacional."""
        score = 70  # Base score
        
        # Concentração
        if 'concentration_analysis' in operational_kpis:
            concentration = operational_kpis['concentration_analysis'].get('concentration_80_20_pct', 0)
            if concentration < 70:
                score += 15
            elif concentration < 80:
                score += 10
            elif concentration < 90:
                score += 5
            else:
                score -= 10
        
        return min(max(score, 0), 100)
    
    def _calculate_inventory_health_score(self, inventory_kpis: Dict[str, Any]) -> float:
        """Calcular score de saúde de inventário."""
        score = 70  # Base score
        
        # Produtos slow-moving
        if 'product_lifecycle' in inventory_kpis:
            slow_moving_pct = inventory_kpis['product_lifecycle'].get('slow_moving_pct', 0)
            if slow_moving_pct < 10:
                score += 15
            elif slow_moving_pct < 20:
                score += 10
            elif slow_moving_pct < 30:
                score += 5
            else:
                score -= 10
        
        return min(max(score, 0), 100)
    
    def _calculate_customer_health_score(self, customer_kpis: Dict[str, Any]) -> float:
        """Calcular score de saúde de clientes."""
        score = 70  # Base score
        
        # Taxa de retenção
        if 'retention_metrics' in customer_kpis:
            repeat_rate = customer_kpis['retention_metrics'].get('repeat_rate', 0)
            if repeat_rate > 40:
                score += 15
            elif repeat_rate > 30:
                score += 10
            elif repeat_rate > 20:
                score += 5
            else:
                score -= 10
        
        return min(max(score, 0), 100)
    
    def _calculate_product_health_score(self, product_kpis: Dict[str, Any]) -> float:
        """Calcular score de saúde de produtos."""
        score = 70  # Base score
        
        # Diversificação por metal
        if 'metal_performance' in product_kpis:
            market_share = product_kpis['metal_performance'].get('market_share', {})
            if len(market_share) > 3:  # Boa diversificação
                score += 15
            elif len(market_share) > 2:
                score += 10
            else:
                score += 5
        
        return min(max(score, 0), 100)
    
    def _generate_health_recommendations(self, overall_score: float, component_scores: Dict[str, float]) -> List[str]:
        """Gerar recomendações baseadas no score de saúde."""
        recommendations = []
        
        if overall_score < 60:
            recommendations.append("🚨 AÇÃO URGENTE: Score crítico - revisar estratégia geral")
        
        # Recomendações por componente
        for component, score in component_scores.items():
            if score < 60:
                if component == 'financial_health':
                    recommendations.append("💰 Revisar precificação e controle de custos")
                elif component == 'operational_health':
                    recommendations.append("⚙️ Otimizar processos operacionais")
                elif component == 'inventory_health':
                    recommendations.append("📦 Implementar gestão de estoque mais eficiente")
                elif component == 'customer_health':
                    recommendations.append("👥 Focar em retenção e fidelização de clientes")
                elif component == 'product_health':
                    recommendations.append("💎 Diversificar portfólio de produtos")
        
        if not recommendations:
            recommendations.append("✅ Negócio em boa saúde - manter estratégias atuais")
        
        return recommendations[:5]  # Top 5 recomendações
    
    # Manter métodos existentes com melhorias mínimas
    def _calculate_financial_kpis_v3(self, df: pd.DataFrame, periodo: str) -> Dict[str, Any]:
        """Calcular KPIs financeiros focados em métricas de negócio."""
        print("💰 Calculando KPIs financeiros v3.0...")
        
        try:
            kpis = {}
            
            # KPIs básicos essenciais
            total_revenue = df['Total_Liquido'].sum()
            kpis['total_revenue'] = round(total_revenue, 2)
            kpis['aov'] = round(df['Total_Liquido'].mean(), 2)
            kpis['median_order_value'] = round(df['Total_Liquido'].median(), 2)
            kpis['total_transactions'] = len(df)
            
            # NOVO: KPI Receita Ano Atual (YTD) com comparação YoY
            ytd_analysis = self._calculate_ytd_comparison(df)
            if ytd_analysis:
                kpis['ytd_analysis'] = ytd_analysis
            
            # KPIs de margem real (usando dados preparados)
            if 'Margem_Real' in df.columns:
                kpis['margem_analysis'] = {
                    'margem_total': round(df['Margem_Real'].sum(), 2),
                    'margem_percentual_media': round(df['Margem_Percentual'].mean(), 2),
                    'margem_mediana': round(df['Margem_Percentual'].median(), 2),
                    'produtos_baixa_margem': len(df[df['Margem_Percentual'] < 30]),
                    'roi_real': round((df['Margem_Real'].sum() / df['Custo_Produto'].sum() * 100), 2) if 'Custo_Produto' in df.columns else 0
                }
            
            # KPIs de crescimento
            if periodo == 'monthly' and 'Ano_Mes' in df.columns:
                monthly_revenue = df.groupby('Ano_Mes')['Total_Liquido'].sum()
                if len(monthly_revenue) >= 2:
                    kpis['growth_analysis'] = {
                        'mom_growth_rate': round(monthly_revenue.pct_change().iloc[-1] * 100, 2),
                        'avg_growth_3months': round(monthly_revenue.tail(3).pct_change().mean() * 100, 2) if len(monthly_revenue) >= 3 else 0,
                        'growth_acceleration': self._calculate_growth_acceleration_v3(monthly_revenue),
                        'revenue_trend': 'crescente' if monthly_revenue.pct_change().iloc[-1] > 0 else 'decrescente'
                    }
            
            # Revenue por categoria (melhorado)
            if 'Grupo_Produto' in df.columns:
                category_revenue = df.groupby('Grupo_Produto')['Total_Liquido'].sum()
                total_cat_revenue = category_revenue.sum()
                kpis['category_performance'] = {
                    'revenue_by_category': category_revenue.round(2).to_dict(),
                    'market_share_by_category': (category_revenue / total_cat_revenue * 100).round(2).to_dict(),
                    'top_category': category_revenue.idxmax(),
                    'category_concentration': round(category_revenue.max() / total_cat_revenue * 100, 2)
                }
            
            # Performance temporal resumida
            kpis['temporal_performance'] = self._calculate_temporal_performance_v3(df, periodo)
            
            return kpis
            
        except Exception as e:
            return {'error': f"Erro nos KPIs financeiros v3.0: {str(e)}"}
    
    def _calculate_operational_kpis_v3(self, df: pd.DataFrame, periodo: str) -> Dict[str, Any]:
        """KPIs operacionais focados em eficiência e performance."""
        print("⚙️ Calculando KPIs operacionais v3.0...")
        
        try:
            kpis = {}
            
            # Métricas básicas de eficiência
            days_in_period = (df['Data'].max() - df['Data'].min()).days + 1
            kpis['efficiency_metrics'] = {
                'produtos_ativos': df['Codigo_Produto'].nunique() if 'Codigo_Produto' in df.columns else len(df),
                'sales_velocity_daily': round(df['Quantidade'].sum() / days_in_period, 2),
                'revenue_velocity_daily': round(df['Total_Liquido'].sum() / days_in_period, 2),
                'avg_items_per_transaction': round(df['Quantidade'].mean(), 2),
                'transactions_per_day': round(len(df) / days_in_period, 2)
            }
            
            # Giro de estoque real (usando dados preparados)
            if 'Estoque_Atual' in df.columns:
                kpis['inventory_turnover'] = self._calculate_inventory_turnover_v3(df)
            
            # Análise de concentração (80/20 rule)
            if 'Codigo_Produto' in df.columns:
                product_sales = df.groupby('Codigo_Produto')['Total_Liquido'].sum().sort_values(ascending=False)
                top_20_pct = int(len(product_sales) * 0.2)
                concentration_80_20 = (product_sales.head(top_20_pct).sum() / product_sales.sum() * 100)
                
                kpis['concentration_analysis'] = {
                    'concentration_80_20_pct': round(concentration_80_20, 2),
                    'gini_coefficient': self._calculate_gini_coefficient(product_sales.values),
                    'top_20_percent_products': top_20_pct,
                    'concentration_status': 'Alta' if concentration_80_20 > 80 else 'Média' if concentration_80_20 > 60 else 'Baixa'
                }
            
            # Performance por dia da semana (simplificado)
            if 'Nome_Dia_Semana' in df.columns:
                weekday_performance = df.groupby('Nome_Dia_Semana')['Total_Liquido'].agg(['sum', 'mean', 'count'])
                best_day = weekday_performance['sum'].idxmax()
                worst_day = weekday_performance['sum'].idxmin()
                
                kpis['weekday_performance'] = {
                    'best_day': best_day,
                    'worst_day': worst_day,
                    'weekday_variation': round((weekday_performance['sum'].max() / weekday_performance['sum'].min() - 1) * 100, 2)
                }
            
            # Sazonalidade usando dados preparados
            if 'Sazonalidade' in df.columns:
                seasonal_performance = df.groupby('Sazonalidade')['Total_Liquido'].sum()
                kpis['seasonality'] = {
                    'seasonal_revenue': seasonal_performance.to_dict(),
                    'peak_season': seasonal_performance.idxmax(),
                    'seasonal_variation': round((seasonal_performance.max() / seasonal_performance.min() - 1) * 100, 2)
                }
            
            return kpis
            
        except Exception as e:
            return {'error': f"Erro nos KPIs operacionais v3.0: {str(e)}"}
    
    def _calculate_inventory_kpis_v3(self, df: pd.DataFrame, periodo: str) -> Dict[str, Any]:
        """KPIs de inventário usando análises consolidadas."""
        print("📦 Calculando KPIs de inventário v3.0...")
        
        try:
            kpis = {}
            
            # Análise ABC usando mixin consolidado
            abc_analysis = self.perform_abc_analysis(df, dimension='product')
            if 'error' not in abc_analysis:
                kpis['abc_analysis'] = abc_analysis
            
            # Análise de produtos slow-moving
            if 'Codigo_Produto' in df.columns:
                last_sale_by_product = df.groupby('Codigo_Produto')['Data'].max()
                current_date = df['Data'].max()
                
                # Produtos sem venda há mais de 60 dias
                slow_moving_cutoff = current_date - timedelta(days=60)
                slow_moving = (last_sale_by_product < slow_moving_cutoff).sum()
                
                # Produtos sem venda há mais de 90 dias (dead stock)
                dead_stock_cutoff = current_date - timedelta(days=90)
                dead_stock = (last_sale_by_product < dead_stock_cutoff).sum()
                
                total_products = len(last_sale_by_product)
                
                kpis['product_lifecycle'] = {
                    'slow_moving_products': slow_moving,
                    'slow_moving_pct': round(slow_moving / total_products * 100, 2),
                    'dead_stock_products': dead_stock,
                    'dead_stock_pct': round(dead_stock / total_products * 100, 2),
                    'active_products': total_products - dead_stock
                }
            
            # Turnover estimado (se não há dados reais de estoque)
            monthly_sales_avg = df.groupby([df['Data'].dt.year, df['Data'].dt.month])['Total_Liquido'].sum().mean()
            if monthly_sales_avg > 0:
                estimated_avg_inventory = monthly_sales_avg * 2.5  # Estimativa conservadora
                inventory_turnover_annual = (monthly_sales_avg * 12) / estimated_avg_inventory
                
                kpis['turnover_estimates'] = {
                    'estimated_inventory_turnover_annual': round(inventory_turnover_annual, 2),
                    'estimated_days_sales_inventory': round(365 / inventory_turnover_annual, 1),
                    'monthly_sales_average': round(monthly_sales_avg, 2)
                }
            
            return kpis
            
        except Exception as e:
            return {'error': f"Erro nos KPIs de inventário v3.0: {str(e)}"}
    
    def _calculate_customer_kpis_v3(self, df: pd.DataFrame, periodo: str) -> Dict[str, Any]:
        """KPIs de clientes focados em métricas de negócio (não demográficas)."""
        print("👥 Calculando KPIs de clientes v3.0...")
        
        try:
            kpis = {}
            
            # Segmentação por valor (mantida)
            value_segments = {
                'Premium (>R$5K)': len(df[df['Total_Liquido'] > 5000]),
                'Alto Valor (R$2K-5K)': len(df[(df['Total_Liquido'] >= 2000) & (df['Total_Liquido'] <= 5000)]),
                'Médio (R$1K-2K)': len(df[(df['Total_Liquido'] >= 1000) & (df['Total_Liquido'] < 2000)]),
                'Entry (< R$1K)': len(df[df['Total_Liquido'] < 1000])
            }
            
            total_transactions = sum(value_segments.values())
            
            kpis['value_segmentation'] = {
                'segment_distribution': value_segments,
                'segment_percentages': {k: round(v/total_transactions*100, 1) for k, v in value_segments.items()},
                'high_value_share': round((value_segments['Premium (>R$5K)'] + value_segments['Alto Valor (R$2K-5K)']) / total_transactions * 100, 2)
            }
            
            # RFM Analysis usando mixin consolidado
            if 'Codigo_Cliente' in df.columns:
                customer_rfm = self.analyze_customer_rfm(df)
                if 'error' not in customer_rfm:
                    kpis['rfm_analysis'] = customer_rfm
            else:
                # Estimativa de CLV e métricas de cliente (mantida como fallback)
                kpis['customer_estimates'] = self._estimate_customer_metrics(df)
            
            # Análise de retenção simples
            if 'Codigo_Cliente' in df.columns:
                customer_frequency = df['Codigo_Cliente'].value_counts()
                repeat_customers = len(customer_frequency[customer_frequency > 1])
                total_customers = len(customer_frequency)
                
                kpis['retention_metrics'] = {
                    'total_unique_customers': total_customers,
                    'repeat_customers': repeat_customers,
                    'repeat_rate': round(repeat_customers / total_customers * 100, 2),
                    'avg_purchases_per_customer': round(customer_frequency.mean(), 2)
                }
            
            return kpis
            
        except Exception as e:
            return {'error': f"Erro nos KPIs de clientes v3.0: {str(e)}"}
    
    def _calculate_product_kpis_v3(self, df: pd.DataFrame, periodo: str) -> Dict[str, Any]:
        """KPIs de produtos usando análises consolidadas."""
        print("💎 Calculando KPIs de produtos v3.0...")
        
        try:
            kpis = {}
            
            # Performance por categoria/metal
            if 'Metal' in df.columns:
                metal_performance = df.groupby('Metal').agg({
                    'Total_Liquido': ['sum', 'mean', 'count'],
                    'Quantidade': 'sum'
                }).round(2)
                
                kpis['metal_performance'] = {
                    'revenue_by_metal': metal_performance['Total_Liquido']['sum'].to_dict(),
                    'aov_by_metal': metal_performance['Total_Liquido']['mean'].to_dict(),
                    'transactions_by_metal': metal_performance['Total_Liquido']['count'].to_dict()
                }
                
                # Market share por metal
                total_revenue = df['Total_Liquido'].sum()
                metal_market_share = metal_performance['Total_Liquido']['sum'] / total_revenue * 100
                kpis['metal_performance']['market_share'] = metal_market_share.round(2).to_dict()
            
            # Matriz BCG usando mixin consolidado
            bcg_analysis = self.create_product_bcg_matrix(df)
            if 'error' not in bcg_analysis:
                kpis['bcg_matrix'] = bcg_analysis
            
            # RFM de produtos usando mixin consolidado
            product_rfm = self.analyze_product_rfm(df)
            if 'error' not in product_rfm:
                kpis['product_rfm'] = product_rfm
            
            # Top produtos por receita
            if 'Codigo_Produto' in df.columns:
                top_products = df.groupby('Codigo_Produto')['Total_Liquido'].sum().nlargest(10)
                kpis['top_products'] = {
                    'by_revenue': top_products.to_dict(),
                    'top_product_share': round(top_products.iloc[0] / df['Total_Liquido'].sum() * 100, 2)
                }
            
            # Elasticidade de preço usando benchmarks consolidados
            price_elasticity = self.get_jewelry_industry_benchmarks()['price_elasticity']
            kpis['price_elasticity'] = price_elasticity
            
            return kpis
            
        except Exception as e:
            return {'error': f"Erro nos KPIs de produtos v3.0: {str(e)}"}
    
    def _calculate_benchmark_comparison_v3(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comparação com benchmarks usando mixin consolidado."""
        print("📈 Comparando com benchmarks do setor...")
        
        try:
            # Preparar métricas atuais
            current_metrics = {
                'aov': df['Total_Liquido'].mean(),
                'gross_margin': df['Margem_Percentual'].mean() if 'Margem_Percentual' in df.columns else 58.0
            }
            
            # Usar mixin para comparação
            benchmark_comparison = self.compare_with_benchmarks(current_metrics)
            
            return benchmark_comparison
            
        except Exception as e:
            return {'error': f"Erro na comparação com benchmarks: {str(e)}"}
    
    def _generate_intelligent_alerts(self, df: pd.DataFrame, kpis: Dict[str, Any], threshold: float = 0.05) -> List[Dict[str, Any]]:
        """Gerar alertas inteligentes baseados nos KPIs calculados."""
        alerts = []
        
        try:
            # Alertas financeiros
            if 'financeiros' in kpis:
                fin = kpis['financeiros']
                
                # Alertas de margem
                if 'margem_analysis' in fin:
                    margem_media = fin['margem_analysis'].get('margem_percentual_media', 0)
                    produtos_baixa_margem = fin['margem_analysis'].get('produtos_baixa_margem', 0)
                    
                    if margem_media < 25:
                        alerts.append({
                            "type": "CRÍTICO",
                            "category": "Financeiro",
                            "message": f"Margem média muito baixa ({margem_media:.1f}%) - Ação imediata necessária",
                            "severity": "high",
                            "action_required": "Revisar precificação urgentemente"
                        })
                    elif margem_media < 40:
                        alerts.append({
                            "type": "ATENÇÃO",
                            "category": "Financeiro", 
                            "message": f"Margem média abaixo do ideal ({margem_media:.1f}%) - Revisar precificação",
                            "severity": "medium",
                            "action_required": "Analisar estrutura de custos"
                        })
                    
                    if produtos_baixa_margem > 0:
                        alerts.append({
                            "type": "MARGEM",
                            "category": "Produtos",
                            "message": f"{produtos_baixa_margem} produtos com margem <30% - Revisar preços",
                            "severity": "medium",
                            "action_required": "Revisar precificação por produto"
                        })
                
                # Alertas de crescimento
                if 'growth_analysis' in fin:
                    growth_rate = fin['growth_analysis'].get('mom_growth_rate', 0)
                    if growth_rate < -20:
                        alerts.append({
                            "type": "CRÍTICO",
                            "category": "Crescimento",
                            "message": f"Queda de vendas severa ({growth_rate:.1f}%) - Ação imediata necessária",
                            "severity": "high",
                            "action_required": "Investigar causas e implementar plano de recuperação"
                        })
                    elif growth_rate < -10:
                        alerts.append({
                            "type": "DECLÍNIO",
                            "category": "Crescimento",
                            "message": f"Queda de vendas detectada ({growth_rate:.1f}%) - Investigar causas",
                            "severity": "medium",
                            "action_required": "Analisar fatores de declínio"
                        })
            
            # Alertas operacionais
            if 'operacionais' in kpis:
                op = kpis['operacionais']
                
                # Alertas de concentração
                if 'concentration_analysis' in op:
                    concentration = op['concentration_analysis'].get('concentration_80_20_pct', 0)
                    if concentration > 90:
                        alerts.append({
                            "type": "RISCO",
                            "category": "Operacional",
                            "message": f"Concentração extrema de vendas ({concentration:.1f}%) - Diversificar portfólio",
                            "severity": "high",
                            "action_required": "Implementar estratégia de diversificação"
                        })
                    elif concentration > 80:
                        alerts.append({
                            "type": "CONCENTRAÇÃO",
                            "category": "Operacional",
                            "message": f"Alta dependência de poucos produtos ({concentration:.1f}%) - Monitorar",
                            "severity": "medium",
                            "action_required": "Desenvolver produtos complementares"
                        })
            
            return alerts[:8]  # Limitar a 8 alertas mais críticos
            
        except Exception as e:
            return [{"type": "ERRO", "message": f"Erro na geração de alertas: {str(e)}", "severity": "low"}]
    
    def _generate_business_insights_v3(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gerar insights de negócio baseados nos KPIs calculados."""
        insights = []
        
        try:
            # Insights financeiros
            if 'financeiros' in kpis:
                fin = kpis['financeiros']
                aov = fin.get('aov', 0)
                
                if aov > 2500:
                    insights.append({
                        "type": "Oportunidade",
                        "category": "Financeiro",
                        "message": f"AOV excelente (R${aov:,.2f}) - Posicionamento premium bem-sucedido",
                        "impact": "high",
                        "recommendation": "Manter estratégia de produtos premium"
                    })
                elif aov < 1000:
                    insights.append({
                        "type": "Melhoria",
                        "category": "Financeiro",
                        "message": f"AOV baixo (R${aov:,.2f}) - Oportunidade para up-sell",
                        "impact": "medium",
                        "recommendation": "Implementar estratégias de up-sell e cross-sell"
                    })
                
                if 'growth_analysis' in fin:
                    growth = fin['growth_analysis'].get('mom_growth_rate', 0)
                    if growth > 15:
                        insights.append({
                            "type": "Sucesso",
                            "category": "Crescimento",
                            "message": f"Crescimento forte ({growth:.1f}%) - Manter estratégias atuais",
                            "impact": "high",
                            "recommendation": "Escalar estratégias que estão funcionando"
                        })
                    elif growth > 5:
                        insights.append({
                            "type": "Performance",
                            "category": "Crescimento",
                            "message": f"Crescimento saudável ({growth:.1f}%) - Bom desempenho",
                            "impact": "medium",
                            "recommendation": "Buscar oportunidades de aceleração"
                        })
                
                if 'category_performance' in fin:
                    top_category = fin['category_performance'].get('top_category', 'N/A')
                    concentration = fin['category_performance'].get('category_concentration', 0)
                    insights.append({
                        "type": "Análise",
                        "category": "Produtos",
                        "message": f"Categoria líder: {top_category} ({concentration:.1f}% da receita)",
                        "impact": "medium",
                        "recommendation": "Balancear portfólio para reduzir dependência"
                    })
            
            return insights[:10]  # Top 10 insights mais relevantes
            
        except Exception as e:
            return [{"type": "Erro", "message": f"Erro na geração de insights: {str(e)}", "impact": "low"}]
    
    def _integrate_statistical_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Integrar insights de análises estatísticas (placeholder para integração futura)."""
        try:
            # Placeholder para integração com Statistical Analysis Tool
            # Esta integração será implementada após refatoração do Statistical Tool
            
            integration_status = {
                'status': 'placeholder',
                'available_analyses': [
                    'demographic_patterns',
                    'geographic_performance', 
                    'correlation_analysis',
                    'clustering_analysis'
                ],
                'message': 'Integração com Statistical Tool será ativada na v3.1',
                'estimated_completion': '2024-Q1'
            }
            
            return integration_status
            
        except Exception as e:
            return {'error': f"Erro na integração estatística: {str(e)}"}
    
    # Métodos auxiliares simplificados (manter implementações existentes)
    def _calculate_growth_acceleration_v3(self, monthly_revenue: pd.Series) -> float:
        """Calcular aceleração do crescimento."""
        if len(monthly_revenue) < 3:
            return 0
        growth_rates = monthly_revenue.pct_change()
        acceleration = growth_rates.diff().iloc[-1]
        return round(acceleration * 100, 2)
    
    def _calculate_temporal_performance_v3(self, df: pd.DataFrame, periodo: str) -> Dict[str, Any]:
        """Performance temporal simplificada."""
        temporal = {}
        
        # Performance por trimestre
        quarterly = df.groupby(df['Data'].dt.quarter)['Total_Liquido'].sum()
        temporal['quarterly_revenue'] = quarterly.to_dict()
        temporal['best_quarter'] = quarterly.idxmax()
        
        return temporal
    
    def _calculate_inventory_turnover_v3(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Giro de estoque usando dados preparados."""
        try:
            if 'Turnover_Estoque' in df.columns:
                turnover_stats = {
                    'turnover_medio': round(df['Turnover_Estoque'].mean(), 2),
                    'produtos_alto_turnover': len(df[df['Turnover_Estoque'] > df['Turnover_Estoque'].quantile(0.8)]),
                    'produtos_baixo_turnover': len(df[df['Turnover_Estoque'] < df['Turnover_Estoque'].quantile(0.2)])
                }
                
                # Alertas baseados nos dias de estoque
                if 'Dias_Estoque' in df.columns:
                    overstock = df[df['Dias_Estoque'] > 180]
                    understock = df[df['Dias_Estoque'] < 30]
                    
                    turnover_stats['produtos_excesso_estoque'] = len(overstock)
                    turnover_stats['produtos_baixo_estoque'] = len(understock)
                    turnover_stats['valor_excesso_estoque'] = round(overstock['Total_Liquido'].sum(), 2)
                
                return turnover_stats
            else:
                return {'error': 'Dados de estoque não disponíveis'}
                
        except Exception as e:
            return {'error': f"Erro no cálculo de turnover: {str(e)}"}
    
    def _estimate_customer_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Estimar métricas de cliente quando não há Codigo_Cliente."""
        # Estimativa conservadora baseada em padrões do setor
        high_value_threshold = 2000
        high_value_sales = df[df['Total_Liquido'] > high_value_threshold]
        low_value_sales = df[df['Total_Liquido'] <= high_value_threshold]
        
        estimated_unique_customers = len(high_value_sales) + (len(low_value_sales) * 0.7)
        
        # CLV estimado
        avg_purchase_value = df['Total_Liquido'].mean()
        estimated_annual_purchases = 2.3  # Benchmark do setor
        estimated_lifetime_years = 3.5
        estimated_clv = avg_purchase_value * estimated_annual_purchases * estimated_lifetime_years
        
        return {
            'estimated_unique_customers': int(estimated_unique_customers),
            'estimated_clv': round(estimated_clv, 2),
            'avg_purchase_value': round(avg_purchase_value, 2),
            'estimation_method': 'Industry benchmarks'
        }
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calcular coeficiente de Gini."""
        if len(values) == 0:
            return 0
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return round((n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n, 3)
    
    def _calculate_ytd_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calcular KPI de Receita Total no Ano Atual (YTD) com comparação year-over-year.
        
        Compara o mesmo período do ano anterior para uma análise precisa.
        Usa no mínimo 2 anos de dados para projeções confiáveis.
        """
        print("📅 Calculando análise YTD (Year-to-Date)...")
        
        try:
            if 'Data' not in df.columns:
                print("⚠️ Coluna 'Data' não encontrada para análise YTD")
                return {}
            
            # Converter Data para datetime se necessário
            df_copy = df.copy()
            df_copy['Data'] = pd.to_datetime(df_copy['Data'])
            
            # Obter data atual e ano atual
            current_date = df_copy['Data'].max()
            current_year = current_date.year
            current_month = current_date.month
            current_day = current_date.day
            
            # Verificar se temos pelo menos 2 anos de dados
            min_date = df_copy['Data'].min()
            years_of_data = (current_date - min_date).days / 365.25
            
            if years_of_data < 2:
                print(f"⚠️ Apenas {years_of_data:.1f} anos de dados disponíveis. Recomendado: mínimo 2 anos para projeções confiáveis")
            
            # Definir período YTD do ano atual
            ytd_start_current = pd.Timestamp(f"{current_year}-01-01")
            ytd_end_current = current_date
            
            # Definir mesmo período do ano anterior
            ytd_start_previous = pd.Timestamp(f"{current_year-1}-01-01") 
            ytd_end_previous = pd.Timestamp(f"{current_year-1}-{current_month:02d}-{current_day:02d}")
            
            # Filtrar dados para YTD atual
            ytd_current_data = df_copy[
                (df_copy['Data'] >= ytd_start_current) & 
                (df_copy['Data'] <= ytd_end_current)
            ]
            
            # Filtrar dados para mesmo período do ano anterior
            ytd_previous_data = df_copy[
                (df_copy['Data'] >= ytd_start_previous) & 
                (df_copy['Data'] <= ytd_end_previous)
            ]
            
            if ytd_current_data.empty:
                print("⚠️ Nenhum dado encontrado para o ano atual")
                return {}
            
            # Calcular métricas YTD atual
            ytd_current_revenue = ytd_current_data['Total_Liquido'].sum()
            ytd_current_transactions = len(ytd_current_data)
            ytd_current_avg_ticket = ytd_current_data['Total_Liquido'].mean()
            ytd_current_days = (ytd_end_current - ytd_start_current).days + 1
            
            # Calcular métricas do ano anterior (mesmo período)
            ytd_previous_revenue = ytd_previous_data['Total_Liquido'].sum() if not ytd_previous_data.empty else 0
            ytd_previous_transactions = len(ytd_previous_data)
            ytd_previous_avg_ticket = ytd_previous_data['Total_Liquido'].mean() if not ytd_previous_data.empty else 0
            ytd_previous_days = (ytd_end_previous - ytd_start_previous).days + 1
            
            # Calcular variações YoY
            revenue_yoy_change = 0
            transactions_yoy_change = 0
            avg_ticket_yoy_change = 0
            
            if ytd_previous_revenue > 0:
                revenue_yoy_change = ((ytd_current_revenue - ytd_previous_revenue) / ytd_previous_revenue) * 100
            
            if ytd_previous_transactions > 0:
                transactions_yoy_change = ((ytd_current_transactions - ytd_previous_transactions) / ytd_previous_transactions) * 100
            
            if ytd_previous_avg_ticket > 0:
                avg_ticket_yoy_change = ((ytd_current_avg_ticket - ytd_previous_avg_ticket) / ytd_previous_avg_ticket) * 100
            
            # Calcular projeção anual baseada em dados históricos (mínimo 2 anos)
            annual_projection = self._calculate_annual_projection(df_copy, current_date, years_of_data)
            
            # Calcular progresso em relação ao ano
            year_progress_pct = (current_date - ytd_start_current).days / 365 * 100
            
            ytd_analysis = {
                'current_year': current_year,
                'analysis_date': current_date.strftime('%Y-%m-%d'),
                'year_progress_percentage': round(year_progress_pct, 1),
                'ytd_current': {
                    'revenue': round(ytd_current_revenue, 2),
                    'transactions': ytd_current_transactions,
                    'avg_ticket': round(ytd_current_avg_ticket, 2),
                    'period_days': ytd_current_days,
                    'daily_average': round(ytd_current_revenue / ytd_current_days, 2)
                },
                'ytd_previous_year': {
                    'revenue': round(ytd_previous_revenue, 2),
                    'transactions': ytd_previous_transactions,
                    'avg_ticket': round(ytd_previous_avg_ticket, 2),
                    'period_days': ytd_previous_days,
                    'daily_average': round(ytd_previous_revenue / ytd_previous_days, 2) if ytd_previous_days > 0 else 0
                },
                'yoy_comparison': {
                    'revenue_change_pct': round(revenue_yoy_change, 2),
                    'revenue_change_abs': round(ytd_current_revenue - ytd_previous_revenue, 2),
                    'transactions_change_pct': round(transactions_yoy_change, 2),
                    'avg_ticket_change_pct': round(avg_ticket_yoy_change, 2),
                    'trend': 'positiva' if revenue_yoy_change > 0 else 'negativa' if revenue_yoy_change < 0 else 'estável'
                },
                'annual_projection': annual_projection,
                'data_quality': {
                    'years_of_data': round(years_of_data, 1),
                    'min_recommended_years': 2,
                    'confidence_level': 'alta' if years_of_data >= 2 else 'média' if years_of_data >= 1 else 'baixa'
                }
            }
            
            print(f"✅ Análise YTD concluída: YTD {current_year} vs {current_year-1} = {revenue_yoy_change:+.1f}%")
            return ytd_analysis
            
        except Exception as e:
            print(f"❌ Erro na análise YTD: {str(e)}")
            return {}
    
    def _calculate_annual_projection(self, df: pd.DataFrame, current_date: pd.Timestamp, years_of_data: float) -> Dict[str, Any]:
        """
        Calcular projeção anual baseada em dados históricos.
        Usa padrões sazonais de pelo menos 2 anos para maior precisão.
        """
        try:
            current_year = current_date.year
            
            # Se temos menos de 2 anos, fazer projeção simples
            if years_of_data < 2:
                ytd_revenue = df[df['Data'].dt.year == current_year]['Total_Liquido'].sum()
                days_elapsed = (current_date - pd.Timestamp(f"{current_year}-01-01")).days + 1
                daily_average = ytd_revenue / days_elapsed
                simple_projection = daily_average * 365
                
                return {
                    'method': 'simple_extrapolation',
                    'projected_annual_revenue': round(simple_projection, 2),
                    'confidence': 'baixa',
                    'note': 'Projeção baseada em tendência linear (dados insuficientes para análise sazonal)'
                }
            
            # Para 2+ anos, usar análise sazonal
            # Calcular receita média por mês dos anos anteriores
            historical_data = df[df['Data'].dt.year < current_year].copy()
            monthly_avg = historical_data.groupby(historical_data['Data'].dt.month)['Total_Liquido'].sum().groupby(level=0).mean()
            
            # YTD atual
            current_ytd = df[df['Data'].dt.year == current_year]['Total_Liquido'].sum()
            current_month = current_date.month
            
            # Projeção baseada em padrão sazonal histórico
            remaining_months_projection = monthly_avg[monthly_avg.index > current_month].sum()
            seasonal_projection = current_ytd + remaining_months_projection
            
            # Calcular crescimento médio anual histórico
            yearly_revenues = historical_data.groupby(historical_data['Data'].dt.year)['Total_Liquido'].sum()
            if len(yearly_revenues) >= 2:
                avg_growth_rate = yearly_revenues.pct_change().mean()
                growth_adjusted_projection = seasonal_projection * (1 + avg_growth_rate)
            else:
                growth_adjusted_projection = seasonal_projection
                avg_growth_rate = 0
            
            return {
                'method': 'seasonal_analysis_with_growth',
                'projected_annual_revenue': round(growth_adjusted_projection, 2),
                'seasonal_baseline': round(seasonal_projection, 2),
                'historical_growth_rate': round(avg_growth_rate * 100, 2),
                'confidence': 'alta' if years_of_data >= 2 else 'média',
                'years_analyzed': round(years_of_data, 1),
                'note': f'Projeção baseada em padrões sazonais de {len(yearly_revenues)} anos completos'
            }
            
        except Exception as e:
            print(f"❌ Erro na projeção anual: {str(e)}")
            return {
                'method': 'error',
                'projected_annual_revenue': 0,
                'confidence': 'nenhuma',
                'note': f'Erro no cálculo: {str(e)}'
            } 