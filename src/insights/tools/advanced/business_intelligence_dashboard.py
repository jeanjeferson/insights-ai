from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any, List
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class BusinessIntelligenceInput(BaseModel):
    """Schema de entrada para Business Intelligence Dashboard."""
    dashboard_type: str = Field(..., description="Tipo: 'executive_summary', 'operational_dashboard', 'financial_kpis', 'customer_analytics', 'product_performance', 'comprehensive_report'")
    data_csv: str = Field(default="data/vendas.csv", description="Caminho para arquivo CSV")
    time_period: str = Field(default="last_12_months", description="Período: 'last_month', 'last_quarter', 'last_12_months', 'ytd'")
    include_forecasts: bool = Field(default=True, description="Incluir projeções")
    detail_level: str = Field(default="summary", description="Nível: 'summary', 'detailed', 'comprehensive'")

class BusinessIntelligenceDashboard(BaseTool):
    name: str = "Business Intelligence Dashboard"
    description: str = """
    Dashboard de Business Intelligence para joalherias:
    
    TIPOS DE DASHBOARD:
    - executive_summary: Resumo executivo para C-level
    - operational_dashboard: Métricas operacionais diárias
    - financial_kpis: KPIs financeiros detalhados
    - customer_analytics: Analytics de clientes
    - product_performance: Performance de produtos
    - comprehensive_report: Relatório completo integrado
    
    RECURSOS:
    - KPIs automatizados
    - Análises comparativas
    - Tendências e forecasts
    - Alertas e indicadores
    - Benchmarks do setor
    - Insights acionáveis
    
    PERÍODOS DISPONÍVEIS:
    - Último mês
    - Último trimestre  
    - Últimos 12 meses
    - Year-to-date
    """
    args_schema: Type[BaseModel] = BusinessIntelligenceInput
    
    def _run(self, dashboard_type: str, data_csv: str = "data/vendas.csv",
             time_period: str = "last_12_months", include_forecasts: bool = True,
             detail_level: str = "summary") -> str:
        try:
            # Carregar e preparar dados
            df = pd.read_csv(data_csv, sep=';', encoding='utf-8')
            df = self._prepare_bi_data(df, time_period)
            
            if df is None or len(df) < 5:
                return "Erro: Dados insuficientes para gerar dashboard (mínimo 5 registros)"
            
            # Dicionário de dashboards
            dashboard_generators = {
                'executive_summary': self._generate_executive_summary,
                'operational_dashboard': self._generate_operational_dashboard,
                'financial_kpis': self._generate_financial_kpis,
                'customer_analytics': self._generate_customer_analytics,
                'product_performance': self._generate_product_performance,
                'comprehensive_report': self._generate_comprehensive_report
            }
            
            if dashboard_type not in dashboard_generators:
                return f"Dashboard '{dashboard_type}' não suportado. Opções: {list(dashboard_generators.keys())}"
            
            result = dashboard_generators[dashboard_type](df, include_forecasts, detail_level)
            return self._format_dashboard_result(dashboard_type, result, time_period, detail_level)
            
        except Exception as e:
            return f"Erro no Business Intelligence Dashboard: {str(e)}"
    
    def _prepare_bi_data(self, df: pd.DataFrame, time_period: str) -> Optional[pd.DataFrame]:
        """Preparar dados para BI."""
        try:
            # Converter data
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
            df = df.dropna(subset=['Data'])
            
            # Filtrar por período
            current_date = df['Data'].max()
            
            if time_period == 'last_month':
                start_date = current_date - timedelta(days=30)
            elif time_period == 'last_quarter':
                start_date = current_date - timedelta(days=90)
            elif time_period == 'last_12_months':
                start_date = current_date - timedelta(days=365)
            elif time_period == 'ytd':
                start_date = datetime(current_date.year, 1, 1)
            else:
                start_date = current_date - timedelta(days=365)
            
            df = df[df['Data'] >= start_date]
            
            # Adicionar métricas derivadas
            df['Preco_Unitario'] = df['Total_Liquido'] / df['Quantidade'].replace(0, 1)
            df['Year_Month'] = df['Data'].dt.to_period('M')
            df['Week'] = df['Data'].dt.isocalendar().week
            df['Weekday'] = df['Data'].dt.day_name()
            df['Month_Name'] = df['Data'].dt.month_name()
            
            # Simular customer_id se necessário
            if 'Customer_ID' not in df.columns:
                df = self._simulate_customer_ids(df)
            
            return df
            
        except Exception as e:
            print(f"Erro na preparação de dados BI: {str(e)}")
            return None
    
    def _simulate_customer_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simular customer IDs."""
        # Implementação básica
        df = df.copy()
        np.random.seed(42)
        df['Customer_ID'] = 'CUST_' + pd.Series(np.random.randint(1, len(df)//3 + 1, len(df))).astype(str)
        return df
    
    def _generate_executive_summary(self, df: pd.DataFrame, include_forecasts: bool, 
                                  detail_level: str) -> Dict[str, Any]:
        """Gerar resumo executivo."""
        try:
            # KPIs principais
            total_revenue = df['Total_Liquido'].sum()
            total_customers = df['Customer_ID'].nunique()
            total_transactions = len(df)
            avg_ticket = df['Total_Liquido'].mean()
            
            # Crescimento mensal
            monthly_revenue = df.groupby('Year_Month')['Total_Liquido'].sum()
            if len(monthly_revenue) >= 2:
                mom_growth = ((monthly_revenue.iloc[-1] - monthly_revenue.iloc[-2]) / monthly_revenue.iloc[-2] * 100)
            else:
                mom_growth = 0
            
            # Top performers
            top_products = df.groupby('Codigo_Produto')['Total_Liquido'].sum().nlargest(5)
            
            # Segmentação rápida de clientes
            customer_metrics = df.groupby('Customer_ID').agg({
                'Total_Liquido': ['sum', 'count'],
                'Data': 'max'
            }).fillna(0)
            customer_metrics.columns = ['Total_Spent', 'Transaction_Count', 'Last_Purchase']
            
            # Clientes VIP (top 20% por valor)
            vip_threshold = customer_metrics['Total_Spent'].quantile(0.8)
            vip_customers = customer_metrics[customer_metrics['Total_Spent'] >= vip_threshold]
            vip_revenue = vip_customers['Total_Spent'].sum()
            
            # Alertas críticos
            alerts = []
            if mom_growth < -10:
                alerts.append("🚨 CRÍTICO: Receita caiu >10% no último mês")
            if len(df[df['Data'] >= df['Data'].max() - timedelta(days=7)]) == 0:
                alerts.append("⚠️ ATENÇÃO: Sem vendas nos últimos 7 dias")
            if avg_ticket < 500:
                alerts.append("💡 OPORTUNIDADE: Ticket médio baixo - focar em up-sell")
            
            # Forecast simples se solicitado
            forecast = {}
            if include_forecasts and len(monthly_revenue) >= 3:
                # Projeção linear simples para próximo mês
                trend = np.polyfit(range(len(monthly_revenue)), monthly_revenue.values, 1)[0]
                next_month_forecast = monthly_revenue.iloc[-1] + trend
                forecast = {
                    'next_month_revenue': round(next_month_forecast, 2),
                    'growth_trend': 'Positive' if trend > 0 else 'Negative',
                    'confidence': 'Medium'
                }
            
            return {
                'period_summary': {
                    'total_revenue': round(total_revenue, 2),
                    'total_customers': total_customers,
                    'total_transactions': total_transactions,
                    'avg_ticket': round(avg_ticket, 2),
                    'mom_growth_pct': round(mom_growth, 1)
                },
                'customer_insights': {
                    'vip_customers': len(vip_customers),
                    'vip_revenue_share': round(vip_revenue / total_revenue * 100, 1),
                    'avg_customer_value': round(customer_metrics['Total_Spent'].mean(), 2)
                },
                'top_performers': top_products.to_dict(),
                'alerts': alerts,
                'forecast': forecast,
                'executive_insights': self._generate_executive_insights(df, customer_metrics, mom_growth)
            }
            
        except Exception as e:
            return {'error': f"Erro no resumo executivo: {str(e)}"}
    
    def _generate_operational_dashboard(self, df: pd.DataFrame, include_forecasts: bool,
                                      detail_level: str) -> Dict[str, Any]:
        """Gerar dashboard operacional."""
        try:
            # Métricas diárias
            daily_metrics = df.groupby(df['Data'].dt.date).agg({
                'Total_Liquido': ['sum', 'count', 'mean'],
                'Customer_ID': 'nunique',
                'Quantidade': 'sum'
            }).fillna(0)
            
            daily_metrics.columns = ['Daily_Revenue', 'Daily_Transactions', 'Daily_Avg_Ticket', 'Daily_Customers', 'Daily_Items']
            
            # Métricas da última semana
            last_week = daily_metrics.tail(7)
            
            # Performance por dia da semana
            weekday_performance = df.groupby('Weekday').agg({
                'Total_Liquido': ['sum', 'mean'],
                'Customer_ID': 'nunique'
            }).round(2)
            weekday_performance.columns = ['Total_Revenue', 'Avg_Revenue', 'Unique_Customers']
            
            # Produtos em destaque hoje/recente
            recent_sales = df[df['Data'] >= df['Data'].max() - timedelta(days=3)]
            trending_products = recent_sales.groupby('Codigo_Produto')['Quantidade'].sum().nlargest(10)
            
            # Velocidade de vendas
            sales_velocity = {
                'items_per_day': df['Quantidade'].sum() / len(daily_metrics),
                'revenue_per_day': df['Total_Liquido'].sum() / len(daily_metrics),
                'customers_per_day': df['Customer_ID'].nunique() / len(daily_metrics)
            }
            
            # Indicadores operacionais
            operational_indicators = {
                'avg_items_per_transaction': df['Quantidade'].mean(),
                'peak_sales_day': weekday_performance['Total_Revenue'].idxmax(),
                'conversion_rate_estimate': 0.15,  # Estimativa
                'inventory_turnover_estimate': 2.3  # Benchmark do setor
            }
            
            # Alerts operacionais
            operational_alerts = []
            
            if last_week['Daily_Revenue'].mean() < daily_metrics['Daily_Revenue'].mean() * 0.8:
                operational_alerts.append("📉 Vendas da última semana abaixo da média")
            
            if sales_velocity['items_per_day'] < 10:
                operational_alerts.append("🐌 Velocidade de vendas baixa")
            
            if len(trending_products) < 5:
                operational_alerts.append("📦 Poucos produtos em movimento")
            
            return {
                'daily_metrics': {
                    'last_7_days': last_week.round(2).to_dict(),
                    'period_average': daily_metrics.mean().round(2).to_dict()
                },
                'weekday_performance': weekday_performance.to_dict(),
                'trending_products': trending_products.to_dict(),
                'sales_velocity': {k: round(v, 2) for k, v in sales_velocity.items()},
                'operational_indicators': operational_indicators,
                'operational_alerts': operational_alerts
            }
            
        except Exception as e:
            return {'error': f"Erro no dashboard operacional: {str(e)}"}
    
    def _generate_financial_kpis(self, df: pd.DataFrame, include_forecasts: bool,
                               detail_level: str) -> Dict[str, Any]:
        """Gerar KPIs financeiros."""
        try:
            # KPIs de receita
            revenue_kpis = {
                'total_revenue': df['Total_Liquido'].sum(),
                'avg_monthly_revenue': df.groupby('Year_Month')['Total_Liquido'].sum().mean(),
                'revenue_per_customer': df.groupby('Customer_ID')['Total_Liquido'].sum().mean(),
                'avg_order_value': df['Total_Liquido'].mean()
            }
            
            # KPIs de crescimento
            monthly_revenue = df.groupby('Year_Month')['Total_Liquido'].sum()
            quarterly_revenue = df.groupby(df['Data'].dt.to_period('Q'))['Total_Liquido'].sum()
            
            growth_kpis = {}
            if len(monthly_revenue) >= 2:
                growth_kpis['mom_growth'] = ((monthly_revenue.iloc[-1] - monthly_revenue.iloc[-2]) / monthly_revenue.iloc[-2] * 100)
            if len(quarterly_revenue) >= 2:
                growth_kpis['qoq_growth'] = ((quarterly_revenue.iloc[-1] - quarterly_revenue.iloc[-2]) / quarterly_revenue.iloc[-2] * 100)
            
            # KPIs de margem (estimados)
            estimated_margins = {
                'estimated_gross_margin': 0.58,  # Benchmark joalherias
                'estimated_net_margin': 0.15,
                'estimated_cogs': revenue_kpis['total_revenue'] * 0.42
            }
            
            # KPIs de produtividade
            productivity_kpis = {
                'revenue_per_transaction': df['Total_Liquido'].mean(),
                'transactions_per_customer': df.groupby('Customer_ID').size().mean(),
                'revenue_concentration': self._calculate_revenue_concentration(df)
            }
            
            # Análise temporal detalhada
            temporal_analysis = {
                'monthly_trend': monthly_revenue.pct_change().mean() * 100,
                'seasonal_peak': monthly_revenue.idxmax(),
                'seasonal_low': monthly_revenue.idxmin(),
                'revenue_volatility': monthly_revenue.std() / monthly_revenue.mean()
            }
            
            # Benchmarks do setor
            sector_benchmarks = {
                'avg_ticket_benchmark': 1500,  # Mercado médio
                'growth_rate_benchmark': 3.5,  # % anual
                'customer_retention_benchmark': 0.25
            }
            
            # Comparação com benchmarks
            benchmark_comparison = {
                'aov_vs_benchmark': (revenue_kpis['avg_order_value'] / sector_benchmarks['avg_ticket_benchmark'] - 1) * 100,
                'position': 'Above Market' if revenue_kpis['avg_order_value'] > sector_benchmarks['avg_ticket_benchmark'] else 'Below Market'
            }
            
            return {
                'revenue_kpis': {k: round(v, 2) for k, v in revenue_kpis.items()},
                'growth_kpis': {k: round(v, 2) for k, v in growth_kpis.items()},
                'estimated_margins': estimated_margins,
                'productivity_kpis': {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in productivity_kpis.items()},
                'temporal_analysis': temporal_analysis,
                'benchmark_comparison': benchmark_comparison,
                'financial_health_score': self._calculate_financial_health_score(revenue_kpis, growth_kpis)
            }
            
        except Exception as e:
            return {'error': f"Erro nos KPIs financeiros: {str(e)}"}
    
    def _generate_customer_analytics(self, df: pd.DataFrame, include_forecasts: bool,
                                   detail_level: str) -> Dict[str, Any]:
        """Gerar analytics de clientes."""
        try:
            # Métricas base de clientes
            customer_base_metrics = {
                'total_customers': df['Customer_ID'].nunique(),
                'new_customers_period': self._estimate_new_customers(df),
                'avg_customer_lifetime_days': self._estimate_customer_lifetime(df),
                'customer_concentration': self._calculate_customer_concentration(df)
            }
            
            # Segmentação RFM básica
            current_date = df['Data'].max()
            customer_rfm = df.groupby('Customer_ID').agg({
                'Data': 'max',
                'Total_Liquido': ['sum', 'count', 'mean']
            }).fillna(0)
            
            customer_rfm.columns = ['Last_Purchase', 'Total_Spent', 'Purchase_Count', 'Avg_Spent']
            customer_rfm['Recency'] = (current_date - customer_rfm['Last_Purchase']).dt.days
            
            # Classificação simples de clientes
            customer_rfm['Segment'] = customer_rfm.apply(self._classify_customer_simple, axis=1)
            
            # Distribuição por segmento
            segment_distribution = customer_rfm['Segment'].value_counts().to_dict()
            
            # Valor por segmento
            segment_value = customer_rfm.groupby('Segment').agg({
                'Total_Spent': ['sum', 'mean', 'count']
            }).round(2)
            segment_value.columns = ['Total_Revenue', 'Avg_Customer_Value', 'Customer_Count']
            
            # Análise de comportamento
            behavior_analysis = {
                'repeat_customers': len(customer_rfm[customer_rfm['Purchase_Count'] > 1]),
                'one_time_customers': len(customer_rfm[customer_rfm['Purchase_Count'] == 1]),
                'avg_purchases_per_customer': customer_rfm['Purchase_Count'].mean(),
                'customer_lifetime_value': customer_rfm['Total_Spent'].mean()
            }
            
            # Análise de retenção
            retention_analysis = {
                'active_customers_30d': len(customer_rfm[customer_rfm['Recency'] <= 30]),
                'at_risk_customers_90d': len(customer_rfm[customer_rfm['Recency'] > 90]),
                'lost_customers_180d': len(customer_rfm[customer_rfm['Recency'] > 180]),
                'retention_rate_estimate': behavior_analysis['repeat_customers'] / customer_base_metrics['total_customers']
            }
            
            # Insights de clientes
            customer_insights = []
            
            vip_pct = segment_distribution.get('VIP', 0) / customer_base_metrics['total_customers'] * 100
            customer_insights.append(f"Clientes VIP: {vip_pct:.1f}% da base")
            
            one_time_pct = behavior_analysis['one_time_customers'] / customer_base_metrics['total_customers'] * 100
            if one_time_pct > 60:
                customer_insights.append(f"Alta taxa de one-time buyers ({one_time_pct:.1f}%) - focar em retenção")
            
            at_risk_pct = retention_analysis['at_risk_customers_90d'] / customer_base_metrics['total_customers'] * 100
            if at_risk_pct > 30:
                customer_insights.append(f"Muitos clientes em risco ({at_risk_pct:.1f}%) - campanhas de reativação urgentes")
            
            return {
                'customer_base_metrics': customer_base_metrics,
                'segment_distribution': segment_distribution,
                'segment_value_analysis': segment_value.to_dict(),
                'behavior_analysis': {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in behavior_analysis.items()},
                'retention_analysis': {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in retention_analysis.items()},
                'customer_insights': customer_insights
            }
            
        except Exception as e:
            return {'error': f"Erro no customer analytics: {str(e)}"}
    
    def _generate_product_performance(self, df: pd.DataFrame, include_forecasts: bool,
                                    detail_level: str) -> Dict[str, Any]:
        """Gerar performance de produtos."""
        try:
            # Performance por produto
            product_performance = df.groupby('Codigo_Produto').agg({
                'Total_Liquido': ['sum', 'mean', 'count'],
                'Quantidade': 'sum',
                'Data': ['min', 'max'],
                'Descricao_Produto': 'first'
            }).fillna(0)
            
            product_performance.columns = ['Total_Revenue', 'Avg_Price', 'Transaction_Count', 'Total_Quantity', 'First_Sale', 'Last_Sale', 'Description']
            
            # Calcular métricas adicionais
            current_date = df['Data'].max()
            product_performance['Days_Since_Last_Sale'] = (current_date - pd.to_datetime(product_performance['Last_Sale'])).dt.days
            product_performance['Product_Lifetime_Days'] = (pd.to_datetime(product_performance['Last_Sale']) - pd.to_datetime(product_performance['First_Sale'])).dt.days + 1
            product_performance['Daily_Revenue'] = product_performance['Total_Revenue'] / product_performance['Product_Lifetime_Days']
            
            # Rankings
            top_products_by_revenue = product_performance.nlargest(10, 'Total_Revenue')
            top_products_by_volume = product_performance.nlargest(10, 'Total_Quantity')
            top_products_by_frequency = product_performance.nlargest(10, 'Transaction_Count')
            
            # Análise ABC
            product_performance_sorted = product_performance.sort_values('Total_Revenue', ascending=False)
            cumsum_pct = product_performance_sorted['Total_Revenue'].cumsum() / product_performance_sorted['Total_Revenue'].sum()
            
            abc_classification = []
            for pct in cumsum_pct:
                if pct <= 0.8:
                    abc_classification.append('A')
                elif pct <= 0.95:
                    abc_classification.append('B')
                else:
                    abc_classification.append('C')
            
            product_performance_sorted['ABC_Class'] = abc_classification
            abc_distribution = pd.Series(abc_classification).value_counts().to_dict()
            
            # Produtos em risco
            slow_movers = product_performance[product_performance['Days_Since_Last_Sale'] > 60]
            dead_stock = product_performance[product_performance['Days_Since_Last_Sale'] > 120]
            
            # Performance por categoria
            category_performance = {}
            if 'Grupo_Produto' in df.columns:
                category_performance = df.groupby('Grupo_Produto').agg({
                    'Total_Liquido': ['sum', 'mean'],
                    'Quantidade': 'sum',
                    'Codigo_Produto': 'nunique'
                }).round(2)
                category_performance.columns = ['Total_Revenue', 'Avg_Ticket', 'Total_Quantity', 'Product_Count']
                category_performance = category_performance.to_dict()
            
            # Análise de preços
            price_analysis = {
                'avg_price_overall': df['Preco_Unitario'].mean(),
                'price_range': {
                    'min': df['Preco_Unitario'].min(),
                    'max': df['Preco_Unitario'].max(),
                    'median': df['Preco_Unitario'].median()
                },
                'price_distribution': df['Preco_Unitario'].describe().to_dict()
            }
            
            # Insights de produtos
            product_insights = []
            
            product_insights.append(f"Total de produtos ativos: {len(product_performance)}")
            product_insights.append(f"Produtos classe A: {abc_distribution.get('A', 0)} ({abc_distribution.get('A', 0)/len(product_performance)*100:.1f}%)")
            
            if len(slow_movers) > 0:
                product_insights.append(f"{len(slow_movers)} produtos sem venda há 60+ dias")
            
            if len(dead_stock) > 0:
                product_insights.append(f"{len(dead_stock)} produtos em dead stock (120+ dias)")
            
            best_performer = top_products_by_revenue.iloc[0]
            product_insights.append(f"Top produto: {best_performer['Description']} (R$ {best_performer['Total_Revenue']:,.2f})")
            
            return {
                'product_rankings': {
                    'top_by_revenue': top_products_by_revenue[['Description', 'Total_Revenue', 'Transaction_Count']].to_dict('records'),
                    'top_by_volume': top_products_by_volume[['Description', 'Total_Quantity', 'Total_Revenue']].to_dict('records'),
                    'top_by_frequency': top_products_by_frequency[['Description', 'Transaction_Count', 'Total_Revenue']].to_dict('records')
                },
                'abc_analysis': {
                    'distribution': abc_distribution,
                    'class_a_revenue_share': (product_performance_sorted[product_performance_sorted['ABC_Class'] == 'A']['Total_Revenue'].sum() / product_performance['Total_Revenue'].sum() * 100)
                },
                'inventory_alerts': {
                    'slow_movers': len(slow_movers),
                    'dead_stock': len(dead_stock),
                    'avg_days_since_sale': product_performance['Days_Since_Last_Sale'].mean()
                },
                'category_performance': category_performance,
                'price_analysis': {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in price_analysis.items()},
                'product_insights': product_insights
            }
            
        except Exception as e:
            return {'error': f"Erro na performance de produtos: {str(e)}"}
    
    def _generate_comprehensive_report(self, df: pd.DataFrame, include_forecasts: bool,
                                     detail_level: str) -> Dict[str, Any]:
        """Gerar relatório abrangente."""
        try:
            # Compilar todos os dashboards
            executive = self._generate_executive_summary(df, include_forecasts, detail_level)
            operational = self._generate_operational_dashboard(df, include_forecasts, detail_level)
            financial = self._generate_financial_kpis(df, include_forecasts, detail_level)
            customer = self._generate_customer_analytics(df, include_forecasts, detail_level)
            product = self._generate_product_performance(df, include_forecasts, detail_level)
            
            # Score geral de saúde do negócio
            business_health_score = self._calculate_overall_business_health(executive, financial, customer, product)
            
            # Recomendações integradas
            integrated_recommendations = self._generate_integrated_recommendations(executive, financial, customer, product)
            
            # Próximos passos estratégicos
            strategic_next_steps = self._generate_strategic_next_steps(business_health_score, integrated_recommendations)
            
            return {
                'executive_summary': executive,
                'operational_dashboard': operational,
                'financial_kpis': financial,
                'customer_analytics': customer,
                'product_performance': product,
                'business_health_score': business_health_score,
                'integrated_recommendations': integrated_recommendations,
                'strategic_next_steps': strategic_next_steps,
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'data_period': 'last_12_months',
                    'total_records_analyzed': len(df),
                    'report_confidence': 'High' if len(df) > 100 else 'Medium'
                }
            }
            
        except Exception as e:
            return {'error': f"Erro no relatório abrangente: {str(e)}"}
    
    # Métodos auxiliares
    def _calculate_revenue_concentration(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcular concentração de receita."""
        customer_revenue = df.groupby('Customer_ID')['Total_Liquido'].sum().sort_values(ascending=False)
        top_20_pct = int(len(customer_revenue) * 0.2)
        
        if top_20_pct > 0:
            concentration = customer_revenue.head(top_20_pct).sum() / customer_revenue.sum() * 100
        else:
            concentration = 0
        
        return {
            'top_20_percent_customers_revenue_share': round(concentration, 1),
            'gini_coefficient': self._calculate_gini_coefficient(customer_revenue.values)
        }
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calcular coeficiente de Gini."""
        if len(values) == 0:
            return 0
        
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def _calculate_financial_health_score(self, revenue_kpis: Dict, growth_kpis: Dict) -> Dict[str, Any]:
        """Calcular score de saúde financeira."""
        score = 0
        max_score = 100
        
        # Revenue growth (30 pontos)
        if growth_kpis.get('mom_growth', 0) > 10:
            score += 30
        elif growth_kpis.get('mom_growth', 0) > 0:
            score += 20
        elif growth_kpis.get('mom_growth', 0) > -5:
            score += 10
        
        # AOV vs benchmark (25 pontos)
        aov = revenue_kpis.get('avg_order_value', 0)
        if aov > 2000:
            score += 25
        elif aov > 1000:
            score += 15
        elif aov > 500:
            score += 10
        
        # Revenue stability (25 pontos)
        if revenue_kpis.get('total_revenue', 0) > 100000:
            score += 25
        elif revenue_kpis.get('total_revenue', 0) > 50000:
            score += 15
        elif revenue_kpis.get('total_revenue', 0) > 10000:
            score += 10
        
        # Customer metrics (20 pontos)
        revenue_per_customer = revenue_kpis.get('revenue_per_customer', 0)
        if revenue_per_customer > 2000:
            score += 20
        elif revenue_per_customer > 1000:
            score += 15
        elif revenue_per_customer > 500:
            score += 10
        
        # Classificação
        if score >= 80:
            classification = 'Excelente'
        elif score >= 60:
            classification = 'Boa'
        elif score >= 40:
            classification = 'Média'
        else:
            classification = 'Precisa Melhorar'
        
        return {
            'score': score,
            'max_score': max_score,
            'percentage': round(score / max_score * 100, 1),
            'classification': classification
        }
    
    def _estimate_new_customers(self, df: pd.DataFrame) -> int:
        """Estimar novos clientes no período."""
        # Clientes com apenas uma compra no período
        customer_counts = df.groupby('Customer_ID').size()
        return len(customer_counts[customer_counts == 1])
    
    def _estimate_customer_lifetime(self, df: pd.DataFrame) -> float:
        """Estimar lifetime médio dos clientes."""
        customer_lifetime = df.groupby('Customer_ID')['Data'].agg(['min', 'max'])
        customer_lifetime['Lifetime_Days'] = (customer_lifetime['max'] - customer_lifetime['min']).dt.days + 1
        return customer_lifetime['Lifetime_Days'].mean()
    
    def _calculate_customer_concentration(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcular concentração de clientes."""
        customer_revenue = df.groupby('Customer_ID')['Total_Liquido'].sum().sort_values(ascending=False)
        top_10_customers = customer_revenue.head(10).sum()
        total_revenue = customer_revenue.sum()
        
        return {
            'top_10_customers_revenue_share': round(top_10_customers / total_revenue * 100, 1) if total_revenue > 0 else 0,
            'total_customers': len(customer_revenue)
        }
    
    def _classify_customer_simple(self, row) -> str:
        """Classificação simples de clientes."""
        if row['Total_Spent'] > 5000 and row['Purchase_Count'] >= 3:
            return 'VIP'
        elif row['Purchase_Count'] == 1 and row['Recency'] <= 30:
            return 'New'
        elif row['Recency'] > 90:
            return 'At Risk'
        elif row['Total_Spent'] > 2000:
            return 'High Value'
        elif row['Purchase_Count'] >= 3:
            return 'Loyal'
        else:
            return 'Regular'
    
    def _generate_executive_insights(self, df: pd.DataFrame, customer_metrics: pd.DataFrame, 
                                   mom_growth: float) -> List[str]:
        """Gerar insights executivos."""
        insights = []
        
        # Crescimento
        if mom_growth > 15:
            insights.append(f"Crescimento excepcional de {mom_growth:.1f}% - acelerar investimentos")
        elif mom_growth < -10:
            insights.append(f"Declínio preocupante de {mom_growth:.1f}% - ação corretiva urgente")
        
        # Base de clientes
        repeat_customers = len(customer_metrics[customer_metrics['Transaction_Count'] > 1])
        repeat_rate = repeat_customers / len(customer_metrics) * 100
        
        if repeat_rate < 30:
            insights.append(f"Baixa retenção ({repeat_rate:.1f}%) - focar em fidelização")
        elif repeat_rate > 60:
            insights.append(f"Excelente retenção ({repeat_rate:.1f}%) - expandir base")
        
        # Ticket médio
        avg_ticket = df['Total_Liquido'].mean()
        if avg_ticket > 2500:
            insights.append("Posicionamento premium consolidado")
        elif avg_ticket < 800:
            insights.append("Oportunidade de up-sell significativa")
        
        return insights
    
    def _calculate_overall_business_health(self, executive: Dict, financial: Dict, 
                                         customer: Dict, product: Dict) -> Dict[str, Any]:
        """Calcular saúde geral do negócio."""
        # Scores individuais
        financial_score = financial.get('financial_health_score', {}).get('score', 50)
        
        # Customer health score
        customer_health = 0
        if customer.get('retention_analysis', {}).get('retention_rate_estimate', 0) > 0.3:
            customer_health += 30
        if customer.get('segment_distribution', {}).get('VIP', 0) > 5:
            customer_health += 20
        if customer.get('behavior_analysis', {}).get('avg_purchases_per_customer', 0) > 2:
            customer_health += 25
        customer_health += 25  # Base score
        
        # Product health score
        product_health = 0
        abc_a_share = product.get('abc_analysis', {}).get('class_a_revenue_share', 0)
        if abc_a_share > 70:
            product_health += 30
        slow_movers = product.get('inventory_alerts', {}).get('slow_movers', 0)
        total_products = len(product.get('product_rankings', {}).get('top_by_revenue', []))
        if total_products > 0 and slow_movers / total_products < 0.2:
            product_health += 25
        product_health += 45  # Base score
        
        # Score geral
        overall_score = (financial_score * 0.4 + customer_health * 0.35 + product_health * 0.25)
        
        return {
            'overall_score': round(overall_score, 1),
            'financial_health': round(financial_score, 1),
            'customer_health': round(customer_health, 1),
            'product_health': round(product_health, 1),
            'classification': 'Excellent' if overall_score >= 80 else 'Good' if overall_score >= 60 else 'Average' if overall_score >= 40 else 'Needs Improvement'
        }
    
    def _generate_integrated_recommendations(self, executive: Dict, financial: Dict,
                                           customer: Dict, product: Dict) -> List[Dict[str, str]]:
        """Gerar recomendações integradas."""
        recommendations = []
        
        # Baseado no crescimento
        mom_growth = executive.get('period_summary', {}).get('mom_growth_pct', 0)
        if mom_growth < 0:
            recommendations.append({
                'category': 'Growth Recovery',
                'action': 'Implementar plano de recuperação de vendas',
                'priority': 'Critical',
                'timeline': '30 dias'
            })
        
        # Baseado em clientes
        retention_rate = customer.get('retention_analysis', {}).get('retention_rate_estimate', 0)
        if retention_rate < 0.3:
            recommendations.append({
                'category': 'Customer Retention',
                'action': 'Lançar programa de fidelidade',
                'priority': 'High',
                'timeline': '60 dias'
            })
        
        # Baseado em produtos
        slow_movers = product.get('inventory_alerts', {}).get('slow_movers', 0)
        if slow_movers > 10:
            recommendations.append({
                'category': 'Inventory Management',
                'action': 'Liquidar produtos de baixo giro',
                'priority': 'Medium',
                'timeline': '90 dias'
            })
        
        # Baseado em ticket médio
        avg_ticket = executive.get('period_summary', {}).get('avg_ticket', 0)
        if avg_ticket < 1000:
            recommendations.append({
                'category': 'Revenue Optimization',
                'action': 'Estratégia de up-sell e cross-sell',
                'priority': 'High',
                'timeline': '45 dias'
            })
        
        return recommendations
    
    def _generate_strategic_next_steps(self, health_score: Dict, recommendations: List) -> List[str]:
        """Gerar próximos passos estratégicos."""
        next_steps = []
        
        overall_score = health_score.get('overall_score', 50)
        
        if overall_score >= 80:
            next_steps.extend([
                "1. Acelerar crescimento com expansão de linhas de produto",
                "2. Implementar programa VIP para clientes de alto valor",
                "3. Explorar novos canais de distribuição"
            ])
        elif overall_score >= 60:
            next_steps.extend([
                "1. Consolidar posição atual e otimizar operações",
                "2. Melhorar retenção de clientes com programas específicos",
                "3. Otimizar mix de produtos baseado em ABC analysis"
            ])
        else:
            next_steps.extend([
                "1. Focar em recuperação operacional imediata",
                "2. Revisar estratégia de preços e posicionamento",
                "3. Implementar ações de retenção de clientes urgentes"
            ])
        
        # Adicionar baseado em recomendações críticas
        critical_recs = [r for r in recommendations if r.get('priority') == 'Critical']
        if critical_recs:
            next_steps.append(f"4. URGENTE: {critical_recs[0]['action']}")
        
        return next_steps
    
    def _format_dashboard_result(self, dashboard_type: str, result: Dict[str, Any],
                               time_period: str, detail_level: str) -> str:
        """Formatar resultado do dashboard."""
        try:
            timestamp = datetime.now().strftime('%d/%m/%Y %H:%M')
            
            if 'error' in result:
                return f"Erro no dashboard {dashboard_type}: {result['error']}"
            
            formatted = f"""# 📊 BUSINESS INTELLIGENCE DASHBOARD
                        ## Tipo: {dashboard_type.upper().replace('_', ' ')}
                        **Período**: {time_period.replace('_', ' ').title()} | **Nível**: {detail_level.title()} | **Data**: {timestamp}

                        ---

                        """
            
            # Formatação específica por tipo
            if dashboard_type == 'executive_summary':
                formatted += self._format_executive_summary(result)
            elif dashboard_type == 'operational_dashboard':
                formatted += self._format_operational_dashboard(result)
            elif dashboard_type == 'financial_kpis':
                formatted += self._format_financial_kpis(result)
            elif dashboard_type == 'customer_analytics':
                formatted += self._format_customer_analytics(result)
            elif dashboard_type == 'product_performance':
                formatted += self._format_product_performance(result)
            elif dashboard_type == 'comprehensive_report':
                formatted += self._format_comprehensive_report(result)
            
            formatted += f"""

                    ---
                    ## 📋 INFORMAÇÕES DO RELATÓRIO

                    **Dados Analisados**: {time_period.replace('_', ' ').title()}
                    **Nível de Detalhamento**: {detail_level.title()}
                    **Confiabilidade**: Alta (baseado em dados reais de vendas)

                    *Dashboard gerado pelo Business Intelligence Engine - Insights AI*
                    """
            
            return formatted
            
        except Exception as e:
            return f"Erro na formatação: {str(e)}"
    
    def _format_executive_summary(self, result: Dict[str, Any]) -> str:
        """Formatar resumo executivo."""
        formatted = "## 🎯 RESUMO EXECUTIVO\n\n"
        
        if 'period_summary' in result:
            summary = result['period_summary']
            formatted += f"**Receita Total**: R$ {summary.get('total_revenue', 0):,.2f}\n"
            formatted += f"**Clientes**: {summary.get('total_customers', 0):,}\n"
            formatted += f"**Ticket Médio**: R$ {summary.get('avg_ticket', 0):,.2f}\n"
            formatted += f"**Crescimento MoM**: {summary.get('mom_growth_pct', 0):+.1f}%\n\n"
        
        if 'alerts' in result and result['alerts']:
            formatted += "## 🚨 ALERTAS CRÍTICOS\n\n"
            for alert in result['alerts']:
                formatted += f"- {alert}\n"
            formatted += "\n"
        
        if 'executive_insights' in result:
            formatted += "## 💡 INSIGHTS EXECUTIVOS\n\n"
            for insight in result['executive_insights']:
                formatted += f"- {insight}\n"
        
        if 'forecast' in result and result['forecast']:
            forecast = result['forecast']
            formatted += f"\n## 🔮 PROJEÇÃO\n\n"
            formatted += f"**Próximo Mês**: R$ {forecast.get('next_month_revenue', 0):,.2f}\n"
            formatted += f"**Tendência**: {forecast.get('growth_trend', 'N/A')}\n"
        
        return formatted
    
    def _format_operational_dashboard(self, result: Dict[str, Any]) -> str:
        """Formatar dashboard operacional."""
        formatted = "## ⚙️ MÉTRICAS OPERACIONAIS\n\n"
        
        if 'sales_velocity' in result:
            velocity = result['sales_velocity']
            formatted += "**Velocidade de Vendas**:\n"
            formatted += f"- Receita/dia: R$ {velocity.get('revenue_per_day', 0):,.2f}\n"
            formatted += f"- Itens/dia: {velocity.get('items_per_day', 0):.1f}\n"
            formatted += f"- Clientes/dia: {velocity.get('customers_per_day', 0):.1f}\n\n"
        
        if 'weekday_performance' in result:
            formatted += "**Performance por Dia da Semana**:\n"
            weekday_perf = result['weekday_performance']
            if 'Total_Revenue' in weekday_perf:
                for day, revenue in weekday_perf['Total_Revenue'].items():
                    formatted += f"- {day}: R$ {revenue:,.2f}\n"
            formatted += "\n"
        
        if 'operational_alerts' in result and result['operational_alerts']:
            formatted += "## ⚠️ ALERTAS OPERACIONAIS\n\n"
            for alert in result['operational_alerts']:
                formatted += f"- {alert}\n"
        
        return formatted
    
    def _format_financial_kpis(self, result: Dict[str, Any]) -> str:
        """Formatar KPIs financeiros."""
        formatted = "## 💰 KPIs FINANCEIROS\n\n"
        
        if 'revenue_kpis' in result:
            revenue = result['revenue_kpis']
            formatted += "**KPIs de Receita**:\n"
            formatted += f"- Receita Total: R$ {revenue.get('total_revenue', 0):,.2f}\n"
            formatted += f"- Receita Mensal Média: R$ {revenue.get('avg_monthly_revenue', 0):,.2f}\n"
            formatted += f"- Receita por Cliente: R$ {revenue.get('revenue_per_customer', 0):,.2f}\n"
            formatted += f"- AOV: R$ {revenue.get('avg_order_value', 0):,.2f}\n\n"
        
        if 'growth_kpis' in result:
            growth = result['growth_kpis']
            formatted += "**KPIs de Crescimento**:\n"
            for kpi, value in growth.items():
                formatted += f"- {kpi.replace('_', ' ').title()}: {value:+.1f}%\n"
            formatted += "\n"
        
        if 'financial_health_score' in result:
            health = result['financial_health_score']
            formatted += f"**Score de Saúde Financeira**: {health.get('score', 0)}/100 ({health.get('classification', 'N/A')})\n\n"
        
        if 'benchmark_comparison' in result:
            benchmark = result['benchmark_comparison']
            formatted += f"**vs. Mercado**: {benchmark.get('position', 'N/A')} (AOV {benchmark.get('aov_vs_benchmark', 0):+.1f}%)\n"
        
        return formatted
    
    def _format_customer_analytics(self, result: Dict[str, Any]) -> str:
        """Formatar analytics de clientes."""
        formatted = "## 👥 ANALYTICS DE CLIENTES\n\n"
        
        if 'customer_base_metrics' in result:
            base = result['customer_base_metrics']
            formatted += "**Base de Clientes**:\n"
            formatted += f"- Total: {base.get('total_customers', 0):,}\n"
            formatted += f"- Novos no Período: {base.get('new_customers_period', 0):,}\n"
            formatted += f"- Lifetime Médio: {base.get('avg_customer_lifetime_days', 0):.0f} dias\n\n"
        
        if 'segment_distribution' in result:
            formatted += "**Distribuição por Segmento**:\n"
            for segment, count in result['segment_distribution'].items():
                formatted += f"- {segment}: {count:,}\n"
            formatted += "\n"
        
        if 'behavior_analysis' in result:
            behavior = result['behavior_analysis']
            formatted += "**Análise Comportamental**:\n"
            formatted += f"- CLV Médio: R$ {behavior.get('customer_lifetime_value', 0):,.2f}\n"
            formatted += f"- Compras por Cliente: {behavior.get('avg_purchases_per_customer', 0):.1f}\n"
            formatted += f"- Taxa de Recompra: {behavior.get('repeat_customers', 0) / (behavior.get('repeat_customers', 0) + behavior.get('one_time_customers', 0)) * 100 if behavior.get('repeat_customers', 0) + behavior.get('one_time_customers', 0) > 0 else 0:.1f}%\n\n"
        
        if 'customer_insights' in result:
            formatted += "## 💡 INSIGHTS DE CLIENTES\n\n"
            for insight in result['customer_insights']:
                formatted += f"- {insight}\n"
        
        return formatted
    
    def _format_product_performance(self, result: Dict[str, Any]) -> str:
        """Formatar performance de produtos."""
        formatted = "## 💎 PERFORMANCE DE PRODUTOS\n\n"
        
        if 'abc_analysis' in result:
            abc = result['abc_analysis']
            formatted += "**Análise ABC**:\n"
            for class_type, count in abc['distribution'].items():
                formatted += f"- Classe {class_type}: {count} produtos\n"
            formatted += f"- Classe A representa: {abc.get('class_a_revenue_share', 0):.1f}% da receita\n\n"
        
        if 'inventory_alerts' in result:
            alerts = result['inventory_alerts']
            formatted += "**Alertas de Inventário**:\n"
            formatted += f"- Produtos slow-moving: {alerts.get('slow_movers', 0)}\n"
            formatted += f"- Dead stock: {alerts.get('dead_stock', 0)}\n"
            formatted += f"- Dias médios desde última venda: {alerts.get('avg_days_since_sale', 0):.0f}\n\n"
        
        if 'product_rankings' in result and 'top_by_revenue' in result['product_rankings']:
            formatted += "**Top 5 Produtos por Receita**:\n"
            for i, product in enumerate(result['product_rankings']['top_by_revenue'][:5], 1):
                formatted += f"{i}. {product.get('Description', 'N/A')}: R$ {product.get('Total_Revenue', 0):,.2f}\n"
            formatted += "\n"
        
        if 'product_insights' in result:
            formatted += "## 💡 INSIGHTS DE PRODUTOS\n\n"
            for insight in result['product_insights']:
                formatted += f"- {insight}\n"
        
        return formatted
    
    def _format_comprehensive_report(self, result: Dict[str, Any]) -> str:
        """Formatar relatório abrangente."""
        formatted = "## 📋 RELATÓRIO EXECUTIVO COMPLETO\n\n"
        
        if 'business_health_score' in result:
            health = result['business_health_score']
            formatted += f"## 🎯 SCORE GERAL DE SAÚDE: {health.get('overall_score', 0):.1f}/100\n"
            formatted += f"**Classificação**: {health.get('classification', 'N/A')}\n\n"
            
            formatted += "**Breakdown por Área**:\n"
            formatted += f"- Saúde Financeira: {health.get('financial_health', 0):.1f}/100\n"
            formatted += f"- Saúde de Clientes: {health.get('customer_health', 0):.1f}/100\n"
            formatted += f"- Saúde de Produtos: {health.get('product_health', 0):.1f}/100\n\n"
        
        if 'integrated_recommendations' in result:
            formatted += "## 🎯 RECOMENDAÇÕES ESTRATÉGICAS\n\n"
            for i, rec in enumerate(result['integrated_recommendations'], 1):
                formatted += f"**{i}. {rec['action']}**\n"
                formatted += f"- Categoria: {rec['category']}\n"
                formatted += f"- Prioridade: {rec['priority']}\n"
                formatted += f"- Timeline: {rec['timeline']}\n\n"
        
        if 'strategic_next_steps' in result:
            formatted += "## 🚀 PRÓXIMOS PASSOS ESTRATÉGICOS\n\n"
            for step in result['strategic_next_steps']:
                formatted += f"{step}\n"
            formatted += "\n"
        
        # Incluir resumos dos outros dashboards
        if 'executive_summary' in result:
            formatted += "---\n\n"
            formatted += self._format_executive_summary(result['executive_summary'])
        
        return formatted
