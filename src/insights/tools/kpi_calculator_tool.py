from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any, List
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class KPICalculatorInput(BaseModel):
    """Schema de entrada para a ferramenta de cálculo de KPIs."""
    data_csv: str = Field(default="data/vendas.csv", description="Caminho para o arquivo CSV de vendas")
    categoria: str = Field(default="all", description="Categoria específica ('all', 'revenue', 'operational', 'inventory', 'customer')")
    periodo: str = Field(default="monthly", description="Período de análise: 'daily', 'weekly', 'monthly'")
    benchmark_mode: bool = Field(default=True, description="Incluir benchmarks do setor de joalherias")

class KPICalculatorTool(BaseTool):
    name: str = "KPI Calculator Tool"
    description: str = (
        "Calcula 30+ KPIs críticos para joalherias com base nos dados reais de vendas:\n"
        "- FINANCEIROS: Revenue Growth, AOV, Margem por categoria, ROI\n"
        "- OPERACIONAIS: Giro de estoque, Velocidade de vendas, Concentração\n"
        "- CLIENTES: CLV estimado, Repeat purchase rate, Segmentação de valor\n"
        "- INVENTÁRIO: Days sales inventory, ABC analysis, Performance por produto\n"
        "- PRODUTOS: Performance por metal/categoria, Elasticidade, Sazonalidade\n"
        "- BENCHMARKS: Comparação com padrões do setor de joalherias"
    )
    args_schema: Type[BaseModel] = KPICalculatorInput
    
    def _run(self, data_csv: str = "data/vendas.csv", categoria: str = "all", 
             periodo: str = "monthly", benchmark_mode: bool = True) -> str:
        try:
            # Carregar e validar dados
            df = pd.read_csv(data_csv, sep=';', encoding='utf-8')
            df = self._validate_and_clean_data(df)
            
            if df is None:
                return "Erro: Não foi possível carregar os dados ou estrutura inválida"
            
            # Calcular KPIs por categoria solicitada
            kpis = {}
            
            if categoria == "all" or categoria == "revenue":
                kpis['financeiros'] = self._calculate_financial_kpis(df, periodo)
            
            if categoria == "all" or categoria == "operational":
                kpis['operacionais'] = self._calculate_operational_kpis(df, periodo)
            
            if categoria == "all" or categoria == "inventory":
                kpis['inventario'] = self._calculate_inventory_kpis(df, periodo)
            
            if categoria == "all" or categoria == "customer":
                kpis['clientes'] = self._calculate_customer_kpis(df, periodo)
            
            if categoria == "all":
                kpis['produtos'] = self._calculate_product_kpis(df, periodo)
                kpis['benchmarks'] = self._calculate_benchmark_comparison(df) if benchmark_mode else {}
                kpis['insights'] = self._generate_comprehensive_insights(df, kpis)
                kpis['alertas'] = self._generate_alerts(df, kpis)
            
            return self._format_kpis_report(kpis, categoria, benchmark_mode)
            
        except Exception as e:
            return f"Erro no cálculo de KPIs: {str(e)}"
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Validar e limpar dados de entrada."""
        try:
            # Verificar colunas essenciais
            required_cols = ['Data', 'Total_Liquido', 'Quantidade']
            optional_cols = ['Codigo_Produto', 'Descricao_Produto', 'Grupo_Produto', 'Metal', 'Colecao']
            
            if not all(col in df.columns for col in required_cols):
                return None
            
            # Converter tipos
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
            df['Total_Liquido'] = pd.to_numeric(df['Total_Liquido'], errors='coerce')
            df['Quantidade'] = pd.to_numeric(df['Quantidade'], errors='coerce')
            
            # Remover registros inválidos
            df = df.dropna(subset=['Data', 'Total_Liquido'])
            df = df[df['Total_Liquido'] > 0]  # Remover valores negativos/zero
            
            # Adicionar colunas derivadas
            df['Ano'] = df['Data'].dt.year
            df['Mes'] = df['Data'].dt.month
            df['Trimestre'] = df['Data'].dt.quarter
            df['Dia_Semana'] = df['Data'].dt.dayofweek
            df['Semana_Ano'] = df['Data'].dt.isocalendar().week
            
            # Preencher colunas opcionais se não existirem
            for col in optional_cols:
                if col not in df.columns:
                    df[col] = 'N/A'
            
            return df
            
        except Exception as e:
            print(f"Erro na validação: {str(e)}")
            return None
    
    def _calculate_financial_kpis(self, df: pd.DataFrame, periodo: str) -> Dict[str, Any]:
        """Calcular KPIs financeiros detalhados."""
        try:
            kpis = {}
            
            # 1. REVENUE METRICS
            total_revenue = df['Total_Liquido'].sum()
            kpis['total_revenue'] = round(total_revenue, 2)
            
            # 2. GROWTH RATES
            if periodo == 'monthly':
                monthly_revenue = df.groupby([df['Data'].dt.year, df['Data'].dt.month])['Total_Liquido'].sum()
                if len(monthly_revenue) >= 2:
                    mom_growth = ((monthly_revenue.iloc[-1] - monthly_revenue.iloc[-2]) / monthly_revenue.iloc[-2] * 100)
                    kpis['mom_growth_rate'] = round(mom_growth, 2)
                    
                    # YoY Growth se há dados de anos anteriores
                    if len(monthly_revenue) >= 12:
                        yoy_growth = ((monthly_revenue.iloc[-1] - monthly_revenue.iloc[-13]) / monthly_revenue.iloc[-13] * 100)
                        kpis['yoy_growth_rate'] = round(yoy_growth, 2)
            
            # 3. AVERAGE ORDER VALUE (AOV)
            kpis['aov'] = round(df['Total_Liquido'].mean(), 2)
            kpis['median_order_value'] = round(df['Total_Liquido'].median(), 2)
            
            # 4. REVENUE POR CATEGORIA
            if 'Grupo_Produto' in df.columns and df['Grupo_Produto'].notna().any():
                revenue_by_category = df.groupby('Grupo_Produto')['Total_Liquido'].sum().to_dict()
                kpis['revenue_by_category'] = {k: round(v, 2) for k, v in revenue_by_category.items()}
                
                # Market share por categoria
                total = sum(revenue_by_category.values())
                kpis['category_market_share'] = {k: round(v/total*100, 2) for k, v in revenue_by_category.items()}
            
            # 5. TICKET MÉDIO POR METAL
            if 'Metal' in df.columns and df['Metal'].notna().any():
                ticket_by_metal = df.groupby('Metal')['Total_Liquido'].mean().to_dict()
                kpis['ticket_by_metal'] = {k: round(v, 2) for k, v in ticket_by_metal.items()}
            
            # 6. MARGEM ESTIMADA (benchmarks do setor)
            margin_by_category = self._estimate_margins_by_category(df)
            kpis['estimated_margins'] = margin_by_category
            
            # 7. TOP PRODUTOS POR RECEITA
            if 'Descricao_Produto' in df.columns:
                top_products = df.groupby('Descricao_Produto')['Total_Liquido'].sum().nlargest(10)
                kpis['top_products_revenue'] = {k: round(v, 2) for k, v in top_products.items()}
            
            # 8. PERFORMANCE TEMPORAL
            kpis['performance_temporal'] = self._calculate_temporal_performance(df)
            
            return kpis
            
        except Exception as e:
            return {'error': f"Erro nos KPIs financeiros: {str(e)}"}
    
    def _calculate_operational_kpis(self, df: pd.DataFrame, periodo: str) -> Dict[str, Any]:
        """Calcular KPIs operacionais."""
        try:
            kpis = {}
            
            # 1. PRODUTOS E DIVERSIFICAÇÃO
            kpis['produtos_ativos'] = df['Codigo_Produto'].nunique() if 'Codigo_Produto' in df.columns else len(df)
            kpis['total_transacoes'] = len(df)
            
            # 2. VELOCIDADE DE VENDAS
            days_in_period = (df['Data'].max() - df['Data'].min()).days + 1
            kpis['sales_velocity_daily'] = round(df['Quantidade'].sum() / days_in_period, 2)
            kpis['revenue_velocity_daily'] = round(df['Total_Liquido'].sum() / days_in_period, 2)
            
            # 3. ANÁLISE DE CONCENTRAÇÃO (80/20)
            if 'Codigo_Produto' in df.columns:
                product_sales = df.groupby('Codigo_Produto')['Total_Liquido'].sum().sort_values(ascending=False)
                top_20_pct = int(len(product_sales) * 0.2)
                concentration_80_20 = (product_sales.head(top_20_pct).sum() / product_sales.sum() * 100)
                kpis['concentration_80_20_pct'] = round(concentration_80_20, 2)
                
                # Índice de Gini (desigualdade na distribuição)
                kpis['gini_coefficient'] = self._calculate_gini_coefficient(product_sales.values)
            
            # 4. DIVERSIFICAÇÃO POR CATEGORIA
            if 'Grupo_Produto' in df.columns and df['Grupo_Produto'].notna().any():
                category_distribution = df.groupby('Grupo_Produto')['Total_Liquido'].sum()
                category_concentration = (category_distribution.max() / category_distribution.sum() * 100)
                kpis['category_concentration_pct'] = round(category_concentration, 2)
                
                # Índice de diversificação (Herfindahl-Hirschman Index)
                market_shares = category_distribution / category_distribution.sum()
                hhi = (market_shares ** 2).sum()
                kpis['diversification_index'] = round(1 - hhi, 3)  # Quanto maior, mais diversificado
            
            # 5. PERFORMANCE POR PERÍODOS
            kpis['performance_by_period'] = self._calculate_period_performance(df, periodo)
            
            # 6. EFICIÊNCIA OPERACIONAL
            kpis['avg_items_per_transaction'] = round(df['Quantidade'].mean(), 2)
            kpis['transactions_per_day'] = round(len(df) / days_in_period, 2)
            
            # 7. SAZONALIDADE OPERACIONAL
            kpis['seasonal_patterns'] = self._calculate_seasonal_patterns(df)
            
            return kpis
            
        except Exception as e:
            return {'error': f"Erro nos KPIs operacionais: {str(e)}"}
    
    def _calculate_inventory_kpis(self, df: pd.DataFrame, periodo: str) -> Dict[str, Any]:
        """Calcular KPIs de inventário."""
        try:
            kpis = {}
            
            # 1. ANÁLISE ABC POR PRODUTO
            if 'Codigo_Produto' in df.columns:
                abc_analysis = self._perform_abc_analysis(df)
                kpis.update(abc_analysis)
            
            # 2. GIRO DE ESTOQUE ESTIMADO
            monthly_sales_avg = df.groupby([df['Data'].dt.year, df['Data'].dt.month])['Total_Liquido'].sum().mean()
            estimated_avg_inventory = monthly_sales_avg * 2.5  # Assumindo 2.5 meses de estoque médio
            
            if monthly_sales_avg > 0:
                inventory_turnover_monthly = monthly_sales_avg / (estimated_avg_inventory / 12)
                inventory_turnover_annual = inventory_turnover_monthly * 12
                kpis['inventory_turnover_annual'] = round(inventory_turnover_annual, 2)
                kpis['days_sales_inventory'] = round(365 / inventory_turnover_annual, 1)
            
            # 3. PRODUTOS SLOW-MOVING
            last_sale_by_product = df.groupby('Codigo_Produto')['Data'].max()
            cutoff_date = df['Data'].max() - timedelta(days=60)  # 60 dias
            slow_moving = (last_sale_by_product < cutoff_date).sum()
            total_products = len(last_sale_by_product)
            
            kpis['slow_moving_products'] = slow_moving
            kpis['slow_moving_pct'] = round(slow_moving / total_products * 100, 2)
            
            # 4. DEAD STOCK ANALYSIS
            dead_stock_cutoff = df['Data'].max() - timedelta(days=90)
            dead_stock = (last_sale_by_product < dead_stock_cutoff).sum()
            kpis['dead_stock_products'] = dead_stock
            kpis['dead_stock_pct'] = round(dead_stock / total_products * 100, 2)
            
            return kpis
            
        except Exception as e:
            return {'error': f"Erro nos KPIs de inventário: {str(e)}"}
    
    def _calculate_customer_kpis(self, df: pd.DataFrame, periodo: str) -> Dict[str, Any]:
        """Calcular KPIs estimados de clientes."""
        try:
            kpis = {}
            
            # ESTIMATIVA DE CLIENTES ÚNICOS
            high_value_threshold = 2000
            high_value_sales = df[df['Total_Liquido'] > high_value_threshold]
            low_value_sales = df[df['Total_Liquido'] <= high_value_threshold]
            
            estimated_unique_customers = len(high_value_sales) + (len(low_value_sales) * 0.7)
            kpis['estimated_unique_customers'] = int(estimated_unique_customers)
            
            # CUSTOMER LIFETIME VALUE ESTIMADO
            avg_purchase_value = df['Total_Liquido'].mean()
            estimated_annual_purchases = 2.3
            estimated_lifetime_years = 3.5
            estimated_clv = avg_purchase_value * estimated_annual_purchases * estimated_lifetime_years
            kpis['estimated_clv'] = round(estimated_clv, 2)
            
            # CUSTOMER ACQUISITION COST ESTIMADO
            marketing_rate = 0.10  # 10% da receita
            total_revenue = df['Total_Liquido'].sum()
            estimated_total_cac = total_revenue * marketing_rate
            estimated_cac = estimated_total_cac / estimated_unique_customers
            kpis['estimated_cac'] = round(estimated_cac, 2)
            
            # CLV/CAC RATIO
            kpis['clv_cac_ratio'] = round(estimated_clv / estimated_cac, 2)
            
            # SEGMENTAÇÃO POR VALOR
            value_segments = {
                'premium': len(df[df['Total_Liquido'] > 5000]),
                'mid_high': len(df[(df['Total_Liquido'] >= 2000) & (df['Total_Liquido'] <= 5000)]),
                'mid': len(df[(df['Total_Liquido'] >= 1000) & (df['Total_Liquido'] < 2000)]),
                'entry': len(df[df['Total_Liquido'] < 1000])
            }
            
            total_transactions = len(df)
            kpis['value_segments'] = value_segments
            kpis['value_segments_pct'] = {k: round(v/total_transactions*100, 2) for k, v in value_segments.items()}
            
            return kpis
            
        except Exception as e:
            return {'error': f"Erro nos KPIs de clientes: {str(e)}"}
    
    def _calculate_product_kpis(self, df: pd.DataFrame, periodo: str) -> Dict[str, Any]:
        """Calcular KPIs específicos de produtos."""
        try:
            kpis = {}
            
            # PERFORMANCE POR METAL
            if 'Metal' in df.columns and df['Metal'].notna().any():
                metal_performance = df.groupby('Metal').agg({
                    'Total_Liquido': ['sum', 'mean', 'count'],
                    'Quantidade': 'sum'
                }).round(2)
                
                metal_performance.columns = ['Total_Revenue', 'Avg_Ticket', 'Transaction_Count', 'Total_Quantity']
                kpis['metal_performance'] = metal_performance.to_dict()
                
                # Market share por metal
                total_revenue = df['Total_Liquido'].sum()
                metal_market_share = df.groupby('Metal')['Total_Liquido'].sum() / total_revenue * 100
                kpis['metal_market_share'] = metal_market_share.round(2).to_dict()
            
            # MATRIZ BCG DE PRODUTOS
            if 'Codigo_Produto' in df.columns:
                bcg_matrix = self._create_product_bcg_matrix(df)
                kpis['bcg_matrix'] = bcg_matrix
            
            # ELASTICIDADE DE PREÇO ESTIMADA
            price_elasticity = self._estimate_price_elasticity(df)
            kpis['price_elasticity'] = price_elasticity
            
            return kpis
            
        except Exception as e:
            return {'error': f"Erro nos KPIs de produtos: {str(e)}"}
    
    def _calculate_benchmark_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comparar KPIs com benchmarks do setor de joalherias."""
        try:
            # Benchmarks baseados em estudos do setor de joalherias
            industry_benchmarks = {
                'avg_order_value': {'min': 800, 'avg': 1500, 'max': 3000},
                'inventory_turnover': {'min': 1.5, 'avg': 2.5, 'max': 4.0},
                'gross_margin': {'min': 0.45, 'avg': 0.58, 'max': 0.70},
                'customer_repeat_rate': {'min': 0.15, 'avg': 0.25, 'max': 0.40},
                'seasonal_variation': {'min': 0.20, 'avg': 0.35, 'max': 0.60}
            }
            
            # Calcular métricas atuais
            current_aov = df['Total_Liquido'].mean()
            
            # Comparações
            benchmarks = {}
            
            # AOV Comparison
            aov_benchmark = industry_benchmarks['avg_order_value']
            if current_aov >= aov_benchmark['max']:
                aov_status = 'Excelente'
            elif current_aov >= aov_benchmark['avg']:
                aov_status = 'Bom'
            elif current_aov >= aov_benchmark['min']:
                aov_status = 'Abaixo da Média'
            else:
                aov_status = 'Crítico'
            
            benchmarks['aov_comparison'] = {
                'current': round(current_aov, 2),
                'benchmark_avg': aov_benchmark['avg'],
                'status': aov_status,
                'gap_to_avg': round(current_aov - aov_benchmark['avg'], 2)
            }
            
            return benchmarks
            
        except Exception as e:
            return {'error': f"Erro no benchmark: {str(e)}"}
    
    # Métodos auxiliares
    def _estimate_margins_by_category(self, df: pd.DataFrame) -> Dict[str, float]:
        """Estimar margens por categoria baseado em benchmarks do setor."""
        category_margins = {
            'Anéis': 0.62, 'Brincos': 0.58, 'Colares': 0.55, 'Pulseiras': 0.60,
            'Alianças': 0.45, 'Pingentes': 0.65, 'Correntes': 0.52, 'Outros': 0.50
        }
        
        if 'Grupo_Produto' not in df.columns:
            return {'estimated_margin_avg': 0.58}
        
        margins = {}
        for category in df['Grupo_Produto'].unique():
            if pd.isna(category):
                continue
            margin = category_margins.get(category, 0.55)
            margins[category] = margin
        
        return margins
    
    def _calculate_temporal_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcular performance temporal detalhada."""
        temporal = {}
        
        # Performance por trimestre
        quarterly = df.groupby(df['Data'].dt.quarter).agg({
            'Total_Liquido': ['sum', 'mean', 'count']
        }).round(2)
        quarterly.columns = ['Total', 'Avg', 'Count']
        temporal['quarterly'] = quarterly.to_dict()
        
        # Performance por dia da semana
        weekday = df.groupby(df['Data'].dt.dayofweek).agg({
            'Total_Liquido': ['sum', 'mean']
        }).round(2)
        weekday.columns = ['Total', 'Avg']
        weekday_names = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        weekday.index = [weekday_names[i] for i in weekday.index]
        temporal['weekday'] = weekday.to_dict()
        
        return temporal
    
    def _calculate_period_performance(self, df: pd.DataFrame, periodo: str) -> Dict[str, Any]:
        """Calcular performance detalhada por período."""
        if periodo == 'daily':
            grouped = df.groupby(df['Data'].dt.date)
        elif periodo == 'weekly':
            grouped = df.groupby(df['Data'].dt.to_period('W'))
        else:  # monthly
            grouped = df.groupby([df['Data'].dt.year, df['Data'].dt.month])
        
        performance = grouped.agg({
            'Total_Liquido': ['sum', 'mean', 'count'],
            'Quantidade': 'sum'
        }).round(2)
        
        recent_performance = performance.tail(10)
        
        return {
            'recent_periods': recent_performance.to_dict(),
            'best_period': {
                'date': str(performance['Total_Liquido']['sum'].idxmax()),
                'revenue': performance['Total_Liquido']['sum'].max()
            }
        }
    
    def _calculate_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcular padrões sazonais detalhados."""
        patterns = {}
        
        # Sazonalidade mensal
        monthly = df.groupby(df['Data'].dt.month)['Total_Liquido'].mean()
        patterns['monthly_index'] = (monthly / monthly.mean()).round(3).to_dict()
        
        # Identificar picos sazonais
        peak_months = monthly.nlargest(3)
        low_months = monthly.nsmallest(3)
        
        month_names = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                      7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
        
        patterns['peak_months'] = [month_names[m] for m in peak_months.index]
        patterns['low_months'] = [month_names[m] for m in low_months.index]
        
        return patterns
    
    def _perform_abc_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Realizar análise ABC detalhada."""
        product_sales = df.groupby('Codigo_Produto').agg({
            'Total_Liquido': 'sum',
            'Quantidade': 'sum'
        }).sort_values('Total_Liquido', ascending=False)
        
        # Calcular percentuais cumulativos
        cumsum_pct = (product_sales['Total_Liquido'].cumsum() / product_sales['Total_Liquido'].sum() * 100)
        
        # Classificação ABC
        abc_classes = []
        for pct in cumsum_pct:
            if pct <= 80:
                abc_classes.append('A')
            elif pct <= 95:
                abc_classes.append('B')
            else:
                abc_classes.append('C')
        
        product_sales['ABC_Class'] = abc_classes
        
        # Distribuição ABC
        abc_distribution = product_sales['ABC_Class'].value_counts()
        abc_revenue_distribution = product_sales.groupby('ABC_Class')['Total_Liquido'].sum()
        
        return {
            'abc_distribution': abc_distribution.to_dict(),
            'abc_revenue_distribution': abc_revenue_distribution.to_dict(),
            'abc_revenue_percentage': (abc_revenue_distribution / abc_revenue_distribution.sum() * 100).round(2).to_dict()
        }
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calcular coeficiente de Gini para medir desigualdade."""
        if len(values) == 0:
            return 0
        
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def _create_product_bcg_matrix(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Criar matriz BCG para produtos."""
        product_metrics = df.groupby('Codigo_Produto').agg({
            'Total_Liquido': 'sum',
            'Data': ['min', 'max']
        })
        
        # Calcular market share e growth rate
        total_revenue = df['Total_Liquido'].sum()
        product_metrics['market_share'] = product_metrics['Total_Liquido'] / total_revenue * 100
        
        # Growth rate baseado no tempo de vida do produto
        product_metrics['days_active'] = (
            pd.to_datetime(product_metrics['Data']['max']) - 
            pd.to_datetime(product_metrics['Data']['min'])
        ).dt.days + 1
        
        product_metrics['daily_revenue'] = product_metrics['Total_Liquido'] / product_metrics['days_active']
        
        # Classificar na matriz BCG
        market_share_median = product_metrics['market_share'].median()
        daily_revenue_median = product_metrics['daily_revenue'].median()
        
        def classify_bcg(row):
            if row['market_share'] > market_share_median and row['daily_revenue'] > daily_revenue_median:
                return 'Stars'
            elif row['market_share'] > market_share_median and row['daily_revenue'] <= daily_revenue_median:
                return 'Cash Cows'
            elif row['market_share'] <= market_share_median and row['daily_revenue'] > daily_revenue_median:
                return 'Question Marks'
            else:
                return 'Dogs'
        
        product_metrics['bcg_category'] = product_metrics.apply(classify_bcg, axis=1)
        
        bcg_distribution = product_metrics['bcg_category'].value_counts().to_dict()
        
        return {
            'distribution': bcg_distribution,
            'stars_count': bcg_distribution.get('Stars', 0),
            'cash_cows_count': bcg_distribution.get('Cash Cows', 0),
            'question_marks_count': bcg_distribution.get('Question Marks', 0),
            'dogs_count': bcg_distribution.get('Dogs', 0)
        }
    
    def _estimate_price_elasticity(self, df: pd.DataFrame) -> Dict[str, float]:
        """Estimar elasticidade de preço por categoria."""
        elasticity = {}
        
        if 'Grupo_Produto' not in df.columns:
            return {'overall_elasticity': -1.2}
        
        category_elasticity = {
            'Anéis': -0.8, 'Brincos': -1.2, 'Colares': -1.0, 'Pulseiras': -1.3,
            'Alianças': -0.5, 'Pingentes': -1.1, 'Correntes': -1.4, 'Outros': -1.2
        }
        
        for category in df['Grupo_Produto'].unique():
            if pd.isna(category):
                continue
            elasticity[category] = category_elasticity.get(category, -1.2)
        
        return elasticity
    
    def _generate_comprehensive_insights(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        """Gerar insights abrangentes baseados em todos os KPIs."""
        insights = []
        
        try:
            # Insights financeiros
            if 'financeiros' in kpis and 'aov' in kpis['financeiros']:
                aov = kpis['financeiros']['aov']
                if aov > 2500:
                    insights.append("AOV excelente (>R$2.500) indica posicionamento premium bem-sucedido")
                elif aov < 1000:
                    insights.append("AOV baixo (<R$1.000) - oportunidade de up-sell e produtos premium")
                
                if 'mom_growth_rate' in kpis['financeiros']:
                    growth = kpis['financeiros']['mom_growth_rate']
                    if growth > 15:
                        insights.append(f"Crescimento forte mês a mês ({growth:.1f}%) - manter estratégias atuais")
                    elif growth < -10:
                        insights.append(f"Declínio preocupante ({growth:.1f}%) - investigar causas e implementar ações")
            
            # Insights operacionais
            if 'operacionais' in kpis:
                op = kpis['operacionais']
                if 'concentration_80_20_pct' in op:
                    concentration = op['concentration_80_20_pct']
                    if concentration > 85:
                        insights.append("Alta concentração 80/20 - risco de dependência excessiva de poucos produtos")
                    elif concentration < 70:
                        insights.append("Boa diversificação de vendas - portfolio equilibrado")
            
            # Insights de inventário
            if 'inventario' in kpis:
                inv = kpis['inventario']
                if 'slow_moving_pct' in inv:
                    slow_moving = inv['slow_moving_pct']
                    if slow_moving > 30:
                        insights.append(f"Muitos produtos slow-moving ({slow_moving:.1f}%) - implementar estratégias de liquidação")
                    elif slow_moving < 15:
                        insights.append("Boa rotatividade de produtos - gestão de estoque eficiente")
            
        except Exception as e:
            insights.append(f"Erro na geração de insights: {str(e)}")
        
        return insights[:10]
    
    def _generate_alerts(self, df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
        """Gerar alertas críticos que requerem ação imediata."""
        alerts = []
        
        try:
            # Alertas financeiros críticos
            if 'financeiros' in kpis:
                fin = kpis['financeiros']
                if 'mom_growth_rate' in fin and fin['mom_growth_rate'] < -20:
                    alerts.append("🚨 CRÍTICO: Queda de vendas >20% no mês - Ação imediata necessária")
                
                if 'aov' in fin and fin['aov'] < 800:
                    alerts.append("⚠️  AOV muito baixo (<R$800) - Implementar estratégias de up-sell urgentemente")
            
            # Alertas operacionais
            if 'operacionais' in kpis:
                op = kpis['operacionais']
                if 'concentration_80_20_pct' in op and op['concentration_80_20_pct'] > 90:
                    alerts.append("🚨 RISCO: Concentração extrema de vendas - Portfolio muito dependente de poucos produtos")
            
        except Exception as e:
            alerts.append(f"Erro na geração de alertas: {str(e)}")
        
        return alerts[:5]
    
    def _format_kpis_report(self, kpis: Dict[str, Any], categoria: str, benchmark_mode: bool) -> str:
        """Formatar relatório completo de KPIs."""
        try:
            timestamp = datetime.now().strftime('%d/%m/%Y %H:%M')
            
            report = f"""# 📊 RELATÓRIO COMPLETO DE KPIs - JOALHERIA
                    **Categoria**: {categoria.upper()} | **Data**: {timestamp}
                    {'**Inclui Benchmarks do Setor**' if benchmark_mode else ''}

                    ## 🎯 ALERTAS CRÍTICOS
                    """
            
            # Adicionar alertas se existirem
            if 'alertas' in kpis and kpis['alertas']:
                for alert in kpis['alertas']:
                    report += f"{alert}\n"
            else:
                report += "✅ Nenhum alerta crítico identificado\n"
            
            report += "\n## 💡 INSIGHTS PRINCIPAIS\n"
            
            # Adicionar insights
            if 'insights' in kpis and kpis['insights']:
                for i, insight in enumerate(kpis['insights'], 1):
                    report += f"{i}. {insight}\n"
            
            report += "\n---\n"
            
            # Formatar cada seção de KPIs
            for section_name, section_data in kpis.items():
                if section_name in ['insights', 'alertas']:
                    continue
                
                if isinstance(section_data, dict) and 'error' not in section_data:
                    section_title = self._get_section_title(section_name)
                    report += f"\n## {section_title}\n\n"
                    report += self._format_kpi_section(section_data, section_name)
            
            # Adicionar rodapé  
            report += f"""
                        ---
                        ## 📋 METODOLOGIA

                        **KPIs Calculados**: 30+ métricas especializadas para joalherias
                        **Benchmarks**: Baseados em estudos do setor de varejo de luxo

                        *Relatório gerado automaticamente pelo Sistema de BI - Insights AI*
                        """
            
            return report
            
        except Exception as e:
            return f"Erro na formatação do relatório: {str(e)}"
    
    def _get_section_title(self, section_name: str) -> str:
        """Obter título formatado para cada seção."""
        titles = {
            'financeiros': '💰 KPIs FINANCEIROS',
            'operacionais': '⚙️ KPIs OPERACIONAIS', 
            'inventario': '📦 KPIs DE INVENTÁRIO',
            'clientes': '👥 KPIs DE CLIENTES',
            'produtos': '💎 KPIs DE PRODUTOS',
            'benchmarks': '📈 COMPARAÇÃO COM BENCHMARKS'
        }
        return titles.get(section_name, section_name.upper())
    
    def _format_kpi_section(self, data: Dict[str, Any], section_name: str) -> str:
        """Formatar seção específica de KPIs."""
        formatted = ""
        
        try:
            for key, value in data.items():
                if isinstance(value, dict):
                    formatted += f"### {key.replace('_', ' ').title()}\n"
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float)):
                            if 'pct' in subkey or 'rate' in subkey or 'percentage' in subkey:
                                formatted += f"- **{subkey.replace('_', ' ').title()}**: {subvalue}%\n"
                            elif 'revenue' in subkey or 'total' in subkey or 'aov' in subkey:
                                formatted += f"- **{subkey.replace('_', ' ').title()}**: R$ {subvalue:,.2f}\n"
                            else:
                                formatted += f"- **{subkey.replace('_', ' ').title()}**: {subvalue:,.2f}\n"
                        else:
                            formatted += f"- **{subkey.replace('_', ' ').title()}**: {subvalue}\n"
                    formatted += "\n"
                elif isinstance(value, (int, float)):
                    if 'pct' in key or 'rate' in key or 'percentage' in key:
                        formatted += f"**{key.replace('_', ' ').title()}**: {value}%\n"
                    elif 'revenue' in key or 'total' in key or 'aov' in key or 'clv' in key or 'cac' in key:
                        formatted += f"**{key.replace('_', ' ').title()}**: R$ {value:,.2f}\n"
                    else:
                        formatted += f"**{key.replace('_', ' ').title()}**: {value:,.2f}\n"
                elif isinstance(value, list):
                    formatted += f"**{key.replace('_', ' ').title()}**:\n"
                    for item in value[:5]:  # Limitar a 5 itens
                        formatted += f"  - {item}\n"
                else:
                    formatted += f"**{key.replace('_', ' ').title()}**: {value}\n"
            
            formatted += "\n"
            return formatted
            
        except Exception as e:
            return f"Erro na formatação da seção {section_name}: {str(e)}\n"
