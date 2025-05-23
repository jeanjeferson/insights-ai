from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any, List
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class CompetitiveIntelligenceInput(BaseModel):
    """Schema de entrada para análise competitiva."""
    analysis_type: str = Field(..., description="Tipo: 'market_position', 'pricing_analysis', 'trend_comparison', 'market_share_estimation', 'competitive_gaps'")
    data_csv: str = Field(default="data/vendas.csv", description="Caminho para arquivo CSV")
    market_segment: str = Field(default="joalherias", description="Segmento de mercado")
    benchmark_period: str = Field(default="quarterly", description="Período de benchmark: monthly, quarterly, yearly")

class CompetitiveIntelligenceTool(BaseTool):
    name: str = "Competitive Intelligence Tool"
    description: str = """
    Ferramenta de inteligência competitiva para joalherias:
    
    ANÁLISES DISPONÍVEIS:
    - market_position: Posicionamento no mercado e análise competitiva
    - pricing_analysis: Análise de estratégias de preço vs. concorrência
    - trend_comparison: Comparação de tendências com mercado
    - market_share_estimation: Estimativa de market share
    - competitive_gaps: Identificação de gaps competitivos e oportunidades
    
    BENCHMARKS INCLUÍDOS:
    - Padrões do setor de joalherias brasileiro
    - Métricas de performance comparativas
    - Análise de posicionamento de preços
    - Identificação de oportunidades de mercado
    """
    args_schema: Type[BaseModel] = CompetitiveIntelligenceInput
    
    def _run(self, analysis_type: str, data_csv: str = "data/vendas.csv", 
             market_segment: str = "joalherias", benchmark_period: str = "quarterly") -> str:
        try:
            # Carregar dados
            df = pd.read_csv(data_csv, sep=';', encoding='utf-8')
            df = self._prepare_competitive_data(df)
            
            if df is None or len(df) < 30:
                return "Erro: Dados insuficientes para análise competitiva (mínimo 30 registros)"
            
            # Carregar benchmarks do setor
            market_benchmarks = self._load_market_benchmarks(market_segment)
            
            # Dicionário de análises competitivas
            competitive_analyses = {
                'market_position': self._analyze_market_position,
                'pricing_analysis': self._analyze_competitive_pricing,
                'trend_comparison': self._compare_market_trends,
                'market_share_estimation': self._estimate_market_share,
                'competitive_gaps': self._identify_competitive_gaps
            }
            
            if analysis_type not in competitive_analyses:
                return f"Análise '{analysis_type}' não suportada. Opções: {list(competitive_analyses.keys())}"
            
            result = competitive_analyses[analysis_type](df, market_benchmarks, benchmark_period)
            return self._format_competitive_result(analysis_type, result, market_segment)
            
        except Exception as e:
            return f"Erro na análise competitiva: {str(e)}"
    
    def _prepare_competitive_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Preparar dados para análise competitiva."""
        try:
            # Converter data
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
            df = df.dropna(subset=['Data'])
            
            # Calcular métricas competitivas
            df['Ano_Mes'] = df['Data'].dt.to_period('M')
            df['Preco_Unitario'] = df['Total_Liquido'] / df['Quantidade'].replace(0, 1)
            
            # Categorização de preços
            df['Faixa_Preco'] = pd.cut(df['Preco_Unitario'], 
                                      bins=[0, 500, 1500, 3000, 10000, float('inf')],
                                      labels=['Economy', 'Mid', 'Premium', 'Luxury', 'Ultra-Luxury'])
            
            # Métricas mensais agregadas
            monthly_metrics = df.groupby('Ano_Mes').agg({
                'Total_Liquido': ['sum', 'mean', 'count'],
                'Quantidade': 'sum',
                'Preco_Unitario': ['mean', 'median', 'std']
            }).reset_index()
            
            # Flatten column names
            monthly_metrics.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                     for col in monthly_metrics.columns]
            
            df['Monthly_Metrics'] = df['Ano_Mes'].map(
                monthly_metrics.set_index('Ano_Mes').to_dict('index')
            )
            
            return df
            
        except Exception as e:
            print(f"Erro na preparação de dados competitivos: {str(e)}")
            return None
    
    def _load_market_benchmarks(self, market_segment: str) -> Dict[str, Any]:
        """Carregar benchmarks do mercado de joalherias."""
        # Benchmarks baseados em pesquisas do setor de joalherias brasileiro
        benchmarks = {
            'joalherias': {
                'market_size_billion_brl': 6.8,  # Mercado brasileiro 2024
                'annual_growth_rate': 0.035,     # 3.5% ao ano
                'average_ticket': {
                    'economy': {'min': 50, 'avg': 350, 'max': 499},
                    'mid': {'min': 500, 'avg': 1000, 'max': 1499},
                    'premium': {'min': 1500, 'avg': 2250, 'max': 2999},
                    'luxury': {'min': 3000, 'avg': 6500, 'max': 9999},
                    'ultra_luxury': {'min': 10000, 'avg': 25000, 'max': 100000}
                },
                'seasonal_patterns': {
                    'peak_months': [5, 12],  # Maio (Dia das Mães), Dezembro (Natal)
                    'low_months': [2, 3, 8], # Fevereiro, Março, Agosto
                    'seasonal_variation': 0.45  # 45% de variação sazonal
                },
                'category_distribution': {
                    'aneis': 0.28,
                    'brincos': 0.22,
                    'colares': 0.18,
                    'pulseiras': 0.15,
                    'aliancas': 0.12,
                    'outros': 0.05
                },
                'margin_benchmarks': {
                    'gross_margin_min': 0.45,
                    'gross_margin_avg': 0.58,
                    'gross_margin_max': 0.72,
                    'net_margin_avg': 0.15
                },
                'operational_benchmarks': {
                    'inventory_turnover': {'min': 1.5, 'avg': 2.3, 'max': 4.0},
                    'customer_repeat_rate': {'min': 0.15, 'avg': 0.25, 'max': 0.40},
                    'avg_items_per_transaction': {'min': 1.2, 'avg': 1.8, 'max': 2.5}
                },
                'digital_penetration': 0.23,  # 23% das vendas online
                'top_competitors_market_share': {
                    'vivara': 0.12,
                    'pandora': 0.08,
                    'rommanel': 0.06,
                    'casa_do_ouro': 0.04,
                    'outros_grandes': 0.20,
                    'independentes': 0.50
                }
            }
        }
        
        return benchmarks.get(market_segment, benchmarks['joalherias'])
    
    def _analyze_market_position(self, df: pd.DataFrame, benchmarks: Dict[str, Any], 
                                period: str) -> Dict[str, Any]:
        """Analisar posicionamento no mercado."""
        try:
            # Métricas da empresa
            company_metrics = {
                'total_revenue': df['Total_Liquido'].sum(),
                'avg_ticket': df['Total_Liquido'].mean(),
                'total_transactions': len(df),
                'avg_monthly_revenue': df.groupby('Ano_Mes')['Total_Liquido'].sum().mean(),
                'price_positioning': self._analyze_price_positioning(df, benchmarks)
            }
            
            # Comparação com benchmarks
            benchmark_comparison = {}
            
            # Ticket médio vs mercado
            market_avg_ticket = benchmarks['average_ticket']['mid']['avg']
            ticket_vs_market = (company_metrics['avg_ticket'] / market_avg_ticket - 1) * 100
            
            if company_metrics['avg_ticket'] >= benchmarks['average_ticket']['luxury']['min']:
                positioning = 'Luxury'
            elif company_metrics['avg_ticket'] >= benchmarks['average_ticket']['premium']['min']:
                positioning = 'Premium'
            elif company_metrics['avg_ticket'] >= benchmarks['average_ticket']['mid']['min']:
                positioning = 'Mid-Market'
            else:
                positioning = 'Economy'
            
            benchmark_comparison['ticket_analysis'] = {
                'company_avg_ticket': round(company_metrics['avg_ticket'], 2),
                'market_avg_ticket': market_avg_ticket,
                'difference_percent': round(ticket_vs_market, 1),
                'positioning': positioning
            }
            
            # Análise de crescimento
            monthly_growth = self._calculate_growth_metrics(df)
            market_growth = benchmarks['annual_growth_rate'] * 100
            
            benchmark_comparison['growth_analysis'] = {
                'company_growth': monthly_growth,
                'market_growth_annual': round(market_growth, 1),
                'growth_vs_market': 'Above' if monthly_growth.get('mom_avg', 0) > market_growth/12 else 'Below'
            }
            
            # Market share estimado
            estimated_market_share = self._estimate_local_market_share(
                company_metrics['total_revenue'], benchmarks
            )
            
            # Análise competitiva por categoria
            category_analysis = self._analyze_category_strength(df, benchmarks)
            
            # Insights de posicionamento
            positioning_insights = []
            
            if positioning == 'Luxury':
                positioning_insights.append("Posicionamento luxury - foco em exclusividade e experiência")
            elif positioning == 'Premium':
                positioning_insights.append("Posicionamento premium - oportunidade de expandir para luxury")
            
            if ticket_vs_market > 20:
                positioning_insights.append("Ticket médio significativamente acima do mercado - posicionamento diferenciado")
            elif ticket_vs_market < -20:
                positioning_insights.append("Ticket médio abaixo do mercado - oportunidade de up-sell")
            
            return {
                'company_metrics': company_metrics,
                'benchmark_comparison': benchmark_comparison,
                'estimated_market_share': estimated_market_share,
                'category_analysis': category_analysis,
                'positioning_insights': positioning_insights
            }
            
        except Exception as e:
            return {'error': f"Erro na análise de posicionamento: {str(e)}"}
    
    def _analyze_competitive_pricing(self, df: pd.DataFrame, benchmarks: Dict[str, Any],
                                   period: str) -> Dict[str, Any]:
        """Analisar estratégia de preços vs. concorrência."""
        try:
            # Distribuição de preços da empresa
            price_distribution = df['Faixa_Preco'].value_counts(normalize=True).to_dict()
            
            # Benchmarks de mercado por categoria
            market_price_benchmarks = benchmarks['average_ticket']
            
            # Análise por faixa de preço
            pricing_analysis = {}
            
            for category, data in market_price_benchmarks.items():
                if category == 'ultra_luxury':
                    continue
                    
                company_prices_in_category = df[
                    (df['Preco_Unitario'] >= data['min']) & 
                    (df['Preco_Unitario'] <= data['max'])
                ]['Preco_Unitario']
                
                if len(company_prices_in_category) > 0:
                    pricing_analysis[category] = {
                        'company_avg_price': round(company_prices_in_category.mean(), 2),
                        'market_avg_price': data['avg'],
                        'company_count': len(company_prices_in_category),
                        'company_percentage': round(len(company_prices_in_category) / len(df) * 100, 1),
                        'price_premium_discount': round(
                            (company_prices_in_category.mean() / data['avg'] - 1) * 100, 1
                        )
                    }
            
            # Análise de elasticidade de preço
            price_elasticity = self._calculate_price_elasticity_competitive(df)
            
            # Oportunidades de pricing
            pricing_opportunities = []
            
            # Identificar gaps de preço
            total_revenue_by_category = {
                category: data['company_count'] * data['company_avg_price'] 
                for category, data in pricing_analysis.items()
            }
            
            dominant_category = max(total_revenue_by_category, key=total_revenue_by_category.get)
            
            for category, data in pricing_analysis.items():
                if data['price_premium_discount'] < -10:
                    pricing_opportunities.append(f"Categoria {category}: Preços 10%+ abaixo do mercado - oportunidade de aumento")
                elif data['price_premium_discount'] > 25:
                    pricing_opportunities.append(f"Categoria {category}: Preços 25%+ acima do mercado - risco de perda de competitividade")
            
            # Análise de mix de preços
            price_mix_analysis = {
                'current_distribution': price_distribution,
                'market_opportunity': self._identify_price_mix_opportunities(price_distribution, benchmarks),
                'revenue_concentration': {
                    'dominant_category': dominant_category,
                    'revenue_percentage': round(
                        total_revenue_by_category[dominant_category] / sum(total_revenue_by_category.values()) * 100, 1
                    )
                }
            }
            
            return {
                'pricing_analysis': pricing_analysis,
                'price_elasticity': price_elasticity,
                'pricing_opportunities': pricing_opportunities,
                'price_mix_analysis': price_mix_analysis
            }
            
        except Exception as e:
            return {'error': f"Erro na análise de preços: {str(e)}"}
    
    def _compare_market_trends(self, df: pd.DataFrame, benchmarks: Dict[str, Any],
                             period: str) -> Dict[str, Any]:
        """Comparar tendências com o mercado."""
        try:
            # Análise temporal da empresa
            if period == 'monthly':
                company_trends = df.groupby(df['Data'].dt.to_period('M')).agg({
                    'Total_Liquido': 'sum',
                    'Quantidade': 'sum',
                    'Preco_Unitario': 'mean'
                })
            elif period == 'quarterly':
                company_trends = df.groupby(df['Data'].dt.to_period('Q')).agg({
                    'Total_Liquido': 'sum',
                    'Quantidade': 'sum',
                    'Preco_Unitario': 'mean'
                })
            else:  # yearly
                company_trends = df.groupby(df['Data'].dt.year).agg({
                    'Total_Liquido': 'sum',
                    'Quantidade': 'sum',
                    'Preco_Unitario': 'mean'
                })
            
            # Calcular taxas de crescimento
            company_growth_rates = {
                'revenue_growth': company_trends['Total_Liquido'].pct_change().mean() * 100,
                'volume_growth': company_trends['Quantidade'].pct_change().mean() * 100,
                'price_growth': company_trends['Preco_Unitario'].pct_change().mean() * 100
            }
            
            # Comparação com benchmarks de mercado
            market_growth_rate = benchmarks['annual_growth_rate'] * 100
            
            # Ajustar para o período
            if period == 'monthly':
                market_growth_rate = market_growth_rate / 12
            elif period == 'quarterly':
                market_growth_rate = market_growth_rate / 4
            
            trend_comparison = {
                'company_vs_market': {
                    'company_revenue_growth': round(company_growth_rates['revenue_growth'], 2),
                    'market_growth_benchmark': round(market_growth_rate, 2),
                    'performance_vs_market': 'Outperforming' if company_growth_rates['revenue_growth'] > market_growth_rate else 'Underperforming'
                }
            }
            
            # Análise de sazonalidade vs. mercado
            company_seasonality = self._analyze_company_seasonality(df)
            market_seasonality = benchmarks['seasonal_patterns']
            
            seasonality_comparison = {
                'company_peak_months': company_seasonality['peak_months'],
                'market_peak_months': market_seasonality['peak_months'],
                'alignment_with_market': len(set(company_seasonality['peak_months']) & 
                                           set(market_seasonality['peak_months'])) > 0,
                'company_seasonal_variation': company_seasonality['variation'],
                'market_seasonal_variation': market_seasonality['seasonal_variation']
            }
            
            # Tendências por categoria
            category_trends = {}
            if 'Grupo_Produto' in df.columns:
                for categoria in df['Grupo_Produto'].unique():
                    if pd.isna(categoria):
                        continue
                    
                    cat_data = df[df['Grupo_Produto'] == categoria]
                    cat_trend = cat_data.groupby(cat_data['Data'].dt.to_period('M'))['Total_Liquido'].sum()
                    
                    if len(cat_trend) > 1:
                        cat_growth = cat_trend.pct_change().mean() * 100
                        market_cat_share = benchmarks['category_distribution'].get(categoria.lower(), 0)
                        
                        category_trends[categoria] = {
                            'growth_rate': round(cat_growth, 2),
                            'market_share_benchmark': round(market_cat_share * 100, 1),
                            'trend_strength': 'Strong' if cat_growth > market_growth_rate else 'Weak'
                        }
            
            # Insights de tendências
            trend_insights = []
            
            if company_growth_rates['revenue_growth'] > market_growth_rate * 1.5:
                trend_insights.append("Crescimento bem acima do mercado - estratégias eficazes")
            elif company_growth_rates['revenue_growth'] < market_growth_rate * 0.5:
                trend_insights.append("Crescimento abaixo do mercado - necessita revisão estratégica")
            
            if seasonality_comparison['alignment_with_market']:
                trend_insights.append("Sazonalidade alinhada com mercado - boa captura de demanda sazonal")
            else:
                trend_insights.append("Sazonalidade divergente do mercado - oportunidade ou risco a investigar")
            
            return {
                'trend_comparison': trend_comparison,
                'seasonality_comparison': seasonality_comparison,
                'category_trends': category_trends,
                'trend_insights': trend_insights
            }
            
        except Exception as e:
            return {'error': f"Erro na comparação de tendências: {str(e)}"}
    
    def _estimate_market_share(self, df: pd.DataFrame, benchmarks: Dict[str, Any],
                             period: str) -> Dict[str, Any]:
        """Estimar market share."""
        try:
            # Receita da empresa
            company_revenue = df['Total_Liquido'].sum()
            
            # Estimativa do mercado total (baseado em benchmarks)
            market_size_billion = benchmarks['market_size_billion_brl']
            market_size_total = market_size_billion * 1_000_000_000  # Converter para reais
            
            # Estimativa de market share nacional
            national_market_share = (company_revenue / market_size_total) * 100
            
            # Estimativa de mercado local/regional
            # Assumindo que a empresa atua em mercado regional (1-5% do mercado nacional)
            estimated_regional_market = market_size_total * 0.02  # 2% do mercado nacional
            regional_market_share = (company_revenue / estimated_regional_market) * 100
            
            # Análise por segmento
            segment_analysis = {}
            
            if 'Faixa_Preco' in df.columns:
                for segment in df['Faixa_Preco'].unique():
                    if pd.isna(segment):
                        continue
                    
                    segment_revenue = df[df['Faixa_Preco'] == segment]['Total_Liquido'].sum()
                    segment_percentage = (segment_revenue / company_revenue) * 100
                    
                    # Estimativa de participação no segmento
                    segment_market_size = estimated_regional_market * self._get_segment_market_percentage(segment)
                    segment_market_share = (segment_revenue / segment_market_size) * 100
                    
                    segment_analysis[segment] = {
                        'revenue': round(segment_revenue, 2),
                        'percentage_of_company': round(segment_percentage, 1),
                        'estimated_segment_market_share': min(round(segment_market_share, 2), 100)  # Cap at 100%
                    }
            
            # Comparação com principais concorrentes
            competitor_analysis = self._analyze_competitive_landscape(company_revenue, benchmarks)
            
            # Potencial de crescimento
            growth_potential = {
                'current_position': 'Niche Player' if regional_market_share < 1 else 
                                  'Regional Player' if regional_market_share < 5 else 'Market Leader',
                'growth_opportunity': max(0, 10 - regional_market_share),  # Assumindo 10% como teto realista
                'expansion_recommendations': self._generate_expansion_recommendations(regional_market_share, segment_analysis)
            }
            
            return {
                'market_share_estimation': {
                    'company_revenue': round(company_revenue, 2),
                    'estimated_national_market_share': round(national_market_share, 6),
                    'estimated_regional_market_share': round(regional_market_share, 2),
                    'market_position': growth_potential['current_position']
                },
                'segment_analysis': segment_analysis,
                'competitor_analysis': competitor_analysis,
                'growth_potential': growth_potential
            }
            
        except Exception as e:
            return {'error': f"Erro na estimativa de market share: {str(e)}"}
    
    def _identify_competitive_gaps(self, df: pd.DataFrame, benchmarks: Dict[str, Any],
                                 period: str) -> Dict[str, Any]:
        """Identificar gaps competitivos e oportunidades."""
        try:
            # Análise de gaps por categoria
            category_gaps = {}
            market_category_dist = benchmarks['category_distribution']
            
            if 'Grupo_Produto' in df.columns:
                company_category_dist = df.groupby('Grupo_Produto')['Total_Liquido'].sum()
                company_category_dist = company_category_dist / company_category_dist.sum()
                
                for categoria, market_share in market_category_dist.items():
                    categoria_title = categoria.title()
                    company_share = company_category_dist.get(categoria_title, 0)
                    
                    gap = market_share - company_share
                    
                    category_gaps[categoria_title] = {
                        'market_share_benchmark': round(market_share * 100, 1),
                        'company_share': round(company_share * 100, 1),
                        'gap_percentage': round(gap * 100, 1),
                        'opportunity_size': 'High' if abs(gap) > 0.05 else 'Medium' if abs(gap) > 0.02 else 'Low'
                    }
            
            # Análise de gaps operacionais
            operational_gaps = {}
            operational_benchmarks = benchmarks['operational_benchmarks']
            
            # Inventory Turnover (estimado)
            company_revenue = df['Total_Liquido'].sum()
            estimated_inventory = company_revenue * 0.4  # Assumindo 40% da receita em estoque
            estimated_turnover = company_revenue / estimated_inventory if estimated_inventory > 0 else 0
            
            turnover_benchmark = operational_benchmarks['inventory_turnover']['avg']
            turnover_gap = (estimated_turnover - turnover_benchmark) / turnover_benchmark * 100
            
            operational_gaps['inventory_turnover'] = {
                'company_estimated': round(estimated_turnover, 2),
                'market_benchmark': turnover_benchmark,
                'gap_percentage': round(turnover_gap, 1),
                'status': 'Above' if turnover_gap > 0 else 'Below'
            }
            
            # Average Items per Transaction
            company_items_per_transaction = df['Quantidade'].mean() if 'Quantidade' in df.columns else 1.0
            items_benchmark = operational_benchmarks['avg_items_per_transaction']['avg']
            items_gap = (company_items_per_transaction - items_benchmark) / items_benchmark * 100
            
            operational_gaps['items_per_transaction'] = {
                'company_avg': round(company_items_per_transaction, 2),
                'market_benchmark': items_benchmark,
                'gap_percentage': round(items_gap, 1),
                'cross_sell_opportunity': 'High' if items_gap < -20 else 'Medium' if items_gap < 0 else 'Low'
            }
            
            # Gaps de preço por segmento
            pricing_gaps = self._identify_pricing_gaps(df, benchmarks)
            
            # Gaps digitais
            digital_gaps = self._analyze_digital_gaps(df, benchmarks)
            
            # Priorização de oportunidades
            opportunity_matrix = self._create_opportunity_matrix(category_gaps, operational_gaps, pricing_gaps)
            
            # Recomendações estratégicas
            strategic_recommendations = []
            
            # Top 3 category gaps
            top_category_gaps = sorted(category_gaps.items(), 
                                     key=lambda x: abs(x[1]['gap_percentage']), reverse=True)[:3]
            
            for categoria, gap_data in top_category_gaps:
                if gap_data['gap_percentage'] > 5:
                    strategic_recommendations.append(
                        f"Expandir em {categoria}: Gap de {gap_data['gap_percentage']:.1f}% vs. mercado"
                    )
            
            # Operational improvements
            if operational_gaps['inventory_turnover']['gap_percentage'] < -20:
                strategic_recommendations.append("Melhorar giro de estoque - 20%+ abaixo do mercado")
            
            if operational_gaps['items_per_transaction']['cross_sell_opportunity'] == 'High':
                strategic_recommendations.append("Implementar estratégias de cross-sell - alta oportunidade")
            
            return {
                'category_gaps': category_gaps,
                'operational_gaps': operational_gaps,
                'pricing_gaps': pricing_gaps,
                'digital_gaps': digital_gaps,
                'opportunity_matrix': opportunity_matrix,
                'strategic_recommendations': strategic_recommendations[:5]  # Top 5
            }
            
        except Exception as e:
            return {'error': f"Erro na identificação de gaps: {str(e)}"}
    
    # Métodos auxiliares
    def _analyze_price_positioning(self, df: pd.DataFrame, benchmarks: Dict[str, Any]) -> Dict[str, Any]:
        """Analisar posicionamento de preços."""
        price_stats = {
            'min_price': df['Preco_Unitario'].min(),
            'max_price': df['Preco_Unitario'].max(),
            'avg_price': df['Preco_Unitario'].mean(),
            'median_price': df['Preco_Unitario'].median(),
            'std_price': df['Preco_Unitario'].std()
        }
        
        # Distribuição por faixa
        price_distribution = df['Faixa_Preco'].value_counts(normalize=True).to_dict()
        
        return {
            'price_statistics': {k: round(v, 2) for k, v in price_stats.items()},
            'price_distribution': {k: round(v * 100, 1) for k, v in price_distribution.items()}
        }
    
    def _calculate_growth_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcular métricas de crescimento."""
        monthly_revenue = df.groupby('Ano_Mes')['Total_Liquido'].sum()
        
        if len(monthly_revenue) < 2:
            return {'mom_avg': 0, 'mom_last': 0}
        
        mom_growth = monthly_revenue.pct_change().dropna()
        
        return {
            'mom_avg': round(mom_growth.mean() * 100, 2),
            'mom_last': round(mom_growth.iloc[-1] * 100, 2) if len(mom_growth) > 0 else 0
        }
    
    def _estimate_local_market_share(self, company_revenue: float, benchmarks: Dict[str, Any]) -> Dict[str, float]:
        """Estimar market share local."""
        # Estimativas conservadoras baseadas em tamanho da empresa
        if company_revenue > 10_000_000:  # > R$ 10M
            estimated_local_share = min((company_revenue / 50_000_000) * 100, 15)  # Max 15%
        elif company_revenue > 1_000_000:  # > R$ 1M
            estimated_local_share = min((company_revenue / 10_000_000) * 100, 8)   # Max 8%
        else:
            estimated_local_share = min((company_revenue / 2_000_000) * 100, 3)    # Max 3%
        
        return {
            'estimated_local_share': round(estimated_local_share, 2),
            'confidence_level': 'Medium'  # Estimativa baseada em benchmarks
        }
    
    def _analyze_category_strength(self, df: pd.DataFrame, benchmarks: Dict[str, Any]) -> Dict[str, Any]:
        """Analisar força por categoria."""
        if 'Grupo_Produto' not in df.columns:
            return {}
        
        category_performance = {}
        category_revenue = df.groupby('Grupo_Produto')['Total_Liquido'].agg(['sum', 'mean', 'count'])
        
        for categoria in category_revenue.index:
            performance = category_revenue.loc[categoria]
            
            category_performance[categoria] = {
                'total_revenue': round(performance['sum'], 2),
                'avg_ticket': round(performance['mean'], 2),
                'transaction_count': int(performance['count']),
                'revenue_share': round(performance['sum'] / df['Total_Liquido'].sum() * 100, 1)
            }
        
        return category_performance
    
    def _calculate_price_elasticity_competitive(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcular elasticidade de preço competitiva."""
        # Análise básica de elasticidade
        monthly_data = df.groupby('Ano_Mes').agg({
            'Preco_Unitario': 'mean',
            'Quantidade': 'sum'
        })
        
        if len(monthly_data) < 3:
            return {'elasticity': -1.2, 'confidence': 'Low'}
        
        price_change = monthly_data['Preco_Unitario'].pct_change()
        quantity_change = monthly_data['Quantidade'].pct_change()
        
        # Elasticidade simples
        elasticity_values = quantity_change / price_change.replace(0, np.nan)
        elasticity_values = elasticity_values.dropna()
        
        if len(elasticity_values) == 0:
            return {'elasticity': -1.2, 'confidence': 'Low'}
        
        avg_elasticity = elasticity_values.mean()
        
        return {
            'elasticity': round(avg_elasticity, 2),
            'confidence': 'Medium' if len(elasticity_values) > 2 else 'Low'
        }
    
    def _identify_price_mix_opportunities(self, current_distribution: Dict[str, float], 
                                        benchmarks: Dict[str, Any]) -> List[str]:
        """Identificar oportunidades no mix de preços."""
        opportunities = []
        
        # Oportunidades baseadas na distribuição atual vs. potencial de mercado
        premium_luxury_share = current_distribution.get('Premium', 0) + current_distribution.get('Luxury', 0)
        
        if premium_luxury_share < 0.3:
            opportunities.append("Expandir produtos premium/luxury - baixa participação atual")
        
        economy_share = current_distribution.get('Economy', 0)
        if economy_share > 0.4:
            opportunities.append("Reduzir dependência de produtos economy - migrar para mid/premium")
        
        return opportunities
    
    def _analyze_company_seasonality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisar sazonalidade da empresa."""
        monthly_sales = df.groupby(df['Data'].dt.month)['Total_Liquido'].sum()
        
        if len(monthly_sales) < 6:
            return {'peak_months': [], 'variation': 0}
        
        avg_monthly = monthly_sales.mean()
        peak_months = monthly_sales[monthly_sales > avg_monthly * 1.2].index.tolist()
        
        variation = (monthly_sales.max() - monthly_sales.min()) / avg_monthly
        
        return {
            'peak_months': peak_months,
            'variation': round(variation, 2)
        }
    
    def _get_segment_market_percentage(self, segment: str) -> float:
        """Obter percentual de mercado por segmento."""
        segment_percentages = {
            'Economy': 0.25,
            'Mid': 0.40,
            'Premium': 0.25,
            'Luxury': 0.08,
            'Ultra-Luxury': 0.02
        }
        return segment_percentages.get(segment, 0.20)
    
    def _analyze_competitive_landscape(self, company_revenue: float, benchmarks: Dict[str, Any]) -> Dict[str, Any]:
        """Analisar cenário competitivo."""
        competitors = benchmarks['top_competitors_market_share']
        
        # Estimar receita dos principais concorrentes (muito aproximado)
        market_size = benchmarks['market_size_billion_brl'] * 1_000_000_000
        
        competitor_revenues = {}
        for competitor, share in competitors.items():
            if competitor != 'independentes':
                competitor_revenues[competitor] = market_size * share
        
        # Posição relativa
        company_position = 'Independent'
        for competitor, revenue in competitor_revenues.items():
            if company_revenue > revenue * 0.1:  # Se for > 10% do concorrente
                company_position = f"Competitor to {competitor}"
                break
        
        return {
            'competitive_position': company_position,
            'market_leaders': list(competitors.keys())[:3],
            'estimated_gap_to_leader': round(
                (competitor_revenues.get('vivara', 0) - company_revenue) / 1_000_000, 1
            )
        }
    
    def _generate_expansion_recommendations(self, market_share: float, segment_analysis: Dict[str, Any]) -> List[str]:
        """Gerar recomendações de expansão."""
        recommendations = []
        
        if market_share < 1:
            recommendations.append("Focar em crescimento orgânico local antes de expansão geográfica")
        elif market_share < 3:
            recommendations.append("Considerar expansão para mercados adjacentes")
        else:
            recommendations.append("Avaliar aquisições ou parcerias estratégicas")
        
        # Baseado na análise de segmentos
        if segment_analysis:
            strongest_segment = max(segment_analysis.items(), 
                                  key=lambda x: x[1]['estimated_segment_market_share'])
            recommendations.append(f"Fortalecer posição em {strongest_segment[0]} - segmento mais forte")
        
        return recommendations
    
    def _identify_pricing_gaps(self, df: pd.DataFrame, benchmarks: Dict[str, Any]) -> Dict[str, Any]:
        """Identificar gaps de pricing."""
        current_avg = df['Preco_Unitario'].mean()
        
        gaps = {}
        for category, price_range in benchmarks['average_ticket'].items():
            if category == 'ultra_luxury':
                continue
                
            market_avg = price_range['avg']
            gap_percentage = (current_avg - market_avg) / market_avg * 100
            
            gaps[category] = {
                'market_price': market_avg,
                'gap_percentage': round(gap_percentage, 1),
                'opportunity': 'Increase' if gap_percentage < -10 else 'Maintain' if abs(gap_percentage) <= 10 else 'Reassess'
            }
        
        return gaps
    
    def _analyze_digital_gaps(self, df: pd.DataFrame, benchmarks: Dict[str, Any]) -> Dict[str, Any]:
        """Analisar gaps digitais."""
        # Análise básica - assumindo dados limitados sobre canal digital
        digital_penetration_market = benchmarks['digital_penetration']
        
        # Estimativa básica baseada em padrões de transação
        # (seria necessário dados específicos de canal para análise precisa)
        
        return {
            'market_digital_penetration': round(digital_penetration_market * 100, 1),
            'estimated_company_digital': 'Data not available',
            'digital_opportunity': 'High' if digital_penetration_market > 0.2 else 'Medium'
        }
    
    def _create_opportunity_matrix(self, category_gaps: Dict, operational_gaps: Dict, 
                                 pricing_gaps: Dict) -> Dict[str, Any]:
        """Criar matriz de oportunidades."""
        opportunities = []
        
        # Category opportunities
        for categoria, gap_data in category_gaps.items():
            if gap_data['opportunity_size'] == 'High':
                opportunities.append({
                    'type': 'Category',
                    'opportunity': categoria,
                    'impact': 'High',
                    'effort': 'Medium',
                    'priority': 'High'
                })
        
        # Operational opportunities
        for metric, gap_data in operational_gaps.items():
            if abs(gap_data['gap_percentage']) > 20:
                opportunities.append({
                    'type': 'Operational',
                    'opportunity': metric.replace('_', ' ').title(),
                    'impact': 'Medium',
                    'effort': 'Medium',
                    'priority': 'Medium'
                })
        
        # Priorizar por impacto vs esforço
        high_priority = [opp for opp in opportunities if opp['priority'] == 'High']
        medium_priority = [opp for opp in opportunities if opp['priority'] == 'Medium']
        
        return {
            'high_priority': high_priority,
            'medium_priority': medium_priority,
            'total_opportunities': len(opportunities)
        }
    
    def _format_competitive_result(self, analysis_type: str, result: Dict[str, Any], 
                                 market_segment: str) -> str:
        """Formatar resultado da análise competitiva."""
        try:
            timestamp = datetime.now().strftime('%d/%m/%Y %H:%M')
            
            if 'error' in result:
                return f"Erro na análise competitiva {analysis_type}: {result['error']}"
            
            formatted = f"""# 🏆 ANÁLISE DE INTELIGÊNCIA COMPETITIVA
                            ## Tipo: {analysis_type.upper().replace('_', ' ')}
                            **Segmento**: {market_segment.title()} | **Data**: {timestamp}

                            ---

                            """
            
            # Formatação específica por tipo
            if analysis_type == 'market_position':
                formatted += self._format_market_position(result)
            elif analysis_type == 'pricing_analysis':
                formatted += self._format_pricing_analysis(result)
            elif analysis_type == 'trend_comparison':
                formatted += self._format_trend_comparison(result)
            elif analysis_type == 'market_share_estimation':
                formatted += self._format_market_share(result)
            elif analysis_type == 'competitive_gaps':
                formatted += self._format_competitive_gaps(result)
            
            formatted += f"""

                    ---
                    ## 📋 DISCLAIMER

                    **Dados de Mercado**: Baseados em pesquisas setoriais e benchmarks públicos
                    **Estimativas**: Market share e análises competitivas são aproximações
                    **Período**: Análise baseada nos dados históricos disponíveis

                    *Relatório gerado pelo Competitive Intelligence Tool - Insights AI*
                    """
            
            return formatted
            
        except Exception as e:
            return f"Erro na formatação: {str(e)}"
    
    def _format_market_position(self, result: Dict[str, Any]) -> str:
        """Formatar análise de posicionamento."""
        formatted = "## 🎯 POSICIONAMENTO NO MERCADO\n\n"
        
        if 'benchmark_comparison' in result:
            ticket = result['benchmark_comparison']['ticket_analysis']
            formatted += f"**Posicionamento**: {ticket['positioning']}\n"
            formatted += f"**Ticket Médio**: R$ {ticket['company_avg_ticket']:,.2f}\n"
            formatted += f"**vs. Mercado**: {ticket['difference_percent']:+.1f}%\n\n"
        
        if 'estimated_market_share' in result:
            share = result['estimated_market_share']
            formatted += f"**Market Share Estimado**: {share['estimated_regional_market_share']:.2f}%\n"
            formatted += f"**Posição**: {share['market_position']}\n\n"
        
        formatted += "## 💡 INSIGHTS DE POSICIONAMENTO\n\n"
        
        if 'positioning_insights' in result:
            for insight in result['positioning_insights']:
                formatted += f"- {insight}\n"
        
        return formatted
    
    def _format_pricing_analysis(self, result: Dict[str, Any]) -> str:
        """Formatar análise de preços."""
        formatted = "## 💰 ANÁLISE COMPETITIVA DE PREÇOS\n\n"
        
        if 'pricing_analysis' in result:
            for category, data in result['pricing_analysis'].items():
                formatted += f"### {category.title()}\n"
                formatted += f"- **Preço Médio Empresa**: R$ {data['company_avg_price']:,.2f}\n"
                formatted += f"- **Preço Médio Mercado**: R$ {data['market_avg_price']:,.2f}\n"
                formatted += f"- **Premium/Desconto**: {data['price_premium_discount']:+.1f}%\n\n"
        
        formatted += "## 🚀 OPORTUNIDADES DE PRICING\n\n"
        
        if 'pricing_opportunities' in result:
            for opportunity in result['pricing_opportunities']:
                formatted += f"- {opportunity}\n"
        
        return formatted
    
    def _format_trend_comparison(self, result: Dict[str, Any]) -> str:
        """Formatar comparação de tendências."""
        formatted = "## 📈 COMPARAÇÃO DE TENDÊNCIAS\n\n"
        
        if 'trend_comparison' in result:
            trend = result['trend_comparison']['company_vs_market']
            formatted += f"**Crescimento Empresa**: {trend['company_revenue_growth']:+.2f}%\n"
            formatted += f"**Crescimento Mercado**: {trend['market_growth_benchmark']:+.2f}%\n"
            formatted += f"**Performance**: {trend['performance_vs_market']}\n\n"
        
        if 'seasonality_comparison' in result:
            season = result['seasonality_comparison']
            formatted += f"**Alinhamento Sazonal**: {'Sim' if season['alignment_with_market'] else 'Não'}\n"
            formatted += f"**Variação Sazonal**: {season['company_seasonal_variation']:.1%}\n\n"
        
        formatted += "## 💡 INSIGHTS DE TENDÊNCIAS\n\n"
        
        if 'trend_insights' in result:
            for insight in result['trend_insights']:
                formatted += f"- {insight}\n"
        
        return formatted
    
    def _format_market_share(self, result: Dict[str, Any]) -> str:
        """Formatar estimativa de market share."""
        formatted = "## 📊 ESTIMATIVA DE MARKET SHARE\n\n"
        
        if 'market_share_estimation' in result:
            share = result['market_share_estimation']
            formatted += f"**Receita da Empresa**: R$ {share['company_revenue']:,.2f}\n"
            formatted += f"**Market Share Regional**: {share['estimated_regional_market_share']:.2f}%\n"
            formatted += f"**Posição no Mercado**: {share['market_position']}\n\n"
        
        if 'segment_analysis' in result:
            formatted += "**Análise por Segmento**:\n"
            for segment, data in list(result['segment_analysis'].items())[:3]:
                formatted += f"- {segment}: {data['estimated_segment_market_share']:.1f}% do segmento\n"
        
        formatted += "\n## 🚀 POTENCIAL DE CRESCIMENTO\n\n"
        
        if 'growth_potential' in result:
            growth = result['growth_potential']
            formatted += f"**Posição Atual**: {growth['current_position']}\n"
            formatted += f"**Oportunidade de Crescimento**: {growth['growth_opportunity']:.1f}%\n"
        
        return formatted
    
    def _format_competitive_gaps(self, result: Dict[str, Any]) -> str:
        """Formatar gaps competitivos."""
        formatted = "## 🔍 GAPS COMPETITIVOS IDENTIFICADOS\n\n"
        
        if 'opportunity_matrix' in result:
            matrix = result['opportunity_matrix']
            formatted += f"**Oportunidades de Alta Prioridade**: {len(matrix['high_priority'])}\n"
            formatted += f"**Oportunidades de Média Prioridade**: {len(matrix['medium_priority'])}\n\n"
        
        formatted += "## 🎯 RECOMENDAÇÕES ESTRATÉGICAS\n\n"
        
        if 'strategic_recommendations' in result:
            for i, rec in enumerate(result['strategic_recommendations'], 1):
                formatted += f"{i}. {rec}\n"
        
        if 'category_gaps' in result:
            formatted += "\n## 📦 GAPS POR CATEGORIA\n\n"
            for category, gap in list(result['category_gaps'].items())[:3]:
                if gap['opportunity_size'] in ['High', 'Medium']:
                    formatted += f"**{category}**: Gap de {gap['gap_percentage']:+.1f}% vs. mercado\n"
        
        return formatted
