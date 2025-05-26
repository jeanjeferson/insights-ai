from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

class CompetitiveIntelligenceInput(BaseModel):
    """Schema otimizado para análise de inteligência competitiva com validações robustas."""
    
    analysis_type: str = Field(
        ..., 
        description="Tipo de análise competitiva: 'market_position' (posicionamento), 'pricing_analysis' (preços), 'trend_comparison' (tendências), 'market_share_estimation' (market share), 'competitive_gaps' (gaps e oportunidades)",
        json_schema_extra={
            "pattern": "^(market_position|pricing_analysis|trend_comparison|market_share_estimation|competitive_gaps)$"
        }
    )
    
    data_csv: str = Field(
        default="data/vendas.csv", 
        description="Caminho para arquivo CSV com dados de vendas atualizados (extraído via SQL Query Tool)"
    )
    
    market_segment: str = Field(
        default="joalherias", 
        description="Segmento de mercado para benchmarks (qualquer valor aceito, fallback para 'joalherias')"
    )
    
    benchmark_period: str = Field(
        default="quarterly", 
        description="Período de benchmark para análises temporais (aceita qualquer valor, fallback para 'quarterly')"
    )
    
    include_recommendations: bool = Field(
        default=True, 
        description="Incluir recomendações estratégicas específicas baseadas na análise"
    )
    
    risk_tolerance: str = Field(
        default="medium", 
        description="Tolerância a risco para recomendações (aceita qualquer valor, fallback para 'medium')"
    )
    
    @field_validator('analysis_type')
    @classmethod
    def validate_analysis_type(cls, v):
        valid_types = ['market_position', 'pricing_analysis', 'trend_comparison', 
                      'market_share_estimation', 'competitive_gaps']
        if v not in valid_types:
            raise ValueError(f"analysis_type deve ser um de: {valid_types}")
        return v
    
    @field_validator('data_csv')
    @classmethod
    def validate_csv_path(cls, v):
        if not v.endswith('.csv'):
            raise ValueError("data_csv deve ser um arquivo CSV válido")
        return v
    
    @field_validator('market_segment')
    @classmethod
    def validate_market_segment(cls, v):
        # Accept any value, fallback handled in _run() method
        return v
    
    @field_validator('benchmark_period')
    @classmethod
    def validate_benchmark_period(cls, v):
        # Accept any value, fallback handled in _run() method  
        return v
    
    @field_validator('risk_tolerance')
    @classmethod
    def validate_risk_tolerance(cls, v):
        # Accept any value, fallback handled in _run() method
        return v

class CompetitiveIntelligenceTool(BaseTool):
    """
    🏆 FERRAMENTA DE INTELIGÊNCIA COMPETITIVA PARA JOALHERIAS
    
    QUANDO USAR CADA ANÁLISE:
    
    📊 MARKET_POSITION:
    - Use quando: Agente precisa entender posicionamento competitivo da empresa
    - Casos de uso: "Como estamos posicionados vs. mercado?", "Qual nossa categoria de preço?", "Somos competitivos?"
    - Entrega: Posicionamento (Economy/Mid/Premium/Luxury), comparação de ticket médio vs. mercado, 
      market share estimado, análise de crescimento vs. benchmarks, insights de posicionamento
    
    💰 PRICING_ANALYSIS:
    - Use quando: Agente analisa estratégias de preço vs. concorrência
    - Casos de uso: "Nossos preços estão competitivos?", "Onde temos premium/desconto?", "Oportunidades de pricing?"
    - Entrega: Análise por faixa de preço vs. benchmarks, identificação de premium/desconto por categoria,
      elasticidade de preço, oportunidades de ajuste, mix de preços otimizado
    
    📈 TREND_COMPARISON:
    - Use quando: Agente compara performance com tendências do mercado
    - Casos de uso: "Estamos crescendo acima do mercado?", "Nossa sazonalidade está alinhada?", "Tendências por categoria?"
    - Entrega: Crescimento empresa vs. mercado, análise sazonal comparativa, tendências por categoria,
      performance relativa, insights de alinhamento com mercado
    
    🎯 MARKET_SHARE_ESTIMATION:
    - Use quando: Agente precisa estimar participação de mercado
    - Casos de uso: "Qual nosso market share?", "Quanto podemos crescer?", "Nossa posição competitiva?"
    - Entrega: Market share estimado (nacional/regional), análise por segmento de preço,
      potencial de crescimento, comparação com principais concorrentes, recomendações de expansão
    
    🔍 COMPETITIVE_GAPS:
    - Use quando: Agente identifica oportunidades e gaps competitivos
    - Casos de uso: "Onde temos oportunidades?", "Quais gaps vs. mercado?", "Prioridades estratégicas?"
    - Entrega: Gaps por categoria vs. mercado, oportunidades operacionais, matriz de prioridades,
      recomendações estratégicas específicas, plano de ação priorizado
    
    INTEGRAÇÃO COM OUTRAS FERRAMENTAS:
    - 🗄️ Use SQL Query Tool ANTES para extrair dados atualizados do banco
    - 📊 Use Statistical Analysis Tool APÓS para análises estatísticas detalhadas
    - 📈 Use Business Intelligence Tool para dashboards executivos
    - 🎯 Use KPI Calculator Tool para métricas complementares
    - ⚠️ Use Risk Assessment Tool para avaliar riscos das recomendações
    
    REQUISITOS DE DADOS:
    - CSV de vendas atualizado (recomendado: extraído via SQL Query Tool)
    - Mínimo 30 registros para análises confiáveis
    - Dados de pelo menos 3 meses para análises temporais
    - Colunas obrigatórias: Data, Total_Liquido, Quantidade, informações de produto
    """
    
    name: str = "Competitive Intelligence Tool"
    description: str = (
        "Ferramenta especializada em inteligência competitiva para joalherias. "
        "Analisa posicionamento de mercado, estratégias de preço, tendências competitivas, "
        "estimativa de market share e identificação de gaps/oportunidades. "
        "Usa benchmarks do setor brasileiro e fornece recomendações estratégicas acionáveis."
    )
    args_schema: Type[BaseModel] = CompetitiveIntelligenceInput
    
    def _run(self, analysis_type: str, data_csv: str = "data/vendas.csv", 
             market_segment: str = "joalherias", benchmark_period: str = "quarterly",
             include_recommendations: bool = True, risk_tolerance: str = "medium") -> str:
        
        # Implement fallback logic for parameters
        valid_segments = ['joalherias', 'relogios', 'acessorios']
        if market_segment not in valid_segments:
            print(f"⚠️ Segmento '{market_segment}' inválido, usando fallback 'joalherias'")
            market_segment = "joalherias"
            
        valid_periods = ['monthly', 'quarterly', 'yearly']
        if benchmark_period not in valid_periods:
            print(f"⚠️ Período '{benchmark_period}' inválido, usando fallback 'quarterly'")
            benchmark_period = "quarterly"
            
        valid_tolerance = ['low', 'medium', 'high']
        if risk_tolerance not in valid_tolerance:
            print(f"⚠️ Tolerância '{risk_tolerance}' inválida, usando fallback 'medium'")
            risk_tolerance = "medium"
        
        print(f"🏆 Competitive Intelligence Tool executando:")
        print(f"   📊 Análise: {analysis_type}")
        print(f"   📁 Dados: {data_csv}")
        print(f"   🏪 Segmento: {market_segment}")
        print(f"   ⏱️ Período: {benchmark_period}")
        print(f"   🎯 Tolerância Risco: {risk_tolerance}")
        
        try:
            # Validar existência do arquivo
            if not os.path.exists(data_csv):
                return f"❌ Erro: Arquivo {data_csv} não encontrado. Use SQL Query Tool para extrair dados atualizados."
            
            # Carregar e validar dados
            df = pd.read_csv(data_csv, sep=';', encoding='utf-8')
            print(f"✅ Dados carregados: {len(df)} registros")
            
            # Preparar dados para análise competitiva
            df = self._prepare_competitive_data(df)
            
            if df is None:
                return "❌ Erro: Falha na preparação dos dados. Verifique formato do CSV."
            
            if len(df) < 30:
                return f"❌ Erro: Dados insuficientes para análise competitiva (mínimo 30 registros, encontrados {len(df)})"
            
            # Validar colunas obrigatórias
            required_columns = ['Data', 'Total_Liquido', 'Quantidade']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return f"❌ Erro: Colunas obrigatórias ausentes: {missing_columns}"
            
            print(f"✅ Dados preparados e validados: {len(df)} registros")
            
            # Carregar benchmarks do setor
            market_benchmarks = self._load_market_benchmarks(market_segment)
            print(f"✅ Benchmarks carregados para segmento: {market_segment}")
            
            # Dicionário de análises competitivas
            competitive_analyses = {
                'market_position': self._analyze_market_position,
                'pricing_analysis': self._analyze_competitive_pricing,
                'trend_comparison': self._compare_market_trends,
                'market_share_estimation': self._estimate_market_share,
                'competitive_gaps': self._identify_competitive_gaps
            }
            
            if analysis_type not in competitive_analyses:
                return f"❌ Análise '{analysis_type}' não suportada. Opções: {list(competitive_analyses.keys())}"
            
            print(f"🔍 Executando análise: {analysis_type}")
            result = competitive_analyses[analysis_type](
                df, market_benchmarks, benchmark_period, include_recommendations, risk_tolerance
            )
            
            return self._format_competitive_result(analysis_type, result, market_segment, 
                                                 benchmark_period, include_recommendations)
            
        except Exception as e:
            return f"❌ Erro na análise competitiva: {str(e)}\n\nDica: Verifique se o CSV foi gerado pelo SQL Query Tool com dados atualizados."
    
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
                                period: str, include_recommendations: bool, risk_tolerance: str) -> Dict[str, Any]:
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
            
            # Insights de posicionamento expandidos
            positioning_insights = []
            
            # Análise de posicionamento detalhada
            if positioning == 'Luxury':
                positioning_insights.append("Posicionamento luxury - empresa opera no segmento premium do mercado brasileiro")
                positioning_insights.append("Estratégia de diferenciação pela exclusividade e experiência do cliente")
                positioning_insights.append("Foco em qualidade, design e atendimento personalizado")
            elif positioning == 'Premium':
                positioning_insights.append("Posicionamento premium - oportunidade de expansão para segmento luxury")
                positioning_insights.append("Competitivo no mid-high market com potencial de crescimento")
                positioning_insights.append("Considerar estratégias de premium pricing e value proposition")
            elif positioning == 'Mid-Market':
                positioning_insights.append("Posicionamento mid-market - competindo no mainstream do setor")
                positioning_insights.append("Foco em eficiência operacional e value for money")
                positioning_insights.append("Oportunidade de migração seletiva para premium")
            else:
                positioning_insights.append("Posicionamento economy - estratégia de liderança em custo")
                positioning_insights.append("Competitividade via preços acessíveis e volume")
                positioning_insights.append("Considerar estratégias de up-selling e cross-selling")
            
            # Análise competitiva detalhada
            if ticket_vs_market > 20:
                positioning_insights.append("Ticket médio 20%+ acima do mercado - posicionamento diferenciado forte")
                positioning_insights.append("Estratégia de premium pricing efetiva")
            elif ticket_vs_market > 0:
                positioning_insights.append("Ticket médio acima da média do mercado - posicionamento competitivo")
            elif ticket_vs_market < -20:
                positioning_insights.append("Ticket médio 20%+ abaixo do mercado - oportunidade de up-sell significativa")
                positioning_insights.append("Considerar revisão de estratégia de preços")
            elif ticket_vs_market < 0:
                positioning_insights.append("Ticket médio abaixo da média - espaço para melhoria de preços")
            
            # Análise de crescimento vs mercado
            growth_vs_market = benchmark_comparison['growth_analysis']['growth_vs_market']
            if growth_vs_market == 'Above':
                positioning_insights.append("Crescimento acima do mercado - estratégias efetivas de expansão")
                positioning_insights.append("Capturando market share dos concorrentes")
            else:
                positioning_insights.append("Crescimento abaixo do mercado - necessita revisão estratégica")
                positioning_insights.append("Avaliar estratégias de aceleração de crescimento")
            
            # Análise de market share
            market_share_pct = estimated_market_share.get('estimated_local_share', 0)
            if market_share_pct > 5:
                positioning_insights.append("Market share regional significativo - posição competitiva forte")
            elif market_share_pct > 2:
                positioning_insights.append("Market share regional moderado - oportunidade de consolidação")
            else:
                positioning_insights.append("Market share regional baixo - potencial de crescimento significativo")
            
            # Recomendações estratégicas expandidas
            if include_recommendations:
                # Use the correct key for market share
                market_share_value = estimated_market_share.get('estimated_local_share', 
                                   estimated_market_share.get('estimated_regional_market_share', 0))
                base_recommendations = self._generate_expansion_recommendations(
                    market_share_value, category_analysis
                )
                
                # Adicionar recomendações específicas de posicionamento
                strategic_recommendations = []
                
                if positioning in ['Economy', 'Mid-Market'] and ticket_vs_market < -10:
                    strategic_recommendations.append("Implementar estratégia de premium pricing seletiva")
                    strategic_recommendations.append("Desenvolver linha de produtos de maior valor agregado")
                
                if growth_vs_market == 'Below':
                    strategic_recommendations.append("Acelerar estratégias de marketing e vendas")
                    strategic_recommendations.append("Avaliar canais de distribuição e expansão geográfica")
                
                if market_share_pct < 3:
                    strategic_recommendations.append("Focar em crescimento orgânico local antes de expansão")
                    strategic_recommendations.append("Fortalecer posicionamento de marca e diferenciação")
                
                # Recomendações baseadas em tolerância a risco
                if risk_tolerance == 'high':
                    strategic_recommendations.append("Considerar estratégias agressivas de expansão de mercado")
                    strategic_recommendations.append("Avaliar aquisições estratégicas ou parcerias")
                elif risk_tolerance == 'low':
                    strategic_recommendations.append("Manter estratégias conservadoras de crescimento orgânico")
                    strategic_recommendations.append("Focar em otimização de operações existentes")
                
                positioning_insights.extend(base_recommendations)
                positioning_insights.extend(strategic_recommendations)
            
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
                                   period: str, include_recommendations: bool, risk_tolerance: str) -> Dict[str, Any]:
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
            
            # Recomendações estratégicas
            if include_recommendations:
                recommendations = self._generate_pricing_recommendations(
                    price_mix_analysis, risk_tolerance
                )
                pricing_opportunities.extend(recommendations)
            
            return {
                'pricing_analysis': pricing_analysis,
                'price_elasticity': price_elasticity,
                'pricing_opportunities': pricing_opportunities,
                'price_mix_analysis': price_mix_analysis
            }
            
        except Exception as e:
            return {'error': f"Erro na análise de preços: {str(e)}"}
    
    def _compare_market_trends(self, df: pd.DataFrame, benchmarks: Dict[str, Any],
                             period: str, include_recommendations: bool, risk_tolerance: str) -> Dict[str, Any]:
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
            
            # Recomendações estratégicas
            if include_recommendations:
                recommendations = self._generate_trend_recommendations(
                    trend_comparison, seasonality_comparison, category_trends
                )
                trend_insights.extend(recommendations)
            
            return {
                'trend_comparison': trend_comparison,
                'seasonality_comparison': seasonality_comparison,
                'category_trends': category_trends,
                'trend_insights': trend_insights
            }
            
        except Exception as e:
            return {'error': f"Erro na comparação de tendências: {str(e)}"}
    
    def _estimate_market_share(self, df: pd.DataFrame, benchmarks: Dict[str, Any],
                             period: str, include_recommendations: bool, risk_tolerance: str) -> Dict[str, Any]:
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
            
            # Recomendações estratégicas
            if include_recommendations:
                recommendations = self._generate_market_share_recommendations(
                    growth_potential, segment_analysis, competitor_analysis
                )
                growth_potential['expansion_recommendations'].extend(recommendations)
            
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
                                 period: str, include_recommendations: bool, risk_tolerance: str) -> Dict[str, Any]:
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
                'status': 'Above' if items_gap > 0 else 'Below',
                'cross_sell_opportunity': 'High' if items_gap < -20 else 'Medium' if items_gap < 0 else 'Low'
            }
            
            # Gaps de preço por segmento
            pricing_gaps = self._identify_pricing_gaps(df, benchmarks)
            
            # Gaps digitais
            digital_gaps = self._analyze_digital_gaps(df, benchmarks)
            
            # Priorização de oportunidades
            opportunity_matrix = self._create_opportunity_matrix(category_gaps, operational_gaps, pricing_gaps)
            
            # Recomendações estratégicas
            if include_recommendations:
                recommendations = self._generate_gaps_recommendations(
                    category_gaps, operational_gaps, pricing_gaps, digital_gaps, opportunity_matrix
                )
                opportunity_matrix['strategic_recommendations'] = recommendations[:5]  # Top 5
            
            return {
                'category_gaps': category_gaps,
                'operational_gaps': operational_gaps,
                'pricing_gaps': pricing_gaps,
                'digital_gaps': digital_gaps,
                'opportunity_matrix': opportunity_matrix
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
        
        # Baseado na análise de segmentos (usando revenue_share ao invés de estimated_segment_market_share)
        if segment_analysis and len(segment_analysis) > 0:
            try:
                # Usar revenue_share que existe no category_analysis
                strongest_segment = max(segment_analysis.items(), 
                                      key=lambda x: x[1].get('revenue_share', 0))
                recommendations.append(f"Fortalecer posição em {strongest_segment[0]} - categoria com maior participação ({strongest_segment[1].get('revenue_share', 0):.1f}%)")
            except (ValueError, KeyError) as e:
                # Fallback caso haja problemas com os dados
                recommendations.append("Fortalecer posição nas categorias principais da empresa")
        
        return recommendations
    
    def _generate_trend_recommendations(self, trend_comparison: Dict[str, Any], 
                                      seasonality_comparison: Dict[str, Any], 
                                      category_trends: Dict[str, Any]) -> List[str]:
        """Gerar recomendações de tendências."""
        recommendations = []
        
        # Performance vs. mercado
        company_vs_market = trend_comparison['company_vs_market']
        if company_vs_market['performance_vs_market'] == 'Underperforming':
            recommendations.append("Revisar estratégias: crescimento abaixo do mercado")
        elif company_vs_market['performance_vs_market'] == 'Outperforming':
            recommendations.append("Manter estratégias atuais: crescimento acima do mercado")
        
        # Sazonalidade
        if not seasonality_comparison['alignment_with_market']:
            recommendations.append("Avaliar alinhamento sazonal com mercado")
        
        # Categorias em crescimento
        strong_categories = [cat for cat, data in category_trends.items() 
                           if data.get('trend_strength') == 'Strong']
        if strong_categories:
            recommendations.append(f"Focar em categorias crescentes: {', '.join(strong_categories[:2])}")
        
        return recommendations
    
    def _generate_pricing_recommendations(self, price_mix_analysis: Dict[str, Any], risk_tolerance: str) -> List[str]:
        """Gerar recomendações de pricing."""
        opportunities = []
        
        # Oportunidades baseadas no mix de preços
        current_distribution = price_mix_analysis['current_distribution']
        market_opportunity = price_mix_analysis['market_opportunity']
        
        if market_opportunity:
            opportunities.append(f"Expandir produtos premium/luxury - baixa participação atual")
        
        economy_share = current_distribution.get('Economy', 0)
        if economy_share > 0.4:
            opportunities.append("Reduzir dependência de produtos economy - migrar para mid/premium")
        
        # Recomendações estratégicas
        if risk_tolerance == 'high':
            opportunities.append("Avaliar oportunidades de pricing agressivas")
        
        return opportunities
    
    def _generate_market_share_recommendations(self, growth_potential: Dict[str, Any], segment_analysis: Dict[str, Any], 
                                              competitor_analysis: Dict[str, Any]) -> List[str]:
        """Gerar recomendações de market share."""
        recommendations = []
        
        if growth_potential['current_position'] == 'Niche Player':
            recommendations.append("Considerar expansão para mercados adjacentes")
        elif growth_potential['current_position'] == 'Regional Player':
            recommendations.append("Avaliar aquisições ou parcerias estratégicas")
        
        # Baseado na análise de segmentos
        if segment_analysis and len(segment_analysis) > 0:
            try:
                # Usar revenue_share ou total_revenue que existem no segment_analysis
                if any('estimated_segment_market_share' in data for data in segment_analysis.values()):
                    # Caso seja segment_analysis do market share estimation
                    strongest_segment = max(segment_analysis.items(), 
                                          key=lambda x: x[1].get('estimated_segment_market_share', 0))
                    recommendations.append(f"Fortalecer posição em {strongest_segment[0]} - segmento mais forte")
                else:
                    # Caso seja category_analysis do market position
                    strongest_segment = max(segment_analysis.items(), 
                                          key=lambda x: x[1].get('revenue_share', 0))
                    recommendations.append(f"Fortalecer posição em {strongest_segment[0]} - categoria com maior participação")
            except (ValueError, KeyError):
                recommendations.append("Fortalecer posição nos segmentos principais da empresa")
        
        # Competitor analysis
        if competitor_analysis['competitive_position'] != 'Independent':
            recommendations.append(f"Avaliar concorrência: {competitor_analysis['competitive_position']}")
        
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
    
    def _generate_gaps_recommendations(self, category_gaps: Dict, operational_gaps: Dict, 
                                      pricing_gaps: Dict, digital_gaps: Dict, opportunity_matrix: Dict) -> List[str]:
        """Gerar recomendações abrangentes de gaps competitivos."""
        recommendations = []
        
        # Análise detalhada de gaps por categoria
        top_category_gaps = sorted(category_gaps.items(), 
                                 key=lambda x: abs(x[1]['gap_percentage']), reverse=True)[:4]
        
        for categoria, gap_data in top_category_gaps:
            gap_pct = gap_data['gap_percentage']
            opportunity_size = gap_data['opportunity_size']
            
            if gap_pct > 5:  # Gap positivo = sub-representação
                if opportunity_size == 'High':
                    recommendations.append(f"PRIORIDADE ALTA: Expandir portfolio em {categoria} - gap de {gap_pct:.1f}% vs. mercado representa oportunidade significativa")
                    recommendations.append(f"Desenvolver estratégia de entrada em {categoria} com foco em diferenciação competitiva")
                else:
                    recommendations.append(f"Considerar expansão seletiva em {categoria} - gap de {gap_pct:.1f}% vs. mercado")
            elif gap_pct < -3:  # Gap negativo = sobre-representação
                recommendations.append(f"Avaliar concentração em {categoria} - participação {abs(gap_pct):.1f}% acima do mercado")
        
        # Recomendações operacionais detalhadas
        inventory_gap = operational_gaps.get('inventory_turnover', {})
        if inventory_gap.get('gap_percentage', 0) < -20:
            recommendations.append("CRÍTICO: Melhorar giro de estoque - 20%+ abaixo do mercado impacta cashflow")
            recommendations.append("Implementar gestão de estoque just-in-time e análise ABC de produtos")
        elif inventory_gap.get('gap_percentage', 0) < -10:
            recommendations.append("Otimizar giro de estoque - oportunidade de melhoria operacional")
        
        items_gap = operational_gaps.get('items_per_transaction', {})
        cross_sell_opp = items_gap.get('cross_sell_opportunity', 'Low')
        if cross_sell_opp == 'High':
            recommendations.append("ALTA OPORTUNIDADE: Implementar estratégias de cross-sell e up-sell")
            recommendations.append("Desenvolver bundles de produtos e treinamento de vendas consultiva")
        elif cross_sell_opp == 'Medium':
            recommendations.append("Explorar oportunidades de cross-sell - potencial de aumento do ticket médio")
        
        # Recomendações de pricing gaps
        for category, pricing_data in pricing_gaps.items():
            opportunity = pricing_data.get('opportunity', 'Maintain')
            gap_pct = pricing_data.get('gap_percentage', 0)
            
            if opportunity == 'Increase' and abs(gap_pct) > 15:
                recommendations.append(f"PRICING: Ajustar preços em {category} - {abs(gap_pct):.1f}% abaixo do mercado")
                recommendations.append(f"Implementar estratégia de premium pricing gradual em {category}")
            elif opportunity == 'Reassess' and gap_pct > 25:
                recommendations.append(f"RISCO: Reavaliar preços em {category} - {gap_pct:.1f}% acima do mercado pode afetar competitividade")
        
        # Oportunidades digitais
        digital_opp = digital_gaps.get('digital_opportunity', 'Low')
        if digital_opp == 'High':
            recommendations.append("TRANSFORMAÇÃO DIGITAL: Desenvolver canal online - alta penetração digital no mercado")
            recommendations.append("Investir em e-commerce, marketing digital e experiência omnichannel")
        elif digital_opp == 'Medium':
            recommendations.append("Avaliar presença digital - oportunidade de crescimento incremental")
        
        # Matriz de oportunidades prioritárias
        high_priority_opps = opportunity_matrix.get('high_priority', [])
        for opp in high_priority_opps[:3]:  # Top 3 high priority
            opp_name = opp.get('opportunity', 'Unknown')
            impact = opp.get('impact', 'Medium')
            effort = opp.get('effort', 'Medium')
            recommendations.append(f"MATRIZ ESTRATÉGICA: {opp_name} (Impacto: {impact}, Esforço: {effort})")
        
        # Recomendações estratégicas macro
        total_opportunities = opportunity_matrix.get('total_opportunities', 0)
        if total_opportunities > 5:
            recommendations.append("ESTRATÉGIA: Priorizar 3-4 iniciativas principais para execução efetiva")
            recommendations.append("Desenvolver roadmap de implementação com milestones trimestrais")
        
        # Recomendações de monitoramento
        recommendations.append("MONITORAMENTO: Implementar KPIs competitivos para acompanhar progresso vs. benchmarks")
        recommendations.append("Revisar análise competitiva trimestralmente para ajustes estratégicos")
        
        return recommendations[:12]  # Limite de 12 recomendações para foco
    
    def _format_competitive_result(self, analysis_type: str, result: Dict[str, Any], 
                                 market_segment: str, benchmark_period: str, include_recommendations: bool) -> str:
        """Formatar resultado da análise competitiva."""
        try:
            timestamp = datetime.now().strftime('%d/%m/%Y %H:%M')
            
            if 'error' in result:
                return f"❌ Erro na análise competitiva {analysis_type}: {result['error']}"
            
            # Header padronizado
            analysis_name = analysis_type.upper().replace('_', ' ')
            formatted = f"""
# 🏆 INTELIGÊNCIA COMPETITIVA - {analysis_name}

**📊 Análise**: {analysis_name}  
**🏪 Segmento**: {market_segment.title()}  
**⏱️ Período**: {benchmark_period.title()}  
**📅 Gerado em**: {timestamp}

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
            
            # Seção de recomendações se incluída
            if include_recommendations:
                formatted += self._format_recommendations_section(result, analysis_type)
            
            # Footer padronizado
            formatted += f"""
---

## 📋 METADADOS DA ANÁLISE

**🔍 Metodologia**: Benchmarks setoriais brasileiros + análise comparativa  
**📊 Fonte de Dados**: CSV de vendas (extraído via SQL Query Tool)  
**⚠️ Disclaimers**: Market share e análises competitivas são estimativas baseadas em benchmarks públicos  
**🔄 Atualização**: Recomenda-se atualizar dados mensalmente via SQL Query Tool  

**🤖 Próximos Passos Sugeridos**:
- Use Statistical Analysis Tool para análises estatísticas detalhadas
- Use Business Intelligence Tool para dashboards executivos  
- Use Risk Assessment Tool para avaliar riscos das recomendações

*Relatório gerado por Competitive Intelligence Tool - Insights AI v2.0*
"""
            
            return formatted
            
        except Exception as e:
                         return f"❌ Erro na formatação da análise competitiva: {str(e)}"
    
    def _format_recommendations_section(self, result: Dict[str, Any], analysis_type: str) -> str:
        """Formatar seção de recomendações."""
        formatted = "\n## 🎯 RECOMENDAÇÕES ESTRATÉGICAS\n\n"
        
        # Buscar recomendações em diferentes campos dependendo do tipo de análise
        recommendations = []
        
        if analysis_type == 'market_position' and 'positioning_insights' in result:
            recommendations = result['positioning_insights']
        elif analysis_type == 'pricing_analysis' and 'pricing_opportunities' in result:
            recommendations = result['pricing_opportunities']
        elif analysis_type == 'trend_comparison' and 'trend_insights' in result:
            recommendations = result['trend_insights']
        elif analysis_type == 'market_share_estimation' and 'growth_potential' in result:
            recommendations = result['growth_potential'].get('expansion_recommendations', [])
        elif analysis_type == 'competitive_gaps' and 'opportunity_matrix' in result:
            recommendations = result['opportunity_matrix'].get('strategic_recommendations', [])
        
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):  # Top 5 recomendações
                formatted += f"**{i}.** {rec}\n\n"
        else:
            formatted += "Nenhuma recomendação específica gerada para esta análise.\n\n"
        
        return formatted
    
    def _format_market_position(self, result: Dict[str, Any]) -> str:
        """Formatar análise de posicionamento."""
        formatted = "## 🎯 POSICIONAMENTO COMPETITIVO\n\n"
        
        # Métricas principais
        if 'benchmark_comparison' in result:
            ticket = result['benchmark_comparison']['ticket_analysis']
            growth = result['benchmark_comparison'].get('growth_analysis', {})
            
            formatted += "### 📊 Métricas vs. Mercado\n\n"
            formatted += f"**🏷️ Categoria de Preço**: {ticket['positioning']}\n"
            formatted += f"**💰 Ticket Médio**: R$ {ticket['company_avg_ticket']:,.2f}\n"
            formatted += f"**📈 Diferença vs. Mercado**: {ticket['difference_percent']:+.1f}%\n"
            
            if growth:
                company_growth = growth.get('company_growth', {})
                if isinstance(company_growth, dict) and 'mom_avg' in company_growth:
                    formatted += f"**📊 Crescimento Médio**: {company_growth['mom_avg']:+.1f}% mensal\n"
            formatted += "\n"
        
        # Market Share
        if 'estimated_market_share' in result:
            share = result['estimated_market_share']
            formatted += "### 🏆 Participação de Mercado\n\n"
            
            # Use a chave correta baseada na estrutura dos dados
            if 'estimated_local_share' in share:
                formatted += f"**📍 Market Share Local**: {share['estimated_local_share']:.2f}%\n"
            elif 'estimated_regional_market_share' in share:
                formatted += f"**📍 Market Share Regional**: {share['estimated_regional_market_share']:.2f}%\n"
            
            if 'market_position' in share:
                formatted += f"**🎯 Posição Competitiva**: {share['market_position']}\n"
            if 'confidence_level' in share:
                formatted += f"**💼 Nível de Confiança**: {share['confidence_level']}\n"
            formatted += "\n"
        
        # Análise por categoria
        if 'category_analysis' in result:
            formatted += "### 📦 Performance por Categoria\n\n"
            for categoria, data in list(result['category_analysis'].items())[:3]:
                formatted += f"**{categoria}**: R$ {data['total_revenue']:,.0f} ({data['revenue_share']:.1f}% do total)\n"
            formatted += "\n"
        
        return formatted
    
    def _format_pricing_analysis(self, result: Dict[str, Any]) -> str:
        """Formatar análise de preços."""
        formatted = "## 💰 ANÁLISE COMPETITIVA DE PREÇOS\n\n"
        
        # Análise por categoria de preço
        if 'pricing_analysis' in result:
            formatted += "### 📊 Comparação por Categoria\n\n"
            for category, data in result['pricing_analysis'].items():
                status_emoji = "🟢" if data['price_premium_discount'] > 0 else "🔴" if data['price_premium_discount'] < -10 else "🟡"
                formatted += f"**{status_emoji} {category.title()}**\n"
                formatted += f"- Preço Empresa: R$ {data['company_avg_price']:,.2f}\n"
                formatted += f"- Benchmark Mercado: R$ {data['market_avg_price']:,.2f}\n"
                formatted += f"- Diferencial: {data['price_premium_discount']:+.1f}%\n"
                formatted += f"- Participação: {data['company_percentage']:.1f}% das vendas\n\n"
        
        # Mix de preços
        if 'price_mix_analysis' in result:
            mix_data = result['price_mix_analysis']
            formatted += "### 🎯 Mix de Preços\n\n"
            
            revenue_conc = mix_data.get('revenue_concentration', {})
            if revenue_conc:
                formatted += f"**📈 Categoria Dominante**: {revenue_conc['dominant_category']}\n"
                formatted += f"**📊 Concentração de Receita**: {revenue_conc['revenue_percentage']:.1f}%\n\n"
        
        # Elasticidade de preços
        if 'price_elasticity' in result:
            elasticity = result['price_elasticity']
            formatted += "### 📈 Elasticidade de Preços\n\n"
            formatted += f"**🔄 Elasticidade**: {elasticity['elasticity']:.2f}\n"
            formatted += f"**📊 Confiança**: {elasticity['confidence']}\n\n"
        
        return formatted
    
    def _format_trend_comparison(self, result: Dict[str, Any]) -> str:
        """Formatar comparação de tendências."""
        formatted = "## 📈 COMPARAÇÃO DE TENDÊNCIAS\n\n"
        
        # Performance vs. mercado
        if 'trend_comparison' in result:
            trend = result['trend_comparison']['company_vs_market']
            performance_emoji = "🟢" if trend['performance_vs_market'] == 'Outperforming' else "🔴"
            
            formatted += "### 📊 Performance vs. Mercado\n\n"
            formatted += f"**{performance_emoji} Status**: {trend['performance_vs_market']}\n"
            formatted += f"**📈 Crescimento Empresa**: {trend['company_revenue_growth']:+.2f}%\n"
            formatted += f"**📊 Benchmark Mercado**: {trend['market_growth_benchmark']:+.2f}%\n"
            
            # Calcular gap de performance
            gap = trend['company_revenue_growth'] - trend['market_growth_benchmark']
            formatted += f"**⚖️ Gap de Performance**: {gap:+.2f} pontos percentuais\n\n"
        
        # Análise sazonal
        if 'seasonality_comparison' in result:
            season = result['seasonality_comparison']
            alignment_emoji = "🟢" if season['alignment_with_market'] else "🟡"
            
            formatted += "### 🌊 Padrões Sazonais\n\n"
            formatted += f"**{alignment_emoji} Alinhamento com Mercado**: {'Sim' if season['alignment_with_market'] else 'Não'}\n"
            formatted += f"**📊 Variação Sazonal Empresa**: {season['company_seasonal_variation']:.1%}\n"
            formatted += f"**📈 Variação Sazonal Mercado**: {season['market_seasonal_variation']:.1%}\n"
            
            if 'company_peak_months' in season:
                months_names = {1:'Jan', 2:'Fev', 3:'Mar', 4:'Abr', 5:'Mai', 6:'Jun',
                              7:'Jul', 8:'Ago', 9:'Set', 10:'Out', 11:'Nov', 12:'Dez'}
                company_peaks = [months_names.get(m, str(m)) for m in season['company_peak_months']]
                market_peaks = [months_names.get(m, str(m)) for m in season['market_peak_months']]
                formatted += f"**🎯 Picos da Empresa**: {', '.join(company_peaks)}\n"
                formatted += f"**🏪 Picos do Mercado**: {', '.join(market_peaks)}\n\n"
        
        # Tendências por categoria
        if 'category_trends' in result and result['category_trends']:
            formatted += "### 📦 Tendências por Categoria\n\n"
            for categoria, data in list(result['category_trends'].items())[:3]:
                trend_emoji = "🟢" if data['trend_strength'] == 'Strong' else "🔴"
                formatted += f"**{trend_emoji} {categoria}**: {data['growth_rate']:+.1f}% (Benchmark: {data['market_share_benchmark']:.1f}%)\n"
            formatted += "\n"
        
        return formatted
    
    def _format_market_share(self, result: Dict[str, Any]) -> str:
        """Formatar estimativa de market share."""
        formatted = "## 📊 ESTIMATIVA DE MARKET SHARE\n\n"
        
        # Métricas principais
        if 'market_share_estimation' in result:
            share = result['market_share_estimation']
            position_emoji = "🏆" if share['market_position'] == 'Market Leader' else "🎯" if share['market_position'] == 'Regional Player' else "🌱"
            
            formatted += "### 📈 Posição Competitiva\n\n"
            formatted += f"**💰 Receita da Empresa**: R$ {share['company_revenue']:,.2f}\n"
            formatted += f"**📍 Market Share Regional**: {share['estimated_regional_market_share']:.2f}%\n"
            formatted += f"**{position_emoji} Posição no Mercado**: {share['market_position']}\n"
            
            if 'estimated_national_market_share' in share:
                formatted += f"**🇧🇷 Market Share Nacional**: {share['estimated_national_market_share']:.4f}%\n"
            formatted += "\n"
        
        # Análise por segmento
        if 'segment_analysis' in result and result['segment_analysis']:
            formatted += "### 🎯 Performance por Segmento\n\n"
            for segment, data in list(result['segment_analysis'].items())[:4]:
                share_pct = data['estimated_segment_market_share']
                segment_emoji = "🟢" if share_pct > 5 else "🟡" if share_pct > 2 else "🔴"
                formatted += f"**{segment_emoji} {segment}**\n"
                formatted += f"- Receita: R$ {data['revenue']:,.0f}\n"
                formatted += f"- Participação na Empresa: {data['percentage_of_company']:.1f}%\n"
                formatted += f"- Market Share do Segmento: {share_pct:.1f}%\n\n"
        
        # Potencial de crescimento
        if 'growth_potential' in result:
            growth = result['growth_potential']
            formatted += "### 🚀 Potencial de Crescimento\n\n"
            formatted += f"**📊 Posição Atual**: {growth['current_position']}\n"
            formatted += f"**📈 Oportunidade de Crescimento**: {growth['growth_opportunity']:.1f} pontos percentuais\n"
            
            if 'expansion_recommendations' in growth and growth['expansion_recommendations']:
                formatted += f"**💡 Próximos Passos**: {growth['expansion_recommendations'][0]}\n"
            formatted += "\n"
        
        # Análise competitiva
        if 'competitor_analysis' in result:
            comp = result['competitor_analysis']
            formatted += "### 🏪 Cenário Competitivo\n\n"
            formatted += f"**🎯 Posição Competitiva**: {comp['competitive_position']}\n"
            
            if 'market_leaders' in comp:
                formatted += f"**👑 Líderes do Mercado**: {', '.join(comp['market_leaders'][:3])}\n"
            
            if 'estimated_gap_to_leader' in comp:
                formatted += f"**📊 Gap vs. Líder**: R$ {comp['estimated_gap_to_leader']:.1f}M\n"
            formatted += "\n"
        
        return formatted
    
    def _format_competitive_gaps(self, result: Dict[str, Any]) -> str:
        """Formatar gaps competitivos."""
        formatted = "## 🔍 GAPS COMPETITIVOS E OPORTUNIDADES\n\n"
        
        # Matrix de oportunidades
        if 'opportunity_matrix' in result:
            matrix = result['opportunity_matrix']
            total_opp = matrix.get('total_opportunities', 0)
            high_priority = len(matrix.get('high_priority', []))
            medium_priority = len(matrix.get('medium_priority', []))
            
            formatted += "### 🎯 Matriz de Oportunidades\n\n"
            formatted += f"**🔥 Alta Prioridade**: {high_priority} oportunidades\n"
            formatted += f"**⚡ Média Prioridade**: {medium_priority} oportunidades\n"
            formatted += f"**📊 Total Identificado**: {total_opp} oportunidades\n\n"
            
            # Listar oportunidades de alta prioridade
            if matrix.get('high_priority'):
                formatted += "#### 🔥 Oportunidades Prioritárias:\n"
                for opp in matrix['high_priority'][:3]:
                    formatted += f"- **{opp['opportunity']}** (Impacto: {opp['impact']}, Esforço: {opp['effort']})\n"
                formatted += "\n"
        
        # Gaps por categoria
        if 'category_gaps' in result and result['category_gaps']:
            formatted += "### 📦 Gaps por Categoria vs. Mercado\n\n"
            for category, gap in list(result['category_gaps'].items())[:4]:
                gap_pct = gap['gap_percentage']
                opportunity_size = gap['opportunity_size']
                
                # Emoji baseado no tamanho da oportunidade
                opp_emoji = "🔥" if opportunity_size == 'High' else "⚡" if opportunity_size == 'Medium' else "💡"
                gap_emoji = "📈" if gap_pct > 0 else "📉"
                
                formatted += f"**{opp_emoji} {category}**\n"
                formatted += f"- Participação Empresa: {gap['company_share']:.1f}%\n"
                formatted += f"- Benchmark Mercado: {gap['market_share_benchmark']:.1f}%\n"
                formatted += f"- {gap_emoji} Gap: {gap_pct:+.1f} pontos percentuais\n"
                formatted += f"- Oportunidade: {opportunity_size}\n\n"
        
        # Gaps operacionais
        if 'operational_gaps' in result:
            formatted += "### ⚙️ Gaps Operacionais\n\n"
            
            op_gaps = result['operational_gaps']
            for metric, data in op_gaps.items():
                # Verificar se 'status' existe, caso contrário usar valor padrão baseado em gap_percentage
                status = data.get('status', 'Above' if data.get('gap_percentage', 0) > 0 else 'Below')
                status_emoji = "🟢" if status == 'Above' else "🔴"
                gap_pct = abs(data.get('gap_percentage', 0))
                
                if gap_pct > 15:  # Mostrar apenas gaps significativos
                    metric_name = metric.replace('_', ' ').title()
                    formatted += f"**{status_emoji} {metric_name}**\n"
                    
                    # Usar chaves diferentes para diferentes tipos de dados
                    if 'company_estimated' in data:
                        formatted += f"- Empresa: {data['company_estimated']:.2f}\n"
                    elif 'company_avg' in data:
                        formatted += f"- Empresa: {data['company_avg']:.2f}\n"
                    
                    formatted += f"- Benchmark: {data['market_benchmark']:.2f}\n"
                    formatted += f"- Gap: {data['gap_percentage']:+.1f}%\n\n"
        
        # Gaps de pricing
        if 'pricing_gaps' in result and result['pricing_gaps']:
            formatted += "### 💰 Gaps de Pricing\n\n"
            for category, data in list(result['pricing_gaps'].items())[:3]:
                opportunity = data['opportunity']
                gap_pct = data['gap_percentage']
                
                opp_emoji = "🔥" if opportunity == 'Increase' else "⚠️" if opportunity == 'Reassess' else "✅"
                formatted += f"**{opp_emoji} {category.title()}**: {gap_pct:+.1f}% vs. mercado ({opportunity})\n"
            formatted += "\n"
        
        # Digital gaps
        if 'digital_gaps' in result:
            digital = result['digital_gaps']
            if digital.get('digital_opportunity') in ['High', 'Medium']:
                formatted += "### 💻 Gap Digital\n\n"
                formatted += f"**📊 Penetração Digital Mercado**: {digital['market_digital_penetration']:.1f}%\n"
                formatted += f"**🚀 Oportunidade Digital**: {digital['digital_opportunity']}\n\n"
        
        return formatted
