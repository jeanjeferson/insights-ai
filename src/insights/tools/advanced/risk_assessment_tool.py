from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any, List
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RiskAssessmentInput(BaseModel):
    """Schema de entrada para avaliação de riscos."""
    assessment_type: str = Field(..., description="Tipo: 'business_risk', 'financial_risk', 'operational_risk', 'market_risk', 'customer_risk', 'comprehensive_risk'")
    data_csv: str = Field(default="data/vendas.csv", description="Caminho para arquivo CSV")
    risk_tolerance: str = Field(default="medium", description="Tolerância: 'low', 'medium', 'high'")
    time_horizon: str = Field(default="6_months", description="Horizonte: '3_months', '6_months', '12_months'")
    include_mitigation: bool = Field(default=True, description="Incluir estratégias de mitigação")

class RiskAssessmentTool(BaseTool):
    name: str = "Risk Assessment Tool"
    description: str = """
    Ferramenta de avaliação de riscos para joalherias:
    
    TIPOS DE AVALIAÇÃO:
    - business_risk: Riscos gerais do negócio
    - financial_risk: Riscos financeiros e de liquidez
    - operational_risk: Riscos operacionais e de processo
    - market_risk: Riscos de mercado e competição
    - customer_risk: Riscos relacionados à base de clientes
    - comprehensive_risk: Avaliação completa de todos os riscos
    
    ANÁLISES INCLUÍDAS:
    - Identificação de riscos críticos
    - Matriz de probabilidade vs impacto
    - Scores de risco por categoria
    - Estratégias de mitigação
    - Monitoramento e alertas
    - Planos de contingência
    
    TOLERÂNCIA A RISCO:
    - Low: Conservadora
    - Medium: Moderada
    - High: Agressiva
    """
    args_schema: Type[BaseModel] = RiskAssessmentInput
    
    def _run(self, assessment_type: str, data_csv: str = "data/vendas.csv",
             risk_tolerance: str = "medium", time_horizon: str = "6_months",
             include_mitigation: bool = True) -> str:
        try:
            # Carregar e preparar dados
            df = pd.read_csv(data_csv, sep=';', encoding='utf-8')
            df = self._prepare_risk_data(df, time_horizon)
            
            if df is None or len(df) < 10:
                return "Erro: Dados insuficientes para avaliação de riscos (mínimo 10 registros)"
            
            # Dicionário de avaliações de risco
            risk_assessments = {
                'business_risk': self._assess_business_risk,
                'financial_risk': self._assess_financial_risk,
                'operational_risk': self._assess_operational_risk,
                'market_risk': self._assess_market_risk,
                'customer_risk': self._assess_customer_risk,
                'comprehensive_risk': self._assess_comprehensive_risk
            }
            
            if assessment_type not in risk_assessments:
                return f"Avaliação '{assessment_type}' não suportada. Opções: {list(risk_assessments.keys())}"
            
            result = risk_assessments[assessment_type](df, risk_tolerance, include_mitigation)
            return self._format_risk_result(assessment_type, result, risk_tolerance, time_horizon)
            
        except Exception as e:
            return f"Erro na avaliação de riscos: {str(e)}"
    
    def _prepare_risk_data(self, df: pd.DataFrame, time_horizon: str) -> Optional[pd.DataFrame]:
        """Preparar dados para avaliação de riscos."""
        try:
            # Converter data
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
            df = df.dropna(subset=['Data'])
            
            # Filtrar por horizonte temporal
            current_date = df['Data'].max()
            
            if time_horizon == '3_months':
                start_date = current_date - timedelta(days=90)
            elif time_horizon == '6_months':
                start_date = current_date - timedelta(days=180)
            elif time_horizon == '12_months':
                start_date = current_date - timedelta(days=365)
            else:
                start_date = current_date - timedelta(days=180)
            
            df = df[df['Data'] >= start_date]
            
            # Adicionar métricas para análise de risco
            df['Preco_Unitario'] = df['Total_Liquido'] / df['Quantidade'].replace(0, 1)
            df['Year_Month'] = df['Data'].dt.to_period('M')
            df['Week'] = df['Data'].dt.isocalendar().week
            
            # Simular customer_id se necessário
            if 'Customer_ID' not in df.columns:
                df = self._simulate_customer_ids(df)
            
            return df
            
        except Exception as e:
            print(f"Erro na preparação de dados de risco: {str(e)}")
            return None
    
    def _simulate_customer_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simular customer IDs."""
        df = df.copy()
        np.random.seed(42)
        df['Customer_ID'] = 'CUST_' + pd.Series(np.random.randint(1, len(df)//4 + 1, len(df))).astype(str)
        return df
    
    def _assess_business_risk(self, df: pd.DataFrame, risk_tolerance: str, 
                            include_mitigation: bool) -> Dict[str, Any]:
        """Avaliar riscos gerais do negócio."""
        try:
            business_risks = []
            
            # 1. Risco de Concentração de Receita
            customer_revenue = df.groupby('Customer_ID')['Total_Liquido'].sum().sort_values(ascending=False)
            top_10_customers = customer_revenue.head(10).sum()
            concentration_risk = top_10_customers / customer_revenue.sum() * 100
            
            if concentration_risk > 60:
                business_risks.append({
                    'risk': 'Alta Concentração de Clientes',
                    'probability': 'High',
                    'impact': 'High',
                    'score': 9,
                    'description': f'Top 10 clientes representam {concentration_risk:.1f}% da receita',
                    'mitigation': 'Diversificar base de clientes, programas de aquisição'
                })
            elif concentration_risk > 40:
                business_risks.append({
                    'risk': 'Concentração Moderada de Clientes',
                    'probability': 'Medium',
                    'impact': 'Medium',
                    'score': 6,
                    'description': f'Top 10 clientes representam {concentration_risk:.1f}% da receita',
                    'mitigation': 'Monitorar dependência, diversificar gradualmente'
                })
            
            # 2. Risco de Sazonalidade Extrema
            monthly_sales = df.groupby('Year_Month')['Total_Liquido'].sum()
            if len(monthly_sales) >= 3:
                seasonal_variation = (monthly_sales.max() - monthly_sales.min()) / monthly_sales.mean()
                
                if seasonal_variation > 0.8:
                    business_risks.append({
                        'risk': 'Alta Variação Sazonal',
                        'probability': 'High',
                        'impact': 'Medium',
                        'score': 6,
                        'description': f'Variação sazonal de {seasonal_variation:.1%}',
                        'mitigation': 'Produtos menos sazonais, planejamento de fluxo de caixa'
                    })
            
            # 3. Risco de Produtos em Declínio
            product_performance = df.groupby('Codigo_Produto').agg({
                'Total_Liquido': 'sum',
                'Data': 'max'
            })
            
            old_products = product_performance[
                (df['Data'].max() - product_performance['Data']).dt.days > 90
            ]
            
            if len(old_products) > len(product_performance) * 0.3:
                business_risks.append({
                    'risk': 'Portfolio de Produtos Envelhecido',
                    'probability': 'Medium',
                    'impact': 'Medium',
                    'score': 4,
                    'description': f'{len(old_products)} produtos sem venda há 90+ dias',
                    'mitigation': 'Renovação de portfolio, liquidação de estoque antigo'
                })
            
            # 4. Risco de Crescimento Negativo
            if len(monthly_sales) >= 2:
                recent_growth = (monthly_sales.iloc[-1] - monthly_sales.iloc[-2]) / monthly_sales.iloc[-2] * 100
                
                if recent_growth < -15:
                    business_risks.append({
                        'risk': 'Declínio Acelerado de Vendas',
                        'probability': 'High',
                        'impact': 'High',
                        'score': 9,
                        'description': f'Queda de {recent_growth:.1f}% no último período',
                        'mitigation': 'Análise de causas, ações corretivas imediatas'
                    })
                elif recent_growth < -5:
                    business_risks.append({
                        'risk': 'Declínio Moderado de Vendas',
                        'probability': 'Medium',
                        'impact': 'Medium',
                        'score': 6,
                        'description': f'Queda de {recent_growth:.1f}% no último período',
                        'mitigation': 'Monitoramento próximo, ajustes de estratégia'
                    })
            
            # 5. Risco de Baixa Diversificação
            if 'Grupo_Produto' in df.columns:
                category_distribution = df.groupby('Grupo_Produto')['Total_Liquido'].sum()
                top_category_share = category_distribution.max() / category_distribution.sum() * 100
                
                if top_category_share > 70:
                    business_risks.append({
                        'risk': 'Baixa Diversificação de Categoria',
                        'probability': 'Medium',
                        'impact': 'Medium',
                        'score': 5,
                        'description': f'Uma categoria representa {top_category_share:.1f}% das vendas',
                        'mitigation': 'Desenvolver outras categorias, balancear portfolio'
                    })
            
            # Score geral de risco do negócio
            if business_risks:
                avg_score = np.mean([risk['score'] for risk in business_risks])
                max_score = max([risk['score'] for risk in business_risks])
            else:
                avg_score = 2
                max_score = 2
            
            return {
                'business_risks': business_risks,
                'risk_summary': {
                    'total_risks_identified': len(business_risks),
                    'avg_risk_score': round(avg_score, 1),
                    'max_risk_score': max_score,
                    'overall_risk_level': self._classify_risk_level(avg_score)
                },
                'risk_tolerance_assessment': self._assess_risk_tolerance_fit(business_risks, risk_tolerance)
            }
            
        except Exception as e:
            return {'error': f"Erro na avaliação de risco do negócio: {str(e)}"}
    
    def _assess_financial_risk(self, df: pd.DataFrame, risk_tolerance: str,
                             include_mitigation: bool) -> Dict[str, Any]:
        """Avaliar riscos financeiros."""
        try:
            financial_risks = []
            
            # 1. Risco de Fluxo de Caixa
            monthly_revenue = df.groupby('Year_Month')['Total_Liquido'].sum()
            if len(monthly_revenue) >= 3:
                revenue_volatility = monthly_revenue.std() / monthly_revenue.mean()
                
                if revenue_volatility > 0.4:
                    financial_risks.append({
                        'risk': 'Alta Volatilidade de Receita',
                        'probability': 'High',
                        'impact': 'High',
                        'score': 8,
                        'description': f'Coeficiente de variação: {revenue_volatility:.2f}',
                        'mitigation': 'Reserva de emergência, contratos recorrentes'
                    })
                elif revenue_volatility > 0.2:
                    financial_risks.append({
                        'risk': 'Volatilidade Moderada de Receita',
                        'probability': 'Medium',
                        'impact': 'Medium',
                        'score': 5,
                        'description': f'Coeficiente de variação: {revenue_volatility:.2f}',
                        'mitigation': 'Monitoramento de fluxo de caixa, planejamento'
                    })
            
            # 2. Risco de Inadimplência (estimado)
            total_revenue = df['Total_Liquido'].sum()
            estimated_receivables = total_revenue * 0.3  # Assumindo 30% a receber
            
            if estimated_receivables > total_revenue * 0.4:
                financial_risks.append({
                    'risk': 'Alto Risco de Inadimplência',
                    'probability': 'Medium',
                    'impact': 'High',
                    'score': 7,
                    'description': 'Contas a receber representam >40% da receita',
                    'mitigation': 'Política de crédito rigorosa, seguro de crédito'
                })
            
            # 3. Risco de Margem
            avg_ticket = df['Total_Liquido'].mean()
            if avg_ticket < 500:  # Baixo ticket médio
                financial_risks.append({
                    'risk': 'Pressão de Margem por Baixo Ticket',
                    'probability': 'High',
                    'impact': 'Medium',
                    'score': 6,
                    'description': f'Ticket médio de R$ {avg_ticket:.2f}',
                    'mitigation': 'Estratégias de up-sell, produtos premium'
                })
            
            # 4. Risco de Concentração Temporal
            daily_sales = df.groupby(df['Data'].dt.date)['Total_Liquido'].sum()
            if len(daily_sales) >= 7:
                # Verificar concentração em poucos dias
                top_10_days = daily_sales.nlargest(10).sum()
                concentration_temporal = top_10_days / daily_sales.sum() * 100
                
                if concentration_temporal > 50:
                    financial_risks.append({
                        'risk': 'Concentração Temporal de Vendas',
                        'probability': 'Medium',
                        'impact': 'Medium',
                        'score': 5,
                        'description': f'Top 10 dias representam {concentration_temporal:.1f}% das vendas',
                        'mitigation': 'Distribuir vendas ao longo do tempo, campanhas regulares'
                    })
            
            # 5. Risco de Capital de Giro
            # Estimativa baseada em padrões do setor
            monthly_avg_revenue = monthly_revenue.mean() if len(monthly_revenue) > 0 else 0
            estimated_working_capital_need = monthly_avg_revenue * 2  # 2 meses
            
            if monthly_avg_revenue > 0:
                working_capital_ratio = estimated_working_capital_need / monthly_avg_revenue
                
                if working_capital_ratio > 3:
                    financial_risks.append({
                        'risk': 'Alto Requisito de Capital de Giro',
                        'probability': 'Medium',
                        'impact': 'Medium',
                        'score': 6,
                        'description': 'Necessidade estimada de 3+ meses de receita em capital',
                        'mitigation': 'Linha de crédito, gestão de estoque eficiente'
                    })
            
            # Score financeiro
            if financial_risks:
                avg_score = np.mean([risk['score'] for risk in financial_risks])
                max_score = max([risk['score'] for risk in financial_risks])
            else:
                avg_score = 3
                max_score = 3
            
            # Análise de liquidez estimada
            liquidity_analysis = self._estimate_liquidity_risk(df, monthly_revenue)
            
            return {
                'financial_risks': financial_risks,
                'liquidity_analysis': liquidity_analysis,
                'risk_summary': {
                    'total_risks_identified': len(financial_risks),
                    'avg_risk_score': round(avg_score, 1),
                    'max_risk_score': max_score,
                    'overall_risk_level': self._classify_risk_level(avg_score)
                },
                'financial_health_indicators': self._calculate_financial_health_indicators(df)
            }
            
        except Exception as e:
            return {'error': f"Erro na avaliação de risco financeiro: {str(e)}"}
    
    def _assess_operational_risk(self, df: pd.DataFrame, risk_tolerance: str,
                               include_mitigation: bool) -> Dict[str, Any]:
        """Avaliar riscos operacionais."""
        try:
            operational_risks = []
            
            # 1. Risco de Dependência de Poucos Produtos
            product_revenue = df.groupby('Codigo_Produto')['Total_Liquido'].sum().sort_values(ascending=False)
            top_5_products_share = product_revenue.head(5).sum() / product_revenue.sum() * 100
            
            if top_5_products_share > 70:
                operational_risks.append({
                    'risk': 'Dependência Excessiva de Poucos Produtos',
                    'probability': 'High',
                    'impact': 'High',
                    'score': 8,
                    'description': f'Top 5 produtos representam {top_5_products_share:.1f}% das vendas',
                    'mitigation': 'Diversificar linha de produtos, desenvolver novos itens'
                })
            elif top_5_products_share > 50:
                operational_risks.append({
                    'risk': 'Concentração Moderada de Produtos',
                    'probability': 'Medium',
                    'impact': 'Medium',
                    'score': 5,
                    'description': f'Top 5 produtos representam {top_5_products_share:.1f}% das vendas',
                    'mitigation': 'Gradualmente diversificar portfolio'
                })
            
            # 2. Risco de Estoque Parado
            current_date = df['Data'].max()
            product_last_sale = df.groupby('Codigo_Produto')['Data'].max()
            old_stock = (current_date - product_last_sale).dt.days > 90
            old_stock_percentage = old_stock.sum() / len(product_last_sale) * 100
            
            if old_stock_percentage > 30:
                operational_risks.append({
                    'risk': 'Alto Percentual de Estoque Parado',
                    'probability': 'High',
                    'impact': 'Medium',
                    'score': 6,
                    'description': f'{old_stock_percentage:.1f}% dos produtos sem venda há 90+ dias',
                    'mitigation': 'Liquidação agressiva, melhoria na gestão de compras'
                })
            elif old_stock_percentage > 15:
                operational_risks.append({
                    'risk': 'Estoque Parado Moderado',
                    'probability': 'Medium',
                    'impact': 'Low',
                    'score': 3,
                    'description': f'{old_stock_percentage:.1f}% dos produtos sem venda há 90+ dias',
                    'mitigation': 'Promoções direcionadas, análise de demanda'
                })
            
            # 3. Risco de Falta de Padronização
            price_variation = df.groupby('Codigo_Produto')['Preco_Unitario'].std().fillna(0)
            high_variation_products = (price_variation > price_variation.mean() * 2).sum()
            
            if high_variation_products > len(price_variation) * 0.2:
                operational_risks.append({
                    'risk': 'Inconsistência de Preços',
                    'probability': 'Medium',
                    'impact': 'Low',
                    'score': 3,
                    'description': f'{high_variation_products} produtos com alta variação de preço',
                    'mitigation': 'Padronizar política de preços, treinamento de equipe'
                })
            
            # 4. Risco de Baixa Rotatividade
            total_products = df['Codigo_Produto'].nunique()
            products_sold_recently = df[df['Data'] >= current_date - timedelta(days=30)]['Codigo_Produto'].nunique()
            rotation_rate = products_sold_recently / total_products * 100
            
            if rotation_rate < 30:
                operational_risks.append({
                    'risk': 'Baixa Rotatividade de Produtos',
                    'probability': 'High',
                    'impact': 'Medium',
                    'score': 6,
                    'description': f'Apenas {rotation_rate:.1f}% dos produtos vendidos no último mês',
                    'mitigation': 'Otimizar mix de produtos, campanhas de ativação'
                })
            elif rotation_rate < 50:
                operational_risks.append({
                    'risk': 'Rotatividade Moderada de Produtos',
                    'probability': 'Medium',
                    'impact': 'Low',
                    'score': 4,
                    'description': f'{rotation_rate:.1f}% dos produtos vendidos no último mês',
                    'mitigation': 'Melhorar exposição de produtos, cross-selling'
                })
            
            # 5. Risco de Sazonalidade Operacional
            if 'Grupo_Produto' in df.columns:
                seasonal_dependency = self._analyze_seasonal_dependency(df)
                
                if seasonal_dependency > 0.6:
                    operational_risks.append({
                        'risk': 'Alta Dependência Sazonal',
                        'probability': 'High',
                        'impact': 'Medium',
                        'score': 6,
                        'description': f'Dependência sazonal de {seasonal_dependency:.1%}',
                        'mitigation': 'Produtos menos sazonais, diversificação temporal'
                    })
            
            # Score operacional
            if operational_risks:
                avg_score = np.mean([risk['score'] for risk in operational_risks])
                max_score = max([risk['score'] for risk in operational_risks])
            else:
                avg_score = 3
                max_score = 3
            
            # Indicadores de eficiência operacional
            efficiency_indicators = self._calculate_operational_efficiency(df)
            
            return {
                'operational_risks': operational_risks,
                'efficiency_indicators': efficiency_indicators,
                'risk_summary': {
                    'total_risks_identified': len(operational_risks),
                    'avg_risk_score': round(avg_score, 1),
                    'max_risk_score': max_score,
                    'overall_risk_level': self._classify_risk_level(avg_score)
                },
                'operational_recommendations': self._generate_operational_recommendations(operational_risks)
            }
            
        except Exception as e:
            return {'error': f"Erro na avaliação de risco operacional: {str(e)}"}
    
    def _assess_market_risk(self, df: pd.DataFrame, risk_tolerance: str,
                          include_mitigation: bool) -> Dict[str, Any]:
        """Avaliar riscos de mercado."""
        try:
            market_risks = []
            
            # 1. Risco de Posicionamento de Preço
            avg_price = df['Preco_Unitario'].mean()
            price_std = df['Preco_Unitario'].std()
            
            # Benchmarks de mercado para joalherias
            market_benchmarks = {
                'economy': (50, 500),
                'mid': (500, 1500),
                'premium': (1500, 3000),
                'luxury': (3000, 10000)
            }
            
            # Determinar posicionamento atual
            current_positioning = None
            for segment, (min_price, max_price) in market_benchmarks.items():
                if min_price <= avg_price <= max_price:
                    current_positioning = segment
                    break
            
            if current_positioning is None:
                if avg_price > 10000:
                    current_positioning = 'ultra_luxury'
                else:
                    current_positioning = 'economy'
            
            # Avaliar risco de posicionamento
            if current_positioning in ['economy'] and price_std / avg_price > 0.5:
                market_risks.append({
                    'risk': 'Posicionamento de Preço Inconsistente',
                    'probability': 'Medium',
                    'impact': 'Medium',
                    'score': 5,
                    'description': f'Posicionamento {current_positioning} com alta variação de preços',
                    'mitigation': 'Definir estratégia clara de posicionamento'
                })
            
            # 2. Risco de Competição
            # Baseado na concentração de ticket médio
            if current_positioning in ['mid', 'economy']:
                market_risks.append({
                    'risk': 'Alta Competição no Segmento',
                    'probability': 'High',
                    'impact': 'Medium',
                    'score': 6,
                    'description': f'Posicionamento {current_positioning} com competição intensa',
                    'mitigation': 'Diferenciação de produto, foco em experiência'
                })
            
            # 3. Risco de Mudança de Tendências
            # Análise de variação temporal em categorias
            if 'Grupo_Produto' in df.columns and len(df['Year_Month'].unique()) >= 3:
                category_trends = self._analyze_category_trends(df)
                declining_categories = sum(1 for trend in category_trends.values() if trend < -0.1)
                
                if declining_categories > len(category_trends) * 0.3:
                    market_risks.append({
                        'risk': 'Mudança de Tendências de Mercado',
                        'probability': 'Medium',
                        'impact': 'High',
                        'score': 7,
                        'description': f'{declining_categories} categorias em declínio',
                        'mitigation': 'Pesquisa de tendências, renovação de portfolio'
                    })
            
            # 4. Risco Macroeconômico
            # Baseado na volatilidade de vendas (proxy para sensibilidade econômica)
            monthly_sales = df.groupby('Year_Month')['Total_Liquido'].sum()
            if len(monthly_sales) >= 3:
                sales_volatility = monthly_sales.std() / monthly_sales.mean()
                
                if sales_volatility > 0.3 and current_positioning in ['luxury', 'premium']:
                    market_risks.append({
                        'risk': 'Sensibilidade a Ciclos Econômicos',
                        'probability': 'Medium',
                        'impact': 'High',
                        'score': 7,
                        'description': f'Alta volatilidade ({sales_volatility:.1%}) em segmento sensível',
                        'mitigation': 'Diversificar para segmentos menos voláteis'
                    })
            
            # 5. Risco de Substituição
            # Baseado na diversidade de categorias
            if 'Grupo_Produto' in df.columns:
                category_count = df['Grupo_Produto'].nunique()
                
                if category_count < 3:
                    market_risks.append({
                        'risk': 'Baixa Diversificação - Risco de Substituição',
                        'probability': 'Medium',
                        'impact': 'Medium',
                        'score': 5,
                        'description': f'Apenas {category_count} categorias principais',
                        'mitigation': 'Expandir para categorias complementares'
                    })
            
            # Score de mercado
            if market_risks:
                avg_score = np.mean([risk['score'] for risk in market_risks])
                max_score = max([risk['score'] for risk in market_risks])
            else:
                avg_score = 4
                max_score = 4
            
            # Análise competitiva
            competitive_analysis = self._analyze_competitive_position(df, current_positioning)
            
            return {
                'market_risks': market_risks,
                'competitive_analysis': competitive_analysis,
                'market_positioning': current_positioning,
                'risk_summary': {
                    'total_risks_identified': len(market_risks),
                    'avg_risk_score': round(avg_score, 1),
                    'max_risk_score': max_score,
                    'overall_risk_level': self._classify_risk_level(avg_score)
                },
                'market_opportunities': self._identify_market_opportunities(df, current_positioning)
            }
            
        except Exception as e:
            return {'error': f"Erro na avaliação de risco de mercado: {str(e)}"}
    
    def _assess_customer_risk(self, df: pd.DataFrame, risk_tolerance: str,
                            include_mitigation: bool) -> Dict[str, Any]:
        """Avaliar riscos relacionados a clientes."""
        try:
            customer_risks = []
            
            # 1. Risco de Concentração de Clientes
            customer_revenue = df.groupby('Customer_ID')['Total_Liquido'].sum().sort_values(ascending=False)
            top_10_pct = int(len(customer_revenue) * 0.1)
            if top_10_pct > 0:
                concentration = customer_revenue.head(top_10_pct).sum() / customer_revenue.sum() * 100
            else:
                concentration = 0
            
            if concentration > 50:
                customer_risks.append({
                    'risk': 'Alta Concentração de Clientes',
                    'probability': 'High',
                    'impact': 'High',
                    'score': 9,
                    'description': f'Top 10% clientes representam {concentration:.1f}% da receita',
                    'mitigation': 'Diversificar base, programa de aquisição'
                })
            elif concentration > 30:
                customer_risks.append({
                    'risk': 'Concentração Moderada de Clientes',
                    'probability': 'Medium',
                    'impact': 'Medium',
                    'score': 6,
                    'description': f'Top 10% clientes representam {concentration:.1f}% da receita',
                    'mitigation': 'Monitorar dependência, expandir base gradualmente'
                })
            
            # 2. Risco de Churn (abandono)
            current_date = df['Data'].max()
            customer_last_purchase = df.groupby('Customer_ID')['Data'].max()
            inactive_customers = (current_date - customer_last_purchase).dt.days > 90
            churn_risk = inactive_customers.sum() / len(customer_last_purchase) * 100
            
            if churn_risk > 40:
                customer_risks.append({
                    'risk': 'Alto Risco de Churn',
                    'probability': 'High',
                    'impact': 'High',
                    'score': 8,
                    'description': f'{churn_risk:.1f}% clientes inativos há 90+ dias',
                    'mitigation': 'Programa de retenção, campanhas de reativação'
                })
            elif churn_risk > 25:
                customer_risks.append({
                    'risk': 'Risco Moderado de Churn',
                    'probability': 'Medium',
                    'impact': 'Medium',
                    'score': 5,
                    'description': f'{churn_risk:.1f}% clientes inativos há 90+ dias',
                    'mitigation': 'Comunicação regular, programa de fidelidade'
                })
            
            # 3. Risco de Baixa Fidelização
            customer_purchase_count = df.groupby('Customer_ID').size()
            one_time_customers = (customer_purchase_count == 1).sum()
            one_time_rate = one_time_customers / len(customer_purchase_count) * 100
            
            if one_time_rate > 70:
                customer_risks.append({
                    'risk': 'Baixa Taxa de Fidelização',
                    'probability': 'High',
                    'impact': 'Medium',
                    'score': 6,
                    'description': f'{one_time_rate:.1f}% são clientes de compra única',
                    'mitigation': 'Programa de segunda compra, follow-up pós-venda'
                })
            elif one_time_rate > 50:
                customer_risks.append({
                    'risk': 'Fidelização Moderada',
                    'probability': 'Medium',
                    'impact': 'Low',
                    'score': 4,
                    'description': f'{one_time_rate:.1f}% são clientes de compra única',
                    'mitigation': 'Melhorar experiência de primeira compra'
                })
            
            # 4. Risco de Baixo Valor por Cliente
            avg_customer_value = customer_revenue.mean()
            if avg_customer_value < 1000:
                customer_risks.append({
                    'risk': 'Baixo Valor por Cliente',
                    'probability': 'Medium',
                    'impact': 'Medium',
                    'score': 5,
                    'description': f'Valor médio por cliente: R$ {avg_customer_value:.2f}',
                    'mitigation': 'Estratégias de up-sell, produtos premium'
                })
            
            # 5. Risco de Segmentação Inadequada
            # Análise da distribuição de valor
            customer_value_std = customer_revenue.std()
            customer_value_cv = customer_value_std / customer_revenue.mean()
            
            if customer_value_cv > 2:  # Alta variação
                customer_risks.append({
                    'risk': 'Base de Clientes Muito Heterogênea',
                    'probability': 'Medium',
                    'impact': 'Low',
                    'score': 3,
                    'description': f'Coeficiente de variação: {customer_value_cv:.1f}',
                    'mitigation': 'Segmentação clara, ofertas personalizadas'
                })
            
            # Score de clientes
            if customer_risks:
                avg_score = np.mean([risk['score'] for risk in customer_risks])
                max_score = max([risk['score'] for risk in customer_risks])
            else:
                avg_score = 3
                max_score = 3
            
            # Análise de lifetime value
            clv_analysis = self._analyze_customer_lifetime_value(df)
            
            return {
                'customer_risks': customer_risks,
                'clv_analysis': clv_analysis,
                'customer_metrics': {
                    'total_customers': len(customer_revenue),
                    'avg_customer_value': round(avg_customer_value, 2),
                    'customer_concentration': round(concentration, 1),
                    'churn_risk_rate': round(churn_risk, 1),
                    'one_time_customer_rate': round(one_time_rate, 1)
                },
                'risk_summary': {
                    'total_risks_identified': len(customer_risks),
                    'avg_risk_score': round(avg_score, 1),
                    'max_risk_score': max_score,
                    'overall_risk_level': self._classify_risk_level(avg_score)
                }
            }
            
        except Exception as e:
            return {'error': f"Erro na avaliação de risco de clientes: {str(e)}"}
    
    def _assess_comprehensive_risk(self, df: pd.DataFrame, risk_tolerance: str,
                                 include_mitigation: bool) -> Dict[str, Any]:
        """Avaliação abrangente de todos os riscos."""
        try:
            # Executar todas as avaliações
            business_risk = self._assess_business_risk(df, risk_tolerance, include_mitigation)
            financial_risk = self._assess_financial_risk(df, risk_tolerance, include_mitigation)
            operational_risk = self._assess_operational_risk(df, risk_tolerance, include_mitigation)
            market_risk = self._assess_market_risk(df, risk_tolerance, include_mitigation)
            customer_risk = self._assess_customer_risk(df, risk_tolerance, include_mitigation)
            
            # Consolidar riscos por categoria
            all_risks = {
                'business': business_risk.get('business_risks', []),
                'financial': financial_risk.get('financial_risks', []),
                'operational': operational_risk.get('operational_risks', []),
                'market': market_risk.get('market_risks', []),
                'customer': customer_risk.get('customer_risks', [])
            }
            
            # Top 10 riscos críticos
            critical_risks = []
            for category, risks in all_risks.items():
                for risk in risks:
                    risk['category'] = category
                    critical_risks.append(risk)
            
            critical_risks.sort(key=lambda x: x['score'], reverse=True)
            top_risks = critical_risks[:10]
            
            # Score geral de risco
            category_scores = {}
            for category in ['business', 'financial', 'operational', 'market', 'customer']:
                risks = all_risks[category]
                if risks:
                    category_scores[category] = np.mean([risk['score'] for risk in risks])
                else:
                    category_scores[category] = 2  # Score neutro
            
            overall_risk_score = np.mean(list(category_scores.values()))
            
            # Matriz de risco
            risk_matrix = self._create_risk_matrix(critical_risks)
            
            # Plano de mitigação integrado
            mitigation_plan = self._create_integrated_mitigation_plan(top_risks, risk_tolerance)
            
            # Monitoramento recomendado
            monitoring_plan = self._create_monitoring_plan(top_risks)
            
            return {
                'risk_overview': {
                    'overall_risk_score': round(overall_risk_score, 1),
                    'risk_level': self._classify_risk_level(overall_risk_score),
                    'total_risks_identified': len(critical_risks),
                    'critical_risks_count': len([r for r in critical_risks if r['score'] >= 7])
                },
                'category_scores': {k: round(v, 1) for k, v in category_scores.items()},
                'top_critical_risks': top_risks,
                'risk_matrix': risk_matrix,
                'all_risks_by_category': all_risks,
                'mitigation_plan': mitigation_plan,
                'monitoring_plan': monitoring_plan,
                'risk_tolerance_assessment': {
                    'current_tolerance': risk_tolerance,
                    'recommended_actions': self._get_tolerance_recommendations(overall_risk_score, risk_tolerance)
                }
            }
            
        except Exception as e:
            return {'error': f"Erro na avaliação abrangente: {str(e)}"}
    
    # Métodos auxiliares
    def _classify_risk_level(self, score: float) -> str:
        """Classificar nível de risco."""
        if score >= 7:
            return 'Alto'
        elif score >= 4:
            return 'Médio'
        else:
            return 'Baixo'
    
    def _assess_risk_tolerance_fit(self, risks: List[Dict], tolerance: str) -> Dict[str, Any]:
        """Avaliar compatibilidade com tolerância a risco."""
        high_risk_count = len([r for r in risks if r['score'] >= 7])
        
        recommendations = []
        if tolerance == 'low' and high_risk_count > 0:
            recommendations.append("Tolerância baixa incompatível com riscos altos identificados")
        elif tolerance == 'high' and high_risk_count == 0:
            recommendations.append("Perfil conservador - considerar mais agressividade estratégica")
        
        return {
            'tolerance_fit': 'Compatible' if len(recommendations) == 0 else 'Misaligned',
            'recommendations': recommendations
        }
    
    def _estimate_liquidity_risk(self, df: pd.DataFrame, monthly_revenue: pd.Series) -> Dict[str, Any]:
        """Estimar risco de liquidez."""
        if len(monthly_revenue) == 0:
            return {'liquidity_risk': 'Cannot assess'}
        
        avg_monthly_revenue = monthly_revenue.mean()
        revenue_volatility = monthly_revenue.std() / avg_monthly_revenue if avg_monthly_revenue > 0 else 0
        
        # Estimativas baseadas em benchmarks do setor
        estimated_monthly_expenses = avg_monthly_revenue * 0.7  # 70% de margem
        estimated_cash_need = estimated_monthly_expenses * 3  # 3 meses
        
        return {
            'avg_monthly_revenue': round(avg_monthly_revenue, 2),
            'revenue_volatility': round(revenue_volatility, 3),
            'estimated_monthly_expenses': round(estimated_monthly_expenses, 2),
            'recommended_cash_reserve': round(estimated_cash_need, 2),
            'liquidity_risk_level': 'High' if revenue_volatility > 0.4 else 'Medium' if revenue_volatility > 0.2 else 'Low'
        }
    
    def _calculate_financial_health_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcular indicadores de saúde financeira."""
        total_revenue = df['Total_Liquido'].sum()
        avg_ticket = df['Total_Liquido'].mean()
        total_customers = df['Customer_ID'].nunique()
        
        return {
            'revenue_per_customer': round(total_revenue / total_customers, 2) if total_customers > 0 else 0,
            'avg_order_value': round(avg_ticket, 2),
            'revenue_concentration_risk': self._calculate_revenue_concentration_risk(df),
            'estimated_gross_margin': 0.58,  # Benchmark setor
            'financial_stability_score': self._calculate_financial_stability_score(df)
        }
    
    def _calculate_revenue_concentration_risk(self, df: pd.DataFrame) -> str:
        """Calcular risco de concentração de receita."""
        customer_revenue = df.groupby('Customer_ID')['Total_Liquido'].sum().sort_values(ascending=False)
        top_20_pct = int(len(customer_revenue) * 0.2)
        
        if top_20_pct > 0:
            concentration = customer_revenue.head(top_20_pct).sum() / customer_revenue.sum() * 100
            
            if concentration > 80:
                return 'Very High'
            elif concentration > 60:
                return 'High'
            elif concentration > 40:
                return 'Medium'
            else:
                return 'Low'
        return 'Low'
    
    def _calculate_financial_stability_score(self, df: pd.DataFrame) -> float:
        """Calcular score de estabilidade financeira."""
        monthly_revenue = df.groupby('Year_Month')['Total_Liquido'].sum()
        
        if len(monthly_revenue) < 2:
            return 5.0  # Score neutro
        
        # Fatores de estabilidade
        growth_stability = 1 - min(abs(monthly_revenue.pct_change().std()), 1.0)
        revenue_trend = 1 if monthly_revenue.iloc[-1] > monthly_revenue.iloc[0] else 0.5
        
        stability_score = (growth_stability * 0.7 + revenue_trend * 0.3) * 10
        return round(stability_score, 1)
    
    def _analyze_seasonal_dependency(self, df: pd.DataFrame) -> float:
        """Analisar dependência sazonal."""
        monthly_sales = df.groupby(df['Data'].dt.month)['Total_Liquido'].sum()
        
        if len(monthly_sales) < 3:
            return 0.0
        
        seasonal_variation = (monthly_sales.max() - monthly_sales.min()) / monthly_sales.mean()
        return min(seasonal_variation, 1.0)  # Cap at 100%
    
    def _calculate_operational_efficiency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcular indicadores de eficiência operacional."""
        return {
            'products_per_transaction': df.groupby(['Customer_ID', 'Data']).size().mean(),
            'avg_transaction_value': df['Total_Liquido'].mean(),
            'product_turnover_rate': df['Codigo_Produto'].nunique() / len(df) * 100,
            'sales_consistency': 1 - (df.groupby(df['Data'].dt.date)['Total_Liquido'].sum().std() / df.groupby(df['Data'].dt.date)['Total_Liquido'].sum().mean())
        }
    
    def _generate_operational_recommendations(self, risks: List[Dict]) -> List[str]:
        """Gerar recomendações operacionais."""
        recommendations = []
        
        for risk in risks:
            if 'Dependência' in risk['risk']:
                recommendations.append("Diversificar portfolio de produtos")
            elif 'Estoque' in risk['risk']:
                recommendations.append("Implementar gestão de estoque just-in-time")
            elif 'Rotatividade' in risk['risk']:
                recommendations.append("Otimizar mix de produtos e campanhas de ativação")
        
        return recommendations[:5]  # Top 5
    
    def _analyze_category_trends(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analisar tendências por categoria."""
        trends = {}
        
        for category in df['Grupo_Produto'].unique():
            if pd.isna(category):
                continue
                
            cat_data = df[df['Grupo_Produto'] == category]
            monthly_sales = cat_data.groupby('Year_Month')['Total_Liquido'].sum()
            
            if len(monthly_sales) >= 3:
                # Calcular tendência linear
                x = np.arange(len(monthly_sales))
                slope = np.polyfit(x, monthly_sales.values, 1)[0]
                trends[category] = slope / monthly_sales.mean()  # Normalizar
        
        return trends
    
    def _analyze_competitive_position(self, df: pd.DataFrame, positioning: str) -> Dict[str, Any]:
        """Analisar posição competitiva."""
        avg_price = df['Preco_Unitario'].mean()
        
        # Benchmarks competitivos
        competitive_benchmarks = {
            'economy': {'avg_price': 350, 'competition': 'High'},
            'mid': {'avg_price': 1000, 'competition': 'High'},
            'premium': {'avg_price': 2250, 'competition': 'Medium'},
            'luxury': {'avg_price': 6500, 'competition': 'Low'}
        }
        
        benchmark = competitive_benchmarks.get(positioning, {'avg_price': 1000, 'competition': 'Medium'})
        
        return {
            'positioning': positioning,
            'price_vs_benchmark': round((avg_price / benchmark['avg_price'] - 1) * 100, 1),
            'competition_level': benchmark['competition'],
            'competitive_advantage': 'Price Premium' if avg_price > benchmark['avg_price'] else 'Cost Leadership'
        }
    
    def _identify_market_opportunities(self, df: pd.DataFrame, positioning: str) -> List[str]:
        """Identificar oportunidades de mercado."""
        opportunities = []
        
        avg_price = df['Preco_Unitario'].mean()
        
        if positioning == 'economy' and avg_price > 400:
            opportunities.append("Migração para segmento mid-market")
        elif positioning == 'mid' and avg_price > 1200:
            opportunities.append("Explorar segmento premium")
        elif positioning == 'premium':
            opportunities.append("Desenvolver linha de entrada para capturar mais mercado")
        
        # Oportunidades baseadas em categorias
        if 'Grupo_Produto' in df.columns:
            category_count = df['Grupo_Produto'].nunique()
            if category_count < 4:
                opportunities.append("Expandir para novas categorias de produto")
        
        return opportunities
    
    def _analyze_customer_lifetime_value(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisar valor vitalício do cliente."""
        customer_metrics = df.groupby('Customer_ID').agg({
            'Total_Liquido': ['sum', 'count', 'mean'],
            'Data': ['min', 'max']
        }).fillna(0)
        
        customer_metrics.columns = ['Total_Spent', 'Purchase_Count', 'Avg_Spent', 'First_Purchase', 'Last_Purchase']
        
        # Calcular lifetime em dias
        customer_metrics['Lifetime_Days'] = (
            pd.to_datetime(customer_metrics['Last_Purchase']) - 
            pd.to_datetime(customer_metrics['First_Purchase'])
        ).dt.days + 1
        
        # CLV estimado
        avg_annual_value = customer_metrics['Total_Spent'] / (customer_metrics['Lifetime_Days'] / 365)
        avg_annual_value = avg_annual_value.replace([np.inf, -np.inf], customer_metrics['Total_Spent'])
        
        estimated_clv = avg_annual_value * 2.5  # Assumindo 2.5 anos de vida média
        
        return {
            'avg_customer_lifetime_days': round(customer_metrics['Lifetime_Days'].mean(), 0),
            'avg_clv': round(estimated_clv.mean(), 2),
            'clv_distribution': {
                'min': round(estimated_clv.min(), 2),
                'max': round(estimated_clv.max(), 2),
                'median': round(estimated_clv.median(), 2)
            }
        }
    
    def _create_risk_matrix(self, risks: List[Dict]) -> Dict[str, List[str]]:
        """Criar matriz de risco (probabilidade vs impacto)."""
        matrix = {
            'high_prob_high_impact': [],
            'high_prob_medium_impact': [],
            'high_prob_low_impact': [],
            'medium_prob_high_impact': [],
            'medium_prob_medium_impact': [],
            'medium_prob_low_impact': [],
            'low_prob_high_impact': [],
            'low_prob_medium_impact': [],
            'low_prob_low_impact': []
        }
        
        for risk in risks:
            prob = risk.get('probability', 'Medium').lower()
            impact = risk.get('impact', 'Medium').lower()
            key = f"{prob}_prob_{impact}_impact"
            
            if key in matrix:
                matrix[key].append(risk['risk'])
        
        return matrix
    
    def _create_integrated_mitigation_plan(self, top_risks: List[Dict], tolerance: str) -> Dict[str, Any]:
        """Criar plano integrado de mitigação."""
        plan = {
            'immediate_actions': [],
            'short_term_actions': [],
            'long_term_actions': [],
            'investment_required': 'Medium'
        }
        
        for risk in top_risks[:5]:  # Top 5 riscos
            if risk['score'] >= 8:
                plan['immediate_actions'].append({
                    'risk': risk['risk'],
                    'action': risk.get('mitigation', 'Análise detalhada necessária'),
                    'timeline': '30 dias'
                })
            elif risk['score'] >= 6:
                plan['short_term_actions'].append({
                    'risk': risk['risk'],
                    'action': risk.get('mitigation', 'Desenvolver estratégia'),
                    'timeline': '3 meses'
                })
            else:
                plan['long_term_actions'].append({
                    'risk': risk['risk'],
                    'action': risk.get('mitigation', 'Monitorar'),
                    'timeline': '6-12 meses'
                })
        
        # Ajustar investimento baseado na tolerância
        if tolerance == 'low':
            plan['investment_required'] = 'High'
        elif tolerance == 'high':
            plan['investment_required'] = 'Low'
        
        return plan
    
    def _create_monitoring_plan(self, top_risks: List[Dict]) -> Dict[str, Any]:
        """Criar plano de monitoramento."""
        monitoring = {
            'daily_kpis': [],
            'weekly_reviews': [],
            'monthly_assessments': [],
            'alert_thresholds': {}
        }
        
        # Baseado nos tipos de risco
        risk_types = [risk.get('category', 'general') for risk in top_risks]
        
        if 'financial' in risk_types:
            monitoring['daily_kpis'].append('Receita diária')
            monitoring['alert_thresholds']['daily_revenue_drop'] = '15%'
        
        if 'customer' in risk_types:
            monitoring['weekly_reviews'].append('Taxa de churn')
            monitoring['alert_thresholds']['churn_rate'] = '5%'
        
        if 'operational' in risk_types:
            monitoring['monthly_assessments'].append('Rotatividade de estoque')
            monitoring['alert_thresholds']['inventory_turnover'] = '2.0'
        
        return monitoring
    
    def _get_tolerance_recommendations(self, risk_score: float, tolerance: str) -> List[str]:
        """Obter recomendações baseadas na tolerância."""
        recommendations = []
        
        if tolerance == 'low' and risk_score > 6:
            recommendations.append("Risco acima da tolerância - ações imediatas necessárias")
            recommendations.append("Considerar estratégias mais conservadoras")
        elif tolerance == 'high' and risk_score < 4:
            recommendations.append("Perfil muito conservador - considerar mais agressividade")
            recommendations.append("Oportunidades de crescimento podem estar sendo perdidas")
        elif tolerance == 'medium':
            if risk_score > 7:
                recommendations.append("Alguns riscos altos identificados - priorizar mitigação")
            elif risk_score < 3:
                recommendations.append("Perfil conservador - avaliar oportunidades de crescimento")
        
        return recommendations
    
    def _format_risk_result(self, assessment_type: str, result: Dict[str, Any],
                          risk_tolerance: str, time_horizon: str) -> str:
        """Formatar resultado da avaliação de risco."""
        try:
            timestamp = datetime.now().strftime('%d/%m/%Y %H:%M')
            
            if 'error' in result:
                return f"Erro na avaliação de risco {assessment_type}: {result['error']}"
            
            formatted = f"""# ⚠️ AVALIAÇÃO DE RISCOS
                        ## Tipo: {assessment_type.upper().replace('_', ' ')}
                        **Tolerância**: {risk_tolerance.title()} | **Horizonte**: {time_horizon.replace('_', ' ')} | **Data**: {timestamp}

                        ---

                        """
            
            # Formatação específica por tipo
            if assessment_type == 'comprehensive_risk':
                formatted += self._format_comprehensive_risk(result)
            else:
                formatted += self._format_specific_risk_assessment(result, assessment_type)
            
            formatted += f"""

                        ---
                        ## 📋 METODOLOGIA DE AVALIAÇÃO

                        **Critérios de Risco**: Probabilidade vs Impacto (Escala 1-10)
                        **Tolerância Configurada**: {risk_tolerance.title()}
                        **Horizonte Temporal**: {time_horizon.replace('_', ' ')}

                        *Avaliação gerada pelo Risk Assessment Tool - Insights AI*
                        """
                                    
            return formatted
            
        except Exception as e:
            return f"Erro na formatação: {str(e)}"
    
    def _format_comprehensive_risk(self, result: Dict[str, Any]) -> str:
        """Formatar avaliação abrangente de riscos."""
        formatted = "## 📊 VISÃO GERAL DE RISCOS\n\n"
        
        if 'risk_overview' in result:
            overview = result['risk_overview']
            formatted += f"**Score Geral de Risco**: {overview.get('overall_risk_score', 0):.1f}/10\n"
            formatted += f"**Nível de Risco**: {overview.get('risk_level', 'N/A')}\n"
            formatted += f"**Total de Riscos**: {overview.get('total_risks_identified', 0)}\n"
            formatted += f"**Riscos Críticos**: {overview.get('critical_risks_count', 0)}\n\n"
        
        if 'category_scores' in result:
            formatted += "## 📈 SCORE POR CATEGORIA\n\n"
            for category, score in result['category_scores'].items():
                level = self._classify_risk_level(score)
                formatted += f"- **{category.title()}**: {score:.1f}/10 ({level})\n"
            formatted += "\n"
        
        if 'top_critical_risks' in result:
            formatted += "## 🚨 TOP 5 RISCOS CRÍTICOS\n\n"
            for i, risk in enumerate(result['top_critical_risks'][:5], 1):
                formatted += f"**{i}. {risk['risk']}**\n"
                formatted += f"- **Score**: {risk['score']}/10\n"
                formatted += f"- **Categoria**: {risk.get('category', 'N/A').title()}\n"
                formatted += f"- **Probabilidade**: {risk.get('probability', 'N/A')}\n"
                formatted += f"- **Impacto**: {risk.get('impact', 'N/A')}\n"
                formatted += f"- **Descrição**: {risk.get('description', 'N/A')}\n"
                formatted += f"- **Mitigação**: {risk.get('mitigation', 'N/A')}\n\n"
        
        if 'mitigation_plan' in result:
            formatted += "## 🛡️ PLANO DE MITIGAÇÃO\n\n"
            plan = result['mitigation_plan']
            
            if plan.get('immediate_actions'):
                formatted += "**Ações Imediatas (30 dias)**:\n"
                for action in plan['immediate_actions']:
                    formatted += f"- {action['action']}\n"
                formatted += "\n"
            
            if plan.get('short_term_actions'):
                formatted += "**Ações de Curto Prazo (3 meses)**:\n"
                for action in plan['short_term_actions']:
                    formatted += f"- {action['action']}\n"
                formatted += "\n"
            
            if plan.get('long_term_actions'):
                formatted += "**Ações de Longo Prazo (6-12 meses)**:\n"
                for action in plan['long_term_actions']:
                    formatted += f"- {action['action']}\n"
                formatted += "\n"
        
        if 'monitoring_plan' in result:
            formatted += "## 📋 PLANO DE MONITORAMENTO\n\n"
            monitoring = result['monitoring_plan']
            
            if monitoring.get('daily_kpis'):
                formatted += f"**KPIs Diários**: {', '.join(monitoring['daily_kpis'])}\n"
            
            if monitoring.get('weekly_reviews'):
                formatted += f"**Revisões Semanais**: {', '.join(monitoring['weekly_reviews'])}\n"
            
            if monitoring.get('alert_thresholds'):
                formatted += "**Limites de Alerta**:\n"
                for metric, threshold in monitoring['alert_thresholds'].items():
                    formatted += f"- {metric.replace('_', ' ').title()}: {threshold}\n"
        
        return formatted
    
    def _format_specific_risk_assessment(self, result: Dict[str, Any], assessment_type: str) -> str:
        """Formatar avaliação específica de risco."""
        formatted = f"## 📋 AVALIAÇÃO DE RISCO: {assessment_type.upper().replace('_', ' ')}\n\n"
        
        # Buscar a chave principal de riscos
        risk_key = f"{assessment_type.split('_')[0]}_risks"
        risks = result.get(risk_key, [])
        
        if 'risk_summary' in result:
            summary = result['risk_summary']
            formatted += f"**Riscos Identificados**: {summary.get('total_risks_identified', 0)}\n"
            formatted += f"**Score Médio**: {summary.get('avg_risk_score', 0):.1f}/10\n"
            formatted += f"**Nível Geral**: {summary.get('overall_risk_level', 'N/A')}\n\n"
        
        if risks:
            formatted += "## ⚠️ RISCOS IDENTIFICADOS\n\n"
            for i, risk in enumerate(risks, 1):
                formatted += f"**{i}. {risk['risk']}**\n"
                formatted += f"- **Score**: {risk['score']}/10\n"
                formatted += f"- **Probabilidade**: {risk.get('probability', 'N/A')}\n"
                formatted += f"- **Impacto**: {risk.get('impact', 'N/A')}\n"
                formatted += f"- **Descrição**: {risk.get('description', 'N/A')}\n"
                formatted += f"- **Mitigação**: {risk.get('mitigation', 'N/A')}\n\n"
        
        # Adicionar análises específicas baseadas no tipo
        if assessment_type == 'financial_risk' and 'liquidity_analysis' in result:
            liquidity = result['liquidity_analysis']
            formatted += "## 💰 ANÁLISE DE LIQUIDEZ\n\n"
            formatted += f"- **Receita Mensal Média**: R$ {liquidity.get('avg_monthly_revenue', 0):,.2f}\n"
            formatted += f"- **Volatilidade**: {liquidity.get('revenue_volatility', 0):.1%}\n"
            formatted += f"- **Reserva Recomendada**: R$ {liquidity.get('recommended_cash_reserve', 0):,.2f}\n"
            formatted += f"- **Nível de Risco de Liquidez**: {liquidity.get('liquidity_risk_level', 'N/A')}\n\n"
        
        elif assessment_type == 'customer_risk' and 'customer_metrics' in result:
            metrics = result['customer_metrics']
            formatted += "## 👥 MÉTRICAS DE CLIENTES\n\n"
            formatted += f"- **Total de Clientes**: {metrics.get('total_customers', 0):,}\n"
            formatted += f"- **Valor Médio por Cliente**: R$ {metrics.get('avg_customer_value', 0):,.2f}\n"
            formatted += f"- **Taxa de Concentração**: {metrics.get('customer_concentration', 0):.1f}%\n"
            formatted += f"- **Taxa de Risco de Churn**: {metrics.get('churn_risk_rate', 0):.1f}%\n\n"
        
        return formatted
