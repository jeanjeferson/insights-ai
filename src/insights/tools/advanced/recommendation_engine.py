from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RecommendationInput(BaseModel):
    """Schema de entrada para sistema de recomendações."""
    recommendation_type: str = Field(..., description="Tipo: 'product_recommendations', 'customer_targeting', 'pricing_optimization', 'inventory_suggestions', 'marketing_campaigns', 'strategic_actions'")
    data_csv: str = Field(default="data/vendas.csv", description="Caminho para arquivo CSV")
    target_segment: str = Field(default="all", description="Segmento alvo: 'all', 'vip', 'new_customers', 'at_risk'")
    recommendation_count: int = Field(default=10, description="Número de recomendações")
    confidence_threshold: float = Field(default=0.7, description="Limiar de confiança (0.5-0.95)")

class RecommendationEngine(BaseTool):
    name: str = "Recommendation Engine"
    description: str = """
    Motor de recomendações inteligentes para joalherias:
    
    TIPOS DE RECOMENDAÇÕES:
    - product_recommendations: Recomendações de produtos baseadas em padrões
    - customer_targeting: Targeting inteligente de clientes
    - pricing_optimization: Sugestões de otimização de preços
    - inventory_suggestions: Recomendações de gestão de estoque
    - marketing_campaigns: Campanhas personalizadas por segmento
    - strategic_actions: Ações estratégicas baseadas em dados
    
    ALGORITMOS:
    - Collaborative Filtering
    - Content-Based Filtering
    - Hybrid Recommendations
    - Market Basket Analysis
    - Customer Similarity
    - Trend Analysis
    
    SEGMENTAÇÃO AUTOMÁTICA:
    - VIP customers
    - New customers  
    - At-risk customers
    - High-value prospects
    """
    args_schema: Type[BaseModel] = RecommendationInput
    
    def _run(self, recommendation_type: str, data_csv: str = "data/vendas.csv",
             target_segment: str = "all", recommendation_count: int = 10,
             confidence_threshold: float = 0.7) -> str:
        try:
            # Carregar e preparar dados
            df = pd.read_csv(data_csv, sep=';', encoding='utf-8')
            df = self._prepare_recommendation_data(df)
            
            if df is None or len(df) < 10:
                return "Erro: Dados insuficientes para gerar recomendações (mínimo 10 registros)"
            
            # Dicionário de tipos de recomendação
            recommendation_engines = {
                'product_recommendations': self._generate_product_recommendations,
                'customer_targeting': self._generate_customer_targeting,
                'pricing_optimization': self._generate_pricing_recommendations,
                'inventory_suggestions': self._generate_inventory_recommendations,
                'marketing_campaigns': self._generate_marketing_campaigns,
                'strategic_actions': self._generate_strategic_actions
            }
            
            if recommendation_type not in recommendation_engines:
                return f"Tipo '{recommendation_type}' não suportado. Opções: {list(recommendation_engines.keys())}"
            
            result = recommendation_engines[recommendation_type](
                df, target_segment, recommendation_count, confidence_threshold
            )
            
            return self._format_recommendation_result(recommendation_type, result, target_segment)
            
        except Exception as e:
            return f"Erro no sistema de recomendações: {str(e)}"
    
    def _prepare_recommendation_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Preparar dados para sistema de recomendações."""
        try:
            # Converter data
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
            df = df.dropna(subset=['Data'])
            
            # Métricas básicas
            df['Preco_Unitario'] = df['Total_Liquido'] / df['Quantidade'].replace(0, 1)
            df['Year_Month'] = df['Data'].dt.to_period('M')
            df['Weekday'] = df['Data'].dt.dayofweek
            df['Month'] = df['Data'].dt.month
            
            # Simular customer_id se não existir
            if 'Customer_ID' not in df.columns:
                df = self._simulate_customer_ids(df)
            
            # Categorizar produtos por preço
            df['Price_Category'] = pd.cut(
                df['Preco_Unitario'],
                bins=[0, 500, 1500, 3000, 10000, float('inf')],
                labels=['Economy', 'Mid', 'Premium', 'Luxury', 'Ultra-Luxury']
            )
            
            # Métricas de recência e frequência por cliente
            current_date = df['Data'].max()
            customer_metrics = df.groupby('Customer_ID').agg({
                'Data': ['min', 'max', 'count'],
                'Total_Liquido': ['sum', 'mean'],
                'Quantidade': 'sum'
            }).fillna(0)
            
            # Flatten columns
            customer_metrics.columns = ['_'.join(col).strip() for col in customer_metrics.columns]
            
            # Calcular RFM
            customer_metrics['Recency'] = (current_date - pd.to_datetime(customer_metrics['Data_max'])).dt.days
            customer_metrics['Frequency'] = customer_metrics['Data_count']
            customer_metrics['Monetary'] = customer_metrics['Total_Liquido_sum']
            
            # Segmentar clientes
            customer_metrics['Customer_Segment'] = customer_metrics.apply(self._classify_customer_segment, axis=1)
            
            # Merge de volta com dados originais
            df = df.merge(
                customer_metrics[['Customer_Segment', 'Recency', 'Frequency', 'Monetary']],
                left_on='Customer_ID', right_index=True, how='left'
            )
            
            return df
            
        except Exception as e:
            print(f"Erro na preparação de dados: {str(e)}")
            return None
    
    def _simulate_customer_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simular IDs de clientes."""
        # Estratégia simples baseada em padrões de valor e tempo
        df = df.copy()
        
        # Criar features para clustering
        df['Date_Numeric'] = df['Data'].astype('int64') // 10**9
        
        # Agrupar por similaridade
        features = ['Total_Liquido', 'Date_Numeric']
        if 'Quantidade' in df.columns:
            features.append('Quantidade')
        
        # Normalizar
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features])
        
        # Clustering simples
        n_clusters = min(max(len(df) // 10, 5), 100)  # Entre 5 e 100 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        
        df['Customer_ID'] = 'CUST_' + pd.Series(clusters).astype(str)
        
        return df
    
    def _classify_customer_segment(self, row) -> str:
        """Classificar segmento do cliente."""
        recency = row['Recency']
        frequency = row['Frequency']
        monetary = row['Monetary']
        
        if monetary > 5000 and frequency >= 3 and recency <= 60:
            return 'VIP'
        elif frequency == 1 and recency <= 30:
            return 'New'
        elif recency > 180:
            return 'At Risk'
        elif monetary > 2000:
            return 'High Value'
        elif frequency >= 3:
            return 'Frequent'
        else:
            return 'Regular'
    
    def _generate_product_recommendations(self, df: pd.DataFrame, target_segment: str,
                                        count: int, confidence: float) -> Dict[str, Any]:
        """Gerar recomendações de produtos."""
        try:
            # Filtrar por segmento se especificado
            if target_segment != 'all':
                segment_map = {
                    'vip': 'VIP',
                    'new_customers': 'New',
                    'at_risk': 'At Risk'
                }
                segment = segment_map.get(target_segment, target_segment)
                df_segment = df[df['Customer_Segment'] == segment]
                
                if len(df_segment) == 0:
                    df_segment = df  # Fallback para todos os dados
            else:
                df_segment = df
            
            # Análise de produtos por performance
            product_performance = df_segment.groupby('Codigo_Produto').agg({
                'Total_Liquido': ['sum', 'mean', 'count'],
                'Quantidade': 'sum',
                'Data': 'max',
                'Descricao_Produto': 'first'
            }).fillna(0)
            
            # Flatten columns
            product_performance.columns = ['_'.join(col).strip() for col in product_performance.columns]
            
            # Calcular scores de recomendação
            product_performance['Revenue_Score'] = self._normalize_score(product_performance['Total_Liquido_sum'])
            product_performance['Frequency_Score'] = self._normalize_score(product_performance['Total_Liquido_count'])
            product_performance['Recency_Score'] = self._calculate_recency_score(product_performance['Data_max'])
            product_performance['AOV_Score'] = self._normalize_score(product_performance['Total_Liquido_mean'])
            
            # Score final ponderado
            product_performance['Recommendation_Score'] = (
                product_performance['Revenue_Score'] * 0.3 +
                product_performance['Frequency_Score'] * 0.25 +
                product_performance['Recency_Score'] * 0.2 +
                product_performance['AOV_Score'] * 0.25
            )
            
            # Filtrar por confiança
            high_confidence = product_performance[
                product_performance['Recommendation_Score'] >= confidence
            ]
            
            # Top recomendações
            top_products = high_confidence.nlargest(count, 'Recommendation_Score')
            
            # Análise de categorias recomendadas
            category_analysis = self._analyze_recommended_categories(df_segment, top_products.index.tolist())
            
            # Padrões de compra
            buying_patterns = self._analyze_buying_patterns(df_segment, top_products.index.tolist())
            
            # Market basket analysis
            basket_analysis = self._perform_market_basket_analysis(df_segment)
            
            # Insights de produtos
            product_insights = []
            
            if len(top_products) > 0:
                best_product = top_products.iloc[0]
                product_insights.append(f"Produto top: {best_product['Descricao_Produto_first']} (Score: {best_product['Recommendation_Score']:.2f})")
                
                avg_revenue = top_products['Total_Liquido_sum'].mean()
                product_insights.append(f"Receita média dos produtos recomendados: R$ {avg_revenue:,.2f}")
                
                if target_segment != 'all':
                    product_insights.append(f"Recomendações específicas para segmento: {target_segment}")
            
            return {
                'recommended_products': top_products[
                    ['Descricao_Produto_first', 'Total_Liquido_sum', 'Total_Liquido_count', 'Recommendation_Score']
                ].to_dict('records'),
                'category_analysis': category_analysis,
                'buying_patterns': buying_patterns,
                'basket_analysis': basket_analysis,
                'target_segment': target_segment,
                'confidence_threshold': confidence,
                'total_analyzed': len(product_performance),
                'high_confidence_count': len(high_confidence),
                'product_insights': product_insights
            }
            
        except Exception as e:
            return {'error': f"Erro nas recomendações de produtos: {str(e)}"}
    
    def _generate_customer_targeting(self, df: pd.DataFrame, target_segment: str,
                                   count: int, confidence: float) -> Dict[str, Any]:
        """Gerar targeting de clientes."""
        try:
            # Análise de clientes por segmento
            customer_analysis = df.groupby(['Customer_ID', 'Customer_Segment']).agg({
                'Total_Liquido': ['sum', 'mean', 'count'],
                'Data': ['min', 'max'],
                'Preco_Unitario': 'mean'
            }).fillna(0)
            
            # Flatten columns
            customer_analysis.columns = ['_'.join(col).strip() for col in customer_analysis.columns]
            
            # Calcular lifetime value e potencial
            current_date = df['Data'].max()
            customer_analysis['Days_Since_First'] = (current_date - pd.to_datetime(customer_analysis['Data_min'])).dt.days
            customer_analysis['Days_Since_Last'] = (current_date - pd.to_datetime(customer_analysis['Data_max'])).dt.days
            
            customer_analysis['CLV_Estimate'] = (
                customer_analysis['Total_Liquido_mean'] * 
                customer_analysis['Total_Liquido_count'] * 
                (customer_analysis['Days_Since_First'] / 365 + 1)
            )
            
            # Scoring para targeting
            customer_analysis['Value_Score'] = self._normalize_score(customer_analysis['Total_Liquido_sum'])
            customer_analysis['Frequency_Score'] = self._normalize_score(customer_analysis['Total_Liquido_count'])
            customer_analysis['Recency_Score'] = self._normalize_score(365 - customer_analysis['Days_Since_Last'])
            customer_analysis['AOV_Score'] = self._normalize_score(customer_analysis['Total_Liquido_mean'])
            
            customer_analysis['Target_Score'] = (
                customer_analysis['Value_Score'] * 0.3 +
                customer_analysis['Frequency_Score'] * 0.25 +
                customer_analysis['Recency_Score'] * 0.25 +
                customer_analysis['AOV_Score'] * 0.2
            )
            
            # Targeting por objetivo
            targeting_strategies = {
                'upsell': customer_analysis[
                    (customer_analysis['Total_Liquido_count'] >= 2) & 
                    (customer_analysis['Total_Liquido_mean'] < customer_analysis['Total_Liquido_mean'].median())
                ].nlargest(count, 'Target_Score'),
                
                'retention': customer_analysis[
                    (customer_analysis['Days_Since_Last'] > 60) & 
                    (customer_analysis['Total_Liquido_sum'] > customer_analysis['Total_Liquido_sum'].median())
                ].nlargest(count, 'Target_Score'),
                
                'reactivation': customer_analysis[
                    customer_analysis['Days_Since_Last'] > 180
                ].nlargest(count, 'Target_Score'),
                
                'vip_cultivation': customer_analysis[
                    customer_analysis['CLV_Estimate'] > customer_analysis['CLV_Estimate'].quantile(0.8)
                ].nlargest(count, 'Target_Score')
            }
            
            # Análise de segmentos
            segment_distribution = df['Customer_Segment'].value_counts().to_dict()
            
            # Potencial de receita por estratégia
            revenue_potential = {}
            for strategy, customers in targeting_strategies.items():
                if len(customers) > 0:
                    revenue_potential[strategy] = {
                        'customer_count': len(customers),
                        'total_clv_estimate': customers['CLV_Estimate'].sum(),
                        'avg_clv_estimate': customers['CLV_Estimate'].mean(),
                        'confidence_level': 'High' if len(customers) >= count * 0.8 else 'Medium'
                    }
            
            # Insights de targeting
            targeting_insights = []
            
            if revenue_potential:
                best_strategy = max(revenue_potential.items(), key=lambda x: x[1]['total_clv_estimate'])
                targeting_insights.append(f"Maior potencial: {best_strategy[0]} (R$ {best_strategy[1]['total_clv_estimate']:,.2f})")
                
                total_customers = sum(len(customers) for customers in targeting_strategies.values())
                targeting_insights.append(f"Total de clientes identificados: {total_customers}")
            
            return {
                'targeting_strategies': {k: v.to_dict('records') for k, v in targeting_strategies.items()},
                'segment_distribution': segment_distribution,
                'revenue_potential': revenue_potential,
                'confidence_threshold': confidence,
                'targeting_insights': targeting_insights
            }
            
        except Exception as e:
            return {'error': f"Erro no targeting de clientes: {str(e)}"}
    
    def _generate_pricing_recommendations(self, df: pd.DataFrame, target_segment: str,
                                        count: int, confidence: float) -> Dict[str, Any]:
        """Gerar recomendações de pricing."""
        try:
            # Análise de elasticidade de preço por produto
            pricing_analysis = {}
            
            if 'Grupo_Produto' in df.columns:
                for categoria in df['Grupo_Produto'].unique():
                    if pd.isna(categoria):
                        continue
                    
                    cat_data = df[df['Grupo_Produto'] == categoria]
                    if len(cat_data) < 5:
                        continue
                    
                    # Análise temporal de preços
                    monthly_data = cat_data.groupby('Year_Month').agg({
                        'Preco_Unitario': 'mean',
                        'Quantidade': 'sum',
                        'Total_Liquido': 'sum'
                    })
                    
                    if len(monthly_data) >= 3:
                        # Calcular tendências
                        price_trend = self._calculate_trend(monthly_data['Preco_Unitario'])
                        volume_trend = self._calculate_trend(monthly_data['Quantidade'])
                        
                        # Elasticidade simplificada
                        price_changes = monthly_data['Preco_Unitario'].pct_change().dropna()
                        volume_changes = monthly_data['Quantidade'].pct_change().dropna()
                        
                        if len(price_changes) > 0 and len(volume_changes) > 0:
                            elasticity = np.corrcoef(price_changes, volume_changes)[0, 1]
                        else:
                            elasticity = -0.5  # Valor padrão
                        
                        # Margem estimada
                        avg_price = cat_data['Preco_Unitario'].mean()
                        estimated_cost = avg_price * 0.4  # Assumindo 40% de custo
                        current_margin = (avg_price - estimated_cost) / avg_price
                        
                        # Recomendação de preço
                        price_recommendation = self._calculate_optimal_price(
                            avg_price, elasticity, current_margin
                        )
                        
                        pricing_analysis[categoria] = {
                            'current_avg_price': round(avg_price, 2),
                            'recommended_price': price_recommendation,
                            'price_change_pct': round((price_recommendation - avg_price) / avg_price * 100, 1),
                            'estimated_elasticity': round(elasticity, 3),
                            'current_margin_pct': round(current_margin * 100, 1),
                            'price_trend': price_trend,
                            'volume_trend': volume_trend,
                            'confidence': 'High' if len(monthly_data) >= 6 else 'Medium'
                        }
            
            # Análise competitiva de preços
            competitive_analysis = self._analyze_competitive_pricing(df)
            
            # Oportunidades de pricing
            pricing_opportunities = []
            
            for categoria, data in pricing_analysis.items():
                if data['price_change_pct'] > 5:
                    pricing_opportunities.append({
                        'category': categoria,
                        'opportunity': 'Increase Price',
                        'potential_increase': data['price_change_pct'],
                        'rationale': f"Elasticidade favorável ({data['estimated_elasticity']:.2f})"
                    })
                elif data['price_change_pct'] < -5:
                    pricing_opportunities.append({
                        'category': categoria,
                        'opportunity': 'Decrease Price',
                        'potential_decrease': abs(data['price_change_pct']),
                        'rationale': f"Alta elasticidade ({data['estimated_elasticity']:.2f}) - volume pode compensar"
                    })
            
            # Segmentos de preço sugeridos
            price_segmentation = self._suggest_price_segmentation(df)
            
            # Insights de pricing
            pricing_insights = []
            
            if pricing_opportunities:
                total_opportunities = len(pricing_opportunities)
                increase_ops = len([op for op in pricing_opportunities if op['opportunity'] == 'Increase Price'])
                pricing_insights.append(f"{total_opportunities} oportunidades identificadas ({increase_ops} de aumento)")
            
            avg_margin = np.mean([data['current_margin_pct'] for data in pricing_analysis.values()])
            pricing_insights.append(f"Margem média estimada: {avg_margin:.1f}%")
            
            return {
                'pricing_analysis': pricing_analysis,
                'competitive_analysis': competitive_analysis,
                'pricing_opportunities': pricing_opportunities,
                'price_segmentation': price_segmentation,
                'confidence_threshold': confidence,
                'pricing_insights': pricing_insights
            }
            
        except Exception as e:
            return {'error': f"Erro nas recomendações de pricing: {str(e)}"}
    
    def _generate_inventory_recommendations(self, df: pd.DataFrame, target_segment: str,
                                          count: int, confidence: float) -> Dict[str, Any]:
        """Gerar recomendações de inventory."""
        try:
            # Análise de performance por produto
            product_analysis = df.groupby('Codigo_Produto').agg({
                'Total_Liquido': ['sum', 'mean', 'count'],
                'Quantidade': 'sum',
                'Data': ['min', 'max'],
                'Descricao_Produto': 'first'
            }).fillna(0)
            
            # Flatten columns
            product_analysis.columns = ['_'.join(col).strip() for col in product_analysis.columns]
            
            # Calcular métricas de inventory
            current_date = df['Data'].max()
            product_analysis['Days_Active'] = (
                pd.to_datetime(product_analysis['Data_max']) - 
                pd.to_datetime(product_analysis['Data_min'])
            ).dt.days + 1
            
            product_analysis['Days_Since_Last_Sale'] = (
                current_date - pd.to_datetime(product_analysis['Data_max'])
            ).dt.days
            
            product_analysis['Daily_Revenue'] = product_analysis['Total_Liquido_sum'] / product_analysis['Days_Active']
            product_analysis['Turnover_Estimate'] = product_analysis['Total_Liquido_count'] / (product_analysis['Days_Active'] / 30)
            
            # Classificação ABC
            revenue_sorted = product_analysis.sort_values('Total_Liquido_sum', ascending=False)
            cumsum_pct = revenue_sorted['Total_Liquido_sum'].cumsum() / revenue_sorted['Total_Liquido_sum'].sum()
            
            abc_class = []
            for pct in cumsum_pct:
                if pct <= 0.8:
                    abc_class.append('A')
                elif pct <= 0.95:
                    abc_class.append('B')
                else:
                    abc_class.append('C')
            
            revenue_sorted['ABC_Class'] = abc_class
            product_analysis = product_analysis.merge(
                revenue_sorted[['ABC_Class']], left_index=True, right_index=True, how='left'
            )
            
            # Recomendações por categoria
            inventory_recommendations = {
                'restock_urgent': product_analysis[
                    (product_analysis['ABC_Class'] == 'A') & 
                    (product_analysis['Days_Since_Last_Sale'] <= 30) &
                    (product_analysis['Turnover_Estimate'] > 2)
                ].nlargest(count//2, 'Daily_Revenue'),
                
                'liquidate': product_analysis[
                    (product_analysis['Days_Since_Last_Sale'] > 90) &
                    (product_analysis['ABC_Class'] == 'C') &
                    (product_analysis['Total_Liquido_count'] <= 2)
                ].nsmallest(count//3, 'Daily_Revenue'),
                
                'reduce_stock': product_analysis[
                    (product_analysis['ABC_Class'] == 'B') &
                    (product_analysis['Turnover_Estimate'] < 1) &
                    (product_analysis['Days_Since_Last_Sale'] > 60)
                ].nsmallest(count//3, 'Turnover_Estimate'),
                
                'monitor_closely': product_analysis[
                    (product_analysis['ABC_Class'] == 'A') &
                    (product_analysis['Days_Since_Last_Sale'] > 45)
                ].nlargest(count//4, 'Total_Liquido_sum')
            }
            
            # Análise sazonal para inventory
            seasonal_analysis = self._analyze_seasonal_inventory_needs(df)
            
            # Potencial financeiro das recomendações
            financial_impact = {}
            for rec_type, products in inventory_recommendations.items():
                if len(products) > 0:
                    if rec_type == 'liquidate':
                        # Valor em estoque parado
                        tied_capital = products['Total_Liquido_sum'].sum() * 0.6  # Assumindo 60% do valor de venda
                        financial_impact[rec_type] = {
                            'tied_capital_estimate': round(tied_capital, 2),
                            'potential_recovery': round(tied_capital * 0.7, 2),  # 70% de recovery
                            'product_count': len(products)
                        }
                    elif rec_type == 'restock_urgent':
                        # Oportunidade perdida
                        lost_opportunity = products['Daily_Revenue'].sum() * 30  # 30 dias
                        financial_impact[rec_type] = {
                            'opportunity_cost_monthly': round(lost_opportunity, 2),
                            'product_count': len(products)
                        }
            
            # Insights de inventory
            inventory_insights = []
            
            a_products = len(product_analysis[product_analysis['ABC_Class'] == 'A'])
            inventory_insights.append(f"Produtos classe A: {a_products} (foco prioritário)")
            
            slow_movers = len(product_analysis[product_analysis['Days_Since_Last_Sale'] > 90])
            if slow_movers > 0:
                inventory_insights.append(f"{slow_movers} produtos sem venda há 90+ dias")
            
            if financial_impact.get('liquidate'):
                tied_capital = financial_impact['liquidate']['tied_capital_estimate']
                inventory_insights.append(f"Capital estimado em produtos parados: R$ {tied_capital:,.2f}")
            
            return {
                'inventory_recommendations': {k: v.to_dict('records') for k, v in inventory_recommendations.items()},
                'abc_analysis': product_analysis['ABC_Class'].value_counts().to_dict(),
                'seasonal_analysis': seasonal_analysis,
                'financial_impact': financial_impact,
                'confidence_threshold': confidence,
                'inventory_insights': inventory_insights
            }
            
        except Exception as e:
            return {'error': f"Erro nas recomendações de inventory: {str(e)}"}
    
    def _generate_marketing_campaigns(self, df: pd.DataFrame, target_segment: str,
                                    count: int, confidence: float) -> Dict[str, Any]:
        """Gerar campanhas de marketing personalizadas."""
        try:
            # Análise de segmentos para campanhas
            segment_analysis = df.groupby('Customer_Segment').agg({
                'Total_Liquido': ['sum', 'mean', 'count'],
                'Preco_Unitario': 'mean',
                'Month': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
            }).fillna(0)
            
            # Flatten columns
            segment_analysis.columns = ['_'.join(col).strip() for col in segment_analysis.columns]
            
            # Campanhas personalizadas por segmento
            campaign_recommendations = {}
            
            for segment in df['Customer_Segment'].unique():
                segment_data = df[df['Customer_Segment'] == segment]
                segment_info = segment_analysis.loc[segment]
                
                campaign_type, campaign_details = self._design_campaign_for_segment(
                    segment, segment_data, segment_info
                )
                
                campaign_recommendations[segment] = {
                    'campaign_type': campaign_type,
                    'details': campaign_details,
                    'target_size': len(segment_data['Customer_ID'].unique()),
                    'avg_ticket': round(segment_info['Total_Liquido_mean'], 2),
                    'total_potential': round(segment_info['Total_Liquido_sum'], 2),
                    'preferred_month': int(segment_info['Month_<lambda>']),
                    'confidence': 'High' if len(segment_data) >= 20 else 'Medium'
                }
            
            # Análise de timing para campanhas
            timing_analysis = self._analyze_campaign_timing(df)
            
            # ROI estimado por campanha
            roi_estimates = {}
            for segment, campaign in campaign_recommendations.items():
                estimated_cost = campaign['target_size'] * 15  # R$ 15 por cliente
                estimated_revenue = campaign['avg_ticket'] * campaign['target_size'] * 0.15  # 15% conversion
                roi = (estimated_revenue - estimated_cost) / estimated_cost if estimated_cost > 0 else 0
                
                roi_estimates[segment] = {
                    'estimated_cost': estimated_cost,
                    'estimated_revenue': round(estimated_revenue, 2),
                    'roi_percentage': round(roi * 100, 1),
                    'payback_days': 30 if roi > 0.5 else 60
                }
            
            # Campanhas sazonais
            seasonal_campaigns = self._suggest_seasonal_campaigns(df)
            
            # Insights de marketing
            marketing_insights = []
            
            best_roi_segment = max(roi_estimates.items(), key=lambda x: x[1]['roi_percentage'])
            marketing_insights.append(f"Melhor ROI estimado: {best_roi_segment[0]} ({best_roi_segment[1]['roi_percentage']}%)")
            
            total_target_size = sum(camp['target_size'] for camp in campaign_recommendations.values())
            marketing_insights.append(f"Base total para campanhas: {total_target_size} clientes")
            
            return {
                'campaign_recommendations': campaign_recommendations,
                'timing_analysis': timing_analysis,
                'roi_estimates': roi_estimates,
                'seasonal_campaigns': seasonal_campaigns,
                'confidence_threshold': confidence,
                'marketing_insights': marketing_insights
            }
            
        except Exception as e:
            return {'error': f"Erro nas campanhas de marketing: {str(e)}"}
    
    def _generate_strategic_actions(self, df: pd.DataFrame, target_segment: str,
                                  count: int, confidence: float) -> Dict[str, Any]:
        """Gerar ações estratégicas baseadas em dados."""
        try:
            # Análise de performance geral
            performance_metrics = {
                'total_revenue': df['Total_Liquido'].sum(),
                'total_customers': df['Customer_ID'].nunique(),
                'avg_ticket': df['Total_Liquido'].mean(),
                'total_transactions': len(df),
                'date_range': (df['Data'].min(), df['Data'].max())
            }
            
            # Identificar tendências principais
            monthly_trends = df.groupby('Year_Month').agg({
                'Total_Liquido': 'sum',
                'Customer_ID': 'nunique',
                'Codigo_Produto': 'nunique'
            })
            
            # Calcular taxas de crescimento
            revenue_growth = monthly_trends['Total_Liquido'].pct_change().mean() * 100
            customer_growth = monthly_trends['Customer_ID'].pct_change().mean() * 100
            
            # Ações estratégicas baseadas em padrões
            strategic_actions = []
            
            # 1. Ações baseadas em crescimento
            if revenue_growth > 10:
                strategic_actions.append({
                    'category': 'Growth',
                    'action': 'Acelerar Expansão',
                    'priority': 'High',
                    'rationale': f'Crescimento forte de {revenue_growth:.1f}% mensal',
                    'expected_impact': 'Aumento de 25-40% na receita',
                    'timeline': '3-6 meses',
                    'investment_required': 'Medium'
                })
            elif revenue_growth < -5:
                strategic_actions.append({
                    'category': 'Recovery',
                    'action': 'Plano de Recuperação',
                    'priority': 'Critical',
                    'rationale': f'Declínio de {revenue_growth:.1f}% mensal',
                    'expected_impact': 'Estabilizar receita',
                    'timeline': '1-3 meses',
                    'investment_required': 'High'
                })
            
            # 2. Ações baseadas em segmentação
            segment_distribution = df['Customer_Segment'].value_counts()
            
            if segment_distribution.get('VIP', 0) / len(df) < 0.1:
                strategic_actions.append({
                    'category': 'Customer Development',
                    'action': 'Programa VIP',
                    'priority': 'High',
                    'rationale': 'Baixa base de clientes VIP (<10%)',
                    'expected_impact': 'Aumento de 15-20% no CLV',
                    'timeline': '2-4 meses',
                    'investment_required': 'Medium'
                })
            
            if segment_distribution.get('At Risk', 0) / len(df) > 0.2:
                strategic_actions.append({
                    'category': 'Retention',
                    'action': 'Programa de Retenção',
                    'priority': 'High',
                    'rationale': 'Alto percentual de clientes em risco (>20%)',
                    'expected_impact': 'Redução de 30-50% no churn',
                    'timeline': '1-2 meses',
                    'investment_required': 'Medium'
                })
            
            # 3. Ações baseadas em produtos
            product_concentration = self._analyze_product_concentration(df)
            
            if product_concentration['top_10_percent'] > 80:
                strategic_actions.append({
                    'category': 'Product Mix',
                    'action': 'Diversificação de Portfolio',
                    'priority': 'Medium',
                    'rationale': 'Alta concentração em poucos produtos',
                    'expected_impact': 'Redução de risco e aumento de 10-15% em vendas',
                    'timeline': '4-6 meses',
                    'investment_required': 'High'
                })
            
            # 4. Ações baseadas em sazonalidade
            seasonal_variation = self._calculate_seasonal_variation(df)
            
            if seasonal_variation > 0.4:
                strategic_actions.append({
                    'category': 'Operational',
                    'action': 'Gestão Sazonal Avançada',
                    'priority': 'Medium',
                    'rationale': f'Alta variação sazonal ({seasonal_variation:.1%})',
                    'expected_impact': 'Melhoria de 20-30% na gestão de estoque',
                    'timeline': '2-3 meses',
                    'investment_required': 'Low'
                })
            
            # Priorização das ações
            prioritized_actions = sorted(strategic_actions, 
                                       key=lambda x: {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}[x['priority']])
            
            # ROI estimado das ações
            action_roi = self._estimate_action_roi(strategic_actions, performance_metrics)
            
            # Roadmap de implementação
            implementation_roadmap = self._create_implementation_roadmap(prioritized_actions)
            
            # Insights estratégicos
            strategic_insights = []
            
            critical_actions = len([a for a in strategic_actions if a['priority'] == 'Critical'])
            if critical_actions > 0:
                strategic_insights.append(f"{critical_actions} ações críticas identificadas - ação imediata necessária")
            
            high_priority = len([a for a in strategic_actions if a['priority'] == 'High'])
            strategic_insights.append(f"{high_priority} ações de alta prioridade para próximos 3-6 meses")
            
            total_investment = sum(action_roi.values())
            strategic_insights.append(f"Investimento total estimado: R$ {total_investment:,.2f}")
            
            return {
                'strategic_actions': prioritized_actions,
                'performance_metrics': performance_metrics,
                'action_roi': action_roi,
                'implementation_roadmap': implementation_roadmap,
                'trends_analysis': {
                    'revenue_growth_monthly': round(revenue_growth, 2),
                    'customer_growth_monthly': round(customer_growth, 2)
                },
                'confidence_threshold': confidence,
                'strategic_insights': strategic_insights
            }
            
        except Exception as e:
            return {'error': f"Erro nas ações estratégicas: {str(e)}"}
    
    # Métodos auxiliares
    def _normalize_score(self, series: pd.Series) -> pd.Series:
        """Normalizar scores de 0 a 1."""
        if series.max() == series.min():
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - series.min()) / (series.max() - series.min())
    
    def _calculate_recency_score(self, dates: pd.Series) -> pd.Series:
        """Calcular score de recência."""
        current_date = pd.Timestamp.now()
        days_since = (current_date - pd.to_datetime(dates)).dt.days
        # Inverter: menor tempo = maior score
        max_days = days_since.max()
        return (max_days - days_since) / max_days if max_days > 0 else pd.Series([1] * len(days_since), index=days_since.index)
    
    def _analyze_recommended_categories(self, df: pd.DataFrame, product_codes: List[str]) -> Dict[str, Any]:
        """Analisar categorias dos produtos recomendados."""
        if 'Grupo_Produto' not in df.columns:
            return {}
        
        recommended_products = df[df['Codigo_Produto'].isin(product_codes)]
        category_dist = recommended_products['Grupo_Produto'].value_counts().to_dict()
        
        return {
            'category_distribution': category_dist,
            'top_category': max(category_dist, key=category_dist.get) if category_dist else None
        }
    
    def _analyze_buying_patterns(self, df: pd.DataFrame, product_codes: List[str]) -> Dict[str, Any]:
        """Analisar padrões de compra."""
        patterns = {}
        
        # Padrão semanal
        weekday_pattern = df['Weekday'].value_counts().to_dict()
        patterns['weekday_preference'] = max(weekday_pattern, key=weekday_pattern.get)
        
        # Padrão mensal
        monthly_pattern = df['Month'].value_counts().to_dict()
        patterns['monthly_preference'] = max(monthly_pattern, key=monthly_pattern.get)
        
        return patterns
    
    def _perform_market_basket_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise de cesta de mercado simplificada."""
        # Agrupar por transação (data + cliente)
        transaction_groups = df.groupby(['Customer_ID', 'Data'])['Codigo_Produto'].apply(list)
        
        # Encontrar combinações mais frequentes
        product_combinations = {}
        for products in transaction_groups:
            if len(products) > 1:
                for i, prod1 in enumerate(products):
                    for prod2 in products[i+1:]:
                        combo = tuple(sorted([prod1, prod2]))
                        product_combinations[combo] = product_combinations.get(combo, 0) + 1
        
        # Top combinações
        top_combinations = sorted(product_combinations.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'top_combinations': [{'products': combo, 'frequency': freq} for combo, freq in top_combinations],
            'total_combinations': len(product_combinations)
        }
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calcular tendência de uma série."""
        if len(series) < 2:
            return 'Stable'
        
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]
        
        if slope > series.std() * 0.1:
            return 'Increasing'
        elif slope < -series.std() * 0.1:
            return 'Decreasing'
        else:
            return 'Stable'
    
    def _calculate_optimal_price(self, current_price: float, elasticity: float, 
                               current_margin: float) -> float:
        """Calcular preço ótimo baseado em elasticidade."""
        # Fórmula simplificada de otimização de preço
        if abs(elasticity) < 0.3:  # Inelástico
            # Pode aumentar preço
            optimal_price = current_price * 1.1
        elif abs(elasticity) > 1.5:  # Muito elástico
            # Considerar redução de preço
            optimal_price = current_price * 0.95
        else:
            # Manter próximo ao atual
            optimal_price = current_price * 1.02
        
        return round(optimal_price, 2)
    
    def _analyze_competitive_pricing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise competitiva de preços."""
        # Benchmarks de mercado simplificados
        market_benchmarks = {
            'economy': {'min': 50, 'avg': 350, 'max': 499},
            'mid': {'min': 500, 'avg': 1000, 'max': 1499},
            'premium': {'min': 1500, 'avg': 2250, 'max': 2999},
            'luxury': {'min': 3000, 'avg': 6500, 'max': 9999}
        }
        
        # Posicionamento atual
        avg_price = df['Preco_Unitario'].mean()
        
        positioning = 'premium'
        if avg_price < 500:
            positioning = 'economy'
        elif avg_price < 1500:
            positioning = 'mid'
        elif avg_price < 3000:
            positioning = 'premium'
        else:
            positioning = 'luxury'
        
        return {
            'current_positioning': positioning,
            'avg_price': round(avg_price, 2),
            'market_benchmark': market_benchmarks[positioning]['avg'],
            'price_gap': round(avg_price - market_benchmarks[positioning]['avg'], 2)
        }
    
    def _suggest_price_segmentation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Sugerir segmentação de preços."""
        price_percentiles = df['Preco_Unitario'].quantile([0.25, 0.5, 0.75, 0.9])
        
        return {
            'entry_level': {'max_price': round(price_percentiles[0.25], 2), 'target': 'Novos clientes'},
            'mid_tier': {'max_price': round(price_percentiles[0.5], 2), 'target': 'Clientes regulares'},
            'premium': {'max_price': round(price_percentiles[0.75], 2), 'target': 'Clientes fiéis'},
            'luxury': {'max_price': round(price_percentiles[0.9], 2), 'target': 'Clientes VIP'}
        }
    
    def _analyze_seasonal_inventory_needs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisar necessidades sazonais de estoque."""
        monthly_sales = df.groupby('Month')['Total_Liquido'].sum()
        seasonal_index = (monthly_sales / monthly_sales.mean()).round(2)
        
        peak_months = seasonal_index[seasonal_index > 1.2].index.tolist()
        low_months = seasonal_index[seasonal_index < 0.8].index.tolist()
        
        return {
            'seasonal_index': seasonal_index.to_dict(),
            'peak_months': peak_months,
            'low_months': low_months,
            'seasonal_variation': round(seasonal_index.std(), 2)
        }
    
    def _design_campaign_for_segment(self, segment: str, segment_data: pd.DataFrame, 
                                   segment_info: pd.Series) -> Tuple[str, Dict[str, Any]]:
        """Desenhar campanha específica para segmento."""
        avg_ticket = segment_info['Total_Liquido_mean']
        
        campaigns = {
            'VIP': ('Exclusivity Campaign', {
                'message': 'Acesso exclusivo a nova coleção',
                'channel': 'Personal contact + Email',
                'offer': 'Preview exclusivo + 10% desconto',
                'timing': 'Immediately'
            }),
            'New': ('Welcome Series', {
                'message': 'Bem-vindo à nossa família',
                'channel': 'Email sequence',
                'offer': '15% desconto na segunda compra',
                'timing': 'Week 2 after first purchase'
            }),
            'At Risk': ('Win-back Campaign', {
                'message': 'Sentimos sua falta',
                'channel': 'Email + SMS',
                'offer': '20% desconto + frete grátis',
                'timing': 'Immediately'
            }),
            'High Value': ('Upsell Campaign', {
                'message': 'Peças especiais para você',
                'channel': 'Email + Whatsapp',
                'offer': 'Produtos premium com 10% desconto',
                'timing': 'Monthly'
            }),
            'Frequent': ('Loyalty Program', {
                'message': 'Recompensas por sua fidelidade',
                'channel': 'App notification + Email',
                'offer': 'Pontos dobrados + brinde especial',
                'timing': 'Bi-weekly'
            }),
            'Regular': ('Engagement Campaign', {
                'message': 'Novidades que você vai adorar',
                'channel': 'Newsletter',
                'offer': 'Conteúdo educativo + 5% desconto',
                'timing': 'Weekly'
            })
        }
        
        return campaigns.get(segment, ('General Campaign', {
            'message': 'Ofertas especiais',
            'channel': 'Email',
            'offer': '10% desconto',
            'timing': 'Monthly'
        }))
    
    def _analyze_campaign_timing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisar timing ótimo para campanhas."""
        # Análise por dia da semana
        weekday_sales = df.groupby('Weekday')['Total_Liquido'].sum()
        best_weekday = weekday_sales.idxmax()
        
        # Análise por mês
        monthly_sales = df.groupby('Month')['Total_Liquido'].sum()
        best_months = monthly_sales.nlargest(3).index.tolist()
        
        return {
            'best_weekday': int(best_weekday),
            'best_months': best_months,
            'avoid_months': monthly_sales.nsmallest(2).index.tolist()
        }
    
    def _suggest_seasonal_campaigns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Sugerir campanhas sazonais."""
        seasonal_campaigns = [
            {
                'name': 'Dia das Mães',
                'month': 5,
                'target': 'All segments',
                'strategy': 'Emotional appeal + gift packaging',
                'expected_lift': '25-40%'
            },
            {
                'name': 'Black Friday',
                'month': 11,
                'target': 'Price-sensitive customers',
                'strategy': 'Discount-heavy + urgency',
                'expected_lift': '60-80%'
            },
            {
                'name': 'Natal',
                'month': 12,
                'target': 'All segments',
                'strategy': 'Gift-focused + premium packaging',
                'expected_lift': '40-60%'
            },
            {
                'name': 'Volta às Aulas',
                'month': 1,
                'target': 'Young professionals',
                'strategy': 'New year, new style',
                'expected_lift': '15-25%'
            }
        ]
        
        return seasonal_campaigns
    
    def _analyze_product_concentration(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analisar concentração de produtos."""
        product_sales = df.groupby('Codigo_Produto')['Total_Liquido'].sum().sort_values(ascending=False)
        total_sales = product_sales.sum()
        
        top_10_percent = int(len(product_sales) * 0.1)
        concentration = product_sales.head(top_10_percent).sum() / total_sales * 100
        
        return {
            'top_10_percent': round(concentration, 1),
            'total_products': len(product_sales)
        }
    
    def _calculate_seasonal_variation(self, df: pd.DataFrame) -> float:
        """Calcular variação sazonal."""
        monthly_sales = df.groupby('Month')['Total_Liquido'].sum()
        if len(monthly_sales) < 3:
            return 0
        
        variation = (monthly_sales.max() - monthly_sales.min()) / monthly_sales.mean()
        return variation
    
    def _estimate_action_roi(self, actions: List[Dict], performance: Dict) -> Dict[str, float]:
        """Estimar ROI das ações estratégicas."""
        roi_estimates = {}
        
        investment_costs = {
            'Low': performance['total_revenue'] * 0.02,    # 2% da receita
            'Medium': performance['total_revenue'] * 0.05, # 5% da receita
            'High': performance['total_revenue'] * 0.10    # 10% da receita
        }
        
        for action in actions:
            investment = investment_costs.get(action['investment_required'], 0)
            roi_estimates[action['action']] = round(investment, 2)
        
        return roi_estimates
    
    def _create_implementation_roadmap(self, actions: List[Dict]) -> Dict[str, List[str]]:
        """Criar roadmap de implementação."""
        roadmap = {
            'Month 1': [],
            'Month 2-3': [],
            'Month 4-6': [],
            'Month 6+': []
        }
        
        for action in actions:
            timeline = action['timeline']
            action_name = action['action']
            
            if '1' in timeline:
                roadmap['Month 1'].append(action_name)
            elif '2' in timeline or '3' in timeline:
                roadmap['Month 2-3'].append(action_name)
            elif '4' in timeline or '6' in timeline:
                roadmap['Month 4-6'].append(action_name)
            else:
                roadmap['Month 6+'].append(action_name)
        
        return roadmap
    
    def _format_recommendation_result(self, rec_type: str, result: Dict[str, Any], 
                                    target_segment: str) -> str:
        """Formatar resultado das recomendações."""
        try:
            timestamp = datetime.now().strftime('%d/%m/%Y %H:%M')
            
            if 'error' in result:
                return f"Erro nas recomendações {rec_type}: {result['error']}"
            
            formatted = f"""# 🎯 RECOMMENDATION ENGINE
                        ## Tipo: {rec_type.upper().replace('_', ' ')}
                        **Segmento Alvo**: {target_segment.title()} | **Data**: {timestamp}

                        ---

                        """
            
            # Formatação específica por tipo
            if rec_type == 'product_recommendations':
                formatted += self._format_product_recommendations(result)
            elif rec_type == 'customer_targeting':
                formatted += self._format_customer_targeting(result)
            elif rec_type == 'pricing_optimization':
                formatted += self._format_pricing_optimization(result)
            elif rec_type == 'inventory_suggestions':
                formatted += self._format_inventory_suggestions(result)
            elif rec_type == 'marketing_campaigns':
                formatted += self._format_marketing_campaigns(result)
            elif rec_type == 'strategic_actions':
                formatted += self._format_strategic_actions(result)
            
            formatted += f"""

                    ---
                    ## 📋 METODOLOGIA

                    **Algoritmos**: Collaborative Filtering, Content-Based, Market Basket Analysis
                    **Confiança**: {result.get('confidence_threshold', 'N/A')}
                    **Segmentação**: Automática baseada em RFM e comportamento

                    *Recomendações geradas pelo Recommendation Engine - Insights AI*
                    """
                                
            return formatted
            
        except Exception as e:
            return f"Erro na formatação: {str(e)}"
    
    def _format_product_recommendations(self, result: Dict[str, Any]) -> str:
        """Formatar recomendações de produtos."""
        formatted = "## 💎 PRODUTOS RECOMENDADOS\n\n"
        
        if 'recommended_products' in result:
            for i, product in enumerate(result['recommended_products'][:5], 1):
                formatted += f"**{i}. {product.get('Descricao_Produto_first', 'N/A')}**\n"
                formatted += f"- Score: {product.get('Recommendation_Score', 0):.2f}\n"
                formatted += f"- Receita Total: R$ {product.get('Total_Liquido_sum', 0):,.2f}\n"
                formatted += f"- Vendas: {product.get('Total_Liquido_count', 0)} transações\n\n"
        
        if 'product_insights' in result:
            formatted += "## 💡 INSIGHTS DE PRODUTOS\n\n"
            for insight in result['product_insights']:
                formatted += f"- {insight}\n"
        
        return formatted
    
    def _format_customer_targeting(self, result: Dict[str, Any]) -> str:
        """Formatar targeting de clientes."""
        formatted = "## 🎯 ESTRATÉGIAS DE TARGETING\n\n"
        
        if 'targeting_strategies' in result:
            for strategy, customers in result['targeting_strategies'].items():
                if customers:
                    formatted += f"### {strategy.title()}\n"
                    formatted += f"- Clientes identificados: {len(customers)}\n"
                    if customers:
                        avg_value = np.mean([c.get('Total_Liquido_sum', 0) for c in customers])
                        formatted += f"- Valor médio: R$ {avg_value:,.2f}\n"
                    formatted += "\n"
        
        if 'targeting_insights' in result:
            formatted += "## 💡 INSIGHTS DE TARGETING\n\n"
            for insight in result['targeting_insights']:
                formatted += f"- {insight}\n"
        
        return formatted
    
    def _format_pricing_optimization(self, result: Dict[str, Any]) -> str:
        """Formatar otimização de preços."""
        formatted = "## 💰 OTIMIZAÇÃO DE PREÇOS\n\n"
        
        if 'pricing_opportunities' in result:
            formatted += "**Oportunidades Identificadas**:\n"
            for i, opp in enumerate(result['pricing_opportunities'][:3], 1):
                formatted += f"{i}. **{opp['category']}**: {opp['opportunity']}\n"
                if 'potential_increase' in opp:
                    formatted += f"   - Potencial: +{opp['potential_increase']}%\n"
                formatted += f"   - Justificativa: {opp['rationale']}\n\n"
        
        if 'pricing_insights' in result:
            formatted += "## 💡 INSIGHTS DE PRICING\n\n"
            for insight in result['pricing_insights']:
                formatted += f"- {insight}\n"
        
        return formatted
    
    def _format_inventory_suggestions(self, result: Dict[str, Any]) -> str:
        """Formatar sugestões de inventário."""
        formatted = "## 📦 SUGESTÕES DE INVENTÁRIO\n\n"
        
        if 'inventory_recommendations' in result:
            for rec_type, products in result['inventory_recommendations'].items():
                if products:
                    formatted += f"### {rec_type.replace('_', ' ').title()}\n"
                    formatted += f"- Produtos identificados: {len(products)}\n"
                    formatted += "\n"
        
        if 'financial_impact' in result:
            formatted += "## 💰 IMPACTO FINANCEIRO\n\n"
            for impact_type, data in result['financial_impact'].items():
                formatted += f"**{impact_type.replace('_', ' ').title()}**:\n"
                if 'tied_capital_estimate' in data:
                    formatted += f"- Capital parado: R$ {data['tied_capital_estimate']:,.2f}\n"
                if 'opportunity_cost_monthly' in data:
                    formatted += f"- Oportunidade mensal: R$ {data['opportunity_cost_monthly']:,.2f}\n"
                formatted += "\n"
        
        if 'inventory_insights' in result:
            formatted += "## 💡 INSIGHTS DE INVENTÁRIO\n\n"
            for insight in result['inventory_insights']:
                formatted += f"- {insight}\n"
        
        return formatted
    
    def _format_marketing_campaigns(self, result: Dict[str, Any]) -> str:
        """Formatar campanhas de marketing."""
        formatted = "## 📢 CAMPANHAS RECOMENDADAS\n\n"
        
        if 'campaign_recommendations' in result:
            for segment, campaign in result['campaign_recommendations'].items():
                formatted += f"### {segment}\n"
                formatted += f"- **Tipo**: {campaign['campaign_type']}\n"
                formatted += f"- **Target**: {campaign['target_size']} clientes\n"
                formatted += f"- **Ticket Médio**: R$ {campaign['avg_ticket']:,.2f}\n"
                formatted += f"- **Confiança**: {campaign['confidence']}\n\n"
        
        if 'roi_estimates' in result:
            formatted += "## 📊 ROI ESTIMADO\n\n"
            for segment, roi_data in result['roi_estimates'].items():
                if roi_data['roi_percentage'] > 0:
                    formatted += f"**{segment}**: {roi_data['roi_percentage']}% ROI\n"
        
        if 'marketing_insights' in result:
            formatted += "\n## 💡 INSIGHTS DE MARKETING\n\n"
            for insight in result['marketing_insights']:
                formatted += f"- {insight}\n"
        
        return formatted
    
    def _format_strategic_actions(self, result: Dict[str, Any]) -> str:
        """Formatar ações estratégicas."""
        formatted = "## 🚀 AÇÕES ESTRATÉGICAS RECOMENDADAS\n\n"
        
        if 'strategic_actions' in result:
            for i, action in enumerate(result['strategic_actions'][:5], 1):
                formatted += f"### {i}. {action['action']}\n"
                formatted += f"- **Prioridade**: {action['priority']}\n"
                formatted += f"- **Categoria**: {action['category']}\n"
                formatted += f"- **Timeline**: {action['timeline']}\n"
                formatted += f"- **Impacto Esperado**: {action['expected_impact']}\n"
                formatted += f"- **Justificativa**: {action['rationale']}\n\n"
        
        if 'implementation_roadmap' in result:
            formatted += "## 📅 ROADMAP DE IMPLEMENTAÇÃO\n\n"
            for period, actions in result['implementation_roadmap'].items():
                if actions:
                    formatted += f"**{period}**: {', '.join(actions)}\n"
        
        if 'strategic_insights' in result:
            formatted += "\n## 💡 INSIGHTS ESTRATÉGICOS\n\n"
            for insight in result['strategic_insights']:
                formatted += f"- {insight}\n"
        
        return formatted
