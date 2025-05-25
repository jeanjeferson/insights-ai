"""
💼 MÓDULO DE BUSINESS MIXINS PARA JOALHERIAS
============================================

Este módulo contém mixins especializados para análises de negócio de joalherias,
com funções que podem ser compartilhadas entre KPI Tool e Statistical Tool.

FUNCIONALIDADES:
✅ Análise RFM especializada para joalherias
✅ Classificação de produtos BCG
✅ Análise ABC de produtos
✅ Benchmarks do setor de joalherias
✅ Insights automáticos especializados
✅ Classificações e segmentações específicas
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class JewelryRFMAnalysisMixin:
    """Mixin para análises RFM especializadas em joalherias."""
    
    def analyze_product_rfm(self, df: pd.DataFrame, current_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Análise RFM para produtos de joalheria.
        
        Args:
            df: DataFrame com dados de vendas
            current_date: Data de referência (default: max data do dataset)
            
        Returns:
            Dicionário com análise RFM de produtos
        """
        try:
            if current_date is None:
                current_date = df['Data'].max()
            
            # RFM por produto - construir agregação dinamicamente
            agg_dict = {
                'Data': lambda x: (current_date - x.max()).days,  # Recency
                'Total_Liquido': ['count', 'sum'],  # Frequency, Monetary
                'Descricao_Produto': 'first',
                'Grupo_Produto': 'first'
            }
            
            # Adicionar Margem_Real apenas se existir
            if 'Margem_Real' in df.columns:
                agg_dict['Margem_Real'] = 'sum'
            
            rfm_products = df.groupby('Codigo_Produto').agg(agg_dict)
            
            # Definir colunas baseado no que foi agregado
            base_columns = ['Recency', 'Frequency', 'Monetary', 'Description', 'Group']
            if 'Margem_Real' in df.columns:
                base_columns.append('Margin')
            
            rfm_products.columns = base_columns
            
            # Scores RFM com ajustes para joalherias
            rfm_products['R_Score'] = pd.qcut(rfm_products['Recency'], 5, labels=[5,4,3,2,1], duplicates='drop').astype(int)
            rfm_products['F_Score'] = pd.qcut(rfm_products['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
            rfm_products['M_Score'] = pd.qcut(rfm_products['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
            
            # Segmentos RFM específicos para joalherias
            rfm_products['RFM_Segment'] = rfm_products.apply(self._categorize_jewelry_product_rfm, axis=1)
            
            # Análise por categoria de joia
            group_analysis = {}
            if 'Group' in rfm_products.columns:
                for group in rfm_products['Group'].unique():
                    if pd.notna(group):
                        group_data = rfm_products[rfm_products['Group'] == group]
                        group_analysis[group] = {
                            'count': len(group_data),
                            'avg_monetary': round(group_data['Monetary'].mean(), 2),
                            'champions_count': len(group_data[group_data['RFM_Segment'] == 'Champions']),
                            'dormant_count': len(group_data[group_data['RFM_Segment'] == 'Dormant']),
                            'avg_recency_days': round(group_data['Recency'].mean(), 1)
                        }
            
            return {
                'analysis_type': 'RFM Produtos Joalheria',
                'total_products': len(rfm_products),
                'segment_distribution': rfm_products['RFM_Segment'].value_counts().to_dict(),
                'group_analysis': group_analysis,
                'top_champions': rfm_products[rfm_products['RFM_Segment'] == 'Champions'].nlargest(10, 'Monetary').to_dict('records'),
                'recommendations': self._generate_jewelry_rfm_recommendations(rfm_products)
            }
            
        except Exception as e:
            return {'error': f"Erro na análise RFM de produtos: {str(e)}"}
    
    def analyze_customer_rfm(self, df: pd.DataFrame, current_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Análise RFM para clientes de joalheria.
        
        Args:
            df: DataFrame com dados de vendas
            current_date: Data de referência
            
        Returns:
            Dicionário com análise RFM de clientes
        """
        try:
            if 'Codigo_Cliente' not in df.columns:
                return {'error': 'Codigo_Cliente não disponível para análise RFM de clientes'}
            
            if current_date is None:
                current_date = df['Data'].max()
            
            # RFM por cliente com características de joalheria
            rfm_customers = df.groupby('Codigo_Cliente').agg({
                'Data': lambda x: (current_date - x.max()).days,  # Recency
                'Total_Liquido': ['count', 'sum', 'mean'],  # Frequency, Monetary, AOV
                'Idade': 'first' if 'Idade' in df.columns else 'count',
                'Sexo': 'first' if 'Sexo' in df.columns else 'count',
                'Grupo_Produto': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A'  # Categoria preferida
            })
            
            rfm_customers.columns = ['Recency', 'Frequency', 'Monetary', 'AOV', 'Age', 'Gender', 'Preferred_Category']
            
            # Scores RFM ajustados para comportamento de compra de joias
            rfm_customers['R_Score'] = pd.qcut(rfm_customers['Recency'], 5, labels=[5,4,3,2,1], duplicates='drop').astype(int)
            rfm_customers['F_Score'] = pd.qcut(rfm_customers['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
            rfm_customers['M_Score'] = pd.qcut(rfm_customers['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop').astype(int)
            
            # Segmentos RFM específicos para clientes de joalheria
            rfm_customers['RFM_Segment'] = rfm_customers.apply(self._categorize_jewelry_customer_rfm, axis=1)
            
            return {
                'analysis_type': 'RFM Clientes Joalheria',
                'total_customers': len(rfm_customers),
                'segment_distribution': rfm_customers['RFM_Segment'].value_counts().to_dict(),
                'segment_profiles': self._create_customer_segment_profiles(rfm_customers),
                'vip_customers': rfm_customers[rfm_customers['RFM_Segment'] == 'VIP Champions'].nlargest(20, 'Monetary').to_dict('records'),
                'at_risk_customers': rfm_customers[rfm_customers['RFM_Segment'] == 'At Risk'].to_dict('records')
            }
            
        except Exception as e:
            return {'error': f"Erro na análise RFM de clientes: {str(e)}"}
    
    def _categorize_jewelry_product_rfm(self, row) -> str:
        """Classificar produtos usando RFM especializado para joalherias."""
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        
        # Lógica específica para produtos de joalheria
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'  # Produtos estrela
        elif r >= 4 and f >= 3 and m >= 3:
            return 'Loyal Performers'  # Produtos consistentes
        elif r >= 4 and f <= 2 and m >= 3:
            return 'New Potential'  # Novos produtos com potencial
        elif r <= 2 and f >= 4 and m >= 3:
            return 'Need Attention'  # Produtos que precisam atenção
        elif r <= 2 and f <= 2 and m >= 4:
            return 'High Value Dormant'  # Alto valor mas dormentes
        elif r <= 1 and f <= 1:
            return 'Dormant'  # Produtos dormentes
        elif m <= 2:
            return 'Low Value'  # Baixo valor
        else:
            return 'Standard'  # Produtos padrão
    
    def _categorize_jewelry_customer_rfm(self, row) -> str:
        """Classificar clientes usando RFM especializado para joalherias."""
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        
        # Lógica específica para clientes de joalheria (compras menos frequentes, alto valor)
        if r >= 4 and f >= 3 and m >= 4:
            return 'VIP Champions'  # Clientes VIP
        elif r >= 3 and f >= 2 and m >= 3:
            return 'Loyal Customers'  # Clientes fiéis
        elif r >= 4 and f <= 2 and m >= 3:
            return 'Occasional High Value'  # Compradores ocasionais de alto valor
        elif r <= 2 and f >= 3 and m >= 2:
            return 'At Risk'  # Em risco de churn
        elif r <= 2 and f <= 2 and m >= 4:
            return 'Cannot Lose'  # Não podem ser perdidos
        elif r <= 1 and f <= 1:
            return 'Lost'  # Perdidos
        elif f >= 4 and m <= 2:
            return 'Frequent Low Value'  # Frequentes mas baixo valor
        else:
            return 'Potential Loyalists'  # Potenciais fiéis
    
    def _generate_jewelry_rfm_recommendations(self, rfm_data: pd.DataFrame) -> List[str]:
        """Gerar recomendações específicas baseadas na análise RFM de joalheria."""
        recommendations = []
        
        # Análise dos segmentos
        segment_counts = rfm_data['RFM_Segment'].value_counts()
        total_products = len(rfm_data)
        
        # Champions
        champions_pct = (segment_counts.get('Champions', 0) / total_products * 100)
        if champions_pct < 15:
            recommendations.append("🏆 Baixo percentual de produtos Champions (<15%) - focar em desenvolver produtos estrela")
        elif champions_pct > 30:
            recommendations.append("⭐ Excelente portfólio com muitos produtos Champions - manter estratégia atual")
        
        # Produtos dormentes
        dormant_pct = (segment_counts.get('Dormant', 0) / total_products * 100)
        if dormant_pct > 25:
            recommendations.append(f"⚠️ Muitos produtos dormentes ({dormant_pct:.1f}%) - considerar liquidação ou relançamento")
        
        # Produtos de alto valor dormentes
        high_value_dormant = segment_counts.get('High Value Dormant', 0)
        if high_value_dormant > 0:
            recommendations.append(f"💎 {high_value_dormant} produtos de alto valor dormentes - prioridade para reativação")
        
        return recommendations
    
    def _create_customer_segment_profiles(self, rfm_data: pd.DataFrame) -> Dict[str, Dict]:
        """Criar perfis detalhados dos segmentos de clientes."""
        profiles = {}
        
        for segment in rfm_data['RFM_Segment'].unique():
            segment_data = rfm_data[rfm_data['RFM_Segment'] == segment]
            
            profiles[segment] = {
                'count': len(segment_data),
                'avg_monetary': round(segment_data['Monetary'].mean(), 2),
                'avg_frequency': round(segment_data['Frequency'].mean(), 2),
                'avg_aov': round(segment_data['AOV'].mean(), 2),
                'avg_recency_days': round(segment_data['Recency'].mean(), 1),
                'preferred_categories': segment_data['Preferred_Category'].value_counts().head(3).to_dict()
            }
            
            # Adicionar perfil demográfico se disponível
            if 'Age' in segment_data.columns and segment_data['Age'].notna().any():
                profiles[segment]['avg_age'] = round(segment_data['Age'].mean(), 1)
            
            if 'Gender' in segment_data.columns and segment_data['Gender'].notna().any():
                profiles[segment]['gender_distribution'] = segment_data['Gender'].value_counts().to_dict()
        
        return profiles

class JewelryBusinessAnalysisMixin:
    """Mixin para análises específicas de negócio de joalherias."""
    
    def create_product_bcg_matrix(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Criar matriz BCG especializada para produtos de joalheria.
        
        Args:
            df: DataFrame com dados de vendas
            
        Returns:
            Dicionário com análise BCG
        """
        try:
            if 'Codigo_Produto' not in df.columns:
                return {'error': 'Codigo_Produto não disponível'}
            
            # Métricas por produto
            product_metrics = df.groupby('Codigo_Produto').agg({
                'Total_Liquido': 'sum',
                'Data': ['min', 'max'],
                'Quantidade': 'sum',
                'Margem_Real': 'sum' if 'Margem_Real' in df.columns else 'count'
            })
            
            # Calcular market share (participação no faturamento)
            total_revenue = df['Total_Liquido'].sum()
            product_metrics['market_share'] = product_metrics['Total_Liquido'] / total_revenue * 100
            
            # Calcular taxa de crescimento (baseada no tempo de vida)
            product_metrics['days_active'] = (
                pd.to_datetime(product_metrics['Data']['max']) - 
                pd.to_datetime(product_metrics['Data']['min'])
            ).dt.days + 1
            
            product_metrics['daily_revenue'] = product_metrics['Total_Liquido'] / product_metrics['days_active']
            
            # Medianas para classificação
            market_share_median = product_metrics['market_share'].median()
            daily_revenue_median = product_metrics['daily_revenue'].median()
            
            # Classificação BCG especializada para joalherias
            def classify_jewelry_bcg(row):
                ms = row['market_share']
                dr = row['daily_revenue']
                
                if ms > market_share_median and dr > daily_revenue_median:
                    return 'Stars'  # Estrelas - alta participação e crescimento
                elif ms > market_share_median and dr <= daily_revenue_median:
                    return 'Cash Cows'  # Vacas leiteiras - alta participação, baixo crescimento
                elif ms <= market_share_median and dr > daily_revenue_median:
                    return 'Question Marks'  # Interrogações - baixa participação, alto crescimento
                else:
                    return 'Dogs'  # Abacaxis - baixa participação e crescimento
            
            product_metrics['bcg_category'] = product_metrics.apply(classify_jewelry_bcg, axis=1)
            
            # Distribuição e recomendações
            bcg_distribution = product_metrics['bcg_category'].value_counts().to_dict()
            
            # Análise por categoria de joia
            category_bcg = {}
            if 'Grupo_Produto' in df.columns:
                for category in df['Grupo_Produto'].unique():
                    if pd.notna(category):
                        cat_products = df[df['Grupo_Produto'] == category]['Codigo_Produto'].unique()
                        cat_bcg = product_metrics[product_metrics.index.isin(cat_products)]['bcg_category'].value_counts()
                        category_bcg[category] = cat_bcg.to_dict()
            
            return {
                'method': 'BCG Matrix for Jewelry Products',
                'distribution': bcg_distribution,
                'category_analysis': category_bcg,
                'stars': product_metrics[product_metrics['bcg_category'] == 'Stars'].nlargest(10, 'Total_Liquido').to_dict(),
                'recommendations': self._generate_bcg_recommendations(bcg_distribution)
            }
            
        except Exception as e:
            return {'error': f"Erro na análise BCG: {str(e)}"}
    
    def perform_abc_analysis(self, df: pd.DataFrame, dimension: str = 'product') -> Dict[str, Any]:
        """
        Análise ABC especializada para joalherias.
        
        Args:
            df: DataFrame com dados
            dimension: 'product', 'customer', 'category'
            
        Returns:
            Dicionário com análise ABC
        """
        try:
            # Definir agrupamento baseado na dimensão
            group_configs = {
                'product': {'group_by': 'Codigo_Produto', 'description': 'Produtos'},
                'customer': {'group_by': 'Codigo_Cliente', 'description': 'Clientes'},
                'category': {'group_by': 'Grupo_Produto', 'description': 'Categorias'}
            }
            
            if dimension not in group_configs:
                return {'error': f"Dimensão '{dimension}' não suportada"}
            
            config = group_configs[dimension]
            group_col = config['group_by']
            
            if group_col not in df.columns:
                return {'error': f"Coluna '{group_col}' não encontrada"}
            
            # Agregação por dimensão
            analysis_data = df.groupby(group_col).agg({
                'Total_Liquido': 'sum',
                'Quantidade': 'sum',
                'Margem_Real': 'sum' if 'Margem_Real' in df.columns else 'count'
            }).sort_values('Total_Liquido', ascending=False)
            
            # Cálculo ABC
            analysis_data['Revenue_Cumsum'] = analysis_data['Total_Liquido'].cumsum()
            analysis_data['Revenue_Pct'] = analysis_data['Total_Liquido'] / analysis_data['Total_Liquido'].sum() * 100
            analysis_data['Revenue_Cumsum_Pct'] = analysis_data['Revenue_Cumsum'] / analysis_data['Total_Liquido'].sum() * 100
            
            # Classificação ABC específica para joalherias
            def classify_abc_jewelry(row):
                cumsum_pct = row['Revenue_Cumsum_Pct']
                if cumsum_pct <= 70:  # Mais restritivo para joalherias
                    return 'A'
                elif cumsum_pct <= 90:
                    return 'B'
                else:
                    return 'C'
            
            analysis_data['ABC_Class'] = analysis_data.apply(classify_abc_jewelry, axis=1)
            
            # Estatísticas por classe
            abc_stats = analysis_data.groupby('ABC_Class').agg({
                'Total_Liquido': ['count', 'sum'],
                'Quantidade': 'sum',
                'Margem_Real': 'sum' if 'Margem_Real' in analysis_data.columns else 'count'
            }).round(2)
            
            # Percentuais
            total_revenue = analysis_data['Total_Liquido'].sum()
            total_items = len(analysis_data)
            
            abc_summary = {}
            for cls in ['A', 'B', 'C']:
                cls_data = analysis_data[analysis_data['ABC_Class'] == cls]
                abc_summary[f'Class_{cls}'] = {
                    'item_count': len(cls_data),
                    'item_percentage': round(len(cls_data) / total_items * 100, 1),
                    'revenue_sum': round(cls_data['Total_Liquido'].sum(), 2),
                    'revenue_percentage': round(cls_data['Total_Liquido'].sum() / total_revenue * 100, 1)
                }
            
            return {
                'analysis_type': f'ABC Analysis - {config["description"]}',
                'dimension': dimension,
                'total_items': total_items,
                'abc_summary': abc_summary,
                'detailed_stats': abc_stats.to_dict(),
                'top_class_a': analysis_data[analysis_data['ABC_Class'] == 'A'].head(10).to_dict(),
                'recommendations': self._generate_abc_recommendations(abc_summary, dimension)
            }
            
        except Exception as e:
            return {'error': f"Erro na análise ABC: {str(e)}"}
    
    def _generate_bcg_recommendations(self, bcg_distribution: Dict[str, int]) -> List[str]:
        """Gerar recomendações baseadas na matriz BCG."""
        recommendations = []
        total = sum(bcg_distribution.values())
        
        # Stars
        stars_pct = (bcg_distribution.get('Stars', 0) / total * 100)
        if stars_pct < 15:
            recommendations.append("⭐ Poucos produtos Stars - investir em desenvolvimento de produtos com alto potencial")
        
        # Cash Cows
        cows_pct = (bcg_distribution.get('Cash Cows', 0) / total * 100)
        if cows_pct > 40:
            recommendations.append("🐄 Muitos Cash Cows - usar receita para investir em inovação")
        
        # Question Marks
        questions_pct = (bcg_distribution.get('Question Marks', 0) / total * 100)
        if questions_pct > 30:
            recommendations.append("❓ Muitos Question Marks - definir estratégias claras: investir ou descontinuar")
        
        # Dogs
        dogs_pct = (bcg_distribution.get('Dogs', 0) / total * 100)
        if dogs_pct > 25:
            recommendations.append("🐕 Muitos produtos Dogs - considerar descontinuação ou reposicionamento")
        
        return recommendations
    
    def _generate_abc_recommendations(self, abc_summary: Dict[str, Dict], dimension: str) -> List[str]:
        """Gerar recomendações baseadas na análise ABC."""
        recommendations = []
        
        class_a = abc_summary.get('Class_A', {})
        class_c = abc_summary.get('Class_C', {})
        
        a_revenue_pct = class_a.get('revenue_percentage', 0)
        a_item_pct = class_a.get('item_percentage', 0)
        c_item_pct = class_c.get('item_percentage', 0)
        
        if a_revenue_pct > 80:
            recommendations.append(f"🎯 Classe A muito concentrada ({a_revenue_pct:.1f}% da receita) - focar na proteção destes {dimension}")
        
        if a_item_pct < 10:
            recommendations.append(f"💎 Poucos itens Classe A ({a_item_pct:.1f}%) geram a maior receita - alta dependência")
        
        if c_item_pct > 60:
            recommendations.append(f"🗑️ Muitos itens Classe C ({c_item_pct:.1f}%) - avaliar descontinuação para simplificar portfólio")
        
        return recommendations

class JewelryBenchmarkMixin:
    """Mixin para benchmarks específicos do setor de joalherias."""
    
    def get_jewelry_industry_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """
        Obter benchmarks do setor de joalherias.
        
        Returns:
            Dicionário com benchmarks por categoria
        """
        return {
            'financial_metrics': {
                'gross_margin_min': 45.0,
                'gross_margin_avg': 58.0,
                'gross_margin_max': 70.0,
                'inventory_turnover_min': 1.5,
                'inventory_turnover_avg': 2.5,
                'inventory_turnover_max': 4.0,
                'aov_min': 800.0,
                'aov_avg': 1500.0,
                'aov_max': 3000.0
            },
            'operational_metrics': {
                'seasonal_variation_min': 20.0,
                'seasonal_variation_avg': 35.0,
                'seasonal_variation_max': 60.0,
                'customer_repeat_rate_min': 15.0,
                'customer_repeat_rate_avg': 25.0,
                'customer_repeat_rate_max': 40.0
            },
            'category_margins': {
                'Anéis': 62.0,
                'Brincos': 58.0,
                'Colares': 55.0,
                'Pulseiras': 60.0,
                'Alianças': 45.0,
                'Pingentes': 65.0,
                'Correntes': 52.0,
                'Outros': 50.0
            },
            'price_elasticity': {
                'Anéis': -0.8,
                'Brincos': -1.2,
                'Colares': -1.0,
                'Pulseiras': -1.3,
                'Alianças': -0.5,
                'Pingentes': -1.1,
                'Correntes': -1.4,
                'Outros': -1.2
            }
        }
    
    def compare_with_benchmarks(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Comparar métricas com benchmarks do setor.
        
        Args:
            metrics: Dicionário com métricas calculadas
            
        Returns:
            Dicionário com comparações e status
        """
        benchmarks = self.get_jewelry_industry_benchmarks()
        comparisons = {}
        
        # Comparações financeiras
        financial_benchmarks = benchmarks['financial_metrics']
        
        # AOV
        if 'aov' in metrics:
            aov = metrics['aov']
            aov_benchmark = financial_benchmarks
            if aov >= aov_benchmark['aov_max']:
                aov_status = 'Excelente'
            elif aov >= aov_benchmark['aov_avg']:
                aov_status = 'Bom'
            elif aov >= aov_benchmark['aov_min']:
                aov_status = 'Abaixo da Média'
            else:
                aov_status = 'Crítico'
            
            comparisons['aov_comparison'] = {
                'current': round(aov, 2),
                'benchmark_avg': aov_benchmark['aov_avg'],
                'status': aov_status,
                'gap_to_avg': round(aov - aov_benchmark['aov_avg'], 2)
            }
        
        # Margem
        if 'gross_margin' in metrics:
            margin = metrics['gross_margin']
            margin_benchmark = financial_benchmarks
            if margin >= margin_benchmark['gross_margin_max']:
                margin_status = 'Excelente'
            elif margin >= margin_benchmark['gross_margin_avg']:
                margin_status = 'Bom'
            elif margin >= margin_benchmark['gross_margin_min']:
                margin_status = 'Abaixo da Média'
            else:
                margin_status = 'Crítico'
            
            comparisons['margin_comparison'] = {
                'current': round(margin, 2),
                'benchmark_avg': margin_benchmark['gross_margin_avg'],
                'status': margin_status,
                'gap_to_avg': round(margin - margin_benchmark['gross_margin_avg'], 2)
            }
        
        return {
            'benchmark_comparisons': comparisons,
            'overall_performance': self._assess_overall_benchmark_performance(comparisons),
            'recommendations': self._generate_benchmark_recommendations(comparisons)
        }
    
    def _assess_overall_benchmark_performance(self, comparisons: Dict[str, Dict]) -> str:
        """Avaliar performance geral comparada aos benchmarks."""
        if not comparisons:
            return 'Dados insuficientes'
        
        excellent_count = sum(1 for comp in comparisons.values() if comp.get('status') == 'Excelente')
        good_count = sum(1 for comp in comparisons.values() if comp.get('status') == 'Bom')
        total_metrics = len(comparisons)
        
        excellent_pct = excellent_count / total_metrics
        good_plus_pct = (excellent_count + good_count) / total_metrics
        
        if excellent_pct >= 0.7:
            return 'Performance Excepcional'
        elif good_plus_pct >= 0.7:
            return 'Performance Boa'
        elif good_plus_pct >= 0.4:
            return 'Performance Média'
        else:
            return 'Performance Abaixo da Média'
    
    def _generate_benchmark_recommendations(self, comparisons: Dict[str, Dict]) -> List[str]:
        """Gerar recomendações baseadas nos benchmarks."""
        recommendations = []
        
        for metric, comparison in comparisons.items():
            status = comparison.get('status', '')
            gap = comparison.get('gap_to_avg', 0)
            
            if status == 'Crítico':
                if 'aov' in metric:
                    recommendations.append(f"🚨 AOV crítico - implementar estratégias de up-sell e cross-sell urgentemente")
                elif 'margin' in metric:
                    recommendations.append(f"🚨 Margem crítica - revisar precificação e custos imediatamente")
            
            elif status == 'Abaixo da Média' and gap < -100:
                if 'aov' in metric:
                    recommendations.append(f"📈 AOV abaixo da média - focar em produtos premium e bundling")
                elif 'margin' in metric:
                    recommendations.append(f"💰 Margem abaixo da média - otimizar mix de produtos e negociação com fornecedores")
        
        return recommendations 