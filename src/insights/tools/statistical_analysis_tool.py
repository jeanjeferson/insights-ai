from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalysisInput(BaseModel):
    """Schema de entrada para análise estatística."""
    analysis_type: str = Field(..., description="Tipo de análise: 'correlation', 'clustering', 'outliers', 'rfm_products', 'trend_test', 'distribution'")
    data_csv: str = Field(default="data/vendas.csv", description="Caminho para o arquivo CSV")
    target_column: str = Field(default="Total_Liquido", description="Coluna alvo para análise")
    group_column: str = Field(default="Grupo_Produto", description="Coluna para agrupamento")

class StatisticalAnalysisTool(BaseTool):
    name: str = "Statistical Analysis Tool" 
    description: str = """
    Análises estatísticas avançadas para joalherias:
    - correlation: Análise de correlação entre variáveis numéricas
    - clustering: Segmentação de produtos/clientes por comportamento
    - outliers: Detecção de anomalias e valores atípicos
    - rfm_products: Análise RFM adaptada para produtos
    - trend_test: Testes de significância de tendências
    - distribution: Análise de distribuição e normalidade
    
    Especializado para dados de joalherias com interpretações específicas do setor.
    """
    args_schema: Type[BaseModel] = StatisticalAnalysisInput
    
    def _run(self, analysis_type: str, data_csv: str = "data/vendas.csv", 
             target_column: str = "Total_Liquido", group_column: str = "Grupo_Produto") -> str:
        try:
            # Carregar dados
            df = pd.read_csv(data_csv, sep=';', encoding='utf-8')
            df['Data'] = pd.to_datetime(df['Data'])
            
            # Dicionário de análises
            analyses = {
                'correlation': self._correlation_analysis,
                'clustering': self._product_clustering, 
                'outliers': self._outlier_detection,
                'rfm_products': self._rfm_product_analysis,
                'trend_test': self._trend_significance_test,
                'distribution': self._distribution_analysis
            }
            
            if analysis_type not in analyses:
                return f"Tipo de análise '{analysis_type}' não suportado. Opções: {list(analyses.keys())}"
            
            result = analyses[analysis_type](df, target_column, group_column)
            return self._format_analysis_result(analysis_type, result)
            
        except Exception as e:
            return f"Erro na análise estatística: {str(e)}"
    
    def _correlation_analysis(self, df: pd.DataFrame, target_column: str, group_column: str) -> Dict[str, Any]:
        """Análise de correlação entre variáveis numéricas."""
        try:
            # Selecionar apenas colunas numéricas
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                return {'error': 'Não há colunas numéricas suficientes para análise de correlação'}
            
            # Matriz de correlação
            corr_matrix = df[numeric_cols].corr()
            
            # Encontrar correlações significativas
            significant_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.3:  # Correlações > 30%
                        var1, var2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        # Teste de significância
                        _, p_value = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
                        
                        significant_corr.append({
                            'variable_1': var1,
                            'variable_2': var2,
                            'correlation': round(corr_value, 3),
                            'strength': self._classify_correlation_strength(abs(corr_value)),
                            'p_value': round(p_value, 4),
                            'significant': p_value < 0.05
                        })
            
            # Ordenar por força da correlação
            significant_corr.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            # Análise específica por categoria se disponível
            category_correlations = {}
            if group_column in df.columns:
                for category in df[group_column].unique():
                    if pd.isna(category):
                        continue
                    cat_data = df[df[group_column] == category][numeric_cols]
                    if len(cat_data) > 10:  # Mínimo de observações
                        cat_corr = cat_data.corr()
                        # Correlação mais forte na categoria
                        max_corr_idx = np.unravel_index(
                            np.argmax(np.abs(cat_corr.values - np.eye(len(cat_corr)))), 
                            cat_corr.shape
                        )
                        if max_corr_idx[0] != max_corr_idx[1]:
                            var1, var2 = cat_corr.columns[max_corr_idx[0]], cat_corr.columns[max_corr_idx[1]]
                            category_correlations[category] = {
                                'variables': f"{var1} x {var2}",
                                'correlation': round(cat_corr.iloc[max_corr_idx], 3)
                            }
            
            return {
                'correlation_matrix': corr_matrix.round(3).to_dict(),
                'significant_correlations': significant_corr[:10],  # Top 10
                'category_specific': category_correlations,
                'total_variables': len(numeric_cols),
                'insights': self._generate_correlation_insights(significant_corr)
            }
            
        except Exception as e:
            return {'error': f"Erro na análise de correlação: {str(e)}"}
    
    def _product_clustering(self, df: pd.DataFrame, target_column: str, group_column: str) -> Dict[str, Any]:
        """Clustering de produtos baseado em performance de vendas."""
        try:
            # Agregar dados por produto
            product_metrics = df.groupby('Codigo_Produto').agg({
                'Total_Liquido': ['sum', 'mean', 'count'],
                'Quantidade': 'sum',
                'Data': ['min', 'max']
            }).fillna(0)
            
            # Flatten column names
            product_metrics.columns = ['_'.join(col).strip() for col in product_metrics.columns]
            
            # Calcular métricas adicionais
            product_metrics['days_since_first_sale'] = (
                pd.to_datetime(product_metrics['Data_max']) - pd.to_datetime(product_metrics['Data_min'])
            ).dt.days
            product_metrics['avg_daily_revenue'] = (
                product_metrics['Total_Liquido_sum'] / 
                (product_metrics['days_since_first_sale'] + 1)
            )
            
            # Selecionar features para clustering
            features = [
                'Total_Liquido_sum', 'Total_Liquido_mean', 'Total_Liquido_count',
                'Quantidade_sum', 'avg_daily_revenue'
            ]
            
            X = product_metrics[features].fillna(0)
            
            # Normalizar dados
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # K-means clustering
            optimal_k = self._find_optimal_clusters(X_scaled)
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            product_metrics['cluster'] = clusters
            
            # Análise dos clusters
            cluster_analysis = {}
            for cluster_id in range(optimal_k):
                cluster_data = product_metrics[product_metrics['cluster'] == cluster_id]
                cluster_analysis[f'Cluster_{cluster_id}'] = {
                    'count': len(cluster_data),
                    'avg_revenue': round(cluster_data['Total_Liquido_sum'].mean(), 2),
                    'avg_frequency': round(cluster_data['Total_Liquido_count'].mean(), 2),
                    'avg_ticket': round(cluster_data['Total_Liquido_mean'].mean(), 2),
                    'profile': self._classify_product_cluster(cluster_data)
                }
            
            # PCA para visualização
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            return {
                'optimal_clusters': optimal_k,
                'cluster_analysis': cluster_analysis,
                'cluster_assignments': product_metrics['cluster'].to_dict(),
                'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
                'insights': self._generate_clustering_insights(cluster_analysis)
            }
            
        except Exception as e:
            return {'error': f"Erro no clustering: {str(e)}"}
    
    def _outlier_detection(self, df: pd.DataFrame, target_column: str, group_column: str) -> Dict[str, Any]:
        """Detecção de outliers usando múltiplos métodos."""
        try:
            if target_column not in df.columns:
                return {'error': f'Coluna {target_column} não encontrada'}
            
            values = df[target_column].dropna()
            
            # Método 1: IQR (Interquartile Range)
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = df[
                (df[target_column] < lower_bound) | (df[target_column] > upper_bound)
            ].copy()
            
            # Método 2: Z-Score
            z_scores = np.abs(stats.zscore(values))
            zscore_outliers = df[z_scores > 3].copy() if len(z_scores) == len(df) else pd.DataFrame()
            
            # Método 3: Modified Z-Score (usando mediana)
            median = values.median()
            mad = np.median(np.abs(values - median))
            modified_z_scores = 0.6745 * (values - median) / mad
            modified_outliers = df[np.abs(modified_z_scores) > 3.5].copy() if len(modified_z_scores) == len(df) else pd.DataFrame()
            
            # Análise por categoria
            category_outliers = {}
            if group_column in df.columns:
                for category in df[group_column].unique():
                    if pd.isna(category):
                        continue
                    cat_data = df[df[group_column] == category][target_column].dropna()
                    if len(cat_data) > 10:
                        cat_Q1 = cat_data.quantile(0.25)
                        cat_Q3 = cat_data.quantile(0.75)
                        cat_IQR = cat_Q3 - cat_Q1
                        cat_outliers = len(cat_data[
                            (cat_data < cat_Q1 - 1.5 * cat_IQR) | 
                            (cat_data > cat_Q3 + 1.5 * cat_IQR)
                        ])
                        category_outliers[category] = {
                            'outlier_count': cat_outliers,
                            'outlier_percentage': round(cat_outliers / len(cat_data) * 100, 2),
                            'max_value': cat_data.max(),
                            'min_value': cat_data.min()
                        }
            
            # Estatísticas dos outliers
            outlier_stats = {
                'total_records': len(df),
                'iqr_method': {
                    'count': len(iqr_outliers),
                    'percentage': round(len(iqr_outliers) / len(df) * 100, 2),
                    'bounds': {'lower': round(lower_bound, 2), 'upper': round(upper_bound, 2)}
                },
                'zscore_method': {
                    'count': len(zscore_outliers),
                    'percentage': round(len(zscore_outliers) / len(df) * 100, 2) if len(zscore_outliers) > 0 else 0
                },
                'modified_zscore_method': {
                    'count': len(modified_outliers),
                    'percentage': round(len(modified_outliers) / len(df) * 100, 2) if len(modified_outliers) > 0 else 0
                }
            }
            
            return {
                'outlier_statistics': outlier_stats,
                'category_analysis': category_outliers,
                'top_outliers': iqr_outliers.nlargest(10, target_column)[[
                    'Codigo_Produto', 'Descricao_Produto', target_column, 'Data'
                ]].to_dict('records') if len(iqr_outliers) > 0 else [],
                'insights': self._generate_outlier_insights(outlier_stats, category_outliers)
            }
            
        except Exception as e:
            return {'error': f"Erro na detecção de outliers: {str(e)}"}
    
    def _rfm_product_analysis(self, df: pd.DataFrame, target_column: str, group_column: str) -> Dict[str, Any]:
        """Análise RFM adaptada para produtos (Recency, Frequency, Monetary)."""
        try:
            # Análise RFM por produto
            current_date = df['Data'].max()
            
            rfm_products = df.groupby('Codigo_Produto').agg({
                'Data': lambda x: (current_date - x.max()).days,  # Recency
                'Total_Liquido': ['count', 'sum'],  # Frequency, Monetary
                'Descricao_Produto': 'first'
            })
            
            # Flatten columns
            rfm_products.columns = ['Recency', 'Frequency', 'Monetary', 'Description']
            
            # Calcular quintis para scoring
            rfm_products['R_Score'] = pd.qcut(rfm_products['Recency'], 5, labels=[5,4,3,2,1]).astype(int)
            rfm_products['F_Score'] = pd.qcut(rfm_products['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
            rfm_products['M_Score'] = pd.qcut(rfm_products['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
            
            # Criar segmentos RFM
            rfm_products['RFM_Score'] = (
                rfm_products['R_Score'].astype(str) + 
                rfm_products['F_Score'].astype(str) + 
                rfm_products['M_Score'].astype(str)
            )
            
            rfm_products['Segment'] = rfm_products.apply(self._categorize_product_rfm, axis=1)
            
            # Análise dos segmentos
            segment_analysis = rfm_products.groupby('Segment').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
                'Description': 'count'
            }).round(2)
            segment_analysis.columns = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Product_Count']
            
            # Top produtos por segmento
            segment_products = {}
            for segment in rfm_products['Segment'].unique():
                segment_data = rfm_products[rfm_products['Segment'] == segment]
                top_products = segment_data.nlargest(5, 'Monetary')[['Description', 'Monetary', 'Frequency']].to_dict('records')
                segment_products[segment] = top_products
            
            return {
                'segment_distribution': rfm_products['Segment'].value_counts().to_dict(),
                'segment_analysis': segment_analysis.to_dict(),
                'top_products_by_segment': segment_products,
                'rfm_scores_distribution': {
                    'R_Score': rfm_products['R_Score'].value_counts().sort_index().to_dict(),
                    'F_Score': rfm_products['F_Score'].value_counts().sort_index().to_dict(),
                    'M_Score': rfm_products['M_Score'].value_counts().sort_index().to_dict()
                },
                'insights': self._generate_rfm_insights(segment_analysis.to_dict(), rfm_products)
            }
            
        except Exception as e:
            return {'error': f"Erro na análise RFM: {str(e)}"}
    
    def _trend_significance_test(self, df: pd.DataFrame, target_column: str, group_column: str) -> Dict[str, Any]:
        """Testes estatísticos de significância de tendências."""
        try:
            # Preparar dados temporais
            df_temporal = df.copy()
            df_temporal['Year_Month'] = df_temporal['Data'].dt.to_period('M')
            
            # Tendência geral
            monthly_sales = df_temporal.groupby('Year_Month')[target_column].sum()
            
            # Teste de Mann-Kendall para tendência
            x = np.arange(len(monthly_sales))
            y = monthly_sales.values
            
            # Correlação de Spearman (não-paramétrica)
            spearman_corr, spearman_p = stats.spearmanr(x, y)
            
            # Regressão linear simples
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Teste de tendência por categoria
            category_trends = {}
            if group_column in df.columns:
                for category in df[group_column].unique():
                    if pd.isna(category):
                        continue
                    cat_data = df_temporal[df_temporal[group_column] == category]
                    if len(cat_data) > 6:  # Mínimo 6 observações
                        cat_monthly = cat_data.groupby('Year_Month')[target_column].sum()
                        if len(cat_monthly) > 3:
                            cat_x = np.arange(len(cat_monthly))
                            cat_y = cat_monthly.values
                            cat_slope, _, cat_r, cat_p, _ = stats.linregress(cat_x, cat_y)
                            
                            category_trends[category] = {
                                'slope': round(cat_slope, 2),
                                'r_squared': round(cat_r**2, 3),
                                'p_value': round(cat_p, 4),
                                'trend_direction': 'crescente' if cat_slope > 0 else 'decrescente',
                                'significant': cat_p < 0.05
                            }
            
            # Teste de sazonalidade (Kruskal-Wallis)
            df_temporal['Month'] = df_temporal['Data'].dt.month
            monthly_groups = [df_temporal[df_temporal['Month'] == month][target_column].values 
                            for month in range(1, 13)]
            monthly_groups = [group for group in monthly_groups if len(group) > 0]
            
            if len(monthly_groups) > 2:
                kw_stat, kw_p = stats.kruskal(*monthly_groups)
                seasonality_significant = kw_p < 0.05
            else:
                kw_stat, kw_p = 0, 1
                seasonality_significant = False
            
            return {
                'overall_trend': {
                    'slope': round(slope, 2),
                    'r_squared': round(r_value**2, 3),
                    'p_value': round(p_value, 4),
                    'significant': p_value < 0.05,
                    'direction': 'crescente' if slope > 0 else 'decrescente',
                    'strength': self._classify_trend_strength(abs(r_value))
                },
                'spearman_test': {
                    'correlation': round(spearman_corr, 3),
                    'p_value': round(spearman_p, 4),
                    'significant': spearman_p < 0.05
                },
                'category_trends': category_trends,
                'seasonality_test': {
                    'kruskal_wallis_stat': round(kw_stat, 3),
                    'p_value': round(kw_p, 4),
                    'significant_seasonality': seasonality_significant
                },
                'insights': self._generate_trend_insights(slope, p_value, category_trends, seasonality_significant)
            }
            
        except Exception as e:
            return {'error': f"Erro no teste de tendências: {str(e)}"}
    
    def _distribution_analysis(self, df: pd.DataFrame, target_column: str, group_column: str) -> Dict[str, Any]:
        """Análise de distribuição e testes de normalidade."""
        try:
            values = df[target_column].dropna()
            
            # Estatísticas descritivas
            desc_stats = {
                'count': len(values),
                'mean': round(values.mean(), 2),
                'median': round(values.median(), 2),
                'std': round(values.std(), 2),
                'min': round(values.min(), 2),
                'max': round(values.max(), 2),
                'skewness': round(stats.skew(values), 3),
                'kurtosis': round(stats.kurtosis(values), 3)
            }
            
            # Testes de normalidade
            shapiro_stat, shapiro_p = stats.shapiro(values[:5000] if len(values) > 5000 else values)
            
            # Teste de normalidade por categoria
            category_distributions = {}
            if group_column in df.columns:
                for category in df[group_column].unique():
                    if pd.isna(category):
                        continue
                    cat_values = df[df[group_column] == category][target_column].dropna()
                    if len(cat_values) > 3:
                        cat_shapiro_stat, cat_shapiro_p = stats.shapiro(
                            cat_values[:5000] if len(cat_values) > 5000 else cat_values
                        )
                        category_distributions[category] = {
                            'count': len(cat_values),
                            'mean': round(cat_values.mean(), 2),
                            'std': round(cat_values.std(), 2),
                            'skewness': round(stats.skew(cat_values), 3),
                            'normal_distribution': cat_shapiro_p > 0.05,
                            'shapiro_p_value': round(cat_shapiro_p, 4)
                        }
            
            # Percentis
            percentiles = {}
            for p in [10, 25, 50, 75, 90, 95, 99]:
                percentiles[f'p{p}'] = round(values.quantile(p/100), 2)
            
            return {
                'descriptive_statistics': desc_stats,
                'normality_test': {
                    'shapiro_wilk_stat': round(shapiro_stat, 4),
                    'p_value': round(shapiro_p, 4),
                    'is_normal': shapiro_p > 0.05
                },
                'percentiles': percentiles,
                'category_distributions': category_distributions,
                'insights': self._generate_distribution_insights(desc_stats, shapiro_p, category_distributions)
            }
            
        except Exception as e:
            return {'error': f"Erro na análise de distribuição: {str(e)}"}
    
    # Métodos auxiliares
    def _classify_correlation_strength(self, corr_value: float) -> str:
        """Classificar força da correlação."""
        if corr_value >= 0.7:
            return 'muito forte'
        elif corr_value >= 0.5:
            return 'forte'
        elif corr_value >= 0.3:
            return 'moderada'
        elif corr_value >= 0.1:
            return 'fraca'
        else:
            return 'muito fraca'
    
    def _find_optimal_clusters(self, X: np.ndarray, max_k: int = 8) -> int:
        """Encontrar número ótimo de clusters usando método do cotovelo."""
        if len(X) < 4:
            return 2
        
        max_k = min(max_k, len(X) - 1)
        inertias = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Método do cotovelo simplificado
        if len(inertias) >= 2:
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            if len(second_diffs) > 0:
                optimal_k = np.argmax(second_diffs) + 3  # +3 porque começamos em k=2
                return min(optimal_k, max_k)
        
        return 3  # Default
    
    def _classify_product_cluster(self, cluster_data: pd.DataFrame) -> str:
        """Classificar perfil do cluster de produtos."""
        avg_revenue = cluster_data['Total_Liquido_sum'].mean()
        avg_frequency = cluster_data['Total_Liquido_count'].mean()
        
        if avg_revenue > cluster_data['Total_Liquido_sum'].quantile(0.75) and avg_frequency > cluster_data['Total_Liquido_count'].quantile(0.75):
            return 'Star Products'
        elif avg_revenue > cluster_data['Total_Liquido_sum'].quantile(0.75):
            return 'Cash Cows'
        elif avg_frequency > cluster_data['Total_Liquido_count'].quantile(0.75):
            return 'Frequent Sellers'
        else:
            return 'Underperformers'
    
    def _categorize_product_rfm(self, row) -> str:
        """Categorizar produtos baseado em scores RFM."""
        if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
            return 'Champions'
        elif row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
            return 'Loyal Products'
        elif row['R_Score'] >= 3 and row['F_Score'] <= 2:
            return 'Potential Stars'
        elif row['R_Score'] <= 2 and row['F_Score'] >= 3:
            return 'At Risk'
        elif row['R_Score'] <= 2 and row['F_Score'] <= 2:
            return 'Lost Products'
        else:
            return 'Others'
    
    def _classify_trend_strength(self, r_value: float) -> str:
        """Classificar força da tendência."""
        if r_value >= 0.8:
            return 'muito forte'
        elif r_value >= 0.6:
            return 'forte'
        elif r_value >= 0.4:
            return 'moderada'
        elif r_value >= 0.2:
            return 'fraca'
        else:
            return 'muito fraca'
    
    def _generate_correlation_insights(self, correlations: list) -> list:
        """Gerar insights da análise de correlação."""
        insights = []
        
        if correlations:
            strongest = correlations[0]
            insights.append(f"Correlação mais forte: {strongest['variable_1']} x {strongest['variable_2']} ({strongest['correlation']})")
            
            significant_count = sum(1 for corr in correlations if corr['significant'])
            insights.append(f"{significant_count} correlações estatisticamente significativas encontradas")
            
            if strongest['correlation'] > 0.7:
                insights.append("Correlação muito forte detectada - investigar relação causal")
            elif strongest['correlation'] < -0.7:
                insights.append("Correlação negativa muito forte - possível relação inversa")
        
        return insights
    
    def _generate_clustering_insights(self, cluster_analysis: Dict) -> list:
        """Gerar insights da análise de clustering."""
        insights = []
        
        total_clusters = len(cluster_analysis)
        insights.append(f"Identificados {total_clusters} grupos distintos de produtos")
        
        # Encontrar cluster dominante
        largest_cluster = max(cluster_analysis.items(), key=lambda x: x[1]['count'])
        insights.append(f"Maior grupo: {largest_cluster[0]} com {largest_cluster[1]['count']} produtos")
        
        # Identificar clusters de alto valor
        high_revenue_clusters = [name for name, data in cluster_analysis.items() 
                               if data['profile'] in ['Star Products', 'Cash Cows']]
        if high_revenue_clusters:
            insights.append(f"Clusters de alto valor: {', '.join(high_revenue_clusters)}")
        
        return insights
    
    def _generate_outlier_insights(self, outlier_stats: Dict, category_outliers: Dict) -> list:
        """Gerar insights da análise de outliers."""
        insights = []
        
        iqr_percentage = outlier_stats['iqr_method']['percentage']
        
        if iqr_percentage > 10:
            insights.append(f"Alto percentual de outliers ({iqr_percentage}%) - investigar qualidade dos dados")
        elif iqr_percentage < 2:
            insights.append("Baixo percentual de outliers - dados consistentes")
        
        if category_outliers:
            max_outlier_category = max(category_outliers.items(), 
                                     key=lambda x: x[1]['outlier_percentage'])
            insights.append(f"Categoria com mais outliers: {max_outlier_category[0]} ({max_outlier_category[1]['outlier_percentage']}%)")
        
        return insights
    
    def _generate_rfm_insights(self, segment_analysis: Dict, rfm_data: pd.DataFrame) -> list:
        """Gerar insights da análise RFM."""
        insights = []
        
        # Analisar distribuição de segmentos
        segment_counts = rfm_data['Segment'].value_counts()
        total_products = len(rfm_data)
        
        champions_pct = (segment_counts.get('Champions', 0) / total_products * 100)
        lost_pct = (segment_counts.get('Lost Products', 0) / total_products * 100)
        
        insights.append(f"{champions_pct:.1f}% dos produtos são Champions")
        
        if champions_pct < 10:
            insights.append("Baixo percentual de products Champions - focar em desenvolvimento de produtos estrela")
        
        if lost_pct > 20:
            insights.append(f"Alto percentual de produtos perdidos ({lost_pct:.1f}%) - considerar descontinuação")
        
        return insights
    
    def _generate_trend_insights(self, slope: float, p_value: float, category_trends: Dict, seasonality: bool) -> list:
        """Gerar insights da análise de tendências."""
        insights = []
        
        if p_value < 0.05:
            direction = "crescente" if slope > 0 else "decrescente"
            insights.append(f"Tendência {direction} estatisticamente significativa")
            
            if abs(slope) > 1000:
                insights.append("Tendência muito forte - mudança acelerada no mercado")
        else:
            insights.append("Não há tendência significativa detectada")
        
        if seasonality:
            insights.append("Padrão sazonal significativo identificado")
        
        if category_trends:
            growing_categories = [cat for cat, trend in category_trends.items() 
                                if trend['trend_direction'] == 'crescente' and trend['significant']]
            if growing_categories:
                insights.append(f"Categorias em crescimento: {', '.join(growing_categories[:3])}")
        
        return insights
    
    def _generate_distribution_insights(self, desc_stats: Dict, shapiro_p: float, category_dists: Dict) -> list:
        """Gerar insights da análise de distribuição."""
        insights = []
        
        # Análise de assimetria
        skewness = desc_stats['skewness']
        if abs(skewness) > 1:
            direction = "positiva" if skewness > 0 else "negativa"
            insights.append(f"Distribuição com assimetria {direction} significativa")
        
        # Normalidade
        if shapiro_p > 0.05:
            insights.append("Distribuição aproximadamente normal")
        else:
            insights.append("Distribuição não-normal - usar estatísticas não-paramétricas")
        
        # Variabilidade
        cv = desc_stats['std'] / desc_stats['mean'] if desc_stats['mean'] > 0 else 0
        if cv > 1:
            insights.append("Alta variabilidade nos dados - considerar segmentação")
        
        return insights
    
    def _format_analysis_result(self, analysis_type: str, result: Dict[str, Any]) -> str:
        """Formatar resultado da análise."""
        try:
            if 'error' in result:
                return f"Erro na análise {analysis_type}: {result['error']}"
            
            formatted = f"# ANÁLISE ESTATÍSTICA: {analysis_type.upper()}\n\n"
            
            for key, value in result.items():
                if key == 'insights':
                    formatted += f"## INSIGHTS\n"
                    for insight in value:
                        formatted += f"- {insight}\n"
                    formatted += "\n"
                elif isinstance(value, dict):
                    formatted += f"## {key.upper().replace('_', ' ')}\n"
                    formatted += self._format_dict_section(value)
                elif isinstance(value, list):
                    formatted += f"## {key.upper().replace('_', ' ')}\n"
                    for i, item in enumerate(value[:5], 1):  # Limit to top 5
                        formatted += f"{i}. {item}\n"
                    formatted += "\n"
                else:
                    formatted += f"**{key.replace('_', ' ').title()}**: {value}\n"
            
            formatted += f"\n---\n*Análise realizada em: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}*\n"
            
            return formatted
            
        except Exception as e:
            return f"Erro na formatação: {str(e)}"
    
    def _format_dict_section(self, data: Dict, level: int = 0) -> str:
        """Formatar seção de dicionário recursivamente."""
        formatted = ""
        indent = "  " * level
        
        for key, value in data.items():
            if isinstance(value, dict):
                formatted += f"{indent}**{key.replace('_', ' ').title()}**:\n"
                formatted += self._format_dict_section(value, level + 1)
            elif isinstance(value, (int, float)):
                formatted += f"{indent}- {key.replace('_', ' ').title()}: {value:,.2f}\n"
            else:
                formatted += f"{indent}- {key.replace('_', ' ').title()}: {value}\n"
        
        return formatted + "\n"
