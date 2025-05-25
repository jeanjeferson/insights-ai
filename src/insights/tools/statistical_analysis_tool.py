from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import json
import warnings
from datetime import datetime

# Importar módulos compartilhados consolidados
from .shared.data_preparation import DataPreparationMixin
from .shared.report_formatter import ReportFormatterMixin
from .shared.business_mixins import JewelryRFMAnalysisMixin

warnings.filterwarnings('ignore')

class StatisticalAnalysisToolInput(BaseModel):
    """Schema otimizado para análise estatística com validações robustas."""
    
    analysis_type: str = Field(
        ..., 
        description="""Tipo de análise estatística especializada:
        
        🔬 ANÁLISES ESTATÍSTICAS CORE:
        - 'correlation': Análise de correlação multi-dimensional com testes de significância
        - 'clustering': Clustering avançado (K-means, Hierárquico, DBSCAN) com validação
        - 'outliers': Detecção de outliers usando múltiplos métodos estatísticos
        - 'distribution': Análise de distribuições e testes de normalidade
        - 'trend_analysis': Testes de tendência temporal e sazonalidade
        
        👥 ANÁLISES DEMOGRÁFICAS ESPECIALIZADAS:
        - 'demographic_patterns': Padrões demográficos avançados (idade, sexo, estado civil)
        - 'generational_analysis': Análise geracional (Gen Z, Millennial, Gen X, Boomer)
        - 'customer_segmentation': Segmentação comportamental avançada
        
        🗺️ ANÁLISES GEOGRÁFICAS DETALHADAS:
        - 'geographic_performance': Performance por estado e cidade com estatísticas
        - 'regional_patterns': Padrões sazonais e comportamentais regionais
        
        💰 ANÁLISES ESPECIALIZADAS:
        - 'price_sensitivity': Análise de elasticidade de preços e sensibilidade
        - 'profitability_patterns': Padrões de rentabilidade com análise estatística
        
        🔗 ANÁLISES INTEGRADAS:
        - 'comprehensive_customer_analysis': Análise completa de clientes com estatísticas
        - 'product_performance_analysis': Performance de produtos com testes estatísticos
        """,
        example="correlation"
    )
    
    data_csv: str = Field(
        default="data/vendas.csv", 
        description="Caminho para arquivo CSV com dados de vendas. Use 'data/vendas.csv' para dados principais.",
        example="data/vendas.csv"
    )
    
    target_column: str = Field(
        default="Total_Liquido", 
        description="Coluna alvo para análise. Use 'Total_Liquido' para receita, 'Quantidade' para volume.",
        example="Total_Liquido"
    )
    
    significance_level: float = Field(
        default=0.05, 
        description="Nível de significância para testes estatísticos (0.01-0.10). Padrão: 0.05 (95% confiança).",
        ge=0.01,
        le=0.10
    )
    
    clustering_method: str = Field(
        default="auto", 
        description="Método de clustering: 'kmeans' (rápido), 'hierarchical' (interpretável), 'dbscan' (outliers), 'auto' (otimizado).",
        pattern="^(kmeans|hierarchical|dbscan|auto)$"
    )
    
    min_correlation: float = Field(
        default=0.3, 
        description="Correlação mínima para reportar (0.1-0.9). Valores altos = apenas correlações fortes.",
        ge=0.1,
        le=0.9
    )
    
    demographic_focus: bool = Field(
        default=True, 
        description="Incluir análise demográfica detalhada. Recomendado: True para insights de cliente."
    )
    
    geographic_focus: bool = Field(
        default=True, 
        description="Incluir análise geográfica detalhada. Recomendado: True para expansão regional."
    )
    
    cache_results: bool = Field(
        default=True, 
        description="Usar cache para análises pesadas. Recomendado: True para datasets grandes."
    )
    
    sample_size: Optional[int] = Field(
        default=None,
        description="Tamanho da amostra para análises pesadas (1000-100000). None = usar todos os dados.",
        ge=1000,
        le=100000
    )
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        valid_types = [
            'correlation', 'clustering', 'outliers', 'distribution', 'trend_analysis',
            'demographic_patterns', 'generational_analysis', 'customer_segmentation',
            'geographic_performance', 'regional_patterns', 'price_sensitivity',
            'profitability_patterns', 'comprehensive_customer_analysis', 'product_performance_analysis'
        ]
        if v not in valid_types:
            raise ValueError(f"analysis_type deve ser um de: {valid_types}")
        return v

class StatisticalAnalysisTool(BaseTool, 
                               DataPreparationMixin, 
                               ReportFormatterMixin,
                               JewelryRFMAnalysisMixin):
    """
    🔬 MOTOR DE ANÁLISES ESTATÍSTICAS AVANÇADAS PARA JOALHERIAS
    
    QUANDO USAR:
    - Descobrir padrões ocultos nos dados de vendas
    - Realizar análises estatísticas rigorosas com testes de significância
    - Segmentar clientes baseado em comportamento estatístico
    - Analisar correlações complexas entre variáveis
    - Detectar outliers e anomalias estatísticas
    - Realizar clustering avançado de produtos/clientes
    
    CASOS DE USO ESPECÍFICOS:
    - analysis_type='correlation': Descobrir quais fatores influenciam vendas
    - analysis_type='clustering': Segmentar clientes por padrões de compra
    - analysis_type='demographic_patterns': Analisar comportamento por idade/sexo
    - analysis_type='geographic_performance': Comparar performance entre regiões
    - analysis_type='price_sensitivity': Medir elasticidade de preços
    - analysis_type='outliers': Identificar vendas anômalas para investigação
    
    RESULTADOS ENTREGUES:
    - Análises estatísticas rigorosas com testes de significância
    - Segmentações automáticas baseadas em dados
    - Correlações significativas com interpretação de negócio
    - Clusters de clientes/produtos com perfis detalhados
    - Insights demográficos e geográficos aprofundados
    - Recomendações baseadas em evidências estatísticas
    """
    
    name: str = "Statistical Analysis Tool"
    description: str = (
        "Motor de análises estatísticas avançadas para descobrir padrões ocultos em dados de joalherias. "
        "Realiza análises rigorosas com testes de significância, clustering, correlações e segmentações. "
        "Ideal para insights profundos sobre comportamento de clientes, performance de produtos e padrões de mercado."
    )
    args_schema: Type[BaseModel] = StatisticalAnalysisToolInput
    
    def __init__(self):
        super().__init__()
        self._analysis_cache = {}  # Cache para análises computacionalmente pesadas
    
    def _run(self, analysis_type: str, data_csv: str = "data/vendas.csv", 
             target_column: str = "Total_Liquido", significance_level: float = 0.05,
             clustering_method: str = "auto", min_correlation: float = 0.3,
             demographic_focus: bool = True, geographic_focus: bool = True,
             cache_results: bool = True, sample_size: Optional[int] = None) -> str:
        try:
            print(f"🔬 Iniciando análise estatística v3.0: {analysis_type}")
            print(f"⚙️ Configurações: significância={significance_level}, correlação_min={min_correlation}")
            
            # 1. Carregar e preparar dados usando módulo consolidado
            df = self._load_and_prepare_statistical_data(data_csv, cache_results, sample_size)
            if df is None:
                return json.dumps({
                    "error": "Não foi possível carregar ou preparar os dados para análise estatística",
                    "troubleshooting": {
                        "check_file_exists": f"Verifique se {data_csv} existe",
                        "check_data_quality": "Confirme que os dados têm qualidade suficiente para análise",
                        "check_sample_size": "Verifique se há dados suficientes para análise estatística"
                    },
                    "metadata": {
                        "tool": "Statistical Analysis",
                        "status": "error",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }, ensure_ascii=False, indent=2)
            
            print(f"✅ Dados preparados: {len(df)} registros, {len(df.columns)} campos")
            
            # 2. Mapeamento de análises especializadas
            analysis_methods = {
                # Análises estatísticas core
                'correlation': self._advanced_correlation_analysis,
                'clustering': self._multidimensional_clustering_analysis,
                'outliers': self._comprehensive_outlier_analysis,
                'distribution': self._advanced_distribution_analysis,
                'trend_analysis': self._temporal_trend_analysis,
                
                # Análises especializadas
                'demographic_patterns': self._demographic_patterns_analysis,
                'generational_analysis': self._generational_analysis,
                'customer_segmentation': self._behavioral_customer_segmentation,
                'geographic_performance': self._geographic_performance_analysis,
                'regional_patterns': self._regional_patterns_analysis,
                'price_sensitivity': self._price_elasticity_analysis,
                'profitability_patterns': self._profitability_pattern_analysis,
                
                # Análises integradas
                'comprehensive_customer_analysis': self._comprehensive_customer_analysis,
                'product_performance_analysis': self._statistical_product_analysis
            }
            
            if analysis_type not in analysis_methods:
                available = list(analysis_methods.keys())
                return json.dumps({
                    "error": f"Análise '{analysis_type}' não suportada",
                    "available_analyses": available,
                    "suggestion": "Use uma das análises disponíveis listadas acima",
                    "metadata": {
                        "tool": "Statistical Analysis",
                        "status": "error",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }, ensure_ascii=False, indent=2)
            
            # 3. Executar análise com parâmetros
            analysis_params = {
                'target_column': target_column,
                'significance_level': significance_level,
                'clustering_method': clustering_method,
                'min_correlation': min_correlation,
                'demographic_focus': demographic_focus,
                'geographic_focus': geographic_focus
            }
            
            print(f"🎯 Executando análise: {analysis_type}")
            result = analysis_methods[analysis_type](df, **analysis_params)
            
            # 4. Adicionar metadados
            result['metadata'] = {
                'tool': 'Statistical Analysis Tool v3.0',
                'analysis_type': analysis_type,
                'target_column': target_column,
                'total_records': len(df),
                'significance_level': significance_level,
                'date_range': {
                    'start': df['Data'].min().strftime("%Y-%m-%d") if 'Data' in df.columns else "N/A",
                    'end': df['Data'].max().strftime("%Y-%m-%d") if 'Data' in df.columns else "N/A"
                },
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 5. Armazenar no cache se solicitado
            if cache_results:
                cache_key = f"stat_{analysis_type}_{hash(data_csv)}_{target_column}"
                self._analysis_cache[cache_key] = result
                print(f"💾 Resultado salvo no cache")
            
            # 6. Formatar resultado final
            print("✅ Análise estatística concluída com sucesso!")
            return json.dumps(result, ensure_ascii=False, indent=2, default=str)
            
        except Exception as e:
            error_response = {
                "error": f"Erro na análise estatística v3.0: {str(e)}",
                "analysis_type": analysis_type,
                "data_csv": data_csv,
                "troubleshooting": {
                    "check_data_format": "Verifique se os dados estão no formato correto",
                    "check_statistical_requirements": "Confirme que há dados suficientes para análise estatística",
                    "try_simpler_analysis": "Tente uma análise mais simples primeiro"
                },
                "metadata": {
                    "tool": "Statistical Analysis",
                    "status": "error",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            return json.dumps(error_response, ensure_ascii=False, indent=2)
    
    def _load_and_prepare_statistical_data(self, data_csv: str, use_cache: bool = True, sample_size: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Carregar e preparar dados especificamente para análises estatísticas."""
        cache_key = f"statistical_data_{hash(data_csv)}_{sample_size}"
        
        # Verificar cache
        if use_cache and cache_key in self._analysis_cache:
            print("📋 Usando dados estatísticos do cache")
            return self._analysis_cache[cache_key]
        
        try:
            # Carregar dados brutos
            df = pd.read_csv(data_csv, sep=';', encoding='utf-8')
            print(f"📁 Arquivo carregado: {len(df)} registros")
            
            # Aplicar amostragem se necessário
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                print(f"📊 Amostra aplicada: {len(df)} registros")
            
            # Preparar dados usando mixin consolidado (nível strict para estatísticas)
            df_prepared = self.prepare_jewelry_data(df, validation_level="strict")
            
            if df_prepared is None:
                print("❌ Falha na preparação dos dados estatísticos")
                return None
            
            # Preparações específicas para análises estatísticas
            df_prepared = self._add_statistical_features(df_prepared)
            
            # Armazenar no cache
            if use_cache:
                self._analysis_cache[cache_key] = df_prepared
                print("💾 Dados estatísticos salvos no cache")
            
            return df_prepared
            
        except Exception as e:
            print(f"❌ Erro no carregamento de dados estatísticos: {str(e)}")
            return None
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adicionar features específicas para análises estatísticas."""
        try:
            print("🧮 Adicionando features estatísticas...")
            
            # FASE 1: CORREÇÃO DO BUG - Padronização de valores para clustering
            numeric_cols = ['Total_Liquido', 'Quantidade', 'Margem_Real', 'Preco_Unitario']
            available_numeric = [col for col in numeric_cols if col in df.columns]
            
            if available_numeric:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[available_numeric])
                
                # CORREÇÃO: Criar colunas escaladas corretamente
                for i, col in enumerate(available_numeric):
                    df[f'{col}_scaled'] = scaled_data[:, i]
                
                print(f"✅ Padronização aplicada: {len(available_numeric)} campos escalados")
            
            # Features temporais para análise de tendência
            if 'Data' in df.columns:
                df['Days_Since_Start'] = (df['Data'] - df['Data'].min()).dt.days
                df['Weeks_Since_Start'] = df['Days_Since_Start'] // 7
                df['Month_Index'] = df['Data'].dt.month
                print("✅ Features temporais adicionadas")
            
            # Encoding de variáveis categóricas para clustering
            categorical_cols = ['Sexo', 'Estado_Civil', 'Estado', 'Grupo_Produto']
            for col in categorical_cols:
                if col in df.columns:
                    # One-hot encoding simples
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
            
            print("✅ Encoding categórico aplicado")
            
            # Quartis e ranks para análises
            if 'Total_Liquido' in df.columns:
                df['Revenue_Quartile'] = pd.qcut(df['Total_Liquido'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                df['Revenue_Rank'] = df['Total_Liquido'].rank(pct=True)
                print("✅ Quartis e ranks calculados")
            
            return df
            
        except Exception as e:
            print(f"⚠️ Erro ao adicionar features estatísticas: {str(e)}")
            return df
    
    def _advanced_correlation_analysis(self, df: pd.DataFrame, target_column: str = "Total_Liquido",
                                     min_correlation: float = 0.3, significance_level: float = 0.05,
                                     **kwargs) -> Dict[str, Any]:
        """Análise de correlação avançada com testes de significância."""
        try:
            print("🔍 Executando análise de correlação avançada...")
            
            result = {
                'analysis_type': 'Advanced Correlation Analysis',
                'target_column': target_column,
                'significance_level': significance_level,
                'min_correlation_threshold': min_correlation
            }
            
            # Selecionar colunas numéricas para correlação
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column not in numeric_cols:
                return {'error': f"Coluna alvo '{target_column}' não é numérica"}
            
            # Matriz de correlação
            corr_matrix = df[numeric_cols].corr()
            target_correlations = corr_matrix[target_column].abs().sort_values(ascending=False)
            
            # Filtrar correlações significativas
            significant_correlations = target_correlations[
                (target_correlations >= min_correlation) & 
                (target_correlations.index != target_column)
            ]
            
            result['correlation_matrix'] = corr_matrix.round(3).to_dict()
            result['significant_correlations'] = significant_correlations.round(3).to_dict()
            
            # Testes de significância para correlações
            correlation_tests = {}
            for col in significant_correlations.index:
                if col in df.columns and not df[col].isna().all():
                    r_stat, p_value = stats.pearsonr(df[target_column].dropna(), df[col].dropna())
                    correlation_tests[col] = {
                        'correlation': round(r_stat, 3),
                        'p_value': round(p_value, 4),
                        'significant': p_value < significance_level,
                        'interpretation': self._interpret_correlation_strength(abs(r_stat))
                    }
            
            result['correlation_tests'] = correlation_tests
            
            # Análise por categorias
            categorical_correlations = {}
            categorical_cols = ['Grupo_Produto', 'Metal', 'Faixa_Etaria', 'Sexo', 'Estado']
            
            for cat_col in categorical_cols:
                if cat_col in df.columns:
                    # ANOVA para testar diferença entre grupos
                    groups = [group[target_column].dropna() for name, group in df.groupby(cat_col)]
                    if len(groups) > 1 and all(len(g) > 0 for g in groups):
                        f_stat, p_val = stats.f_oneway(*groups)
                        
                        # Eta-squared (effect size)
                        group_means = df.groupby(cat_col)[target_column].mean()
                        overall_mean = df[target_column].mean()
                        ss_between = sum(df.groupby(cat_col).size() * (group_means - overall_mean)**2)
                        ss_total = sum((df[target_column] - overall_mean)**2)
                        eta_squared = ss_between / ss_total if ss_total > 0 else 0
                        
                        categorical_correlations[cat_col] = {
                            'f_statistic': round(f_stat, 3),
                            'p_value': round(p_val, 4),
                            'significant': p_val < significance_level,
                            'eta_squared': round(eta_squared, 3),
                            'effect_size': self._interpret_effect_size(eta_squared),
                            'group_means': group_means.round(2).to_dict()
                        }
            
            result['categorical_analysis'] = categorical_correlations
            
            # Insights automáticos
            result['insights'] = self._generate_correlation_insights(result)
            result['business_recommendations'] = self._generate_correlation_recommendations(result)
            
            return result
            
        except Exception as e:
            return {'error': f"Erro na análise de correlação: {str(e)}"}
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpretar força da correlação."""
        if correlation >= 0.7:
            return "Muito forte"
        elif correlation >= 0.5:
            return "Forte"
        elif correlation >= 0.3:
            return "Moderada"
        elif correlation >= 0.1:
            return "Fraca"
        else:
            return "Muito fraca"
    
    def _interpret_effect_size(self, eta_squared: float) -> str:
        """Interpretar tamanho do efeito."""
        if eta_squared >= 0.14:
            return "Grande"
        elif eta_squared >= 0.06:
            return "Médio"
        elif eta_squared >= 0.01:
            return "Pequeno"
        else:
            return "Desprezível"
    
    def _generate_correlation_insights(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gerar insights de correlação."""
        insights = []
        
        if 'significant_correlations' in result:
            correlations = result['significant_correlations']
            if correlations:
                strongest = max(correlations.items(), key=lambda x: x[1])
                insights.append({
                    "type": "Correlação Forte",
                    "message": f"Correlação mais forte: {strongest[0]} ({strongest[1]:.3f})",
                    "impact": "high",
                    "recommendation": f"Investigar relação causal entre {strongest[0]} e vendas"
                })
                
        # Correlações por categoria
        if 'categorical_analysis' in result:
            cat_analysis = result['categorical_analysis']
            significant_categories = [cat for cat, data in cat_analysis.items() 
                                   if data.get('significant', False)]
            if significant_categories:
                insights.append({
                    "type": "Impacto Categórico",
                    "message": f"Categorias com impacto significativo: {', '.join(significant_categories)}",
                    "impact": "medium",
                    "recommendation": "Focar estratégias nestas categorias"
                })
        
        return insights
    
    def _generate_correlation_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Gerar recomendações baseadas em correlações."""
        recommendations = []
        
        if 'correlation_tests' in result:
            for var, test in result['correlation_tests'].items():
                if test['significant'] and test['correlation'] > 0.5:
                    recommendations.append(f"Otimizar {var} para aumentar vendas (correlação forte)")
        
        return recommendations[:5]
    
    def _multidimensional_clustering_analysis(self, df: pd.DataFrame, target_column: str = "Total_Liquido",
                                            clustering_method: str = "auto", **kwargs) -> Dict[str, Any]:
        """FASE 3: Análise de clustering multidimensional com cache otimizado."""
        
        # Verificar cache primeiro
        cache_key = f"clustering_{len(df)}_{target_column}_{clustering_method}"
        if cache_key in self._analysis_cache:
            print("📋 Usando resultado de clustering do cache")
            return self._analysis_cache[cache_key]
        
        try:
            print("🎯 Executando análise de clustering multidimensional...")
            
            # Preparar dados com amostragem se necessário
            if len(df) > 50000:
                print(f"📊 Dataset grande ({len(df)} registros) - usando amostra de 30.000 para clustering")
                df_sample = df.sample(n=30000, random_state=42)
            else:
                df_sample = df
            
            result = {
                'analysis_type': 'Multidimensional Clustering Analysis',
                'target_column': target_column,
                'method': clustering_method,
                'sample_size': len(df_sample),
                'original_size': len(df)
            }
            
            # Preparar dados para clustering
            feature_cols = self._select_clustering_features(df_sample, target_column)
            if len(feature_cols) < 2:
                return {'error': 'Insuficientes features numéricas para clustering'}
            
            X = df_sample[feature_cols].fillna(0)
            
            # Usar dados já escalados se disponíveis, senão escalar
            if any(col.endswith('_scaled') for col in feature_cols):
                X_scaled = X.values
                print("✅ Usando dados pré-escalados")
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                print("⚙️ Aplicando padronização aos dados")
            
            # Escolher método automaticamente se necessário
            if clustering_method == "auto":
                clustering_method = self._select_optimal_clustering_method(X_scaled)
                print(f"🤖 Método automático selecionado: {clustering_method}")
            
            # Executar clustering
            if clustering_method == "kmeans":
                cluster_result = self._perform_kmeans_clustering(X_scaled, df_sample, feature_cols)
            elif clustering_method == "hierarchical":
                cluster_result = self._perform_hierarchical_clustering(X_scaled, df_sample, feature_cols)
            elif clustering_method == "dbscan":
                cluster_result = self._perform_dbscan_clustering(X_scaled, df_sample, feature_cols)
            else:
                return {'error': f"Método de clustering '{clustering_method}' não suportado"}
            
            result.update(cluster_result)
            
            # Análise dos clusters
            if 'cluster_labels' in result:
                df_clustered = df_sample.copy()
                df_clustered['Cluster'] = result['cluster_labels']
                
                # Perfil dos clusters
                cluster_profiles = {}
                for cluster_id in df_clustered['Cluster'].unique():
                    if cluster_id != -1:  # Ignorar outliers do DBSCAN
                        cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
                        
                        profile = {
                            'size': len(cluster_data),
                            'size_percentage': round(len(cluster_data) / len(df_clustered) * 100, 1),
                            'avg_revenue': round(cluster_data[target_column].mean(), 2),
                            'total_revenue': round(cluster_data[target_column].sum(), 2),
                            'revenue_share': round(cluster_data[target_column].sum() / df_sample[target_column].sum() * 100, 1)
                        }
                        
                        # Características demográficas se disponíveis
                        if 'Faixa_Etaria' in cluster_data.columns:
                            profile['predominant_age_group'] = cluster_data['Faixa_Etaria'].mode().iloc[0] if len(cluster_data['Faixa_Etaria'].mode()) > 0 else 'N/A'
                        
                        if 'Sexo' in cluster_data.columns:
                            profile['gender_distribution'] = cluster_data['Sexo'].value_counts().to_dict()
                        
                        # Produtos preferidos
                        if 'Grupo_Produto' in cluster_data.columns:
                            profile['preferred_products'] = cluster_data['Grupo_Produto'].value_counts().head(3).to_dict()
                        
                        cluster_profiles[f'Cluster_{cluster_id}'] = profile
                
                result['cluster_profiles'] = cluster_profiles
            
            # Insights de clustering
            result['insights'] = self._generate_clustering_insights(result)
            result['business_recommendations'] = self._generate_clustering_recommendations(result)
            
            # Salvar no cache
            self._analysis_cache[cache_key] = result
            print(f"💾 Resultado salvo no cache (chave: {cache_key})")
            
            return result
            
        except Exception as e:
            return {'error': f"Erro na análise de clustering: {str(e)}"}
    
    def _select_clustering_features(self, df: pd.DataFrame, target_column: str) -> List[str]:
        """Selecionar features otimizadas para clustering."""
        # Priorizar features escaladas se disponíveis
        scaled_features = [col for col in df.columns if col.endswith('_scaled')]
        
        if scaled_features:
            print(f"🎯 Usando {len(scaled_features)} features escaladas para clustering")
            return scaled_features
        
        # Fallback para features numéricas básicas
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = [target_column, 'Data', 'Codigo_Cliente', 'Codigo_Produto', 'Ano', 'Mes']
        selected_features = [col for col in numeric_features[:10] 
                           if col not in exclude_cols and not df[col].isna().all()]
        
        return selected_features
    
    def _select_optimal_clustering_method(self, X_scaled: np.ndarray) -> str:
        """Selecionar método de clustering automaticamente."""
        n_samples = X_scaled.shape[0]
        
        if n_samples < 50:
            return "hierarchical"
        elif n_samples > 1000:
            return "kmeans"
        else:
            return "kmeans"  # fallback seguro
    
    def _perform_kmeans_clustering(self, X_scaled: np.ndarray, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """Executar clustering K-means."""
        optimal_k = min(5, len(X_scaled) // 10)  # Heurística simples
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        silhouette_avg = silhouette_score(X_scaled, cluster_labels) if len(set(cluster_labels)) > 1 else -1
        
        return {
            'method_used': 'K-Means',
            'optimal_k': optimal_k,
            'cluster_labels': cluster_labels,
            'silhouette_score': round(silhouette_avg, 3),
            'feature_names': feature_cols
        }
    
    def _perform_hierarchical_clustering(self, X_scaled: np.ndarray, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """Executar clustering hierárquico."""
        linkage_matrix = linkage(X_scaled, method='ward')
        optimal_clusters = min(5, len(X_scaled) // 10)
        cluster_labels = fcluster(linkage_matrix, optimal_clusters, criterion='maxclust') - 1
        
        silhouette_avg = silhouette_score(X_scaled, cluster_labels) if len(set(cluster_labels)) > 1 else -1
        
        return {
            'method_used': 'Hierarchical',
            'optimal_clusters': optimal_clusters,
            'cluster_labels': cluster_labels,
            'silhouette_score': round(silhouette_avg, 3),
            'feature_names': feature_cols
        }
    
    def _perform_dbscan_clustering(self, X_scaled: np.ndarray, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """Executar clustering DBSCAN."""
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_outliers = list(cluster_labels).count(-1)
        
        silhouette_avg = silhouette_score(X_scaled, cluster_labels) if n_clusters > 1 else -1
        
        return {
            'method_used': 'DBSCAN',
            'cluster_labels': cluster_labels,
            'n_clusters': n_clusters,
            'n_outliers': n_outliers,
            'silhouette_score': round(silhouette_avg, 3),
            'feature_names': feature_cols
        }
    
    def _generate_clustering_insights(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gerar insights de clustering."""
        insights = []
        
        if 'cluster_profiles' in result:
            profiles = result['cluster_profiles']
            
            # Cluster mais valioso
            revenue_clusters = {k: v.get('total_revenue', 0) for k, v in profiles.items()}
            if revenue_clusters:
                top_cluster = max(revenue_clusters, key=revenue_clusters.get)
                top_revenue_share = profiles[top_cluster].get('revenue_share', 0)
                insights.append({
                    "type": "Cluster Valioso",
                    "message": f"{top_cluster} gera {top_revenue_share}% da receita",
                    "impact": "high",
                    "recommendation": f"Focar estratégias de retenção no {top_cluster}"
                })
        
        return insights
    
    def _generate_clustering_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Gerar recomendações de clustering."""
        recommendations = []
        
        if 'cluster_profiles' in result:
            recommendations.append("Desenvolver estratégias específicas para cada cluster identificado")
            recommendations.append("Personalizar comunicação baseada no perfil do cluster")
        
        return recommendations
    
    def _comprehensive_outlier_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Análise de outliers usando múltiplos métodos."""
        try:
            print("🔍 Executando análise de outliers...")
            
            target_col = kwargs.get('target_column', 'Total_Liquido')
            if target_col not in df.columns:
                return {'error': f"Coluna {target_col} não encontrada"}
            
            # Método IQR
            Q1 = df[target_col].quantile(0.25)
            Q3 = df[target_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_iqr = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]
            
            # Método Z-score
            z_scores = np.abs(stats.zscore(df[target_col].dropna()))
            outliers_zscore = df[z_scores > 3]
            
            result = {
                'analysis_type': 'Comprehensive Outlier Analysis',
                'target_column': target_col,
                'iqr_method': {
                    'outliers_count': len(outliers_iqr),
                    'outliers_percentage': round(len(outliers_iqr) / len(df) * 100, 2),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2)
                },
                'zscore_method': {
                    'outliers_count': len(outliers_zscore),
                    'outliers_percentage': round(len(outliers_zscore) / len(df) * 100, 2)
                },
                'insights': [
                    {
                        "type": "Outliers Detectados",
                        "message": f"{len(outliers_iqr)} outliers identificados pelo método IQR",
                        "impact": "medium",
                        "recommendation": "Investigar transações anômalas para identificar oportunidades ou problemas"
                    }
                ]
            }
            
            return result
            
        except Exception as e:
            return {'error': f"Erro na análise de outliers: {str(e)}"}
    
    def _advanced_distribution_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Análise de distribuição com testes de normalidade."""
        try:
            print("📊 Executando análise de distribuição...")
            
            target_col = kwargs.get('target_column', 'Total_Liquido')
            if target_col not in df.columns:
                return {'error': f"Coluna {target_col} não encontrada"}
            
            data = df[target_col].dropna()
            
            # Teste de normalidade Shapiro-Wilk (para amostras pequenas)
            if len(data) <= 5000:
                shapiro_stat, shapiro_p = stats.shapiro(data.sample(min(5000, len(data))))
            else:
                shapiro_stat, shapiro_p = stats.shapiro(data.sample(5000))
            
            # Estatísticas descritivas
            result = {
                'analysis_type': 'Advanced Distribution Analysis',
                'target_column': target_col,
                'descriptive_stats': {
                    'mean': round(data.mean(), 2),
                    'median': round(data.median(), 2),
                    'std': round(data.std(), 2),
                    'skewness': round(stats.skew(data), 3),
                    'kurtosis': round(stats.kurtosis(data), 3)
                },
                'normality_test': {
                    'shapiro_statistic': round(shapiro_stat, 4),
                    'shapiro_p_value': round(shapiro_p, 4),
                    'is_normal': shapiro_p > 0.05
                },
                'insights': [
                    {
                        "type": "Distribuição",
                        "message": f"Dados {'seguem' if shapiro_p > 0.05 else 'não seguem'} distribuição normal",
                        "impact": "medium",
                        "recommendation": "Usar testes paramétricos" if shapiro_p > 0.05 else "Usar testes não-paramétricos"
                    }
                ]
            }
            
            return result
            
        except Exception as e:
            return {'error': f"Erro na análise de distribuição: {str(e)}"}
    
    def _temporal_trend_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Análise de tendência temporal."""
        try:
            print("📈 Executando análise de tendência temporal...")
            
            if 'Data' not in df.columns:
                return {'error': 'Coluna Data não encontrada'}
            
            target_col = kwargs.get('target_column', 'Total_Liquido')
            
            # Agregar por mês
            monthly_data = df.groupby(df['Data'].dt.to_period('M'))[target_col].sum()
            
            # Teste de tendência Mann-Kendall
            from scipy.stats import kendalltau
            x = range(len(monthly_data))
            tau, p_value = kendalltau(x, monthly_data.values)
            
            result = {
                'analysis_type': 'Temporal Trend Analysis',
                'target_column': target_col,
                'trend_test': {
                    'kendall_tau': round(tau, 3),
                    'p_value': round(p_value, 4),
                    'has_trend': p_value < 0.05,
                    'trend_direction': 'crescente' if tau > 0 else 'decrescente' if tau < 0 else 'estável'
                },
                'monthly_summary': {
                    'periods': len(monthly_data),
                    'avg_monthly': round(monthly_data.mean(), 2),
                    'growth_rate': round(monthly_data.pct_change().mean() * 100, 2)
                },
                'insights': [
                    {
                        "type": "Tendência Temporal",
                        "message": f"Tendência {result['trend_test']['trend_direction']} detectada",
                        "impact": "high" if p_value < 0.05 else "low",
                        "recommendation": "Ajustar estratégias baseado na tendência identificada"
                    }
                ]
            }
            
            return result
            
        except Exception as e:
            return {'error': f"Erro na análise temporal: {str(e)}"}
    
    def _demographic_patterns_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Análise de padrões demográficos."""
        try:
            print("👥 Executando análise demográfica...")
            
            target_col = kwargs.get('target_column', 'Total_Liquido')
            
            result = {
                'analysis_type': 'Demographic Patterns Analysis',
                'target_column': target_col
            }
            
            # Análise por sexo
            if 'Sexo' in df.columns:
                gender_stats = df.groupby('Sexo')[target_col].agg(['count', 'mean', 'sum']).round(2)
                result['gender_analysis'] = gender_stats.to_dict()
            
            # Análise por faixa etária
            if 'Faixa_Etaria' in df.columns:
                age_stats = df.groupby('Faixa_Etaria')[target_col].agg(['count', 'mean', 'sum']).round(2)
                result['age_analysis'] = age_stats.to_dict()
            
            result['insights'] = [
                {
                    "type": "Demografia",
                    "message": "Padrões demográficos identificados",
                    "impact": "medium",
                    "recommendation": "Personalizar ofertas por segmento demográfico"
                }
            ]
            
            return result
            
        except Exception as e:
            return {'error': f"Erro na análise demográfica: {str(e)}"}
    
    def _generational_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Análise geracional."""
        return {'message': 'Análise geracional em desenvolvimento', 'status': 'placeholder'}
    
    def _behavioral_customer_segmentation(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Segmentação comportamental de clientes."""
        return {'message': 'Segmentação comportamental em desenvolvimento', 'status': 'placeholder'}
    
    def _geographic_performance_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Análise de performance geográfica."""
        return {'message': 'Análise geográfica em desenvolvimento', 'status': 'placeholder'}
    
    def _regional_patterns_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Análise de padrões regionais."""
        return {'message': 'Análise de padrões regionais em desenvolvimento', 'status': 'placeholder'}
    
    def _price_elasticity_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Análise de elasticidade de preços."""
        return {'message': 'Análise de elasticidade em desenvolvimento', 'status': 'placeholder'}
    
    def _profitability_pattern_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Análise de padrões de rentabilidade."""
        return {'message': 'Análise de rentabilidade em desenvolvimento', 'status': 'placeholder'}
    
    def _comprehensive_customer_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Análise abrangente de clientes."""
        return {'message': 'Análise abrangente de clientes em desenvolvimento', 'status': 'placeholder'}
    
    def _statistical_product_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Análise estatística de produtos."""
        return {'message': 'Análise estatística de produtos em desenvolvimento', 'status': 'placeholder'} 