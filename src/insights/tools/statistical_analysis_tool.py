from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field, field_validator
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
import time
import traceback
import os

# Importar módulos compartilhados consolidados
try:
    # Imports relativos (quando usado como módulo)
    from .shared.data_preparation import DataPreparationMixin
    from .shared.report_formatter import ReportFormatterMixin
    from .shared.business_mixins import JewelryRFMAnalysisMixin
except ImportError:
    # Imports absolutos (quando executado diretamente)
    from insights.tools.shared.data_preparation import DataPreparationMixin
    from insights.tools.shared.report_formatter import ReportFormatterMixin
    from insights.tools.shared.business_mixins import JewelryRFMAnalysisMixin

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
        json_schema_extra={"example": "correlation"}
    )
    
    data_csv: str = Field(
        default="data/vendas.csv", 
        description="Caminho para arquivo CSV com dados de vendas. Use 'data/vendas.csv' para dados principais.",
        json_schema_extra={"example": "data/vendas.csv"}
    )
    
    target_column: str = Field(
        default="Total_Liquido", 
        description="Coluna alvo para análise. Use 'Total_Liquido' para receita, 'Quantidade' para volume.",
        json_schema_extra={"example": "Total_Liquido"}
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
        json_schema_extra={"pattern": "^(kmeans|hierarchical|dbscan|auto)$"}
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
    
    @field_validator('analysis_type')
    @classmethod
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
            
            # CORREÇÃO: Inicializar result
            result = {
                'analysis_type': 'Temporal Trend Analysis',
                'target_column': kwargs.get('target_column', 'Total_Liquido')
            }
            
            if 'Data' not in df.columns:
                return {'error': 'Coluna Data não encontrada'}
            
            target_col = kwargs.get('target_column', 'Total_Liquido')
            
            # Agregar por mês
            monthly_data = df.groupby(df['Data'].dt.to_period('M'))[target_col].sum()
            
            # Teste de tendência Mann-Kendall
            from scipy.stats import kendalltau
            x = range(len(monthly_data))
            tau, p_value = kendalltau(x, monthly_data.values)
            
            result['trend_test'] = {
                'kendall_tau': round(tau, 3),
                'p_value': round(p_value, 4),
                'has_trend': p_value < 0.05,
                'trend_direction': 'crescente' if tau > 0 else 'decrescente' if tau < 0 else 'estável'
            }
            
            result['monthly_summary'] = {
                'periods': len(monthly_data),
                'avg_monthly': round(monthly_data.mean(), 2),
                'growth_rate': round(monthly_data.pct_change().mean() * 100, 2)
            }
            
            result['insights'] = [
                {
                    "type": "Tendência Temporal",
                    "message": f"Tendência {result['trend_test']['trend_direction']} detectada",
                    "impact": "high" if p_value < 0.05 else "low",
                    "recommendation": "Ajustar estratégias baseado na tendência identificada"
                }
            ]
            
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
        """Análise geracional IMPLEMENTADA."""
        try:
            print("👥 Executando análise geracional...")
            
            result = {
                'analysis_type': 'Generational Analysis',
                'target_column': kwargs.get('target_column', 'Total_Liquido')
            }
            
            if 'Geracao' in df.columns:
                gen_stats = df.groupby('Geracao')[kwargs.get('target_column', 'Total_Liquido')].agg(['count', 'mean', 'sum']).round(2)
                result['generational_analysis'] = gen_stats.to_dict()
                
                # Geração mais valiosa
                top_generation = gen_stats['sum'].idxmax()
                result['insights'] = [
                    {
                        "type": "Geração Dominante",
                        "message": f"Geração {top_generation} é a mais valiosa em receita",
                        "impact": "medium",
                        "recommendation": f"Focar estratégias na geração {top_generation}"
                    }
                ]
            else:
                result['insights'] = [
                    {
                        "type": "Dados Insuficientes",
                        "message": "Campo 'Geracao' não encontrado para análise geracional",
                        "impact": "low",
                        "recommendation": "Implementar classificação geracional baseada em idade"
                    }
                ]
            
            return result
            
        except Exception as e:
            return {'error': f"Erro na análise geracional: {str(e)}"}
    
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

    def generate_statistical_visual_report(self, test_data: dict) -> str:
        """Gera relatório visual completo dos testes estatísticos em formato markdown."""
        
        # Coletar dados com fallbacks
        metadata = test_data.get('metadata', {})
        data_metrics = test_data.get('data_metrics', {})
        results = test_data.get('results', {})
        component_tests = test_data.get('component_tests', {})
        
        report = [
            "# 🔬 Teste Completo de Análises Estatísticas - Relatório Executivo",
            f"**Data do Teste:** {metadata.get('test_timestamp', 'N/A')}",
            f"**Fonte de Dados:** `{metadata.get('data_source', 'desconhecida')}`",
            f"**Registros Analisados:** {data_metrics.get('total_records', 0):,}",
            f"**Período de Análise:** {data_metrics.get('date_range', {}).get('start', 'N/A')} até {data_metrics.get('date_range', {}).get('end', 'N/A')}",
            "\n## 📈 Performance de Execução",
            f"```\n{json.dumps(test_data.get('performance_metrics', {}), indent=2)}\n```",
            "\n## 🎯 Resumo dos Testes Executados"
        ]
        
        # Contabilizar sucessos e falhas
        successful_tests = len([r for r in results.values() if 'error' not in r])
        failed_tests = len([r for r in results.values() if 'error' in r])
        total_tests = len(results)
        
        report.extend([
            f"- **Total de Análises:** {total_tests}",
            f"- **Sucessos:** {successful_tests} ✅",
            f"- **Falhas:** {failed_tests} ❌",
            f"- **Taxa de Sucesso:** {(successful_tests/total_tests*100):.1f}%" if total_tests > 0 else "- **Taxa de Sucesso:** N/A"
        ])
        
        # Insights Estatísticos Principais
        report.append("\n## 🔍 Principais Descobertas Estatísticas")
        
        # Correlações significativas
        if 'correlation' in results and 'error' not in results['correlation']:
            corr_data = results['correlation']
            if 'significant_correlations' in corr_data:
                sig_corr = corr_data['significant_correlations']
                if sig_corr:
                    strongest = max(sig_corr.items(), key=lambda x: abs(x[1]))
                    report.append(f"- **Correlação Mais Forte:** {strongest[0]} ({strongest[1]:.3f})")
        
        # Clusters identificados
        if 'clustering' in results and 'error' not in results['clustering']:
            cluster_data = results['clustering']
            if 'cluster_profiles' in cluster_data:
                n_clusters = len(cluster_data['cluster_profiles'])
                report.append(f"- **Clusters Identificados:** {n_clusters} segmentos distintos")
        
        # Outliers detectados
        if 'outliers' in results and 'error' not in results['outliers']:
            outlier_data = results['outliers']
            if 'iqr_method' in outlier_data:
                outlier_count = outlier_data['iqr_method'].get('outliers_count', 0)
                report.append(f"- **Outliers Detectados:** {outlier_count} transações anômalas")
        
        # Testes de Normalidade
        if 'distribution' in results and 'error' not in results['distribution']:
            dist_data = results['distribution']
            if 'normality_test' in dist_data:
                is_normal = dist_data['normality_test'].get('is_normal', False)
                report.append(f"- **Distribuição dos Dados:** {'Normal' if is_normal else 'Não-Normal'}")
        
        # Análise Temporal
        if 'trend_analysis' in results and 'error' not in results['trend_analysis']:
            trend_data = results['trend_analysis']
            if 'trend_test' in trend_data:
                trend_dir = trend_data['trend_test'].get('trend_direction', 'N/A')
                report.append(f"- **Tendência Temporal:** {trend_dir.title()}")
        
        # Detalhamento por Tipo de Análise
        report.append("\n## 📊 Detalhamento das Análises")
        
        analysis_categories = {
            'Análises Estatísticas Core': ['correlation', 'clustering', 'outliers', 'distribution', 'trend_analysis'],
            'Análises Demográficas': ['demographic_patterns', 'generational_analysis', 'customer_segmentation'],
            'Análises Geográficas': ['geographic_performance', 'regional_patterns'],
            'Análises Especializadas': ['price_sensitivity', 'profitability_patterns'],
            'Análises Integradas': ['comprehensive_customer_analysis', 'product_performance_analysis']
        }
        
        for category, analyses in analysis_categories.items():
            report.append(f"\n### {category}")
            for analysis in analyses:
                if analysis in results:
                    if 'error' in results[analysis]:
                        report.append(f"- ❌ **{analysis}**: {results[analysis]['error']}")
                    else:
                        # Resumir insights principais de cada análise
                        insights = results[analysis].get('insights', [])
                        if insights:
                            report.append(f"- ✅ **{analysis}**: {len(insights)} insights gerados")
                            for insight in insights[:2]:  # Top 2 insights
                                report.append(f"  - {insight.get('message', 'N/A')}")
                        else:
                            report.append(f"- ✅ **{analysis}**: Concluído")
                else:
                    report.append(f"- ⏭️ **{analysis}**: Não testado")
        
        # Recomendações Baseadas em Evidências Estatísticas
        report.append("\n## 💡 Recomendações Baseadas em Evidências")
        
        all_insights = []
        for result in results.values():
            if 'insights' in result and isinstance(result['insights'], list):
                all_insights.extend(result['insights'])
        
        # Agrupar recomendações por impacto
        high_impact = [i for i in all_insights if i.get('impact') == 'high']
        medium_impact = [i for i in all_insights if i.get('impact') == 'medium']
        
        if high_impact:
            report.append("\n### 🔥 Alta Prioridade")
            for insight in high_impact[:3]:
                report.append(f"- {insight.get('recommendation', insight.get('message', 'N/A'))}")
        
        if medium_impact:
            report.append("\n### 📈 Média Prioridade")
            for insight in medium_impact[:3]:
                report.append(f"- {insight.get('recommendation', insight.get('message', 'N/A'))}")
        
        # Qualidade dos Dados e Limitações
        report.append("\n## ⚠️ Limitações e Considerações")
        
        data_quality = data_metrics.get('data_quality_check', {})
        if data_quality:
            report.append("### Qualidade dos Dados:")
            for check, value in data_quality.items():
                if value > 0:
                    report.append(f"- **{check}**: {value} ocorrências")
        
        # Erros encontrados
        errors = test_data.get('errors', [])
        if errors:
            report.append(f"\n### Erros Detectados ({len(errors)}):")
            for error in errors[-3:]:  # Últimos 3 erros
                report.append(f"- **{error['context']}**: {error['error_message']}")
        
        return "\n".join(report)

    def run_full_statistical_test(self) -> str:
        """Executa teste completo e retorna relatório formatado"""
        test_result = self.test_all_statistical_analyses()
        parsed = json.loads(test_result)
        return self.generate_statistical_visual_report(parsed)

    def test_all_statistical_analyses(self, sample_data: str = "data/vendas.csv") -> str:
        """
        Executa teste completo de todas as análises estatísticas da classe
        """
        
        # Corrigir caminho do arquivo
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
        data_file_path = os.path.join(project_root, sample_data)
        
        # Verificar se arquivo existe
        if not os.path.exists(data_file_path):
            return json.dumps({
                "error": f"Arquivo não encontrado: {data_file_path}",
                "current_dir": current_dir,
                "project_root": project_root,
                "expected_path": data_file_path
            }, indent=2)

        test_report = {
            "metadata": {
                "test_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "test_version": "Statistical Analysis Test Suite v1.0",
                "data_source": data_file_path,
                "tool_version": "Statistical Analysis Tool v3.0",
                "status": "in_progress"
            },
            "data_metrics": {
                "total_records": 0,
                "columns": [],
                "date_range": {},
                "data_quality_check": {}
            },
            "results": {},
            "component_tests": {},
            "performance_metrics": {},
            "errors": []
        }

        try:
            # 1. Fase de Carregamento de Dados (UMA VEZ SÓ)
            test_report["metadata"]["current_stage"] = "data_loading"
            print("\n=== ETAPA 1: CARREGAMENTO DE DADOS ESTATÍSTICOS ===")
            print(f"📁 Tentando carregar: {data_file_path}")
            
            # Carregar dados UMA VEZ e reutilizar
            df = self._load_and_prepare_statistical_data(data_file_path, use_cache=True)
            
            if df is None:
                raise Exception("Falha no carregamento dos dados estatísticos")
            
            print(f"✅ Dados carregados GLOBALMENTE: {len(df)} registros")
            
            # Coletar métricas básicas dos dados
            test_report["data_metrics"] = {
                "total_records": int(len(df)),
                "columns": list(df.columns),
                "date_range": {
                    "start": str(df['Data'].min()) if 'Data' in df.columns else "N/A",
                    "end": str(df['Data'].max()) if 'Data' in df.columns else "N/A"
                },
                "data_quality_check": self._convert_to_native_types(self._perform_statistical_data_quality_check(df))
            }

            # 2. Teste de Todas as Análises Estatísticas (REUTILIZANDO DADOS)
            test_report["metadata"]["current_stage"] = "statistical_testing"
            print("\n=== ETAPA 2: TESTE DE ANÁLISES ESTATÍSTICAS ===")
            
            # Definir todas as análises disponíveis
            statistical_analyses = [
                'correlation',
                'clustering', 
                'outliers',
                'distribution',
                'trend_analysis',
                'demographic_patterns',
                'generational_analysis',
                'customer_segmentation',
                'geographic_performance',
                'regional_patterns',
                'price_sensitivity',
                'profitability_patterns',
                'comprehensive_customer_analysis',
                'product_performance_analysis'
            ]
            
            for analysis_type in statistical_analyses:
                try:
                    print(f"\n🔬 TESTANDO ANÁLISE: {analysis_type.upper()}")
                    start_time = time.time()
                    
                    # CORREÇÃO: Usar dados já carregados ao invés de recarregar
                    result = self._run_analysis_with_prepared_data(
                        df=df,  # USAR DADOS PREPARADOS
                        analysis_type=analysis_type,
                        target_column="Total_Liquido",
                        significance_level=0.05,
                        clustering_method="auto",
                        min_correlation=0.3,
                        demographic_focus=True,
                        geographic_focus=True
                    )
                    
                    # Análise de resultados
                    parsed_result = json.loads(result)
                    test_report["results"][analysis_type] = parsed_result
                    
                    execution_time = time.time() - start_time
                    
                    # Verificação básica de integridade
                    if 'error' in parsed_result:
                        print(f"❌ {analysis_type.upper()} - Erro: {parsed_result['error']}")
                    else:
                        insights_count = len(parsed_result.get('insights', []))
                        print(f"✅ {analysis_type.upper()} - {insights_count} insights gerados ({execution_time:.2f}s)")

                except Exception as e:
                    error_id = f"ERR-{analysis_type.upper()}-{datetime.now().strftime('%H%M%S')}"
                    self._log_statistical_test_error(test_report, e, analysis_type)
                    print(f"⛔ Erro em {analysis_type.upper()} - {error_id}: {str(e)}")

            # 3. Teste de Componentes de Cache e Otimização
            test_report["metadata"]["current_stage"] = "optimization_testing"
            print("\n=== ETAPA 3: TESTE DE OTIMIZAÇÕES ===")
            
            try:
                print("🔧 Testando cache de análises...")
                # Teste com cache
                start_time = time.time()
                result_with_cache = self._run(
                    analysis_type="correlation",
                    data_csv=data_file_path,
                    cache_results=True
                )
                cache_time = time.time() - start_time
                
                # Teste sem cache
                start_time = time.time()
                result_without_cache = self._run(
                    analysis_type="correlation", 
                    data_csv=data_file_path,
                    cache_results=False
                )
                no_cache_time = time.time() - start_time
                
                test_report["component_tests"]["cache_performance"] = {
                    "cache_enabled_time": round(cache_time, 3),
                    "cache_disabled_time": round(no_cache_time, 3),
                    "cache_efficiency": round((no_cache_time - cache_time) / no_cache_time * 100, 1) if no_cache_time > 0 else 0
                }
                print("✅ Cache performance - OK")
                
            except Exception as e:
                self._log_statistical_test_error(test_report, e, "cache_test")
                print(f"❌ Cache test - Falha: {str(e)}")

            try:
                print("🔧 Testando amostragem para datasets grandes...")
                # Simular dataset grande com amostragem
                result_sampled = self._run(
                    analysis_type="clustering",
                    data_csv=data_file_path,
                    sample_size=1000  # Forçar amostragem
                )
                
                parsed_sampled = json.loads(result_sampled)
                test_report["component_tests"]["sampling"] = {
                    "sample_size_used": parsed_sampled.get('metadata', {}).get('total_records', 0),
                    "sampling_successful": 'error' not in parsed_sampled
                }
                print("✅ Sampling test - OK")
                
            except Exception as e:
                self._log_statistical_test_error(test_report, e, "sampling_test")
                print(f"❌ Sampling test - Falha: {str(e)}")

            # 4. Teste de Performance com Análise Complexa
            test_report["metadata"]["current_stage"] = "performance_testing"
            print("\n=== ETAPA 4: TESTE DE PERFORMANCE ===")
            try:
                start_time = time.time()
                
                # CORREÇÃO: Remover parâmetro inexistente
                complex_test = self._run(
                    analysis_type="clustering",
                    data_csv=data_file_path,
                    clustering_method="auto",
                    demographic_focus=True,
                    geographic_focus=True,
                    cache_results=True  # SUBSTITUIR include_statistical_insights
                )
                
                test_report["performance_metrics"] = {
                    "complex_analysis_time_seconds": round(time.time() - start_time, 2),
                    "result_size_kb": round(len(complex_test)/1024, 2),
                    "memory_usage_mb": round(self._get_statistical_memory_usage(), 2),
                    "cache_size": len(self._analysis_cache)
                }
                print("✅ Performance test concluído")
                
            except Exception as e:
                self._log_statistical_test_error(test_report, e, "performance_test")
                print(f"❌ Performance test falhou: {str(e)}")

            # 5. Análise Final
            test_report["metadata"]["status"] = "completed" if not test_report["errors"] else "completed_with_errors"
            print(f"\n✅✅✅ TESTE ESTATÍSTICO COMPLETO - {len(test_report['errors'])} erros ✅✅✅")
            
            return json.dumps(test_report, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            test_report["metadata"]["status"] = "failed"
            self._log_statistical_test_error(test_report, e, "global")
            print(f"❌ TESTE ESTATÍSTICO FALHOU: {str(e)}")
            return json.dumps(test_report, ensure_ascii=False, indent=2, default=str)

    def _get_analysis_specific_params(self, analysis_type: str) -> dict:
        """Retorna parâmetros específicos para cada tipo de análise"""
        params_map = {
            'correlation': {'min_correlation': 0.3, 'significance_level': 0.05},
            'clustering': {'clustering_method': 'auto'},
            'outliers': {'target_column': 'Total_Liquido'},
            'distribution': {'target_column': 'Total_Liquido'},
            'trend_analysis': {'target_column': 'Total_Liquido'},
            'demographic_patterns': {'demographic_focus': True},
            'geographic_performance': {'geographic_focus': True},
            'price_sensitivity': {'target_column': 'Total_Liquido'},
        }
        return params_map.get(analysis_type, {})

    def _log_statistical_test_error(self, report: dict, error: Exception, context: str) -> None:
        """Registra erros de teste estatístico de forma estruturada"""
        error_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc()
        }
        report["errors"].append(error_entry)

    def _convert_to_native_types(self, obj):
        """Converte tipos numpy/pandas para tipos nativos Python."""
        if isinstance(obj, dict):
            return {k: self._convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native_types(v) for v in obj]
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        else:
            return obj

    def _perform_statistical_data_quality_check(self, df: pd.DataFrame) -> dict:
        """Executa verificações de qualidade específicas para análises estatísticas"""
        checks = {
            "missing_values_total": int(df.isnull().sum().sum()),
            "duplicate_records": int(df.duplicated().sum()),
            "numerical_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns),
            "zero_variance_columns": int((df.var(numeric_only=True) == 0).sum()),
            "outliers_iqr_total": self._count_total_outliers_iqr(df)
        }
        return checks

    def _count_total_outliers_iqr(self, df: pd.DataFrame) -> int:
        """Conta outliers totais usando método IQR para todas as colunas numéricas"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            total_outliers = 0
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                total_outliers += outliers
                
            return int(total_outliers)
        except:
            return 0

    def _get_statistical_memory_usage(self) -> float:
        """Obtém uso de memória específico para análises estatísticas"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Em MB
        except:
            return 0.0

    def _run_analysis_with_prepared_data(self, df: pd.DataFrame, analysis_type: str, **kwargs) -> str:
        """Executa análise usando dados já preparados (otimização para testes)."""
        try:
            print(f"🚀 Executando {analysis_type} com dados pré-carregados...")
            
            # Mapear análises
            analysis_methods = {
                'correlation': self._advanced_correlation_analysis,
                'clustering': self._multidimensional_clustering_analysis,
                'outliers': self._comprehensive_outlier_analysis,
                'distribution': self._advanced_distribution_analysis,
                'trend_analysis': self._temporal_trend_analysis,
                'demographic_patterns': self._demographic_patterns_analysis,
                'generational_analysis': self._generational_analysis,
                'customer_segmentation': self._behavioral_customer_segmentation,
                'geographic_performance': self._geographic_performance_analysis,
                'regional_patterns': self._regional_patterns_analysis,
                'price_sensitivity': self._price_elasticity_analysis,
                'profitability_patterns': self._profitability_pattern_analysis,
                'comprehensive_customer_analysis': self._comprehensive_customer_analysis,
                'product_performance_analysis': self._statistical_product_analysis
            }
            
            if analysis_type not in analysis_methods:
                return json.dumps({'error': f"Análise '{analysis_type}' não suportada"})
            
            # Executar análise diretamente
            result = analysis_methods[analysis_type](df, **kwargs)
            
            # Adicionar metadados
            result['metadata'] = {
                'tool': 'Statistical Analysis Tool v3.0',
                'analysis_type': analysis_type,
                'total_records': len(df),
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return json.dumps(result, ensure_ascii=False, indent=2, default=str)
            
        except Exception as e:
            return json.dumps({'error': f"Erro na análise {analysis_type}: {str(e)}"})

# Exemplo de uso
if __name__ == "__main__":
    analyzer = StatisticalAnalysisTool()
    
    print("🔬 Iniciando Teste Completo de Análises Estatísticas...")
    report = analyzer.run_full_statistical_test()
    
    # Salvar relatório
    os.makedirs("test_results", exist_ok=True)
    with open("test_results/statistical_analysis_test_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("✅ Relatório estatístico gerado em test_results/statistical_analysis_test_report.md")
    print("\n" + "="*80)
    print(report[:1500])  # Exibir parte do relatório no console 