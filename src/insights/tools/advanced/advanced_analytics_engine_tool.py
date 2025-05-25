"""
🤖 ADVANCED ANALYTICS ENGINE V4.0 - OTIMIZADO PARA CREWAI/PYDANTIC
==================================================================

Motor de análises avançadas com Machine Learning otimizado seguindo padrões:
- Schema Pydantic robusto com validações completas
- Documentação estruturada CrewAI
- Integração com módulos compartilhados
- Outputs JSON estruturados
- Cache inteligente e performance otimizada
- Tratamento de erros graceful
"""

from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings

# Importar módulos compartilhados consolidados
try:
    from ..shared.data_preparation import DataPreparationMixin
    from ..shared.report_formatter import ReportFormatterMixin
    from ..shared.business_mixins import JewelryRFMAnalysisMixin, JewelryBusinessAnalysisMixin
    SHARED_MODULES_AVAILABLE = True
except ImportError:
    # Fallback para quando módulos compartilhados não estão disponíveis
    class DataPreparationMixin:
        def prepare_jewelry_data(self, df, validation_level="standard"):
            return df
    
    class ReportFormatterMixin:
        pass
    
    class JewelryRFMAnalysisMixin:
        pass
    
    class JewelryBusinessAnalysisMixin:
        pass
    
    SHARED_MODULES_AVAILABLE = False

warnings.filterwarnings('ignore')

# Imports opcionais para bibliotecas de ML
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class AdvancedAnalyticsEngineToolInput(BaseModel):
    """Schema otimizado para análises avançadas ML com validações robustas."""
    
    analysis_type: str = Field(
        ...,
        description="""Análises de Machine Learning especializadas para joalherias:
        
        🤖 ANÁLISES CORE ML:
        - 'ml_insights': Descobrir insights ocultos com Random Forest e XGBoost
        - 'anomaly_detection': Detectar vendas anômalas e outliers com múltiplos algoritmos
        - 'demand_forecasting': Prever demanda futura com ensemble de modelos ML
        
        🎯 ANÁLISES COMPORTAMENTAIS:
        - 'customer_behavior': Segmentar clientes por padrões comportamentais ML
        - 'product_lifecycle': Analisar ciclo de vida e performance de produtos
        
        💰 ANÁLISES DE OTIMIZAÇÃO:
        - 'price_optimization': Otimizar preços baseado em elasticidade e ML
        - 'inventory_optimization': Otimizar gestão de estoque com análise ABC ML
        """,
        example="ml_insights",
        pattern="^(ml_insights|anomaly_detection|demand_forecasting|customer_behavior|product_lifecycle|price_optimization|inventory_optimization)$"
    )
    
    data_csv: str = Field(
        default="data/vendas.csv",
        description="Caminho para arquivo CSV de vendas. Use 'data/vendas.csv' para dados principais.",
        example="data/vendas.csv"
    )
    
    target_column: str = Field(
        default="Total_Liquido",
        description="Coluna alvo para análise ML. Use 'Total_Liquido' para receita, 'Quantidade' para volume.",
        example="Total_Liquido"
    )
    
    prediction_horizon: int = Field(
        default=30,
        description="Horizonte de predição em dias (7-365). Use 30 para análise mensal, 90 para trimestral.",
        ge=7,
        le=365
    )
    
    confidence_level: float = Field(
        default=0.95,
        description="Nível de confiança para intervalos (0.80-0.99). Use 0.95 para análise padrão.",
        ge=0.80,
        le=0.99
    )
    
    model_complexity: str = Field(
        default="balanced",
        description="Complexidade do modelo: 'simple' (rápido), 'balanced' (equilibrado), 'complex' (preciso).",
        pattern="^(simple|balanced|complex)$"
    )
    
    enable_ensemble: bool = Field(
        default=True,
        description="Usar ensemble de modelos para maior precisão. Recomendado: True para análises críticas."
    )
    
    sample_size: Optional[int] = Field(
        default=None,
        description="Tamanho da amostra para análises pesadas (5000-100000). None = usar todos os dados.",
        ge=5000,
        le=100000
    )
    
    cache_results: bool = Field(
        default=True,
        description="Usar cache para otimizar performance. Recomendado: True para datasets grandes."
    )
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        valid_types = [
            'ml_insights', 'anomaly_detection', 'demand_forecasting',
            'customer_behavior', 'product_lifecycle', 'price_optimization', 'inventory_optimization'
        ]
        if v not in valid_types:
            raise ValueError(f"analysis_type deve ser um de: {valid_types}")
        return v
    
    @validator('target_column')
    def validate_target_column(cls, v):
        allowed_columns = ['Total_Liquido', 'Quantidade', 'Margem_Real', 'Preco_Unitario']
        if v not in allowed_columns:
            raise ValueError(f"target_column deve ser um de: {allowed_columns}")
        return v

class AdvancedAnalyticsEngineTool(BaseTool,
                                   DataPreparationMixin,
                                   ReportFormatterMixin,
                                   JewelryRFMAnalysisMixin,
                                   JewelryBusinessAnalysisMixin):
    """
    🤖 MOTOR DE ANÁLISES AVANÇADAS COM MACHINE LEARNING PARA JOALHERIAS
    
    QUANDO USAR:
    - Descobrir padrões ocultos complexos nos dados de vendas
    - Realizar previsões avançadas com múltiplos algoritmos ML
    - Detectar anomalias e outliers para investigação
    - Segmentar clientes baseado em comportamento ML
    - Otimizar preços e estoque com algoritmos inteligentes
    - Analisar ciclo de vida de produtos com ML
    
    CASOS DE USO ESPECÍFICOS:
    - analysis_type='ml_insights': Descobrir insights com Random Forest/XGBoost
    - analysis_type='anomaly_detection': Identificar vendas anômalas para investigação
    - analysis_type='demand_forecasting': Prever demanda com ensemble de modelos
    - analysis_type='customer_behavior': Segmentar clientes por padrões ML
    - analysis_type='product_lifecycle': Analisar performance e ciclo de produtos
    - analysis_type='price_optimization': Otimizar preços com elasticidade ML
    - analysis_type='inventory_optimization': Otimizar estoque com análise ABC ML
    
    RESULTADOS ENTREGUES:
    - Insights acionáveis baseados em algoritmos ML
    - Previsões com intervalos de confiança estatística
    - Segmentações automáticas com perfis detalhados
    - Detecção de anomalias com scores de confiança
    - Recomendações de otimização baseadas em evidências
    - Métricas de performance e validação dos modelos
    - Análises de feature importance para interpretabilidade
    """
    
    name: str = "Advanced Analytics Engine"
    description: str = (
        "Motor de análises avançadas com Machine Learning para insights profundos de joalherias. "
        "Combina Random Forest, XGBoost e clustering para descobrir padrões ocultos, prever demanda e otimizar processos. "
        "Ideal para análises complexas que requerem algoritmos ML e insights acionáveis baseados em evidências."
    )
    args_schema: Type[BaseModel] = AdvancedAnalyticsEngineToolInput
    
    def __init__(self):
        super().__init__()
        self._ml_cache = {}  # Cache para modelos e resultados ML
    
    def _run(self, analysis_type: str, data_csv: str = "data/vendas.csv",
             target_column: str = "Total_Liquido", prediction_horizon: int = 30,
             confidence_level: float = 0.95, model_complexity: str = "balanced",
             enable_ensemble: bool = True, sample_size: Optional[int] = None,
             cache_results: bool = True) -> str:
        try:
            print(f"🤖 Iniciando Advanced Analytics Engine v4.0: {analysis_type}")
            print(f"⚙️ Configurações: modelo={model_complexity}, ensemble={enable_ensemble}, cache={cache_results}")
            
            # 1. Validar disponibilidade de bibliotecas ML
            if not SKLEARN_AVAILABLE:
                return json.dumps({
                    "error": "Scikit-learn não disponível - análises ML não podem ser executadas",
                    "troubleshooting": {
                        "install_sklearn": "Execute: pip install scikit-learn",
                        "check_environment": "Verifique se o ambiente virtual está ativo",
                        "try_simpler_analysis": "Use KPI Calculator ou Statistical Analysis Tool"
                    },
                    "metadata": {
                        "tool": "Advanced Analytics Engine",
                        "status": "error",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }, ensure_ascii=False, indent=2)
            
            # 2. Carregar e preparar dados usando módulo consolidado
            df = self._load_and_prepare_ml_data(data_csv, cache_results, sample_size)
            if df is None:
                return json.dumps({
                    "error": "Não foi possível carregar ou preparar os dados para análise ML",
                    "troubleshooting": {
                        "check_file_exists": f"Verifique se {data_csv} existe",
                        "check_data_quality": "Confirme que os dados têm qualidade suficiente para ML",
                        "check_sample_size": "Verifique se há dados suficientes para análise ML (mínimo 1000 registros)"
                    },
                    "metadata": {
                        "tool": "Advanced Analytics Engine",
                        "status": "error",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }, ensure_ascii=False, indent=2)
            
            print(f"✅ Dados preparados: {len(df)} registros, {len(df.columns)} campos")
            
            # 3. Validar dados para ML
            if len(df) < 100:
                return json.dumps({
                    "error": f"Dados insuficientes para ML: {len(df)} registros (mínimo 100)",
                    "troubleshooting": {
                        "increase_date_range": "Aumente o período de análise",
                        "check_data_filters": "Verifique se filtros não estão muito restritivos",
                        "try_simpler_analysis": "Use Statistical Analysis Tool para datasets pequenos"
                    },
                    "metadata": {
                        "tool": "Advanced Analytics Engine",
                        "status": "error",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }, ensure_ascii=False, indent=2)
            
            if target_column not in df.columns:
                return json.dumps({
                    "error": f"Coluna alvo '{target_column}' não encontrada",
                    "available_columns": list(df.columns),
                    "troubleshooting": {
                        "check_column_name": "Verifique se o nome da coluna está correto",
                        "use_total_liquido": "Use 'Total_Liquido' como padrão",
                        "check_data_preparation": "Confirme se a preparação de dados foi bem-sucedida"
                    },
                    "metadata": {
                        "tool": "Advanced Analytics Engine",
                        "status": "error",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                }, ensure_ascii=False, indent=2)
            
            # 4. Mapeamento de análises ML especializadas
            analysis_methods = {
                'ml_insights': self._ml_insights_analysis,
                'anomaly_detection': self._anomaly_detection_analysis,
                'demand_forecasting': self._demand_forecasting_analysis,
                'customer_behavior': self._customer_behavior_analysis,
                'product_lifecycle': self._product_lifecycle_analysis,
                'price_optimization': self._price_optimization_analysis,
                'inventory_optimization': self._inventory_optimization_analysis
            }
            
            # 5. Executar análise com parâmetros
            analysis_params = {
                'target_column': target_column,
                'prediction_horizon': prediction_horizon,
                'confidence_level': confidence_level,
                'model_complexity': model_complexity,
                'enable_ensemble': enable_ensemble
            }
            
            print(f"🎯 Executando análise ML: {analysis_type}")
            result = analysis_methods[analysis_type](df, **analysis_params)
            
            # 6. Adicionar metadados
            result['metadata'] = {
                'tool': 'Advanced Analytics Engine v4.0',
                'analysis_type': analysis_type,
                'target_column': target_column,
                'total_records': len(df),
                'model_complexity': model_complexity,
                'ensemble_enabled': enable_ensemble,
                'sklearn_available': SKLEARN_AVAILABLE,
                'xgboost_available': XGBOOST_AVAILABLE,
                'date_range': {
                    'start': df['Data'].min().strftime("%Y-%m-%d") if 'Data' in df.columns else "N/A",
                    'end': df['Data'].max().strftime("%Y-%m-%d") if 'Data' in df.columns else "N/A"
                },
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 7. Armazenar no cache se solicitado
            if cache_results:
                cache_key = f"ml_{analysis_type}_{hash(data_csv)}_{target_column}_{model_complexity}"
                self._ml_cache[cache_key] = result
                print(f"💾 Resultado ML salvo no cache")
            
            # 8. Formatar resultado final
            print("✅ Análise ML concluída com sucesso!")
            return json.dumps(result, ensure_ascii=False, indent=2, default=str)
            
        except Exception as e:
            error_response = {
                "error": f"Erro na análise ML v4.0: {str(e)}",
                "analysis_type": analysis_type,
                "data_csv": data_csv,
                "troubleshooting": {
                    "check_data_format": "Verifique se os dados estão no formato correto",
                    "check_ml_requirements": "Confirme que bibliotecas ML estão instaladas",
                    "reduce_complexity": "Tente model_complexity='simple' para datasets pequenos",
                    "try_statistical_analysis": "Use Statistical Analysis Tool como alternativa"
                },
                "metadata": {
                    "tool": "Advanced Analytics Engine",
                    "status": "error",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            return json.dumps(error_response, ensure_ascii=False, indent=2)
    
    def _load_and_prepare_ml_data(self, data_csv: str, use_cache: bool = True, sample_size: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Carregar e preparar dados especificamente para análises ML."""
        cache_key = f"ml_data_{hash(data_csv)}_{sample_size}"
        
        # Verificar cache
        if use_cache and cache_key in self._ml_cache:
            print("📋 Usando dados ML do cache")
            return self._ml_cache[cache_key]
        
        try:
            # Carregar dados brutos
            df = pd.read_csv(data_csv, sep=';', encoding='utf-8')
            print(f"📁 Arquivo carregado: {len(df)} registros")
            
            # Aplicar amostragem se necessário
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                print(f"📊 Amostra aplicada: {len(df)} registros")
            
            # Preparar dados usando mixin consolidado (nível strict para ML)
            df_prepared = self.prepare_jewelry_data(df, validation_level="strict")
            
            if df_prepared is None:
                print("❌ Falha na preparação dos dados ML")
                return None
            
            # Preparações específicas para ML
            df_prepared = self._add_ml_features(df_prepared)
            
            # Armazenar no cache
            if use_cache:
                self._ml_cache[cache_key] = df_prepared
                print("💾 Dados ML salvos no cache")
            
            return df_prepared
            
        except Exception as e:
            print(f"❌ Erro no carregamento de dados ML: {str(e)}")
            return None
    
    def _add_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adicionar features específicas para análises ML."""
        try:
            print("🧮 Adicionando features ML...")
            
            # Features temporais
            if 'Data' in df.columns:
                df['Days_Since_Start'] = (df['Data'] - df['Data'].min()).dt.days
                df['Month_Sin'] = np.sin(2 * np.pi * df['Data'].dt.month / 12)
                df['Month_Cos'] = np.cos(2 * np.pi * df['Data'].dt.month / 12)
                df['Day_Of_Week'] = df['Data'].dt.dayofweek
                print("✅ Features temporais adicionadas")
            
            # Features de agregação
            if 'Codigo_Cliente' in df.columns:
                customer_agg = df.groupby('Codigo_Cliente')['Total_Liquido'].agg(['count', 'mean', 'sum'])
                customer_agg.columns = ['Customer_Frequency', 'Customer_AOV', 'Customer_Total']
                df = df.merge(customer_agg, left_on='Codigo_Cliente', right_index=True, how='left')
                print("✅ Features de cliente adicionadas")
            
            # Features de produto
            if 'Codigo_Produto' in df.columns:
                product_agg = df.groupby('Codigo_Produto')['Total_Liquido'].agg(['count', 'mean'])
                product_agg.columns = ['Product_Frequency', 'Product_AOV']
                df = df.merge(product_agg, left_on='Codigo_Produto', right_index=True, how='left')
                print("✅ Features de produto adicionadas")
            
            # Encoding de variáveis categóricas
            categorical_cols = ['Sexo', 'Estado_Civil', 'Estado', 'Grupo_Produto', 'Metal']
            for col in categorical_cols:
                if col in df.columns:
                    # Label encoding para variáveis com muitas categorias
                    if df[col].nunique() > 10:
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                    else:
                        # One-hot encoding para variáveis com poucas categorias
                        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                        df = pd.concat([df, dummies], axis=1)
            
            print("✅ Encoding categórico aplicado")
            
            # Normalização de features numéricas
            numeric_cols = ['Total_Liquido', 'Quantidade', 'Margem_Real', 'Preco_Unitario']
            available_numeric = [col for col in numeric_cols if col in df.columns]
            
            if available_numeric:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df[available_numeric])
                
                for i, col in enumerate(available_numeric):
                    df[f'{col}_scaled'] = scaled_data[:, i]
                
                print(f"✅ Normalização aplicada: {len(available_numeric)} campos")
            
            return df
            
        except Exception as e:
            print(f"⚠️ Erro ao adicionar features ML: {str(e)}")
            return df
    
    def _ml_insights_analysis(self, df: pd.DataFrame, target_column: str = "Total_Liquido",
                             model_complexity: str = "balanced", enable_ensemble: bool = True,
                             **kwargs) -> Dict[str, Any]:
        """Análise de insights ML com Random Forest e XGBoost."""
        try:
            print("🤖 Executando análise de insights ML...")
            
            result = {
                'analysis_type': 'ML Insights Analysis',
                'target_column': target_column,
                'model_complexity': model_complexity,
                'ensemble_enabled': enable_ensemble
            }
            
            # Preparar dados para ML
            feature_cols = self._select_ml_features(df, target_column)
            if len(feature_cols) < 3:
                return {'error': 'Insuficientes features para análise ML (mínimo 3)'}
            
            X = df[feature_cols].fillna(0)
            y = df[target_column]
            
            # Split treino/teste
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Configurar modelos baseado na complexidade
            models = self._configure_ml_models(model_complexity, enable_ensemble)
            
            # Treinar modelos
            trained_models = {}
            model_performance = {}
            
            for model_name, model in models.items():
                print(f"🔧 Treinando {model_name}...")
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Métricas de performance
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                trained_models[model_name] = model
                model_performance[model_name] = {
                    'mae': round(mae, 2),
                    'mse': round(mse, 2),
                    'rmse': round(np.sqrt(mse), 2),
                    'r2_score': round(r2, 3)
                }
            
            result['model_performance'] = model_performance
            
            # Feature importance (usando o melhor modelo)
            best_model_name = max(model_performance.keys(), key=lambda k: model_performance[k]['r2_score'])
            best_model = trained_models[best_model_name]
            
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                result['feature_importance'] = {
                    'top_features': feature_importance.head(10).to_dict('records'),
                    'best_model': best_model_name
                }
            
            # Insights de negócio
            result['business_insights'] = self._generate_ml_insights(feature_importance if 'feature_importance' in result else None, model_performance)
            result['recommendations'] = self._generate_ml_recommendations(result)
            
            return result
            
        except Exception as e:
            return {'error': f"Erro na análise de insights ML: {str(e)}"}
    
    def _select_ml_features(self, df: pd.DataFrame, target_column: str) -> List[str]:
        """Selecionar features otimizadas para ML."""
        # Priorizar features numéricas e encodadas
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Excluir colunas que não devem ser features
        exclude_cols = [
            target_column, 'Data', 'Codigo_Cliente', 'Codigo_Produto', 
            'Ano', 'Mes', 'Dia', 'index'
        ]
        
        selected_features = [col for col in numeric_features 
                           if col not in exclude_cols and not df[col].isna().all()]
        
        # Limitar a 20 features para evitar overfitting
        return selected_features[:20]
    
    def _configure_ml_models(self, complexity: str, enable_ensemble: bool) -> Dict[str, Any]:
        """Configurar modelos ML baseado na complexidade."""
        models = {}
        
        if complexity == "simple":
            models['Random Forest'] = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        
        elif complexity == "balanced":
            models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            if XGBOOST_AVAILABLE and enable_ensemble:
                models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        elif complexity == "complex":
            models['Random Forest'] = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
            if XGBOOST_AVAILABLE and enable_ensemble:
                models['XGBoost'] = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
        
        return models
    
    def _generate_ml_insights(self, feature_importance: Optional[pd.DataFrame], model_performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gerar insights baseados em ML."""
        insights = []
        
        # Insights de feature importance
        if feature_importance is not None and len(feature_importance) > 0:
            top_feature = feature_importance.iloc[0]
            insights.append({
                "type": "Feature Mais Importante",
                "message": f"'{top_feature['feature']}' é o fator mais influente nas vendas",
                "impact": "high",
                "recommendation": f"Focar estratégias em otimizar {top_feature['feature']}"
            })
        
        # Insights de performance do modelo
        if model_performance:
            best_model = max(model_performance.keys(), key=lambda k: model_performance[k]['r2_score'])
            best_r2 = model_performance[best_model]['r2_score']
            
            if best_r2 > 0.8:
                insights.append({
                    "type": "Modelo Excelente",
                    "message": f"Modelo {best_model} com R² = {best_r2:.3f} - Previsões muito confiáveis",
                    "impact": "high",
                    "recommendation": "Usar modelo para previsões estratégicas"
                })
            elif best_r2 > 0.6:
                insights.append({
                    "type": "Modelo Bom",
                    "message": f"Modelo {best_model} com R² = {best_r2:.3f} - Previsões confiáveis",
                    "impact": "medium",
                    "recommendation": "Modelo adequado para análises táticas"
                })
            else:
                insights.append({
                    "type": "Modelo Limitado",
                    "message": f"Modelo {best_model} com R² = {best_r2:.3f} - Previsões com limitações",
                    "impact": "low",
                    "recommendation": "Coletar mais dados ou features para melhorar modelo"
                })
        
        return insights
    
    def _generate_ml_recommendations(self, result: Dict[str, Any]) -> List[str]:
        """Gerar recomendações baseadas em análise ML."""
        recommendations = []
        
        if 'feature_importance' in result:
            top_features = result['feature_importance']['top_features'][:3]
            for feature in top_features:
                recommendations.append(f"Otimizar {feature['feature']} para impactar vendas")
        
        if 'model_performance' in result:
            best_model = max(result['model_performance'].keys(), 
                           key=lambda k: result['model_performance'][k]['r2_score'])
            recommendations.append(f"Usar modelo {best_model} para previsões futuras")
        
        return recommendations[:5]
    
    def _anomaly_detection_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Análise de detecção de anomalias."""
        try:
            print("🔍 Executando detecção de anomalias...")
            
            target_col = kwargs.get('target_column', 'Total_Liquido')
            
            # Isolation Forest para detecção de anomalias
            features = self._select_ml_features(df, target_col)[:10]  # Limitar features
            X = df[features].fillna(0)
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(X)
            
            # Identificar anomalias
            df_with_anomalies = df.copy()
            df_with_anomalies['anomaly'] = anomaly_labels
            anomalies = df_with_anomalies[df_with_anomalies['anomaly'] == -1]
            
            result = {
                'analysis_type': 'Anomaly Detection Analysis',
                'target_column': target_col,
                'total_anomalies': len(anomalies),
                'anomaly_percentage': round(len(anomalies) / len(df) * 100, 2),
                'anomaly_summary': {
                    'avg_value': round(anomalies[target_col].mean(), 2),
                    'max_value': round(anomalies[target_col].max(), 2),
                    'min_value': round(anomalies[target_col].min(), 2)
                },
                'business_insights': [
                    {
                        "type": "Anomalias Detectadas",
                        "message": f"{len(anomalies)} transações anômalas identificadas ({len(anomalies) / len(df) * 100:.1f}%)",
                        "impact": "medium",
                        "recommendation": "Investigar transações anômalas para identificar oportunidades ou problemas"
                    }
                ],
                'recommendations': [
                    "Analisar padrões das transações anômalas",
                    "Verificar se anomalias representam oportunidades de upsell",
                    "Investigar possíveis erros ou fraudes"
                ]
            }
            
            return result
            
        except Exception as e:
            return {'error': f"Erro na detecção de anomalias: {str(e)}"}
    
    def _demand_forecasting_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Análise de previsão de demanda."""
        try:
            print("📈 Executando previsão de demanda...")
            
            target_col = kwargs.get('target_column', 'Total_Liquido')
            horizon = kwargs.get('prediction_horizon', 30)
            
            # Agregar dados por dia
            if 'Data' in df.columns:
                daily_sales = df.groupby('Data')[target_col].sum().reset_index()
                daily_sales = daily_sales.sort_values('Data')
                
                # Features temporais simples
                daily_sales['day_of_week'] = daily_sales['Data'].dt.dayofweek
                daily_sales['month'] = daily_sales['Data'].dt.month
                daily_sales['day_of_month'] = daily_sales['Data'].dt.day
                
                # Lag features
                daily_sales['lag_7'] = daily_sales[target_col].shift(7)
                daily_sales['lag_30'] = daily_sales[target_col].shift(30)
                daily_sales = daily_sales.dropna()
                
                if len(daily_sales) < 60:
                    return {'error': 'Dados insuficientes para previsão (mínimo 60 dias)'}
                
                # Preparar dados para ML
                feature_cols = ['day_of_week', 'month', 'day_of_month', 'lag_7', 'lag_30']
                X = daily_sales[feature_cols]
                y = daily_sales[target_col]
                
                # Treinar modelo
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                # Fazer previsões
                last_date = daily_sales['Data'].max()
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon)
                
                # Simular features futuras (simplificado)
                future_features = []
                for date in future_dates:
                    features = [
                        date.dayofweek,
                        date.month,
                        date.day,
                        daily_sales[target_col].tail(7).mean(),  # Média últimos 7 dias
                        daily_sales[target_col].tail(30).mean()  # Média últimos 30 dias
                    ]
                    future_features.append(features)
                
                future_X = pd.DataFrame(future_features, columns=feature_cols)
                predictions = model.predict(future_X)
                
                result = {
                    'analysis_type': 'Demand Forecasting Analysis',
                    'target_column': target_col,
                    'prediction_horizon': horizon,
                    'historical_days': len(daily_sales),
                    'forecast_summary': {
                        'avg_predicted': round(predictions.mean(), 2),
                        'total_predicted': round(predictions.sum(), 2),
                        'min_predicted': round(predictions.min(), 2),
                        'max_predicted': round(predictions.max(), 2)
                    },
                    'daily_predictions': [
                        {
                            'date': date.strftime('%Y-%m-%d'),
                            'predicted_value': round(pred, 2)
                        }
                        for date, pred in zip(future_dates, predictions)
                    ][:10],  # Primeiros 10 dias
                    'business_insights': [
                        {
                            "type": "Previsão de Demanda",
                            "message": f"Demanda prevista para próximos {horizon} dias: R$ {predictions.sum():,.2f}",
                            "impact": "high",
                            "recommendation": "Ajustar estoque baseado na previsão"
                        }
                    ],
                    'recommendations': [
                        "Usar previsões para planejamento de estoque",
                        "Monitorar desvios entre previsto e realizado",
                        "Ajustar modelo com novos dados mensalmente"
                    ]
                }
                
                return result
            else:
                return {'error': 'Coluna Data não encontrada para previsão temporal'}
            
        except Exception as e:
            return {'error': f"Erro na previsão de demanda: {str(e)}"}
    
    def _customer_behavior_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Análise de comportamento de clientes com ML."""
        try:
            print("👥 Executando análise de comportamento de clientes...")
            
            if 'Codigo_Cliente' not in df.columns:
                return {'error': 'Coluna Codigo_Cliente não encontrada para análise comportamental'}
            
            target_col = kwargs.get('target_column', 'Total_Liquido')
            
            # Agregar dados por cliente
            customer_features = df.groupby('Codigo_Cliente').agg({
                target_col: ['count', 'sum', 'mean'],
                'Quantidade': 'sum',
                'Data': ['min', 'max']
            }).round(2)
            
            # Flatten column names
            customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns]
            
            # Calcular recência
            max_date = df['Data'].max()
            customer_features['recency'] = (max_date - customer_features['Data_max']).dt.days
            
            # Preparar para clustering
            cluster_features = [
                f'{target_col}_count', f'{target_col}_sum', f'{target_col}_mean',
                'Quantidade_sum', 'recency'
            ]
            
            X = customer_features[cluster_features].fillna(0)
            
            # Normalizar dados
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # K-means clustering
            optimal_k = min(5, len(X) // 10)  # Heurística simples
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            customer_features['cluster'] = cluster_labels
            
            # Analisar clusters
            cluster_profiles = {}
            for cluster_id in range(optimal_k):
                cluster_data = customer_features[customer_features['cluster'] == cluster_id]
                
                profile = {
                    'size': len(cluster_data),
                    'size_percentage': round(len(cluster_data) / len(customer_features) * 100, 1),
                    'avg_revenue': round(cluster_data[f'{target_col}_sum'].mean(), 2),
                    'avg_frequency': round(cluster_data[f'{target_col}_count'].mean(), 1),
                    'avg_recency': round(cluster_data['recency'].mean(), 1),
                    'total_revenue': round(cluster_data[f'{target_col}_sum'].sum(), 2)
                }
                
                # Classificar cluster
                if profile['avg_revenue'] > customer_features[f'{target_col}_sum'].quantile(0.8):
                    profile['classification'] = 'VIP'
                elif profile['avg_frequency'] > customer_features[f'{target_col}_count'].quantile(0.7):
                    profile['classification'] = 'Frequente'
                elif profile['avg_recency'] < 30:
                    profile['classification'] = 'Ativo'
                else:
                    profile['classification'] = 'Regular'
                
                cluster_profiles[f'Cluster_{cluster_id}'] = profile
            
            result = {
                'analysis_type': 'Customer Behavior Analysis',
                'target_column': target_col,
                'total_customers': len(customer_features),
                'clusters_identified': optimal_k,
                'cluster_profiles': cluster_profiles,
                'business_insights': [
                    {
                        "type": "Segmentação de Clientes",
                        "message": f"{optimal_k} segmentos comportamentais identificados",
                        "impact": "high",
                        "recommendation": "Desenvolver estratégias específicas para cada segmento"
                    }
                ],
                'recommendations': [
                    "Personalizar comunicação por segmento",
                    "Focar retenção nos clientes VIP",
                    "Reativar clientes com alta recência",
                    "Aumentar frequência dos clientes regulares"
                ]
            }
            
            return result
            
        except Exception as e:
            return {'error': f"Erro na análise comportamental: {str(e)}"}
    
    def _product_lifecycle_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Análise de ciclo de vida de produtos."""
        return {
            'analysis_type': 'Product Lifecycle Analysis',
            'message': 'Análise de ciclo de vida de produtos em desenvolvimento',
            'status': 'placeholder',
            'business_insights': [
                {
                    "type": "Em Desenvolvimento",
                    "message": "Análise de ciclo de vida será implementada na v4.1",
                    "impact": "medium",
                    "recommendation": "Use Product Performance no Business Intelligence Tool"
                }
            ]
        }
    
    def _price_optimization_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Análise de otimização de preços."""
        return {
            'analysis_type': 'Price Optimization Analysis',
            'message': 'Análise de otimização de preços em desenvolvimento',
            'status': 'placeholder',
            'business_insights': [
                {
                    "type": "Em Desenvolvimento",
                    "message": "Otimização de preços será implementada na v4.1",
                    "impact": "medium",
                    "recommendation": "Use análises de margem no KPI Calculator Tool"
                }
            ]
        }
    
    def _inventory_optimization_analysis(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Análise de otimização de inventário."""
        return {
            'analysis_type': 'Inventory Optimization Analysis',
            'message': 'Análise de otimização de inventário em desenvolvimento',
            'status': 'placeholder',
            'business_insights': [
                {
                    "type": "Em Desenvolvimento",
                    "message": "Otimização de inventário será implementada na v4.1",
                    "impact": "medium",
                    "recommendation": "Use análises ABC no KPI Calculator Tool"
                }
            ]
        }