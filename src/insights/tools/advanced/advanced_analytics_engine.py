from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalyticsInput(BaseModel):
    """Schema de entrada para análises avançadas."""
    analysis_type: str = Field(..., description="Tipo: 'ml_insights', 'anomaly_detection', 'customer_behavior', 'demand_forecasting', 'price_optimization', 'inventory_optimization'")
    data_csv: str = Field(default="data/vendas.csv", description="Caminho para o arquivo CSV")
    target_column: str = Field(default="Total_Liquido", description="Coluna alvo para ML")
    prediction_horizon: int = Field(default=30, description="Horizonte de predição em dias")
    confidence_level: float = Field(default=0.95, description="Nível de confiança (0.90-0.99)")

class AdvancedAnalyticsEngine(BaseTool):
    name: str = "Advanced Analytics Engine"
    description: str = """
    Motor de análises avançadas com Machine Learning para joalherias:
    
    ANÁLISES DISPONÍVEIS:
    - ml_insights: Insights baseados em ML (Random Forest, XGBoost)
    - anomaly_detection: Detecção de anomalias em vendas usando múltiplos algoritmos
    - customer_behavior: Análise comportamental avançada de clientes
    - demand_forecasting: Previsão de demanda com ML ensemble
    - price_optimization: Otimização de preços baseada em elasticidade
    - inventory_optimization: Otimização de estoque com ML
    
    ALGORITMOS UTILIZADOS:
    - Random Forest, XGBoost para predição
    - Isolation Forest, DBSCAN para anomalias
    - PCA, K-means para segmentação
    - Elastic Net para regularização
    - Cross-validation para validação
    """
    args_schema: Type[BaseModel] = AdvancedAnalyticsInput
    
    def _run(self, analysis_type: str, data_csv: str = "data/vendas.csv", 
             target_column: str = "Total_Liquido", prediction_horizon: int = 30,
             confidence_level: float = 0.95) -> str:
        try:
            # Carregar e preparar dados
            df = pd.read_csv(data_csv, sep=';', encoding='utf-8')
            df = self._prepare_advanced_features(df)
            
            if df is None or len(df) < 50:
                return "Erro: Dados insuficientes para análise de ML (mínimo 50 registros)"
            
            # Dicionário de análises avançadas
            advanced_analyses = {
                'ml_insights': self._generate_ml_insights,
                'anomaly_detection': self._advanced_anomaly_detection,
                'customer_behavior': self._advanced_customer_behavior,
                'demand_forecasting': self._ml_demand_forecasting,
                'price_optimization': self._price_optimization_ml,
                'inventory_optimization': self._inventory_optimization_ml
            }
            
            if analysis_type not in advanced_analyses:
                return f"Análise '{analysis_type}' não suportada. Opções: {list(advanced_analyses.keys())}"
            
            result = advanced_analyses[analysis_type](df, target_column, prediction_horizon, confidence_level)
            return self._format_advanced_result(analysis_type, result)
            
        except Exception as e:
            return f"Erro na análise avançada: {str(e)}"
    
    def _prepare_advanced_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Preparar features avançadas para Machine Learning."""
        try:
            # Converter data
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
            df = df.dropna(subset=['Data'])
            
            # Features temporais avançadas
            df['Ano'] = df['Data'].dt.year
            df['Mes'] = df['Data'].dt.month
            df['Dia'] = df['Data'].dt.day
            df['Dia_Semana'] = df['Data'].dt.dayofweek
            df['Trimestre'] = df['Data'].dt.quarter
            df['Semana_Ano'] = df['Data'].dt.isocalendar().week
            df['Dia_Mes'] = df['Data'].dt.day
            df['Is_Weekend'] = df['Dia_Semana'].isin([5, 6]).astype(int)
            df['Is_Month_End'] = (df['Data'].dt.day > 25).astype(int)
            df['Is_Month_Start'] = (df['Data'].dt.day <= 5).astype(int)
            
            # Features sazonais
            df['Sin_Month'] = np.sin(2 * np.pi * df['Mes'] / 12)
            df['Cos_Month'] = np.cos(2 * np.pi * df['Mes'] / 12)
            df['Sin_Day'] = np.sin(2 * np.pi * df['Dia_Semana'] / 7)
            df['Cos_Day'] = np.cos(2 * np.pi * df['Dia_Semana'] / 7)
            
            # Features de lag temporal
            df = df.sort_values('Data')
            df['Total_Liquido_Lag1'] = df['Total_Liquido'].shift(1)
            df['Total_Liquido_Lag7'] = df['Total_Liquido'].shift(7)
            df['Total_Liquido_MA7'] = df['Total_Liquido'].rolling(window=7).mean()
            df['Total_Liquido_MA30'] = df['Total_Liquido'].rolling(window=30).mean()
            
            # Features estatísticas móveis
            df['Total_Liquido_Std7'] = df['Total_Liquido'].rolling(window=7).std()
            df['Total_Liquido_Min7'] = df['Total_Liquido'].rolling(window=7).min()
            df['Total_Liquido_Max7'] = df['Total_Liquido'].rolling(window=7).max()
            
            # Features de categoria (se disponível)
            if 'Grupo_Produto' in df.columns:
                le_grupo = LabelEncoder()
                df['Grupo_Produto_Encoded'] = le_grupo.fit_transform(df['Grupo_Produto'].fillna('Outros'))
            
            if 'Metal' in df.columns:
                le_metal = LabelEncoder()
                df['Metal_Encoded'] = le_metal.fit_transform(df['Metal'].fillna('Outros'))
            
            # Features de interação
            if 'Quantidade' in df.columns:
                df['Preco_Unitario'] = df['Total_Liquido'] / df['Quantidade'].replace(0, 1)
                df['Log_Preco_Unitario'] = np.log1p(df['Preco_Unitario'])
            
            # Remover valores infinitos e NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            return df
            
        except Exception as e:
            print(f"Erro na preparação de features: {str(e)}")
            return None
    
    def _generate_ml_insights(self, df: pd.DataFrame, target_column: str, 
                            prediction_horizon: int, confidence_level: float) -> Dict[str, Any]:
        """Gerar insights usando Machine Learning."""
        try:
            # Preparar features para ML
            feature_columns = [
                'Mes', 'Dia_Semana', 'Trimestre', 'Is_Weekend', 'Is_Month_End',
                'Sin_Month', 'Cos_Month', 'Sin_Day', 'Cos_Day'
            ]
            
            # Adicionar features disponíveis
            available_features = []
            for col in feature_columns:
                if col in df.columns:
                    available_features.append(col)
            
            if 'Grupo_Produto_Encoded' in df.columns:
                available_features.append('Grupo_Produto_Encoded')
            if 'Metal_Encoded' in df.columns:
                available_features.append('Metal_Encoded')
            if 'Quantidade' in df.columns:
                available_features.append('Quantidade')
            
            if len(available_features) < 3:
                return {'error': 'Features insuficientes para ML'}
            
            # Preparar dataset
            ml_df = df[available_features + [target_column]].dropna()
            
            if len(ml_df) < 50:
                return {'error': 'Dados insuficientes após limpeza'}
            
            X = ml_df[available_features]
            y = ml_df[target_column]
            
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 1. Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            rf_score = r2_score(y_test, rf_pred)
            
            # Feature importance
            feature_importance = dict(zip(available_features, rf_model.feature_importances_))
            feature_importance = {k: round(v, 4) for k, v in 
                                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)}
            
            # 2. XGBoost
            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
            xgb_model.fit(X_train, y_train)
            xgb_pred = xgb_model.predict(X_test)
            xgb_score = r2_score(y_test, xgb_pred)
            
            # 3. Ensemble prediction
            ensemble_pred = (rf_pred + xgb_pred) / 2
            ensemble_score = r2_score(y_test, ensemble_pred)
            
            # Validação cruzada
            cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
            
            # Insights baseados em ML
            ml_insights = []
            
            # Feature mais importante
            top_feature = list(feature_importance.keys())[0]
            top_importance = list(feature_importance.values())[0]
            ml_insights.append(f"Fator mais importante: {top_feature} (importância: {top_importance:.3f})")
            
            # Performance do modelo
            if ensemble_score > 0.8:
                ml_insights.append("Modelo com alta precisão - padrões bem definidos nos dados")
            elif ensemble_score > 0.6:
                ml_insights.append("Modelo com precisão moderada - padrões identificáveis")
            else:
                ml_insights.append("Padrões complexos - alta variabilidade nos dados")
            
            # Previsões para período futuro
            future_predictions = self._generate_future_predictions(
                rf_model, X, available_features, prediction_horizon
            )
            
            return {
                'model_performance': {
                    'random_forest_r2': round(rf_score, 4),
                    'xgboost_r2': round(xgb_score, 4),
                    'ensemble_r2': round(ensemble_score, 4),
                    'cross_validation_mean': round(cv_scores.mean(), 4),
                    'cross_validation_std': round(cv_scores.std(), 4)
                },
                'feature_importance': feature_importance,
                'future_predictions': future_predictions,
                'model_insights': ml_insights,
                'prediction_intervals': self._calculate_prediction_intervals(
                    ensemble_pred, y_test, confidence_level
                )
            }
            
        except Exception as e:
            return {'error': f"Erro no ML insights: {str(e)}"}
    
    def _advanced_anomaly_detection(self, df: pd.DataFrame, target_column: str,
                                   prediction_horizon: int, confidence_level: float) -> Dict[str, Any]:
        """Detecção avançada de anomalias usando múltiplos algoritmos."""
        try:
            # Preparar dados
            anomaly_features = ['Total_Liquido', 'Quantidade'] if 'Quantidade' in df.columns else ['Total_Liquido']
            
            # Adicionar features temporais
            temporal_features = ['Mes', 'Dia_Semana', 'Dia_Mes']
            available_temporal = [col for col in temporal_features if col in df.columns]
            anomaly_features.extend(available_temporal)
            
            anomaly_df = df[anomaly_features].dropna()
            
            if len(anomaly_df) < 50:
                return {'error': 'Dados insuficientes para detecção de anomalias'}
            
            # Padronizar dados
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(anomaly_df)
            
            # 1. Isolation Forest
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            iso_anomalies = iso_forest.fit_predict(X_scaled)
            iso_scores = iso_forest.decision_function(X_scaled)
            
            # 2. DBSCAN Clustering
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(X_scaled)
            dbscan_anomalies = (dbscan_labels == -1).astype(int)
            dbscan_anomalies = np.where(dbscan_anomalies == 1, -1, 1)  # Converter para formato padrão
            
            # 3. Estatístico (Z-score modificado)
            z_scores = np.abs(stats.zscore(anomaly_df[target_column]))
            statistical_anomalies = (z_scores > 3).astype(int)
            statistical_anomalies = np.where(statistical_anomalies == 1, -1, 1)
            
            # Combinar resultados
            anomaly_df['ISO_Anomaly'] = iso_anomalies
            anomaly_df['DBSCAN_Anomaly'] = dbscan_anomalies
            anomaly_df['Statistical_Anomaly'] = statistical_anomalies
            anomaly_df['ISO_Score'] = iso_scores
            anomaly_df['Z_Score'] = z_scores
            
            # Anomalias consenso (detectadas por pelo menos 2 métodos)
            anomaly_df['Consensus_Anomaly'] = (
                (anomaly_df['ISO_Anomaly'] == -1).astype(int) +
                (anomaly_df['DBSCAN_Anomaly'] == -1).astype(int) +
                (anomaly_df['Statistical_Anomaly'] == -1).astype(int)
            ) >= 2
            
            # Adicionar dados originais para contexto
            anomaly_df['Data'] = df['Data'].iloc[:len(anomaly_df)]
            if 'Codigo_Produto' in df.columns:
                anomaly_df['Codigo_Produto'] = df['Codigo_Produto'].iloc[:len(anomaly_df)]
            
            # Identificar top anomalias
            consensus_anomalies = anomaly_df[anomaly_df['Consensus_Anomaly']]
            top_anomalies = consensus_anomalies.nlargest(10, 'Z_Score') if len(consensus_anomalies) > 0 else pd.DataFrame()
            
            # Análise temporal de anomalias
            if len(consensus_anomalies) > 0:
                anomaly_by_month = consensus_anomalies.groupby(
                    consensus_anomalies['Data'].dt.month
                ).size().to_dict()
                anomaly_by_weekday = consensus_anomalies.groupby(
                    consensus_anomalies['Data'].dt.dayofweek
                ).size().to_dict()
            else:
                anomaly_by_month = {}
                anomaly_by_weekday = {}
            
            # Insights de anomalias
            anomaly_insights = []
            
            total_anomalies = len(consensus_anomalies)
            anomaly_rate = total_anomalies / len(anomaly_df) * 100
            
            if anomaly_rate > 10:
                anomaly_insights.append(f"Alta taxa de anomalias ({anomaly_rate:.1f}%) - investigar qualidade dos dados")
            elif anomaly_rate < 2:
                anomaly_insights.append("Baixa taxa de anomalias - dados consistentes")
            else:
                anomaly_insights.append(f"Taxa normal de anomalias ({anomaly_rate:.1f}%)")
            
            if anomaly_by_month:
                peak_anomaly_month = max(anomaly_by_month, key=anomaly_by_month.get)
                anomaly_insights.append(f"Mês com mais anomalias: {peak_anomaly_month}")
            
            return {
                'anomaly_summary': {
                    'total_records': len(anomaly_df),
                    'isolation_forest_anomalies': int((iso_anomalies == -1).sum()),
                    'dbscan_anomalies': int((dbscan_anomalies == -1).sum()),
                    'statistical_anomalies': int((statistical_anomalies == -1).sum()),
                    'consensus_anomalies': total_anomalies,
                    'anomaly_rate_percent': round(anomaly_rate, 2)
                },
                'temporal_patterns': {
                    'anomalies_by_month': anomaly_by_month,
                    'anomalies_by_weekday': anomaly_by_weekday
                },
                'top_anomalies': top_anomalies[[
                    'Data', target_column, 'Z_Score', 'ISO_Score'
                ]].to_dict('records') if len(top_anomalies) > 0 else [],
                'insights': anomaly_insights
            }
            
        except Exception as e:
            return {'error': f"Erro na detecção de anomalias: {str(e)}"}
    
    def _advanced_customer_behavior(self, df: pd.DataFrame, target_column: str,
                                  prediction_horizon: int, confidence_level: float) -> Dict[str, Any]:
        """Análise comportamental avançada usando ML."""
        try:
            # Simular customer_id baseado em padrões de compra
            df_behavior = df.copy()
            
            # Cluster baseado em valor e frequência de compra
            behavior_features = []
            
            # Features de valor
            if target_column in df.columns:
                behavior_features.append(target_column)
            
            # Features temporais para identificar padrões
            if 'Dia_Semana' in df.columns:
                behavior_features.append('Dia_Semana')
            if 'Mes' in df.columns:
                behavior_features.append('Mes')
            
            # Features de produto se disponível
            if 'Grupo_Produto_Encoded' in df.columns:
                behavior_features.append('Grupo_Produto_Encoded')
            
            if len(behavior_features) < 2:
                return {'error': 'Features insuficientes para análise comportamental'}
            
            # Preparar dados para clustering
            behavior_df = df[behavior_features].dropna()
            
            if len(behavior_df) < 100:
                return {'error': 'Dados insuficientes para análise comportamental'}
            
            # Padronizar
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(behavior_df)
            
            # K-means clustering para identificar padrões comportamentais
            optimal_k = min(5, len(behavior_df) // 20)  # Máximo 5 clusters
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            behavior_df['Behavior_Cluster'] = clusters
            
            # Análise dos clusters comportamentais
            cluster_analysis = {}
            for cluster_id in range(optimal_k):
                cluster_data = behavior_df[behavior_df['Behavior_Cluster'] == cluster_id]
                
                cluster_profile = {
                    'size': len(cluster_data),
                    'percentage': round(len(cluster_data) / len(behavior_df) * 100, 2),
                    'avg_value': round(cluster_data[target_column].mean(), 2),
                    'profile': self._classify_behavior_cluster(cluster_data, target_column)
                }
                
                # Padrões temporais do cluster
                if 'Dia_Semana' in cluster_data.columns:
                    cluster_profile['preferred_weekdays'] = cluster_data['Dia_Semana'].mode().tolist()
                if 'Mes' in cluster_data.columns:
                    cluster_profile['preferred_months'] = cluster_data['Mes'].mode().tolist()
                
                cluster_analysis[f'Cluster_{cluster_id}'] = cluster_profile
            
            # PCA para visualização comportamental
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Análise de transição comportamental (se há dados temporais)
            transition_analysis = {}
            if 'Data' in df.columns:
                transition_analysis = self._analyze_behavior_transitions(df, clusters)
            
            # Insights comportamentais
            behavior_insights = []
            
            # Cluster dominante
            largest_cluster = max(cluster_analysis.items(), key=lambda x: x[1]['size'])
            behavior_insights.append(f"Padrão dominante: {largest_cluster[1]['profile']} ({largest_cluster[1]['percentage']}%)")
            
            # Cluster de alto valor
            high_value_clusters = [name for name, data in cluster_analysis.items() 
                                 if data['profile'] in ['High Value', 'Premium']]
            if high_value_clusters:
                behavior_insights.append(f"Segmentos de alto valor identificados: {len(high_value_clusters)} clusters")
            
            return {
                'cluster_analysis': cluster_analysis,
                'behavior_insights': behavior_insights,
                'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
                'transition_analysis': transition_analysis,
                'optimal_clusters': optimal_k
            }
            
        except Exception as e:
            return {'error': f"Erro na análise comportamental: {str(e)}"}
    
    def _ml_demand_forecasting(self, df: pd.DataFrame, target_column: str,
                             prediction_horizon: int, confidence_level: float) -> Dict[str, Any]:
        """Previsão de demanda usando ensemble de modelos ML."""
        try:
            # Preparar dados temporais
            df_forecast = df.copy()
            df_forecast = df_forecast.sort_values('Data')
            
            # Agregar por dia para forecasting
            daily_sales = df_forecast.groupby('Data').agg({
                target_column: 'sum',
                'Quantidade': 'sum' if 'Quantidade' in df_forecast.columns else 'count'
            }).fillna(0)
            
            # Adicionar features temporais
            daily_sales['Dia_Semana'] = daily_sales.index.dayofweek
            daily_sales['Mes'] = daily_sales.index.month
            daily_sales['Dia_Mes'] = daily_sales.index.day
            daily_sales['Sin_Month'] = np.sin(2 * np.pi * daily_sales['Mes'] / 12)
            daily_sales['Cos_Month'] = np.cos(2 * np.pi * daily_sales['Mes'] / 12)
            
            # Features de lag
            for lag in [1, 7, 14]:
                daily_sales[f'{target_column}_lag_{lag}'] = daily_sales[target_column].shift(lag)
            
            # Moving averages
            for window in [3, 7, 14]:
                daily_sales[f'{target_column}_ma_{window}'] = daily_sales[target_column].rolling(window=window).mean()
            
            # Remover NaN
            forecast_df = daily_sales.dropna()
            
            if len(forecast_df) < 30:
                return {'error': 'Dados insuficientes para forecasting (mínimo 30 dias)'}
            
            # Preparar features e target
            feature_cols = [col for col in forecast_df.columns if col != target_column and 'lag' in col or 'ma' in col or col in ['Dia_Semana', 'Mes', 'Sin_Month', 'Cos_Month']]
            
            X = forecast_df[feature_cols]
            y = forecast_df[target_column]
            
            # Split temporal (últimos 20% para teste)
            split_point = int(len(X) * 0.8)
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            # Modelos ensemble
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
                'Elastic Net': ElasticNet(random_state=42)
            }
            
            # Treinar modelos
            model_performance = {}
            predictions = {}
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                
                model_performance[name] = {
                    'mae': round(mean_absolute_error(y_test, pred), 2),
                    'rmse': round(np.sqrt(mean_squared_error(y_test, pred)), 2),
                    'r2': round(r2_score(y_test, pred), 4)
                }
                predictions[name] = pred
            
            # Ensemble prediction (média ponderada baseada em performance)
            weights = [model_performance[name]['r2'] for name in models.keys()]
            total_weight = sum(weights)
            
            if total_weight > 0:
                ensemble_pred = sum(pred * weight / total_weight 
                                  for pred, weight in zip(predictions.values(), weights))
            else:
                ensemble_pred = np.mean(list(predictions.values()), axis=0)
            
            ensemble_performance = {
                'mae': round(mean_absolute_error(y_test, ensemble_pred), 2),
                'rmse': round(np.sqrt(mean_squared_error(y_test, ensemble_pred)), 2),
                'r2': round(r2_score(y_test, ensemble_pred), 4)
            }
            
            # Previsão futura
            future_predictions = self._generate_future_forecast(
                models, X, feature_cols, prediction_horizon, daily_sales
            )
            
            return {
                'model_performance': model_performance,
                'ensemble_performance': ensemble_performance,
                'future_forecast': future_predictions,
                'forecast_insights': self._generate_forecast_insights(
                    future_predictions, ensemble_performance
                )
            }
            
        except Exception as e:
            return {'error': f"Erro no forecasting ML: {str(e)}"}
    
    def _price_optimization_ml(self, df: pd.DataFrame, target_column: str,
                             prediction_horizon: int, confidence_level: float) -> Dict[str, Any]:
        """Otimização de preços usando ML."""
        try:
            # Calcular preço unitário
            if 'Quantidade' not in df.columns:
                return {'error': 'Coluna Quantidade necessária para otimização de preços'}
            
            df_price = df.copy()
            df_price['Preco_Unitario'] = df_price[target_column] / df_price['Quantidade'].replace(0, 1)
            
            # Remover outliers extremos de preço
            price_q99 = df_price['Preco_Unitario'].quantile(0.99)
            price_q01 = df_price['Preco_Unitario'].quantile(0.01)
            df_price = df_price[
                (df_price['Preco_Unitario'] >= price_q01) & 
                (df_price['Preco_Unitario'] <= price_q99)
            ]
            
            if len(df_price) < 50:
                return {'error': 'Dados insuficientes para otimização de preços'}
            
            # Análise de elasticidade por categoria
            elasticity_analysis = {}
            
            if 'Grupo_Produto' in df_price.columns:
                for categoria in df_price['Grupo_Produto'].unique():
                    if pd.isna(categoria):
                        continue
                        
                    cat_data = df_price[df_price['Grupo_Produto'] == categoria]
                    if len(cat_data) < 20:
                        continue
                    
                    # Calcular elasticidade usando regressão
                    elasticity = self._calculate_price_elasticity(cat_data)
                    
                    # Simular otimização de preço
                    optimization = self._simulate_price_optimization(cat_data, elasticity)
                    
                    elasticity_analysis[categoria] = {
                        'current_avg_price': round(cat_data['Preco_Unitario'].mean(), 2),
                        'price_elasticity': round(elasticity, 3),
                        'optimization': optimization
                    }
            
            # Otimização geral
            general_elasticity = self._calculate_price_elasticity(df_price)
            general_optimization = self._simulate_price_optimization(df_price, general_elasticity)
            
            # Insights de otimização
            optimization_insights = []
            
            if general_elasticity > -0.5:
                optimization_insights.append("Demanda inelástica - oportunidade de aumento de preços")
            elif general_elasticity < -1.5:
                optimization_insights.append("Demanda elástica - cuidado com aumentos de preço")
            
            # Identificar categorias com maior potencial
            high_potential_categories = []
            for cat, data in elasticity_analysis.items():
                if data['optimization']['potential_revenue_increase'] > 5:
                    high_potential_categories.append(cat)
            
            if high_potential_categories:
                optimization_insights.append(f"Categorias com maior potencial: {', '.join(high_potential_categories[:3])}")
            
            return {
                'general_analysis': {
                    'current_avg_price': round(df_price['Preco_Unitario'].mean(), 2),
                    'price_elasticity': round(general_elasticity, 3),
                    'optimization': general_optimization
                },
                'category_analysis': elasticity_analysis,
                'optimization_insights': optimization_insights,
                'price_distribution': {
                    'min': round(df_price['Preco_Unitario'].min(), 2),
                    'max': round(df_price['Preco_Unitario'].max(), 2),
                    'median': round(df_price['Preco_Unitario'].median(), 2),
                    'std': round(df_price['Preco_Unitario'].std(), 2)
                }
            }
            
        except Exception as e:
            return {'error': f"Erro na otimização de preços: {str(e)}"}
    
    def _inventory_optimization_ml(self, df: pd.DataFrame, target_column: str,
                                 prediction_horizon: int, confidence_level: float) -> Dict[str, Any]:
        """Otimização de inventário usando ML."""
        try:
            # Análise por produto
            if 'Codigo_Produto' not in df.columns:
                return {'error': 'Coluna Codigo_Produto necessária para otimização de inventário'}
            
            df_inventory = df.copy()
            
            # Métricas por produto
            product_metrics = df_inventory.groupby('Codigo_Produto').agg({
                target_column: ['sum', 'mean', 'count', 'std'],
                'Quantidade': 'sum' if 'Quantidade' in df_inventory.columns else 'count',
                'Data': ['min', 'max']
            }).fillna(0)
            
            # Flatten columns
            product_metrics.columns = ['_'.join(col).strip() for col in product_metrics.columns]
            
            # Calcular métricas de inventário
            product_metrics['days_active'] = (
                pd.to_datetime(product_metrics['Data_max']) - 
                pd.to_datetime(product_metrics['Data_min'])
            ).dt.days + 1
            
            product_metrics['avg_daily_sales'] = (
                product_metrics[f'{target_column}_sum'] / product_metrics['days_active']
            )
            
            product_metrics['coefficient_variation'] = (
                product_metrics[f'{target_column}_std'] / product_metrics[f'{target_column}_mean']
            ).fillna(0)
            
            # Classificação ABC-XYZ
            abc_xyz_classification = self._perform_abc_xyz_analysis(product_metrics, target_column)
            
            # ML para previsão de demanda por produto
            demand_predictions = {}
            top_products = product_metrics.nlargest(20, f'{target_column}_sum')
            
            for produto in top_products.index[:10]:  # Top 10 produtos
                produto_data = df_inventory[df_inventory['Codigo_Produto'] == produto]
                if len(produto_data) >= 10:
                    daily_demand = produto_data.groupby('Data')[target_column].sum()
                    prediction = self._predict_product_demand(daily_demand, prediction_horizon)
                    demand_predictions[produto] = prediction
            
            # Otimização de níveis de estoque
            inventory_optimization = {}
            for produto in top_products.index:
                metrics = top_products.loc[produto]
                optimization = self._optimize_inventory_levels(metrics, target_column)
                inventory_optimization[produto] = optimization
            
            # Insights de inventário
            inventory_insights = []
            
            # Produtos de alto giro
            high_turnover = product_metrics[product_metrics['avg_daily_sales'] > 
                                         product_metrics['avg_daily_sales'].quantile(0.8)]
            inventory_insights.append(f"{len(high_turnover)} produtos de alto giro identificados")
            
            # Produtos de baixo giro
            low_turnover = product_metrics[product_metrics['avg_daily_sales'] < 
                                         product_metrics['avg_daily_sales'].quantile(0.2)]
            inventory_insights.append(f"{len(low_turnover)} produtos de baixo giro - candidatos à liquidação")
            
            # Produtos com alta variabilidade
            high_variation = product_metrics[product_metrics['coefficient_variation'] > 1]
            inventory_insights.append(f"{len(high_variation)} produtos com alta variabilidade - necessitam safety stock maior")
            
            return {
                'abc_xyz_analysis': abc_xyz_classification,
                'demand_predictions': demand_predictions,
                'inventory_optimization': dict(list(inventory_optimization.items())[:10]),
                'product_performance': {
                    'high_performers': high_turnover.index.tolist()[:10],
                    'low_performers': low_turnover.index.tolist()[:10],
                    'high_variation': high_variation.index.tolist()[:10]
                },
                'inventory_insights': inventory_insights
            }
            
        except Exception as e:
            return {'error': f"Erro na otimização de inventário: {str(e)}"}
    
    # Métodos auxiliares
    def _generate_future_predictions(self, model, X: pd.DataFrame, features: List[str], 
                                   horizon: int) -> Dict[str, Any]:
        """Gerar previsões futuras."""
        try:
            # Criar dados futuros baseados nos padrões dos últimos dados
            last_data = X.tail(1)
            future_predictions = []
            
            for i in range(horizon):
                # Simular dados futuros com base nos últimos padrões
                future_row = last_data.copy()
                
                # Ajustar features temporais se existirem
                if 'Mes' in features:
                    # Simular progressão temporal
                    current_month = int(future_row['Mes'].iloc[0])
                    future_month = ((current_month + i // 30) % 12) + 1
                    future_row['Mes'] = future_month
                
                if 'Dia_Semana' in features:
                    current_day = int(future_row['Dia_Semana'].iloc[0])
                    future_day = (current_day + i) % 7
                    future_row['Dia_Semana'] = future_day
                
                pred = model.predict(future_row)[0]
                future_predictions.append(round(pred, 2))
            
            return {
                'predictions': future_predictions,
                'horizon_days': horizon,
                'total_predicted': round(sum(future_predictions), 2),
                'avg_daily': round(np.mean(future_predictions), 2)
            }
            
        except Exception as e:
            return {'error': f"Erro nas previsões futuras: {str(e)}"}
    
    def _calculate_prediction_intervals(self, predictions: np.ndarray, actual: np.ndarray, 
                                      confidence_level: float) -> Dict[str, float]:
        """Calcular intervalos de confiança."""
        try:
            residuals = actual - predictions
            residual_std = np.std(residuals)
            
            # Z-score para o nível de confiança
            from scipy.stats import norm
            alpha = 1 - confidence_level
            z_score = norm.ppf(1 - alpha/2)
            
            margin_error = z_score * residual_std
            
            return {
                'lower_bound': round(np.mean(predictions) - margin_error, 2),
                'upper_bound': round(np.mean(predictions) + margin_error, 2),
                'margin_error': round(margin_error, 2),
                'confidence_level': confidence_level
            }
            
        except Exception as e:
            return {'error': f"Erro no cálculo de intervalos: {str(e)}"}
    
    def _classify_behavior_cluster(self, cluster_data: pd.DataFrame, target_column: str) -> str:
        """Classificar cluster comportamental."""
        avg_value = cluster_data[target_column].mean()
        overall_avg = cluster_data[target_column].mean()  # Aproximação
        
        if avg_value > overall_avg * 1.5:
            return 'High Value'
        elif avg_value > overall_avg * 1.2:
            return 'Premium'
        elif avg_value > overall_avg * 0.8:
            return 'Regular'
        else:
            return 'Budget'
    
    def _analyze_behavior_transitions(self, df: pd.DataFrame, clusters: np.ndarray) -> Dict[str, Any]:
        """Analisar transições comportamentais ao longo do tempo."""
        try:
            # Simplificado - análise básica de mudanças de cluster ao longo do tempo
            df_temp = df.copy()
            df_temp['Cluster'] = clusters[:len(df_temp)]
            
            monthly_clusters = df_temp.groupby([df_temp['Data'].dt.year, df_temp['Data'].dt.month])['Cluster'].apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
            
            transitions = {}
            if len(monthly_clusters) > 1:
                transitions['stability'] = (monthly_clusters.diff().fillna(0) == 0).sum() / len(monthly_clusters)
                transitions['most_common_cluster'] = int(monthly_clusters.mode().iloc[0])
            
            return transitions
            
        except Exception as e:
            return {'error': f"Erro na análise de transições: {str(e)}"}
    
    def _generate_future_forecast(self, models: Dict, X: pd.DataFrame, features: List[str], 
                                horizon: int, daily_sales: pd.DataFrame) -> Dict[str, Any]:
        """Gerar forecast futuro usando ensemble."""
        try:
            # Último registro para base da previsão
            last_date = daily_sales.index.max()
            
            predictions = []
            for i in range(horizon):
                future_date = last_date + timedelta(days=i+1)
                
                # Criar features para data futura
                future_features = {
                    'Dia_Semana': future_date.dayofweek,
                    'Mes': future_date.month,
                    'Dia_Mes': future_date.day,
                    'Sin_Month': np.sin(2 * np.pi * future_date.month / 12),
                    'Cos_Month': np.cos(2 * np.pi * future_date.month / 12)
                }
                
                # Usar últimos valores para lags (simplificado)
                for feature in features:
                    if 'lag' in feature or 'ma' in feature:
                        future_features[feature] = daily_sales.iloc[-1][feature] if feature in daily_sales.columns else 0
                
                # Criar DataFrame para predição
                future_df = pd.DataFrame([future_features])
                future_df = future_df.reindex(columns=features, fill_value=0)
                
                # Média das previsões dos modelos
                model_preds = []
                for model in models.values():
                    try:
                        pred = model.predict(future_df)[0]
                        model_preds.append(pred)
                    except:
                        continue
                
                if model_preds:
                    predictions.append(round(np.mean(model_preds), 2))
                else:
                    predictions.append(0)
            
            return {
                'daily_predictions': predictions,
                'total_forecast': round(sum(predictions), 2),
                'avg_daily_forecast': round(np.mean(predictions), 2),
                'forecast_dates': [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                                 for i in range(horizon)]
            }
            
        except Exception as e:
            return {'error': f"Erro no forecast futuro: {str(e)}"}
    
    def _generate_forecast_insights(self, forecast: Dict[str, Any], performance: Dict[str, float]) -> List[str]:
        """Gerar insights do forecast."""
        insights = []
        
        if 'total_forecast' in forecast:
            total = forecast['total_forecast']
            avg_daily = forecast['avg_daily_forecast']
            
            insights.append(f"Previsão total: R$ {total:,.2f}")
            insights.append(f"Média diária prevista: R$ {avg_daily:,.2f}")
            
            if performance['r2'] > 0.8:
                insights.append("Modelo com alta confiabilidade")
            elif performance['r2'] > 0.6:
                insights.append("Modelo com confiabilidade moderada")
            else:
                insights.append("Previsões com alta incerteza")
        
        return insights
    
    def _calculate_price_elasticity(self, data: pd.DataFrame) -> float:
        """Calcular elasticidade-preço da demanda."""
        try:
            if len(data) < 10:
                return -1.0  # Valor padrão
            
            # Usar regressão simples para estimar elasticidade
            price = data['Preco_Unitario']
            quantity = data['Quantidade'] if 'Quantidade' in data.columns else data.index
            
            # Log transformation para elasticidade
            log_price = np.log(price)
            log_quantity = np.log(quantity + 1)  # +1 para evitar log(0)
            
            # Regressão linear simples
            correlation = np.corrcoef(log_price, log_quantity)[0, 1]
            
            # Elasticidade aproximada
            elasticity = correlation * (log_quantity.std() / log_price.std())
            
            return max(-3.0, min(0.0, elasticity))  # Limitar entre -3 e 0
            
        except Exception as e:
            return -1.0
    
    def _simulate_price_optimization(self, data: pd.DataFrame, elasticity: float) -> Dict[str, Any]:
        """Simular otimização de preços."""
        try:
            current_price = data['Preco_Unitario'].mean()
            current_quantity = data['Quantidade'].sum() if 'Quantidade' in data.columns else len(data)
            current_revenue = data['Total_Liquido'].sum()
            
            # Simular diferentes aumentos de preço
            price_changes = [0.05, 0.10, 0.15, 0.20, -0.05, -0.10]  # 5%, 10%, 15%, 20%, -5%, -10%
            optimization_results = []
            
            for change in price_changes:
                new_price = current_price * (1 + change)
                
                # Calcular nova quantidade baseada na elasticidade
                quantity_change = elasticity * change
                new_quantity = current_quantity * (1 + quantity_change)
                new_revenue = new_price * new_quantity
                
                revenue_change = (new_revenue - current_revenue) / current_revenue * 100
                
                optimization_results.append({
                    'price_change_pct': round(change * 100, 1),
                    'new_price': round(new_price, 2),
                    'expected_quantity_change_pct': round(quantity_change * 100, 1),
                    'expected_revenue_change_pct': round(revenue_change, 1)
                })
            
            # Melhor opção
            best_option = max(optimization_results, key=lambda x: x['expected_revenue_change_pct'])
            
            return {
                'current_price': round(current_price, 2),
                'scenarios': optimization_results,
                'best_scenario': best_option,
                'potential_revenue_increase': round(best_option['expected_revenue_change_pct'], 1)
            }
            
        except Exception as e:
            return {'error': f"Erro na simulação: {str(e)}"}
    
    def _perform_abc_xyz_analysis(self, product_metrics: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Realizar análise ABC-XYZ."""
        try:
            # Análise ABC (por valor)
            revenue_col = f'{target_column}_sum'
            sorted_products = product_metrics.sort_values(revenue_col, ascending=False)
            
            cumsum_pct = sorted_products[revenue_col].cumsum() / sorted_products[revenue_col].sum()
            
            abc_class = []
            for pct in cumsum_pct:
                if pct <= 0.8:
                    abc_class.append('A')
                elif pct <= 0.95:
                    abc_class.append('B')
                else:
                    abc_class.append('C')
            
            sorted_products['ABC_Class'] = abc_class
            
            # Análise XYZ (por variabilidade)
            cv_quantiles = product_metrics['coefficient_variation'].quantile([0.33, 0.67])
            
            def classify_xyz(cv):
                if cv <= cv_quantiles.iloc[0]:
                    return 'X'  # Baixa variabilidade
                elif cv <= cv_quantiles.iloc[1]:
                    return 'Y'  # Média variabilidade
                else:
                    return 'Z'  # Alta variabilidade
            
            product_metrics['XYZ_Class'] = product_metrics['coefficient_variation'].apply(classify_xyz)
            
            # Combinar ABC-XYZ
            sorted_products['XYZ_Class'] = sorted_products.index.map(product_metrics['XYZ_Class'])
            sorted_products['ABC_XYZ'] = sorted_products['ABC_Class'] + sorted_products['XYZ_Class']
            
            # Análise das combinações
            abc_xyz_distribution = sorted_products['ABC_XYZ'].value_counts().to_dict()
            
            return {
                'abc_distribution': sorted_products['ABC_Class'].value_counts().to_dict(),
                'xyz_distribution': sorted_products['XYZ_Class'].value_counts().to_dict(),
                'abc_xyz_distribution': abc_xyz_distribution,
                'top_products_by_class': {
                    'AX': sorted_products[sorted_products['ABC_XYZ'] == 'AX'].index.tolist()[:5],
                    'AY': sorted_products[sorted_products['ABC_XYZ'] == 'AY'].index.tolist()[:5],
                    'AZ': sorted_products[sorted_products['ABC_XYZ'] == 'AZ'].index.tolist()[:5]
                }
            }
            
        except Exception as e:
            return {'error': f"Erro na análise ABC-XYZ: {str(e)}"}
    
    def _predict_product_demand(self, daily_demand: pd.Series, horizon: int) -> Dict[str, Any]:
        """Prever demanda individual de produto."""
        try:
            if len(daily_demand) < 7:
                return {'error': 'Dados insuficientes'}
            
            # Média móvel simples
            ma_7 = daily_demand.rolling(window=7).mean().iloc[-1]
            
            # Tendência simples
            recent_trend = (daily_demand.iloc[-3:].mean() - daily_demand.iloc[-7:-3].mean())
            
            # Previsão básica
            base_prediction = ma_7 + recent_trend
            
            predictions = [max(0, base_prediction + np.random.normal(0, daily_demand.std() * 0.1)) 
                         for _ in range(horizon)]
            
            return {
                'predictions': [round(p, 2) for p in predictions],
                'total_predicted': round(sum(predictions), 2),
                'avg_daily': round(np.mean(predictions), 2)
            }
            
        except Exception as e:
            return {'error': f"Erro na previsão de produto: {str(e)}"}
    
    def _optimize_inventory_levels(self, product_metrics: pd.Series, target_column: str) -> Dict[str, Any]:
        """Otimizar níveis de estoque para produto."""
        try:
            avg_daily_sales = product_metrics['avg_daily_sales']
            cv = product_metrics['coefficient_variation']
            
            # Lead time assumido (7 dias)
            lead_time = 7
            
            # Safety stock baseado na variabilidade
            if cv < 0.5:
                safety_factor = 1.2
            elif cv < 1.0:
                safety_factor = 1.5
            else:
                safety_factor = 2.0
            
            reorder_point = (avg_daily_sales * lead_time) * safety_factor
            optimal_order_quantity = avg_daily_sales * 30  # 30 dias de estoque
            
            return {
                'reorder_point': round(reorder_point, 0),
                'optimal_order_quantity': round(optimal_order_quantity, 0),
                'safety_stock_factor': safety_factor,
                'classification': 'High Priority' if cv > 1.0 else 'Standard'
            }
            
        except Exception as e:
            return {'error': f"Erro na otimização: {str(e)}"}
    
    def _format_advanced_result(self, analysis_type: str, result: Dict[str, Any]) -> str:
        """Formatar resultado da análise avançada."""
        try:
            timestamp = datetime.now().strftime('%d/%m/%Y %H:%M')
            
            if 'error' in result:
                return f"Erro na análise {analysis_type}: {result['error']}"
            
            formatted = f"""# 🤖 ANÁLISE AVANÇADA COM MACHINE LEARNING
                                ## Tipo: {analysis_type.upper().replace('_', ' ')}
                                **Data**: {timestamp}

                                ---

                                """
            
            # Formatação específica por tipo de análise
            if analysis_type == 'ml_insights':
                formatted += self._format_ml_insights(result)
            elif analysis_type == 'anomaly_detection':
                formatted += self._format_anomaly_results(result)
            elif analysis_type == 'customer_behavior':
                formatted += self._format_behavior_results(result)
            elif analysis_type == 'demand_forecasting':
                formatted += self._format_forecast_results(result)
            elif analysis_type == 'price_optimization':
                formatted += self._format_price_results(result)
            elif analysis_type == 'inventory_optimization':
                formatted += self._format_inventory_results(result)
            
            formatted += f"""

                            ---
                            ## 📋 METODOLOGIA

                            **Algoritmos Utilizados**: Random Forest, XGBoost, Isolation Forest, DBSCAN, K-means
                            **Validação**: Cross-validation e split temporal
                            **Confiabilidade**: {result.get('confidence_level', 'N/A')}

                            *Análise gerada pelo Advanced Analytics Engine - Insights AI*
                            """
            
            return formatted
            
        except Exception as e:
            return f"Erro na formatação: {str(e)}"
    
    def _format_ml_insights(self, result: Dict[str, Any]) -> str:
        """Formatar resultados de ML insights."""
        formatted = "## 📊 PERFORMANCE DOS MODELOS\n\n"
        
        if 'model_performance' in result:
            perf = result['model_performance']
            for model, metrics in perf.items():
                formatted += f"**{model}**: R² = {metrics.get('r2', 'N/A')}\n"
        
        formatted += "\n## 🎯 FEATURES MAIS IMPORTANTES\n\n"
        
        if 'feature_importance' in result:
            for i, (feature, importance) in enumerate(list(result['feature_importance'].items())[:5], 1):
                formatted += f"{i}. **{feature.replace('_', ' ').title()}**: {importance:.3f}\n"
        
        formatted += "\n## 🔮 PREVISÕES FUTURAS\n\n"
        
        if 'future_predictions' in result:
            pred = result['future_predictions']
            formatted += f"- **Total Previsto**: R$ {pred.get('total_predicted', 0):,.2f}\n"
            formatted += f"- **Média Diária**: R$ {pred.get('avg_daily', 0):,.2f}\n"
        
        formatted += "\n## 💡 INSIGHTS PRINCIPAIS\n\n"
        
        if 'model_insights' in result:
            for insight in result['model_insights']:
                formatted += f"- {insight}\n"
        
        return formatted
    
    def _format_anomaly_results(self, result: Dict[str, Any]) -> str:
        """Formatar resultados de detecção de anomalias."""
        formatted = "## 🚨 DETECÇÃO DE ANOMALIAS\n\n"
        
        if 'anomaly_summary' in result:
            summary = result['anomaly_summary']
            formatted += f"- **Total de Registros**: {summary.get('total_records', 0):,}\n"
            formatted += f"- **Anomalias por Consenso**: {summary.get('consensus_anomalies', 0)}\n"
            formatted += f"- **Taxa de Anomalias**: {summary.get('anomaly_rate_percent', 0)}%\n"
        
        formatted += "\n## 📅 PADRÕES TEMPORAIS\n\n"
        
        if 'temporal_patterns' in result:
            temp = result['temporal_patterns']
            if temp.get('anomalies_by_month'):
                formatted += "**Anomalias por Mês**:\n"
                for month, count in temp['anomalies_by_month'].items():
                    formatted += f"- Mês {month}: {count} anomalias\n"
        
        formatted += "\n## 🔍 TOP ANOMALIAS\n\n"
        
        if 'top_anomalies' in result and result['top_anomalies']:
            for i, anomaly in enumerate(result['top_anomalies'][:5], 1):
                formatted += f"{i}. **{anomaly.get('Data', 'N/A')}**: R$ {anomaly.get('Total_Liquido', 0):,.2f} (Z-Score: {anomaly.get('Z_Score', 0):.2f})\n"
        
        return formatted
    
    def _format_behavior_results(self, result: Dict[str, Any]) -> str:
        """Formatar resultados de análise comportamental."""
        formatted = "## 👥 SEGMENTAÇÃO COMPORTAMENTAL\n\n"
        
        if 'cluster_analysis' in result:
            for cluster, data in result['cluster_analysis'].items():
                formatted += f"### {cluster}\n"
                formatted += f"- **Tamanho**: {data.get('size', 0)} ({data.get('percentage', 0)}%)\n"
                formatted += f"- **Valor Médio**: R$ {data.get('avg_value', 0):,.2f}\n"
                formatted += f"- **Perfil**: {data.get('profile', 'N/A')}\n\n"
        
        formatted += "## 💡 INSIGHTS COMPORTAMENTAIS\n\n"
        
        if 'behavior_insights' in result:
            for insight in result['behavior_insights']:
                formatted += f"- {insight}\n"
        
        return formatted
    
    def _format_forecast_results(self, result: Dict[str, Any]) -> str:
        """Formatar resultados de forecasting."""
        formatted = "## 📈 PREVISÃO DE DEMANDA\n\n"
        
        if 'ensemble_performance' in result:
            perf = result['ensemble_performance']
            formatted += f"**Performance do Ensemble**:\n"
            formatted += f"- R²: {perf.get('r2', 0):.4f}\n"
            formatted += f"- MAE: {perf.get('mae', 0):,.2f}\n"
            formatted += f"- RMSE: {perf.get('rmse', 0):,.2f}\n\n"
        
        if 'future_forecast' in result:
            forecast = result['future_forecast']
            formatted += "**Previsão Futura**:\n"
            formatted += f"- Total Previsto: R$ {forecast.get('total_forecast', 0):,.2f}\n"
            formatted += f"- Média Diária: R$ {forecast.get('avg_daily_forecast', 0):,.2f}\n\n"
        
        if 'forecast_insights' in result:
            formatted += "## 💡 INSIGHTS DE PREVISÃO\n\n"
            for insight in result['forecast_insights']:
                formatted += f"- {insight}\n"
        
        return formatted
    
    def _format_price_results(self, result: Dict[str, Any]) -> str:
        """Formatar resultados de otimização de preços."""
        formatted = "## 💰 OTIMIZAÇÃO DE PREÇOS\n\n"
        
        if 'general_analysis' in result:
            general = result['general_analysis']
            formatted += f"**Análise Geral**:\n"
            formatted += f"- Preço Médio Atual: R$ {general.get('current_avg_price', 0):,.2f}\n"
            formatted += f"- Elasticidade: {general.get('price_elasticity', 0):.3f}\n"
            
            if 'optimization' in general and 'best_scenario' in general['optimization']:
                best = general['optimization']['best_scenario']
                formatted += f"- Melhor Cenário: {best.get('price_change_pct', 0)}% de mudança\n"
                formatted += f"- Potencial Aumento de Receita: {best.get('expected_revenue_change_pct', 0)}%\n\n"
        
        if 'category_analysis' in result:
            formatted += "**Análise por Categoria** (Top 3):\n"
            for i, (category, data) in enumerate(list(result['category_analysis'].items())[:3], 1):
                formatted += f"{i}. **{category}**: Elasticidade {data.get('price_elasticity', 0):.3f}\n"
        
        return formatted
    
    def _format_inventory_results(self, result: Dict[str, Any]) -> str:
        """Formatar resultados de otimização de inventário."""
        formatted = "## 📦 OTIMIZAÇÃO DE INVENTÁRIO\n\n"
        
        if 'abc_xyz_analysis' in result:
            abc_xyz = result['abc_xyz_analysis']
            formatted += "**Classificação ABC-XYZ**:\n"
            
            if 'abc_xyz_distribution' in abc_xyz:
                for class_type, count in list(abc_xyz['abc_xyz_distribution'].items())[:5]:
                    formatted += f"- {class_type}: {count} produtos\n"
        
        formatted += "\n**Performance de Produtos**:\n"
        
        if 'product_performance' in result:
            perf = result['product_performance']
            formatted += f"- Alto Desempenho: {len(perf.get('high_performers', []))} produtos\n"
            formatted += f"- Baixo Desempenho: {len(perf.get('low_performers', []))} produtos\n"
            formatted += f"- Alta Variação: {len(perf.get('high_variation', []))} produtos\n"
        
        if 'inventory_insights' in result:
            formatted += "\n## 💡 INSIGHTS DE INVENTÁRIO\n\n"
            for insight in result['inventory_insights']:
                formatted += f"- {insight}\n"
        
        return formatted
