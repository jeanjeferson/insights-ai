"""
🚀 ADVANCED ANALYTICS ENGINE V2.0 - Com Otimizações de Performance
==================================================================

Versão otimizada do Advanced Analytics Engine incluindo:
- Cache inteligente para resultados
- Paralelização de modelos ML
- Sampling estratificado para datasets grandes
- Detecção de data drift
"""

from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import logging
import os

# Imports das otimizações de performance
try:
    from .performance_optimizations import (
        CacheManager, ParallelProcessor, StratifiedSampler, DataDriftDetector
    )
    OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    OPTIMIZATIONS_AVAILABLE = False

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Imports opcionais para bibliotecas de ML
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("SciPy não disponível - algumas análises estatísticas serão limitadas")
    SCIPY_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.linear_model import ElasticNet
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("Scikit-learn não disponível - análises ML serão limitadas")
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class AdvancedAnalyticsEngineToolInput(BaseModel):
    """Schema de entrada para análises avançadas V2.0."""
    analysis_type: str = Field(..., description="Tipo de análise")
    data_csv: str = Field(default="data/vendas.csv", description="Caminho para o arquivo CSV")
    target_column: str = Field(default="Total_Liquido", description="Coluna alvo para ML")
    prediction_horizon: int = Field(default=30, description="Horizonte de predição em dias", ge=1, le=365)
    confidence_level: float = Field(default=0.95, description="Nível de confiança", ge=0.80, le=0.99)
    
    # Parâmetros de otimização
    enable_cache: bool = Field(default=True, description="Habilitar cache inteligente")
    enable_parallel: bool = Field(default=True, description="Habilitar processamento paralelo")
    enable_sampling: bool = Field(default=True, description="Habilitar sampling estratificado")
    enable_drift_detection: bool = Field(default=True, description="Habilitar detecção de data drift")
    max_sample_size: int = Field(default=50000, description="Tamanho máximo da amostra", ge=1000, le=200000)

class AdvancedAnalyticsEngineTool(BaseTool):
    """
    🤖 MOTOR DE ANÁLISES AVANÇADAS COM MACHINE LEARNING
    
    QUANDO USAR:
    - Análises complexas que requerem Machine Learning
    - Detecção de padrões ocultos nos dados de vendas
    - Previsões avançadas com múltiplos algoritmos
    - Otimização de processos de negócio
    - Análise comportamental de clientes
    
    CASOS DE USO ESPECÍFICOS:
    - ml_insights: Descobrir insights com Random Forest e XGBoost
    - anomaly_detection: Identificar vendas anômalas e outliers
    - customer_behavior: Segmentar clientes por comportamento
    - demand_forecasting: Prever demanda com ensemble de modelos
    - price_optimization: Otimizar preços baseado em elasticidade
    - inventory_optimization: Otimizar estoque com análise ABC
    
    RESULTADOS ENTREGUES:
    - Insights acionáveis baseados em ML
    - Previsões com intervalos de confiança
    - Segmentações automáticas de clientes
    - Recomendações de otimização
    - Métricas de performance dos modelos
    """
    
    name: str = "Advanced Analytics Engine"
    description: str = (
        "Motor de análises avançadas com Machine Learning para insights profundos de vendas. "
        "Use para descobrir padrões ocultos, prever demanda, detectar anomalias e otimizar processos. "
        "Combina múltiplos algoritmos ML para análises robustas e recomendações acionáveis."
    )
    args_schema: Type[BaseModel] = AdvancedAnalyticsEngineToolInput
    
    def __init__(self):
        super().__init__()
        # Inicializar componentes de otimização
        self._cache_manager = None
        self._parallel_processor = None
        self._sampler = None
        self._drift_detector = None
    
    def _run(self, analysis_type: str, data_csv: str = "data/vendas.csv", 
             target_column: str = "Total_Liquido", prediction_horizon: int = 30,
             confidence_level: float = 0.95, enable_cache: bool = True,
             enable_parallel: bool = True, enable_sampling: bool = True,
             enable_drift_detection: bool = True, max_sample_size: int = 50000) -> str:
        
        try:
            start_time = datetime.now()
            logger.info(f"🚀 Iniciando análise avançada V2.0: {analysis_type}")
            
            # Inicializar componentes de otimização se disponíveis
            if OPTIMIZATIONS_AVAILABLE:
                if enable_cache and self._cache_manager is None:
                    self._cache_manager = CacheManager()
                
                if enable_parallel and self._parallel_processor is None:
                    self._parallel_processor = ParallelProcessor()
                
                if enable_sampling and self._sampler is None:
                    self._sampler = StratifiedSampler(max_sample_size=max_sample_size)
                
                if enable_drift_detection and self._drift_detector is None:
                    self._drift_detector = DataDriftDetector()
            
            # Validação de entrada
            validation_result = self._validate_inputs(analysis_type, data_csv, target_column, 
                                                    prediction_horizon, confidence_level)
            if validation_result['error']:
                return f"❌ Erro de validação: {validation_result['message']}"
            
            # Carregar e validar dados
            df = self._load_data(data_csv)
            if df is None:
                return "❌ Erro: Falha ao carregar dados"
            
            # Preparar features avançadas
            df = self._prepare_features(df)
            if df is None:
                return "❌ Erro: Falha na preparação de features"
            
            # Validação final dos dados
            data_validation = self._validate_data_quality(df, target_column)
            if data_validation['error']:
                return f"❌ Erro de qualidade dos dados: {data_validation['message']}"
            
            logger.info(f"Dados validados: {len(df)} registros, {len(df.columns)} colunas")
            
            # Verificar cache
            cache_params = {
                'analysis_type': analysis_type,
                'target_column': target_column,
                'prediction_horizon': prediction_horizon,
                'confidence_level': confidence_level
            }
            
            if enable_cache and self._cache_manager:
                cached_result = self._cache_manager.get_cached_result(df, analysis_type, cache_params)
                if cached_result is not None:
                    logger.info("⚡ Cache hit - retornando resultado em cache")
                    return self._format_result(analysis_type, cached_result, start_time, cache_hit=True)
            
            # Aplicar sampling se necessário
            original_df = df
            sample_info = {'sampled': False, 'original_size': len(df)}
            
            if enable_sampling and self._sampler and self._sampler.should_sample(df):
                logger.info(f"📊 Aplicando sampling estratificado para {len(df)} registros")
                df, sample_info = self._sampler.create_stratified_sample(df, analysis_type)
                logger.info(f"✅ Amostra criada: {len(df)} registros")
            
            # Detectar data drift
            drift_info = {}
            if enable_drift_detection and self._drift_detector:
                try:
                    logger.info("🔍 Detectando data drift...")
                    drift_info = self._drift_detector.detect_drift(df)
                    if drift_info.get('drift_detected', False):
                        logger.warning(f"⚠️ Data drift detectado")
                except Exception as e:
                    logger.warning(f"Erro na detecção de drift: {e}")
                    drift_info = {'drift_detected': False, 'error': str(e)}
            

            
            # Executar análise
            result = self._execute_analysis(df, analysis_type, target_column, enable_parallel, 
                                          prediction_horizon, confidence_level)
            
            if 'error' in result:
                return f"❌ Erro na análise: {result['error']}"
            
            # Adicionar informações de performance
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            result['_performance_info'] = {
                'cache_hit': False,
                'execution_time_seconds': round(execution_time, 2),
                'sample_info': sample_info,
                'drift_info': drift_info,
                'version': '2.0'
            }
            
            # Salvar no cache
            if enable_cache and self._cache_manager and 'error' not in result:
                self._cache_manager.save_result_to_cache(original_df, analysis_type, cache_params, result)
            
            logger.info(f"🎉 Análise concluída em {execution_time:.2f}s: {analysis_type}")
            return self._format_advanced_result(analysis_type, result)
            
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            return f"❌ Erro inesperado na análise V2.0: {str(e)}"
    
    def _load_data(self, data_csv: str) -> Optional[pd.DataFrame]:
        """Carregar dados do CSV."""
        try:
            # Tentar diferentes separadores
            for sep in [';', ',', '\t']:
                try:
                    df = pd.read_csv(data_csv, sep=sep)
                    if len(df) > 0 and len(df.columns) > 1:
                        logger.info(f"✅ Dados carregados: {len(df)} linhas, {len(df.columns)} colunas")
                        return df
                except:
                    continue
            
            logger.error("Falha ao carregar dados")
            return None
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            return None
    
    def _validate_inputs(self, analysis_type: str, data_csv: str, target_column: str, 
                        prediction_horizon: int, confidence_level: float) -> Dict[str, Any]:
        """Validar parâmetros de entrada."""
        try:
            # Validar tipo de análise
            valid_analyses = ['ml_insights', 'anomaly_detection', 'customer_behavior', 
                            'demand_forecasting', 'price_optimization', 'inventory_optimization']
            if analysis_type not in valid_analyses:
                return {'error': True, 'message': f"Tipo de análise inválido. Opções: {valid_analyses}"}
            
            # Validar arquivo
            if not isinstance(data_csv, str) or not data_csv.strip():
                return {'error': True, 'message': "Caminho do arquivo CSV é obrigatório"}
            
            # Verificar se arquivo existe
            if not os.path.exists(data_csv):
                return {'error': True, 'message': f"Arquivo não encontrado: {data_csv}"}
            
            # Validar coluna alvo
            if not isinstance(target_column, str) or not target_column.strip():
                return {'error': True, 'message': "Nome da coluna alvo é obrigatório"}
            
            # Validar horizonte de predição
            if not isinstance(prediction_horizon, int) or prediction_horizon < 1 or prediction_horizon > 365:
                return {'error': True, 'message': "Horizonte de predição deve ser entre 1 e 365 dias"}
            
            # Validar nível de confiança
            if not isinstance(confidence_level, (int, float)) or confidence_level < 0.8 or confidence_level > 0.99:
                return {'error': True, 'message': "Nível de confiança deve ser entre 0.8 e 0.99"}
            
            return {'error': False, 'message': 'Validação bem-sucedida'}
            
        except Exception as e:
            return {'error': True, 'message': f"Erro na validação: {str(e)}"}
    
    def _validate_data_quality(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Validar qualidade dos dados."""
        try:
            # Verificar se dataset não está vazio
            if df is None or len(df) == 0:
                return {'error': True, 'message': 'Dataset está vazio'}
            
            # Verificar número mínimo de registros
            if len(df) < 50:
                return {'error': True, 'message': f'Dados insuficientes: {len(df)} registros (mínimo: 50)'}
            
            # Verificar se coluna alvo existe
            if target_column not in df.columns:
                available_cols = list(df.columns)
                return {'error': True, 'message': f'Coluna alvo "{target_column}" não encontrada. Disponíveis: {available_cols}'}
            
            # Verificar se coluna alvo tem dados válidos
            target_data = df[target_column].dropna()
            if len(target_data) == 0:
                return {'error': True, 'message': f'Coluna alvo "{target_column}" não possui dados válidos'}
            
            # Verificar se coluna alvo é numérica
            if not pd.api.types.is_numeric_dtype(df[target_column]):
                try:
                    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
                    target_data = df[target_column].dropna()
                    if len(target_data) == 0:
                        return {'error': True, 'message': f'Coluna alvo "{target_column}" não pode ser convertida para numérico'}
                except:
                    return {'error': True, 'message': f'Coluna alvo "{target_column}" deve ser numérica'}
            
            # Verificar variabilidade dos dados
            if target_data.std() == 0:
                return {'error': True, 'message': f'Coluna alvo "{target_column}" não possui variabilidade (todos valores iguais)'}
            
            # Verificar se há coluna de data
            date_columns = ['Data', 'data', 'DATE', 'date', 'Data_Venda', 'data_venda']
            has_date = any(col in df.columns for col in date_columns)
            if not has_date:
                logger.warning("Nenhuma coluna de data identificada - análises temporais podem ser limitadas")
            
            # Estatísticas de qualidade
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            logger.info(f"Qualidade dos dados: {missing_pct:.1f}% valores ausentes")
            
            return {'error': False, 'message': 'Dados validados com sucesso'}
            
        except Exception as e:
            return {'error': True, 'message': f'Erro na validação de qualidade: {str(e)}'}

    def _prepare_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Preparar features avançadas para Machine Learning."""
        try:
            logger.info("🔧 Iniciando preparação de features avançadas")
            original_rows = len(df)
            
            # Identificar coluna de data
            date_columns = ['Data', 'data', 'DATE', 'date', 'Data_Venda', 'data_venda']
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                logger.info(f"Coluna de data identificada: {date_col}")
                # Converter data com múltiplos formatos
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
                
                # Remover registros com datas inválidas
                before_date_filter = len(df)
                df = df.dropna(subset=[date_col])
                after_date_filter = len(df)
                
                if after_date_filter < before_date_filter:
                    logger.warning(f"Removidos {before_date_filter - after_date_filter} registros com datas inválidas")
                
                if len(df) == 0:
                    logger.error("Todos os registros foram removidos devido a datas inválidas")
                    return None
                
                # Padronizar nome da coluna
                if date_col != 'Data':
                    df['Data'] = df[date_col]
                
                # Features temporais avançadas
                try:
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
                    logger.info("Features temporais básicas criadas")
                except Exception as e:
                    logger.error(f"Erro ao criar features temporais: {e}")
                    return None
            else:
                logger.warning("Nenhuma coluna de data encontrada - features temporais não serão criadas")
            
            # Features sazonais (apenas se temos dados temporais)
            if date_col and 'Mes' in df.columns and 'Dia_Semana' in df.columns:
                try:
                    df['Sin_Month'] = np.sin(2 * np.pi * df['Mes'] / 12)
                    df['Cos_Month'] = np.cos(2 * np.pi * df['Mes'] / 12)
                    df['Sin_Day'] = np.sin(2 * np.pi * df['Dia_Semana'] / 7)
                    df['Cos_Day'] = np.cos(2 * np.pi * df['Dia_Semana'] / 7)
                    logger.info("Features sazonais criadas")
                except Exception as e:
                    logger.warning(f"Erro ao criar features sazonais: {e}")
            
            # Features de lag temporal (apenas se temos coluna alvo e data)
            if date_col and 'Total_Liquido' in df.columns:
                try:
                    df = df.sort_values('Data')
                    df['Total_Liquido_Lag1'] = df['Total_Liquido'].shift(1)
                    df['Total_Liquido_Lag7'] = df['Total_Liquido'].shift(7)
                    df['Total_Liquido_MA7'] = df['Total_Liquido'].rolling(window=7, min_periods=1).mean()
                    df['Total_Liquido_MA30'] = df['Total_Liquido'].rolling(window=30, min_periods=1).mean()
                    
                    # Features estatísticas móveis
                    df['Total_Liquido_Std7'] = df['Total_Liquido'].rolling(window=7, min_periods=1).std()
                    df['Total_Liquido_Min7'] = df['Total_Liquido'].rolling(window=7, min_periods=1).min()
                    df['Total_Liquido_Max7'] = df['Total_Liquido'].rolling(window=7, min_periods=1).max()
                    logger.info("Features de lag temporal criadas")
                except Exception as e:
                    logger.warning(f"Erro ao criar features de lag: {e}")
            
            # Features de categoria (se disponível)
            categorical_features = 0
            if 'Grupo_Produto' in df.columns:
                try:
                    le_grupo = LabelEncoder()
                    df['Grupo_Produto_Encoded'] = le_grupo.fit_transform(df['Grupo_Produto'].fillna('Outros'))
                    categorical_features += 1
                    logger.info("Feature Grupo_Produto codificada")
                except Exception as e:
                    logger.warning(f"Erro ao codificar Grupo_Produto: {e}")
            
            if 'Metal' in df.columns:
                try:
                    le_metal = LabelEncoder()
                    df['Metal_Encoded'] = le_metal.fit_transform(df['Metal'].fillna('Outros'))
                    categorical_features += 1
                    logger.info("Feature Metal codificada")
                except Exception as e:
                    logger.warning(f"Erro ao codificar Metal: {e}")
            
            # Features de interação
            if 'Quantidade' in df.columns and 'Total_Liquido' in df.columns:
                try:
                    # Evitar divisão por zero
                    df['Preco_Unitario'] = df['Total_Liquido'] / df['Quantidade'].replace(0, 1)
                    df['Log_Preco_Unitario'] = np.log1p(df['Preco_Unitario'].clip(lower=0))
                    logger.info("Features de interação criadas")
                except Exception as e:
                    logger.warning(f"Erro ao criar features de interação: {e}")
            
            # Limpeza final
            try:
                # Remover valores infinitos e NaN problemáticos
                df = df.replace([np.inf, -np.inf], np.nan)
                
                # Estatísticas finais
                final_rows = len(df)
                total_columns = len(df.columns)
                
                logger.info(f"Preparação concluída: {final_rows} registros, {total_columns} colunas totais")
                logger.info(f"Taxa de retenção de dados: {(final_rows/original_rows)*100:.1f}%")
                
                return df
                
            except Exception as e:
                logger.error(f"Erro na limpeza final: {e}")
                return df  # Retornar mesmo com erro na limpeza
            
        except Exception as e:
            logger.error(f"Erro crítico na preparação de features: {str(e)}")
            return None
    
    def _execute_analysis(self, df: pd.DataFrame, analysis_type: str, 
                         target_column: str, enable_parallel: bool, 
                         prediction_horizon: int = 30, confidence_level: float = 0.95) -> Dict[str, Any]:
        """Executar análise específica."""
        try:
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
                return {'error': f"Análise '{analysis_type}' não suportada. Opções: {list(advanced_analyses.keys())}"}
            
            # Executar análise
            logger.info(f"Executando análise: {analysis_type}")
            result = advanced_analyses[analysis_type](df, target_column, prediction_horizon, confidence_level)
            
            return result
                
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_ml_insights(self, df: pd.DataFrame, target_column: str, 
                            prediction_horizon: int, confidence_level: float) -> Dict[str, Any]:
        """Gerar insights usando Machine Learning."""
        try:
            logger.info("Iniciando geração de insights ML")
            
            # Verificar dependências
            if not SKLEARN_AVAILABLE:
                return {'error': 'Scikit-learn não disponível - instale com: pip install scikit-learn'}
            
            # Verificar se coluna alvo existe
            if target_column not in df.columns:
                return {'error': f'Coluna alvo "{target_column}" não encontrada'}
            
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
            
            # Adicionar features categóricas se disponíveis
            if 'Grupo_Produto_Encoded' in df.columns:
                available_features.append('Grupo_Produto_Encoded')
            if 'Metal_Encoded' in df.columns:
                available_features.append('Metal_Encoded')
            if 'Quantidade' in df.columns:
                available_features.append('Quantidade')
            
            logger.info(f"Features disponíveis para ML: {len(available_features)} - {available_features}")
            
            if len(available_features) < 2:
                return {'error': f'Features insuficientes para ML: {len(available_features)} (mínimo: 2)'}
            
            # Preparar dataset
            try:
                required_columns = available_features + [target_column]
                ml_df = df[required_columns].copy()
                
                # Remover linhas com valores ausentes
                initial_rows = len(ml_df)
                ml_df = ml_df.dropna()
                final_rows = len(ml_df)
                
                logger.info(f"Dataset ML: {final_rows} registros após limpeza ({initial_rows - final_rows} removidos)")
                
                if final_rows < 30:
                    return {'error': f'Dados insuficientes após limpeza: {final_rows} registros (mínimo: 30)'}
                
            except Exception as e:
                return {'error': f'Erro na preparação do dataset: {str(e)}'}
            
            # Separar features e target
            X = ml_df[available_features]
            y = ml_df[target_column]
            
            # Verificar variabilidade do target
            if y.std() == 0:
                return {'error': 'Coluna alvo não possui variabilidade (todos valores iguais)'}
            
            # Dividir dados
            try:
                test_size = min(0.3, max(0.1, 20/len(ml_df)))  # Adaptativo baseado no tamanho
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                logger.info(f"Divisão dos dados: {len(X_train)} treino, {len(X_test)} teste")
                
            except Exception as e:
                return {'error': f'Erro na divisão dos dados: {str(e)}'}
            
            # Resultados dos modelos
            models_results = {}
            
            # 1. Random Forest
            try:
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                rf_model.fit(X_train, y_train)
                rf_pred = rf_model.predict(X_test)
                rf_score = r2_score(y_test, rf_pred)
                rf_mae = mean_absolute_error(y_test, rf_pred)
                
                # Feature importance
                feature_importance = dict(zip(available_features, rf_model.feature_importances_))
                feature_importance = {k: round(v, 4) for k, v in 
                                    sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)}
                
                models_results['random_forest'] = {
                    'r2_score': round(rf_score, 4),
                    'mae': round(rf_mae, 2),
                    'feature_importance': feature_importance
                }
                logger.info(f"Random Forest R²: {rf_score:.4f}")
                
            except Exception as e:
                logger.error(f"Erro no Random Forest: {e}")
                models_results['random_forest'] = {'error': str(e)}
            
            # 2. XGBoost (se disponível)
            if XGBOOST_AVAILABLE:
                try:
                    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=-1)
                    xgb_model.fit(X_train, y_train)
                    xgb_pred = xgb_model.predict(X_test)
                    xgb_score = r2_score(y_test, xgb_pred)
                    xgb_mae = mean_absolute_error(y_test, xgb_pred)
                    
                    models_results['xgboost'] = {
                        'r2_score': round(xgb_score, 4),
                        'mae': round(xgb_mae, 2)
                    }
                    logger.info(f"XGBoost R²: {xgb_score:.4f}")
                    
                    # 3. Ensemble prediction
                    if 'random_forest' in models_results and 'error' not in models_results['random_forest']:
                        ensemble_pred = (rf_pred + xgb_pred) / 2
                        ensemble_score = r2_score(y_test, ensemble_pred)
                        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                        
                        models_results['ensemble'] = {
                            'r2_score': round(ensemble_score, 4),
                            'mae': round(ensemble_mae, 2)
                        }
                        logger.info(f"Ensemble R²: {ensemble_score:.4f}")
                    
                except Exception as e:
                    logger.error(f"Erro no XGBoost: {e}")
                    models_results['xgboost'] = {'error': str(e)}
            else:
                logger.info("XGBoost não disponível - usando apenas Random Forest")
                models_results['xgboost'] = {'error': 'XGBoost não instalado'}
            
            # Validação cruzada (apenas se Random Forest funcionou)
            cv_results = {}
            if 'random_forest' in models_results and 'error' not in models_results['random_forest']:
                try:
                    cv_scores = cross_val_score(rf_model, X, y, cv=min(5, len(X)//10), scoring='r2')
                    cv_results = {
                        'mean_cv_score': round(cv_scores.mean(), 4),
                        'std_cv_score': round(cv_scores.std(), 4),
                        'cv_scores': [round(score, 4) for score in cv_scores]
                    }
                    logger.info(f"CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                except Exception as e:
                    logger.warning(f"Erro na validação cruzada: {e}")
                    cv_results = {'error': str(e)}
            
            # Insights baseados em ML
            ml_insights = []
            
            # Verificar se temos resultados válidos
            if 'random_forest' in models_results and 'error' not in models_results['random_forest']:
                feature_importance = models_results['random_forest']['feature_importance']
                
                # Feature mais importante
                if feature_importance:
                    top_feature = list(feature_importance.keys())[0]
                    top_importance = list(feature_importance.values())[0]
                    ml_insights.append(f"Fator mais importante: {top_feature} (importância: {top_importance:.3f})")
                
                # Performance do modelo
                best_score = 0
                if 'ensemble' in models_results:
                    best_score = models_results['ensemble']['r2_score']
                elif 'random_forest' in models_results:
                    best_score = models_results['random_forest']['r2_score']
                
                if best_score > 0.8:
                    ml_insights.append("Modelo com alta precisão - padrões bem definidos nos dados")
                elif best_score > 0.6:
                    ml_insights.append("Modelo com precisão moderada - padrões identificáveis")
                else:
                    ml_insights.append("Padrões complexos - alta variabilidade nos dados")
                
                # Previsões para período futuro
                try:
                    future_predictions = self._generate_future_predictions(
                        rf_model, X, available_features, prediction_horizon
                    )
                except Exception as e:
                    logger.warning(f"Erro ao gerar previsões futuras: {e}")
                    future_predictions = {'error': str(e)}
                
                # Intervalos de predição
                try:
                    if 'ensemble' in models_results:
                        prediction_intervals = self._calculate_prediction_intervals(
                            ensemble_pred, y_test, confidence_level
                        )
                    else:
                        prediction_intervals = self._calculate_prediction_intervals(
                            rf_pred, y_test, confidence_level
                        )
                except Exception as e:
                    logger.warning(f"Erro ao calcular intervalos de predição: {e}")
                    prediction_intervals = {'error': str(e)}
            else:
                ml_insights.append("Não foi possível treinar modelos ML com os dados disponíveis")
                future_predictions = {'error': 'Modelos não disponíveis'}
                prediction_intervals = {'error': 'Modelos não disponíveis'}
            
            # Compilar resultados finais
            result = {
                'models_results': models_results,
                'cross_validation': cv_results,
                'future_predictions': future_predictions,
                'model_insights': ml_insights,
                'prediction_intervals': prediction_intervals,
                'data_summary': {
                    'total_records': len(ml_df),
                    'features_used': len(available_features),
                    'feature_names': available_features
                }
            }
            
            logger.info("Insights ML gerados com sucesso")
            return result
            
        except Exception as e:
            logger.error(f"Erro crítico no ML insights: {e}")
            return {'error': f"Erro no ML insights: {str(e)}"}
    
    def _ml_insights(self, df: pd.DataFrame, target_column: str, enable_parallel: bool) -> Dict[str, Any]:
        """Wrapper para compatibilidade - redireciona para _generate_ml_insights."""
        return self._generate_ml_insights(df, target_column, 30, 0.95)
    
    def _train_sequential(self, models: Dict[str, Any], X_train: pd.DataFrame,
                         y_train: pd.Series, X_test: pd.DataFrame,
                         y_test: pd.Series) -> Dict[str, Dict]:
        """Treinamento sequencial de modelos."""
        results = {}
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                metrics = {
                    'mae': round(mean_absolute_error(y_test, predictions), 2),
                    'rmse': round(np.sqrt(mean_squared_error(y_test, predictions)), 2),
                    'r2': round(r2_score(y_test, predictions), 4)
                }
                
                results[name] = metrics
                logger.info(f"✅ {name} - R²: {metrics['r2']}")
                
            except Exception as e:
                results[name] = {'error': str(e)}
        
        return results
    
    def _demand_forecasting(self, df: pd.DataFrame, target_column: str, enable_parallel: bool) -> Dict[str, Any]:
        """Previsão de demanda simplificada."""
        try:
            # Agregar por dia
            if 'Data' not in df.columns:
                return {'error': 'Coluna Data não encontrada'}
            
            daily_sales = df.groupby('Data')[target_column].sum().fillna(0)
            
            if len(daily_sales) < 30:
                return {'error': 'Dados insuficientes para forecasting'}
            
            # Estatísticas básicas
            mean_sales = daily_sales.mean()
            std_sales = daily_sales.std()
            trend = daily_sales.tail(7).mean() - daily_sales.head(7).mean()
            
            return {
                'forecast_summary': {
                    'mean_daily_sales': round(mean_sales, 2),
                    'std_daily_sales': round(std_sales, 2),
                    'trend': 'crescente' if trend > 0 else 'decrescente',
                    'trend_value': round(trend, 2)
                },
                'forecast_insights': [
                    f"Vendas médias diárias: {mean_sales:.2f}",
                    f"Variabilidade: {std_sales:.2f}",
                    f"Tendência: {trend:.2f}"
                ]
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _format_result(self, analysis_type: str, result: Dict[str, Any], 
                      start_time: datetime, cache_hit: bool = False) -> str:
        """Formatar resultado da análise."""
        try:
            timestamp = datetime.now().strftime('%d/%m/%Y %H:%M')
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if 'error' in result:
                return f"❌ Erro na análise {analysis_type}: {result['error']}"
            
            # Informações de performance
            perf_info = result.get('_performance_info', {})
            
            formatted = f"""# 🚀 ANÁLISE AVANÇADA V2.0 - OTIMIZADA
                            ## Tipo: {analysis_type.upper().replace('_', ' ')}
                            **Data**: {timestamp} | **Tempo**: {execution_time:.2f}s

                            ---

                            ## ⚡ OTIMIZAÇÕES APLICADAS
                            """
            
            if cache_hit:
                formatted += "- **Cache Inteligente**: 🎯 HIT\n"
            else:
                formatted += "- **Cache Inteligente**: 💾 MISS\n"
            
            sample_info = perf_info.get('sample_info', {})
            if sample_info.get('sampled', False):
                ratio = sample_info.get('sample_ratio', 0)
                formatted += f"- **Sampling Estratificado**: ✅ {ratio:.1%} dos dados\n"
            
            drift_info = perf_info.get('drift_info', {})
            if drift_info:
                drift_status = "⚠️ DETECTADO" if drift_info.get('drift_detected') else "✅ ESTÁVEL"
                formatted += f"- **Detecção de Drift**: {drift_status}\n"
            
            # Resultados específicos
            formatted += f"""

                        ---

                        ## 📈 RESULTADOS DA ANÁLISE
                        """
                                    
            if analysis_type == 'ml_insights':
                models_results = result.get('models_results', {})
                if models_results:
                    formatted += "**Performance dos Modelos**:\n"
                    for model_name, metrics in models_results.items():
                        if 'error' not in metrics and 'r2' in metrics:
                            formatted += f"- **{model_name}**: R² = {metrics['r2']:.4f}, MAE = {metrics['mae']}\n"
            
            elif analysis_type == 'demand_forecasting':
                forecast_summary = result.get('forecast_summary', {})
                if forecast_summary:
                    formatted += f"**Resumo da Previsão**:\n"
                    formatted += f"- Vendas médias diárias: {forecast_summary.get('mean_daily_sales', 0):.2f}\n"
                    formatted += f"- Tendência: {forecast_summary.get('trend', 'N/A')}\n"
            
            formatted += f"""

                        ---
                        ## 🔧 INFORMAÇÕES TÉCNICAS

                        **Versão**: Advanced Analytics Engine V2.0
                        **Performance**: {execution_time:.2f}s de execução
                        **Otimizações**: Cache, Paralelização, Sampling, Drift Detection

                        *Análise gerada pelo Advanced Analytics Engine V2.0 - Insights AI*
                        """
            
            return formatted
            
        except Exception as e:
            return f"❌ Erro na formatação: {str(e)}"
    
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
            if SCIPY_AVAILABLE:
                from scipy.stats import norm
                alpha = 1 - confidence_level
                z_score = norm.ppf(1 - alpha/2)
            else:
                # Aproximação para z-score comum
                z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
                z_score = z_scores.get(confidence_level, 1.96)
            
            margin_error = z_score * residual_std
            
            return {
                'lower_bound': round(np.mean(predictions) - margin_error, 2),
                'upper_bound': round(np.mean(predictions) + margin_error, 2),
                'margin_error': round(margin_error, 2),
                'confidence_level': confidence_level
            }
            
        except Exception as e:
            return {'error': f"Erro no cálculo de intervalos: {str(e)}"}
    
    def _advanced_anomaly_detection(self, df: pd.DataFrame, target_column: str,
                                   prediction_horizon: int, confidence_level: float) -> Dict[str, Any]:
        """Detecção avançada de anomalias usando múltiplos algoritmos."""
        try:
            if not SKLEARN_AVAILABLE:
                return {'error': 'Scikit-learn não disponível para detecção de anomalias'}
            
            # Preparar dados
            anomaly_features = [target_column]
            if 'Quantidade' in df.columns:
                anomaly_features.append('Quantidade')
            
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
            if SCIPY_AVAILABLE:
                z_scores = np.abs(stats.zscore(anomaly_df[target_column]))
            else:
                # Z-score manual
                mean_val = anomaly_df[target_column].mean()
                std_val = anomaly_df[target_column].std()
                z_scores = np.abs((anomaly_df[target_column] - mean_val) / std_val)
            
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
            if 'Data' in df.columns:
                anomaly_df['Data'] = df['Data'].iloc[:len(anomaly_df)]
            if 'Codigo_Produto' in df.columns:
                anomaly_df['Codigo_Produto'] = df['Codigo_Produto'].iloc[:len(anomaly_df)]
            
            # Identificar top anomalias
            consensus_anomalies = anomaly_df[anomaly_df['Consensus_Anomaly']]
            top_anomalies = consensus_anomalies.nlargest(10, 'Z_Score') if len(consensus_anomalies) > 0 else pd.DataFrame()
            
            # Análise temporal de anomalias
            anomaly_by_month = {}
            anomaly_by_weekday = {}
            if len(consensus_anomalies) > 0 and 'Data' in anomaly_df.columns:
                try:
                    anomaly_by_month = consensus_anomalies.groupby(
                        consensus_anomalies['Data'].dt.month
                    ).size().to_dict()
                    anomaly_by_weekday = consensus_anomalies.groupby(
                        consensus_anomalies['Data'].dt.dayofweek
                    ).size().to_dict()
                except:
                    pass
            
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
                    col for col in ['Data', target_column, 'Z_Score', 'ISO_Score'] if col in top_anomalies.columns
                ]].to_dict('records') if len(top_anomalies) > 0 else [],
                'insights': anomaly_insights
            }
            
        except Exception as e:
            return {'error': f"Erro na detecção de anomalias: {str(e)}"}
    
    def _advanced_customer_behavior(self, df: pd.DataFrame, target_column: str,
                                  prediction_horizon: int, confidence_level: float) -> Dict[str, Any]:
        """Análise comportamental avançada usando ML."""
        try:
            if not SKLEARN_AVAILABLE:
                return {'error': 'Scikit-learn não disponível para análise comportamental'}
            
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
    
    def _classify_behavior_cluster(self, cluster_data: pd.DataFrame, target_column: str) -> str:
        """Classificar cluster comportamental."""
        try:
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
        except:
            return 'Unknown'
    
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
    
    def _ml_demand_forecasting(self, df: pd.DataFrame, target_column: str,
                             prediction_horizon: int, confidence_level: float) -> Dict[str, Any]:
        """Previsão de demanda usando ensemble de modelos ML."""
        try:
            if not SKLEARN_AVAILABLE:
                return {'error': 'Scikit-learn não disponível para forecasting'}
            
            # Preparar dados temporais
            df_forecast = df.copy()
            
            if 'Data' not in df_forecast.columns:
                return {'error': 'Coluna Data necessária para forecasting'}
            
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
            feature_cols = [col for col in forecast_df.columns if col != target_column and ('lag' in col or 'ma' in col or col in ['Dia_Semana', 'Mes', 'Sin_Month', 'Cos_Month'])]
            
            X = forecast_df[feature_cols]
            y = forecast_df[target_column]
            
            # Split temporal (últimos 20% para teste)
            split_point = int(len(X) * 0.8)
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            # Modelos ensemble
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Elastic Net': ElasticNet(random_state=42)
            }
            
            if XGBOOST_AVAILABLE:
                models['XGBoost'] = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
            
            # Treinar modelos
            model_performance = {}
            predictions = {}
            
            for name, model in models.items():
                try:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    
                    model_performance[name] = {
                        'mae': round(mean_absolute_error(y_test, pred), 2),
                        'rmse': round(np.sqrt(mean_squared_error(y_test, pred)), 2),
                        'r2': round(r2_score(y_test, pred), 4)
                    }
                    predictions[name] = pred
                except Exception as e:
                    model_performance[name] = {'error': str(e)}
            
            # Ensemble prediction (média ponderada baseada em performance)
            weights = [model_performance[name].get('r2', 0) for name in models.keys() if 'error' not in model_performance[name]]
            total_weight = sum(weights) if weights else 1
            
            if total_weight > 0 and len(predictions) > 0:
                ensemble_pred = sum(pred * weight / total_weight 
                                  for pred, weight in zip(predictions.values(), weights) if weight > 0)
                if len(ensemble_pred) == 0:
                    ensemble_pred = np.mean(list(predictions.values()), axis=0)
            else:
                ensemble_pred = np.mean(list(predictions.values()), axis=0) if predictions else np.zeros(len(y_test))
            
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
            
            if performance.get('r2', 0) > 0.8:
                insights.append("Modelo com alta confiabilidade")
            elif performance.get('r2', 0) > 0.6:
                insights.append("Modelo com confiabilidade moderada")
            else:
                insights.append("Previsões com alta incerteza")
        
        return insights
    
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
                    
                    # Calcular elasticidade usando correlação simples
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
    
    def _calculate_price_elasticity(self, data: pd.DataFrame) -> float:
        """Calcular elasticidade-preço da demanda."""
        try:
            if len(data) < 10:
                return -1.0  # Valor padrão
            
            # Usar correlação simples entre preço e quantidade
            price = data['Preco_Unitario']
            quantity = data['Quantidade'] if 'Quantidade' in data.columns else data.index
            
            # Correlação de Pearson
            correlation = np.corrcoef(price, quantity)[0, 1]
            
            # Elasticidade aproximada
            elasticity = correlation * (quantity.std() / price.std()) if price.std() > 0 else -1.0
            
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
                'Data': ['min', 'max'] if 'Data' in df_inventory.columns else 'count'
            }).fillna(0)
            
            # Flatten columns
            product_metrics.columns = ['_'.join(col).strip() for col in product_metrics.columns]
            
            # Calcular métricas de inventário
            if 'Data_min' in product_metrics.columns and 'Data_max' in product_metrics.columns:
                product_metrics['days_active'] = (
                    pd.to_datetime(product_metrics['Data_max']) - 
                    pd.to_datetime(product_metrics['Data_min'])
                ).dt.days + 1
                
                product_metrics['avg_daily_sales'] = (
                    product_metrics[f'{target_column}_sum'] / product_metrics['days_active']
                )
            else:
                product_metrics['avg_daily_sales'] = product_metrics[f'{target_column}_mean']
            
            product_metrics['coefficient_variation'] = (
                product_metrics[f'{target_column}_std'] / product_metrics[f'{target_column}_mean']
            ).fillna(0)
            
            # Classificação ABC simplificada
            abc_xyz_classification = self._perform_abc_analysis(product_metrics, target_column)
            
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
                'abc_analysis': abc_xyz_classification,
                'product_performance': {
                    'high_performers': high_turnover.index.tolist()[:10],
                    'low_performers': low_turnover.index.tolist()[:10],
                    'high_variation': high_variation.index.tolist()[:10]
                },
                'inventory_insights': inventory_insights,
                'summary_stats': {
                    'total_products': len(product_metrics),
                    'avg_daily_sales': round(product_metrics['avg_daily_sales'].mean(), 2),
                    'avg_variation': round(product_metrics['coefficient_variation'].mean(), 2)
                }
            }
            
        except Exception as e:
            return {'error': f"Erro na otimização de inventário: {str(e)}"}
    
    def _perform_abc_analysis(self, product_metrics: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Realizar análise ABC simplificada."""
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
            
            # Análise das distribuições
            abc_distribution = sorted_products['ABC_Class'].value_counts().to_dict()
            
            return {
                'abc_distribution': abc_distribution,
                'top_products_class_a': sorted_products[sorted_products['ABC_Class'] == 'A'].index.tolist()[:10]
            }
            
        except Exception as e:
            return {'error': f"Erro na análise ABC: {str(e)}"}
    
    # Atualizar método _format_result para usar _format_advanced_result
    def _format_advanced_result(self, analysis_type: str, result: Dict[str, Any]) -> str:
        """Formatar resultado da análise avançada."""
        try:
            timestamp = datetime.now().strftime('%d/%m/%Y %H:%M')
            
            if 'error' in result:
                return f"❌ Erro na análise {analysis_type}: {result['error']}"
            
            formatted = f"""# 🤖 ANÁLISE AVANÇADA COM MACHINE LEARNING V2.0
## Tipo: {analysis_type.upper().replace('_', ' ')}
**Data**: {timestamp}

---

"""
            
            # Formatação específica por tipo de análise
            if analysis_type == 'ml_insights':
                formatted += self._format_ml_insights_results(result)
            elif analysis_type == 'anomaly_detection':
                formatted += self._format_anomaly_results_v2(result)
            elif analysis_type == 'customer_behavior':
                formatted += self._format_behavior_results_v2(result)
            elif analysis_type == 'demand_forecasting':
                formatted += self._format_forecast_results_v2(result)
            elif analysis_type == 'price_optimization':
                formatted += self._format_price_results_v2(result)
            elif analysis_type == 'inventory_optimization':
                formatted += self._format_inventory_results_v2(result)
            
            formatted += f"""

---
## 📋 METODOLOGIA V2.0

**Algoritmos Utilizados**: Random Forest, XGBoost, Isolation Forest, DBSCAN, K-means
**Validação**: Cross-validation e split temporal
**Otimizações**: Cache, Paralelização, Sampling, Drift Detection

*Análise gerada pelo Advanced Analytics Engine V2.0 - Insights AI*
"""
            
            return formatted
            
        except Exception as e:
            return f"❌ Erro na formatação: {str(e)}"
    
    def _format_ml_insights_results(self, result: Dict[str, Any]) -> str:
        """Formatar resultados de ML insights."""
        formatted = "## 📊 PERFORMANCE DOS MODELOS\n\n"
        
        if 'models_results' in result:
            for model, metrics in result['models_results'].items():
                if 'error' not in metrics:
                    formatted += f"**{model}**: R² = {metrics.get('r2_score', 'N/A')}, MAE = {metrics.get('mae', 'N/A')}\n"
        
        if 'model_insights' in result:
            formatted += "\n## 💡 INSIGHTS PRINCIPAIS\n\n"
            for insight in result['model_insights']:
                formatted += f"- {insight}\n"
        
        return formatted
    
    def _format_anomaly_results_v2(self, result: Dict[str, Any]) -> str:
        """Formatar resultados de detecção de anomalias."""
        formatted = "## 🚨 DETECÇÃO DE ANOMALIAS\n\n"
        
        if 'anomaly_summary' in result:
            summary = result['anomaly_summary']
            formatted += f"- **Anomalias Detectadas**: {summary.get('consensus_anomalies', 0)}\n"
            formatted += f"- **Taxa de Anomalias**: {summary.get('anomaly_rate_percent', 0)}%\n"
        
        return formatted
    
    def _format_behavior_results_v2(self, result: Dict[str, Any]) -> str:
        """Formatar resultados de análise comportamental."""
        formatted = "## 👥 SEGMENTAÇÃO COMPORTAMENTAL\n\n"
        
        if 'cluster_analysis' in result:
            for cluster, data in result['cluster_analysis'].items():
                formatted += f"**{cluster}**: {data.get('size', 0)} registros ({data.get('percentage', 0)}%)\n"
        
        return formatted
    
    def _format_forecast_results_v2(self, result: Dict[str, Any]) -> str:
        """Formatar resultados de forecasting."""
        formatted = "## 📈 PREVISÃO DE DEMANDA\n\n"
        
        if 'future_forecast' in result:
            forecast = result['future_forecast']
            formatted += f"- **Total Previsto**: R$ {forecast.get('total_forecast', 0):,.2f}\n"
            formatted += f"- **Média Diária**: R$ {forecast.get('avg_daily_forecast', 0):,.2f}\n"
        
        return formatted
    
    def _format_price_results_v2(self, result: Dict[str, Any]) -> str:
        """Formatar resultados de otimização de preços."""
        formatted = "## 💰 OTIMIZAÇÃO DE PREÇOS\n\n"
        
        if 'general_analysis' in result:
            general = result['general_analysis']
            formatted += f"- **Preço Médio Atual**: R$ {general.get('current_avg_price', 0):,.2f}\n"
            formatted += f"- **Elasticidade**: {general.get('price_elasticity', 0):.3f}\n"
        
        return formatted
    
    def _format_inventory_results_v2(self, result: Dict[str, Any]) -> str:
        """Formatar resultados de otimização de inventário."""
        formatted = "## 📦 OTIMIZAÇÃO DE INVENTÁRIO\n\n"
        
        if 'summary_stats' in result:
            stats = result['summary_stats']
            formatted += f"- **Total de Produtos**: {stats.get('total_products', 0)}\n"
            formatted += f"- **Vendas Médias Diárias**: {stats.get('avg_daily_sales', 0):.2f}\n"
        
        return formatted