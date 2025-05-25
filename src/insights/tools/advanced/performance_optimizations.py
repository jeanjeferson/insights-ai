"""
🚀 PERFORMANCE OPTIMIZATIONS - Fase 2 Advanced Analytics Engine
================================================================

Sistema completo de otimizações de performance incluindo:
- Cache inteligente com invalidação automática
- Paralelização de modelos ML
- Sampling estratificado para datasets grandes
- Detecção de data drift
"""

import pandas as pd
import numpy as np
import hashlib
import pickle
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import wraps
import threading
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logger = logging.getLogger(__name__)

class CacheManager:
    """Sistema de cache inteligente para resultados de análises."""
    
    def __init__(self, cache_dir: str = "cache/analytics"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index = self._load_cache_index()
        self.max_cache_size_mb = 500  # 500MB máximo
        
    def _load_cache_index(self) -> Dict[str, Dict]:
        """Carregar índice do cache."""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache_index(self):
        """Salvar índice do cache."""
        index_file = self.cache_dir / "cache_index.json"
        with open(index_file, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
    
    def _generate_cache_key(self, data_hash: str, analysis_type: str, 
                          params: Dict[str, Any]) -> str:
        """Gerar chave única para cache."""
        # Criar hash dos parâmetros
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        
        return f"{analysis_type}_{data_hash}_{params_hash}"
    
    def _calculate_data_hash(self, df: pd.DataFrame) -> str:
        """Calcular hash dos dados para detecção de mudanças."""
        # Usar amostra dos dados para performance
        sample_size = min(1000, len(df))
        sample_df = df.sample(sample_size) if len(df) > sample_size else df
        
        # Hash baseado em shape, colunas e amostra de valores
        data_info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'sample_values': sample_df.to_dict()
        }
        
        data_str = json.dumps(data_info, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()[:16]
    
    def get_cached_result(self, df: pd.DataFrame, analysis_type: str, 
                         params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Recuperar resultado do cache se disponível."""
        try:
            data_hash = self._calculate_data_hash(df)
            cache_key = self._generate_cache_key(data_hash, analysis_type, params)
            
            if cache_key in self.cache_index:
                cache_info = self.cache_index[cache_key]
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                
                if cache_file.exists():
                    # Verificar se cache não expirou (24 horas)
                    cache_time = datetime.fromisoformat(cache_info['timestamp'])
                    if datetime.now() - cache_time < timedelta(hours=24):
                        with open(cache_file, 'rb') as f:
                            result = pickle.load(f)
                        
                        logger.info(f"Cache hit para {analysis_type}")
                        return result
                    else:
                        # Cache expirado
                        self._remove_cache_entry(cache_key)
            
            return None
            
        except Exception as e:
            logger.warning(f"Erro ao recuperar cache: {e}")
            return None
    
    def save_result_to_cache(self, df: pd.DataFrame, analysis_type: str,
                           params: Dict[str, Any], result: Dict[str, Any]):
        """Salvar resultado no cache."""
        try:
            # Verificar tamanho do cache
            self._cleanup_cache_if_needed()
            
            data_hash = self._calculate_data_hash(df)
            cache_key = self._generate_cache_key(data_hash, analysis_type, params)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            # Salvar resultado
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            # Atualizar índice
            self.cache_index[cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': analysis_type,
                'data_hash': data_hash,
                'file_size': cache_file.stat().st_size,
                'params': params
            }
            
            self._save_cache_index()
            logger.info(f"Resultado salvo no cache: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Erro ao salvar no cache: {e}")
    
    def _cleanup_cache_if_needed(self):
        """Limpar cache se necessário."""
        total_size = sum(info.get('file_size', 0) for info in self.cache_index.values())
        max_size_bytes = self.max_cache_size_mb * 1024 * 1024
        
        if total_size > max_size_bytes:
            # Remover entradas mais antigas
            sorted_entries = sorted(
                self.cache_index.items(),
                key=lambda x: x[1]['timestamp']
            )
            
            for cache_key, _ in sorted_entries[:len(sorted_entries)//4]:
                self._remove_cache_entry(cache_key)
    
    def _remove_cache_entry(self, cache_key: str):
        """Remover entrada do cache."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            cache_file.unlink()
        
        if cache_key in self.cache_index:
            del self.cache_index[cache_key]
    
    def clear(self):
        """Limpar todo o cache."""
        try:
            # Remover todos os arquivos de cache
            for cache_key in list(self.cache_index.keys()):
                self._remove_cache_entry(cache_key)
            
            # Limpar o índice
            self.cache_index.clear()
            
            # Salvar índice vazio
            self._save_cache_index()
            
            logger.info("Cache limpo completamente")
            
        except Exception as e:
            logger.warning(f"Erro ao limpar cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obter estatísticas do cache."""
        try:
            total_entries = len(self.cache_index)
            total_size = sum(info.get('file_size', 0) for info in self.cache_index.values())
            
            return {
                'total_entries': total_entries,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'cache_dir': str(self.cache_dir),
                'max_size_mb': self.max_cache_size_mb
            }
        except Exception as e:
            return {'error': str(e)}


class ParallelProcessor:
    """Sistema de paralelização para modelos ML e análises."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(4, os.cpu_count())
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        
    def parallel_model_training(self, models: Dict[str, Any], X_train: pd.DataFrame,
                              y_train: pd.Series, X_test: pd.DataFrame,
                              y_test: pd.Series) -> Dict[str, Dict]:
        """Treinar múltiplos modelos em paralelo."""
        def train_single_model(model_name: str, model) -> Tuple[str, Dict]:
            try:
                logger.info(f"Treinando modelo: {model_name}")
                
                # Treinar modelo
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                # Calcular métricas
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                
                metrics = {
                    'mae': round(mean_absolute_error(y_test, predictions), 2),
                    'rmse': round(np.sqrt(mean_squared_error(y_test, predictions)), 2),
                    'r2': round(r2_score(y_test, predictions), 4),
                    'predictions': predictions
                }
                
                # Feature importance se disponível
                if hasattr(model, 'feature_importances_'):
                    feature_names = X_train.columns.tolist()
                    importance = dict(zip(feature_names, model.feature_importances_))
                    metrics['feature_importance'] = {
                        k: round(v, 4) for k, v in 
                        sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    }
                
                logger.info(f"Modelo {model_name} treinado - R²: {metrics['r2']}")
                return model_name, metrics
                
            except Exception as e:
                logger.error(f"Erro no treinamento de {model_name}: {e}")
                return model_name, {'error': str(e)}
        
        # Executar treinamento em paralelo
        futures = {
            self.thread_pool.submit(train_single_model, name, model): name
            for name, model in models.items()
        }
        
        results = {}
        for future in as_completed(futures):
            model_name, result = future.result()
            results[model_name] = result
        
        return results
    
    def parallel_category_analysis(self, df: pd.DataFrame, categories: List[str],
                                 analysis_func: Callable) -> Dict[str, Any]:
        """Executar análise por categoria em paralelo."""
        def analyze_category(category: str) -> Tuple[str, Dict]:
            try:
                category_data = df[df['Grupo_Produto'] == category]
                if len(category_data) < 10:
                    return category, {'error': 'Dados insuficientes'}
                
                result = analysis_func(category_data)
                return category, result
                
            except Exception as e:
                return category, {'error': str(e)}
        
        futures = {
            self.thread_pool.submit(analyze_category, cat): cat
            for cat in categories
        }
        
        results = {}
        for future in as_completed(futures):
            category, result = future.result()
            results[category] = result
        
        return results
    
    def __del__(self):
        """Cleanup do thread pool."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)


class StratifiedSampler:
    """Sistema de sampling estratificado para datasets grandes."""
    
    def __init__(self, max_sample_size: int = 50000):
        self.max_sample_size = max_sample_size
        
    def should_sample(self, df: pd.DataFrame) -> bool:
        """Determinar se sampling é necessário."""
        return len(df) > self.max_sample_size
    
    def create_stratified_sample(self, df: pd.DataFrame, 
                               analysis_type: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Criar amostra estratificada baseada no tipo de análise."""
        if not self.should_sample(df):
            return df, {'sampled': False, 'original_size': len(df)}
        
        logger.info(f"Criando amostra estratificada para {len(df)} registros")
        
        try:
            # Estratégias diferentes por tipo de análise
            if analysis_type in ['ml_insights', 'demand_forecasting']:
                sample_df = self._temporal_stratified_sample(df)
            elif analysis_type in ['customer_behavior', 'price_optimization']:
                sample_df = self._value_stratified_sample(df)
            elif analysis_type == 'inventory_optimization':
                sample_df = self._product_stratified_sample(df)
            else:
                sample_df = self._random_stratified_sample(df)
            
            # Validar representatividade
            representativeness = self._validate_sample_representativeness(df, sample_df)
            
            sample_info = {
                'sampled': True,
                'original_size': len(df),
                'sample_size': len(sample_df),
                'sample_ratio': len(sample_df) / len(df),
                'representativeness': representativeness,
                'strategy': analysis_type
            }
            
            logger.info(f"Amostra criada: {len(sample_df)} registros ({sample_info['sample_ratio']:.1%})")
            return sample_df, sample_info
            
        except Exception as e:
            logger.warning(f"Erro no sampling, usando dados completos: {e}")
            return df, {'sampled': False, 'error': str(e)}
    
    def _temporal_stratified_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sampling estratificado por período temporal."""
        if 'Data' not in df.columns:
            return self._random_stratified_sample(df)
        
        # Criar estratos por mês
        df_temp = df.copy()
        df_temp['Year_Month'] = pd.to_datetime(df_temp['Data']).dt.to_period('M')
        
        # Calcular tamanho da amostra por estrato
        strata_counts = df_temp['Year_Month'].value_counts()
        total_sample_size = min(self.max_sample_size, len(df))
        
        samples = []
        for period, count in strata_counts.items():
            stratum_data = df_temp[df_temp['Year_Month'] == period]
            stratum_sample_size = int((count / len(df)) * total_sample_size)
            stratum_sample_size = max(1, min(stratum_sample_size, len(stratum_data)))
            
            if len(stratum_data) > stratum_sample_size:
                stratum_sample = stratum_data.sample(stratum_sample_size)
            else:
                stratum_sample = stratum_data
            
            samples.append(stratum_sample.drop('Year_Month', axis=1))
        
        return pd.concat(samples, ignore_index=True)
    
    def _value_stratified_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sampling estratificado por valor de venda."""
        if 'Total_Liquido' not in df.columns:
            return self._random_stratified_sample(df)
        
        # Criar quartis de valor
        df_temp = df.copy()
        df_temp['Value_Quartile'] = pd.qcut(df_temp['Total_Liquido'], 
                                          q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'],
                                          duplicates='drop')
        
        # Amostra proporcional por quartil
        total_sample_size = min(self.max_sample_size, len(df))
        samples = []
        
        for quartile in df_temp['Value_Quartile'].unique():
            if pd.isna(quartile):
                continue
                
            quartile_data = df_temp[df_temp['Value_Quartile'] == quartile]
            quartile_sample_size = int((len(quartile_data) / len(df)) * total_sample_size)
            quartile_sample_size = max(1, min(quartile_sample_size, len(quartile_data)))
            
            if len(quartile_data) > quartile_sample_size:
                quartile_sample = quartile_data.sample(quartile_sample_size)
            else:
                quartile_sample = quartile_data
            
            samples.append(quartile_sample.drop('Value_Quartile', axis=1))
        
        return pd.concat(samples, ignore_index=True)
    
    def _product_stratified_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sampling estratificado por categoria de produto."""
        if 'Grupo_Produto' not in df.columns:
            return self._random_stratified_sample(df)
        
        total_sample_size = min(self.max_sample_size, len(df))
        samples = []
        
        for category in df['Grupo_Produto'].unique():
            if pd.isna(category):
                continue
                
            category_data = df[df['Grupo_Produto'] == category]
            category_sample_size = int((len(category_data) / len(df)) * total_sample_size)
            category_sample_size = max(1, min(category_sample_size, len(category_data)))
            
            if len(category_data) > category_sample_size:
                category_sample = category_data.sample(category_sample_size)
            else:
                category_sample = category_data
            
            samples.append(category_sample)
        
        return pd.concat(samples, ignore_index=True)
    
    def _random_stratified_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sampling aleatório simples."""
        sample_size = min(self.max_sample_size, len(df))
        return df.sample(sample_size)
    
    def _validate_sample_representativeness(self, original_df: pd.DataFrame,
                                          sample_df: pd.DataFrame) -> Dict[str, float]:
        """Validar representatividade da amostra."""
        representativeness = {}
        
        # Comparar distribuições de colunas numéricas
        numeric_cols = original_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in sample_df.columns:
                try:
                    # Teste Kolmogorov-Smirnov
                    ks_stat, p_value = stats.ks_2samp(
                        original_df[col].dropna(),
                        sample_df[col].dropna()
                    )
                    representativeness[f'{col}_ks_pvalue'] = round(p_value, 4)
                except:
                    representativeness[f'{col}_ks_pvalue'] = 0.0
        
        # Score geral de representatividade
        p_values = [v for k, v in representativeness.items() if 'pvalue' in k]
        if p_values:
            # Quanto maior o p-value, mais similar as distribuições
            avg_p_value = np.mean(p_values)
            representativeness['overall_score'] = round(avg_p_value, 4)
        else:
            representativeness['overall_score'] = 1.0
        
        return representativeness


class DataDriftDetector:
    """Sistema de detecção de data drift."""
    
    def __init__(self, reference_window_days: int = 90):
        self.reference_window_days = reference_window_days
        self.drift_threshold = 0.05  # p-value threshold
        
    def detect_drift(self, df: pd.DataFrame, 
                    reference_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Detectar drift nos dados."""
        try:
            if reference_df is None:
                reference_df = self._create_reference_data(df)
            
            if reference_df is None or len(reference_df) < 100:
                return {'drift_detected': False, 'reason': 'Dados de referência insuficientes'}
            
            # Detectar drift em diferentes dimensões
            drift_results = {
                'timestamp': datetime.now().isoformat(),
                'drift_detected': False,
                'drift_details': {},
                'recommendations': []
            }
            
            # 1. Drift temporal
            temporal_drift = self._detect_temporal_drift(df, reference_df)
            drift_results['drift_details']['temporal'] = temporal_drift
            
            # 2. Drift de distribuição
            distribution_drift = self._detect_distribution_drift(df, reference_df)
            drift_results['drift_details']['distribution'] = distribution_drift
            
            # 3. Drift de padrões de negócio
            business_drift = self._detect_business_pattern_drift(df, reference_df)
            drift_results['drift_details']['business_patterns'] = business_drift
            
            # Determinar se há drift significativo
            significant_drifts = []
            for drift_type, drift_info in drift_results['drift_details'].items():
                if drift_info.get('drift_detected', False):
                    significant_drifts.append(drift_type)
            
            if significant_drifts:
                drift_results['drift_detected'] = True
                drift_results['drift_types'] = significant_drifts
                drift_results['recommendations'] = self._generate_drift_recommendations(
                    significant_drifts, drift_results['drift_details']
                )
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Erro na detecção de drift: {e}")
            return {'drift_detected': False, 'error': str(e)}
    
    def _create_reference_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Criar dados de referência a partir do período mais antigo."""
        if 'Data' not in df.columns:
            return None
        
        try:
            df_sorted = df.sort_values('Data')
            reference_end_date = df_sorted['Data'].min() + timedelta(days=self.reference_window_days)
            
            reference_df = df_sorted[df_sorted['Data'] <= reference_end_date]
            
            if len(reference_df) < 100:
                return None
            
            return reference_df
            
        except Exception:
            return None
    
    def _detect_temporal_drift(self, current_df: pd.DataFrame,
                             reference_df: pd.DataFrame) -> Dict[str, Any]:
        """Detectar drift temporal nos padrões."""
        try:
            # Comparar padrões sazonais
            if 'Data' not in current_df.columns or 'Total_Liquido' not in current_df.columns:
                return {'drift_detected': False, 'reason': 'Colunas necessárias não encontradas'}
            
            # Agregar por dia da semana
            current_weekly = current_df.groupby(
                pd.to_datetime(current_df['Data']).dt.dayofweek
            )['Total_Liquido'].mean()
            
            reference_weekly = reference_df.groupby(
                pd.to_datetime(reference_df['Data']).dt.dayofweek
            )['Total_Liquido'].mean()
            
            # Teste estatístico
            if len(current_weekly) > 0 and len(reference_weekly) > 0:
                ks_stat, p_value = stats.ks_2samp(current_weekly, reference_weekly)
                
                return {
                    'drift_detected': p_value < self.drift_threshold,
                    'ks_statistic': round(ks_stat, 4),
                    'p_value': round(p_value, 4),
                    'pattern_change': 'Padrão semanal mudou significativamente' if p_value < self.drift_threshold else 'Padrão semanal estável'
                }
            
            return {'drift_detected': False, 'reason': 'Dados insuficientes para comparação'}
            
        except Exception as e:
            return {'drift_detected': False, 'error': str(e)}
    
    def _detect_distribution_drift(self, current_df: pd.DataFrame,
                                 reference_df: pd.DataFrame) -> Dict[str, Any]:
        """Detectar drift nas distribuições das variáveis."""
        try:
            numeric_cols = ['Total_Liquido', 'Quantidade']
            available_cols = [col for col in numeric_cols if col in current_df.columns and col in reference_df.columns]
            
            if not available_cols:
                return {'drift_detected': False, 'reason': 'Nenhuma coluna numérica para comparação'}
            
            drift_results = {}
            significant_drifts = 0
            
            for col in available_cols:
                current_values = current_df[col].dropna()
                reference_values = reference_df[col].dropna()
                
                if len(current_values) > 10 and len(reference_values) > 10:
                    # Teste Kolmogorov-Smirnov
                    ks_stat, p_value = stats.ks_2samp(current_values, reference_values)
                    
                    # Comparar estatísticas descritivas
                    current_mean = current_values.mean()
                    reference_mean = reference_values.mean()
                    mean_change_pct = ((current_mean - reference_mean) / reference_mean) * 100
                    
                    drift_results[col] = {
                        'ks_statistic': round(ks_stat, 4),
                        'p_value': round(p_value, 4),
                        'drift_detected': p_value < self.drift_threshold,
                        'mean_change_percent': round(mean_change_pct, 2),
                        'current_mean': round(current_mean, 2),
                        'reference_mean': round(reference_mean, 2)
                    }
                    
                    if p_value < self.drift_threshold:
                        significant_drifts += 1
            
            return {
                'drift_detected': significant_drifts > 0,
                'variables_with_drift': significant_drifts,
                'total_variables_tested': len(available_cols),
                'variable_details': drift_results
            }
            
        except Exception as e:
            return {'drift_detected': False, 'error': str(e)}
    
    def _detect_business_pattern_drift(self, current_df: pd.DataFrame,
                                     reference_df: pd.DataFrame) -> Dict[str, Any]:
        """Detectar drift em padrões de negócio."""
        try:
            business_metrics = {}
            
            # 1. AOV (Average Order Value)
            if 'Total_Liquido' in current_df.columns:
                current_aov = current_df['Total_Liquido'].mean()
                reference_aov = reference_df['Total_Liquido'].mean()
                aov_change = ((current_aov - reference_aov) / reference_aov) * 100
                
                business_metrics['aov'] = {
                    'current': round(current_aov, 2),
                    'reference': round(reference_aov, 2),
                    'change_percent': round(aov_change, 2),
                    'significant_change': abs(aov_change) > 10  # >10% change
                }
            
            # 2. Mix de produtos
            if 'Grupo_Produto' in current_df.columns:
                current_mix = current_df['Grupo_Produto'].value_counts(normalize=True)
                reference_mix = reference_df['Grupo_Produto'].value_counts(normalize=True)
                
                # Calcular divergência no mix
                common_products = set(current_mix.index) & set(reference_mix.index)
                if common_products:
                    mix_divergence = sum(
                        abs(current_mix.get(prod, 0) - reference_mix.get(prod, 0))
                        for prod in common_products
                    ) / 2  # Dividir por 2 para normalizar
                    
                    business_metrics['product_mix'] = {
                        'divergence': round(mix_divergence, 4),
                        'significant_change': mix_divergence > 0.1  # >10% divergence
                    }
            
            # Determinar se há drift nos padrões de negócio
            significant_changes = sum(
                1 for metric in business_metrics.values()
                if metric.get('significant_change', False)
            )
            
            return {
                'drift_detected': significant_changes > 0,
                'metrics_with_drift': significant_changes,
                'business_metrics': business_metrics
            }
            
        except Exception as e:
            return {'drift_detected': False, 'error': str(e)}
    
    def _generate_drift_recommendations(self, drift_types: List[str],
                                      drift_details: Dict[str, Any]) -> List[str]:
        """Gerar recomendações baseadas no drift detectado."""
        recommendations = []
        
        if 'temporal' in drift_types:
            recommendations.append("Revisar modelos sazonais - padrões temporais mudaram")
        
        if 'distribution' in drift_types:
            recommendations.append("Retreinar modelos ML - distribuições das variáveis mudaram")
            
            # Recomendações específicas por variável
            dist_details = drift_details.get('distribution', {}).get('variable_details', {})
            for var, details in dist_details.items():
                if details.get('drift_detected', False):
                    change_pct = details.get('mean_change_percent', 0)
                    if abs(change_pct) > 20:
                        recommendations.append(f"Investigar mudança significativa em {var} ({change_pct:+.1f}%)")
        
        if 'business_patterns' in drift_types:
            recommendations.append("Revisar estratégias de negócio - padrões comportamentais mudaram")
            
            business_details = drift_details.get('business_patterns', {}).get('business_metrics', {})
            if 'aov' in business_details and business_details['aov'].get('significant_change'):
                aov_change = business_details['aov'].get('change_percent', 0)
                recommendations.append(f"AOV mudou {aov_change:+.1f}% - revisar estratégia de pricing")
        
        if not recommendations:
            recommendations.append("Monitorar continuamente - drift detectado mas sem ações específicas necessárias")
        
        return recommendations


def with_performance_optimizations(cache_enabled: bool = True,
                                 parallel_enabled: bool = True,
                                 sampling_enabled: bool = True,
                                 drift_detection_enabled: bool = True):
    """Decorador para aplicar otimizações de performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, df: pd.DataFrame, analysis_type: str, *args, **kwargs) -> Dict[str, Any]:
            # Inicializar componentes de otimização
            if not hasattr(self, '_cache_manager') and cache_enabled:
                self._cache_manager = CacheManager()
            
            if not hasattr(self, '_parallel_processor') and parallel_enabled:
                self._parallel_processor = ParallelProcessor()
            
            if not hasattr(self, '_sampler') and sampling_enabled:
                self._sampler = StratifiedSampler()
            
            if not hasattr(self, '_drift_detector') and drift_detection_enabled:
                self._drift_detector = DataDriftDetector()
            
            # Preparar parâmetros para cache
            cache_params = {
                'analysis_type': analysis_type,
                'args': args,
                'kwargs': {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))}
            }
            
            # 1. Verificar cache
            if cache_enabled and hasattr(self, '_cache_manager'):
                cached_result = self._cache_manager.get_cached_result(df, analysis_type, cache_params)
                if cached_result is not None:
                    cached_result['_performance_info'] = {
                        'cache_hit': True,
                        'optimizations_applied': ['cache']
                    }
                    return cached_result
            
            # 2. Aplicar sampling se necessário
            original_df = df
            sample_info = {'sampled': False}
            
            if sampling_enabled and hasattr(self, '_sampler'):
                df, sample_info = self._sampler.create_stratified_sample(df, analysis_type)
            
            # 3. Detectar data drift
            drift_info = {}
            if drift_detection_enabled and hasattr(self, '_drift_detector'):
                try:
                    drift_info = self._drift_detector.detect_drift(df)
                except Exception as e:
                    logger.warning(f"Erro na detecção de drift: {e}")
                    drift_info = {'drift_detected': False, 'error': str(e)}
            
            # 4. Executar análise original
            try:
                result = func(self, df, analysis_type, *args, **kwargs)
                
                # Adicionar informações de performance
                optimizations_applied = []
                if cache_enabled:
                    optimizations_applied.append('cache_enabled')
                if parallel_enabled:
                    optimizations_applied.append('parallel_enabled')
                if sample_info['sampled']:
                    optimizations_applied.append('sampling')
                if drift_detection_enabled:
                    optimizations_applied.append('drift_detection')
                
                result['_performance_info'] = {
                    'cache_hit': False,
                    'optimizations_applied': optimizations_applied,
                    'sample_info': sample_info,
                    'drift_info': drift_info
                }
                
                # 5. Salvar no cache
                if cache_enabled and hasattr(self, '_cache_manager') and 'error' not in result:
                    self._cache_manager.save_result_to_cache(original_df, analysis_type, cache_params, result)
                
                return result
                
            except Exception as e:
                logger.error(f"Erro na análise otimizada: {e}")
                return {
                    'error': str(e),
                    '_performance_info': {
                        'cache_hit': False,
                        'optimizations_applied': ['error_occurred'],
                        'sample_info': sample_info
                    }
                }
        
        return wrapper
    return decorator 