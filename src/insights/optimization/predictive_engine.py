#!/usr/bin/env python
"""
🚀 INSIGHTS-AI PREDICTIVE ENGINE
Sistema de predição baseado em ML para otimização proativa

Características:
- Predição de carga de trabalho
- Detecção de anomalias
- Previsão de necessidades de recursos
- Análise de padrões temporais
- ML adaptativo
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
predictive_logger = logging.getLogger(__name__)
predictive_logger.setLevel(logging.INFO)

class PredictionType(Enum):
    """Tipos de predições disponíveis"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    EXECUTION_TIME = "execution_time"
    CACHE_HIT_RATE = "cache_hit_rate"
    REQUEST_VOLUME = "request_volume"
    RESOURCE_DEMAND = "resource_demand"

class PredictionTimeframe(Enum):
    """Janelas de tempo para predições"""
    NEXT_5_MINUTES = 5
    NEXT_15_MINUTES = 15
    NEXT_30_MINUTES = 30
    NEXT_HOUR = 60
    NEXT_4_HOURS = 240
    NEXT_DAY = 1440

@dataclass
class PredictionInput:
    """Dados de entrada para predições"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    execution_time_ms: float
    cache_hit_rate: float
    request_count: int
    active_flows: int
    system_load: float
    hour_of_day: int
    day_of_week: int
    is_weekend: bool

@dataclass
class PredictionResult:
    """Resultado de uma predição"""
    prediction_type: PredictionType
    timeframe: PredictionTimeframe
    predicted_value: float
    confidence_score: float
    prediction_bounds: Tuple[float, float]
    factors_contribution: Dict[str, float]
    timestamp: datetime
    model_version: str

@dataclass
class AnomalyDetection:
    """Resultado de detecção de anomalia"""
    timestamp: datetime
    is_anomaly: bool
    anomaly_score: float
    affected_metrics: List[str]
    severity: str
    description: str

class PredictiveEngine:
    """Engine de predição baseado em ML"""
    
    def __init__(self, 
                 model_dir: str = "data/optimization/models",
                 enable_auto_retrain: bool = True,
                 retrain_threshold_hours: int = 24):
        
        self.model_dir = model_dir
        self.enable_auto_retrain = enable_auto_retrain
        self.retrain_threshold_hours = retrain_threshold_hours
        
        # Estado do sistema
        self.is_initialized = False
        self.last_training = None
        self.model_version = "1.0.0"
        
        # Modelos ML
        self.prediction_models: Dict[PredictionType, RandomForestRegressor] = {}
        self.anomaly_detector = None
        self.feature_scalers: Dict[str, StandardScaler] = {}
        
        # Dados e histórico
        self.training_data: List[PredictionInput] = []
        self.prediction_history: List[PredictionResult] = []
        self.anomaly_history: List[AnomalyDetection] = []
        
        # Configurações dos modelos
        self.model_configs = {
            PredictionType.CPU_USAGE: {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            PredictionType.MEMORY_USAGE: {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            PredictionType.EXECUTION_TIME: {
                "n_estimators": 150,
                "max_depth": 12,
                "random_state": 42
            }
        }
        
        # Configurar sistema
        self._setup_predictive_engine()
        
        predictive_logger.info("🎯 PredictiveEngine inicializado")
    
    def _setup_predictive_engine(self):
        """Configurar engine de predição"""
        try:
            # Criar diretório para modelos
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Carregar modelos existentes
            self._load_existing_models()
            
            # Inicializar modelos se necessário
            if not self.prediction_models:
                self._initialize_models()
            
            self.is_initialized = True
            predictive_logger.info("✅ PredictiveEngine configurado")
            
        except Exception as e:
            predictive_logger.error(f"❌ Erro ao configurar PredictiveEngine: {e}")
    
    def add_training_data(self, data: PredictionInput):
        """Adicionar dados de treinamento"""
        self.training_data.append(data)
        
        # Limitar tamanho dos dados de treinamento
        if len(self.training_data) > 10000:
            self.training_data = self.training_data[-5000:]
        
        # Auto-retreinar se necessário ou se tivermos dados suficientes pela primeira vez
        if (len(self.training_data) >= 10 and 
            (self.enable_auto_retrain or not self.prediction_models or 
             all(not hasattr(model, 'tree_') for model in self.prediction_models.values()))):
            self._retrain_models()
        elif (self.enable_auto_retrain and 
              len(self.training_data) > 100 and
              self._should_retrain()):
            self._retrain_models()
    
    def predict(self, 
                prediction_type: PredictionType, 
                timeframe: PredictionTimeframe,
                current_data: Optional[PredictionInput] = None) -> PredictionResult:
        """Fazer predição para tipo e timeframe específicos"""
        
        if not self.is_initialized:
            raise RuntimeError("PredictiveEngine não inicializado")
        
        try:
            # Usar dados atuais ou dados mais recentes
            if current_data is None:
                if not self.training_data:
                    raise ValueError("Nenhum dado disponível para predição")
                current_data = self.training_data[-1]
            
            # Preparar features
            features = self._prepare_features(current_data, timeframe)
            
            # Obter modelo apropriado
            model = self.prediction_models.get(prediction_type)
            if model is None:
                raise ValueError(f"Modelo não disponível para {prediction_type}")
            
            # Fazer predição
            prediction = model.predict([features])[0]
            
            # Calcular confiança e bounds
            confidence = self._calculate_prediction_confidence(model, features, prediction_type)
            bounds = self._calculate_prediction_bounds(prediction, confidence)
            
            # Analisar contribuição dos fatores
            factors_contribution = self._analyze_feature_importance(model, features, prediction_type)
            
            result = PredictionResult(
                prediction_type=prediction_type,
                timeframe=timeframe,
                predicted_value=prediction,
                confidence_score=confidence,
                prediction_bounds=bounds,
                factors_contribution=factors_contribution,
                timestamp=datetime.now(),
                model_version=self.model_version
            )
            
            # Salvar no histórico
            self.prediction_history.append(result)
            self._cleanup_old_predictions()
            
            predictive_logger.info(
                f"🔮 Predição: {prediction_type.value} = {prediction:.2f} "
                f"(confiança: {confidence:.2f})"
            )
            
            return result
            
        except Exception as e:
            predictive_logger.error(f"❌ Erro na predição: {e}")
            raise
    
    def predict_multiple(self, 
                        prediction_types: List[PredictionType],
                        timeframe: PredictionTimeframe,
                        current_data: Optional[PredictionInput] = None) -> List[PredictionResult]:
        """Fazer múltiplas predições"""
        results = []
        
        for prediction_type in prediction_types:
            try:
                result = self.predict(prediction_type, timeframe, current_data)
                results.append(result)
            except Exception as e:
                predictive_logger.error(f"❌ Erro na predição {prediction_type}: {e}")
        
        return results
    
    def detect_anomalies(self, current_data: PredictionInput) -> AnomalyDetection:
        """Detectar anomalias nos dados atuais"""
        
        if self.anomaly_detector is None:
            self._train_anomaly_detector()
        
        try:
            # Preparar features para detecção de anomalias
            features = self._prepare_features_for_anomaly_detection(current_data)
            
            # Detectar anomalia
            anomaly_score = self.anomaly_detector.decision_function([features])[0]
            is_anomaly = self.anomaly_detector.predict([features])[0] == -1
            
            # Analisar quais métricas são anômalas
            affected_metrics = self._identify_anomalous_metrics(current_data, features)
            
            # Determinar severidade
            severity = self._determine_anomaly_severity(anomaly_score, affected_metrics)
            
            # Gerar descrição
            description = self._generate_anomaly_description(affected_metrics, severity)
            
            result = AnomalyDetection(
                timestamp=current_data.timestamp,
                is_anomaly=is_anomaly,
                anomaly_score=anomaly_score,
                affected_metrics=affected_metrics,
                severity=severity,
                description=description
            )
            
            # Salvar no histórico se for anomalia
            if is_anomaly:
                self.anomaly_history.append(result)
                self._cleanup_old_anomalies()
                
                predictive_logger.warning(
                    f"🚨 Anomalia detectada: {description} (score: {anomaly_score:.3f})"
                )
            
            return result
            
        except Exception as e:
            predictive_logger.error(f"❌ Erro na detecção de anomalias: {e}")
            return AnomalyDetection(
                timestamp=current_data.timestamp,
                is_anomaly=False,
                anomaly_score=0.0,
                affected_metrics=[],
                severity="normal",
                description="Erro na detecção"
            )
    
    def get_prediction_accuracy(self, hours_back: int = 24) -> Dict[str, float]:
        """Calcular precisão das predições recentes"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        recent_predictions = [
            p for p in self.prediction_history
            if p.timestamp >= cutoff_time
        ]
        
        if not recent_predictions:
            return {}
        
        accuracy_by_type = {}
        
        for pred_type in PredictionType:
            type_predictions = [p for p in recent_predictions if p.prediction_type == pred_type]
            
            if type_predictions:
                # Calcular MAE médio como proxy para precisão
                total_confidence = sum(p.confidence_score for p in type_predictions)
                avg_confidence = total_confidence / len(type_predictions)
                accuracy_by_type[pred_type.value] = avg_confidence
        
        return accuracy_by_type
    
    def get_system_forecast(self, hours_ahead: int = 4) -> Dict[str, Any]:
        """Obter previsão geral do sistema"""
        
        if not self.training_data:
            return {"status": "no_data"}
        
        current_data = self.training_data[-1]
        
        # Prever principais métricas
        forecasts = {}
        
        for pred_type in [PredictionType.CPU_USAGE, PredictionType.MEMORY_USAGE, PredictionType.EXECUTION_TIME]:
            try:
                # Determinar timeframe baseado em hours_ahead
                if hours_ahead <= 1:
                    timeframe = PredictionTimeframe.NEXT_HOUR
                else:
                    timeframe = PredictionTimeframe.NEXT_4_HOURS
                
                result = self.predict(pred_type, timeframe, current_data)
                
                forecasts[pred_type.value] = {
                    "predicted_value": result.predicted_value,
                    "confidence": result.confidence_score,
                    "bounds": result.prediction_bounds,
                    "trend": self._calculate_trend(pred_type)
                }
                
            except Exception as e:
                predictive_logger.error(f"❌ Erro na previsão {pred_type}: {e}")
        
        # Detectar anomalias atuais
        anomaly_result = self.detect_anomalies(current_data)
        
        # Calcular score geral de saúde do sistema
        health_score = self._calculate_system_health_score(forecasts, anomaly_result)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "forecast_horizon_hours": hours_ahead,
            "forecasts": forecasts,
            "anomaly_status": {
                "is_anomaly": anomaly_result.is_anomaly,
                "severity": anomaly_result.severity,
                "description": anomaly_result.description
            },
            "system_health_score": health_score,
            "recommendations": self._generate_system_recommendations(forecasts, anomaly_result)
        }
    
    # Métodos privados
    
    def _initialize_models(self):
        """Inicializar modelos ML"""
        for pred_type in [PredictionType.CPU_USAGE, PredictionType.MEMORY_USAGE, PredictionType.EXECUTION_TIME]:
            config = self.model_configs.get(pred_type, {})
            self.prediction_models[pred_type] = RandomForestRegressor(**config)
        
        # Inicializar detector de anomalias
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        predictive_logger.info("🤖 Modelos ML inicializados")
    
    def _prepare_features(self, data: PredictionInput, timeframe: PredictionTimeframe) -> List[float]:
        """Preparar features para predição"""
        features = [
            data.cpu_percent,
            data.memory_percent,
            data.execution_time_ms,
            data.cache_hit_rate,
            data.request_count,
            data.active_flows,
            data.system_load,
            data.hour_of_day,
            data.day_of_week,
            float(data.is_weekend),
            timeframe.value  # Adicionar timeframe como feature
        ]
        
        # Adicionar features temporais
        if len(self.training_data) > 1:
            # Taxa de mudança recente
            recent_data = self.training_data[-5:]
            cpu_trend = self._calculate_simple_trend([d.cpu_percent for d in recent_data])
            memory_trend = self._calculate_simple_trend([d.memory_percent for d in recent_data])
            
            features.extend([cpu_trend, memory_trend])
        else:
            features.extend([0.0, 0.0])
        
        return features
    
    def _prepare_features_for_anomaly_detection(self, data: PredictionInput) -> List[float]:
        """Preparar features para detecção de anomalias"""
        return [
            data.cpu_percent,
            data.memory_percent,
            data.execution_time_ms,
            data.cache_hit_rate,
            data.request_count,
            data.active_flows,
            data.system_load
        ]
    
    def _calculate_simple_trend(self, values: List[float]) -> float:
        """Calcular tendência simples"""
        if len(values) < 2:
            return 0.0
        
        # Diferença entre último e primeiro valor
        return (values[-1] - values[0]) / len(values)
    
    def _should_retrain(self) -> bool:
        """Verificar se deve retreinar modelos"""
        if self.last_training is None:
            return True
        
        hours_since_training = (datetime.now() - self.last_training).total_seconds() / 3600
        return hours_since_training >= self.retrain_threshold_hours
    
    def _retrain_models(self):
        """Retreinar modelos com novos dados"""
        try:
            if len(self.training_data) < 50:
                return
            
            predictive_logger.info("🔄 Retreinando modelos ML...")
            
            # Preparar dados de treinamento
            X, y_dict = self._prepare_training_data()
            
            # Treinar cada modelo
            for pred_type in self.prediction_models.keys():
                if pred_type.value in y_dict:
                    y = y_dict[pred_type.value]
                    
                    # Split train/test
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Treinar modelo
                    self.prediction_models[pred_type].fit(X_train, y_train)
                    
                    # Avaliar performance
                    y_pred = self.prediction_models[pred_type].predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    predictive_logger.info(
                        f"📊 Modelo {pred_type.value}: MAE={mae:.3f}, R²={r2:.3f}"
                    )
            
            # Treinar detector de anomalias
            self._train_anomaly_detector()
            
            # Salvar modelos
            self._save_models()
            
            self.last_training = datetime.now()
            predictive_logger.info("✅ Retreinamento concluído")
            
        except Exception as e:
            predictive_logger.error(f"❌ Erro no retreinamento: {e}")
    
    def _prepare_training_data(self) -> Tuple[List[List[float]], Dict[str, List[float]]]:
        """Preparar dados para treinamento"""
        X = []
        y_dict = {
            PredictionType.CPU_USAGE.value: [],
            PredictionType.MEMORY_USAGE.value: [],
            PredictionType.EXECUTION_TIME.value: []
        }
        
        for i, data in enumerate(self.training_data[:-1]):  # Excluir último item
            # Features
            features = self._prepare_features(data, PredictionTimeframe.NEXT_15_MINUTES)
            X.append(features)
            
            # Targets (valores futuros)
            if i + 1 < len(self.training_data):
                next_data = self.training_data[i + 1]
                y_dict[PredictionType.CPU_USAGE.value].append(next_data.cpu_percent)
                y_dict[PredictionType.MEMORY_USAGE.value].append(next_data.memory_percent)
                y_dict[PredictionType.EXECUTION_TIME.value].append(next_data.execution_time_ms)
        
        return X, y_dict
    
    def _train_anomaly_detector(self):
        """Treinar detector de anomalias"""
        if len(self.training_data) < 20:
            return
        
        try:
            # Preparar features para anomalia
            X_anomaly = []
            for data in self.training_data:
                features = self._prepare_features_for_anomaly_detection(data)
                X_anomaly.append(features)
            
            # Treinar detector
            self.anomaly_detector.fit(X_anomaly)
            
            predictive_logger.info("🤖 Detector de anomalias treinado")
            
        except Exception as e:
            predictive_logger.error(f"❌ Erro no treinamento do detector: {e}")
    
    def _calculate_prediction_confidence(self, model, features: List[float], pred_type: PredictionType) -> float:
        """Calcular confiança da predição"""
        try:
            # Para Random Forest, usar variância das predições das árvores
            if hasattr(model, 'estimators_'):
                predictions = [tree.predict([features])[0] for tree in model.estimators_]
                variance = np.var(predictions)
                # Converter variância em score de confiança (0-1)
                confidence = max(0.0, min(1.0, 1.0 - (variance / 100.0)))
                return confidence
            else:
                return 0.7  # Confiança padrão
        except:
            return 0.5
    
    def _calculate_prediction_bounds(self, prediction: float, confidence: float) -> Tuple[float, float]:
        """Calcular bounds da predição"""
        margin = prediction * (1.0 - confidence) * 0.5
        return (prediction - margin, prediction + margin)
    
    def _analyze_feature_importance(self, model, features: List[float], pred_type: PredictionType) -> Dict[str, float]:
        """Analisar importância das features"""
        try:
            if hasattr(model, 'feature_importances_'):
                feature_names = [
                    "cpu_percent", "memory_percent", "execution_time_ms",
                    "cache_hit_rate", "request_count", "active_flows",
                    "system_load", "hour_of_day", "day_of_week",
                    "is_weekend", "timeframe", "cpu_trend", "memory_trend"
                ]
                
                importance_dict = {}
                importances = model.feature_importances_
                
                for i, name in enumerate(feature_names[:len(importances)]):
                    importance_dict[name] = float(importances[i])
                
                return importance_dict
            
        except:
            pass
        
        return {}
    
    def _identify_anomalous_metrics(self, data: PredictionInput, features: List[float]) -> List[str]:
        """Identificar quais métricas são anômalas"""
        anomalous_metrics = []
        
        # Análise simples baseada em thresholds
        if data.cpu_percent > 90:
            anomalous_metrics.append("cpu_percent")
        if data.memory_percent > 85:
            anomalous_metrics.append("memory_percent")
        if data.execution_time_ms > 10000:  # 10 segundos
            anomalous_metrics.append("execution_time_ms")
        if data.cache_hit_rate < 30:
            anomalous_metrics.append("cache_hit_rate")
        
        return anomalous_metrics
    
    def _determine_anomaly_severity(self, anomaly_score: float, affected_metrics: List[str]) -> str:
        """Determinar severidade da anomalia"""
        if anomaly_score < -0.5 or len(affected_metrics) >= 3:
            return "critical"
        elif anomaly_score < -0.2 or len(affected_metrics) >= 2:
            return "high"
        elif anomaly_score < 0 or len(affected_metrics) >= 1:
            return "medium"
        else:
            return "low"
    
    def _generate_anomaly_description(self, affected_metrics: List[str], severity: str) -> str:
        """Gerar descrição da anomalia"""
        if not affected_metrics:
            return "Sistema funcionando normalmente"
        
        metrics_str = ", ".join(affected_metrics)
        return f"Anomalia {severity} detectada em: {metrics_str}"
    
    def _calculate_trend(self, pred_type: PredictionType) -> str:
        """Calcular tendência para um tipo de predição"""
        if len(self.training_data) < 5:
            return "stable"
        
        recent_values = []
        for data in self.training_data[-5:]:
            if pred_type == PredictionType.CPU_USAGE:
                recent_values.append(data.cpu_percent)
            elif pred_type == PredictionType.MEMORY_USAGE:
                recent_values.append(data.memory_percent)
            elif pred_type == PredictionType.EXECUTION_TIME:
                recent_values.append(data.execution_time_ms)
        
        if len(recent_values) >= 2:
            trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
            if trend > 1.0:
                return "increasing"
            elif trend < -1.0:
                return "decreasing"
        
        return "stable"
    
    def _calculate_system_health_score(self, forecasts: Dict[str, Any], anomaly: AnomalyDetection) -> float:
        """Calcular score de saúde do sistema"""
        health_score = 100.0
        
        # Penalizar baseado nas predições
        for pred_type, forecast in forecasts.items():
            predicted_value = forecast.get("predicted_value", 0)
            confidence = forecast.get("confidence", 0)
            
            if pred_type == "cpu_usage" and predicted_value > 80:
                health_score -= (predicted_value - 80) * 0.5
            elif pred_type == "memory_usage" and predicted_value > 85:
                health_score -= (predicted_value - 85) * 0.6
            
            # Penalizar baixa confiança
            health_score -= (1.0 - confidence) * 10
        
        # Penalizar anomalias
        if anomaly.is_anomaly:
            severity_penalties = {
                "critical": 30,
                "high": 20,
                "medium": 10,
                "low": 5
            }
            health_score -= severity_penalties.get(anomaly.severity, 0)
        
        return max(0.0, min(100.0, health_score))
    
    def _generate_system_recommendations(self, forecasts: Dict[str, Any], anomaly: AnomalyDetection) -> List[str]:
        """Gerar recomendações do sistema"""
        recommendations = []
        
        # Recomendações baseadas nas predições
        for pred_type, forecast in forecasts.items():
            predicted_value = forecast.get("predicted_value", 0)
            trend = forecast.get("trend", "stable")
            
            if pred_type == "cpu_usage":
                if predicted_value > 80:
                    recommendations.append("Considere otimização de CPU ou scaling up")
                if trend == "increasing":
                    recommendations.append("Tendência de aumento no uso de CPU - monitore de perto")
            
            elif pred_type == "memory_usage":
                if predicted_value > 85:
                    recommendations.append("Memória alta prevista - considere limpeza ou scaling")
                if trend == "increasing":
                    recommendations.append("Uso de memória crescente - verifique vazamentos")
        
        # Recomendações baseadas em anomalias
        if anomaly.is_anomaly:
            if anomaly.severity in ["critical", "high"]:
                recommendations.append("Anomalia crítica detectada - investigação imediata necessária")
            elif "cpu_percent" in anomaly.affected_metrics:
                recommendations.append("CPU anômala - verifique processos intensivos")
            elif "memory_percent" in anomaly.affected_metrics:
                recommendations.append("Memória anômala - verifique vazamentos ou carga incomum")
        
        # Recomendação padrão se não houver outras
        if not recommendations:
            recommendations.append("Sistema operando dentro dos parâmetros normais")
        
        return recommendations
    
    def _save_models(self):
        """Salvar modelos treinados"""
        try:
            # Salvar modelos de predição
            for pred_type, model in self.prediction_models.items():
                model_path = os.path.join(self.model_dir, f"{pred_type.value}_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Salvar detector de anomalias
            if self.anomaly_detector:
                anomaly_path = os.path.join(self.model_dir, "anomaly_detector.pkl")
                with open(anomaly_path, 'wb') as f:
                    pickle.dump(self.anomaly_detector, f)
            
            # Salvar metadados
            metadata = {
                "model_version": self.model_version,
                "last_training": self.last_training.isoformat() if self.last_training else None,
                "training_data_size": len(self.training_data)
            }
            
            metadata_path = os.path.join(self.model_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            predictive_logger.info("💾 Modelos salvos com sucesso")
            
        except Exception as e:
            predictive_logger.error(f"❌ Erro ao salvar modelos: {e}")
    
    def _load_existing_models(self):
        """Carregar modelos existentes"""
        try:
            # Carregar metadados
            metadata_path = os.path.join(self.model_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.model_version = metadata.get("model_version", "1.0.0")
                    last_training_str = metadata.get("last_training")
                    if last_training_str:
                        self.last_training = datetime.fromisoformat(last_training_str)
            
            # Carregar modelos de predição
            for pred_type in [PredictionType.CPU_USAGE, PredictionType.MEMORY_USAGE, PredictionType.EXECUTION_TIME]:
                model_path = os.path.join(self.model_dir, f"{pred_type.value}_model.pkl")
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.prediction_models[pred_type] = pickle.load(f)
            
            # Carregar detector de anomalias
            anomaly_path = os.path.join(self.model_dir, "anomaly_detector.pkl")
            if os.path.exists(anomaly_path):
                with open(anomaly_path, 'rb') as f:
                    self.anomaly_detector = pickle.load(f)
            
            if self.prediction_models:
                predictive_logger.info(f"📦 {len(self.prediction_models)} modelos carregados")
            
        except Exception as e:
            predictive_logger.warning(f"⚠️ Erro ao carregar modelos existentes: {e}")
    
    def _cleanup_old_predictions(self):
        """Limpar predições antigas"""
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-500:]
    
    def _cleanup_old_anomalies(self):
        """Limpar anomalias antigas"""
        if len(self.anomaly_history) > 500:
            self.anomaly_history = self.anomaly_history[-250:] 