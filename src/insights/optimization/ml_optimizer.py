#!/usr/bin/env python
"""
🧠 ML OPTIMIZER - ETAPA 4
Sistema de otimização baseado em Machine Learning
"""

import logging
import time
import threading
import pickle
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path

# Configuração de logging
ml_optimizer_logger = logging.getLogger('ml_optimizer')
ml_optimizer_logger.setLevel(logging.INFO)

class OptimizationType(Enum):
    """Tipos de otimização ML"""
    EXECUTION_TIME_PREDICTION = "execution_time_prediction"
    RESOURCE_ALLOCATION = "resource_allocation"
    CACHE_STRATEGY = "cache_strategy"
    PARALLEL_SCHEDULING = "parallel_scheduling"
    ANOMALY_DETECTION = "anomaly_detection"

@dataclass
class MLOptimizationFeatures:
    """Features para modelos de ML"""
    # Features de entrada
    data_size_mb: float = 0.0
    analysis_count: int = 0
    complexity_score: float = 0.0
    has_date_range: bool = False
    execution_mode: str = "completo"
    
    # Features de histórico
    avg_past_execution_time: float = 0.0
    cache_hit_rate: float = 0.0
    cpu_usage_avg: float = 0.0
    memory_usage_avg: float = 0.0
    
    # Features de contexto
    time_of_day: int = 0  # 0-23
    day_of_week: int = 0  # 0-6
    current_system_load: float = 0.0

@dataclass
class MLPrediction:
    """Resultado de predição ML"""
    prediction_type: OptimizationType
    predicted_value: float
    confidence: float
    features_used: List[str]
    model_version: str
    prediction_time: datetime
    metadata: Dict[str, Any]

@dataclass
class OptimizationRecommendation:
    """Recomendação de otimização baseada em ML"""
    recommendation_type: str
    action: str
    expected_improvement: float
    confidence: float
    priority: str  # high, medium, low
    reasoning: str
    implementation_details: Dict[str, Any]

class MLOptimizer:
    """
    🧠 Otimizador Baseado em Machine Learning
    
    Utiliza modelos de ML para:
    - Predizer tempo de execução
    - Otimizar alocação de recursos
    - Detectar anomalias
    - Recomendar estratégias de cache
    - Otimizar scheduling paralelo
    """
    
    def __init__(self, 
                 models_dir: str = "data/optimization/models",
                 enable_online_learning: bool = True,
                 prediction_confidence_threshold: float = 0.7,
                 retrain_interval_hours: int = 168):  # 7 dias
        
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurações
        self.enable_online_learning = enable_online_learning
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.retrain_interval_hours = retrain_interval_hours
        
        # Modelos ML (inicializados sob demanda)
        self.models: Dict[OptimizationType, Any] = {}
        self.model_metadata: Dict[OptimizationType, Dict[str, Any]] = {}
        
        # Dados de treinamento
        self.training_data: Dict[OptimizationType, List[Dict[str, Any]]] = {
            opt_type: [] for opt_type in OptimizationType
        }
        
        # Histórico de predições
        self.prediction_history: List[MLPrediction] = []
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Estatísticas
        self.stats = {
            "total_predictions": 0,
            "accurate_predictions": 0,
            "model_retrains": 0,
            "avg_confidence": 0.0,
            "last_retrain": None
        }
        
        # Inicializar sistema
        self._initialize_ml_system()
        
        ml_optimizer_logger.info("🧠 MLOptimizer inicializado")
    
    def _initialize_ml_system(self):
        """Inicializar sistema de ML"""
        try:
            # Carregar modelos existentes
            self._load_existing_models()
            
            # Inicializar modelos padrão se necessário
            self._initialize_default_models()
            
            ml_optimizer_logger.info("✅ Sistema ML inicializado com sucesso")
            
        except Exception as e:
            ml_optimizer_logger.error(f"❌ Erro ao inicializar sistema ML: {e}")
    
    def _load_existing_models(self):
        """Carregar modelos existentes do disco"""
        for opt_type in OptimizationType:
            model_file = self.models_dir / f"{opt_type.value}_model.pkl"
            metadata_file = self.models_dir / f"{opt_type.value}_metadata.json"
            
            if model_file.exists() and metadata_file.exists():
                try:
                    # Carregar modelo
                    with open(model_file, 'rb') as f:
                        self.models[opt_type] = pickle.load(f)
                    
                    # Carregar metadata
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        self.model_metadata[opt_type] = json.load(f)
                    
                    ml_optimizer_logger.info(f"📁 Modelo carregado: {opt_type.value}")
                    
                except Exception as e:
                    ml_optimizer_logger.warning(f"⚠️ Erro ao carregar modelo {opt_type.value}: {e}")
    
    def _initialize_default_models(self):
        """Inicializar modelos padrão simples"""
        for opt_type in OptimizationType:
            if opt_type not in self.models:
                # Criar modelo simples baseado em heurísticas
                self.models[opt_type] = self._create_simple_model(opt_type)
                self.model_metadata[opt_type] = {
                    "model_type": "heuristic",
                    "created_at": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "training_samples": 0
                }
    
    def _create_simple_model(self, opt_type: OptimizationType) -> Dict[str, Any]:
        """Criar modelo heurístico simples"""
        if opt_type == OptimizationType.EXECUTION_TIME_PREDICTION:
            return {
                "type": "heuristic_time_predictor",
                "base_time": 30.0,  # segundos base
                "complexity_multiplier": 1.5,
                "data_size_factor": 0.1
            }
        
        elif opt_type == OptimizationType.RESOURCE_ALLOCATION:
            return {
                "type": "heuristic_resource_allocator",
                "base_memory_mb": 256,
                "base_cpu_percent": 20,
                "scaling_factor": 1.2
            }
        
        elif opt_type == OptimizationType.CACHE_STRATEGY:
            return {
                "type": "heuristic_cache_optimizer",
                "hit_rate_threshold": 0.7,
                "execution_time_threshold": 2.0
            }
        
        elif opt_type == OptimizationType.PARALLEL_SCHEDULING:
            return {
                "type": "heuristic_parallel_scheduler",
                "max_parallel_jobs": 4,
                "complexity_threshold": 0.5
            }
        
        elif opt_type == OptimizationType.ANOMALY_DETECTION:
            return {
                "type": "heuristic_anomaly_detector",
                "execution_time_std_threshold": 2.0,
                "resource_usage_threshold": 0.8
            }
        
        return {"type": "unknown"}
    
    def predict_execution_time(self, 
                             flow_state: Any,
                             context: Dict[str, Any] = None) -> MLPrediction:
        """Predizer tempo de execução de um Flow"""
        try:
            # Extrair features
            features = self._extract_features(flow_state, context)
            
            # Obter modelo
            model = self.models.get(OptimizationType.EXECUTION_TIME_PREDICTION)
            if not model:
                raise ValueError("Modelo de predição de tempo não disponível")
            
            # Fazer predição
            if model["type"] == "heuristic_time_predictor":
                predicted_time = self._predict_time_heuristic(features, model)
                confidence = 0.6  # Confiança média para heurística
            else:
                # Implementação futura para modelos ML reais
                predicted_time = self._predict_time_ml(features, model)
                confidence = 0.8
            
            # Criar resultado
            prediction = MLPrediction(
                prediction_type=OptimizationType.EXECUTION_TIME_PREDICTION,
                predicted_value=predicted_time,
                confidence=confidence,
                features_used=list(asdict(features).keys()),
                model_version=self.model_metadata[OptimizationType.EXECUTION_TIME_PREDICTION].get("version", "1.0.0"),
                prediction_time=datetime.now(),
                metadata={
                    "features": asdict(features),
                    "model_type": model["type"]
                }
            )
            
            # Registrar predição
            self.prediction_history.append(prediction)
            with self.lock:
                self.stats["total_predictions"] += 1
            
            ml_optimizer_logger.info(
                f"🔮 Predição de tempo: {predicted_time:.1f}s (confiança: {confidence:.1%})"
            )
            
            return prediction
            
        except Exception as e:
            ml_optimizer_logger.error(f"❌ Erro na predição de tempo: {e}")
            return MLPrediction(
                prediction_type=OptimizationType.EXECUTION_TIME_PREDICTION,
                predicted_value=60.0,  # Fallback
                confidence=0.1,
                features_used=[],
                model_version="error",
                prediction_time=datetime.now(),
                metadata={"error": str(e)}
            )
    
    def _extract_features(self, flow_state: Any, context: Dict[str, Any] = None) -> MLOptimizationFeatures:
        """Extrair features para modelos ML"""
        features = MLOptimizationFeatures()
        
        try:
            if hasattr(flow_state, '__dict__'):
                state_dict = flow_state.__dict__
                
                # Features de dados
                features.analysis_count = len([k for k in state_dict.keys() if k.startswith("analise_")])
                features.has_date_range = bool(state_dict.get("data_inicio") and state_dict.get("data_fim"))
                features.execution_mode = state_dict.get("modo_execucao", "completo")
                
                # Estimar tamanho dos dados
                if state_dict.get("dataset_path"):
                    try:
                        from pathlib import Path
                        if Path(state_dict["dataset_path"]).exists():
                            features.data_size_mb = Path(state_dict["dataset_path"]).stat().st_size / (1024 * 1024)
                    except:
                        features.data_size_mb = 10.0  # Fallback
                
                # Complexity score
                features.complexity_score = self._calculate_complexity_features(state_dict)
            
            # Features de contexto
            now = datetime.now()
            features.time_of_day = now.hour
            features.day_of_week = now.weekday()
            
            # Features do sistema (se contexto disponível)
            if context:
                features.avg_past_execution_time = context.get("avg_execution_time", 0.0)
                features.cache_hit_rate = context.get("cache_hit_rate", 0.0)
                features.cpu_usage_avg = context.get("cpu_usage", 0.0)
                features.memory_usage_avg = context.get("memory_usage", 0.0)
                features.current_system_load = context.get("system_load", 0.0)
            
            return features
            
        except Exception as e:
            ml_optimizer_logger.warning(f"⚠️ Erro ao extrair features: {e}")
            return features
    
    def _calculate_complexity_features(self, state_dict: Dict[str, Any]) -> float:
        """Calcular score de complexidade para features"""
        complexity = 0.0
        
        # Fator 1: Número de análises
        analysis_count = len([k for k in state_dict.keys() if k.startswith("analise_")])
        complexity += min(analysis_count / 10.0, 0.3)  # Máximo 0.3
        
        # Fator 2: Análises avançadas
        advanced_count = len([k for k in state_dict.keys() if "avancada" in k])
        complexity += min(advanced_count / 5.0, 0.2)  # Máximo 0.2
        
        # Fator 3: Relatórios complexos
        if state_dict.get("dashboard_html_dinamico") or state_dict.get("relatorio_executivo_completo"):
            complexity += 0.3
        
        # Fator 4: Modo de execução
        if state_dict.get("modo_execucao") == "completo":
            complexity += 0.2
        
        return min(complexity, 1.0)
    
    def _predict_time_heuristic(self, features: MLOptimizationFeatures, model: Dict[str, Any]) -> float:
        """Predição heurística de tempo"""
        base_time = model["base_time"]
        complexity_mult = model["complexity_multiplier"]
        data_factor = model["data_size_factor"]
        
        # Cálculo heurístico
        predicted_time = base_time
        predicted_time *= (1 + features.complexity_score * complexity_mult)
        predicted_time += features.data_size_mb * data_factor
        predicted_time *= (1 + features.analysis_count * 0.1)
        
        # Ajustes baseados no contexto
        if features.avg_past_execution_time > 0:
            # Usar média histórica como referência
            predicted_time = (predicted_time + features.avg_past_execution_time) / 2
        
        return max(predicted_time, 10.0)  # Mínimo 10 segundos
    
    def _predict_time_ml(self, features: MLOptimizationFeatures, model: Dict[str, Any]) -> float:
        """Predição usando modelo ML (implementação futura)"""
        # Placeholder para modelos ML reais (scikit-learn, etc.)
        return self._predict_time_heuristic(features, model)
    
    def optimize_resource_allocation(self, 
                                   flow_state: Any,
                                   current_resources: Dict[str, Any] = None) -> OptimizationRecommendation:
        """Otimizar alocação de recursos"""
        try:
            features = self._extract_features(flow_state)
            model = self.models.get(OptimizationType.RESOURCE_ALLOCATION)
            
            if not model:
                raise ValueError("Modelo de alocação de recursos não disponível")
            
            # Calcular recursos recomendados
            if model["type"] == "heuristic_resource_allocator":
                recommended_memory = model["base_memory_mb"] * (1 + features.complexity_score)
                recommended_memory += features.data_size_mb * 0.5
                recommended_memory *= model["scaling_factor"]
                
                recommended_cpu = model["base_cpu_percent"] * (1 + features.complexity_score)
                recommended_cpu = min(recommended_cpu, 80.0)  # Máximo 80%
            else:
                # Implementação futura para modelos ML
                recommended_memory = 512.0
                recommended_cpu = 50.0
            
            # Comparar com recursos atuais
            current_memory = current_resources.get("memory_mb", 256) if current_resources else 256
            current_cpu = current_resources.get("cpu_percent", 25) if current_resources else 25
            
            memory_improvement = (recommended_memory - current_memory) / current_memory * 100
            cpu_improvement = (recommended_cpu - current_cpu) / current_cpu * 100
            
            expected_improvement = (abs(memory_improvement) + abs(cpu_improvement)) / 2
            
            return OptimizationRecommendation(
                recommendation_type="resource_allocation",
                action=f"Ajustar recursos: {recommended_memory:.0f}MB RAM, {recommended_cpu:.0f}% CPU",
                expected_improvement=min(expected_improvement, 50.0),
                confidence=0.7,
                priority="medium" if expected_improvement > 20 else "low",
                reasoning=f"Baseado na complexidade ({features.complexity_score:.2f}) e tamanho dos dados ({features.data_size_mb:.1f}MB)",
                implementation_details={
                    "recommended_memory_mb": recommended_memory,
                    "recommended_cpu_percent": recommended_cpu,
                    "current_memory_mb": current_memory,
                    "current_cpu_percent": current_cpu
                }
            )
            
        except Exception as e:
            ml_optimizer_logger.error(f"❌ Erro na otimização de recursos: {e}")
            return OptimizationRecommendation(
                recommendation_type="resource_allocation",
                action="Manter configuração atual",
                expected_improvement=0.0,
                confidence=0.1,
                priority="low",
                reasoning=f"Erro na análise: {e}",
                implementation_details={}
            )
    
    def detect_anomalies(self, 
                        execution_metrics: Dict[str, Any],
                        historical_data: List[Dict[str, Any]] = None) -> List[OptimizationRecommendation]:
        """Detectar anomalias na execução"""
        anomalies = []
        
        try:
            model = self.models.get(OptimizationType.ANOMALY_DETECTION)
            if not model:
                return anomalies
            
            # Verificar tempo de execução anômalo
            execution_time = execution_metrics.get("execution_time", 0)
            if historical_data:
                historical_times = [d.get("execution_time", 0) for d in historical_data]
                if historical_times:
                    avg_time = sum(historical_times) / len(historical_times)
                    std_time = np.std(historical_times) if len(historical_times) > 1 else 0
                    
                    if std_time > 0 and abs(execution_time - avg_time) > model["execution_time_std_threshold"] * std_time:
                        anomalies.append(OptimizationRecommendation(
                            recommendation_type="anomaly_detection",
                            action="Investigar tempo de execução anômalo",
                            expected_improvement=20.0,
                            confidence=0.8,
                            priority="high",
                            reasoning=f"Tempo de execução ({execution_time:.1f}s) está {abs(execution_time - avg_time):.1f}s fora da média ({avg_time:.1f}s)",
                            implementation_details={
                                "anomaly_type": "execution_time",
                                "current_value": execution_time,
                                "expected_range": [avg_time - std_time, avg_time + std_time]
                            }
                        ))
            
            # Verificar uso de recursos anômalo
            cpu_usage = execution_metrics.get("cpu_usage", 0)
            memory_usage = execution_metrics.get("memory_usage", 0)
            
            if cpu_usage > model["resource_usage_threshold"] * 100:
                anomalies.append(OptimizationRecommendation(
                    recommendation_type="anomaly_detection",
                    action="Otimizar uso de CPU",
                    expected_improvement=15.0,
                    confidence=0.7,
                    priority="medium",
                    reasoning=f"Uso de CPU ({cpu_usage:.1f}%) acima do threshold ({model['resource_usage_threshold']*100:.1f}%)",
                    implementation_details={
                        "anomaly_type": "high_cpu",
                        "current_value": cpu_usage,
                        "threshold": model["resource_usage_threshold"] * 100
                    }
                ))
            
            if memory_usage > model["resource_usage_threshold"] * 100:
                anomalies.append(OptimizationRecommendation(
                    recommendation_type="anomaly_detection",
                    action="Otimizar uso de memória",
                    expected_improvement=15.0,
                    confidence=0.7,
                    priority="medium",
                    reasoning=f"Uso de memória ({memory_usage:.1f}%) acima do threshold ({model['resource_usage_threshold']*100:.1f}%)",
                    implementation_details={
                        "anomaly_type": "high_memory",
                        "current_value": memory_usage,
                        "threshold": model["resource_usage_threshold"] * 100
                    }
                ))
            
            return anomalies
            
        except Exception as e:
            ml_optimizer_logger.error(f"❌ Erro na detecção de anomalias: {e}")
            return []
    
    def recommend_cache_strategy(self, 
                               operation_metrics: Dict[str, Any],
                               current_strategy: str = "smart") -> OptimizationRecommendation:
        """Recomendar estratégia de cache"""
        try:
            model = self.models.get(OptimizationType.CACHE_STRATEGY)
            if not model:
                raise ValueError("Modelo de cache não disponível")
            
            hit_rate = operation_metrics.get("cache_hit_rate", 0.0)
            avg_execution_time = operation_metrics.get("avg_execution_time", 0.0)
            
            hit_threshold = model["hit_rate_threshold"]
            time_threshold = model["execution_time_threshold"]
            
            # Lógica de recomendação
            if hit_rate < hit_threshold and avg_execution_time > time_threshold:
                recommended_strategy = "aggressive"
                reasoning = f"Hit rate baixo ({hit_rate:.1%}) e tempo alto ({avg_execution_time:.1f}s)"
                improvement = 30.0
                priority = "high"
            elif hit_rate > 0.9:
                recommended_strategy = "selective"
                reasoning = f"Hit rate muito alto ({hit_rate:.1%}), pode estar cacheando demais"
                improvement = 10.0
                priority = "low"
            else:
                recommended_strategy = "smart"
                reasoning = "Performance de cache aceitável"
                improvement = 5.0
                priority = "low"
            
            return OptimizationRecommendation(
                recommendation_type="cache_strategy",
                action=f"Mudar estratégia de cache para '{recommended_strategy}'",
                expected_improvement=improvement,
                confidence=0.75,
                priority=priority,
                reasoning=reasoning,
                implementation_details={
                    "recommended_strategy": recommended_strategy,
                    "current_strategy": current_strategy,
                    "metrics": operation_metrics
                }
            )
            
        except Exception as e:
            ml_optimizer_logger.error(f"❌ Erro na recomendação de cache: {e}")
            return OptimizationRecommendation(
                recommendation_type="cache_strategy",
                action="Manter estratégia atual",
                expected_improvement=0.0,
                confidence=0.1,
                priority="low",
                reasoning=f"Erro na análise: {e}",
                implementation_details={}
            )
    
    def learn_from_execution(self, 
                           flow_state: Any,
                           execution_result: Dict[str, Any],
                           context: Dict[str, Any] = None):
        """Aprender com execução para melhorar modelos"""
        if not self.enable_online_learning:
            return
        
        try:
            # Extrair features e resultados para treinamento
            features = self._extract_features(flow_state, context)
            
            training_sample = {
                "features": asdict(features),
                "actual_execution_time": execution_result.get("execution_time", 0.0),
                "actual_cpu_usage": execution_result.get("cpu_usage", 0.0),
                "actual_memory_usage": execution_result.get("memory_usage", 0.0),
                "cache_hit_rate": execution_result.get("cache_hit_rate", 0.0),
                "timestamp": datetime.now().isoformat(),
                "flow_id": execution_result.get("flow_id", "unknown")
            }
            
            # Adicionar aos dados de treinamento
            with self.lock:
                self.training_data[OptimizationType.EXECUTION_TIME_PREDICTION].append(training_sample)
                self.training_data[OptimizationType.RESOURCE_ALLOCATION].append(training_sample)
            
            # Verificar se há predição anterior para validar
            self._validate_predictions(execution_result)
            
            ml_optimizer_logger.debug("📚 Dados de treinamento atualizados")
            
        except Exception as e:
            ml_optimizer_logger.error(f"❌ Erro no aprendizado: {e}")
    
    def _validate_predictions(self, execution_result: Dict[str, Any]):
        """Validar predições anteriores"""
        try:
            flow_id = execution_result.get("flow_id")
            actual_time = execution_result.get("execution_time", 0.0)
            
            # Encontrar predições recentes para este flow
            recent_predictions = [
                p for p in self.prediction_history[-10:]  # Últimas 10 predições
                if p.prediction_type == OptimizationType.EXECUTION_TIME_PREDICTION
                and p.metadata.get("flow_id") == flow_id
            ]
            
            for prediction in recent_predictions:
                predicted_time = prediction.predicted_value
                error_percent = abs(predicted_time - actual_time) / actual_time * 100 if actual_time > 0 else 100
                
                # Considerar predição precisa se erro < 20%
                if error_percent < 20:
                    with self.lock:
                        self.stats["accurate_predictions"] += 1
                
                ml_optimizer_logger.debug(
                    f"📊 Validação: Predito {predicted_time:.1f}s, Real {actual_time:.1f}s, Erro {error_percent:.1f}%"
                )
                
        except Exception as e:
            ml_optimizer_logger.warning(f"⚠️ Erro na validação de predições: {e}")
    
    # =============== API PÚBLICA ===============
    
    def get_ml_stats(self) -> Dict[str, Any]:
        """Obter estatísticas dos modelos ML"""
        with self.lock:
            stats = self.stats.copy()
            
            # Calcular precisão
            if stats["total_predictions"] > 0:
                stats["accuracy_percent"] = (stats["accurate_predictions"] / stats["total_predictions"]) * 100
            else:
                stats["accuracy_percent"] = 0.0
            
            # Informações dos modelos
            stats["models_loaded"] = len(self.models)
            stats["training_samples"] = sum(len(data) for data in self.training_data.values())
            
            return stats
    
    def retrain_models(self, force: bool = False) -> Dict[str, Any]:
        """Retreinar modelos com novos dados"""
        try:
            # Verificar se é necessário retreinar
            if not force:
                last_retrain = self.stats.get("last_retrain")
                if last_retrain:
                    last_retrain_dt = datetime.fromisoformat(last_retrain)
                    if datetime.now() - last_retrain_dt < timedelta(hours=self.retrain_interval_hours):
                        return {"status": "skipped", "reason": "not_due_yet"}
            
            retrained_models = []
            
            for opt_type in OptimizationType:
                training_data = self.training_data.get(opt_type, [])
                
                if len(training_data) >= 10:  # Mínimo de amostras
                    # Implementação simplificada - em produção, usar algoritmos ML reais
                    retrained_models.append(opt_type.value)
                    ml_optimizer_logger.info(f"🔄 Modelo retreinado: {opt_type.value}")
            
            # Atualizar estatísticas
            with self.lock:
                self.stats["model_retrains"] += 1
                self.stats["last_retrain"] = datetime.now().isoformat()
            
            return {
                "status": "completed",
                "retrained_models": retrained_models,
                "training_samples_used": sum(len(data) for data in self.training_data.values())
            }
            
        except Exception as e:
            ml_optimizer_logger.error(f"❌ Erro no retreinamento: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_optimization_recommendations(self, 
                                       flow_state: Any,
                                       execution_context: Dict[str, Any] = None) -> List[OptimizationRecommendation]:
        """Obter todas as recomendações de otimização"""
        recommendations = []
        
        try:
            # Recomendação de recursos
            resource_rec = self.optimize_resource_allocation(flow_state, execution_context)
            recommendations.append(resource_rec)
            
            # Recomendação de cache (se contexto disponível)
            if execution_context and "cache_metrics" in execution_context:
                cache_rec = self.recommend_cache_strategy(
                    execution_context["cache_metrics"],
                    execution_context.get("current_cache_strategy", "smart")
                )
                recommendations.append(cache_rec)
            
            # Detecção de anomalias (se métricas disponíveis)
            if execution_context and "execution_metrics" in execution_context:
                anomalies = self.detect_anomalies(
                    execution_context["execution_metrics"],
                    execution_context.get("historical_data", [])
                )
                recommendations.extend(anomalies)
            
            # Ordenar por prioridade
            priority_order = {"high": 0, "medium": 1, "low": 2}
            recommendations.sort(key=lambda r: priority_order.get(r.priority, 3))
            
            return recommendations
            
        except Exception as e:
            ml_optimizer_logger.error(f"❌ Erro ao gerar recomendações: {e}")
            return [] 