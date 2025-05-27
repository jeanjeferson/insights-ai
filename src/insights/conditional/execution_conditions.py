#!/usr/bin/env python
"""
ETAPA 4 - CONDIÇÕES DE EXECUÇÃO
Sistema de condições inteligentes para execução otimizada
"""

import os
import hashlib
import logging
import psutil
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Configuração de logging
condition_logger = logging.getLogger('execution_conditions')
condition_logger.setLevel(logging.INFO)

class ConditionType(Enum):
    """Tipos de condições disponíveis"""
    DATA_CHANGE = "data_change"
    BUSINESS_RELEVANCE = "business_relevance"
    PERFORMANCE = "performance"
    TEMPORAL = "temporal"
    RESOURCE = "resource"
    DEPENDENCY = "dependency"

@dataclass
class ConditionResult:
    """Resultado da avaliação de uma condição"""
    should_execute: bool
    confidence: float  # 0.0 a 1.0
    reason: str
    metadata: Dict[str, Any]
    condition_type: ConditionType
    
    def __post_init__(self):
        # Validar confidence
        self.confidence = max(0.0, min(1.0, self.confidence))

class ExecutionCondition(ABC):
    """Condição abstrata de execução"""
    
    def __init__(self, name: str, condition_type: ConditionType, weight: float = 1.0):
        self.name = name
        self.condition_type = condition_type
        self.weight = weight
        self.created_at = datetime.now()
        self.evaluation_count = 0
        self.last_result: Optional[ConditionResult] = None
    
    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> ConditionResult:
        """Avaliar a condição e retornar resultado"""
        pass
    
    def _update_stats(self, result: ConditionResult):
        """Atualizar estatísticas da condição"""
        self.evaluation_count += 1
        self.last_result = result
        
        condition_logger.debug(
            f"📊 Condição '{self.name}' avaliada: "
            f"execute={result.should_execute}, confidence={result.confidence:.2f}, "
            f"reason={result.reason}"
        )

class DataChangeCondition(ExecutionCondition):
    """Condição baseada em mudanças nos dados"""
    
    def __init__(self, 
                 data_sources: List[str],
                 check_interval: int = 3600,  # segundos
                 min_change_threshold: float = 0.01,  # 1% de mudança mínima
                 hash_algorithm: str = "md5"):
        
        super().__init__("DataChange", ConditionType.DATA_CHANGE)
        self.data_sources = data_sources
        self.check_interval = check_interval
        self.min_change_threshold = min_change_threshold
        self.hash_algorithm = hash_algorithm
        
        # Cache de hashes dos dados
        self.data_hashes: Dict[str, str] = {}
        self.last_check: Dict[str, datetime] = {}
    
    def evaluate(self, context: Dict[str, Any]) -> ConditionResult:
        """Avaliar se os dados mudaram significativamente"""
        try:
            now = datetime.now()
            changes_detected = 0
            total_sources = len(self.data_sources)
            change_details = {}
            
            for source in self.data_sources:
                # Verificar se precisa checar este source
                last_check = self.last_check.get(source, datetime.min)
                if (now - last_check).total_seconds() < self.check_interval:
                    continue
                
                # Obter dados do source
                source_data = self._get_source_data(source, context)
                if source_data is None:
                    continue
                
                # Calcular hash atual
                current_hash = self._calculate_hash(source_data)
                previous_hash = self.data_hashes.get(source)
                
                # Verificar mudança
                if previous_hash is None:
                    # Primeira execução - considerar como mudança
                    changes_detected += 1
                    change_details[source] = "initial_load"
                elif current_hash != previous_hash:
                    # Dados mudaram
                    change_percentage = self._calculate_change_percentage(
                        source_data, context.get(f"{source}_previous_data")
                    )
                    
                    if change_percentage >= self.min_change_threshold:
                        changes_detected += 1
                        change_details[source] = f"changed_{change_percentage:.1%}"
                
                # Atualizar cache
                self.data_hashes[source] = current_hash
                self.last_check[source] = now
            
            # Calcular resultado
            if total_sources == 0:
                should_execute = True
                confidence = 0.5
                reason = "No data sources configured"
            else:
                change_ratio = changes_detected / total_sources
                should_execute = change_ratio > 0
                confidence = min(change_ratio * 2.0, 1.0)  # Máximo 1.0
                
                if changes_detected == 0:
                    reason = "No significant data changes detected"
                else:
                    reason = f"Data changes in {changes_detected}/{total_sources} sources"
            
            result = ConditionResult(
                should_execute=should_execute,
                confidence=confidence,
                reason=reason,
                metadata={
                    "changes_detected": changes_detected,
                    "total_sources": total_sources,
                    "change_details": change_details,
                    "sources_checked": list(self.last_check.keys())
                },
                condition_type=self.condition_type
            )
            
            self._update_stats(result)
            return result
            
        except Exception as e:
            condition_logger.error(f"❌ Erro na condição DataChange: {e}")
            return ConditionResult(
                should_execute=True,  # Executar por segurança
                confidence=0.1,
                reason=f"Error evaluating data changes: {e}",
                metadata={"error": str(e)},
                condition_type=self.condition_type
            )
    
    def _get_source_data(self, source: str, context: Dict[str, Any]) -> Optional[Any]:
        """Obter dados de uma fonte"""
        try:
            # Verificar se é um arquivo
            if os.path.isfile(source):
                with open(source, 'rb') as f:
                    return f.read()
            
            # Verificar se é uma chave no contexto
            if source in context:
                return context[source]
            
            # Verificar se é um diretório
            if os.path.isdir(source):
                file_info = []
                for file_path in Path(source).rglob("*"):
                    if file_path.is_file():
                        stat = file_path.stat()
                        file_info.append((str(file_path), stat.st_size, stat.st_mtime))
                return file_info
            
            return None
            
        except Exception as e:
            condition_logger.warning(f"⚠️ Erro ao obter dados de {source}: {e}")
            return None
    
    def _calculate_hash(self, data: Any) -> str:
        """Calcular hash dos dados"""
        try:
            if isinstance(data, bytes):
                content = data
            elif isinstance(data, str):
                content = data.encode('utf-8')
            else:
                content = str(data).encode('utf-8')
            
            if self.hash_algorithm == "md5":
                return hashlib.md5(content).hexdigest()
            elif self.hash_algorithm == "sha256":
                return hashlib.sha256(content).hexdigest()
            else:
                return hashlib.md5(content).hexdigest()
                
        except Exception as e:
            condition_logger.warning(f"⚠️ Erro ao calcular hash: {e}")
            return str(hash(str(data)))
    
    def _calculate_change_percentage(self, current_data: Any, previous_data: Any) -> float:
        """Calcular porcentagem de mudança"""
        try:
            if previous_data is None:
                return 1.0  # 100% de mudança se não há dados anteriores
            
            # Para dados binários/texto, usar diferença de tamanho
            if isinstance(current_data, (bytes, str)):
                current_size = len(current_data)
                previous_size = len(previous_data) if previous_data else 0
                
                if previous_size == 0:
                    return 1.0 if current_size > 0 else 0.0
                
                size_diff = abs(current_size - previous_size)
                return size_diff / previous_size
            
            # Para listas/diretórios, usar diferença de contagem
            if isinstance(current_data, list):
                current_count = len(current_data)
                previous_count = len(previous_data) if previous_data else 0
                
                if previous_count == 0:
                    return 1.0 if current_count > 0 else 0.0
                
                count_diff = abs(current_count - previous_count)
                return count_diff / previous_count
            
            # Para outros tipos, assumir mudança se diferente
            return 1.0 if current_data != previous_data else 0.0
            
        except Exception:
            return 0.5  # Assumir mudança moderada em caso de erro

class BusinessRelevanceCondition(ExecutionCondition):
    """Condição baseada na relevância de negócio"""
    
    def __init__(self, 
                 relevance_factors: Dict[str, float],
                 min_relevance_score: float = 0.5,
                 time_decay_factor: float = 0.1):
        
        super().__init__("BusinessRelevance", ConditionType.BUSINESS_RELEVANCE)
        self.relevance_factors = relevance_factors
        self.min_relevance_score = min_relevance_score
        self.time_decay_factor = time_decay_factor
    
    def evaluate(self, context: Dict[str, Any]) -> ConditionResult:
        """Avaliar relevância de negócio"""
        try:
            relevance_score = 0.0
            factor_scores = {}
            
            # Calcular score baseado nos fatores
            total_weight = sum(self.relevance_factors.values())
            if total_weight == 0:
                total_weight = 1.0
            
            for factor, weight in self.relevance_factors.items():
                factor_value = self._evaluate_factor(factor, context)
                weighted_score = (factor_value * weight) / total_weight
                relevance_score += weighted_score
                factor_scores[factor] = {
                    "value": factor_value,
                    "weight": weight,
                    "weighted_score": weighted_score
                }
            
            # Aplicar decay temporal se configurado
            last_execution = context.get('last_execution_time')
            if last_execution and isinstance(last_execution, datetime):
                time_since_last = (datetime.now() - last_execution).total_seconds() / 3600  # horas
                time_boost = min(time_since_last * self.time_decay_factor, 0.5)
                relevance_score += time_boost
                factor_scores["time_decay"] = {
                    "value": time_since_last,
                    "boost": time_boost,
                    "weighted_score": time_boost
                }
            
            # Normalizar score
            relevance_score = max(0.0, min(1.0, relevance_score))
            
            # Determinar se deve executar
            should_execute = relevance_score >= self.min_relevance_score
            confidence = relevance_score
            
            if should_execute:
                reason = f"High business relevance (score: {relevance_score:.2f})"
            else:
                reason = f"Low business relevance (score: {relevance_score:.2f}, min: {self.min_relevance_score:.2f})"
            
            result = ConditionResult(
                should_execute=should_execute,
                confidence=confidence,
                reason=reason,
                metadata={
                    "relevance_score": relevance_score,
                    "min_required": self.min_relevance_score,
                    "factor_scores": factor_scores
                },
                condition_type=self.condition_type
            )
            
            self._update_stats(result)
            return result
            
        except Exception as e:
            condition_logger.error(f"❌ Erro na condição BusinessRelevance: {e}")
            return ConditionResult(
                should_execute=True,
                confidence=0.5,
                reason=f"Error evaluating business relevance: {e}",
                metadata={"error": str(e)},
                condition_type=self.condition_type
            )
    
    def _evaluate_factor(self, factor: str, context: Dict[str, Any]) -> float:
        """Avaliar um fator específico de relevância"""
        try:
            # Fatores baseados em contexto
            if factor == "data_freshness":
                # Quanto mais recente, maior a relevância
                last_update = context.get('data_last_update')
                if isinstance(last_update, datetime):
                    hours_old = (datetime.now() - last_update).total_seconds() / 3600
                    return max(0.0, 1.0 - hours_old / 24.0)  # Decai em 24h
                return 0.5
            
            elif factor == "user_engagement":
                # Baseado em métricas de engajamento
                dashboard_views = context.get('dashboard_views_last_24h', 0)
                return min(dashboard_views / 100.0, 1.0)  # Normalizar por 100 views
            
            elif factor == "business_hours":
                # Maior relevância durante horário comercial
                now = datetime.now()
                if 8 <= now.hour <= 18 and now.weekday() < 5:  # 8h-18h, seg-sex
                    return 1.0
                elif 6 <= now.hour <= 22:  # Horário estendido
                    return 0.7
                else:
                    return 0.3
            
            elif factor == "data_volume":
                # Maior volume = maior relevância
                data_volume = context.get('data_volume', 0)
                if data_volume > 10000:
                    return 1.0
                elif data_volume > 1000:
                    return 0.8
                elif data_volume > 100:
                    return 0.5
                else:
                    return 0.2
            
            elif factor == "error_rate":
                # Menor taxa de erro = maior relevância
                error_rate = context.get('error_rate', 0.0)
                return max(0.0, 1.0 - error_rate)
            
            elif factor in context:
                # Valor direto do contexto
                value = context[factor]
                if isinstance(value, (int, float)):
                    return max(0.0, min(1.0, value))
                elif isinstance(value, bool):
                    return 1.0 if value else 0.0
                else:
                    return 0.5
            
            else:
                # Fator desconhecido
                return 0.5
                
        except Exception as e:
            condition_logger.warning(f"⚠️ Erro ao avaliar fator {factor}: {e}")
            return 0.5

class PerformanceCondition(ExecutionCondition):
    """Condição baseada em performance do sistema"""
    
    def __init__(self, 
                 max_cpu_usage: float = 80.0,
                 max_memory_usage: float = 85.0,
                 min_disk_space_gb: float = 5.0,
                 max_load_average: float = 2.0):
        
        super().__init__("Performance", ConditionType.PERFORMANCE)
        self.max_cpu_usage = max_cpu_usage
        self.max_memory_usage = max_memory_usage
        self.min_disk_space_gb = min_disk_space_gb
        self.max_load_average = max_load_average
    
    def evaluate(self, context: Dict[str, Any]) -> ConditionResult:
        """Avaliar condições de performance"""
        try:
            metrics = {}
            violations = []
            
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics["cpu_usage"] = cpu_percent
            if cpu_percent > self.max_cpu_usage:
                violations.append(f"CPU usage too high: {cpu_percent:.1f}% > {self.max_cpu_usage}%")
            
            # Memory Usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            metrics["memory_usage"] = memory_percent
            if memory_percent > self.max_memory_usage:
                violations.append(f"Memory usage too high: {memory_percent:.1f}% > {self.max_memory_usage}%")
            
            # Disk Space
            try:
                disk = psutil.disk_usage('/')
                disk_free_gb = disk.free / (1024**3)
                metrics["disk_free_gb"] = disk_free_gb
                if disk_free_gb < self.min_disk_space_gb:
                    violations.append(f"Disk space too low: {disk_free_gb:.1f}GB < {self.min_disk_space_gb}GB")
            except:
                # Fallback para Windows
                try:
                    disk = psutil.disk_usage('C:')
                    disk_free_gb = disk.free / (1024**3)
                    metrics["disk_free_gb"] = disk_free_gb
                    if disk_free_gb < self.min_disk_space_gb:
                        violations.append(f"Disk space too low: {disk_free_gb:.1f}GB < {self.min_disk_space_gb}GB")
                except:
                    metrics["disk_free_gb"] = None
            
            # Load Average (apenas Linux/Mac)
            try:
                load_avg = psutil.getloadavg()[0]  # 1-minute load average
                metrics["load_average"] = load_avg
                if load_avg > self.max_load_average:
                    violations.append(f"Load average too high: {load_avg:.2f} > {self.max_load_average}")
            except:
                metrics["load_average"] = None
            
            # Calcular resultado
            if violations:
                should_execute = False
                confidence = 1.0 - (len(violations) / 4.0)  # Reduzir confidence por violação
                reason = f"Performance constraints violated: {'; '.join(violations)}"
            else:
                should_execute = True
                # Calcular confidence baseada na proximidade dos limites
                cpu_factor = 1.0 - (cpu_percent / 100.0)
                memory_factor = 1.0 - (memory_percent / 100.0)
                confidence = (cpu_factor + memory_factor) / 2.0
                reason = "All performance constraints satisfied"
            
            result = ConditionResult(
                should_execute=should_execute,
                confidence=max(0.1, confidence),
                reason=reason,
                metadata={
                    "metrics": metrics,
                    "violations": violations,
                    "thresholds": {
                        "max_cpu_usage": self.max_cpu_usage,
                        "max_memory_usage": self.max_memory_usage,
                        "min_disk_space_gb": self.min_disk_space_gb,
                        "max_load_average": self.max_load_average
                    }
                },
                condition_type=self.condition_type
            )
            
            self._update_stats(result)
            return result
            
        except Exception as e:
            condition_logger.error(f"❌ Erro na condição Performance: {e}")
            return ConditionResult(
                should_execute=True,  # Executar por segurança
                confidence=0.3,
                reason=f"Error evaluating performance: {e}",
                metadata={"error": str(e)},
                condition_type=self.condition_type
            )

class TemporalCondition(ExecutionCondition):
    """Condição baseada em aspectos temporais"""
    
    def __init__(self, 
                 allowed_hours: List[int] = None,
                 allowed_days: List[int] = None,
                 min_interval_hours: float = 1.0,
                 max_interval_hours: float = 24.0):
        
        super().__init__("Temporal", ConditionType.TEMPORAL)
        self.allowed_hours = allowed_hours or list(range(24))  # Todas as horas por padrão
        self.allowed_days = allowed_days or list(range(7))     # Todos os dias por padrão
        self.min_interval_hours = min_interval_hours
        self.max_interval_hours = max_interval_hours
    
    def evaluate(self, context: Dict[str, Any]) -> ConditionResult:
        """Avaliar condições temporais"""
        try:
            now = datetime.now()
            violations = []
            
            # Verificar hora permitida
            if now.hour not in self.allowed_hours:
                violations.append(f"Current hour {now.hour} not in allowed hours {self.allowed_hours}")
            
            # Verificar dia permitido
            if now.weekday() not in self.allowed_days:
                violations.append(f"Current day {now.weekday()} not in allowed days {self.allowed_days}")
            
            # Verificar intervalo mínimo
            last_execution = context.get('last_execution_time')
            if isinstance(last_execution, datetime):
                hours_since_last = (now - last_execution).total_seconds() / 3600
                
                if hours_since_last < self.min_interval_hours:
                    violations.append(
                        f"Too soon since last execution: {hours_since_last:.1f}h < {self.min_interval_hours}h"
                    )
                elif hours_since_last > self.max_interval_hours:
                    # Forçar execução se passou muito tempo
                    force_reason = f"Forced execution: {hours_since_last:.1f}h > {self.max_interval_hours}h"
                    
                    return ConditionResult(
                        should_execute=True,
                        confidence=1.0,
                        reason=force_reason,
                        metadata={
                            "hours_since_last": hours_since_last,
                            "max_interval": self.max_interval_hours,
                            "forced": True
                        },
                        condition_type=self.condition_type
                    )
            
            # Calcular resultado
            if violations:
                should_execute = False
                confidence = 0.0
                reason = f"Temporal constraints violated: {'; '.join(violations)}"
            else:
                should_execute = True
                
                # Calcular confidence baseada na proximidade do próximo intervalo
                if isinstance(last_execution, datetime):
                    hours_since_last = (now - last_execution).total_seconds() / 3600
                    interval_progress = hours_since_last / self.max_interval_hours
                    confidence = min(interval_progress, 1.0)
                else:
                    confidence = 1.0  # Primeira execução
                
                reason = "All temporal constraints satisfied"
            
            result = ConditionResult(
                should_execute=should_execute,
                confidence=confidence,
                reason=reason,
                metadata={
                    "current_hour": now.hour,
                    "current_day": now.weekday(),
                    "allowed_hours": self.allowed_hours,
                    "allowed_days": self.allowed_days,
                    "violations": violations,
                    "last_execution": last_execution.isoformat() if isinstance(last_execution, datetime) else None
                },
                condition_type=self.condition_type
            )
            
            self._update_stats(result)
            return result
            
        except Exception as e:
            condition_logger.error(f"❌ Erro na condição Temporal: {e}")
            return ConditionResult(
                should_execute=True,
                confidence=0.5,
                reason=f"Error evaluating temporal conditions: {e}",
                metadata={"error": str(e)},
                condition_type=self.condition_type
            )

class CompositeCondition(ExecutionCondition):
    """Condição composta que combina múltiplas condições"""
    
    def __init__(self, 
                 conditions: List[ExecutionCondition],
                 operator: str = "AND",  # "AND", "OR", "WEIGHTED"
                 weights: List[float] = None,
                 min_confidence: float = 0.5):
        
        super().__init__("Composite", ConditionType.DEPENDENCY)
        self.conditions = conditions
        self.operator = operator.upper()
        self.weights = weights or [1.0] * len(conditions)
        self.min_confidence = min_confidence
        
        if len(self.weights) != len(self.conditions):
            raise ValueError("Number of weights must match number of conditions")
    
    def evaluate(self, context: Dict[str, Any]) -> ConditionResult:
        """Avaliar condição composta"""
        try:
            if not self.conditions:
                return ConditionResult(
                    should_execute=True,
                    confidence=1.0,
                    reason="No conditions to evaluate",
                    metadata={},
                    condition_type=self.condition_type
                )
            
            # Avaliar todas as condições
            results = []
            for condition in self.conditions:
                result = condition.evaluate(context)
                results.append(result)
            
            # Aplicar operador
            if self.operator == "AND":
                should_execute = all(r.should_execute for r in results)
                confidence = min(r.confidence for r in results) if results else 0.0
                reason = "AND: " + (
                    "All conditions satisfied" if should_execute 
                    else f"Some conditions failed: {[r.reason for r in results if not r.should_execute]}"
                )
            
            elif self.operator == "OR":
                should_execute = any(r.should_execute for r in results)
                confidence = max(r.confidence for r in results) if results else 0.0
                reason = "OR: " + (
                    f"At least one condition satisfied: {[r.reason for r in results if r.should_execute]}"
                    if should_execute else "No conditions satisfied"
                )
            
            elif self.operator == "WEIGHTED":
                total_weight = sum(self.weights)
                weighted_score = sum(
                    r.confidence * w for r, w in zip(results, self.weights)
                ) / total_weight if total_weight > 0 else 0.0
                
                should_execute = weighted_score >= self.min_confidence
                confidence = weighted_score
                reason = f"WEIGHTED: Score {weighted_score:.2f} (min: {self.min_confidence:.2f})"
            
            else:
                raise ValueError(f"Unknown operator: {self.operator}")
            
            # Compilar metadados
            metadata = {
                "operator": self.operator,
                "condition_results": [
                    {
                        "name": cond.name,
                        "type": cond.condition_type.value,
                        "should_execute": res.should_execute,
                        "confidence": res.confidence,
                        "reason": res.reason
                    }
                    for cond, res in zip(self.conditions, results)
                ],
                "weights": self.weights if self.operator == "WEIGHTED" else None
            }
            
            result = ConditionResult(
                should_execute=should_execute,
                confidence=confidence,
                reason=reason,
                metadata=metadata,
                condition_type=self.condition_type
            )
            
            self._update_stats(result)
            return result
            
        except Exception as e:
            condition_logger.error(f"❌ Erro na condição Composite: {e}")
            return ConditionResult(
                should_execute=True,
                confidence=0.3,
                reason=f"Error evaluating composite condition: {e}",
                metadata={"error": str(e)},
                condition_type=self.condition_type
            ) 