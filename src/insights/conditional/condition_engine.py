#!/usr/bin/env python
"""
ETAPA 4 - ENGINE DE CONDIÇÕES
Motor de execução para avaliação de condições inteligentes
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from .execution_conditions import ExecutionCondition, ConditionResult, ConditionType

# Configuração de logging
engine_logger = logging.getLogger('condition_engine')
engine_logger.setLevel(logging.INFO)

@dataclass
class ConditionEngineResult:
    """Resultado da avaliação do engine"""
    should_execute: bool
    overall_confidence: float
    primary_reason: str
    condition_results: List[ConditionResult]
    execution_recommendation: str
    metadata: Dict[str, Any]

class ConditionEngine:
    """Motor de avaliação de condições"""
    
    def __init__(self, 
                 min_confidence_threshold: float = 0.5,
                 require_all_conditions: bool = False):
        self.min_confidence_threshold = min_confidence_threshold
        self.require_all_conditions = require_all_conditions
        self.conditions: List[ExecutionCondition] = []
        self.evaluation_history: List[ConditionEngineResult] = []
        
        engine_logger.info("🔧 ConditionEngine inicializado")
    
    def add_condition(self, condition: ExecutionCondition):
        """Adicionar condição ao engine"""
        self.conditions.append(condition)
        engine_logger.info(f"➕ Condição adicionada: {condition.name} ({condition.condition_type.value})")
    
    def remove_condition(self, condition_name: str) -> bool:
        """Remover condição pelo nome"""
        original_count = len(self.conditions)
        self.conditions = [c for c in self.conditions if c.name != condition_name]
        removed = len(self.conditions) < original_count
        
        if removed:
            engine_logger.info(f"➖ Condição removida: {condition_name}")
        
        return removed
    
    def evaluate_all(self, context: Dict[str, Any]) -> ConditionEngineResult:
        """Avaliar todas as condições"""
        try:
            if not self.conditions:
                # Nenhuma condição = execução livre
                result = ConditionEngineResult(
                    should_execute=True,
                    overall_confidence=1.0,
                    primary_reason="No conditions configured - execution allowed",
                    condition_results=[],
                    execution_recommendation="Execute immediately",
                    metadata={"no_conditions": True}
                )
                self.evaluation_history.append(result)
                return result
            
            # Avaliar cada condição
            condition_results = []
            for condition in self.conditions:
                try:
                    result = condition.evaluate(context)
                    condition_results.append(result)
                except Exception as e:
                    # Condição com erro - criar resultado padrão
                    error_result = ConditionResult(
                        should_execute=False,
                        confidence=0.0,
                        reason=f"Error in condition {condition.name}: {e}",
                        metadata={"error": str(e)},
                        condition_type=condition.condition_type
                    )
                    condition_results.append(error_result)
                    engine_logger.error(f"❌ Erro na condição {condition.name}: {e}")
            
            # Analisar resultados
            analysis = self._analyze_results(condition_results)
            
            # Construir resultado final
            final_result = ConditionEngineResult(
                should_execute=analysis["should_execute"],
                overall_confidence=analysis["overall_confidence"],
                primary_reason=analysis["primary_reason"],
                condition_results=condition_results,
                execution_recommendation=analysis["recommendation"],
                metadata=analysis["metadata"]
            )
            
            # Salvar no histórico
            self.evaluation_history.append(final_result)
            
            engine_logger.info(
                f"🎯 Avaliação concluída: execute={final_result.should_execute}, "
                f"confidence={final_result.overall_confidence:.2f}"
            )
            
            return final_result
            
        except Exception as e:
            engine_logger.error(f"❌ Erro na avaliação do engine: {e}")
            error_result = ConditionEngineResult(
                should_execute=False,
                overall_confidence=0.0,
                primary_reason=f"Engine evaluation error: {e}",
                condition_results=[],
                execution_recommendation="Do not execute due to engine error",
                metadata={"engine_error": str(e)}
            )
            self.evaluation_history.append(error_result)
            return error_result
    
    def get_condition_summary(self) -> Dict[str, Any]:
        """Obter resumo das condições"""
        try:
            summary = {
                "total_conditions": len(self.conditions),
                "conditions_by_type": {},
                "condition_details": [],
                "engine_config": {
                    "min_confidence_threshold": self.min_confidence_threshold,
                    "require_all_conditions": self.require_all_conditions
                }
            }
            
            # Agrupar por tipo
            for condition in self.conditions:
                condition_type = condition.condition_type.value
                if condition_type not in summary["conditions_by_type"]:
                    summary["conditions_by_type"][condition_type] = 0
                summary["conditions_by_type"][condition_type] += 1
                
                # Detalhes da condição
                summary["condition_details"].append({
                    "name": condition.name,
                    "type": condition_type,
                    "weight": condition.weight,
                    "evaluations": condition.evaluation_count,
                    "last_result": {
                        "should_execute": condition.last_result.should_execute if condition.last_result else None,
                        "confidence": condition.last_result.confidence if condition.last_result else None,
                        "reason": condition.last_result.reason if condition.last_result else None
                    } if condition.last_result else None
                })
            
            return summary
            
        except Exception as e:
            engine_logger.error(f"❌ Erro ao gerar resumo: {e}")
            return {"error": str(e)}
    
    def get_evaluation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obter histórico de avaliações"""
        try:
            # Pegar últimas avaliações
            recent_evaluations = self.evaluation_history[-limit:] if limit > 0 else self.evaluation_history
            
            history = []
            for evaluation in recent_evaluations:
                history.append({
                    "timestamp": datetime.now().isoformat(),  # Idealmente seria salvo no momento da avaliação
                    "should_execute": evaluation.should_execute,
                    "overall_confidence": evaluation.overall_confidence,
                    "primary_reason": evaluation.primary_reason,
                    "recommendation": evaluation.execution_recommendation,
                    "conditions_evaluated": len(evaluation.condition_results),
                    "conditions_passed": sum(1 for r in evaluation.condition_results if r.should_execute),
                    "conditions_failed": sum(1 for r in evaluation.condition_results if not r.should_execute)
                })
            
            return history
            
        except Exception as e:
            engine_logger.error(f"❌ Erro ao obter histórico: {e}")
            return []
    
    def _analyze_results(self, condition_results: List[ConditionResult]) -> Dict[str, Any]:
        """Analisar resultados das condições"""
        try:
            if not condition_results:
                return {
                    "should_execute": True,
                    "overall_confidence": 1.0,
                    "primary_reason": "No conditions to evaluate",
                    "recommendation": "Execute immediately",
                    "metadata": {}
                }
            
            # Contar sucessos e falhas
            passed_conditions = [r for r in condition_results if r.should_execute]
            failed_conditions = [r for r in condition_results if not r.should_execute]
            
            passed_count = len(passed_conditions)
            failed_count = len(failed_conditions)
            total_count = len(condition_results)
            
            # Calcular confidence geral
            if total_count > 0:
                # Média ponderada das confidences
                total_confidence = sum(r.confidence for r in condition_results)
                overall_confidence = total_confidence / total_count
            else:
                overall_confidence = 0.0
            
            # Determinar se deve executar
            if self.require_all_conditions:
                # Modo rigoroso - todas as condições devem passar
                should_execute = failed_count == 0
                decision_logic = "require_all"
            else:
                # Modo flexível - baseado em threshold de confidence
                should_execute = overall_confidence >= self.min_confidence_threshold
                decision_logic = "confidence_threshold"
            
            # Determinar razão primária
            if should_execute:
                if passed_count == total_count:
                    primary_reason = f"All {total_count} conditions passed successfully"
                else:
                    primary_reason = f"Overall confidence {overall_confidence:.2f} meets threshold {self.min_confidence_threshold:.2f}"
            else:
                if failed_count > 0:
                    failed_reasons = [r.reason for r in failed_conditions[:2]]  # Top 2 falhas
                    primary_reason = f"Conditions failed: {'; '.join(failed_reasons)}"
                else:
                    primary_reason = f"Low confidence {overall_confidence:.2f} below threshold {self.min_confidence_threshold:.2f}"
            
            # Gerar recomendação
            if should_execute:
                if overall_confidence > 0.8:
                    recommendation = "Execute immediately with high confidence"
                elif overall_confidence > 0.6:
                    recommendation = "Execute with moderate confidence"
                else:
                    recommendation = "Execute with caution - low confidence"
            else:
                if failed_count > total_count / 2:
                    recommendation = "Do not execute - multiple conditions failed"
                else:
                    recommendation = "Postpone execution - conditions not met"
            
            # Metadados
            metadata = {
                "total_conditions": total_count,
                "passed_conditions": passed_count,
                "failed_conditions": failed_count,
                "decision_logic": decision_logic,
                "confidence_by_type": {},
                "top_failures": [
                    {"condition": r.condition_type.value, "reason": r.reason}
                    for r in failed_conditions[:3]
                ]
            }
            
            # Confidence por tipo de condição
            by_type = {}
            for result in condition_results:
                ctype = result.condition_type.value
                if ctype not in by_type:
                    by_type[ctype] = []
                by_type[ctype].append(result.confidence)
            
            for ctype, confidences in by_type.items():
                metadata["confidence_by_type"][ctype] = {
                    "avg": sum(confidences) / len(confidences),
                    "count": len(confidences)
                }
            
            return {
                "should_execute": should_execute,
                "overall_confidence": overall_confidence,
                "primary_reason": primary_reason,
                "recommendation": recommendation,
                "metadata": metadata
            }
            
        except Exception as e:
            engine_logger.error(f"❌ Erro na análise dos resultados: {e}")
            return {
                "should_execute": False,
                "overall_confidence": 0.0,
                "primary_reason": f"Analysis error: {e}",
                "recommendation": "Do not execute due to analysis error",
                "metadata": {"analysis_error": str(e)}
            }

# ========== FUNÇÕES UTILITÁRIAS ==========

def create_default_condition_engine() -> ConditionEngine:
    """Criar engine com condições padrão"""
    from .execution_conditions import PerformanceCondition, TemporalCondition, BusinessRelevanceCondition
    
    engine = ConditionEngine(min_confidence_threshold=0.6)
    
    # Adicionar condições padrão
    engine.add_condition(PerformanceCondition())
    engine.add_condition(TemporalCondition())
    engine.add_condition(BusinessRelevanceCondition(
        relevance_factors={
            "business_hours": 0.3,
            "data_freshness": 0.4,
            "user_engagement": 0.3
        }
    ))
    
    return engine

def create_performance_focused_engine() -> ConditionEngine:
    """Criar engine focado em performance"""
    from .execution_conditions import PerformanceCondition
    
    engine = ConditionEngine(
        min_confidence_threshold=0.8,
        require_all_conditions=True
    )
    
    # Condições rigorosas de performance
    engine.add_condition(PerformanceCondition(
        max_cpu_usage=70.0,
        max_memory_usage=75.0,
        min_disk_space_gb=10.0
    ))
    
    return engine

def create_flexible_engine() -> ConditionEngine:
    """Criar engine flexível para desenvolvimento/teste"""
    from .execution_conditions import TemporalCondition
    
    engine = ConditionEngine(
        min_confidence_threshold=0.3,
        require_all_conditions=False
    )
    
    # Apenas condições básicas
    engine.add_condition(TemporalCondition(
        min_interval_hours=0.1  # 6 minutos
    ))
    
    return engine 