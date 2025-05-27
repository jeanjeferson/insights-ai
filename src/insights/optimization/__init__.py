#!/usr/bin/env python
"""
🚀 INSIGHTS-AI OPTIMIZATION MODULE - ETAPA 4
Sistema avançado de otimização e performance para CrewAI Flows

Características:
- Otimização inteligente baseada em ML
- Sistema de auto-scaling
- Cache avançado integrado
- Monitoramento preditivo
- Análise de performance em tempo real
"""

# Imports dos módulos já implementados
from .optimization_controller import (
    OptimizationController, 
    OptimizationConfig, 
    OptimizationMode,
    OptimizationStatus
)
from .flow_optimizer import FlowOptimizer
from .cache_integration import CacheIntegration
from .ml_optimizer import MLOptimizer
from .performance_analytics import PerformanceAnalytics

# Imports dos novos módulos implementados
from .auto_scaler import AutoScaler
from .resource_manager import ResourceManager
from .predictive_engine import PredictiveEngine

__version__ = "1.0.0"
__author__ = "Insights-AI Team"

# Instância global do controlador de otimização
_global_optimization_controller = None

def get_global_optimization_controller() -> 'OptimizationController':
    """Obter controlador global de otimização"""
    global _global_optimization_controller
    if _global_optimization_controller is None:
        _global_optimization_controller = OptimizationController()
    return _global_optimization_controller

def init_optimization_system(**kwargs) -> 'OptimizationController':
    """Inicializar sistema de otimização com configurações customizadas"""
    global _global_optimization_controller
    _global_optimization_controller = OptimizationController(**kwargs)
    return _global_optimization_controller

__all__ = [
    'OptimizationController',
    'OptimizationConfig',
    'OptimizationMode', 
    'OptimizationStatus',
    'FlowOptimizer',
    'CacheIntegration',
    'MLOptimizer',
    'PerformanceAnalytics',
    'AutoScaler',
    'ResourceManager', 
    'PredictiveEngine',
    'get_global_optimization_controller',
    'init_optimization_system'
] 