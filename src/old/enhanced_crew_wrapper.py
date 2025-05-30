"""
🔧 ENHANCED CREW WRAPPER - FASE 1
=================================

Wrapper não invasivo para o crew existente que adiciona:
- Enhanced logging
- Context-aware error handling  
- Progress tracking
- Retry logic melhorado

Mantém 100% de compatibilidade com o código existente.
"""

import time
import functools
from typing import Any, Dict, Optional
from old.enhanced_logging import get_enhanced_logger, with_enhanced_logging

class EnhancedCrewWrapper:
    """Wrapper para o crew existente com logging melhorado"""
    
    def __init__(self, original_crew_function):
        self.original_function = original_crew_function
        self.logger = get_enhanced_logger("enhanced_crew")
        self.retry_count = 0
        self.max_retries = 2
    
    def __call__(self, data_inicio: str, data_fim: str, **kwargs) -> Any:
        """Executar crew com enhanced logging e retry logic"""
        
        # Calcular período em dias para contexto
        try:
            from datetime import datetime
            start_date = datetime.strptime(data_inicio, '%Y-%m-%d')
            end_date = datetime.strptime(data_fim, '%Y-%m-%d')
            days = (end_date - start_date).days
        except:
            days = 0
        
        # Iniciar operação principal
        op_id = self.logger.start_operation(
            operation="Insights-AI Analysis",
            agent="Crew_System",
            expected_duration=600.0,  # 10 minutos estimado
            total_steps=8  # 8 agentes principais
        )
        
        # Log milestone inicial
        self.logger.log_milestone("INÍCIO DA ANÁLISE", {
            "Período": f"{data_inicio} a {data_fim}",
            "Dias": days,
            "Sistema": "Enhanced Crew v1.0"
        })
        
        # Executar com retry logic para erros conhecidos
        for attempt in range(self.max_retries + 1):
            try:
                # Log tentativa
                if attempt > 0:
                    self.logger.warning(f"Tentativa {attempt + 1}/{self.max_retries + 1}")
                
                # Executar função original
                result = self._execute_with_monitoring(data_inicio, data_fim, **kwargs)
                
                # Sucesso
                self.logger.log_milestone("ANÁLISE CONCLUÍDA", {
                    "Resultado": "Sucesso",
                    "Tentativas": attempt + 1,
                    "Tamanho_Resultado": len(str(result)) if result else 0
                })
                
                self.logger.finish_operation(success=True, message=f"Tentativa {attempt + 1}")
                return result
                
            except Exception as e:
                self.logger.error_with_context(e, "Crew_Execution")
                
                # Verificar se é erro conhecido que justifica retry
                if self._should_retry(e, attempt):
                    self.logger.warning(f"Erro conhecido detectado - tentando novamente")
                    # Pequeno delay antes do retry
                    time.sleep(2 ** attempt)  # Backoff exponencial
                    continue
                else:
                    # Erro não recuperável ou excedeu tentativas
                    self.logger.finish_operation(success=False, message=f"Falha após {attempt + 1} tentativas")
                    raise
        
        # Se chegou aqui, excedeu todas as tentativas
        self.logger.finish_operation(success=False, message="Excedeu tentativas máximas")
        raise RuntimeError(f"Falha após {self.max_retries + 1} tentativas")
    
    def _execute_with_monitoring(self, data_inicio: str, data_fim: str, **kwargs) -> Any:
        """Executar com monitoramento de progresso"""
        
        # Wrapper para capturar logs específicos e estimar progresso
        start_time = time.time()
        last_progress_update = 0
        
        try:
            # Executar função original
            result = self.original_function(data_inicio, data_fim, **kwargs)
            
            # Atualizar progresso final
            self.logger.update_progress(step=8, message="Todas as análises concluídas")
            
            return result
            
        except Exception as e:
            # Log erro com contexto de timing
            elapsed = time.time() - start_time
            self.logger.error_with_context(e, f"Crew_Execution_after_{elapsed:.1f}s")
            raise
    
    def _should_retry(self, error: Exception, attempt: int) -> bool:
        """Determinar se deve tentar novamente baseado no erro"""
        
        if attempt >= self.max_retries:
            return False
        
        error_str = str(error).lower()
        
        # Erros que justificam retry
        retry_conditions = [
            "create_reasoning_plan" in error_str,
            "missing 1 required positional argument: 'ready'" in error_str,
            "connection" in error_str and "timeout" not in error_str,  # Não retry timeout
            "temporary" in error_str,
            "rate limit" in error_str
        ]
        
        should_retry = any(retry_conditions)
        
        if should_retry:
            self.logger.info(f"🔄 Retry autorizado para erro: {type(error).__name__}")
        else:
            self.logger.warning(f"🚫 Retry não recomendado para erro: {type(error).__name__}")
        
        return should_retry

# =============== FUNÇÃO WRAPPER PRINCIPAL ===============

def wrap_crew_function(original_function):
    """Decorator para aplicar enhanced logging a função de crew"""
    
    @functools.wraps(original_function)
    def wrapper(*args, **kwargs):
        enhanced_wrapper = EnhancedCrewWrapper(original_function)
        return enhanced_wrapper(*args, **kwargs)
    
    return wrapper

# =============== ENHANCED TOOLS WRAPPER ===============

class EnhancedToolWrapper:
    """Wrapper para ferramentas individuais"""
    
    def __init__(self, tool_instance):
        self.tool = tool_instance
        self.logger = get_enhanced_logger("enhanced_tools")
        self.tool_name = tool_instance.__class__.__name__
    
    def __getattr__(self, name):
        """Interceptar chamadas de métodos da ferramenta"""
        attr = getattr(self.tool, name)
        
        if callable(attr) and name.startswith(('_run', 'run', 'execute', 'analyze')):
            return self._wrap_tool_method(attr, name)
        
        return attr
    
    def _wrap_tool_method(self, method, method_name):
        """Wrapper para métodos da ferramenta"""
        
        @functools.wraps(method)
        def wrapped(*args, **kwargs):
            # Extrair informações para contexto
            operation_name = f"{self.tool_name}.{method_name}"
            
            # Estimar duração baseada no tipo de ferramenta
            expected_duration = self._estimate_duration(method_name, args, kwargs)
            
            # Iniciar operação
            op_id = self.logger.start_operation(
                operation=operation_name,
                agent=self.tool_name,
                expected_duration=expected_duration
            )
            
            try:
                # Executar método original
                result = method(*args, **kwargs)
                
                # Log resultado
                result_size = len(str(result)) if result else 0
                self.logger.finish_operation(
                    success=True, 
                    message=f"Resultado: {result_size} chars"
                )
                
                return result
                
            except Exception as e:
                self.logger.error_with_context(e, operation_name)
                self.logger.finish_operation(success=False)
                raise
        
        return wrapped
    
    def _estimate_duration(self, method_name: str, args, kwargs) -> float:
        """Estimar duração baseada no tipo de operação"""
        
        # Estimativas baseadas no tipo de ferramenta
        base_estimates = {
            'SQL': 15.0,  # Queries SQL podem demorar
            'Statistical': 30.0,  # Análises estatísticas
            'Advanced': 45.0,  # ML engines
            'Prophet': 60.0,  # Forecasting
            'KPI': 10.0,  # Cálculos KPI
            'Risk': 20.0,  # Análise de risco
        }
        
        for key, duration in base_estimates.items():
            if key in self.tool_name:
                return duration
        
        return 5.0  # Default

# =============== HELPER FUNCTIONS ===============

def enhance_existing_crew_system():
    """Aplicar enhanced logging ao sistema existente de forma não invasiva"""
    
    logger = get_enhanced_logger("system_enhancement")
    
    try:
        # Tentar importar e wrapear a função principal do crew
        from insights.crew_optimized import run_optimized_crew
        
        # Aplicar wrapper
        enhanced_run_optimized_crew = wrap_crew_function(run_optimized_crew)
        
        logger.info("✅ Enhanced logging aplicado ao crew_optimized")
        return enhanced_run_optimized_crew
        
    except ImportError as e:
        logger.warning(f"⚠️ Não foi possível importar crew_optimized: {e}")
        
        try:
            # Fallback para crew original
            from old.crew import Insights
            logger.info("✅ Fallback para crew original")
            return None
            
        except ImportError as e2:
            logger.error_with_context(e2, "system_enhancement")
            raise

def create_enhanced_tool(tool_class, *args, **kwargs):
    """Factory para criar ferramentas com enhanced logging"""
    
    original_tool = tool_class(*args, **kwargs)
    enhanced_tool = EnhancedToolWrapper(original_tool)
    
    logger = get_enhanced_logger("tool_factory")
    logger.info(f"🔧 Enhanced tool criada: {tool_class.__name__}")
    
    return enhanced_tool 