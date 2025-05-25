"""
📝 SIMPLE ENHANCED LOGGER - Logger sem dependências externas
===========================================================

Logger simplificado sem dependência do psutil para logging básico de testes.
"""

import time
import logging
import json
import os
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Dict, Any, List, Optional


class SimpleEnhancedTestLogger:
    """
    Logger simplificado para testes sem dependências externas.
    """
    
    def __init__(self, test_name: str, log_level: str = "INFO"):
        self.test_name = test_name
        self.start_time = None
        self.logs = []
        self.performance_metrics = {}
        
        # Configurar logging padrão
        self.logger = logging.getLogger(f"test_{test_name}")
        self.logger.setLevel(getattr(logging, log_level))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def start_test(self, **context):
        """Iniciar logging do teste com contexto adicional."""
        self.start_time = time.time()
        
        self.log("INFO", f"🚀 Iniciando teste: {self.test_name}", **context)
        
        # Log de ambiente
        self.log("INFO", "Ambiente de teste", **context)
    
    def end_test(self, success: bool, **context) -> Dict[str, Any]:
        """Finalizar teste e gerar métricas."""
        if not self.start_time:
            self.log("WARNING", "Teste não foi iniciado corretamente")
            return {}
        
        duration = time.time() - self.start_time
        
        # Métricas finais
        final_metrics = {
            'test_name': self.test_name,
            'success': success,
            'duration_seconds': round(duration, 3),
            'timestamp': datetime.now().isoformat(),
            'total_logs': len(self.logs),
            'logs': self.logs,
            **context
        }
        
        # Log final
        status = "✅ SUCESSO" if success else "❌ FALHA"
        self.log("INFO", f"{status} - Teste finalizado",
                duration=f"{duration:.2f}s")
        
        # Salvar relatório detalhado
        self._save_detailed_report(final_metrics)
        
        return final_metrics
    
    def log(self, level: str, message: str, **kwargs):
        """Adicionar entrada de log."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'elapsed_time_s': round(elapsed_time, 3),
            **kwargs
        }
        
        self.logs.append(log_entry)
        
        # Log no console com formatação
        console_msg = f"[{level}] {message}"
        if kwargs:
            details = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            console_msg += f" | {details}"
        
        print(f"[{elapsed_time:6.2f}s] {console_msg}")
        
        # Log no logger padrão
        getattr(self.logger, level.lower(), self.logger.info)(console_msg)
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """Log específico para métricas de performance."""
        self.performance_metrics[operation] = {
            'duration_s': round(duration, 3),
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.log("PERFORMANCE", f"⚡ {operation}",
                duration=f"{duration:.3f}s",
                **metrics)
    
    def log_error(self, error: Exception, context: str = ""):
        """Log específico para erros com stack trace."""
        import traceback
        
        error_details = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'stack_trace': traceback.format_exc()
        }
        
        self.log("ERROR", f"💥 Erro capturado: {str(error)}", **error_details)
    
    def create_checkpoint(self, name: str, **context):
        """Criar checkpoint durante execução."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        checkpoint = {
            'name': name,
            'elapsed_time_s': round(elapsed, 3),
            'timestamp': datetime.now().isoformat(),
            **context
        }
        
        self.log("CHECKPOINT", f"🏁 Checkpoint: {name}", **checkpoint)
        return checkpoint
    
    def _save_detailed_report(self, metrics: Dict[str, Any]):
        """Salvar relatório detalhado em arquivo."""
        try:
            # Criar diretório de logs
            log_dir = Path("test_logs")
            log_dir.mkdir(exist_ok=True)
            
            # Nome do arquivo com timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{self.test_name}_{timestamp}.json"
            
            # Salvar JSON detalhado
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            
            print(f"📁 Log salvo em: {log_file}")
            
        except Exception as e:
            print(f"⚠️ Erro ao salvar log: {str(e)}")


def simple_log_test_execution(include_performance: bool = True, log_level: str = "INFO"):
    """
    Decorador simplificado para logging automático de execução de testes.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extrair instância da classe de teste se disponível
            test_instance = None
            if args and hasattr(args[0], '__class__'):
                test_instance = args[0]
            
            # Obter logger se disponível na instância
            logger = getattr(test_instance, 'logger', None) if test_instance else None
            
            # Configurar métricas iniciais
            start_time = time.time()
            
            # Log início
            if logger:
                logger.log(log_level, f"🚀 Iniciando {func.__name__}")
            else:
                print(f"🚀 [{datetime.now().strftime('%H:%M:%S')}] Iniciando {func.__name__}")
            
            try:
                # Executar função original
                result = func(*args, **kwargs)
                success = True
                
            except Exception as e:
                success = False
                
                if logger:
                    logger.log_error(e, f"Erro em {func.__name__}")
                else:
                    print(f"❌ Erro em {func.__name__}: {str(e)}")
                
                # Re-raise a exceção para pytest
                raise
                
            finally:
                # Calcular métricas finais
                duration = time.time() - start_time
                
                # Log final
                status = "✅ SUCESSO" if success else "❌ FALHA"
                
                if logger:
                    logger.log(log_level, f"{status} - {func.__name__} concluído",
                             duration=f"{duration:.2f}s")
                else:
                    print(f"{status} [{datetime.now().strftime('%H:%M:%S')}] {func.__name__} | {duration:.2f}s")
            
            return result
        
        return wrapper
    return decorator 