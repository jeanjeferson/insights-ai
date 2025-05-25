"""
🔍 ENHANCED LOGGER - Sistema de Logging Avançado para Testes
============================================================

Sistema completo de logging com métricas de performance, rastreamento
de execução e geração de relatórios detalhados para testes.
"""

import time
import psutil
import logging
import json
import os
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Dict, Any, List, Optional


class EnhancedTestLogger:
    """
    Logger avançado para testes com métricas de performance e trace completo.
    """
    
    def __init__(self, test_name: str, log_level: str = "INFO"):
        self.test_name = test_name
        self.start_time = None
        self.start_memory = None
        self.logs = []
        self.performance_metrics = {}
        
        # Configurar processo para métricas
        try:
            self.process = psutil.Process()
        except:
            self.process = None
        
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
        
        if self.process:
            self.start_memory = self.process.memory_info().rss / 1024 / 1024
        else:
            self.start_memory = 0
        
        self.log("INFO", f"🚀 Iniciando teste: {self.test_name}", **context)
        
        # Log de ambiente
        self.log("INFO", "Ambiente de teste",
                python_version=f"{psutil.cpu_count()} CPUs",
                available_memory=f"{psutil.virtual_memory().available / 1024 / 1024:.1f}MB",
                **context)
    
    def end_test(self, success: bool, **context) -> Dict[str, Any]:
        """Finalizar teste e gerar métricas."""
        if not self.start_time:
            self.log("WARNING", "Teste não foi iniciado corretamente")
            return {}
        
        duration = time.time() - self.start_time
        
        if self.process:
            end_memory = self.process.memory_info().rss / 1024 / 1024
            memory_used = end_memory - self.start_memory
        else:
            end_memory = 0
            memory_used = 0
        
        # Métricas finais
        final_metrics = {
            'test_name': self.test_name,
            'success': success,
            'duration_seconds': round(duration, 3),
            'memory_used_mb': round(memory_used, 2),
            'memory_peak_mb': round(end_memory, 2),
            'timestamp': datetime.now().isoformat(),
            'total_logs': len(self.logs),
            'logs': self.logs,
            **context
        }
        
        # Log final
        status = "✅ SUCESSO" if success else "❌ FALHA"
        self.log("INFO", f"{status} - Teste finalizado",
                duration=f"{duration:.2f}s",
                memory_used=f"{memory_used:.1f}MB")
        
        # Salvar relatório detalhado
        self._save_detailed_report(final_metrics)
        
        return final_metrics
    
    def log(self, level: str, message: str, **kwargs):
        """Adicionar entrada de log com métricas."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'elapsed_time_s': round(elapsed_time, 3),
            **kwargs
        }
        
        # Adicionar métricas instantâneas se disponível
        if self.process:
            try:
                cpu_percent = self.process.cpu_percent()
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                log_entry.update({
                    'cpu_percent': round(cpu_percent, 1),
                    'memory_current_mb': round(memory_mb, 1)
                })
            except:
                pass
        
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
            
            # Salvar resumo em TXT
            summary_file = log_dir / f"{self.test_name}_{timestamp}_summary.txt"
            self._save_summary_report(summary_file, metrics)
            
            self.log("INFO", f"📄 Relatório salvo: {log_file}")
            
        except Exception as e:
            print(f"⚠️ Erro ao salvar relatório: {str(e)}")
    
    def _save_summary_report(self, file_path: Path, metrics: Dict[str, Any]):
        """Salvar relatório resumido legível."""
        summary = f"""
# 📊 RELATÓRIO DE TESTE: {self.test_name}
==================================================

## ℹ️ Informações Gerais
- **Teste**: {self.test_name}
- **Status**: {'✅ SUCESSO' if metrics['success'] else '❌ FALHA'}
- **Data/Hora**: {datetime.fromisoformat(metrics['timestamp']).strftime('%d/%m/%Y %H:%M:%S')}
- **Duração**: {metrics['duration_seconds']}s
- **Uso de Memória**: {metrics['memory_used_mb']}MB

## 📈 Métricas de Performance
- **Tempo Total**: {metrics['duration_seconds']}s
- **Pico de Memória**: {metrics['memory_peak_mb']}MB
- **Total de Logs**: {metrics['total_logs']}

## 📋 Log de Execução (Últimas 10 entradas)
"""
        
        # Adicionar últimas 10 entradas de log
        recent_logs = metrics['logs'][-10:] if metrics['logs'] else []
        for log_entry in recent_logs:
            timestamp = datetime.fromisoformat(log_entry['timestamp']).strftime('%H:%M:%S')
            summary += f"[{timestamp}] {log_entry['level']}: {log_entry['message']}\n"
        
        summary += f"""
## 🎯 Performance Benchmarks
- **Tempo de Execução**: {'✅ Rápido' if metrics['duration_seconds'] < 30 else '⚠️ Lento'}
- **Uso de Memória**: {'✅ Eficiente' if metrics['memory_used_mb'] < 100 else '⚠️ Alto'}
- **Estabilidade**: {'✅ Estável' if metrics['success'] else '❌ Instável'}

## 📊 Dados de Performance
{json.dumps(self.performance_metrics, indent=2) if self.performance_metrics else 'Nenhuma métrica específica coletada'}
"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(summary)


def log_test_execution(include_performance=True, log_level="INFO"):
    """
    Decorator para logging automático de testes.
    
    Args:
        include_performance: Se deve incluir métricas de performance
        log_level: Nível de logging (INFO, DEBUG, WARNING, ERROR)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extrair classe de teste se disponível
            test_class = ""
            if args and hasattr(args[0], '__class__'):
                test_class = f"{args[0].__class__.__name__}."
            
            test_name = f"{test_class}{func.__name__}"
            logger = EnhancedTestLogger(test_name, log_level)
            
            # Contexto inicial
            context = {
                'function': func.__name__,
                'module': func.__module__,
                'include_performance': include_performance
            }
            
            logger.start_test(**context)
            
            success = False
            result = None
            
            try:
                # Executar função com checkpoint
                logger.create_checkpoint("execution_start")
                
                if include_performance:
                    start_perf = time.time()
                
                result = func(*args, **kwargs)
                
                if include_performance:
                    perf_duration = time.time() - start_perf
                    logger.log_performance("function_execution", perf_duration)
                
                logger.create_checkpoint("execution_end", result_type=type(result).__name__)
                success = True
                
            except Exception as e:
                logger.log_error(e, f"Erro durante execução de {func.__name__}")
                success = False
                raise
            
            finally:
                # Finalizar logging
                final_metrics = logger.end_test(success, result_available=result is not None)
                
                # Adicionar métricas ao resultado se for um dicionário
                if isinstance(result, dict) and success:
                    result['_test_metrics'] = final_metrics
            
            return result
        
        return wrapper
    return decorator


class TestExecutionTracker:
    """
    Rastreador global de execução de testes para análise consolidada.
    """
    
    def __init__(self):
        self.executions = []
        self.start_time = time.time()
    
    def add_execution(self, metrics: Dict[str, Any]):
        """Adicionar execução ao tracker."""
        self.executions.append(metrics)
    
    def generate_consolidated_report(self) -> str:
        """Gerar relatório consolidado de todas as execuções."""
        if not self.executions:
            return "Nenhuma execução registrada"
        
        total_duration = time.time() - self.start_time
        successful_tests = [e for e in self.executions if e.get('success', False)]
        failed_tests = [e for e in self.executions if not e.get('success', False)]
        
        avg_duration = sum(e.get('duration_seconds', 0) for e in self.executions) / len(self.executions)
        avg_memory = sum(e.get('memory_used_mb', 0) for e in self.executions) / len(self.executions)
        
        report = f"""
# 📊 RELATÓRIO CONSOLIDADO DE TESTES
=====================================

## 📈 Estatísticas Gerais
- **Total de Testes**: {len(self.executions)}
- **Sucessos**: {len(successful_tests)} ({len(successful_tests)/len(self.executions)*100:.1f}%)
- **Falhas**: {len(failed_tests)} ({len(failed_tests)/len(self.executions)*100:.1f}%)
- **Tempo Total**: {total_duration:.2f}s
- **Tempo Médio por Teste**: {avg_duration:.2f}s
- **Memória Média**: {avg_memory:.1f}MB

## 🏆 Top 5 Testes Mais Rápidos
{chr(10).join([f"- {e['test_name']}: {e['duration_seconds']:.2f}s" for e in sorted(successful_tests, key=lambda x: x['duration_seconds'])[:5]])}

## 🐌 Top 5 Testes Mais Lentos
{chr(10).join([f"- {e['test_name']}: {e['duration_seconds']:.2f}s" for e in sorted(successful_tests, key=lambda x: x['duration_seconds'], reverse=True)[:5]])}

## ❌ Testes Falharam
{chr(10).join([f"- {e['test_name']}: {e.get('error', 'Erro não especificado')}" for e in failed_tests]) if failed_tests else 'Nenhum teste falhou! 🎉'}

## 📊 Análise de Performance
- **Teste mais eficiente em memória**: {min(successful_tests, key=lambda x: x.get('memory_used_mb', float('inf')))['test_name'] if successful_tests else 'N/A'}
- **Maior uso de memória**: {max(successful_tests, key=lambda x: x.get('memory_used_mb', 0))['test_name'] if successful_tests else 'N/A'}
"""
        
        return report
    
    def save_consolidated_report(self, filename: str = None):
        """Salvar relatório consolidado em arquivo."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"consolidated_test_report_{timestamp}.md"
        
        report = self.generate_consolidated_report()
        
        log_dir = Path("test_logs")
        log_dir.mkdir(exist_ok=True)
        
        file_path = log_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Relatório consolidado salvo: {file_path}")
        return file_path


# Instância global do tracker
global_test_tracker = TestExecutionTracker()


if __name__ == "__main__":
    # Exemplo de uso do logger
    logger = EnhancedTestLogger("exemplo_teste")
    logger.start_test(exemplo=True)
    
    logger.log("INFO", "Testando funcionalidade básica")
    logger.create_checkpoint("meio_do_teste", progress=50)
    
    time.sleep(1)  # Simular processamento
    
    logger.log("SUCCESS", "Teste concluído com sucesso")
    metrics = logger.end_test(success=True)
    
    print(f"\n📊 Métricas finais: {json.dumps(metrics, indent=2)}")
