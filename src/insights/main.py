#!/usr/bin/env python
import sys
import warnings
import logging
import time
import threading
from datetime import datetime, timedelta

from insights.crew import Insights

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Configurar logging detalhado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('insights_execution.log')
    ]
)

logger = logging.getLogger(__name__)

# Definindo as datas para a consulta (últimos 2 anos)
data_fim = datetime.now().strftime('%Y-%m-%d')
data_inicio = (datetime.now() - timedelta(days=1460)).strftime('%Y-%m-%d')

def progress_monitor(start_time, stop_event):
    """Monitor de progresso que roda em thread separada"""
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        logger.info(f"⏱️ EXECUTANDO há {minutes:02d}:{seconds:02d} - Sistema ativo...")
        
        # Log de memória/recursos se possível
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            logger.info(f"📊 Recursos: {memory_mb:.1f}MB RAM, {cpu_percent:.1f}% CPU")
        except ImportError:
            pass
        
        # Aguardar 30 segundos antes do próximo log
        stop_event.wait(30)

def run():
    """
    Run the crew with detailed logging.
    """
    start_time = time.time()
    
    inputs = {
        'data_inicio': data_inicio,
        'data_fim': data_fim
    }
    
    logger.info("🚀 INICIANDO INSIGHTS-AI CREW")
    logger.info(f"📅 Período: {data_inicio} até {data_fim}")
    logger.info(f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        logger.info("🔄 Inicializando crew...")
        crew_instance = Insights()
        
        logger.info("✅ Crew inicializada com sucesso")
        logger.info("🚀 Iniciando kickoff...")
        
        # Iniciar monitor de progresso em thread separada
        stop_event = threading.Event()
        progress_thread = threading.Thread(
            target=progress_monitor, 
            args=(start_time, stop_event),
            daemon=True
        )
        progress_thread.start()
        logger.info("📡 Monitor de progresso iniciado")
        
        try:
            # Executar com callback de progresso
            result = crew_instance.crew().kickoff(inputs=inputs)
            
            execution_time = time.time() - start_time
            logger.info(f"✅ EXECUÇÃO CONCLUÍDA em {execution_time:.2f} segundos")
            logger.info(f"📊 Resultado: {type(result)}")
            
            return result
        finally:
            # Parar monitor de progresso
            stop_event.set()
            logger.info("📡 Monitor de progresso finalizado")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ EXECUÇÃO INTERROMPIDA pelo usuário")
        raise
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"❌ ERRO após {execution_time:.2f} segundos: {str(e)}")
        logger.exception("Stack trace completo:")
        raise Exception(f"An error occurred while running the crew: {e}")


if __name__ == "__main__":
    run()