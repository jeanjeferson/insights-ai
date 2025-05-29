#!/usr/bin/env python
"""
🚀 INSIGHTS-AI - Sistema Simplificado e Eficiente

Mantém todas as funcionalidades do crew.py original mas elimina a complexidade desnecessária.
Todas as ferramentas, agentes e tasks permanecem intactas.
"""

import sys
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Importar o crew existente (mantido intacto)
from insights.crew import Insights

def setup_simple_logging():
    """Configuração de logging simplificada"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('insights_execution.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_simple_logging()

def parse_arguments():
    """Processar argumentos da linha de comando de forma simples"""
    parser = argparse.ArgumentParser(
        description='🚀 Insights-AI - Análise Inteligente para Joalherias',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--start', 
        type=str, 
        help='Data de início (YYYY-MM-DD). Padrão: 4 anos atrás'
    )
    
    parser.add_argument(
        '--end', 
        type=str, 
        help='Data de fim (YYYY-MM-DD). Padrão: hoje'
    )
    
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Habilitar logging detalhado'
    )
    
    return parser.parse_args()

def configure_dates(args):
    """Configurar datas baseado nos argumentos"""
    # Data fim padrão: hoje
    if args.end:
        data_fim = args.end
        # Validar formato
        try:
            datetime.strptime(data_fim, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Formato de data inválido para --end: {data_fim}. Use YYYY-MM-DD")
    else:
        data_fim = datetime.now().strftime('%Y-%m-%d')
    
    # Data início padrão: 4 anos atrás
    if args.start:
        data_inicio = args.start
        # Validar formato
        try:
            datetime.strptime(data_inicio, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Formato de data inválido para --start: {data_inicio}. Use YYYY-MM-DD")
    else:
        data_inicio = (datetime.now() - timedelta(days=1460)).strftime('%Y-%m-%d')
    
    return data_inicio, data_fim

def run():
    """
    Função principal simplificada - executa o crew existente sem complexidade adicional
    """
    try:
        # Processar argumentos
        args = parse_arguments()
        
        # Configurar logging verbose se solicitado
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("🔧 Modo verbose habilitado")
        
        # Configurar datas
        data_inicio, data_fim = configure_dates(args)
        
        # Log inicial
        logger.info("🚀 INICIANDO INSIGHTS-AI")
        logger.info("=" * 50)
        logger.info(f"📅 Período de análise: {data_inicio} até {data_fim}")
        logger.info(f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 50)
        
        # Criar e executar o crew (sua implementação existente)
        logger.info("🤖 Inicializando crew com todas as ferramentas...")
        crew_instance = Insights()
        
        # Preparar inputs
        inputs = {
            'data_inicio': data_inicio,
            'data_fim': data_fim
        }
        
        logger.info(f"📊 Executando análise com inputs: {inputs}")
        
        # Executar o crew
        start_time = datetime.now()
        result = crew_instance.crew().kickoff(inputs=inputs)
        end_time = datetime.now()
        
        # Log final
        execution_time = (end_time - start_time).total_seconds()
        logger.info("=" * 50)
        logger.info("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
        logger.info(f"⏱️ Tempo total de execução: {execution_time:.2f} segundos")
        logger.info(f"📄 Resultado: {len(str(result))} caracteres gerados")
        logger.info(f"⏰ Finalizado em: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 50)
        
        return result
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Execução interrompida pelo usuário")
        return None
    except Exception as e:
        logger.error(f"❌ Erro durante a execução: {e}")
        raise

def train(n_iterations: int = 1, filename: str = None):
    """Função de treinamento simplificada"""
    logger.info(f"🎯 Iniciando treinamento com {n_iterations} iterações")
    
    crew_instance = Insights()
    
    # Usar dados padrão para treinamento se não especificado
    data_fim = datetime.now().strftime('%Y-%m-%d')
    data_inicio = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    inputs = {
        'data_inicio': data_inicio,
        'data_fim': data_fim
    }
    
    try:
        crew_instance.crew().train(
            n_iterations=n_iterations,
            filename=filename,
            inputs=inputs
        )
        logger.info("✅ Treinamento concluído com sucesso")
    except Exception as e:
        logger.error(f"❌ Erro durante o treinamento: {e}")
        raise

def replay(task_id: str):
    """Função de replay simplificada"""
    logger.info(f"🔄 Executando replay da task: {task_id}")
    
    crew_instance = Insights()
    
    try:
        crew_instance.crew().replay(task_id=task_id)
        logger.info("✅ Replay concluído com sucesso")
    except Exception as e:
        logger.error(f"❌ Erro durante o replay: {e}")
        raise

def test():
    """Função de teste simplificada"""
    logger.info("🧪 Executando teste do sistema")
    
    # Teste com período pequeno
    data_fim = datetime.now().strftime('%Y-%m-%d')
    data_inicio = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    logger.info(f"📊 Teste com período reduzido: {data_inicio} a {data_fim}")
    
    try:
        crew_instance = Insights()
        inputs = {
            'data_inicio': data_inicio,
            'data_fim': data_fim
        }
        
        result = crew_instance.crew().kickoff(inputs=inputs)
        logger.info("✅ Teste concluído com sucesso")
        return result
    except Exception as e:
        logger.error(f"❌ Erro durante o teste: {e}")
        raise

if __name__ == "__main__":
    run() 