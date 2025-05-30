#!/usr/bin/env python
"""
🚀 INSIGHTS-AI - MAIN ATUALIZADO PARA SISTEMA HÍBRIDO

Agora suporta:
- Sistema tradicional Crew (compatibilidade total)
- Sistema otimizado Flow (nova funcionalidade) 
- Modo automático (decide a melhor opção)
- Monitoramento em tempo real
- Recovery automático

Para usar:
- python main.py                    # Modo automático (recomendado)
- python main.py --mode crew        # Força uso do sistema tradicional
- python main.py --mode flow        # Força uso do sistema otimizado
"""

import sys
import warnings
import logging
import time
import threading
import argparse
from datetime import datetime, timedelta

# Importar novo sistema híbrido
from insights.flow_integration import run_insights, InsightsRunner

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# =============== CONFIGURAÇÃO DE LOGGING MELHORADA ===============

def setup_main_logging():
    """Configurar logging específico para o main"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] MAIN: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('insights_execution.log')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_main_logging()

# =============== CONFIGURAÇÃO DE PARÂMETROS ===============

def parse_arguments():
    """Processar argumentos da linha de comando"""
    parser = argparse.ArgumentParser(
        description='🚀 Insights-AI - Sistema de Análise Inteligente para Joalherias',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Exemplos de uso:
                python main.py                                    # Modo automático (recomendado)
                python main.py --mode flow                        # Força sistema otimizado
                python main.py --mode crew                        # Força sistema tradicional
                python main.py --start 2023-01-01 --end 2024-01-01  # Período específico
                python main.py --mode flow --quick                # Modo rápido com Flow
                        """
                    )
                    
    parser.add_argument(
        '--mode', 
        choices=['auto', 'crew', 'flow'], 
        default='auto',
        help='Modo de execução (padrão: auto)'
    )
    
    parser.add_argument(
        '--start', 
        type=str, 
        help='Data de início (YYYY-MM-DD). Padrão: 2 anos atrás'
    )
    
    parser.add_argument(
        '--end', 
        type=str, 
        help='Data de fim (YYYY-MM-DD). Padrão: hoje'
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Modo de execução rápida (só funciona com Flow)'
    )
    
    parser.add_argument(
        '--no-monitor', 
        action='store_true',
        help='Desabilitar monitoramento detalhado'
    )
    
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Habilitar logging verbose'
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
    
    # Data início padrão: 2 anos atrás (ou 4 anos se especificado)
    if args.start:
        data_inicio = args.start
        # Validar formato
        try:
            datetime.strptime(data_inicio, '%Y-%m-%d')
        except ValueError:
            raise ValueError(f"Formato de data inválido para --start: {data_inicio}. Use YYYY-MM-DD")
    else:
        # Padrão: últimos 4 anos para análise robusta
        data_inicio = (datetime.now() - timedelta(days=1460)).strftime('%Y-%m-%d')
    
    return data_inicio, data_fim

# =============== FUNÇÃO DE PROGRESSO MELHORADA ===============

def progress_monitor(start_time, stop_event, modo_execucao):
    """Monitor de progresso aprimorado que se adapta ao modo de execução"""
    logger.info(f"📡 Monitor de progresso iniciado para modo: {modo_execucao}")
    
    intervalo = 30 if modo_execucao == 'flow' else 60  # Flow atualiza mais frequentemente
    
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        logger.info(f"⏱️ EXECUTANDO há {minutes:02d}:{seconds:02d} - Sistema {modo_execucao.upper()} ativo...")
        
        # Log de recursos se disponível
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            logger.info(f"📊 Recursos: {memory_mb:.1f}MB RAM, {cpu_percent:.1f}% CPU")
        except ImportError:
            pass
        
        # Aguardar antes do próximo log
        stop_event.wait(intervalo)

# =============== FUNÇÃO PRINCIPAL ATUALIZADA ===============

def run():
    """
    Função principal atualizada com sistema híbrido
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
        logger.info("🚀 INICIANDO INSIGHTS-AI HÍBRIDO")
        logger.info("=" * 60)
        logger.info(f"📅 Período de análise: {data_inicio} até {data_fim}")
        logger.info(f"🔧 Modo solicitado: {args.mode}")
        logger.info(f"⚡ Modo rápido: {'Sim' if args.quick else 'Não'}")
        logger.info(f"📡 Monitoramento: {'Não' if args.no_monitor else 'Sim'}")
        logger.info(f"⏰ Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Preparar parâmetros para execução
        kwargs = {
            'modo_rapido': args.quick,
            'monitoramento_detalhado': not args.no_monitor,
            'verbose': args.verbose
        }
        
        # Ajustar modo se quick foi solicitado com crew
        modo_final = args.mode
        if args.quick and args.mode == 'crew':
            logger.warning("⚠️ Modo rápido não suportado com Crew - Mudando para Flow")
            modo_final = 'flow'
        
        # Iniciar monitor de progresso se monitoramento não foi desabilitado
        if not args.no_monitor:
            stop_event = threading.Event()
            progress_thread = threading.Thread(
                target=progress_monitor, 
                args=(start_time, stop_event, modo_final),
                daemon=True
            )
            progress_thread.start()
            logger.info("📡 Monitor de progresso iniciado")
        else:
            stop_event = None
            progress_thread = None
        
        try:
            # EXECUÇÃO PRINCIPAL COM SISTEMA HÍBRIDO
            logger.info(f"🚀 Executando com sistema híbrido - Modo: {modo_final}")
            
            resultado = run_insights(
                data_inicio=data_inicio,
                data_fim=data_fim,
                modo=modo_final,
                **kwargs
            )
            
            execution_time = time.time() - start_time
            
            # Log do resultado
            logger.info("✅ EXECUÇÃO CONCLUÍDA COM SUCESSO")
            logger.info("=" * 60)
            logger.info(f"📊 Sistema utilizado: {resultado['modo_execucao'].upper()}")
            logger.info(f"⏱️ Tempo total: {execution_time:.2f} segundos")
            logger.info(f"🎯 Status: {resultado['status']}")
            
            # Logs específicos por modo
            if resultado['modo_execucao'] == 'flow':
                logger.info(f"📈 Análises concluídas: {resultado.get('analises_concluidas', 0)}")
                logger.info(f"📁 Arquivos gerados: {len(resultado.get('arquivos_gerados', []))}")
                logger.info(f"⚠️ Warnings: {len(resultado.get('warnings', []))}")
                logger.info(f"📊 Score qualidade dados: {resultado.get('qualidade_dados', {}).get('score_confiabilidade', 0):.1f}/100")
            elif resultado['modo_execucao'] == 'crew':
                crew_info = resultado.get('crew_info', {})
                logger.info(f"👥 Agentes utilizados: {crew_info.get('agents_count', 0)}")
                logger.info(f"📋 Tasks executadas: {crew_info.get('tasks_count', 0)}")
            
            logger.info("=" * 60)
            
            return resultado['resultado_principal']
            
        finally:
            # Parar monitor de progresso
            if stop_event and progress_thread:
                stop_event.set()
                logger.info("📡 Monitor de progresso finalizado")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ EXECUÇÃO INTERROMPIDA pelo usuário")
        raise
    except ValueError as e:
        logger.error(f"❌ ERRO de parâmetros: {e}")
        sys.exit(1)
    except Exception as e:
        execution_time = time.time() - start_time if 'start_time' in locals() else 0
        logger.error(f"❌ ERRO após {execution_time:.2f} segundos: {str(e)}")
        logger.exception("Stack trace completo:")
        raise Exception(f"An error occurred while running the crew: {e}")

# =============== FUNCÕES DE CONVENIÊNCIA ===============

def run_with_flow(data_inicio=None, data_fim=None):
    """Executar forçando uso do Flow (para testes)"""
    if data_fim is None:
        data_fim = datetime.now().strftime('%Y-%m-%d')
    if data_inicio is None:
        data_inicio = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    return run_insights(data_inicio, data_fim, modo="flow")

def run_with_crew(data_inicio=None, data_fim=None):
    """Executar forçando uso do Crew tradicional (para compatibilidade)"""
    if data_fim is None:
        data_fim = datetime.now().strftime('%Y-%m-%d')
    if data_inicio is None:
        data_inicio = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    return run_insights(data_inicio, data_fim, modo="crew")

def run_quick():
    """Execução rápida com Flow (para desenvolvimento)"""
    data_fim = datetime.now().strftime('%Y-%m-%d')
    data_inicio = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')  # 3 meses
    
    return run_insights(
        data_inicio, 
        data_fim, 
        modo="flow", 
        modo_rapido=True
    )

# =============== STATUS E HELP ===============

def print_system_status():
    """Imprimir status do sistema híbrido"""
    print("🚀 INSIGHTS-AI - STATUS DO SISTEMA")
    print("=" * 50)
    
    # Verificar disponibilidade do Flow
    try:
        import crewai.flow.flow
        print("✅ CrewAI Flow: Disponível")
    except ImportError:
        print("❌ CrewAI Flow: Não disponível")
    
    # Verificar crew tradicional
    try:
        from old.crew import Insights
        print("✅ Crew Tradicional: Disponível")
    except ImportError:
        print("❌ Crew Tradicional: Não disponível")
    
    # Verificar integração
    try:
        from insights.flow_integration import InsightsRunner
        print("✅ Sistema Híbrido: Operacional")
    except ImportError:
        print("❌ Sistema Híbrido: Erro na integração")
    
    # Verificar recursos
    try:
        import psutil
        memoria_gb = psutil.virtual_memory().available / (1024**3)
        print(f"📊 Memória disponível: {memoria_gb:.1f}GB")
    except ImportError:
        print("ℹ️ Monitoramento de recursos: Não disponível")
    
    print("=" * 50)

def print_usage_examples():
    """Imprimir exemplos de uso"""
    print("📚 EXEMPLOS DE USO DO INSIGHTS-AI")
    print("=" * 50)
    print()
    print("💻 Via linha de comando:")
    print("  python main.py                           # Modo automático")
    print("  python main.py --mode flow               # Força Flow")
    print("  python main.py --mode crew               # Força Crew")
    print("  python main.py --quick                   # Execução rápida")
    print("  python main.py --start 2023-01-01       # Período específico")
    print()
    print("🐍 Via código Python:")
    print("  from insights.main import run_with_flow, run_quick")
    print("  resultado = run_with_flow('2023-01-01', '2024-01-01')")
    print("  resultado_rapido = run_quick()")
    print()
    print("🔧 Via integração direta:")
    print("  from insights.flow_integration import run_insights")
    print("  resultado = run_insights('2023-01-01', '2024-01-01', 'auto')")
    print("=" * 50)

if __name__ == "__main__":
    # Verificar se foi solicitado help ou status
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print_usage_examples()
            sys.exit(0)
        elif sys.argv[1] == '--status':
            print_system_status()
            sys.exit(0)
    
    # Execução normal
    run()