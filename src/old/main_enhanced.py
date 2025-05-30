#!/usr/bin/env python
"""
⚡ MAIN ENHANCED - INSIGHTS-AI FASE 1
====================================

Versão melhorada do main_optimized.py com:
- Enhanced logging com context-aware error handling
- Progress indicators inteligentes
- Retry logic para erros conhecidos
- Mantém total compatibilidade com código existente

Uso igual ao main_optimized.py:
    python main_enhanced.py                    # Último mês
    python main_enhanced.py --days 60          # Últimos 60 dias
    python main_enhanced.py --start 2024-01-01 --end 2024-12-31
    python main_enhanced.py --debug            # Modo debug
"""

import os
import sys
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

# Adicionar src ao Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Importar enhanced logging
from old.enhanced_logging import get_enhanced_logger, reset_enhanced_logger
from old.enhanced_crew_wrapper import enhance_existing_crew_system

# =============== CONFIGURAÇÃO DE ENVIRONMENT ===============

def setup_environment(args):
    """Configurar ambiente baseado nos argumentos"""
    
    logger = get_enhanced_logger("environment_setup")
    
    # Configurar nível de logging
    if args.debug:
        os.environ['INSIGHTS_DEBUG'] = 'true'
        logger.info("🐛 Modo DEBUG ativado")
    elif args.minimal:
        os.environ['INSIGHTS_LOG_LEVEL'] = 'MINIMAL'
        logger.info("🔇 Modo MINIMAL ativado")
    elif args.verbose:
        os.environ['INSIGHTS_LOG_LEVEL'] = 'VERBOSE'
        logger.info("📢 Modo VERBOSE ativado")
    
    # Configurar environment
    if args.production:
        os.environ['ENVIRONMENT'] = 'production'
        logger.info("🏭 Modo PRODUÇÃO ativado")
    
    # Configurar cache
    if args.no_cache:
        os.environ['INSIGHTS_DISABLE_CACHE'] = 'true'
        logger.info("🚫 Cache DESABILITADO")
    
    logger.info("✅ Environment configurado")

# =============== VALIDAÇÃO DE DATAS ===============

def validate_dates(data_inicio: str, data_fim: str):
    """Validar formato e lógica das datas"""
    
    logger = get_enhanced_logger("date_validation")
    
    try:
        start_date = datetime.strptime(data_inicio, '%Y-%m-%d')
        end_date = datetime.strptime(data_fim, '%Y-%m-%d')
        
        if start_date > end_date:
            raise ValueError("Data início não pode ser posterior à data fim")
        
        if end_date > datetime.now():
            logger.warning("Data fim está no futuro - usando dados disponíveis")
        
        days_diff = (end_date - start_date).days
        if days_diff > 365:
            logger.warning(f"Período muito longo ({days_diff} dias) - pode impactar performance")
        
        logger.info(f"✅ Datas validadas: {days_diff} dias de análise")
        return True
        
    except ValueError as e:
        logger.error_with_context(e, "date_validation")
        return False

# =============== CÁLCULO DE DATAS ===============

def calculate_dates(args):
    """Calcular datas baseado nos argumentos"""
    
    if args.start and args.end:
        # Datas específicas
        return args.start, args.end
    
    elif args.days:
        # Últimos N dias
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    else:
        # Padrão: período configurável (reduzido para performance)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # Reduzido de 1430 para 90 dias
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

# =============== EXIBIÇÃO DE INFORMAÇÕES ===============

def show_execution_info(data_inicio: str, data_fim: str, args):
    """Exibir informações da execução"""
    
    logger = get_enhanced_logger("execution_info")
    
    # Período
    start_date = datetime.strptime(data_inicio, '%Y-%m-%d')
    end_date = datetime.strptime(data_fim, '%Y-%m-%d')
    days = (end_date - start_date).days
    
    # Configurações
    config_info = []
    if args.debug:
        config_info.append("DEBUG")
    elif args.minimal:
        config_info.append("MINIMAL") 
    elif args.verbose:
        config_info.append("VERBOSE")
    else:
        config_info.append("NORMAL")
        
    if args.production:
        config_info.append("PRODUÇÃO")
    else:
        config_info.append("DESENVOLVIMENTO")
        
    if args.no_cache:
        config_info.append("SEM CACHE")
    else:
        config_info.append("COM CACHE")
    
    # Log milestone de configuração
    logger.log_milestone("CONFIGURAÇÃO DO SISTEMA", {
        "Período": f"{data_inicio} até {data_fim} ({days} dias)",
        "Início": datetime.now().strftime('%H:%M:%S'),
        "Configuração": ' | '.join(config_info),
        "Sistema": "Enhanced Logging v1.0"
    })

def show_performance_metrics():
    """Exibir métricas de performance"""
    
    logger = get_enhanced_logger("performance_metrics")
    
    try:
        # Métricas do enhanced logger
        stats = logger.get_stats()
        
        logger.log_milestone("MÉTRICAS DE PERFORMANCE", {
            "Operações_ativas": stats.get('active_operations', 0),
            "Total_warnings": stats.get('total_warnings', 0),
            "Total_errors": stats.get('total_errors', 0),
            "Memória_MB": f"{stats.get('memory_usage_mb', 0):.1f}"
        })
        
        # Tentar obter métricas do sistema original
        try:
            from insights.crew_optimized import get_performance_metrics
            original_metrics = get_performance_metrics()
            
            if original_metrics:
                logger.info("📊 Métricas do sistema original:")
                for key, value in original_metrics.items():
                    logger.info(f"   • {key}: {value}")
                    
        except Exception as e:
            logger.debug(f"Não foi possível obter métricas originais: {e}")
            
    except Exception as e:
        logger.error_with_context(e, "performance_metrics")

# =============== EXECUÇÃO PRINCIPAL ===============

def run_insights_enhanced(data_inicio: str, data_fim: str):
    """Executar o Insights-AI com enhanced logging"""
    
    logger = get_enhanced_logger("main_execution")
    execution_start = time.time()
    
    # Iniciar operação principal
    main_op_id = logger.start_operation(
        operation="Enhanced Insights-AI Execution",
        agent="Main_System",
        expected_duration=600.0,  # 10 minutos
        total_steps=3  # Setup, Execute, Cleanup
    )
    
    try:
        # Passo 1: Setup do sistema enhanced
        logger.update_progress(step=1, message="Configurando sistema enhanced")
        
        enhanced_crew_function = enhance_existing_crew_system()
        
        if enhanced_crew_function is None:
            logger.warning("⚠️ Enhanced crew não disponível - usando fallback")
            # Importar função original
            from insights.crew_optimized import run_optimized_crew
            enhanced_crew_function = run_optimized_crew
        
        # Passo 2: Executar análise
        logger.update_progress(step=2, message="Executando análise principal")
        
        result = enhanced_crew_function(data_inicio, data_fim)
        
        # Passo 3: Finalização
        logger.update_progress(step=3, message="Finalizando análise")
        
        # Calcular tempo total
        total_time = time.time() - execution_start
        
        # Log resultado final
        logger.log_milestone("EXECUÇÃO CONCLUÍDA", {
            "Tempo_total": f"{total_time:.2f}s",
            "Resultado_chars": len(str(result)) if result else 0,
            "Status": "Sucesso"
        })
        
        # Salvar resultado se necessário
        if hasattr(result, 'raw') and len(str(result.raw)) > 100:
            save_file = f"output/insights_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            Path("output").mkdir(exist_ok=True)
            
            with open(save_file, 'w', encoding='utf-8') as f:
                f.write(str(result.raw))
            
            logger.info(f"💾 Resultado salvo em: {save_file}")
        
        # Métricas finais
        show_performance_metrics()
        
        logger.finish_operation(success=True, message=f"Tempo total: {total_time:.2f}s")
        return result
        
    except KeyboardInterrupt:
        execution_time = time.time() - execution_start
        logger.warning(f"⚠️ Execução interrompida pelo usuário após {execution_time:.2f}s")
        logger.finish_operation(success=False, message="Interrompido pelo usuário")
        return None
        
    except Exception as e:
        execution_time = time.time() - execution_start
        logger.error_with_context(e, "main_execution")
        logger.finish_operation(success=False, message=f"Erro após {execution_time:.2f}s")
        raise

# =============== INTERFACE DE LINHA DE COMANDO ===============

def create_parser():
    """Criar parser de argumentos de linha de comando"""
    
    parser = argparse.ArgumentParser(
        description="Insights-AI Enhanced - Análise com logging melhorado",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Melhorias implementadas:
  • Context-aware error handling
  • Progress indicators inteligentes  
  • Retry logic para erros conhecidos
  • Logging estruturado e eficiente

Exemplos de uso:
  %(prog)s                           # Últimos 90 dias (otimizado)
  %(prog)s --days 60                 # Últimos 60 dias  
  %(prog)s --start 2024-01-01 --end 2024-12-31
  %(prog)s --debug                   # Modo debug com stack traces
  %(prog)s --minimal                 # Logs mínimos para produção
        """
    )
    
    # Grupo de datas
    date_group = parser.add_argument_group('Período de análise')
    date_group.add_argument(
        '--start', 
        type=str, 
        help='Data início (YYYY-MM-DD)'
    )
    date_group.add_argument(
        '--end', 
        type=str, 
        help='Data fim (YYYY-MM-DD)'
    )
    date_group.add_argument(
        '--days', 
        type=int, 
        help='Últimos N dias (padrão: 90)'
    )
    
    # Grupo de logging
    log_group = parser.add_argument_group('Configuração de logs')
    log_group.add_argument(
        '--debug', 
        action='store_true', 
        help='Modo debug (logs detalhados + stack traces)'
    )
    log_group.add_argument(
        '--verbose', 
        action='store_true', 
        help='Modo verbose (mais logs)'
    )
    log_group.add_argument(
        '--minimal', 
        action='store_true', 
        help='Modo minimal (poucos logs)'
    )
    
    # Grupo de performance
    perf_group = parser.add_argument_group('Configuração de performance')
    perf_group.add_argument(
        '--production', 
        action='store_true', 
        help='Modo produção'
    )
    perf_group.add_argument(
        '--no-cache', 
        action='store_true', 
        help='Desabilitar cache'
    )
    
    # Opções específicas do enhanced
    enhanced_group = parser.add_argument_group('Opções Enhanced')
    enhanced_group.add_argument(
        '--reset-logger', 
        action='store_true', 
        help='Reset do sistema de logging'
    )
    
    # Opções gerais
    parser.add_argument(
        '--version', 
        action='version', 
        version='Insights-AI Enhanced v1.0 (Fase 1)'
    )
    
    return parser

# =============== FUNÇÃO PRINCIPAL ===============

def main():
    """Função principal"""
    
    # Reset logger se necessário
    reset_enhanced_logger()
    
    # Parser de argumentos
    parser = create_parser()
    args = parser.parse_args()
    
    # Reset adicional se solicitado
    if args.reset_logger:
        reset_enhanced_logger()
        print("🔄 Sistema de logging resetado")
    
    # Validar argumentos
    if (args.start and not args.end) or (args.end and not args.start):
        parser.error("--start e --end devem ser usados juntos")
    
    if args.debug and args.minimal:
        parser.error("--debug e --minimal são mutuamente exclusivos")
    
    # Configurar ambiente
    setup_environment(args)
    
    # Calcular datas
    data_inicio, data_fim = calculate_dates(args)
    
    # Validar datas
    if not validate_dates(data_inicio, data_fim):
        sys.exit(1)
    
    # Exibir informações
    show_execution_info(data_inicio, data_fim, args)
    
    try:
        # Executar análise enhanced
        result = run_insights_enhanced(data_inicio, data_fim)
        
        if result:
            print(f"\n🎉 Insights-AI Enhanced executado com sucesso!")
            print(f"🕒 Finalizado: {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"\n💥 Falha na execução: {e}")
        # Em modo debug, mostrar stack trace adicional
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 