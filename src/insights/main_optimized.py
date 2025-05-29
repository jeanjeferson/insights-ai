#!/usr/bin/env python
"""
⚡ MAIN OTIMIZADO - INSIGHTS-AI
=============================

Script principal para executar o Insights-AI com otimizações de performance:
- Logging estruturado menos verbose
- Lazy loading de ferramentas
- Cache inteligente de validações
- Métricas de performance em tempo real
- Configuração automática por ambiente

Uso:
    python main_optimized.py                    # Último mês
    python main_optimized.py --days 60          # Últimos 60 dias
    python main_optimized.py --start 2024-01-01 --end 2024-12-31
    python main_optimized.py --debug            # Modo debug
    python main_optimized.py --minimal          # Logs mínimos
"""

import os
import sys
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

# Adicionar src ao Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# =============== CONFIGURAÇÃO DE ENVIRONMENT ===============

def setup_environment(args):
    """Configurar ambiente baseado nos argumentos"""
    
    # Configurar nível de logging
    if args.debug:
        os.environ['INSIGHTS_DEBUG'] = 'true'
        print("🐛 Modo DEBUG ativado")
    elif args.minimal:
        os.environ['INSIGHTS_LOG_LEVEL'] = 'MINIMAL'
        print("🔇 Modo MINIMAL ativado")
    elif args.verbose:
        os.environ['INSIGHTS_LOG_LEVEL'] = 'VERBOSE'
        print("📢 Modo VERBOSE ativado")
    
    # Configurar environment
    if args.production:
        os.environ['ENVIRONMENT'] = 'production'
        print("🏭 Modo PRODUÇÃO ativado")
    
    # Configurar cache
    if args.no_cache:
        os.environ['INSIGHTS_DISABLE_CACHE'] = 'true'
        print("🚫 Cache DESABILITADO")

# =============== VALIDAÇÃO DE DATAS ===============

def validate_dates(data_inicio: str, data_fim: str):
    """Validar formato e lógica das datas"""
    
    try:
        start_date = datetime.strptime(data_inicio, '%Y-%m-%d')
        end_date = datetime.strptime(data_fim, '%Y-%m-%d')
        
        if start_date > end_date:
            raise ValueError("Data início não pode ser posterior à data fim")
        
        if end_date > datetime.now():
            print("⚠️ Data fim está no futuro - usando dados disponíveis")
        
        days_diff = (end_date - start_date).days
        if days_diff > 365:
            print(f"⚠️ Período muito longo ({days_diff} dias) - pode impactar performance")
        
        return True
        
    except ValueError as e:
        print(f"❌ Erro na data: {e}")
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
        # Padrão: último mês
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1430)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

# =============== EXIBIÇÃO DE INFORMAÇÕES ===============

def show_execution_info(data_inicio: str, data_fim: str, args):
    """Exibir informações da execução"""
    
    print("\n" + "="*60)
    print("⚡ INSIGHTS-AI OTIMIZADO")
    print("="*60)
    
    # Período
    start_date = datetime.strptime(data_inicio, '%Y-%m-%d')
    end_date = datetime.strptime(data_fim, '%Y-%m-%d')
    days = (end_date - start_date).days
    
    print(f"📅 Período:        {data_inicio} até {data_fim} ({days} dias)")
    print(f"🕒 Início:         {datetime.now().strftime('%H:%M:%S')}")
    
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
    
    print(f"⚙️ Configuração:   {' | '.join(config_info)}")
    print("-" * 60)

def show_performance_metrics():
    """Exibir métricas de performance"""
    
    try:
        from insights.crew_optimized import get_performance_metrics
        
        metrics = get_performance_metrics()
        
        print(f"\n📊 MÉTRICAS DE PERFORMANCE:")
        print("-" * 30)
        print(f"   • Nível de log:    {metrics.get('log_level', 'N/A')}")
        print(f"   • Cache:           {'✅' if metrics.get('cache_enabled') else '❌'}")
        print(f"   • Lazy loading:    {'✅' if metrics.get('lazy_loading') else '❌'}")
        print(f"   • Cache entries:   {metrics.get('cache_size', 0)}")
        
        if metrics.get('cache_hits', 0) > 0 or metrics.get('cache_misses', 0) > 0:
            total_requests = metrics.get('cache_hits', 0) + metrics.get('cache_misses', 0)
            hit_rate = (metrics.get('cache_hits', 0) / total_requests) * 100 if total_requests > 0 else 0
            print(f"   • Cache hit rate:  {hit_rate:.1f}%")
            
    except Exception as e:
        print(f"⚠️ Erro ao obter métricas: {e}")

# =============== EXECUÇÃO PRINCIPAL ===============

def run_insights_optimized(data_inicio: str, data_fim: str):
    """Executar o Insights-AI otimizado"""
    
    execution_start = time.time()
    
    try:
        print("🚀 Carregando sistema otimizado...")
        
        # Importar função otimizada
        from insights.crew_optimized import run_optimized_crew
        
        # Executar análise
        print("⚡ Executando análise de insights...")
        result = run_optimized_crew(data_inicio, data_fim)
        
        # Calcular tempo total
        total_time = time.time() - execution_start
        
        # Exibir resultado
        print(f"\n✅ ANÁLISE CONCLUÍDA!")
        print(f"⏱️ Tempo total: {total_time:.2f}s")
        
        # Métricas de performance
        show_performance_metrics()
        
        # Salvar resultado se solicitado
        if hasattr(result, 'raw') and len(str(result.raw)) > 100:
            print(f"📄 Resultado: {len(str(result.raw))} caracteres gerados")
            
            # Opção de salvar
            save_file = f"output/insights_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            Path("output").mkdir(exist_ok=True)
            
            with open(save_file, 'w', encoding='utf-8') as f:
                f.write(str(result.raw))
            
            print(f"💾 Resultado salvo em: {save_file}")
        
        return result
        
    except KeyboardInterrupt:
        execution_time = time.time() - execution_start
        print(f"\n⚠️ Execução interrompida pelo usuário após {execution_time:.2f}s")
        return None
        
    except Exception as e:
        execution_time = time.time() - execution_start
        print(f"\n❌ Erro após {execution_time:.2f}s: {e}")
        
        # Exibir stack trace em modo debug
        if os.getenv('INSIGHTS_DEBUG', 'false').lower() == 'true':
            import traceback
            traceback.print_exc()
        
        raise

# =============== INTERFACE DE LINHA DE COMANDO ===============

def create_parser():
    """Criar parser de argumentos de linha de comando"""
    
    parser = argparse.ArgumentParser(
        description="Insights-AI Otimizado - Análise de dados com performance avançada",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  %(prog)s                           # Último mês
  %(prog)s --days 60                 # Últimos 60 dias  
  %(prog)s --start 2024-01-01 --end 2024-12-31
  %(prog)s --debug                   # Modo debug
  %(prog)s --minimal                 # Logs mínimos
  %(prog)s --production              # Modo produção
  %(prog)s --no-cache                # Sem cache
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
        help='Últimos N dias (padrão: 30)'
    )
    
    # Grupo de logging
    log_group = parser.add_argument_group('Configuração de logs')
    log_group.add_argument(
        '--debug', 
        action='store_true', 
        help='Modo debug (logs detalhados)'
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
    
    # Opções gerais
    parser.add_argument(
        '--version', 
        action='version', 
        version='Insights-AI Otimizado v1.0'
    )
    
    return parser

# =============== FUNÇÃO PRINCIPAL ===============

def main():
    """Função principal"""
    
    # Parser de argumentos
    parser = create_parser()
    args = parser.parse_args()
    
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
        # Executar análise
        result = run_insights_optimized(data_inicio, data_fim)
        
        if result:
            print(f"\n🎉 Insights-AI executado com sucesso!")
            print(f"🕒 Finalizado: {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"\n💥 Falha na execução: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 