#!/usr/bin/env python
"""
⚡ MAIN OTIMIZADO - INSIGHTS-AI (UNIFICADO)
==========================================

Script principal unificado para executar o Insights-AI com otimizações de performance
e funcionalidades selecionadas da versão enhanced v2:

FUNCIONALIDADES PRINCIPAIS:
- Logging estruturado menos verbose
- Lazy loading de ferramentas
- Cache inteligente de validações
- Métricas de performance em tempo real
- Configuração automática por ambiente

FUNCIONALIDADES ADICIONAIS (do v2):
- Log file management básico
- Relatórios de ferramentas opcionais
- Validação de arquivos gerados
- Consolidação básica de logs repetitivos
- Período padrão otimizado (90 dias)

Uso:
    python main_optimized.py                    # Últimos 90 dias
    python main_optimized.py --days 60          # Últimos 60 dias
    python main_optimized.py --start 2024-01-01 --end 2024-12-31
    python main_optimized.py --debug            # Modo debug
    python main_optimized.py --minimal          # Logs mínimos
    python main_optimized.py --tools-report     # Relatório de ferramentas
    python main_optimized.py --validate-files   # Validar arquivos gerados
"""

import os
import sys
import argparse
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

# Adicionar src ao Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# =============== SISTEMA DE LOG CONSOLIDATION BÁSICO ===============

class BasicLogConsolidator:
    """Consolidador básico de logs repetitivos (versão simplificada do v2)"""
    
    def __init__(self):
        self.message_counts = defaultdict(int)
        self.last_log_time = {}
        self.consolidation_threshold = 5  # Consolidar após 5 repetições
        self.time_window = 30  # 30 segundos
    
    def should_log(self, message: str) -> tuple[bool, str]:
        """Determinar se deve fazer log ou consolidar"""
        current_time = time.time()
        
        # Normalizar mensagem para padrão
        pattern = self._normalize_message(message)
        
        self.message_counts[pattern] += 1
        count = self.message_counts[pattern]
        
        # Se é primeira vez ou passou do threshold
        if count == 1:
            self.last_log_time[pattern] = current_time
            return True, message
        elif count == self.consolidation_threshold:
            return True, f"[CONSOLIDADO] {message} (repetido {count}x)"
        elif count > self.consolidation_threshold:
            # Log apenas a cada 10 repetições após o threshold
            if count % 10 == 0:
                return True, f"[CONSOLIDADO] {message} (repetido {count}x)"
            return False, ""
        else:
            return True, message
    
    def _normalize_message(self, message: str) -> str:
        """Normalizar mensagem removendo partes dinâmicas"""
        import re
        # Remover números, timestamps, IDs dinâmicos
        normalized = re.sub(r'\d+', 'N', message)
        normalized = re.sub(r'\d{2}:\d{2}:\d{2}', 'HH:MM:SS', normalized)
        return normalized
    
    def get_stats(self) -> dict:
        """Obter estatísticas de consolidação"""
        total_messages = sum(self.message_counts.values())
        unique_patterns = len(self.message_counts)
        suppressed = total_messages - unique_patterns
        return {
            'total_messages': total_messages,
            'unique_patterns': unique_patterns,
            'suppressed_messages': suppressed
        }

# =============== ENHANCED LOGGER SIMPLES ===============

class SimpleEnhancedLogger:
    """Logger simples com funcionalidades básicas do enhanced v2"""
    
    def __init__(self):
        self.consolidator = BasicLogConsolidator()
        self.log_file_path = None
        self.enable_consolidation = True
        self.performance_metrics = {
            'operations': {},
            'start_time': time.time()
        }
        self._setup_file_logging()
    
    def _setup_file_logging(self):
        """Configurar logging em arquivo (versão simplificada)"""
        try:
            log_dir = Path("logs/optimized")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d")
            self.log_file_path = log_dir / f"insights_optimized_{timestamp}.log"
            
            # Configurar handler de arquivo
            file_handler = logging.FileHandler(self.log_file_path, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            # Adicionar ao logger principal
            logger = logging.getLogger()
            logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)
            
        except Exception as e:
            print(f"⚠️ Não foi possível configurar log em arquivo: {e}")
    
    def _log_with_consolidation(self, level: str, message: str):
        """Log com consolidação básica"""
        if self.enable_consolidation:
            should_log, final_message = self.consolidator.should_log(message)
            if should_log and final_message:
                if level.upper() == 'INFO':
                    print(final_message)
                elif level.upper() == 'WARNING':
                    print(f"⚠️ {final_message}")
                elif level.upper() == 'ERROR':
                    print(f"❌ {final_message}")
                elif level.upper() == 'DEBUG':
                    if os.getenv('INSIGHTS_DEBUG', 'false').lower() == 'true':
                        print(f"🐛 {final_message}")
                
                # Log em arquivo sempre
                if self.log_file_path:
                    logging.getLogger().log(getattr(logging, level.upper()), final_message)
        else:
            # Log direto sem consolidação
            print(message)
            if self.log_file_path:
                logging.getLogger().log(getattr(logging, level.upper()), message)
    
    def info(self, message: str):
        self._log_with_consolidation('INFO', message)
    
    def warning(self, message: str):
        self._log_with_consolidation('WARNING', message)
    
    def error(self, message: str):
        self._log_with_consolidation('ERROR', message)
    
    def debug(self, message: str):
        self._log_with_consolidation('DEBUG', message)
    
    def log_milestone(self, title: str, data: dict):
        """Log milestone simplificado"""
        self.info(f"🎯 {title}")
        for key, value in data.items():
            self.info(f"   📊 {key}: {value}")
    
    def track_operation(self, operation: str, duration: float):
        """Track operation performance"""
        if operation not in self.performance_metrics['operations']:
            self.performance_metrics['operations'][operation] = []
        self.performance_metrics['operations'][operation].append(duration)
    
    def get_performance_summary(self) -> dict:
        """Obter resumo de performance"""
        total_time = time.time() - self.performance_metrics['start_time']
        consolidation_stats = self.consolidator.get_stats()
        
        return {
            'total_execution_time': total_time,
            'operations_tracked': len(self.performance_metrics['operations']),
            'log_file': str(self.log_file_path) if self.log_file_path else None,
            'consolidation': consolidation_stats
        }

# Instância global do logger
enhanced_logger = SimpleEnhancedLogger()

# =============== CONFIGURAÇÃO DE ENVIRONMENT ===============

def setup_environment(args):
    """Configurar ambiente baseado nos argumentos"""
    
    # Configurar nível de logging
    if args.debug:
        os.environ['INSIGHTS_DEBUG'] = 'true'
        enhanced_logger.info("🐛 Modo DEBUG ativado")
    elif args.minimal:
        os.environ['INSIGHTS_LOG_LEVEL'] = 'MINIMAL'
        enhanced_logger.info("🔇 Modo MINIMAL ativado")
    elif args.verbose:
        os.environ['INSIGHTS_LOG_LEVEL'] = 'VERBOSE'
        enhanced_logger.info("📢 Modo VERBOSE ativado")
    
    # Configurar environment
    if args.production:
        os.environ['ENVIRONMENT'] = 'production'
        enhanced_logger.info("🏭 Modo PRODUÇÃO ativado")
    
    # Configurar cache
    if args.no_cache:
        os.environ['INSIGHTS_DISABLE_CACHE'] = 'true'
        enhanced_logger.info("🚫 Cache DESABILITADO")
    
    # Configurações do v2
    if args.no_consolidation:
        enhanced_logger.enable_consolidation = False
        enhanced_logger.info("📝 Consolidação de logs DESABILITADA")
    
    if args.tools_report:
        enhanced_logger.info("🔧 Relatório de ferramentas ATIVADO")

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
        # Padrão: 90 dias (otimizado do v2)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

# =============== EXIBIÇÃO DE INFORMAÇÕES ===============

def show_execution_info(data_inicio: str, data_fim: str, args):
    """Exibir informações da execução"""
    
    enhanced_logger.info("="*60)
    enhanced_logger.info("⚡ INSIGHTS-AI OTIMIZADO (UNIFICADO)")
    enhanced_logger.info("="*60)
    
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
    
    # Features do v2
    v2_features = []
    if not args.no_consolidation:
        v2_features.append("CONSOLIDAÇÃO")
    if args.tools_report:
        v2_features.append("TOOLS_REPORT")
    if args.validate_files:
        v2_features.append("VALIDAÇÃO_ARQUIVOS")
    
    # Log milestone
    enhanced_logger.log_milestone("CONFIGURAÇÃO DO SISTEMA", {
        "Período": f"{data_inicio} até {data_fim} ({days} dias)",
        "Início": datetime.now().strftime('%H:%M:%S'),
        "Configuração": ' | '.join(config_info),
        "Features_v2": ' + '.join(v2_features) if v2_features else "Padrão",
        "Log_File": str(enhanced_logger.log_file_path) if enhanced_logger.log_file_path else "N/A"
    })

def show_performance_metrics():
    """Exibir métricas de performance"""
    
    try:
        from insights.crew_optimized import get_performance_metrics
        
        metrics = get_performance_metrics()
        
        enhanced_logger.info("📊 MÉTRICAS DE PERFORMANCE:")
        enhanced_logger.info("-" * 30)
        enhanced_logger.info(f"   • Nível de log:    {metrics.get('log_level', 'N/A')}")
        enhanced_logger.info(f"   • Cache:           {'✅' if metrics.get('cache_enabled') else '❌'}")
        enhanced_logger.info(f"   • Lazy loading:    {'✅' if metrics.get('lazy_loading') else '❌'}")
        enhanced_logger.info(f"   • Cache entries:   {metrics.get('cache_size', 0)}")
        
        if metrics.get('cache_hits', 0) > 0 or metrics.get('cache_misses', 0) > 0:
            total_requests = metrics.get('cache_hits', 0) + metrics.get('cache_misses', 0)
            hit_rate = (metrics.get('cache_hits', 0) / total_requests) * 100 if total_requests > 0 else 0
            enhanced_logger.info(f"   • Cache hit rate:  {hit_rate:.1f}%")
        
        # Adicionar métricas do enhanced logger
        enhanced_metrics = enhanced_logger.get_performance_summary()
        enhanced_logger.info(f"   • Ops rastreadas:  {enhanced_metrics['operations_tracked']}")
        enhanced_logger.info(f"   • Logs suprimidos: {enhanced_metrics['consolidation']['suppressed_messages']}")
            
    except Exception as e:
        enhanced_logger.warning(f"Erro ao obter métricas: {e}")

# =============== FUNCIONALIDADES DO V2 SIMPLIFICADAS ===============

def run_tools_report():
    """Executar relatório de ferramentas (simplificado do v2)"""
    try:
        enhanced_logger.info("🔧 Gerando relatório de ferramentas...")
        
        # Relatório básico das ferramentas do sistema (sem dependência do enhanced_tool_integration)
        try:
            from insights.config.tools_config_v3 import get_tools_statistics
            stats = get_tools_statistics()
            
            enhanced_logger.log_milestone("RELATÓRIO DE FERRAMENTAS", {
                "Total_ferramentas": stats.get('total_tools', 0),
                "Categorias": stats.get('categories', 0),
                "Integrações": stats.get('integrations', 0)
            })
        except ImportError:
            enhanced_logger.warning("Sistema de estatísticas de ferramentas não disponível")
        
        # Relatório básico adicional
        enhanced_logger.info("📊 Ferramentas do sistema:")
        enhanced_logger.info("   • SQL Query Tool: Extração de dados")
        enhanced_logger.info("   • File Read Tool: Leitura de arquivos")
        enhanced_logger.info("   • Statistical Analysis Tool: Análises estatísticas")
        enhanced_logger.info("   • Business Intelligence Tool: Dashboards")
        enhanced_logger.info("   • File Generation Tool: Geração de relatórios")
            
    except Exception as e:
        enhanced_logger.error(f"Erro no relatório de ferramentas: {e}")

def validate_generated_files(data_inicio: str, data_fim: str):
    """Validar arquivos gerados (simplificado do v2)"""
    try:
        enhanced_logger.info("📁 Validando arquivos gerados...")
        
        # Arquivos esperados básicos
        expected_files = [
            "data/vendas.csv",
            "output/insights_*.txt"
        ]
        
        found_files = []
        missing_files = []
        
        for file_pattern in expected_files:
            if "*" in file_pattern:
                # Pattern matching básico
                base_dir = Path(file_pattern).parent
                if base_dir.exists():
                    pattern = Path(file_pattern).name.replace("*", "")
                    matching_files = list(base_dir.glob(f"*{pattern}*"))
                    if matching_files:
                        found_files.extend(matching_files)
                    else:
                        missing_files.append(file_pattern)
                else:
                    missing_files.append(file_pattern)
            else:
                file_path = Path(file_pattern)
                if file_path.exists():
                    found_files.append(file_path)
                else:
                    missing_files.append(file_pattern)
        
        # Log resultado
        enhanced_logger.log_milestone("VALIDAÇÃO DE ARQUIVOS", {
            "Arquivos_encontrados": len(found_files),
            "Arquivos_esperados": len(expected_files),
            "Taxa_completude": f"{(len(found_files)/len(expected_files)*100):.1f}%" if expected_files else "N/A"
        })
        
        if missing_files:
            enhanced_logger.warning(f"Arquivos não encontrados: {missing_files}")
        
        return len(found_files) >= len(expected_files) * 0.8  # 80% de completude
        
    except Exception as e:
        enhanced_logger.error(f"Erro na validação de arquivos: {e}")
        return False

# =============== EXECUÇÃO PRINCIPAL ===============

def run_insights_optimized(data_inicio: str, data_fim: str, args):
    """Executar o Insights-AI otimizado com funcionalidades v2"""
    
    execution_start = time.time()
    
    try:
        enhanced_logger.info("🚀 Carregando sistema otimizado...")
        
        # Importar função otimizada
        from insights.crew_optimized import run_optimized_crew
        
        # Executar análise
        enhanced_logger.info("⚡ Executando análise de insights...")
        result = run_optimized_crew(data_inicio, data_fim)
        
        # Calcular tempo total
        total_time = time.time() - execution_start
        enhanced_logger.track_operation("main_execution", total_time)
        
        # Relatório de ferramentas se solicitado
        if args.tools_report:
            run_tools_report()
        
        # Validação de arquivos se solicitada
        if args.validate_files:
            validation_ok = validate_generated_files(data_inicio, data_fim)
            if not validation_ok:
                enhanced_logger.warning("⚠️ Validação de arquivos incompleta")
        
        # Log resultado final
        enhanced_logger.log_milestone("EXECUÇÃO CONCLUÍDA", {
            "Tempo_total": f"{total_time:.2f}s",
            "Resultado_chars": len(str(result)) if result else 0,
            "Status": "Sucesso"
        })
        
        # Métricas de performance
        show_performance_metrics()
        
        # Salvar resultado se necessário
        if hasattr(result, 'raw') and len(str(result.raw)) > 100:
            save_file = f"output/insights_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            Path("output").mkdir(exist_ok=True)
            
            with open(save_file, 'w', encoding='utf-8') as f:
                f.write(str(result.raw))
            
            enhanced_logger.info(f"💾 Resultado salvo em: {save_file}")
        
        return result
        
    except KeyboardInterrupt:
        execution_time = time.time() - execution_start
        enhanced_logger.warning(f"Execução interrompida pelo usuário após {execution_time:.2f}s")
        return None
        
    except Exception as e:
        execution_time = time.time() - execution_start
        enhanced_logger.error(f"Erro após {execution_time:.2f}s: {e}")
        
        # Exibir stack trace em modo debug
        if os.getenv('INSIGHTS_DEBUG', 'false').lower() == 'true':
            import traceback
            traceback.print_exc()
        
        raise

# =============== INTERFACE DE LINHA DE COMANDO ===============

def create_parser():
    """Criar parser de argumentos de linha de comando"""
    
    parser = argparse.ArgumentParser(
        description="Insights-AI Otimizado (Unificado) - Análise de dados com performance avançada",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Funcionalidades integradas das versões optimized + enhanced v2:
  • Sistema de logging otimizado com consolidação básica
  • Cache inteligente e lazy loading de ferramentas  
  • Log file management automático
  • Relatórios de performance e ferramentas
  • Validação de arquivos gerados
  • Período padrão otimizado (90 dias)

Exemplos de uso:
  %(prog)s                           # Últimos 90 dias
  %(prog)s --days 60                 # Últimos 60 dias  
  %(prog)s --start 2024-01-01 --end 2024-12-31
  %(prog)s --debug                   # Modo debug
  %(prog)s --minimal                 # Logs mínimos
  %(prog)s --production              # Modo produção
  %(prog)s --tools-report            # Relatório de ferramentas
  %(prog)s --validate-files          # Validar arquivos gerados
  %(prog)s --no-consolidation        # Desabilitar consolidação
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
    log_group.add_argument(
        '--no-consolidation', 
        action='store_true', 
        help='Desabilitar consolidação de logs'
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
    
    # Funcionalidades do v2
    v2_group = parser.add_argument_group('Funcionalidades avançadas (do v2)')
    v2_group.add_argument(
        '--tools-report', 
        action='store_true', 
        help='Gerar relatório detalhado de ferramentas'
    )
    v2_group.add_argument(
        '--validate-files', 
        action='store_true', 
        help='Validar se arquivos obrigatórios foram gerados'
    )
    
    # Opções gerais
    parser.add_argument(
        '--version', 
        action='version', 
        version='Insights-AI Otimizado v2.0 (Unificado)'
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
        result = run_insights_optimized(data_inicio, data_fim, args)
        
        if result:
            enhanced_logger.info("🎉 Insights-AI executado com sucesso!")
            enhanced_logger.info(f"🕒 Finalizado: {datetime.now().strftime('%H:%M:%S')}")
            
            # Estatísticas finais
            performance_summary = enhanced_logger.get_performance_summary()
            if performance_summary['consolidation']['suppressed_messages'] > 0:
                enhanced_logger.info(f"📦 Logs consolidados: {performance_summary['consolidation']['suppressed_messages']} mensagens suprimidas")
            if performance_summary['log_file']:
                enhanced_logger.info(f"📁 Log salvo em: {performance_summary['log_file']}")
        
    except Exception as e:
        enhanced_logger.error(f"Falha na execução: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 