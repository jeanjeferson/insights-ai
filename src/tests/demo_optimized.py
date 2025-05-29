#!/usr/bin/env python
"""
🎬 DEMO OTIMIZADO - INSIGHTS-AI
==============================

Script de demonstração das otimizações implementadas:
- Comparação de verbosidade entre modos
- Demonstração de lazy loading
- Métricas de performance
- Configurações automáticas por ambiente

Uso:
    python demo_optimized.py
"""

import os
import sys
import time
from pathlib import Path

# Adicionar src ao Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def demo_logging_levels():
    """Demonstrar diferentes níveis de logging"""
    
    print("🎭 DEMONSTRAÇÃO DOS NÍVEIS DE LOGGING")
    print("="*50)
    
    from insights.config.performance_config import LogLevel, OptimizedLogger, PerformanceSettings
    
    levels = [
        (LogLevel.SILENT, "🔇 SILENT - Apenas erros críticos"),
        (LogLevel.MINIMAL, "📝 MINIMAL - Resumos essenciais"),
        (LogLevel.NORMAL, "📄 NORMAL - Logs importantes"),
        (LogLevel.VERBOSE, "📢 VERBOSE - Todos os logs"),
        (LogLevel.DEBUG, "🐛 DEBUG - Logs de debug")
    ]
    
    for level, description in levels:
        print(f"\n{description}")
        print("-" * 30)
        
        # Criar configuração temporária
        config = PerformanceSettings(log_level=level, enable_file_logging=False)
        logger = OptimizedLogger('demo', config)
        
        # Simular logs
        logger.debug("Log de debug detalhado")
        logger.info("Informação importante")
        logger.warning("Aviso do sistema")
        logger.error("Erro simulado")
        
        # Forçar flush para demo
        logger._flush_buffer()
        
        time.sleep(0.5)

def demo_environment_detection():
    """Demonstrar detecção automática de ambiente"""
    
    print("\n🌍 DEMONSTRAÇÃO DE DETECÇÃO DE AMBIENTE")
    print("="*50)
    
    from insights.config.performance_config import get_performance_config
    
    # Ambientes de teste
    environments = [
        ("🏭 PRODUÇÃO", {"ENVIRONMENT": "production"}),
        ("🧪 TESTES", {"PYTEST_CURRENT_TEST": "test_demo"}),
        ("🐛 DEBUG", {"INSIGHTS_DEBUG": "true"}),
        ("💻 DESENVOLVIMENTO", {})
    ]
    
    for env_name, env_vars in environments:
        print(f"\n{env_name}")
        print("-" * 20)
        
        # Limpar ambiente
        for key in ["ENVIRONMENT", "PYTEST_CURRENT_TEST", "INSIGHTS_DEBUG"]:
            os.environ.pop(key, None)
        
        # Definir ambiente de teste
        for key, value in env_vars.items():
            os.environ[key] = value
        
        # Obter configuração
        config = get_performance_config()
        
        print(f"   • Log Level: {config.log_level.name}")
        print(f"   • Lazy Loading: {'✅' if config.lazy_tool_loading else '❌'}")
        print(f"   • Cache: {'✅' if config.enable_tool_cache else '❌'}")
        print(f"   • File Logging: {'✅' if config.enable_file_logging else '❌'}")
        print(f"   • Flush Frequency: {config.log_flush_frequency}")
        print(f"   • Max Agents: {config.max_concurrent_agents}")
    
    # Limpar ambiente
    for key in ["ENVIRONMENT", "PYTEST_CURRENT_TEST", "INSIGHTS_DEBUG"]:
        os.environ.pop(key, None)

def demo_lazy_loading():
    """Demonstrar lazy loading de ferramentas"""
    
    print("\n🧰 DEMONSTRAÇÃO DE LAZY LOADING")
    print("="*40)
    
    from insights.config.performance_config import get_optimized_tool_list
    
    # Simular lista de ferramentas
    all_tools = [f"Tool_{i}" for i in range(1, 18)]  # 17 ferramentas
    
    agents = [
        ('engenheiro_dados', '🔧 Engenheiro de Dados'),
        ('analista_financeiro', '💰 Analista Financeiro'),
        ('especialista_clientes', '👥 Especialista em Clientes'),
        ('analista_vendas_tendencias', '📈 Analista de Vendas'),
        ('diretor_insights', '🎯 Diretor de Insights')
    ]
    
    print(f"📦 Total de ferramentas disponíveis: {len(all_tools)}")
    print("\n🎯 Ferramentas otimizadas por agente:")
    
    for agent_role, agent_name in agents:
        optimized_tools = get_optimized_tool_list(agent_role, all_tools)
        reduction = ((len(all_tools) - len(optimized_tools)) / len(all_tools)) * 100
        
        print(f"   • {agent_name}: {len(optimized_tools)}/{len(all_tools)} "
              f"(-{reduction:.0f}%)")

def demo_performance_tracking():
    """Demonstrar tracking de performance"""
    
    print("\n⚡ DEMONSTRAÇÃO DE TRACKING DE PERFORMANCE")
    print("="*50)
    
    from insights.config.performance_config import performance_tracked, cached_result
    
    @performance_tracked("demo_operation")
    def simulated_operation():
        """Operação simulada para demo"""
        time.sleep(0.1)  # Simular processamento
        return "Resultado da operação"
    
    @cached_result()
    def cached_operation(param: str):
        """Operação com cache para demo"""
        time.sleep(0.2)  # Simular operação custosa
        return f"Resultado cached para {param}"
    
    print("🔄 Executando operação com tracking...")
    result1 = simulated_operation()
    
    print("\n💾 Executando operação com cache...")
    start_time = time.time()
    result2 = cached_operation("test_param")
    first_call_time = time.time() - start_time
    
    print(f"   Primeira chamada: {first_call_time:.3f}s")
    
    start_time = time.time()
    result3 = cached_operation("test_param")  # Deve vir do cache
    cached_call_time = time.time() - start_time
    
    print(f"   Segunda chamada (cache): {cached_call_time:.3f}s")
    print(f"   Melhoria: {((first_call_time - cached_call_time) / first_call_time * 100):.1f}%")

def demo_metrics():
    """Demonstrar métricas de performance"""
    
    print("\n📊 DEMONSTRAÇÃO DE MÉTRICAS")
    print("="*35)
    
    try:
        from insights.config.performance_config import PERFORMANCE_CONFIG, performance_cache
        
        print("⚙️ Configuração atual:")
        print(f"   • Log Level: {PERFORMANCE_CONFIG.log_level.name}")
        print(f"   • Lazy Loading: {'✅' if PERFORMANCE_CONFIG.lazy_tool_loading else '❌'}")
        print(f"   • Cache habilitado: {'✅' if PERFORMANCE_CONFIG.enable_tool_cache else '❌'}")
        print(f"   • Cache timeout: {PERFORMANCE_CONFIG.cache_timeout_seconds}s")
        print(f"   • Flush frequency: {PERFORMANCE_CONFIG.log_flush_frequency}")
        
        print(f"\n📈 Estado do cache:")
        print(f"   • Entradas: {len(performance_cache.cache)}")
        print(f"   • Max size: {PERFORMANCE_CONFIG.cache_max_size}")
        
    except Exception as e:
        print(f"⚠️ Erro ao obter métricas: {e}")

def main():
    """Função principal do demo"""
    
    print("🎬 DEMO INSIGHTS-AI OTIMIZADO")
    print("=" * 60)
    print("Demonstração das otimizações de performance implementadas")
    print("=" * 60)
    
    try:
        # 1. Demonstrar níveis de logging
        demo_logging_levels()
        
        # 2. Demonstrar detecção de ambiente
        demo_environment_detection()
        
        # 3. Demonstrar lazy loading
        demo_lazy_loading()
        
        # 4. Demonstrar tracking de performance
        demo_performance_tracking()
        
        # 5. Demonstrar métricas
        demo_metrics()
        
        print("\n🎉 DEMO CONCLUÍDO!")
        print("=" * 20)
        print("✅ Todas as otimizações demonstradas com sucesso!")
        print("\n💡 Para usar na prática:")
        print("   python main_optimized.py --help")
        
    except Exception as e:
        print(f"\n❌ Erro no demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 