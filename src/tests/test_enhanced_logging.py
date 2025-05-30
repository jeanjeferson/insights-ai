#!/usr/bin/env python
"""
🧪 TESTE DO ENHANCED LOGGING SYSTEM - FASE 1
============================================

Script para testar rapidamente o sistema de enhanced logging
sem executar o crew completo.

Testa:
- Context-aware error handling
- Progress indicators
- Retry logic
- Milestone logging
"""

import sys
import time
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_logging():
    """Teste básico do enhanced logging"""
    
    print("🧪 Testando Enhanced Logging System...")
    print("=" * 50)
    
    try:
        from old.enhanced_logging import get_enhanced_logger, reset_enhanced_logger
        
        # Reset para teste limpo
        reset_enhanced_logger()
        
        # Obter logger
        logger = get_enhanced_logger("test_basic")
        
        # Teste de operação simples
        op_id = logger.start_operation(
            operation="Teste Básico",
            agent="Test_Agent",
            expected_duration=3.0,
            total_steps=3
        )
        
        # Simular progresso
        logger.update_progress(step=1, message="Inicializando")
        time.sleep(0.5)
        
        logger.update_progress(step=2, message="Processando")
        time.sleep(0.5)
        
        logger.update_progress(step=3, message="Finalizando")
        time.sleep(0.5)
        
        # Finalizar operação
        logger.finish_operation(success=True, message="Teste concluído")
        
        # Log milestone
        logger.log_milestone("TESTE BÁSICO CONCLUÍDO", {
            "Operações": 1,
            "Status": "Sucesso",
            "Tempo": "~1.5s"
        })
        
        print("✅ Teste básico: PASSOU")
        return True
        
    except Exception as e:
        print(f"❌ Teste básico: FALHOU - {e}")
        return False

def test_error_handling():
    """Teste do context-aware error handling"""
    
    print("\n🧪 Testando Error Handling...")
    print("-" * 30)
    
    try:
        from old.enhanced_logging import get_enhanced_logger
        
        logger = get_enhanced_logger("test_error")
        
        # Simular operação com erro
        logger.start_operation(
            operation="Teste de Erro",
            agent="Error_Agent",
            expected_duration=1.0
        )
        
        try:
            # Simular erro conhecido
            raise Exception("create_reasoning_plan missing 1 required positional argument: 'ready'")
            
        except Exception as e:
            logger.error_with_context(e, "test_error_operation")
            logger.finish_operation(success=False)
        
        print("✅ Teste error handling: PASSOU")
        return True
        
    except Exception as e:
        print(f"❌ Teste error handling: FALHOU - {e}")
        return False

def test_crew_wrapper():
    """Teste do crew wrapper"""
    
    print("\n🧪 Testando Crew Wrapper...")
    print("-" * 30)
    
    try:
        from old.enhanced_crew_wrapper import EnhancedCrewWrapper
        
        # Função mock para testar
        def mock_crew_function(data_inicio, data_fim):
            time.sleep(1)  # Simular processamento
            return f"Resultado mock para {data_inicio} - {data_fim}"
        
        # Wrapper
        enhanced_crew = EnhancedCrewWrapper(mock_crew_function)
        
        # Executar
        result = enhanced_crew("2024-01-01", "2024-01-31")
        
        if result and "mock" in result:
            print("✅ Teste crew wrapper: PASSOU")
            return True
        else:
            print("❌ Teste crew wrapper: FALHOU - resultado inválido")
            return False
        
    except Exception as e:
        print(f"❌ Teste crew wrapper: FALHOU - {e}")
        return False

def test_reasoning_fix():
    """Teste do reasoning fix patch"""
    
    print("\n🧪 Testando Reasoning Fix...")
    print("-" * 30)
    
    try:
        from insights.tools.reasoning_fix_patch import apply_reasoning_fix, monitor_reasoning_health
        
        # Aplicar fix
        fix_applied = apply_reasoning_fix()
        
        # Monitorar saúde
        health_stats = monitor_reasoning_health()
        
        if fix_applied and health_stats is not None:
            print("✅ Teste reasoning fix: PASSOU")
            return True
        else:
            print("❌ Teste reasoning fix: FALHOU")
            return False
            
    except Exception as e:
        print(f"❌ Teste reasoning fix: FALHOU - {e}")
        return False

def test_performance():
    """Teste de performance do enhanced logging"""
    
    print("\n🧪 Testando Performance...")
    print("-" * 30)
    
    try:
        from old.enhanced_logging import get_enhanced_logger
        
        logger = get_enhanced_logger("test_performance")
        
        # Teste de múltiplas operações rápidas
        start_time = time.time()
        
        for i in range(10):
            op_id = logger.start_operation(
                operation=f"Operação {i+1}",
                agent="Performance_Agent",
                total_steps=2
            )
            
            logger.update_progress(step=1, message=f"Processando {i+1}")
            logger.update_progress(step=2, message=f"Concluindo {i+1}")
            logger.finish_operation(success=True)
        
        elapsed = time.time() - start_time
        
        # Verificar se foi rápido (< 1s para 10 operações)
        if elapsed < 1.0:
            print(f"✅ Teste performance: PASSOU ({elapsed:.3f}s para 10 operações)")
            return True
        else:
            print(f"❌ Teste performance: LENTO ({elapsed:.3f}s para 10 operações)")
            return False
            
    except Exception as e:
        print(f"❌ Teste performance: FALHOU - {e}")
        return False

def main():
    """Executar todos os testes"""
    
    print("🚀 ENHANCED LOGGING SYSTEM - TESTES FASE 1")
    print("=" * 55)
    
    tests = [
        ("Logging Básico", test_basic_logging),
        ("Error Handling", test_error_handling),
        ("Crew Wrapper", test_crew_wrapper),
        ("Reasoning Fix", test_reasoning_fix),
        ("Performance", test_performance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: ERRO CRÍTICO - {e}")
            results.append((test_name, False))
    
    # Sumário final
    print("\n" + "=" * 55)
    print("📊 SUMÁRIO DOS TESTES:")
    print("-" * 25)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print("-" * 25)
    print(f"RESULTADO FINAL: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("✅ Sistema Enhanced Logging está funcionando corretamente")
    else:
        print(f"⚠️ {total - passed} teste(s) falharam")
        print("🔧 Verifique os logs acima para detalhes")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 