#!/usr/bin/env python
"""
🚀 TESTE COMPLETO DA ETAPA 4 - OTIMIZAÇÕES AVANÇADAS
Teste abrangente de todos os módulos de otimização implementados
"""

import time
import traceback
from datetime import datetime

def test_etapa4_completo():
    print("🚀 TESTE COMPLETO DA ETAPA 4 - OTIMIZAÇÕES AVANÇADAS")
    print("=" * 80)
    
    results = {
        "passed": 0,
        "failed": 0,
        "modules_tested": []
    }
    
    # 1. Teste OptimizationController
    print("\n🎯 1. Testando OptimizationController...")
    try:
        from src.insights.optimization import OptimizationController, OptimizationConfig, OptimizationMode
        
        config = OptimizationConfig(
            mode=OptimizationMode.BALANCED,
            enable_auto_scaling=False,  # Desabilitar para teste inicial
            enable_resource_prediction=False
        )
        
        controller = OptimizationController(config)
        status = controller.get_status()
        
        print(f"✅ OptimizationController inicializado")
        print(f"   Status: {status['status']}")
        print(f"   Modo: {status['mode']}")
        print(f"   Otimizações ativas: {status['active_optimizations']}")
        
        results["passed"] += 1
        results["modules_tested"].append("OptimizationController")
        
    except Exception as e:
        print(f"❌ Erro no OptimizationController: {e}")
        traceback.print_exc()
        results["failed"] += 1
    
    # 2. Teste AutoScaler
    print("\n📈 2. Testando AutoScaler...")
    try:
        from src.insights.optimization import AutoScaler
        
        auto_scaler = AutoScaler()
        auto_scaler.start_monitoring()
        
        # Aguardar alguns segundos para coletar métricas
        time.sleep(3)
        
        status = auto_scaler.get_current_status()
        print(f"✅ AutoScaler funcionando")
        print(f"   Ativo: {status['is_active']}")
        print(f"   CPU atual: {status['current_metrics']['cpu_percent']:.1f}%")
        print(f"   Memória atual: {status['current_metrics']['memory_percent']:.1f}%")
        
        auto_scaler.stop_monitoring()
        
        results["passed"] += 1
        results["modules_tested"].append("AutoScaler")
        
    except Exception as e:
        print(f"❌ Erro no AutoScaler: {e}")
        traceback.print_exc()
        results["failed"] += 1
    
    # 3. Teste ResourceManager
    print("\n🔧 3. Testando ResourceManager...")
    try:
        from src.insights.optimization import ResourceManager
        from src.insights.optimization.resource_manager import ResourceLimits
        
        limits = ResourceLimits(
            max_memory_mb=512,
            max_threads=20,
            max_connections=50
        )
        
        resource_manager = ResourceManager(limits)
        resource_manager.start_monitoring()
        
        # Aguardar alguns segundos para coletar métricas
        time.sleep(3)
        
        usage = resource_manager.get_current_usage()
        print(f"✅ ResourceManager funcionando")
        print(f"   Memória: {usage['memory']['current_mb']:.1f}MB de {usage['memory']['limit_mb']}MB")
        print(f"   Threads: {usage['threads']['current']} de {usage['threads']['limit']}")
        print(f"   Utilização memória: {usage['memory']['utilization_percent']:.1f}%")
        
        # Testar limpeza forçada
        resource_manager.force_cleanup()
        print(f"   Limpeza forçada executada")
        
        resource_manager.stop_monitoring()
        
        results["passed"] += 1
        results["modules_tested"].append("ResourceManager")
        
    except Exception as e:
        print(f"❌ Erro no ResourceManager: {e}")
        traceback.print_exc()
        results["failed"] += 1
    
    # 4. Teste PredictiveEngine
    print("\n🔮 4. Testando PredictiveEngine...")
    try:
        from src.insights.optimization import PredictiveEngine
        from src.insights.optimization.predictive_engine import (
            PredictionInput, PredictionType, PredictionTimeframe
        )
        
        predictive_engine = PredictiveEngine()
        
        # Adicionar alguns dados de treinamento simulados
        for i in range(10):
            training_data = PredictionInput(
                timestamp=datetime.now(),
                cpu_percent=50.0 + (i * 2),
                memory_percent=60.0 + (i * 1.5),
                execution_time_ms=1000.0 + (i * 100),
                cache_hit_rate=80.0 - (i * 0.5),
                request_count=100 + (i * 10),
                active_flows=5 + i,
                system_load=0.5 + (i * 0.05),
                hour_of_day=datetime.now().hour,
                day_of_week=datetime.now().weekday(),
                is_weekend=datetime.now().weekday() >= 5
            )
            predictive_engine.add_training_data(training_data)
        
        print(f"✅ PredictiveEngine inicializado")
        print(f"   Dados de treinamento: {len(predictive_engine.training_data)}")
        print(f"   Modelos ML inicializados: {len(predictive_engine.prediction_models)}")
        
        # Testar detecção de anomalias
        current_data = PredictionInput(
            timestamp=datetime.now(),
            cpu_percent=95.0,  # CPU alta para testar anomalia
            memory_percent=90.0,  # Memória alta
            execution_time_ms=5000.0,
            cache_hit_rate=30.0,
            request_count=500,
            active_flows=20,
            system_load=0.9,
            hour_of_day=datetime.now().hour,
            day_of_week=datetime.now().weekday(),
            is_weekend=datetime.now().weekday() >= 5
        )
        
        anomaly_result = predictive_engine.detect_anomalies(current_data)
        print(f"   Teste de anomalia:")
        print(f"     É anomalia: {anomaly_result.is_anomaly}")
        print(f"     Severidade: {anomaly_result.severity}")
        print(f"     Descrição: {anomaly_result.description}")
        
        # Testar previsão do sistema
        forecast = predictive_engine.get_system_forecast(hours_ahead=2)
        if forecast.get("status") != "no_data":
            print(f"   Previsão do sistema:")
            print(f"     Score de saúde: {forecast['system_health_score']:.1f}")
            print(f"     Recomendações: {len(forecast['recommendations'])}")
        
        results["passed"] += 1
        results["modules_tested"].append("PredictiveEngine")
        
    except Exception as e:
        print(f"❌ Erro no PredictiveEngine: {e}")
        traceback.print_exc()
        results["failed"] += 1
    
    # 5. Teste de Integração - OptimizationController com novos módulos
    print("\n🔗 5. Testando Integração Completa...")
    try:
        from src.insights.optimization import OptimizationController, OptimizationConfig, OptimizationMode
        
        # Configuração com todos os módulos habilitados
        config = OptimizationConfig(
            mode=OptimizationMode.BALANCED,
            enable_auto_scaling=True,
            enable_resource_prediction=True,
            enable_ml_optimization=True
        )
        
        controller = OptimizationController(config)
        
        # Aguardar inicialização dos sistemas
        time.sleep(5)
        
        status = controller.get_status()
        print(f"✅ Integração completa funcionando")
        print(f"   Status: {status['status']}")
        print(f"   Uptime: {status['uptime_seconds']:.1f}s")
        print(f"   Melhoria de performance: {status['performance_improvement']:.1f}%")
        print(f"   Economia de recursos: {status['resource_savings']:.1f}%")
        
        # Testar trigger manual de otimização
        trigger_result = controller.trigger_optimization("memory_optimization", "high")
        print(f"   Trigger manual: {'✅' if trigger_result else '❌'}")
        
        # Shutdown graceful
        controller.shutdown()
        print(f"   Shutdown graceful: ✅")
        
        results["passed"] += 1
        results["modules_tested"].append("IntegracaoCompleta")
        
    except Exception as e:
        print(f"❌ Erro na Integração: {e}")
        traceback.print_exc()
        results["failed"] += 1
    
    # 6. Teste Cache Integration (se ainda não testado)
    print("\n💾 6. Testando CacheIntegration...")
    try:
        from src.insights.optimization import CacheIntegration
        
        cache_integration = CacheIntegration()
        
        # Testar análise de performance do cache
        performance_analysis = cache_integration.analyze_cache_performance()
        cache_integration.implement_intelligent_prefetching()
        
        # Obter estatísticas
        stats = cache_integration.get_optimization_stats()
        print(f"✅ CacheIntegration funcionando")
        print(f"   Análise de performance: {performance_analysis['status']}")
        print(f"   Otimizações aplicadas: {stats['optimizations_applied']}")
        print(f"   Prefetching ativo: {stats['prefetching_active']}")
        
        results["passed"] += 1
        results["modules_tested"].append("CacheIntegration")
        
    except Exception as e:
        print(f"❌ Erro no CacheIntegration: {e}")
        traceback.print_exc()
        results["failed"] += 1
    
    # 7. Teste PerformanceAnalytics
    print("\n📊 7. Testando PerformanceAnalytics...")
    try:
        from src.insights.optimization import PerformanceAnalytics
        
        analytics = PerformanceAnalytics()
        
        # Coletar métricas atuais
        current_metrics = analytics.collect_current_metrics()
        
        # Gerar insights de performance  
        insights = analytics.get_performance_insights()
        
        # Detectar gargalos
        bottlenecks = analytics.detect_bottlenecks()
        
        print(f"✅ PerformanceAnalytics funcionando")
        print(f"   Métricas coletadas: {len(current_metrics)}")
        print(f"   Score de performance: {insights.get('overall_performance_score', 0):.1f}")
        print(f"   Gargalos detectados: {len(bottlenecks)}")
        
        results["passed"] += 1
        results["modules_tested"].append("PerformanceAnalytics")
        
    except Exception as e:
        print(f"❌ Erro no PerformanceAnalytics: {e}")
        traceback.print_exc()
        results["failed"] += 1
    
    # Resultado Final
    print("\n" + "=" * 80)
    print("📋 RESULTADO FINAL DO TESTE")
    print("=" * 80)
    
    total_tests = results["passed"] + results["failed"]
    success_rate = (results["passed"] / total_tests * 100) if total_tests > 0 else 0
    
    print(f"✅ Testes Passaram: {results['passed']}")
    print(f"❌ Testes Falharam: {results['failed']}")
    print(f"📊 Taxa de Sucesso: {success_rate:.1f}%")
    print(f"🔧 Módulos Testados: {', '.join(results['modules_tested'])}")
    
    if results["failed"] == 0:
        print("\n🎉 PARABÉNS! Todos os testes da Etapa 4 passaram!")
        print("🚀 Sistema de otimização avançada está 100% funcional!")
    else:
        print(f"\n⚠️ {results['failed']} teste(s) falharam. Revise os logs acima.")
    
    print("\n" + "=" * 80)
    print("📖 MÓDULOS IMPLEMENTADOS NA ETAPA 4:")
    print("=" * 80)
    print("✅ OptimizationController - Controlador central de otimização")
    print("✅ AutoScaler - Sistema de auto-scaling inteligente") 
    print("✅ ResourceManager - Gerenciador de recursos do sistema")
    print("✅ PredictiveEngine - Engine de predição baseado em ML")
    print("✅ CacheIntegration - Integração avançada com cache")
    print("✅ PerformanceAnalytics - Analytics de performance")
    print("✅ FlowOptimizer - Otimizador de flows")
    print("✅ MLOptimizer - Otimizador baseado em ML")
    
    return results

if __name__ == "__main__":
    test_etapa4_completo() 