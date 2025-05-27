#!/usr/bin/env python
"""
Teste da Etapa 4 - Otimizações Avançadas
"""

def test_etapa4():
    print("🚀 TESTE DA ETAPA 4 - OTIMIZAÇÕES AVANÇADAS")
    print("=" * 60)
    
    # 1. Teste Cache Inteligente
    print("\n📦 1. Testando Cache Inteligente...")
    try:
        from src.insights.cache import IntelligentCacheSystem, CacheType
        
        cache = IntelligentCacheSystem()
        print("✅ Cache System inicializado")
        
        # Testar operações básicas
        cache.set('test_key', 'test_data', CacheType.ANALYSIS_RESULT)
        result = cache.get('test_key')
        
        if result == 'test_data':
            print("✅ Cache funcionando corretamente")
        else:
            print("❌ Cache não retornou dados corretos")
        
        # Testar estatísticas
        stats = cache.get_stats()
        print(f"📊 Stats: Hit Rate: {stats.get('hit_rate_percent', 0):.1f}%")
        
    except Exception as e:
        print(f"❌ Erro no Cache: {e}")
    
    # 2. Teste Condições de Execução
    print("\n🔄 2. Testando Condições de Execução...")
    try:
        from src.insights.conditional import PerformanceCondition, TemporalCondition
        
        # Teste condição de performance
        perf_condition = PerformanceCondition()
        context = {}
        result = perf_condition.evaluate(context)
        
        print(f"✅ Condição Performance: {result.should_execute} (confidence: {result.confidence:.2f})")
        print(f"   Razão: {result.reason}")
        
        # Teste condição temporal
        temporal_condition = TemporalCondition()
        result = temporal_condition.evaluate(context)
        
        print(f"✅ Condição Temporal: {result.should_execute} (confidence: {result.confidence:.2f})")
        print(f"   Razão: {result.reason}")
        
    except Exception as e:
        print(f"❌ Erro nas Condições: {e}")
    
    # 3. Teste Analytics
    print("\n📊 3. Testando Analytics...")
    try:
        from src.insights.cache import IntelligentCacheSystem, CacheAnalytics
        
        cache = IntelligentCacheSystem()
        analytics = CacheAnalytics(cache)
        
        # Coletar métricas
        metrics = analytics.collect_metrics()
        print(f"✅ Métricas coletadas: {len(metrics)} tipos")
        
        # Gerar insights
        insights = analytics.get_performance_insights()
        health = insights.get('overall_health', 'unknown')
        score = insights.get('optimization_score', 0)
        
        print(f"✅ Health: {health}, Score: {score:.1f}")
        
    except Exception as e:
        print(f"❌ Erro no Analytics: {e}")
    
    print("\n🎯 ETAPA 4 - TESTE CONCLUÍDO!")

if __name__ == "__main__":
    test_etapa4() 