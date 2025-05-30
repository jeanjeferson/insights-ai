"""
EXEMPLO: Como Usar as Funções de Teste do CustomerDataExporter
==============================================================

Este exemplo demonstra como utilizar as novas funcionalidades de teste
implementadas no CustomerDataExporter.
"""

from src.insights.tools.customer_data_exporter import CustomerDataExporter

def demonstrar_testes_customer_exporter():
    """Demonstra como usar as funções de teste do CustomerDataExporter"""
    
    print("🎯 DEMONSTRAÇÃO: Testes do CustomerDataExporter")
    print("=" * 60)
    
    # Instanciar a ferramenta
    exporter = CustomerDataExporter()
    
    print("\n📋 TIPOS DE TESTE DISPONÍVEIS:")
    print("1. Teste Completo com Relatório (run_full_customer_test)")
    print("2. Teste JSON Estruturado (test_all_customer_components)")
    print("3. Teste de Componente Individual (usando _run)")
    
    # 1. TESTE COMPLETO COM RELATÓRIO MARKDOWN
    print("\n" + "="*60)
    print("1️⃣ EXECUTANDO TESTE COMPLETO COM RELATÓRIO")
    print("="*60)
    
    try:
        print("📊 Executando teste completo...")
        report_markdown = exporter.run_full_customer_test()
        
        print("✅ Teste concluído com sucesso!")
        print(f"📄 Relatório gerado: {len(report_markdown):,} caracteres")
        
        # Salvar relatório com timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"exemplos_output/customer_test_report_{timestamp}.md"
        
        import os
        os.makedirs("exemplos_output", exist_ok=True)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_markdown)
        
        print(f"💾 Relatório salvo em: {report_file}")
        
        # Exibir preview do relatório
        print("\n📖 PREVIEW DO RELATÓRIO:")
        print("-" * 40)
        lines = report_markdown.split('\n')[:20]  # Primeiras 20 linhas
        for line in lines:
            print(line)
        print("...")
        
    except Exception as e:
        print(f"❌ Erro no teste completo: {str(e)}")
    
    # 2. TESTE JSON ESTRUTURADO
    print("\n" + "="*60)
    print("2️⃣ EXECUTANDO TESTE JSON ESTRUTURADO")
    print("="*60)
    
    try:
        print("📊 Executando teste com saída JSON...")
        json_result = exporter.test_all_customer_components()
        
        import json
        parsed_result = json.loads(json_result)
        
        print("✅ Teste JSON concluído!")
        print(f"📊 Status: {parsed_result['metadata']['status']}")
        print(f"🔢 Componentes testados: {len(parsed_result['results'])}")
        print(f"❌ Erros encontrados: {len(parsed_result['errors'])}")
        
        # Mostrar resumo dos resultados
        print("\n📋 RESUMO DOS RESULTADOS:")
        for component, result in parsed_result['results'].items():
            status = "✅" if result.get('success') else "❌"
            time_taken = result.get('metrics', {}).get('processing_time', 0)
            print(f"  {status} {component}: {time_taken:.3f}s")
        
    except Exception as e:
        print(f"❌ Erro no teste JSON: {str(e)}")
    
    # 3. TESTE DE FUNCIONALIDADE ESPECÍFICA
    print("\n" + "="*60)
    print("3️⃣ TESTE DE FUNCIONALIDADE ESPECÍFICA")
    print("="*60)
    
    try:
        print("🎯 Testando funcionalidade específica (análise RFM + CLV)...")
        
        # Usar a função _run com parâmetros específicos
        result = exporter._run(
            data_csv="data/vendas.csv",
            output_path="exemplos_output/customer_test_specific.csv",
            include_rfm_analysis=True,
            include_clv_calculation=True,
            include_geographic_analysis=False,  # Desabilitar para teste rápido
            include_behavioral_insights=False,
            clv_months=12  # CLV para 12 meses
        )
        
        print("✅ Teste específico concluído!")
        print("📄 Resultado:")
        print(result[:500] + "..." if len(result) > 500 else result)
        
    except Exception as e:
        print(f"❌ Erro no teste específico: {str(e)}")
    
    # 4. ANÁLISE COMPARATIVA DE PERFORMANCE
    print("\n" + "="*60)
    print("4️⃣ ANÁLISE COMPARATIVA DE PERFORMANCE")
    print("="*60)
    
    try:
        import time
        
        print("⏱️ Testando performance de diferentes configurações...")
        
        configurations = [
            {
                "name": "Análise Básica (RFM + CLV)",
                "params": {
                    "include_rfm_analysis": True,
                    "include_clv_calculation": True,
                    "include_geographic_analysis": False,
                    "include_behavioral_insights": False
                }
            },
            {
                "name": "Análise Completa",
                "params": {
                    "include_rfm_analysis": True,
                    "include_clv_calculation": True,
                    "include_geographic_analysis": True,
                    "include_behavioral_insights": True
                }
            }
        ]
        
        performance_results = []
        
        for config in configurations:
            print(f"🧪 Testando: {config['name']}")
            start_time = time.time()
            
            try:
                result = exporter._run(
                    data_csv="data/vendas.csv",
                    output_path=f"exemplos_output/perf_test_{config['name'].lower().replace(' ', '_')}.csv",
                    **config['params']
                )
                execution_time = time.time() - start_time
                
                performance_results.append({
                    "config": config['name'],
                    "time": execution_time,
                    "success": True
                })
                print(f"  ✅ Concluído em {execution_time:.2f}s")
                
            except Exception as e:
                execution_time = time.time() - start_time
                performance_results.append({
                    "config": config['name'],
                    "time": execution_time,
                    "success": False,
                    "error": str(e)
                })
                print(f"  ❌ Falhou em {execution_time:.2f}s: {str(e)}")
        
        # Exibir comparativo de performance
        print("\n📊 COMPARATIVO DE PERFORMANCE:")
        print("-" * 50)
        for result in performance_results:
            status = "✅" if result['success'] else "❌"
            print(f"{status} {result['config']}: {result['time']:.2f}s")
            if not result['success']:
                print(f"   Erro: {result.get('error', 'Desconhecido')}")
        
    except Exception as e:
        print(f"❌ Erro na análise de performance: {str(e)}")
    
    print("\n" + "="*60)
    print("🎯 DEMONSTRAÇÃO CONCLUÍDA!")
    print("="*60)
    print("📁 Arquivos gerados em: exemplos_output/")
    print("📊 Relatórios de teste disponíveis")
    print("🔍 Use os testes para validar modificações na classe")

if __name__ == "__main__":
    demonstrar_testes_customer_exporter() 