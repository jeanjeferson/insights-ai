#!/usr/bin/env python3
"""
🚀 EXECUTOR DE TESTES - ADVANCED ANALYTICS ENGINE TOOL
======================================================

Script para executar testes do Advanced Analytics Engine Tool
de forma independente e gerar relatórios detalhados.

Uso:
    python run_analytics_tests.py
    python run_analytics_tests.py --quick
    python run_analytics_tests.py --full --data-path custom_data.csv
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Configurar caminhos de importação de forma robusta
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# Adicionar caminhos possíveis ao sys.path
possible_paths = [
    project_root,
    os.path.join(project_root, 'src'),
    current_dir,
    os.path.dirname(current_dir)
]

for path in possible_paths:
    if path not in sys.path:
        sys.path.insert(0, path)

def main():
    parser = argparse.ArgumentParser(description='Executar testes do Advanced Analytics Engine Tool')
    parser.add_argument('--quick', action='store_true', help='Executar apenas testes essenciais')
    parser.add_argument('--full', action='store_true', help='Executar todos os testes (padrão)')
    parser.add_argument('--data-path', default='data/vendas.csv', help='Caminho para arquivo de dados')
    parser.add_argument('--output-dir', default='test_results', help='Diretório para resultados')
    
    args = parser.parse_args()
    
    # Verificar se arquivo de dados existe
    if not os.path.exists(args.data_path):
        print(f"❌ Arquivo de dados não encontrado: {args.data_path}")
        print("💡 Certifique-se de que o arquivo existe ou use --data-path para especificar outro arquivo")
        return 1
    
    # Importar classe de testes
    try:
        from test_advanced_analytics_engine_tool import TestAdvancedAnalyticsEngineTool
    except ImportError as e:
        print(f"❌ Erro ao importar testes: {e}")
        print("💡 Certifique-se de estar no diretório correto e que as dependências estão instaladas")
        return 1
    
    # Criar diretório de resultados
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🧪 ADVANCED ANALYTICS ENGINE TOOL - SUITE DE TESTES")
    print("=" * 60)
    print(f"📁 Dados: {args.data_path}")
    print(f"📊 Modo: {'Rápido' if args.quick else 'Completo'}")
    print(f"📋 Resultados: {output_dir}")
    print("=" * 60)
    
    # Inicializar testes
    test_instance = TestAdvancedAnalyticsEngineTool()
    test_instance.setup_standalone(args.data_path)
    
    # Definir testes a executar
    if args.quick:
        # Testes essenciais (mais rápidos)
        tests_to_run = [
            ('Validação de Dados', test_instance.test_data_loading_and_validation),
            ('ML Insights', test_instance.test_ml_insights_analysis),
            ('Anomaly Detection', test_instance.test_anomaly_detection_analysis),
            ('Tratamento de Erros', test_instance.test_error_handling_comprehensive),
            ('Compatibilidade', test_instance.test_library_compatibility)
        ]
    else:
        # Todos os testes
        tests_to_run = [
            ('Validação de Dados', test_instance.test_data_loading_and_validation),
            ('Preparação de Features', test_instance.test_feature_preparation),
            ('ML Insights', test_instance.test_ml_insights_analysis),
            ('Anomaly Detection', test_instance.test_anomaly_detection_analysis),
            ('Customer Behavior', test_instance.test_customer_behavior_analysis),
            ('Demand Forecasting', test_instance.test_demand_forecasting_analysis),
            ('Price Optimization', test_instance.test_price_optimization_analysis),
            ('Inventory Optimization', test_instance.test_inventory_optimization_analysis),
            ('Integração Completa', test_instance.test_all_analysis_types_integration),
            ('Cache Avançado', test_instance.test_cache_functionality_advanced),
            ('Performance', test_instance.test_performance_benchmarks),
            ('Tratamento de Erros', test_instance.test_error_handling_comprehensive),
            ('Casos Extremos', test_instance.test_edge_cases),
            ('Compatibilidade', test_instance.test_library_compatibility),
            ('Qualidade de Saída', test_instance.test_output_quality_and_formatting)
        ]
    
    # Executar testes
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests_to_run:
        print(f"\n{'='*60}")
        print(f"🔄 Executando: {test_name}")
        print("-" * 60)
        
        test_start = time.time()
        
        try:
            test_func()
            test_duration = time.time() - test_start
            print(f"✅ {test_name} - PASSOU ({test_duration:.2f}s)")
            
            results.append({
                'name': test_name,
                'status': 'PASSOU',
                'duration': round(test_duration, 2),
                'error': None
            })
            
        except Exception as e:
            test_duration = time.time() - test_start
            error_msg = str(e)
            print(f"❌ {test_name} - FALHOU ({test_duration:.2f}s)")
            print(f"   Erro: {error_msg[:100]}...")
            
            results.append({
                'name': test_name,
                'status': 'FALHOU',
                'duration': round(test_duration, 2),
                'error': error_msg
            })
        
        finally:
            test_instance.teardown_method(test_func)
    
    total_duration = time.time() - start_time
    
    # Gerar relatório
    passed = sum(1 for r in results if r['status'] == 'PASSOU')
    failed = len(results) - passed
    success_rate = passed / len(results) if results else 0
    
    print(f"\n{'='*60}")
    print(f"🎯 RELATÓRIO FINAL")
    print(f"{'='*60}")
    print(f"⏱️  Tempo Total: {total_duration:.2f}s")
    print(f"✅ Testes Aprovados: {passed}/{len(results)} ({success_rate:.1%})")
    print(f"❌ Testes Falharam: {failed}")
    
    # Salvar relatório detalhado
    report_file = output_dir / f"analytics_test_report_{int(time.time())}.json"
    
    import json
    report_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'mode': 'quick' if args.quick else 'full',
        'data_path': args.data_path,
        'total_duration': round(total_duration, 2),
        'summary': {
            'total_tests': len(results),
            'passed': passed,
            'failed': failed,
            'success_rate': round(success_rate, 3)
        },
        'results': results
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Relatório salvo: {report_file}")
    
    # Status de saída
    if success_rate >= 0.9:
        print(f"\n🎉 EXCELENTE! Advanced Analytics Engine Tool está funcionando perfeitamente!")
        return 0
    elif success_rate >= 0.8:
        print(f"\n✅ BOM! Maioria dos testes passou. Algumas funcionalidades podem precisar de ajustes.")
        return 0
    else:
        print(f"\n⚠️ ATENÇÃO! Muitos testes falharam. Engine precisa de correções.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 