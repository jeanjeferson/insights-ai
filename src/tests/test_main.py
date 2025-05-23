#!/usr/bin/env python3
"""
🧪 INSIGHTS-AI TESTING SUITE
================================

Suíte completa de testes para validar todas as ferramentas do projeto Insights-AI.
Execute este arquivo para testar todas as ferramentas individualmente.

Usage:
    python test_main.py              # Executar todos os testes
    python test_main.py --tool sql   # Testar apenas SQL tool
    python test_main.py --verbose    # Modo verboso
    python test_main.py --quick      # Testes rápidos apenas
"""

import sys
import os
import argparse
import time
from datetime import datetime
from pathlib import Path

# Adicionar o diretório pai ao path para importar as ferramentas
sys.path.append(str(Path(__file__).parent.parent))

# Importar módulos de teste
from tests.test_sql_query_tool import test_sql_query_tool
from tests.test_kpi_calculator_tool import test_kpi_calculator_tool
from tests.test_prophet_tool import test_prophet_tool
from tests.test_statistical_analysis_tool import test_statistical_analysis_tool
from tests.test_visualization_tool import test_visualization_tool
from tests.test_advanced_tools import test_advanced_tools
from tests.test_data_validation import test_data_validation
from tests.test_integration import test_integration
from tests.test_security import test_security
from tests.test_regression import test_regression
from tests.test_monitoring import test_monitoring
from tests.test_performance import test_performance
from tests.test_duckduck_tool import test_duckduck_tool

class InsightsTestRunner:
    """Coordenador principal dos testes do Insights-AI"""
    
    def __init__(self, verbose=False, quick=False):
        self.verbose = verbose
        self.quick = quick
        self.results = {}
        self.start_time = None
        
    def log(self, message, level="INFO"):
        """Log com timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if level == "ERROR":
            print(f"🔴 [{timestamp}] {message}")
        elif level == "SUCCESS":
            print(f"✅ [{timestamp}] {message}")
        elif level == "WARNING":
            print(f"⚠️  [{timestamp}] {message}")
        else:
            print(f"ℹ️  [{timestamp}] {message}")
            
    def print_header(self):
        """Imprimir cabeçalho dos testes"""
        print("\n" + "="*80)
        print("🧪 INSIGHTS-AI COMPREHENSIVE TESTING SUITE")
        print("="*80)
        print(f"📅 Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print(f"🚀 Modo: {'Quick Tests' if self.quick else 'Full Tests'}")
        print(f"📝 Verbose: {'Enabled' if self.verbose else 'Disabled'}")
        print("="*80 + "\n")
        
    def run_test(self, test_name, test_function):
        """Executar um teste individual"""
        self.log(f"Iniciando teste: {test_name}")
        
        try:
            start_time = time.time()
            result = test_function(verbose=self.verbose, quick=self.quick)
            duration = time.time() - start_time
            
            if result.get('success', False):
                self.log(f"✅ {test_name} - OK ({duration:.2f}s)", "SUCCESS")
                self.results[test_name] = {
                    'status': 'PASS',
                    'duration': duration,
                    'details': result.get('details', ''),
                    'warnings': result.get('warnings', [])
                }
            else:
                self.log(f"❌ {test_name} - FAILED ({duration:.2f}s)", "ERROR")
                self.results[test_name] = {
                    'status': 'FAIL',
                    'duration': duration,
                    'error': result.get('error', 'Unknown error'),
                    'details': result.get('details', '')
                }
                
        except Exception as e:
            self.log(f"💥 {test_name} - EXCEPTION: {str(e)}", "ERROR")
            self.results[test_name] = {
                'status': 'EXCEPTION',
                'duration': 0,
                'error': str(e)
            }
    
    def run_all_tests(self):
        """Executar todos os testes"""
        self.print_header()
        self.start_time = time.time()
        
        # Definir ordem dos testes (dependências primeiro)
        test_suite = [
            ("Data Validation", test_data_validation),
            ("SQL Query Tool", test_sql_query_tool),
            ("KPI Calculator Tool", test_kpi_calculator_tool),
            ("Statistical Analysis Tool", test_statistical_analysis_tool),
            ("Prophet Forecast Tool", test_prophet_tool),
            ("Visualization Tool", test_visualization_tool),
            ("DuckDuckGo Search Tool", test_duckduck_tool),
            ("Advanced Tools Suite", test_advanced_tools),
            ("Performance Tests", test_performance),
            ("Integration Tests", test_integration),
            ("Security Tests", test_security),
            ("Regression Tests", test_regression),
            ("Monitoring Tests", test_monitoring)
        ]
        
        # Executar testes
        for test_name, test_function in test_suite:
            self.run_test(test_name, test_function)
            
        # Imprimir relatório final
        self.print_final_report()
    
    def run_specific_tool(self, tool_name):
        """Executar teste de uma ferramenta específica"""
        tool_map = {
            'sql': ("SQL Query Tool", test_sql_query_tool),
            'kpi': ("KPI Calculator Tool", test_kpi_calculator_tool),
            'prophet': ("Prophet Forecast Tool", test_prophet_tool),
            'stats': ("Statistical Analysis Tool", test_statistical_analysis_tool),
            'viz': ("Visualization Tool", test_visualization_tool),
            'duckduck': ("DuckDuckGo Search Tool", test_duckduck_tool),
            'advanced': ("Advanced Tools Suite", test_advanced_tools),
            'data': ("Data Validation", test_data_validation),
            'performance': ("Performance Tests", test_performance),
            'integration': ("Integration Tests", test_integration),
            'security': ("Security Tests", test_security),
            'regression': ("Regression Tests", test_regression),
            'monitoring': ("Monitoring Tests", test_monitoring)
        }
        
        if tool_name not in tool_map:
            self.log(f"❌ Tool '{tool_name}' não encontrada. Opções: {list(tool_map.keys())}", "ERROR")
            return
            
        self.print_header()
        self.start_time = time.time()
        
        test_name, test_function = tool_map[tool_name]
        self.run_test(test_name, test_function)
        self.print_final_report()
        
    def print_final_report(self):
        """Imprimir relatório final dos testes"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("📊 RELATÓRIO FINAL DOS TESTES")
        print("="*80)
        
        # Estatísticas gerais
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results.values() if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.results.values() if r['status'] in ['FAIL', 'EXCEPTION']])
        
        print(f"📈 Total de Testes: {total_tests}")
        print(f"✅ Sucessos: {passed_tests}")
        print(f"❌ Falhas: {failed_tests}")
        print(f"⏱️  Tempo Total: {total_time:.2f}s")
        print(f"📊 Taxa de Sucesso: {(passed_tests/total_tests*100):.1f}%")
        
        # Detalhes por teste
        print("\n📋 DETALHES POR TESTE:")
        print("-" * 60)
        
        for test_name, result in self.results.items():
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            print(f"{status_icon} {test_name:<30} {result['status']:<10} ({result['duration']:.2f}s)")
            
            if result['status'] != 'PASS':
                print(f"    🔍 Erro: {result.get('error', 'N/A')}")
                
            if result.get('warnings'):
                for warning in result['warnings']:
                    print(f"    ⚠️  Warning: {warning}")
        
        # Recomendações
        print("\n💡 RECOMENDAÇÕES:")
        print("-" * 40)
        
        if failed_tests == 0:
            print("🎉 Todos os testes passaram! Sistema funcionando corretamente.")
        else:
            print("🔧 Algumas ferramentas precisam de atenção:")
            for test_name, result in self.results.items():
                if result['status'] != 'PASS':
                    print(f"   • {test_name}: {result.get('error', 'Verificar logs')}")
        
        print("\n" + "="*80)

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Insights-AI Testing Suite')
    parser.add_argument('--tool', type=str, help='Testar ferramenta específica (sql, kpi, prophet, stats, viz, duckduck, advanced, data, performance, integration, security, regression, monitoring)')
    parser.add_argument('--verbose', action='store_true', help='Modo verboso')
    parser.add_argument('--quick', action='store_true', help='Testes rápidos apenas')
    
    args = parser.parse_args()
    
    # Criar runner de testes
    runner = InsightsTestRunner(verbose=args.verbose, quick=args.quick)
    
    # Executar testes
    if args.tool:
        runner.run_specific_tool(args.tool)
    else:
        runner.run_all_tests()

if __name__ == "__main__":
    main()
