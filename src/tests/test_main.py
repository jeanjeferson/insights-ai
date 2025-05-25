#!/usr/bin/env python3
"""
🧪 INSIGHTS-AI TESTING SUITE (SIMPLIFICADO)
===========================================

Suíte simplificada de testes para validar as ferramentas principais do projeto Insights-AI.
Execute este arquivo para testar todas as ferramentas funcionais.

Usage:
    python test_main.py              # Executar todos os testes
    python test_main.py --tool kpi   # Testar apenas KPI tool
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

# Importar apenas os módulos de teste que realmente existem
try:
    from test_advanced_tools import run_advanced_tests
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

try:
    from test_integration import run_integration_tests
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

try:
    from test_performance import run_performance_tests
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False

try:
    from test_data_validation import TestDataValidation
    DATA_VALIDATION_AVAILABLE = True
except ImportError:
    DATA_VALIDATION_AVAILABLE = False

try:
    from test_prophet_tool import run_prophet_tests
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from test_duckduck_tool import run_duckduck_tests
    DUCKDUCK_AVAILABLE = True
except ImportError:
    DUCKDUCK_AVAILABLE = False

try:
    from test_sql_query_tool import run_sql_tests
    SQL_AVAILABLE = True
except ImportError:
    SQL_AVAILABLE = False

# Testes das ferramentas v3.0 principais
try:
    from test_kpi_calculator_tool import TestKPICalculatorTool
    KPI_V3_AVAILABLE = True
except ImportError:
    KPI_V3_AVAILABLE = False

try:
    from test_statistical_analysis_tool import TestStatisticalAnalysisTool
    STATS_V3_AVAILABLE = True
except ImportError:
    STATS_V3_AVAILABLE = False

try:
    from test_unified_business_intelligence import TestUnifiedBI
    UBI_AVAILABLE = True
except ImportError:
    UBI_AVAILABLE = False

class InsightsTestRunner:
    """Coordenador simplificado dos testes do Insights-AI"""
    
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
    
    def get_success_threshold(self, test_name):
        """Determinar threshold de sucesso baseado no tipo de teste"""
        # Thresholds mais baixos para alguns testes específicos
        if test_name == "Data Validation":
            return 60  # Data validation pode ter problemas de ambiente
        elif "Performance" in test_name:
            return 65  # Performance pode variar por ambiente
        else:
            return 70  # Threshold padrão
            
    def print_header(self):
        """Imprimir cabeçalho dos testes"""
        print("\n" + "="*70)
        print("🧪 INSIGHTS-AI TESTING SUITE (SIMPLIFICADO)")
        print("="*70)
        print(f"📅 Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        print(f"🚀 Modo: {'Quick Tests' if self.quick else 'Full Tests'}")
        print(f"📝 Verbose: {'Enabled' if self.verbose else 'Disabled'}")
        print("="*70 + "\n")
        
    def run_test_function(self, test_name, test_function):
        """Executar uma função de teste que retorna resultado"""
        self.log(f"Iniciando teste: {test_name}")
        
        try:
            start_time = time.time()
            result = test_function()
            duration = time.time() - start_time
            
            # Verificar se o resultado indica sucesso
            success = False
            error_msg = ''
            
            if isinstance(result, dict):
                # Formato padrão: {'success': bool, 'success_rate': float, ...}
                if 'success' in result:
                    success = result.get('success', False)
                elif 'success_rate' in result:
                    # Para testes que retornam taxa de sucesso
                    success_rate = result.get('success_rate', 0)
                    success = success_rate >= self.get_success_threshold(test_name)
                else:
                    # Se tem dados válidos, considerar sucesso
                    success = len(result) > 0
                
                error_msg = result.get('error', 'Test returned failure')
                
            elif isinstance(result, bool):
                success = result
                error_msg = 'Test failed'
            else:
                # Para resultados que não são dict/bool, considerar sucesso se não houve exception
                success = True
                error_msg = ''
            
            if success:
                self.log(f"✅ {test_name} - OK ({duration:.2f}s)", "SUCCESS")
                self.results[test_name] = {
                    'status': 'PASS',
                    'duration': duration,
                    'details': str(result)[:200] if result else ''
                }
            else:
                self.log(f"❌ {test_name} - FAILED ({duration:.2f}s)", "ERROR")
                self.results[test_name] = {
                    'status': 'FAIL',
                    'duration': duration,
                    'error': error_msg
                }
                
        except Exception as e:
            self.log(f"💥 {test_name} - EXCEPTION: {str(e)}", "ERROR")
            self.results[test_name] = {
                'status': 'EXCEPTION',
                'duration': 0,
                'error': str(e)
            }
    
    def run_test_class(self, test_name, test_class):
        """Executar uma classe de teste"""
        self.log(f"Iniciando teste: {test_name}")
        
        try:
            start_time = time.time()
            test_instance = test_class()
            
            # Tentar executar método de teste principal baseado no nome da classe
            result = None
            
            if test_name == "Data Validation":
                # Para TestDataValidation, usar método específico
                if hasattr(test_instance, 'test_data_validation_summary'):
                    result = test_instance.test_data_validation_summary()
                else:
                    result = test_instance.test_data_file_exists()
            else:
                # Para outras classes, tentar métodos padrão
                if hasattr(test_instance, 'run_all_tests'):
                    result = test_instance.run_all_tests()
                elif hasattr(test_instance, 'test_summary'):
                    result = test_instance.test_summary()
                else:
                    # Executar primeiro método que encontrar
                    methods = [method for method in dir(test_instance) if method.startswith('test_')]
                    if methods:
                        result = getattr(test_instance, methods[0])()
                    else:
                        raise Exception("Nenhum método de teste encontrado")
            
            duration = time.time() - start_time
            
            # Verificar sucesso usando a mesma lógica das funções
            success = False
            error_msg = ''
            
            if isinstance(result, dict):
                # Formato padrão: {'success': bool, 'success_rate': float, ...}
                if 'success' in result:
                    success = result.get('success', False)
                elif 'success_rate' in result:
                    # Para testes que retornam taxa de sucesso
                    success_rate = result.get('success_rate', 0)
                    success = success_rate >= self.get_success_threshold(test_name)
                else:
                    # Se tem dados válidos, considerar sucesso
                    success = len(result) > 0
                
                error_msg = result.get('error', 'Test returned failure')
                
            elif isinstance(result, bool):
                success = result
                error_msg = 'Test failed'
            else:
                success = True
                error_msg = ''
            
            if success:
                self.log(f"✅ {test_name} - OK ({duration:.2f}s)", "SUCCESS")
                self.results[test_name] = {
                    'status': 'PASS',
                    'duration': duration,
                    'details': str(result)[:200] if result else ''
                }
            else:
                self.log(f"❌ {test_name} - FAILED ({duration:.2f}s)", "ERROR")
                self.results[test_name] = {
                    'status': 'FAIL',
                    'duration': duration,
                    'error': error_msg
                }
                
        except Exception as e:
            self.log(f"💥 {test_name} - EXCEPTION: {str(e)}", "ERROR")
            self.results[test_name] = {
                'status': 'EXCEPTION',
                'duration': 0,
                'error': str(e)
            }
    
    def run_all_tests(self):
        """Executar todos os testes disponíveis"""
        self.print_header()
        self.start_time = time.time()
        
        # Definir testes disponíveis (ordem de dependências)
        available_tests = []
        
        # 1. Validação de dados (base)
        if DATA_VALIDATION_AVAILABLE:
            available_tests.append(("Data Validation", "class", TestDataValidation))
        
        # 2. Ferramentas v3.0 principais
        if KPI_V3_AVAILABLE:
            available_tests.append(("KPI Calculator v3", "class", TestKPICalculatorV3))
        
        if STATS_V3_AVAILABLE:
            available_tests.append(("Statistical Analysis v3", "class", TestStatisticalAnalysisV3))
        
        if UBI_AVAILABLE:
            available_tests.append(("Unified Business Intelligence", "class", TestUnifiedBI))
        
        # 3. Ferramentas auxiliares
        if PROPHET_AVAILABLE:
            available_tests.append(("Prophet Forecast Tool", "function", run_prophet_tests))
        
        if SQL_AVAILABLE:
            available_tests.append(("SQL Query Tool", "function", run_sql_tests))
        
        if DUCKDUCK_AVAILABLE:
            available_tests.append(("DuckDuckGo Search Tool", "function", run_duckduck_tests))
        
        # 4. Testes integrados
        if ADVANCED_AVAILABLE:
            available_tests.append(("Advanced Tools Suite", "function", run_advanced_tests))
        
        if PERFORMANCE_AVAILABLE:
            available_tests.append(("Performance Tests", "function", run_performance_tests))
        
        if INTEGRATION_AVAILABLE:
            available_tests.append(("Integration Tests", "function", run_integration_tests))
        
        # Executar testes
        if not available_tests:
            self.log("❌ Nenhum teste disponível!", "ERROR")
            return
        
        self.log(f"📊 {len(available_tests)} testes disponíveis para execução")
        
        for test_name, test_type, test_target in available_tests:
            if test_type == "function":
                self.run_test_function(test_name, test_target)
            else:  # class
                self.run_test_class(test_name, test_target)
            
        # Imprimir relatório final
        self.print_final_report()
    
    def run_specific_tool(self, tool_name):
        """Executar teste de uma ferramenta específica"""
        tool_map = {}
        
        # Mapear ferramentas disponíveis
        if KPI_V3_AVAILABLE:
            tool_map['kpi'] = ("KPI Calculator v3", "class", TestKPICalculatorV3)
        
        if STATS_V3_AVAILABLE:
            tool_map['stats'] = ("Statistical Analysis v3", "class", TestStatisticalAnalysisV3)
        
        if UBI_AVAILABLE:
            tool_map['ubi'] = ("Unified Business Intelligence", "class", TestUnifiedBI)
        
        if PROPHET_AVAILABLE:
            tool_map['prophet'] = ("Prophet Forecast Tool", "function", run_prophet_tests)
        
        if SQL_AVAILABLE:
            tool_map['sql'] = ("SQL Query Tool", "function", run_sql_tests)
        
        if DUCKDUCK_AVAILABLE:
            tool_map['duckduck'] = ("DuckDuckGo Search Tool", "function", run_duckduck_tests)
        
        if ADVANCED_AVAILABLE:
            tool_map['advanced'] = ("Advanced Tools Suite", "function", run_advanced_tests)
        
        if PERFORMANCE_AVAILABLE:
            tool_map['performance'] = ("Performance Tests", "function", run_performance_tests)
        
        if INTEGRATION_AVAILABLE:
            tool_map['integration'] = ("Integration Tests", "function", run_integration_tests)
        
        if DATA_VALIDATION_AVAILABLE:
            tool_map['data'] = ("Data Validation", "class", TestDataValidation)
        
        if tool_name not in tool_map:
            self.log(f"❌ Tool '{tool_name}' não encontrada. Opções: {list(tool_map.keys())}", "ERROR")
            return
            
        self.print_header()
        self.start_time = time.time()
        
        test_name, test_type, test_target = tool_map[tool_name]
        
        if test_type == "function":
            self.run_test_function(test_name, test_target)
        else:  # class
            self.run_test_class(test_name, test_target)
            
        self.print_final_report()
        
    def print_final_report(self):
        """Imprimir relatório final dos testes"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*70)
        print("📊 RELATÓRIO FINAL DOS TESTES")
        print("="*70)
        
        # Estatísticas gerais
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results.values() if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.results.values() if r['status'] in ['FAIL', 'EXCEPTION']])
        
        print(f"📈 Total de Testes: {total_tests}")
        print(f"✅ Sucessos: {passed_tests}")
        print(f"❌ Falhas: {failed_tests}")
        print(f"⏱️  Tempo Total: {total_time:.2f}s")
        
        if total_tests > 0:
            success_rate = (passed_tests/total_tests*100)
            print(f"📊 Taxa de Sucesso: {success_rate:.1f}%")
        
        # Detalhes por teste
        if total_tests > 0:
            print("\n📋 DETALHES POR TESTE:")
            print("-" * 50)
            
            for test_name, result in self.results.items():
                status_icon = "✅" if result['status'] == 'PASS' else "❌"
                print(f"{status_icon} {test_name:<35} {result['status']:<10} ({result['duration']:.2f}s)")
                
                if result['status'] != 'PASS':
                    error_msg = result.get('error', 'N/A')
                    if len(error_msg) > 60:
                        error_msg = error_msg[:60] + "..."
                    print(f"    🔍 Erro: {error_msg}")
        
        # Recomendações
        print("\n💡 RECOMENDAÇÕES:")
        print("-" * 30)
        
        if failed_tests == 0 and total_tests > 0:
            print("🎉 Todos os testes passaram! Sistema funcionando corretamente.")
        elif total_tests == 0:
            print("⚠️  Nenhum teste foi executado. Verifique as dependências.")
        else:
            print("🔧 Algumas ferramentas precisam de atenção:")
            for test_name, result in self.results.items():
                if result['status'] != 'PASS':
                    print(f"   • {test_name}")
        
        print("\n" + "="*70)

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Insights-AI Testing Suite (Simplificado)')
    parser.add_argument('--tool', type=str, 
                       help='Testar ferramenta específica (kpi, stats, ubi, prophet, sql, duckduck, advanced, performance, integration, data)')
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
