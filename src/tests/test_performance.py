"""
⚡ TESTE: PERFORMANCE BÁSICA (SIMPLIFICADO)
==========================================

Testa a performance básica das ferramentas v3.0 reais.
Versão simplificada focada em métricas essenciais.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import tempfile

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent))

# Importar ferramentas v3.0 reais
try:
    from insights.tools.kpi_calculator_tool import KPICalculatorTool
    KPI_V3_AVAILABLE = True
except ImportError:
    KPI_V3_AVAILABLE = False

try:
    from insights.tools.statistical_analysis_tool import StatisticalAnalysisTool
    STATS_V3_AVAILABLE = True
except ImportError:
    STATS_V3_AVAILABLE = False

try:
    from insights.tools.business_intelligence_tool import BusinessIntelligenceTool
    UBI_AVAILABLE = True
except ImportError:
    UBI_AVAILABLE = False

def create_performance_test_data(size='small'):
    """Criar dados simples para testes de performance"""
    np.random.seed(42)
    
    sizes = {
        'small': 100,
        'medium': 1000,
        'large': 5000
    }
    
    n_records = sizes.get(size, sizes['small'])
    
    # Gerar dados simples e rápidos
    dates = pd.date_range('2024-01-01', periods=n_records, freq='D')
    
    data = []
    for i, date in enumerate(dates):
        data.append({
            'Data': date.strftime('%Y-%m-%d'),
            'Codigo_Cliente': f"CLI_{(i % 20) + 1:03d}",
            'Codigo_Produto': f"PROD_{(i % 10) + 1:03d}",
            'Categoria': np.random.choice(['Anéis', 'Brincos', 'Colares']),
            'Quantidade': np.random.randint(1, 4),
            'Total_Liquido': np.random.uniform(100, 2000),
            'Preco_Unitario': np.random.uniform(50, 500)
        })
    
    return pd.DataFrame(data)

class TestPerformance:
    """Classe simplificada para testes de performance"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.test_data_small = create_performance_test_data('small')
        self.test_data_medium = create_performance_test_data('medium')
        
        # Criar arquivos temporários
        self.test_csv_small = "temp_perf_small.csv"
        self.test_csv_medium = "temp_perf_medium.csv"
        
        self.test_data_small.to_csv(self.test_csv_small, sep=';', index=False, encoding='utf-8')
        self.test_data_medium.to_csv(self.test_csv_medium, sep=';', index=False, encoding='utf-8')
    
    def teardown_method(self):
        """Cleanup após cada teste"""
        for csv_file in [self.test_csv_small, self.test_csv_medium]:
            if os.path.exists(csv_file):
                os.remove(csv_file)
    
    def test_kpi_performance(self):
        """Teste de performance do KPI Calculator v3"""
        if not KPI_V3_AVAILABLE:
            print("⚠️ KPI Calculator v3 não disponível - pulando teste")
            return
        
        try:
            kpi_tool = KPICalculatorTool()
            
            # Teste com dados pequenos
            start_time = time.time()
            result_small = kpi_tool._run(
                data_csv=self.test_csv_small,
                categoria="revenue",
                periodo="monthly"
            )
            time_small = time.time() - start_time
            
            # Teste com dados médios
            start_time = time.time()
            result_medium = kpi_tool._run(
                data_csv=self.test_csv_medium,
                categoria="revenue",
                periodo="monthly"
            )
            time_medium = time.time() - start_time
            
            # Validações
            assert result_small is not None, "KPI pequeno retornou None"
            assert result_medium is not None, "KPI médio retornou None"
            assert time_small < 30, f"KPI pequeno muito lento: {time_small:.2f}s"
            assert time_medium < 60, f"KPI médio muito lento: {time_medium:.2f}s"
            
            print(f"✅ KPI Performance: {len(self.test_data_small)} reg em {time_small:.2f}s, {len(self.test_data_medium)} reg em {time_medium:.2f}s")
            
        except Exception as e:
            print(f"❌ KPI Performance: FALHOU - {e}")
            raise
    
    def test_stats_performance(self):
        """Teste de performance do Statistical Analysis v3"""
        if not STATS_V3_AVAILABLE:
            print("⚠️ Statistical Analysis v3 não disponível - pulando teste")
            return
        
        try:
            stats_tool = StatisticalAnalysisTool()
            
            # Teste com dados pequenos
            start_time = time.time()
            result_small = stats_tool._run(
                analysis_type="correlation",
                data_csv=self.test_csv_small,
                target_column="Total_Liquido"
            )
            time_small = time.time() - start_time
            
            # Validações
            assert result_small is not None, "Stats pequeno retornou None"
            assert time_small < 30, f"Stats muito lento: {time_small:.2f}s"
            
            print(f"✅ Stats Performance: {len(self.test_data_small)} reg em {time_small:.2f}s")
            
        except Exception as e:
            print(f"❌ Stats Performance: FALHOU - {e}")
            raise
    
    def test_ubi_performance(self):
        """Teste de performance do Unified BI"""
        if not UBI_AVAILABLE:
            print("⚠️ Unified BI não disponível - pulando teste")
            return
        
        try:
            ubi_tool = BusinessIntelligenceTool()
            
            # Teste com dados pequenos
            start_time = time.time()
            result_small = ubi_tool._run(
                data_csv=self.test_csv_small,
                analysis_type="executive_summary",
                output_format="interactive"
            )
            time_small = time.time() - start_time
            
            # Validações
            assert result_small is not None, "UBI pequeno retornou None"
            assert time_small < 45, f"UBI muito lento: {time_small:.2f}s"
            
            print(f"✅ UBI Performance: {len(self.test_data_small)} reg em {time_small:.2f}s")
            
        except Exception as e:
            print(f"❌ UBI Performance: FALHOU - {e}")
            raise
    
    def test_performance_summary(self):
        """Teste resumo de performance de todas as ferramentas"""
        success_count = 0
        total_tests = 0
        performance_results = {}
        
        # Testar cada ferramenta disponível
        tools_to_test = [
            ("KPI v3", KPI_V3_AVAILABLE, self.test_kpi_performance),
            ("Stats v3", STATS_V3_AVAILABLE, self.test_stats_performance),
            ("UBI", UBI_AVAILABLE, self.test_ubi_performance)
        ]
        
        for tool_name, available, test_func in tools_to_test:
            if available:
                total_tests += 1
                try:
                    start_time = time.time()
                    test_func()
                    execution_time = time.time() - start_time
                    
                    performance_results[tool_name] = {
                        'status': 'SUCCESS',
                        'execution_time': round(execution_time, 2)
                    }
                    success_count += 1
                except Exception as e:
                    performance_results[tool_name] = {
                        'status': 'FAILED',
                        'error': str(e)
                    }
                    print(f"❌ {tool_name}: {e}")
        
        # Validação final
        success_rate = (success_count / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📊 RESUMO DE PERFORMANCE:")
        print(f"   ✅ Sucessos: {success_count}/{total_tests}")
        print(f"   📈 Taxa de sucesso: {success_rate:.1f}%")
        print(f"   ⏱️ Resultados por ferramenta:")
        
        for tool_name, result in performance_results.items():
            if result['status'] == 'SUCCESS':
                print(f"     {tool_name}: {result['execution_time']}s")
            else:
                print(f"     {tool_name}: FALHOU")
        
        # Aceitar 70% de sucesso como satisfatório
        assert success_rate >= 70, f"Performance insatisfatória: {success_rate:.1f}%"
        
        return {
            'success_count': success_count,
            'total_tests': total_tests,
            'success_rate': success_rate,
            'performance_results': performance_results
        }

def run_performance_tests():
    """Função principal para executar todos os testes de performance"""
    print("⚡ INICIANDO TESTES DE PERFORMANCE")
    print("=" * 40)
    
    test_suite = TestPerformance()
    test_suite.setup_method()
    
    try:
        result = test_suite.test_performance_summary()
        print(f"\n🎉 TESTES DE PERFORMANCE CONCLUÍDOS!")
        return result
    except Exception as e:
        print(f"\n❌ ERRO NOS TESTES DE PERFORMANCE: {e}")
        raise
    finally:
        test_suite.teardown_method()

if __name__ == "__main__":
    run_performance_tests()
