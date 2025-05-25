"""
🔗 TESTE: INTEGRAÇÃO ENTRE FERRAMENTAS (SIMPLIFICADO)
====================================================

Testa a integração básica entre as ferramentas v3.0 que realmente existem.
Versão simplificada focada em funcionalidade core.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import tempfile

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent))

# Importar ferramentas v3.0 que realmente existem
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
    from insights.tools.unified_business_intelligence import UnifiedBusinessIntelligence
    UBI_AVAILABLE = True
except ImportError:
    UBI_AVAILABLE = False

def create_simple_integration_data():
    """Criar dados simples para testes de integração"""
    np.random.seed(42)
    
    dates = pd.date_range('2024-01-01', periods=90, freq='D')
    data = []
    
    for i, date in enumerate(dates):
        data.append({
            'Data': date.strftime('%Y-%m-%d'),
            'Codigo_Cliente': f"CLI_{(i % 10) + 1:03d}",
            'Codigo_Produto': f"PROD_{(i % 5) + 1:03d}",
            'Descricao_Produto': f"Produto {(i % 5) + 1}",
            'Categoria': np.random.choice(['Anéis', 'Brincos', 'Colares']),
            'Quantidade': np.random.randint(1, 4),
            'Total_Liquido': np.random.uniform(100, 2000),
            'Preco_Unitario': np.random.uniform(50, 500)
        })
    
    return pd.DataFrame(data)

class TestIntegration:
    """Classe simplificada para testes de integração"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.test_data = create_simple_integration_data()
        self.test_csv = "temp_integration_test.csv"
        self.test_data.to_csv(self.test_csv, sep=';', index=False, encoding='utf-8')
    
    def teardown_method(self):
        """Cleanup após cada teste"""
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)
    
    def test_kpi_to_stats_integration(self):
        """Teste de integração KPI Calculator v3 → Statistical Analysis v3"""
        if not (KPI_V3_AVAILABLE and STATS_V3_AVAILABLE):
            print("⚠️ Ferramentas v3.0 não disponíveis - pulando teste")
            return
        
        try:
            # 1. Executar KPI Calculator v3
            kpi_tool = KPICalculatorToolV3()
            kpi_result = kpi_tool._run(
                data_csv=self.test_csv,
                categoria="revenue",
                periodo="monthly"
            )
            
            # 2. Usar os mesmos dados para Statistical Analysis v3
            stats_tool = StatisticalAnalysisToolV3()
            stats_result = stats_tool._run(
                analysis_type="demographic_patterns",
                data_csv=self.test_csv,
                target_column="Total_Liquido"
            )
            
            # Validações
            assert kpi_result is not None, "KPI v3 retornou None"
            assert stats_result is not None, "Stats v3 retornou None"
            assert isinstance(kpi_result, str), "KPI resultado deve ser string"
            assert isinstance(stats_result, str), "Stats resultado deve ser string"
            assert len(kpi_result) > 50, "KPI resultado muito curto"
            assert len(stats_result) > 50, "Stats resultado muito curto"
            
            print("✅ Integração KPI v3 → Stats v3: PASSOU")
            
        except Exception as e:
            print(f"❌ Integração KPI v3 → Stats v3: FALHOU - {e}")
            raise
    
    def test_stats_to_ubi_integration(self):
        """Teste de integração Statistical Analysis v3 → Unified BI"""
        if not (STATS_V3_AVAILABLE and UBI_AVAILABLE):
            print("⚠️ Ferramentas v3.0 não disponíveis - pulando teste")
            return
        
        try:
            # 1. Executar Statistical Analysis v3
            stats_tool = StatisticalAnalysisToolV3()
            stats_result = stats_tool._run(
                analysis_type="correlation",
                data_csv=self.test_csv,
                target_column="Total_Liquido"
            )
            
            # 2. Usar os mesmos dados para Unified BI
            ubi_tool = UnifiedBusinessIntelligence()
            ubi_result = ubi_tool._run(
                data_csv=self.test_csv,
                analysis_type="executive_summary",
                output_format="interactive"
            )
            
            # Validações
            assert stats_result is not None, "Stats v3 retornou None"
            assert ubi_result is not None, "UBI retornou None"
            assert isinstance(stats_result, str), "Stats resultado deve ser string"
            assert isinstance(ubi_result, str), "UBI resultado deve ser string"
            assert len(stats_result) > 50, "Stats resultado muito curto"
            assert len(ubi_result) > 100, "UBI resultado muito curto"
            
            print("✅ Integração Stats v3 → UBI: PASSOU")
            
        except Exception as e:
            print(f"❌ Integração Stats v3 → UBI: FALHOU - {e}")
            raise
    
    def test_full_pipeline_integration(self):
        """Teste de pipeline completo: KPI v3 → Stats v3 → UBI"""
        if not (KPI_V3_AVAILABLE and STATS_V3_AVAILABLE and UBI_AVAILABLE):
            print("⚠️ Pipeline completo não disponível - pulando teste")
            return
        
        try:
            results = {}
            
            # 1. KPI Calculator v3
            kpi_tool = KPICalculatorToolV3()
            results['kpi'] = kpi_tool._run(
                data_csv=self.test_csv,
                categoria="revenue",
                periodo="monthly"
            )
            
            # 2. Statistical Analysis v3
            stats_tool = StatisticalAnalysisToolV3()
            results['stats'] = stats_tool._run(
                analysis_type="demographic_patterns",
                data_csv=self.test_csv,
                target_column="Total_Liquido"
            )
            
            # 3. Unified Business Intelligence
            ubi_tool = UnifiedBusinessIntelligence()
            results['ubi'] = ubi_tool._run(
                data_csv=self.test_csv,
                analysis_type="executive_summary",
                output_format="interactive"
            )
            
            # Validações do pipeline
            for tool_name, result in results.items():
                assert result is not None, f"{tool_name} retornou None"
                assert isinstance(result, str), f"{tool_name} resultado deve ser string"
                assert len(result) > 30, f"{tool_name} resultado muito curto"
            
            # Verificar se todos os resultados são consistentes
            success_count = len([r for r in results.values() if r and len(r) > 30])
            success_rate = (success_count / len(results)) * 100
            
            assert success_rate >= 100, f"Pipeline incompleto: {success_rate:.1f}% de sucesso"
            
            print(f"✅ Pipeline completo: PASSOU ({success_count}/3 ferramentas)")
            
            return {
                'success_count': success_count,
                'total_tools': len(results),
                'success_rate': success_rate,
                'results_lengths': {k: len(v) if v else 0 for k, v in results.items()}
            }
            
        except Exception as e:
            print(f"❌ Pipeline completo: FALHOU - {e}")
            raise

def run_integration_tests():
    """Função principal para executar todos os testes de integração"""
    print("🔗 INICIANDO TESTES DE INTEGRAÇÃO")
    print("=" * 40)
    
    test_suite = TestIntegration()
    test_suite.setup_method()
    
    try:
        success_count = 0
        total_tests = 0
        
        # Teste 1: KPI → Stats
        if KPI_V3_AVAILABLE and STATS_V3_AVAILABLE:
            total_tests += 1
            try:
                test_suite.test_kpi_to_stats_integration()
                success_count += 1
            except Exception as e:
                print(f"❌ Teste 1 falhou: {e}")
        
        # Teste 2: Stats → UBI
        if STATS_V3_AVAILABLE and UBI_AVAILABLE:
            total_tests += 1
            try:
                test_suite.test_stats_to_ubi_integration()
                success_count += 1
            except Exception as e:
                print(f"❌ Teste 2 falhou: {e}")
        
        # Teste 3: Pipeline completo
        if KPI_V3_AVAILABLE and STATS_V3_AVAILABLE and UBI_AVAILABLE:
            total_tests += 1
            try:
                result = test_suite.test_full_pipeline_integration()
                success_count += 1
            except Exception as e:
                print(f"❌ Teste 3 falhou: {e}")
        
        # Resultado final
        success_rate = (success_count / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📊 RESUMO DOS TESTES DE INTEGRAÇÃO:")
        print(f"   ✅ Sucessos: {success_count}/{total_tests}")
        print(f"   📈 Taxa de sucesso: {success_rate:.1f}%")
        
        if success_rate >= 70:
            print(f"\n🎉 TESTES DE INTEGRAÇÃO CONCLUÍDOS COM SUCESSO!")
        else:
            print(f"\n⚠️ ALGUNS TESTES DE INTEGRAÇÃO FALHARAM")
        
        return {
            'success_count': success_count,
            'total_tests': total_tests,
            'success_rate': success_rate
        }
        
    except Exception as e:
        print(f"\n❌ ERRO NOS TESTES DE INTEGRAÇÃO: {e}")
        raise
    finally:
        test_suite.teardown_method()

if __name__ == "__main__":
    run_integration_tests()
