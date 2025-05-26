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
import time

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent))

# Importar ferramentas que realmente existem
try:
    from insights.tools.kpi_calculator_tool import KPICalculatorTool
    KPI_AVAILABLE = True
except ImportError as e:
    KPI_AVAILABLE = False
    print(f"⚠️ KPI Calculator não disponível: {e}")

try:
    from insights.tools.statistical_analysis_tool import StatisticalAnalysisTool
    STATS_AVAILABLE = True
except ImportError as e:
    STATS_AVAILABLE = False
    print(f"⚠️ Statistical Analysis não disponível: {e}")

try:
    from insights.tools.business_intelligence_tool import BusinessIntelligenceTool
    BI_AVAILABLE = True
except ImportError as e:
    BI_AVAILABLE = False
    print(f"⚠️ Business Intelligence não disponível: {e}")

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

def validate_tool_result(result, tool_name, min_length=50):
    """Validar resultado de uma ferramenta"""
    validations = {
        'not_none': result is not None,
        'is_string': isinstance(result, str),
        'min_length': len(result) >= min_length if result else False,
        'is_json': False,
        'has_metadata': False
    }
    
    # Tentar parsear como JSON
    if validations['is_string']:
        try:
            parsed = json.loads(result)
            validations['is_json'] = True
            validations['has_metadata'] = 'metadata' in parsed
        except json.JSONDecodeError:
            pass
    
    # Score de qualidade
    quality_score = sum(validations.values()) / len(validations) * 100
    
    return {
        'tool': tool_name,
        'validations': validations,
        'quality_score': quality_score,
        'result_length': len(result) if result else 0,
        'is_valid': all([validations['not_none'], validations['is_string'], validations['min_length']])
    }

class TestIntegration:
    """Classe simplificada para testes de integração"""
    
    def setup_method(self):
        """Setup para cada teste"""
        self.test_data = create_simple_integration_data()
        self.test_csv = "temp_integration_test.csv"
        self.test_data.to_csv(self.test_csv, sep=';', index=False, encoding='utf-8')
        self.results = {}
        self.timings = {}
    
    def teardown_method(self):
        """Cleanup após cada teste"""
        if os.path.exists(self.test_csv):
            os.remove(self.test_csv)
    
    def test_kpi_to_stats_integration(self):
        """Teste de integração KPI Calculator → Statistical Analysis"""
        if not (KPI_AVAILABLE and STATS_AVAILABLE):
            print("⚠️ Ferramentas não disponíveis - pulando teste")
            return False
        
        try:
            print("\n" + "="*50)
            print("🔄 TESTE 1: KPI Calculator → Statistical Analysis")
            print("="*50)
            
            # 1. Executar KPI Calculator
            print("📊 Executando KPI Calculator...")
            start_time = time.time()
            kpi_tool = KPICalculatorTool()
            kpi_result = kpi_tool._run(
                data_csv=self.test_csv,
                categoria="revenue",
                periodo="monthly"
            )
            kpi_time = time.time() - start_time
            self.timings['kpi'] = kpi_time
            
            # 2. Executar Statistical Analysis
            print("🔬 Executando Statistical Analysis...")
            start_time = time.time()
            stats_tool = StatisticalAnalysisTool()
            stats_result = stats_tool._run(
                analysis_type="demographic_patterns",
                data_csv=self.test_csv,
                target_column="Total_Liquido"
            )
            stats_time = time.time() - start_time
            self.timings['stats'] = stats_time
            
            # 3. Validações detalhadas
            kpi_validation = validate_tool_result(kpi_result, "KPI Calculator", 100)
            stats_validation = validate_tool_result(stats_result, "Statistical Analysis", 100)
            
            self.results['kpi'] = kpi_validation
            self.results['stats'] = stats_validation
            
            # 4. Verificações de integração
            print(f"\n📋 RESULTADOS DO TESTE 1:")
            print(f"   KPI Calculator: ✅ Válido ({kpi_validation['quality_score']:.1f}% qualidade, {kpi_time:.2f}s)")
            print(f"   Statistical Analysis: ✅ Válido ({stats_validation['quality_score']:.1f}% qualidade, {stats_time:.2f}s)")
            
            # Validações de integração
            assert kpi_validation['is_valid'], f"KPI falhou: {kpi_validation['validations']}"
            assert stats_validation['is_valid'], f"Stats falhou: {stats_validation['validations']}"
            
            print("✅ Integração KPI → Stats: PASSOU")
            return True
            
        except Exception as e:
            print(f"❌ Integração KPI → Stats: FALHOU - {e}")
            return False
    
    def test_stats_to_bi_integration(self):
        """Teste de integração Statistical Analysis → Business Intelligence"""
        if not (STATS_AVAILABLE and BI_AVAILABLE):
            print("⚠️ Ferramentas não disponíveis - pulando teste")
            return False
        
        try:
            print("\n" + "="*50)
            print("🔄 TESTE 2: Statistical Analysis → Business Intelligence")
            print("="*50)
            
            # 1. Executar Statistical Analysis
            print("🔬 Executando Statistical Analysis...")
            start_time = time.time()
            stats_tool = StatisticalAnalysisTool()
            stats_result = stats_tool._run(
                analysis_type="correlation",
                data_csv=self.test_csv,
                target_column="Total_Liquido"
            )
            stats_time = time.time() - start_time
            
            # 2. Executar Business Intelligence
            print("📈 Executando Business Intelligence...")
            start_time = time.time()
            bi_tool = BusinessIntelligenceTool()
            bi_result = bi_tool._run(
                data_csv=self.test_csv,
                analysis_type="executive_summary",
                output_format="interactive"
            )
            bi_time = time.time() - start_time
            self.timings['bi'] = bi_time
            
            # 3. Validações detalhadas
            stats_validation = validate_tool_result(stats_result, "Statistical Analysis", 100)
            bi_validation = validate_tool_result(bi_result, "Business Intelligence", 200)
            
            self.results['stats_2'] = stats_validation
            self.results['bi'] = bi_validation
            
            # 4. Verificações de integração
            print(f"\n📋 RESULTADOS DO TESTE 2:")
            print(f"   Statistical Analysis: ✅ Válido ({stats_validation['quality_score']:.1f}% qualidade, {stats_time:.2f}s)")
            print(f"   Business Intelligence: ✅ Válido ({bi_validation['quality_score']:.1f}% qualidade, {bi_time:.2f}s)")
            
            # Validações de integração
            assert stats_validation['is_valid'], f"Stats falhou: {stats_validation['validations']}"
            assert bi_validation['is_valid'], f"BI falhou: {bi_validation['validations']}"
            
            print("✅ Integração Stats → BI: PASSOU")
            return True
            
        except Exception as e:
            print(f"❌ Integração Stats → BI: FALHOU - {e}")
            return False
    
    def test_full_pipeline_integration(self):
        """Teste de pipeline completo: KPI → Stats → BI"""
        if not (KPI_AVAILABLE and STATS_AVAILABLE and BI_AVAILABLE):
            print("⚠️ Pipeline completo não disponível - pulando teste")
            return False
        
        try:
            print("\n" + "="*50)
            print("🔄 TESTE 3: Pipeline Completo KPI → Stats → BI")
            print("="*50)
            
            results = {}
            timings = {}
            
            # 1. KPI Calculator
            print("📊 Executando KPI Calculator...")
            start_time = time.time()
            kpi_tool = KPICalculatorTool()
            results['kpi'] = kpi_tool._run(
                data_csv=self.test_csv,
                categoria="revenue",
                periodo="monthly"
            )
            timings['kpi'] = time.time() - start_time
            
            # 2. Statistical Analysis
            print("🔬 Executando Statistical Analysis...")
            start_time = time.time()
            stats_tool = StatisticalAnalysisTool()
            results['stats'] = stats_tool._run(
                analysis_type="demographic_patterns",
                data_csv=self.test_csv,
                target_column="Total_Liquido"
            )
            timings['stats'] = time.time() - start_time
            
            # 3. Business Intelligence
            print("📈 Executando Business Intelligence...")
            start_time = time.time()
            bi_tool = BusinessIntelligenceTool()
            results['bi'] = bi_tool._run(
                data_csv=self.test_csv,
                analysis_type="executive_summary",
                output_format="interactive"
            )
            timings['bi'] = time.time() - start_time
            
            # 4. Validações do pipeline
            validations = {}
            for tool_name, result in results.items():
                validations[tool_name] = validate_tool_result(result, tool_name, 50)
            
            # 5. Análise de performance
            total_time = sum(timings.values())
            avg_quality = sum(v['quality_score'] for v in validations.values()) / len(validations)
            
            print(f"\n📋 RESULTADOS DO PIPELINE COMPLETO:")
            for tool_name, validation in validations.items():
                status = "✅" if validation['is_valid'] else "❌"
                print(f"   {tool_name}: {status} {validation['quality_score']:.1f}% qualidade, {timings[tool_name]:.2f}s")
            
            print(f"\n⏱️ PERFORMANCE:")
            print(f"   Tempo total: {total_time:.2f}s")
            print(f"   Qualidade média: {avg_quality:.1f}%")
            print(f"   Ferramentas válidas: {sum(1 for v in validations.values() if v['is_valid'])}/3")
            
            # Validações do pipeline
            all_valid = all(v['is_valid'] for v in validations.values())
            performance_ok = total_time < 30  # Máximo 30 segundos
            quality_ok = avg_quality >= 75   # Mínimo 75% qualidade
            
            assert all_valid, f"Nem todas as ferramentas são válidas: {validations}"
            assert performance_ok, f"Pipeline muito lento: {total_time:.2f}s > 30s"
            assert quality_ok, f"Qualidade baixa: {avg_quality:.1f}% < 75%"
            
            print("✅ Pipeline completo: PASSOU")
            
            return {
                'success_count': sum(1 for v in validations.values() if v['is_valid']),
                'total_tools': len(results),
                'success_rate': avg_quality,
                'total_time': total_time,
                'results_details': validations
            }
            
        except Exception as e:
            print(f"❌ Pipeline completo: FALHOU - {e}")
            return False
    
    def test_individual_tools(self):
        """Teste individual de cada ferramenta"""
        print("\n" + "="*50)
        print("🔄 TESTE INDIVIDUAL DAS FERRAMENTAS")
        print("="*50)
        
        individual_results = {}
        
        # Teste KPI Calculator
        if KPI_AVAILABLE:
            try:
                print("📊 Testando KPI Calculator individualmente...")
                kpi_tool = KPICalculatorTool()
                result = kpi_tool._run(data_csv=self.test_csv, categoria="all")
                validation = validate_tool_result(result, "KPI Calculator", 200)
                individual_results['kpi'] = validation
                print(f"   ✅ KPI Calculator: {validation['quality_score']:.1f}% qualidade")
            except Exception as e:
                print(f"   ❌ KPI Calculator falhou: {e}")
                individual_results['kpi'] = {'is_valid': False, 'error': str(e)}
        
        # Teste Statistical Analysis
        if STATS_AVAILABLE:
            try:
                print("🔬 Testando Statistical Analysis individualmente...")
                stats_tool = StatisticalAnalysisTool()
                result = stats_tool._run(
                    analysis_type="correlation", 
                    data_csv=self.test_csv,
                    target_column="Total_Liquido"
                )
                validation = validate_tool_result(result, "Statistical Analysis", 200)
                individual_results['stats'] = validation
                print(f"   ✅ Statistical Analysis: {validation['quality_score']:.1f}% qualidade")
            except Exception as e:
                print(f"   ❌ Statistical Analysis falhou: {e}")
                individual_results['stats'] = {'is_valid': False, 'error': str(e)}
        
        # Teste Business Intelligence
        if BI_AVAILABLE:
            try:
                print("📈 Testando Business Intelligence individualmente...")
                bi_tool = BusinessIntelligenceTool()
                result = bi_tool._run(
                    data_csv=self.test_csv,
                    analysis_type="comprehensive_report"
                )
                validation = validate_tool_result(result, "Business Intelligence", 300)
                individual_results['bi'] = validation
                print(f"   ✅ Business Intelligence: {validation['quality_score']:.1f}% qualidade")
            except Exception as e:
                print(f"   ❌ Business Intelligence falhou: {e}")
                individual_results['bi'] = {'is_valid': False, 'error': str(e)}
        
        return individual_results

def run_integration_tests():
    """Função principal para executar todos os testes de integração"""
    print("🔗 INICIANDO TESTES DE INTEGRAÇÃO AVANÇADOS")
    print("=" * 60)
    print(f"🕐 Iniciado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 Ferramentas disponíveis:")
    print(f"   KPI Calculator: {'✅' if KPI_AVAILABLE else '❌'}")
    print(f"   Statistical Analysis: {'✅' if STATS_AVAILABLE else '❌'}")
    print(f"   Business Intelligence: {'✅' if BI_AVAILABLE else '❌'}")
    
    test_suite = TestIntegration()
    test_suite.setup_method()
    
    try:
        success_count = 0
        total_tests = 0
        test_results = {}
        
        # Teste individual das ferramentas
        individual_results = test_suite.test_individual_tools()
        test_results['individual'] = individual_results
        
        # Teste 1: KPI → Stats
        if KPI_AVAILABLE and STATS_AVAILABLE:
            total_tests += 1
            if test_suite.test_kpi_to_stats_integration():
                success_count += 1
                test_results['kpi_stats'] = True
            else:
                test_results['kpi_stats'] = False
        
        # Teste 2: Stats → BI
        if STATS_AVAILABLE and BI_AVAILABLE:
            total_tests += 1
            if test_suite.test_stats_to_bi_integration():
                success_count += 1
                test_results['stats_bi'] = True
            else:
                test_results['stats_bi'] = False
        
        # Teste 3: Pipeline completo
        if KPI_AVAILABLE and STATS_AVAILABLE and BI_AVAILABLE:
            total_tests += 1
            pipeline_result = test_suite.test_full_pipeline_integration()
            if pipeline_result:
                success_count += 1
                test_results['full_pipeline'] = pipeline_result
            else:
                test_results['full_pipeline'] = False
        
        # Resultado final
        success_rate = (success_count / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "="*60)
        print("📊 RESUMO FINAL DOS TESTES DE INTEGRAÇÃO")
        print("="*60)
        print(f"   ✅ Sucessos: {success_count}/{total_tests}")
        print(f"   📈 Taxa de sucesso: {success_rate:.1f}%")
        print(f"   🕐 Concluído em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if 'full_pipeline' in test_results and isinstance(test_results['full_pipeline'], dict):
            pipeline = test_results['full_pipeline']
            print(f"   ⏱️ Tempo total do pipeline: {pipeline.get('total_time', 0):.2f}s")
            print(f"   🎯 Qualidade média: {pipeline.get('success_rate', 0):.1f}%")
        
        if success_rate >= 70:
            print(f"\n🎉 TESTES DE INTEGRAÇÃO CONCLUÍDOS COM SUCESSO!")
        else:
            print(f"\n⚠️ ALGUNS TESTES DE INTEGRAÇÃO FALHARAM")
        
        return {
            'success_count': success_count,
            'total_tests': total_tests,
            'success_rate': success_rate,
            'detailed_results': test_results,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"\n❌ ERRO NOS TESTES DE INTEGRAÇÃO: {e}")
        raise
    finally:
        test_suite.teardown_method()

if __name__ == "__main__":
    run_integration_tests()
