"""
🧪 TESTE SIMPLIFICADO PARA KPI CALCULATOR TOOL V3.0
====================================================

Suite de testes simplificada que funciona sem dependências externas como psutil.
Focada em validar a funcionalidade core do KPI Calculator Tool V3.0.
"""

import pytest
import pandas as pd
import numpy as np
import time
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import tracemalloc

# Importar ferramenta a ser testada
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.insights.tools.kpi_calculator_tool import KPICalculatorTool


class TestKPICalculatorToolSimple:
    """
    Suite simplificada de testes para KPI Calculator Tool v3.0
    
    Focada em validação funcional sem dependências externas.
    """
    
    @pytest.fixture(autouse=True)
    def setup(self, real_vendas_data):
        """Setup automático para cada teste."""
        self.kpi_tool = KPICalculatorTool()
        self.real_data_path = real_vendas_data
        self.test_logs = []
        self.start_time = time.time()
        
        print(f"🚀 Iniciando teste com dados: {self.real_data_path}")
    
    def setup_standalone(self, data_path):
        """Setup para execução standalone."""
        self.kpi_tool = KPICalculatorTool()
        self.real_data_path = data_path
        self.test_logs = []
        self.start_time = time.time()
        
        print(f"🚀 Iniciando teste com dados: {self.real_data_path}")
    
    def log_test(self, level: str, message: str, **kwargs):
        """Logging simplificado para testes."""
        elapsed = time.time() - self.start_time
        log_entry = {
            'elapsed': round(elapsed, 2),
            'level': level,
            'message': message,
            **kwargs
        }
        self.test_logs.append(log_entry)
        print(f"[{elapsed:6.2f}s] [{level}] {message}")
        if kwargs:
            print(f"    {kwargs}")
    
    def test_financial_kpis_basic(self):
        """
        Teste básico dos KPIs financeiros.
        """
        self.log_test("INFO", "Iniciando teste de KPIs financeiros")
        
        # Medir performance básica
        start_time = time.time()
        tracemalloc.start()
        
        result = self.kpi_tool._run(
            data_csv=self.real_data_path,
            categoria="revenue",
            periodo="monthly",
            benchmark_mode=True,
            cache_data=True
        )
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Validações básicas
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 100, "Resultado muito curto"
        assert "erro" not in result.lower(), f"Erro detectado: {result[:200]}..."
        
        # Validações de conteúdo (aceita termos em português e inglês)
        financial_terms = [
            "receita", "revenue", "aov", "margem", "margin", 
            "crescimento", "growth", "kpi", "total", "vendas"
        ]
        found_terms = [term for term in financial_terms if term in result.lower()]
        assert len(found_terms) >= 3, f"Poucos termos financeiros encontrados: {found_terms}"
        
        # Validações de formato
        assert "R$" in result or "real" in result.lower(), "Deve incluir valores monetários"
        
        self.log_test("SUCCESS", "KPIs financeiros validados", 
                     execution_time=f"{execution_time:.2f}s",
                     memory_peak=f"{peak/1024/1024:.1f}MB",
                     terms_found=len(found_terms),
                     result_length=len(result))
        
        return result
    
    def test_all_categories_basic(self):
        """
        Teste básico de todas as categorias de KPI.
        """
        categories = ["revenue", "operational", "inventory", "customer", "products"]
        results = {}
        
        self.log_test("INFO", f"Testando {len(categories)} categorias")
        
        for category in categories:
            start_time = time.time()
            
            try:
                result = self.kpi_tool._run(
                    data_csv=self.real_data_path,
                    categoria=category,
                    periodo="monthly",
                    benchmark_mode=False,  # Acelerar
                    cache_data=True
                )
                
                execution_time = time.time() - start_time
                success = "erro" not in result.lower() and len(result) > 50
                
                results[category] = {
                    'success': success,
                    'execution_time': round(execution_time, 2),
                    'output_length': len(result)
                }
                
                self.log_test("SUCCESS" if success else "ERROR", 
                             f"Categoria {category}",
                             **results[category])
                
            except Exception as e:
                results[category] = {
                    'success': False,
                    'error': str(e),
                    'execution_time': time.time() - start_time
                }
                self.log_test("ERROR", f"Erro em {category}: {str(e)}")
        
        # Validações
        successful = [cat for cat, res in results.items() if res['success']]
        success_rate = len(successful) / len(categories)
        
        assert success_rate >= 0.8, f"Taxa de sucesso baixa: {success_rate:.1%}"
        
        self.log_test("SUCCESS", "Teste de categorias concluído",
                     success_rate=f"{success_rate:.1%}",
                     successful_categories=successful)
        
        return results
    
    def test_cache_functionality(self):
        """
        Teste básico da funcionalidade de cache.
        """
        self.log_test("INFO", "Testando funcionalidade de cache")
        
        # Limpar cache
        self.kpi_tool._data_cache.clear()
        
        # Primeira execução
        start_time = time.time()
        result1 = self.kpi_tool._run(
            data_csv=self.real_data_path,
            categoria="revenue",
            cache_data=True
        )
        first_time = time.time() - start_time
        
        # Verificar se cache foi populado
        cache_populated = len(self.kpi_tool._data_cache) > 0
        assert cache_populated, "Cache não foi populado"
        
        # Segunda execução (com cache)
        start_time = time.time()
        result2 = self.kpi_tool._run(
            data_csv=self.real_data_path,
            categoria="revenue",
            cache_data=True
        )
        second_time = time.time() - start_time
        
        # Validações
        assert result1 == result2, "Resultados com cache diferem"
        cache_benefit = first_time > second_time
        
        self.log_test("SUCCESS", "Cache funcionando",
                     first_time=f"{first_time:.2f}s",
                     second_time=f"{second_time:.2f}s",
                     cache_benefit=cache_benefit)
        
        return {
            'first_execution': first_time,
            'cached_execution': second_time,
            'cache_benefit': cache_benefit
        }
    
    def test_benchmark_functionality(self):
        """
        Teste da funcionalidade de benchmarks.
        """
        self.log_test("INFO", "Testando funcionalidade de benchmarks")
        
        result = self.kpi_tool._run(
            data_csv=self.real_data_path,
            categoria="all",
            benchmark_mode=True
        )
        
        # Verificar elementos de benchmark (aceita termos em português e inglês)
        benchmark_terms = [
            "benchmark", "setor", "sector", "comparação", "comparison", 
            "média", "average", "padrão", "standard"
        ]
        found_terms = [term for term in benchmark_terms if term in result.lower()]
        
        assert len(found_terms) >= 2, f"Poucos termos de benchmark: {found_terms}"
        
        self.log_test("SUCCESS", "Benchmarks validados",
                     terms_found=len(found_terms))
    
    def test_error_handling_basic(self):
        """
        Teste básico de tratamento de erros.
        """
        self.log_test("INFO", "Testando tratamento de erros")
        
        error_tests = []
        
        # Arquivo inexistente
        try:
            result = self.kpi_tool._run(data_csv="arquivo_inexistente.csv")
            handled = "erro" in result.lower() or "error" in result.lower()
            error_tests.append(('arquivo_inexistente', handled))
        except Exception:
            error_tests.append(('arquivo_inexistente', False))
        
        # Categoria inválida (deve funcionar com fallback)
        try:
            result = self.kpi_tool._run(
                data_csv=self.real_data_path,
                categoria="categoria_invalida"
            )
            handled = len(result) > 50  # Deve gerar algum resultado
            error_tests.append(('categoria_invalida', handled))
        except Exception:
            error_tests.append(('categoria_invalida', False))
        
        # Pelo menos metade dos testes de erro deve passar
        passed = sum(1 for _, handled in error_tests if handled)
        assert passed >= len(error_tests) // 2, f"Poucos erros tratados: {error_tests}"
        
        self.log_test("SUCCESS", "Tratamento de erros validado",
                     tests_passed=f"{passed}/{len(error_tests)}")
    
    def test_data_quality_basic(self):
        """
        Teste básico de qualidade dos dados.
        """
        self.log_test("INFO", "Validando qualidade básica dos dados")
        
        # Verificar se arquivo existe e não está vazio
        assert os.path.exists(self.real_data_path), "Arquivo de dados não encontrado"
        
        file_size = os.path.getsize(self.real_data_path)
        assert file_size > 1000, f"Arquivo muito pequeno: {file_size} bytes"
        
        # Tentar carregar dados
        try:
            df = pd.read_csv(self.real_data_path, sep=';', encoding='utf-8', nrows=100)
            assert len(df) > 0, "DataFrame vazio"
            assert len(df.columns) >= 5, f"Poucas colunas: {len(df.columns)}"
            
            # Verificar colunas essenciais
            essential_cols = ['Data', 'Total_Liquido']
            missing = [col for col in essential_cols if col not in df.columns]
            assert len(missing) == 0, f"Colunas essenciais faltando: {missing}"
            
        except Exception as e:
            pytest.fail(f"Erro ao validar dados: {str(e)}")
        
        self.log_test("SUCCESS", "Qualidade básica dos dados validada",
                     file_size_mb=f"{file_size/1024/1024:.1f}MB",
                     sample_rows=len(df),
                     columns=len(df.columns))
    
    def test_performance_basic(self):
        """
        Teste básico de performance.
        """
        self.log_test("INFO", "Testando performance básica")
        
        # Medir execução de categoria revenue
        start_time = time.time()
        tracemalloc.start()
        
        result = self.kpi_tool._run(
            data_csv=self.real_data_path,
            categoria="revenue",
            periodo="monthly",
            benchmark_mode=False
        )
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Validações de performance
        assert execution_time < 30, f"Execução muito lenta: {execution_time:.2f}s"
        assert peak < 500 * 1024 * 1024, f"Uso de memória excessivo: {peak/1024/1024:.1f}MB"
        assert len(result) > 1000, "Resultado muito curto"
        
        self.log_test("SUCCESS", "Performance validada",
                     execution_time=f"{execution_time:.2f}s",
                     memory_peak=f"{peak/1024/1024:.1f}MB",
                     result_length=len(result))
    
    def teardown_method(self, method):
        """Limpeza após cada teste."""
        test_name = method.__name__
        duration = time.time() - self.start_time
        
        # Salvar logs do teste
        log_dir = Path("test_logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{test_name}_{timestamp}_simple.json"
        
        log_data = {
            'test_name': test_name,
            'timestamp': timestamp,
            'duration': round(duration, 2),
            'logs': self.test_logs
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Log salvo: {log_file}")
        
        # Limpar cache
        if hasattr(self, 'kpi_tool'):
            self.kpi_tool._data_cache.clear()


if __name__ == "__main__":
    # Executar teste standalone
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    test_instance = TestKPICalculatorToolSimple()
    
    # Setup standalone
    test_instance.setup_standalone("data/vendas.csv")
    
    print("🧪 Executando testes simplificados...")
    
    tests = [
        test_instance.test_data_quality_basic,
        test_instance.test_financial_kpis_basic,
        test_instance.test_cache_functionality,
        test_instance.test_error_handling_basic,
        test_instance.test_performance_basic
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            print(f"\n{'='*50}")
            print(f"🔄 Executando: {test_func.__name__}")
            test_func()
            print(f"✅ {test_func.__name__} - PASSOU")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} - FALHOU: {str(e)}")
        finally:
            test_instance.teardown_method(test_func)
    
    print(f"\n{'='*50}")
    print(f"🎯 RESULTADO FINAL: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 TODOS OS TESTES PASSARAM!")
    else:
        print(f"⚠️ {total - passed} teste(s) falharam") 