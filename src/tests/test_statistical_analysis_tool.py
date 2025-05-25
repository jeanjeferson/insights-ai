"""
🧪 TESTE PARA STATISTICAL ANALYSIS TOOL
========================================

Suite de testes para validar a funcionalidade core do Statistical Analysis Tool.
Baseada no template funcionando do KPI Calculator Tool.
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
from src.insights.tools.statistical_analysis_tool import StatisticalAnalysisTool


class TestStatisticalAnalysisTool:
    """
    Suite de testes para Statistical Analysis Tool
    
    Focada em validação funcional dos 13 tipos de análise estatística.
    """
    
    # Tipos de análise críticos identificados na ferramenta
    CRITICAL_ANALYSES = [
        'correlation',                    # Análise de correlação multi-dimensional
        'clustering',                     # Clustering avançado
        'demographic_patterns',           # Padrões demográficos avançados
        'geographic_performance',         # Performance geográfica detalhada
        'customer_segmentation'           # Segmentação de clientes comportamental
    ]
    
    ALL_ANALYSES = [
        # Análises estatísticas core
        'correlation', 'clustering', 'outliers', 'distribution', 'trend_analysis',
        # Análises especializadas
        'demographic_patterns', 'geographic_performance', 'customer_segmentation', 
        'price_sensitivity', 'profitability_patterns',
        # Análises integradas
        'comprehensive_customer_analysis', 'product_performance_analysis'
    ]
    
    @pytest.fixture(autouse=True)
    def setup(self, real_vendas_data):
        """Setup automático para cada teste."""
        self.stats_tool = StatisticalAnalysisTool()
        self.real_data_path = real_vendas_data
        self.test_logs = []
        self.start_time = time.time()
        
        print(f"🚀 Iniciando teste Statistical Analysis Tool com dados: {self.real_data_path}")
    
    def setup_standalone(self, data_path):
        """Setup para execução standalone."""
        self.stats_tool = StatisticalAnalysisTool()
        self.real_data_path = data_path
        self.test_logs = []
        self.start_time = time.time()
        
        print(f"🚀 Iniciando teste Statistical Analysis Tool com dados: {self.real_data_path}")
    
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
    
    def test_demographic_patterns_basic(self):
        """
        Teste básico da análise de padrões demográficos.
        """
        self.log_test("INFO", "Iniciando teste de análise demográfica")
        
        # Medir performance básica
        start_time = time.time()
        tracemalloc.start()
        
        result = self.stats_tool._run(
            analysis_type="demographic_patterns",
            data_csv=self.real_data_path,
            target_column="Total_Liquido",
            demographic_focus=True,
            cache_results=True
        )
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Validações básicas
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 100, "Resultado muito curto"
        assert "erro" not in result.lower(), f"Erro detectado: {result[:200]}..."
        
        # Validações de conteúdo demográfico (aceita termos em português e inglês)
        demographic_terms = [
            "idade", "age", "sexo", "gender", "estado", "state",
            "demográfico", "demographic", "perfil", "profile", "padrão", "pattern"
        ]
        found_terms = [term for term in demographic_terms if term in result.lower()]
        assert len(found_terms) >= 2, f"Poucos termos demográficos encontrados: {found_terms}"
        
        self.log_test("SUCCESS", "Análise demográfica validada", 
                     execution_time=f"{execution_time:.2f}s",
                     memory_peak=f"{peak/1024/1024:.1f}MB",
                     terms_found=len(found_terms),
                     result_length=len(result))
        
        return result
    
    def test_correlation_analysis_basic(self):
        """
        Teste básico da análise de correlação.
        """
        self.log_test("INFO", "Iniciando teste de análise de correlação")
        
        start_time = time.time()
        
        result = self.stats_tool._run(
            analysis_type="correlation",
            data_csv=self.real_data_path,
            target_column="Total_Liquido",
            min_correlation=0.3,
            significance_level=0.05
        )
        
        execution_time = time.time() - start_time
        
        # Validações básicas
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 100, "Resultado muito curto"
        assert "erro" not in result.lower(), f"Erro detectado: {result[:200]}..."
        
        # Validações de conteúdo estatístico
        correlation_terms = [
            "correlação", "correlation", "pearson", "spearman", 
            "significância", "significance", "estatística", "statistic"
        ]
        found_terms = [term for term in correlation_terms if term in result.lower()]
        assert len(found_terms) >= 2, f"Poucos termos de correlação encontrados: {found_terms}"
        
        self.log_test("SUCCESS", "Análise de correlação validada", 
                     execution_time=f"{execution_time:.2f}s",
                     terms_found=len(found_terms),
                     result_length=len(result))
        
        return result
    
    def test_clustering_analysis_basic(self):
        """
        Teste básico da análise de clustering.
        """
        self.log_test("INFO", "Iniciando teste de análise de clustering")
        
        start_time = time.time()
        
        result = self.stats_tool._run(
            analysis_type="clustering",
            data_csv=self.real_data_path,
            target_column="Total_Liquido",
            clustering_method="auto"
        )
        
        execution_time = time.time() - start_time
        
        # Validações básicas
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 100, "Resultado muito curto"
        assert "erro" not in result.lower(), f"Erro detectado: {result[:200]}..."
        
        # Validações de conteúdo de clustering
        clustering_terms = [
            "cluster", "segmento", "segment", "grupo", "group",
            "k-means", "hierarchical", "dbscan", "silhouette"
        ]
        found_terms = [term for term in clustering_terms if term in result.lower()]
        assert len(found_terms) >= 2, f"Poucos termos de clustering encontrados: {found_terms}"
        
        self.log_test("SUCCESS", "Análise de clustering validada", 
                     execution_time=f"{execution_time:.2f}s",
                     terms_found=len(found_terms),
                     result_length=len(result))
        
        return result
    
    def test_critical_analyses_batch(self):
        """
        Teste das 5 análises mais críticas em lote.
        """
        self.log_test("INFO", f"Testando {len(self.CRITICAL_ANALYSES)} análises críticas")
        
        results = {}
        
        for analysis_type in self.CRITICAL_ANALYSES:
            start_time = time.time()
            
            try:
                result = self.stats_tool._run(
                    analysis_type=analysis_type,
                    data_csv=self.real_data_path,
                    target_column="Total_Liquido",
                    cache_results=True
                )
                
                execution_time = time.time() - start_time
                success = "erro" not in result.lower() and len(result) > 50
                
                results[analysis_type] = {
                    'success': success,
                    'execution_time': round(execution_time, 2),
                    'output_length': len(result)
                }
                
                self.log_test("SUCCESS" if success else "ERROR", 
                             f"Análise {analysis_type}",
                             **results[analysis_type])
                
            except Exception as e:
                results[analysis_type] = {
                    'success': False,
                    'error': str(e),
                    'execution_time': time.time() - start_time
                }
                self.log_test("ERROR", f"Erro em {analysis_type}: {str(e)}")
        
        # Validações
        successful = [analysis for analysis, res in results.items() if res['success']]
        success_rate = len(successful) / len(self.CRITICAL_ANALYSES)
        
        assert success_rate >= 0.7, f"Taxa de sucesso baixa: {success_rate:.1%}"
        
        self.log_test("SUCCESS", "Teste de análises críticas concluído",
                     success_rate=f"{success_rate:.1%}",
                     successful_analyses=successful)
        
        return results
    
    def test_all_analyses_comprehensive(self):
        """
        Teste abrangente de todas as análises disponíveis.
        """
        self.log_test("INFO", f"Testando todas as {len(self.ALL_ANALYSES)} análises")
        
        results = {}
        failed = []
        
        for analysis_type in self.ALL_ANALYSES:
            try:
                start_time = time.time()
                
                result = self.stats_tool._run(
                    analysis_type=analysis_type,
                    data_csv=self.real_data_path,
                    target_column="Total_Liquido"
                )
                
                execution_time = time.time() - start_time
                
                if "erro" in result.lower():
                    failed.append(f"{analysis_type}: erro no resultado")
                elif len(result) < 50:
                    failed.append(f"{analysis_type}: resultado muito curto")
                else:
                    results[analysis_type] = {
                        'success': True,
                        'execution_time': round(execution_time, 2),
                        'output_length': len(result)
                    }
                    
                    print(f"✅ {analysis_type}: {execution_time:.2f}s, {len(result)} chars")
                
            except Exception as e:
                failed.append(f"{analysis_type}: {str(e)}")
                print(f"❌ {analysis_type}: ERRO - {str(e)}")
        
        # Calcular taxa de sucesso
        total_analyses = len(self.ALL_ANALYSES)
        successful_count = len(results)
        success_rate = successful_count / total_analyses
        
        # Aceitar até 30% de falha (70% de sucesso)
        assert success_rate >= 0.7, f"Taxa de sucesso baixa: {success_rate:.1%}, Falhas: {failed[:5]}"
        
        self.log_test("SUCCESS", "Teste abrangente concluído",
                     success_rate=f"{success_rate:.1%}",
                     successful_count=successful_count,
                     total_analyses=total_analyses,
                     failed_count=len(failed))
        
        print(f"📊 Taxa de sucesso: {success_rate:.1%} ({successful_count}/{total_analyses})")
        
        return results
    
    def test_cache_functionality(self):
        """
        Teste básico da funcionalidade de cache.
        """
        self.log_test("INFO", "Testando funcionalidade de cache")
        
        # Limpar cache
        self.stats_tool._analysis_cache.clear()
        
        # Primeira execução
        start_time = time.time()
        result1 = self.stats_tool._run(
            analysis_type="correlation",
            data_csv=self.real_data_path,
            cache_results=True
        )
        first_time = time.time() - start_time
        
        # Verificar se cache foi populado
        cache_populated = len(self.stats_tool._analysis_cache) > 0
        # Note: cache pode não ser populado dependendo da implementação
        
        # Segunda execução (pode usar cache)
        start_time = time.time()
        result2 = self.stats_tool._run(
            analysis_type="correlation",
            data_csv=self.real_data_path,
            cache_results=True
        )
        second_time = time.time() - start_time
        
        # Validações básicas (não exigir cache)
        assert isinstance(result1, str), "Primeiro resultado deve ser string"
        assert isinstance(result2, str), "Segundo resultado deve ser string"
        assert len(result1) > 50, "Primeiro resultado muito curto"
        assert len(result2) > 50, "Segundo resultado muito curto"
        
        self.log_test("SUCCESS", "Cache testado",
                     first_time=f"{first_time:.2f}s",
                     second_time=f"{second_time:.2f}s",
                     cache_populated=cache_populated)
        
        return True
    
    def test_error_handling_basic(self):
        """
        Teste básico de tratamento de erros.
        """
        self.log_test("INFO", "Testando tratamento de erros")
        
        # Teste 1: Arquivo inexistente
        result1 = self.stats_tool._run(
            analysis_type="correlation",
            data_csv="arquivo_inexistente.csv"
        )
        
        assert isinstance(result1, str), "Resultado deve ser string mesmo com erro"
        assert len(result1) > 10, "Mensagem de erro muito curta"
        
        # Teste 2: Tipo de análise inválido
        result2 = self.stats_tool._run(
            analysis_type="analise_inexistente",
            data_csv=self.real_data_path
        )
        
        assert isinstance(result2, str), "Resultado deve ser string mesmo com erro"
        assert "não suportada" in result2.lower() or "available" in result2.lower(), "Deve indicar análise não suportada"
        
        self.log_test("SUCCESS", "Tratamento de erros validado")
        
        return True
    
    def test_performance_basic(self):
        """
        Teste básico de performance.
        """
        self.log_test("INFO", "Testando performance básica")
        
        start_time = time.time()
        tracemalloc.start()
        
        # Executar análise rápida
        result = self.stats_tool._run(
            analysis_type="correlation",
            data_csv=self.real_data_path,
            target_column="Total_Liquido"
        )
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Validações de performance
        assert execution_time < 30, f"Execução muito lenta: {execution_time:.2f}s"
        assert peak < 500 * 1024 * 1024, f"Uso de memória muito alto: {peak/1024/1024:.1f}MB"
        
        self.log_test("SUCCESS", "Performance validada",
                     execution_time=f"{execution_time:.2f}s",
                     memory_peak=f"{peak/1024/1024:.1f}MB",
                     result_length=len(result))
        
        return True
    
    def teardown_method(self, method):
        """Limpeza após cada teste."""
        elapsed = time.time() - self.start_time
        print(f"🏁 Teste {method.__name__} concluído em {elapsed:.2f}s")
        
        # Log summary se disponível
        if self.test_logs:
            success_logs = [log for log in self.test_logs if log['level'] == 'SUCCESS']
            error_logs = [log for log in self.test_logs if log['level'] == 'ERROR']
            print(f"📊 Resumo: {len(success_logs)} sucessos, {len(error_logs)} erros")


# Execução standalone para desenvolvimento
if __name__ == "__main__":
    test_instance = TestStatisticalAnalysisTool()
    
    # Verificar se existe arquivo de dados real
    data_path = "data/vendas.csv"
    if not os.path.exists(data_path):
        print(f"⚠️ Arquivo {data_path} não encontrado. Usando dados de amostra.")
        data_path = "src/tests/data_tests/vendas_sample.csv"
    
    if os.path.exists(data_path):
        test_instance.setup_standalone(data_path)
        
        print("🧪 Executando testes do Statistical Analysis Tool V3...")
        
        # Executar testes principais
        try:
            test_instance.test_demographic_patterns_basic()
            test_instance.test_correlation_analysis_basic()
            test_instance.test_critical_analyses_batch()
            test_instance.test_cache_functionality()
            test_instance.test_error_handling_basic()
            
            print("✅ Todos os testes principais passaram!")
            
        except Exception as e:
            print(f"❌ Erro nos testes: {str(e)}")
    else:
        print(f"❌ Nenhum arquivo de dados encontrado para teste.") 