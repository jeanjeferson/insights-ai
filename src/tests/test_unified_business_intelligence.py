"""
🧪 TESTE PARA UNIFIED BUSINESS INTELLIGENCE
===========================================

Suite de testes para validar a funcionalidade core do Unified Business Intelligence.
Baseada no template funcionando do KPI Calculator Tool V3.0.
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
from src.insights.tools.business_intelligence_tool import BusinessIntelligenceTool


class TestBusinessIntelligenceTool:
    """
    Suite de testes para Unified Business Intelligence
    
    Focada em validação funcional dos 10 tipos de análise de BI.
    """
    
    # Tipos de análise executivos identificados na ferramenta
    EXECUTIVE_ANALYSES = [
        'executive_summary',              # Resumo C-level
        'executive_dashboard',            # Dashboard visual executivo
        'financial_analysis',             # KPIs + visualizações + benchmarks
        'customer_intelligence',          # RFM + segmentação + retenção
        'product_performance'             # ABC + rankings + categoria
    ]
    
    ALL_ANALYSES = [
        # Executivos
        'executive_summary', 'executive_dashboard',
        # Financeiros  
        'financial_analysis', 'profitability_analysis',
        # Clientes e produtos
        'customer_intelligence', 'product_performance',
        # Especializados
        'demographic_analysis', 'geographic_analysis', 
        'sales_team_analysis', 'comprehensive_report'
    ]
    
    @pytest.fixture(autouse=True)
    def setup(self, real_vendas_data):
        """Setup automático para cada teste."""
        self.bi_tool = BusinessIntelligenceTool()
        self.real_data_path = real_vendas_data
        self.test_logs = []
        self.start_time = time.time()
        
        print(f"🚀 Iniciando teste Unified Business Intelligence com dados: {self.real_data_path}")
    
    def setup_standalone(self, data_path):
        """Setup para execução standalone."""
        self.bi_tool = BusinessIntelligenceTool()
        self.real_data_path = data_path
        self.test_logs = []
        self.start_time = time.time()
        
        print(f"🚀 Iniciando teste Unified Business Intelligence com dados: {self.real_data_path}")
    
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
    
    def test_executive_summary_basic(self):
        """
        Teste básico do resumo executivo.
        """
        self.log_test("INFO", "Iniciando teste de resumo executivo")
        
        # Medir performance básica
        start_time = time.time()
        tracemalloc.start()
        
        result = self.bi_tool._run(
            analysis_type="executive_summary",
            data_csv=self.real_data_path,
            time_period="last_12_months",
            detail_level="summary",
            include_forecasts=True
        )
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Validações básicas
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 200, "Resultado muito curto para um resumo executivo"
        # Verificar erros mais específicos
        error_indicators = ["❌ erro", "error:", "exception:", "traceback"]
        has_error = any(indicator in result.lower() for indicator in error_indicators)
        assert not has_error, f"Erro detectado: {result[:200]}..."
        
        # Validações de conteúdo executivo (aceita termos em português e inglês)
        executive_terms = [
            "executivo", "executive", "sumário", "summary", 
            "kpi", "receita", "revenue", "performance", "negócio", "business"
        ]
        found_terms = [term for term in executive_terms if term in result.lower()]
        assert len(found_terms) >= 3, f"Poucos termos executivos encontrados: {found_terms}"
        
        self.log_test("SUCCESS", "Resumo executivo validado", 
                     execution_time=f"{execution_time:.2f}s",
                     memory_peak=f"{peak/1024/1024:.1f}MB",
                     terms_found=len(found_terms),
                     result_length=len(result))
        
        return result
    
    def test_executive_dashboard_basic(self):
        """
        Teste básico do dashboard executivo.
        """
        self.log_test("INFO", "Iniciando teste de dashboard executivo")
        
        start_time = time.time()
        
        result = self.bi_tool._run(
            analysis_type="executive_dashboard",
            data_csv=self.real_data_path,
            output_format="interactive",
            include_forecasts=True,
            export_file=False  # Não exportar arquivo para acelerar teste
        )
        
        execution_time = time.time() - start_time
        
        # Validações básicas
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 100, "Resultado muito curto"
        # Verificar erros mais específicos (evitar false positives no HTML)
        error_indicators = ["❌ erro", "error:", "exception:", "traceback"]
        has_error = any(indicator in result.lower() for indicator in error_indicators)
        assert not has_error, f"Erro detectado: {result[:200]}..."
        
        # Validações de conteúdo de dashboard
        dashboard_terms = [
            "dashboard", "gráfico", "chart", "visual", "plotly",
            "interativo", "interactive", "figure", "html"
        ]
        found_terms = [term for term in dashboard_terms if term in result.lower()]
        assert len(found_terms) >= 2, f"Poucos termos de dashboard encontrados: {found_terms}"
        
        self.log_test("SUCCESS", "Dashboard executivo validado", 
                     execution_time=f"{execution_time:.2f}s",
                     terms_found=len(found_terms),
                     result_length=len(result))
        
        return result
    
    def test_financial_analysis_basic(self):
        """
        Teste básico da análise financeira.
        """
        self.log_test("INFO", "Iniciando teste de análise financeira")
        
        start_time = time.time()
        
        result = self.bi_tool._run(
            analysis_type="financial_analysis",
            data_csv=self.real_data_path,
            time_period="last_12_months",
            include_forecasts=True
        )
        
        execution_time = time.time() - start_time
        
        # Validações básicas
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 200, "Resultado muito curto para análise financeira"
        # Verificar erros mais específicos
        error_indicators = ["❌ erro", "error:", "exception:", "traceback"]
        has_error = any(indicator in result.lower() for indicator in error_indicators)
        assert not has_error, f"Erro detectado: {result[:200]}..."
        
        # Validações de conteúdo financeiro
        financial_terms = [
            "receita", "revenue", "kpi", "financeiro", "financial",
            "lucro", "profit", "margem", "margin", "crescimento", "growth"
        ]
        found_terms = [term for term in financial_terms if term in result.lower()]
        assert len(found_terms) >= 3, f"Poucos termos financeiros encontrados: {found_terms}"
        
        # Verificar valores monetários
        assert "R$" in result or "real" in result.lower(), "Deve incluir valores monetários"
        
        self.log_test("SUCCESS", "Análise financeira validada", 
                     execution_time=f"{execution_time:.2f}s",
                     terms_found=len(found_terms),
                     result_length=len(result))
        
        return result
    
    def test_executive_analyses_batch(self):
        """
        Teste das 5 análises executivas mais importantes em lote.
        """
        self.log_test("INFO", f"Testando {len(self.EXECUTIVE_ANALYSES)} análises executivas")
        
        results = {}
        
        for analysis_type in self.EXECUTIVE_ANALYSES:
            start_time = time.time()
            
            try:
                result = self.bi_tool._run(
                    analysis_type=analysis_type,
                    data_csv=self.real_data_path,
                    time_period="last_12_months",
                    detail_level="summary"
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
        success_rate = len(successful) / len(self.EXECUTIVE_ANALYSES)
        
        assert success_rate >= 0.6, f"Taxa de sucesso baixa: {success_rate:.1%}"
        
        self.log_test("SUCCESS", "Teste de análises executivas concluído",
                     success_rate=f"{success_rate:.1%}",
                     successful_analyses=successful)
        
        return results
    
    def test_all_analyses_comprehensive(self):
        """
        Teste abrangente de todas as análises de BI disponíveis.
        """
        self.log_test("INFO", f"Testando todas as {len(self.ALL_ANALYSES)} análises de BI")
        
        results = {}
        failed = []
        
        for analysis_type in self.ALL_ANALYSES:
            try:
                start_time = time.time()
                
                result = self.bi_tool._run(
                    analysis_type=analysis_type,
                    data_csv=self.real_data_path,
                    time_period="last_12_months"
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
        
        # Aceitar até 40% de falha (60% de sucesso)
        assert success_rate >= 0.6, f"Taxa de sucesso baixa: {success_rate:.1%}, Falhas: {failed[:5]}"
        
        self.log_test("SUCCESS", "Teste abrangente concluído",
                     success_rate=f"{success_rate:.1%}",
                     successful_count=successful_count,
                     total_analyses=total_analyses,
                     failed_count=len(failed))
        
        print(f"📊 Taxa de sucesso: {success_rate:.1%} ({successful_count}/{total_analyses})")
        
        return results
    
    def test_output_formats(self):
        """
        Teste dos diferentes formatos de saída.
        """
        self.log_test("INFO", "Testando formatos de saída")
        
        formats = ["text", "interactive", "html"]
        results = {}
        
        for output_format in formats:
            try:
                start_time = time.time()
                
                result = self.bi_tool._run(
                    analysis_type="executive_summary",
                    data_csv=self.real_data_path,
                    output_format=output_format,
                    export_file=False  # Não exportar arquivo para acelerar
                )
                
                execution_time = time.time() - start_time
                
                results[output_format] = {
                    'success': isinstance(result, str) and len(result) > 50,
                    'execution_time': round(execution_time, 2),
                    'output_length': len(result) if isinstance(result, str) else 0
                }
                
                self.log_test("SUCCESS", f"Formato {output_format}",
                             **results[output_format])
                
            except Exception as e:
                results[output_format] = {
                    'success': False,
                    'error': str(e)
                }
                self.log_test("ERROR", f"Erro no formato {output_format}: {str(e)}")
        
        # Pelo menos um formato deve funcionar
        successful = [fmt for fmt, res in results.items() if res['success']]
        assert len(successful) >= 1, f"Nenhum formato funcionou: {results}"
        
        self.log_test("SUCCESS", "Formatos de saída testados",
                     successful_formats=successful)
        
        return results
    
    def test_error_handling_basic(self):
        """
        Teste básico de tratamento de erros.
        """
        self.log_test("INFO", "Testando tratamento de erros")
        
        # Teste 1: Arquivo inexistente
        result1 = self.bi_tool._run(
            analysis_type="executive_summary",
            data_csv="arquivo_inexistente.csv"
        )
        
        assert isinstance(result1, str), "Resultado deve ser string mesmo com erro"
        assert len(result1) > 10, "Mensagem de erro muito curta"
        
        # Teste 2: Tipo de análise inválido
        result2 = self.bi_tool._run(
            analysis_type="analise_inexistente",
            data_csv=self.real_data_path
        )
        
        assert isinstance(result2, str), "Resultado deve ser string mesmo com erro"
        assert "não suportada" in result2.lower() or "opções" in result2.lower(), "Deve indicar análise não suportada"
        
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
        result = self.bi_tool._run(
            analysis_type="executive_summary",
            data_csv=self.real_data_path,
            detail_level="summary",
            include_forecasts=False  # Acelerar
        )
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Validações de performance
        assert execution_time < 45, f"Execução muito lenta: {execution_time:.2f}s"
        assert peak < 600 * 1024 * 1024, f"Uso de memória muito alto: {peak/1024/1024:.1f}MB"
        
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
    test_instance = TestBusinessIntelligenceTool()
    
    # Verificar se existe arquivo de dados real
    data_path = "data/vendas.csv"
    if not os.path.exists(data_path):
        print(f"⚠️ Arquivo {data_path} não encontrado. Usando dados de amostra.")
        data_path = "src/tests/data_tests/vendas_sample.csv"
    
    if os.path.exists(data_path):
        test_instance.setup_standalone(data_path)
        
        print("🧪 Executando testes do Unified Business Intelligence...")
        
        # Executar testes principais
        try:
            test_instance.test_executive_summary_basic()
            test_instance.test_financial_analysis_basic()
            test_instance.test_executive_analyses_batch()
            test_instance.test_output_formats()
            test_instance.test_error_handling_basic()
            
            print("✅ Todos os testes principais passaram!")
            
        except Exception as e:
            print(f"❌ Erro nos testes: {str(e)}")
    else:
        print(f"❌ Nenhum arquivo de dados encontrado para teste.") 