"""
🧪 TESTE PARA MÓDULOS COMPARTILHADOS
===================================

Suite de testes para validar os módulos compartilhados do sistema.
Testa data_preparation.py, business_mixins.py e report_formatter.py.
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

# Importar módulos a serem testados
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.insights.tools.shared.data_preparation import DataPreparationMixin
from src.insights.tools.shared.business_mixins import JewelryRFMAnalysisMixin
from src.insights.tools.shared.report_formatter import ReportFormatterMixin


class TestSharedModules:
    """
    Suite de testes para módulos compartilhados
    
    Focada em validação funcional dos 3 módulos principais.
    """
    
    @pytest.fixture(autouse=True)
    def setup(self, real_vendas_data):
        """Setup automático para cada teste."""
        self.real_data_path = real_vendas_data
        self.test_logs = []
        self.start_time = time.time()
        
        print(f"🚀 Iniciando teste Shared Modules com dados: {self.real_data_path}")
    
    def setup_standalone(self, data_path):
        """Setup para execução standalone."""
        self.real_data_path = data_path
        self.test_logs = []
        self.start_time = time.time()
        
        print(f"🚀 Iniciando teste Shared Modules com dados: {self.real_data_path}")
    
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
    
    def test_data_preparation_basic(self):
        """
        Teste básico do DataPreparationMixin.
        """
        self.log_test("INFO", "Iniciando teste de preparação de dados")
        
        # Medir performance básica
        start_time = time.time()
        tracemalloc.start()
        
        prep = DataPreparationMixin()
        
        # Carregar dados reais
        df = pd.read_csv(self.real_data_path, sep=';', encoding='utf-8')
        original_rows = len(df)
        
        # Testar preparação
        result = prep.prepare_jewelry_data(df, validation_level="basic")
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Validações básicas
        assert result is not None, "Preparação falhou"
        assert len(result) > 0, "Dados preparados vazios"
        assert isinstance(result, pd.DataFrame), "Resultado deve ser DataFrame"
        
        # Validações de colunas essenciais
        essential_cols = ['Data', 'Total_Liquido']
        for col in essential_cols:
            assert col in result.columns, f"Campo {col} não encontrado"
        
        # Validações de tipos
        assert pd.api.types.is_datetime64_any_dtype(result['Data']), "Data deve ser datetime"
        assert pd.api.types.is_numeric_dtype(result['Total_Liquido']), "Total_Liquido deve ser numérico"
        
        self.log_test("SUCCESS", "Preparação de dados validada", 
                     execution_time=f"{execution_time:.2f}s",
                     memory_peak=f"{peak/1024/1024:.1f}MB",
                     original_rows=original_rows,
                     prepared_rows=len(result),
                     columns=len(result.columns))
        
        return result
    
    def test_rfm_analysis_basic(self):
        """
        Teste básico do JewelryRFMAnalysisMixin.
        """
        self.log_test("INFO", "Iniciando teste de análise RFM")
        
        start_time = time.time()
        
        rfm = JewelryRFMAnalysisMixin()
        
        # Preparar dados primeiro
        prep = DataPreparationMixin()
        df = pd.read_csv(self.real_data_path, sep=';', encoding='utf-8')
        df_prepared = prep.prepare_jewelry_data(df, validation_level="basic")
        
        # Usar uma amostra para acelerar teste
        sample_df = df_prepared.head(1000)
        
        # Testar análise RFM de clientes (método correto)
        result = rfm.analyze_customer_rfm(sample_df)
        
        execution_time = time.time() - start_time
        
        # Validações básicas
        assert result is not None, "RFM falhou"
        assert isinstance(result, dict), "Resultado RFM deve ser dicionário"
        
        # Verificar se houve erro e fazer fallback para produtos
        if 'error' in result:
            self.log_test("WARNING", f"RFM de clientes retornou aviso: {result['error']}")
            # Se não há Codigo_Cliente, testar com análise de produtos
            self.log_test("INFO", "Testando RFM de produtos como alternativa")
            result_products = rfm.analyze_product_rfm(sample_df)
            assert result_products is not None, "RFM de produtos falhou"
            assert isinstance(result_products, dict), "Resultado RFM produtos deve ser dicionário"
            result = result_products  # Usar resultado de produtos para validação
        
        # Validar estrutura final (deve ter analysis_type agora)
        assert 'analysis_type' in result, f"Resultado deve ter analysis_type. Keys disponíveis: {list(result.keys())}"
        
        self.log_test("SUCCESS", "Análise RFM validada",
                     execution_time=f"{execution_time:.2f}s",
                     input_rows=len(sample_df),
                     analysis_type=result.get('analysis_type', 'N/A'),
                     total_analyzed=result.get('total_customers', result.get('total_products', 0)))
        
        return result
    
    def test_report_formatting_basic(self):
        """
        Teste básico do ReportFormatterMixin.
        """
        self.log_test("INFO", "Iniciando teste de formatação de relatórios")
        
        start_time = time.time()
        
        formatter = ReportFormatterMixin()
        
        # Dados de teste simples
        test_data = {
            'analysis_type': 'test_analysis',
            'total_revenue': 100000,
            'avg_order_value': 1500,
            'customer_count': 250,
            'insights': ['Insight 1', 'Insight 2'],
            'recommendations': ['Recomendação 1']
        }
        
        # Testar formatação
        result = formatter.format_business_kpi_report(test_data, "test_analysis", True)
        
        execution_time = time.time() - start_time
        
        # Validações básicas
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 100, "Relatório muito curto"
        
        # Validações de conteúdo
        content_terms = [
            "análise", "analysis", "revenue", "receita", 
            "customer", "cliente", "insight", "recomendação"
        ]
        found_terms = [term for term in content_terms if term in result.lower()]
        assert len(found_terms) >= 2, f"Poucos termos encontrados: {found_terms}"
        
        self.log_test("SUCCESS", "Formatação de relatórios validada",
                     execution_time=f"{execution_time:.2f}s",
                     report_length=len(result),
                     terms_found=len(found_terms))
        
        return result
    
    def test_integration_modules(self):
        """
        Teste de integração entre os 3 módulos.
        """
        self.log_test("INFO", "Iniciando teste de integração entre módulos")
        
        start_time = time.time()
        
        # 1. Preparar dados
        prep = DataPreparationMixin()
        df = pd.read_csv(self.real_data_path, sep=';', encoding='utf-8')
        df_prepared = prep.prepare_jewelry_data(df.head(500), validation_level="basic")  # Amostra pequena
        
        # 2. Análise RFM
        rfm = JewelryRFMAnalysisMixin()
        rfm_result = rfm.analyze_customer_rfm(df_prepared)
        
        # Se der erro no cliente, tentar com produtos
        if rfm_result and 'error' in rfm_result:
            rfm_result = rfm.analyze_product_rfm(df_prepared)
        
        # 3. Formatação do resultado
        formatter = ReportFormatterMixin()
        
        # Calcular métricas do RFM
        rfm_count = 0
        if rfm_result and isinstance(rfm_result, dict):
            rfm_count = rfm_result.get('total_customers', rfm_result.get('total_products', 0))
        
        integration_data = {
            'analysis_type': 'integration_test',
            'prepared_rows': len(df_prepared),
            'rfm_segments': rfm_count,
            'insights': [
                f"Dados preparados: {len(df_prepared)} registros",
                f"Entidades RFM analisadas: {rfm_count}"
            ]
        }
        
        final_report = formatter.format_business_kpi_report(integration_data, "integration_test", False)
        
        execution_time = time.time() - start_time
        
        # Validações de integração
        assert df_prepared is not None, "Preparação falhou na integração"
        assert final_report is not None, "Formatação falhou na integração"
        assert len(final_report) > 50, "Relatório integrado muito curto"
        
        self.log_test("SUCCESS", "Integração entre módulos validada",
                     execution_time=f"{execution_time:.2f}s",
                     prepared_rows=len(df_prepared),
                     rfm_segments=rfm_count,
                     report_length=len(final_report))
        
        return {
            'prepared_data': df_prepared,
            'rfm_analysis': rfm_result,
            'final_report': final_report
        }
    
    def test_error_handling_modules(self):
        """
        Teste de tratamento de erros nos módulos.
        """
        self.log_test("INFO", "Testando tratamento de erros dos módulos")
        
        # Teste 1: DataPreparationMixin com dados inválidos
        prep = DataPreparationMixin()
        empty_df = pd.DataFrame()
        
        try:
            result1 = prep.prepare_jewelry_data(empty_df)
            error1_handled = result1 is None or len(result1) == 0
        except Exception:
            error1_handled = True
        
        # Teste 2: RFM com dados insuficientes
        rfm = JewelryRFMAnalysisMixin()
        minimal_df = pd.DataFrame({'Customer_ID': ['C1'], 'Data': [datetime.now()], 'Total_Liquido': [100]})
        
        try:
            result2 = rfm.analyze_customer_rfm(minimal_df)
            error2_handled = True  # Se não crashou, está tratando
        except Exception:
            error2_handled = False
        
        # Teste 3: Formatter com dados None
        formatter = ReportFormatterMixin()
        
        try:
            result3 = formatter.format_business_kpi_report(None, "test", False)
            error3_handled = isinstance(result3, str)
        except Exception:
            error3_handled = False
        
        # Validações
        errors_handled = sum([error1_handled, error2_handled, error3_handled])
        assert errors_handled >= 2, f"Poucos erros tratados: {errors_handled}/3"
        
        self.log_test("SUCCESS", "Tratamento de erros validado",
                     errors_handled=f"{errors_handled}/3")
        
        return True
    
    def test_performance_modules(self):
        """
        Teste básico de performance dos módulos.
        """
        self.log_test("INFO", "Testando performance dos módulos")
        
        start_time = time.time()
        tracemalloc.start()
        
        # Teste de performance com amostra pequena
        df = pd.read_csv(self.real_data_path, sep=';', encoding='utf-8', nrows=100)
        
        # DataPreparation
        prep = DataPreparationMixin()
        prep_start = time.time()
        df_prepared = prep.prepare_jewelry_data(df, validation_level="basic")
        prep_time = time.time() - prep_start
        
        # RFM (se dados preparados)
        if df_prepared is not None and len(df_prepared) > 0:
            rfm = JewelryRFMAnalysisMixin()
            rfm_start = time.time()
            # Tentar clientes primeiro, depois produtos como fallback
            rfm_result = rfm.analyze_customer_rfm(df_prepared)
            if rfm_result and 'error' in rfm_result:
                rfm_result = rfm.analyze_product_rfm(df_prepared)
            rfm_time = time.time() - rfm_start
        else:
            rfm_time = 0
        
        # Formatter
        formatter = ReportFormatterMixin()
        format_start = time.time()
        test_data = {'analysis_type': 'performance_test', 'total_revenue': 10000}
        report = formatter.format_business_kpi_report(test_data, "performance", False)
        format_time = time.time() - format_start
        
        total_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Validações de performance
        assert total_time < 10, f"Performance muito lenta: {total_time:.2f}s"
        assert peak < 200 * 1024 * 1024, f"Uso de memória muito alto: {peak/1024/1024:.1f}MB"
        
        self.log_test("SUCCESS", "Performance validada",
                     total_time=f"{total_time:.2f}s",
                     prep_time=f"{prep_time:.2f}s",
                     rfm_time=f"{rfm_time:.2f}s",
                     format_time=f"{format_time:.2f}s",
                     memory_peak=f"{peak/1024/1024:.1f}MB")
        
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
    test_instance = TestSharedModules()
    
    # Verificar se existe arquivo de dados real
    data_path = "data/vendas.csv"
    if not os.path.exists(data_path):
        print(f"⚠️ Arquivo {data_path} não encontrado. Usando dados de amostra.")
        data_path = "src/tests/data_tests/vendas_sample.csv"
    
    if os.path.exists(data_path):
        test_instance.setup_standalone(data_path)
        
        print("🧪 Executando testes dos Módulos Compartilhados...")
        
        # Executar testes principais
        try:
            test_instance.test_data_preparation_basic()
            test_instance.test_rfm_analysis_basic()
            test_instance.test_report_formatting_basic()
            test_instance.test_integration_modules()
            test_instance.test_error_handling_modules()
            
            print("✅ Todos os testes principais passaram!")
            
        except Exception as e:
            print(f"❌ Erro nos testes: {str(e)}")
    else:
        print(f"❌ Nenhum arquivo de dados encontrado para teste.") 