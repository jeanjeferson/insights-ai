"""
🧪 TESTE COMPLETO PARA CUSTOMER INSIGHTS ENGINE TOOL V3.0
============================================================

Suite de testes abrangente para validar todas as funcionalidades do
Customer Insights Engine Tool, incluindo:
- 6 tipos de análise de clientes
- Segmentação comportamental, demográfica e geográfica
- Predição de churn e análise de valor
- Mineração de preferências e mapeamento de jornada
- Validação robusta e casos edge
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
import warnings

# Importar ferramenta a ser testada
import sys
import os

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

# Tentar importar o Customer Insights Engine Tool
try:
    from src.insights.tools.advanced.customer_insights_engine import CustomerInsightsEngine
except ImportError:
    # Fallback para importação direta
    sys.path.insert(0, os.path.join(project_root, 'src', 'insights', 'tools', 'advanced'))
    from customer_insights_engine import CustomerInsightsEngine

# Suprimir warnings para testes mais limpos
warnings.filterwarnings('ignore')


class TestCustomerInsightsEngine:
    """
    Suite completa de testes para Customer Insights Engine Tool V3.0
    
    Cobertura:
    - Todas as 6 análises de insights de clientes
    - Segmentação comportamental, demográfica e geográfica
    - Predição de churn avançada
    - Análise de valor e CLV
    - Mineração de preferências
    - Mapeamento de jornada do cliente
    - Casos de erro e edge cases
    """
    
    @pytest.fixture(autouse=True)
    def setup(self, real_vendas_data):
        """Setup automático para cada teste."""
        self.insights_engine = CustomerInsightsEngine()
        self.real_data_path = real_vendas_data
        self.test_logs = []
        self.start_time = time.time()
        
        print(f"🚀 Iniciando teste Customer Insights Engine: {self.real_data_path}")
    
    def setup_standalone(self, data_path):
        """Setup para execução standalone."""
        self.insights_engine = CustomerInsightsEngine()
        self.real_data_path = data_path
        self.test_logs = []
        self.start_time = time.time()
        
        print(f"🚀 Iniciando teste Customer Insights Engine: {self.real_data_path}")
    
    def log_test(self, level: str, message: str, **kwargs):
        """Logging detalhado para testes."""
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
    
    # ==========================================
    # TESTES DE VALIDAÇÃO BÁSICA
    # ==========================================
    
    def test_data_loading_and_validation(self):
        """Teste de carregamento e validação de dados de clientes."""
        self.log_test("INFO", "Testando carregamento e validação de dados de clientes")
        
        # Verificar se arquivo existe
        assert os.path.exists(self.real_data_path), "Arquivo de dados não encontrado"
        
        # Testar carregamento direto
        df = pd.read_csv(self.real_data_path, sep=';', encoding='utf-8')
        assert df is not None, "Falha no carregamento de dados"
        assert len(df) > 0, "DataFrame vazio"
        assert len(df.columns) >= 3, f"Poucas colunas: {len(df.columns)}"
        
        # Verificar colunas essenciais para análise de clientes
        essential_cols = ['Data', 'Total_Liquido']
        missing_cols = [col for col in essential_cols if col not in df.columns]
        assert len(missing_cols) == 0, f"Colunas essenciais ausentes: {missing_cols}"
        
        # Verificar dados de clientes se disponíveis
        customer_cols = ['Codigo_Cliente', 'Nome_Cliente']
        available_customer_cols = [col for col in customer_cols if col in df.columns]
        
        # Verificar dados demográficos se disponíveis
        demographic_cols = ['Idade', 'Sexo', 'Estado_Civil']
        available_demographic_cols = [col for col in demographic_cols if col in df.columns]
        
        # Verificar dados geográficos se disponíveis  
        geographic_cols = ['Cidade', 'Estado']
        available_geographic_cols = [col for col in geographic_cols if col in df.columns]
        
        self.log_test("SUCCESS", "Validação de dados aprovada",
                     rows=len(df), 
                     columns=len(df.columns),
                     customer_cols=len(available_customer_cols),
                     demographic_cols=len(available_demographic_cols),
                     geographic_cols=len(available_geographic_cols))
    
    def test_customer_data_preparation(self):
        """Teste de preparação de dados de clientes."""
        self.log_test("INFO", "Testando preparação de dados de clientes")
        
        # Carregar dados brutos
        df = pd.read_csv(self.real_data_path, sep=';', encoding='utf-8')
        
        # Testar preparação com dados reais
        customer_data = self.insights_engine._prepare_customer_data(
            df, 
            customer_id_column="Codigo_Cliente",
            include_demographics=True,
            include_geographic=True,
            min_transactions=1
        )
        
        # Se não há Codigo_Cliente, o método pode retornar None ou tentar alternativas
        if customer_data is not None:
            assert len(customer_data) > 0, "Nenhum cliente preparado"
            
            # Verificar métricas RFM obrigatórias
            rfm_cols = ['Recency', 'Frequency', 'Monetary']
            missing_rfm = [col for col in rfm_cols if col not in customer_data.columns]
            assert len(missing_rfm) == 0, f"Métricas RFM ausentes: {missing_rfm}"
            
            # Verificar métricas comportamentais
            behavioral_cols = ['AOV_Real', 'CLV_Estimado', 'Customer_Lifetime_Days']
            found_behavioral = [col for col in behavioral_cols if col in customer_data.columns]
            
            self.log_test("SUCCESS", "Dados de clientes preparados",
                         customers=len(customer_data),
                         total_cols=len(customer_data.columns),
                         behavioral_metrics=len(found_behavioral))
        else:
            self.log_test("WARNING", "Preparação de dados retornou None - possível ausência de identificador de cliente")
    
    # ==========================================
    # TESTES DAS 6 ANÁLISES PRINCIPAIS
    # ==========================================
    
    def test_behavioral_segmentation_analysis(self):
        """Teste completo da segmentação comportamental."""
        self.log_test("INFO", "Testando Behavioral Segmentation")
        
        start_time = time.time()
        tracemalloc.start()
        
        result = self.insights_engine._run(
            analysis_type="behavioral_segmentation",
            data_csv=self.real_data_path,
            customer_id_column="Codigo_Cliente",
            segmentation_method="rfm",
            include_demographics=True,
            include_geographic=True,
            min_transactions=1
        )
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Validações básicas
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 300, "Resultado muito curto para segmentação comportamental"
        
        # Verificar se é JSON válido ou contém termos esperados
        segmentation_terms = ["segmentation", "rfm", "cluster", "behavior", "customer"]
        found_terms = [term for term in segmentation_terms if term.lower() in result.lower()]
        
        assert len(found_terms) >= 2, f"Poucos termos de segmentação encontrados: {found_terms}"
        
        # Verificar se não há erros
        error_indicators = ["erro", "error", "falha", "failed"]
        found_errors = [error for error in error_indicators if error.lower() in result.lower()]
        
        # Se houver erro sobre dados insuficientes, é aceitável
        if found_errors and "insuficientes" not in result.lower():
            pytest.fail(f"Possíveis erros detectados: {found_errors} em {result[:200]}...")
        
        self.log_test("SUCCESS", "Segmentação comportamental executada",
                     execution_time=round(execution_time, 2),
                     memory_peak_mb=round(peak / 1024 / 1024, 2),
                     result_length=len(result),
                     terms_found=len(found_terms))
    
    def test_lifecycle_analysis(self):
        """Teste da análise de ciclo de vida."""
        self.log_test("INFO", "Testando Lifecycle Analysis")
        
        start_time = time.time()
        
        result = self.insights_engine._run(
            analysis_type="lifecycle_analysis",
            data_csv=self.real_data_path,
            customer_id_column="Codigo_Cliente",
            include_demographics=True,
            min_transactions=1
        )
        
        execution_time = time.time() - start_time
        
        # Validações básicas
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 200, "Resultado muito curto para análise de ciclo de vida"
        
        # Verificar termos de ciclo de vida
        lifecycle_terms = ["lifecycle", "ciclo", "estágio", "stage", "novo", "leal", "vip", "risco"]
        found_terms = [term for term in lifecycle_terms if term.lower() in result.lower()]
        
        assert len(found_terms) >= 2, f"Poucos termos de ciclo de vida encontrados: {found_terms}"
        
        self.log_test("SUCCESS", "Análise de ciclo de vida executada",
                     execution_time=round(execution_time, 2),
                     result_length=len(result),
                     lifecycle_terms=len(found_terms))
    
    def test_churn_prediction_analysis(self):
        """Teste da predição de churn."""
        self.log_test("INFO", "Testando Churn Prediction")
        
        start_time = time.time()
        
        result = self.insights_engine._run(
            analysis_type="churn_prediction",
            data_csv=self.real_data_path,
            customer_id_column="Codigo_Cliente",
            prediction_horizon=90,
            min_transactions=1
        )
        
        execution_time = time.time() - start_time
        
        # Validações básicas
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 200, "Resultado muito curto para predição de churn"
        
        # Verificar termos de churn
        churn_terms = ["churn", "risco", "risk", "abandono", "predição", "retenção"]
        found_terms = [term for term in churn_terms if term.lower() in result.lower()]
        
        assert len(found_terms) >= 2, f"Poucos termos de churn encontrados: {found_terms}"
        
        self.log_test("SUCCESS", "Predição de churn executada",
                     execution_time=round(execution_time, 2),
                     result_length=len(result),
                     churn_terms=len(found_terms))
    
    def test_value_analysis(self):
        """Teste da análise de valor."""
        self.log_test("INFO", "Testando Value Analysis")
        
        start_time = time.time()
        
        result = self.insights_engine._run(
            analysis_type="value_analysis",
            data_csv=self.real_data_path,
            customer_id_column="Codigo_Cliente",
            segmentation_method="value_based",
            min_transactions=1
        )
        
        execution_time = time.time() - start_time
        
        # Validações básicas
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 200, "Resultado muito curto para análise de valor"
        
        # Verificar termos de valor
        value_terms = ["valor", "value", "clv", "monetary", "diamond", "platinum", "bronze", "pareto"]
        found_terms = [term for term in value_terms if term.lower() in result.lower()]
        
        assert len(found_terms) >= 2, f"Poucos termos de valor encontrados: {found_terms}"
        
        self.log_test("SUCCESS", "Análise de valor executada",
                     execution_time=round(execution_time, 2),
                     result_length=len(result),
                     value_terms=len(found_terms))
    
    def test_preference_mining_analysis(self):
        """Teste da mineração de preferências."""
        self.log_test("INFO", "Testando Preference Mining")
        
        start_time = time.time()
        
        result = self.insights_engine._run(
            analysis_type="preference_mining",
            data_csv=self.real_data_path,
            customer_id_column="Codigo_Cliente",
            include_demographics=True,
            include_geographic=True,
            min_transactions=1
        )
        
        execution_time = time.time() - start_time
        
        # Validações básicas
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 200, "Resultado muito curto para mineração de preferências"
        
        # Verificar termos de preferências
        preference_terms = ["preference", "preferência", "mining", "demographic", "geographic", "produto", "idade", "sexo"]
        found_terms = [term for term in preference_terms if term.lower() in result.lower()]
        
        assert len(found_terms) >= 2, f"Poucos termos de preferência encontrados: {found_terms}"
        
        self.log_test("SUCCESS", "Mineração de preferências executada",
                     execution_time=round(execution_time, 2),
                     result_length=len(result),
                     preference_terms=len(found_terms))
    
    def test_journey_mapping_analysis(self):
        """Teste do mapeamento de jornada."""
        self.log_test("INFO", "Testando Journey Mapping")
        
        start_time = time.time()
        
        result = self.insights_engine._run(
            analysis_type="journey_mapping",
            data_csv=self.real_data_path,
            customer_id_column="Codigo_Cliente",
            include_demographics=True,
            min_transactions=1
        )
        
        execution_time = time.time() - start_time
        
        # Validações básicas
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 200, "Resultado muito curto para mapeamento de jornada"
        
        # Verificar termos de jornada
        journey_terms = ["journey", "jornada", "mapping", "estágio", "friction", "atrito", "experiência"]
        found_terms = [term for term in journey_terms if term.lower() in result.lower()]
        
        assert len(found_terms) >= 2, f"Poucos termos de jornada encontrados: {found_terms}"
        
        self.log_test("SUCCESS", "Mapeamento de jornada executado",
                     execution_time=round(execution_time, 2),
                     result_length=len(result),
                     journey_terms=len(found_terms))
    
    # ==========================================
    # TESTES DE INTEGRAÇÃO E EDGE CASES
    # ==========================================
    
    def test_all_analysis_types_integration(self):
        """Teste de integração executando todas as análises sequencialmente."""
        self.log_test("INFO", "Testando integração de todas as análises")
        
        analysis_types = [
            "behavioral_segmentation",
            "lifecycle_analysis", 
            "churn_prediction",
            "value_analysis",
            "preference_mining",
            "journey_mapping"
        ]
        
        results = {}
        total_start_time = time.time()
        
        for analysis_type in analysis_types:
            start_time = time.time()
            
            try:
                result = self.insights_engine._run(
                    analysis_type=analysis_type,
                    data_csv=self.real_data_path,
                    customer_id_column="Codigo_Cliente",
                    min_transactions=1
                )
                
                execution_time = time.time() - start_time
                
                # Validar resultado básico
                assert isinstance(result, str), f"Resultado de {analysis_type} deve ser string"
                assert len(result) > 100, f"Resultado de {analysis_type} muito curto"
                
                results[analysis_type] = {
                    'success': True,
                    'execution_time': round(execution_time, 2),
                    'result_length': len(result)
                }
                
                self.log_test("SUCCESS", f"✅ {analysis_type} executado com sucesso",
                             execution_time=round(execution_time, 2),
                             result_length=len(result))
                
            except Exception as e:
                results[analysis_type] = {
                    'success': False,
                    'error': str(e),
                    'execution_time': time.time() - start_time
                }
                
                self.log_test("ERROR", f"❌ {analysis_type} falhou", error=str(e))
        
        total_execution_time = time.time() - total_start_time
        successful_analyses = sum(1 for r in results.values() if r['success'])
        
        # Validar que pelo menos algumas análises funcionaram
        assert successful_analyses >= len(analysis_types) // 2, \
            f"Muitas análises falharam: {successful_analyses}/{len(analysis_types)}"
        
        self.log_test("SUCCESS", "Teste de integração concluído",
                     total_time=round(total_execution_time, 2),
                     successful=successful_analyses,
                     total=len(analysis_types),
                     success_rate=round(successful_analyses/len(analysis_types)*100, 1))
    
    def test_error_handling_comprehensive(self):
        """Teste abrangente de tratamento de erros."""
        self.log_test("INFO", "Testando tratamento de erros")
        
        # Teste 1: Arquivo inexistente
        result_bad_file = self.insights_engine._run(
            analysis_type="behavioral_segmentation",
            data_csv="arquivo_inexistente.csv"
        )
        assert "erro" in result_bad_file.lower() or "error" in result_bad_file.lower(), \
            "Deve retornar erro para arquivo inexistente"
        
        # Teste 2: Tipo de análise inválido
        result_bad_type = self.insights_engine._run(
            analysis_type="invalid_analysis_type",
            data_csv=self.real_data_path
        )
        assert "erro" in result_bad_type.lower() or "error" in result_bad_type.lower() or \
               "suportada" in result_bad_type.lower(), \
            "Deve retornar erro para tipo de análise inválido"
        
        # Teste 3: Coluna de cliente inexistente
        result_bad_column = self.insights_engine._run(
            analysis_type="behavioral_segmentation",
            data_csv=self.real_data_path,
            customer_id_column="coluna_inexistente"
        )
        # Este pode funcionar se o sistema usar alternativas
        
        self.log_test("SUCCESS", "Tratamento de erros testado",
                     bad_file_handled=True,
                     bad_type_handled=True)
    
    def test_edge_cases(self):
        """Teste de casos extremos."""
        self.log_test("INFO", "Testando casos extremos")
        
        # Teste com min_transactions alto
        result_high_min = self.insights_engine._run(
            analysis_type="behavioral_segmentation",
            data_csv=self.real_data_path,
            min_transactions=100  # Valor alto que pode filtrar todos os clientes
        )
        
        # Deve funcionar ou retornar erro explicativo
        assert isinstance(result_high_min, str), "Resultado deve ser string"
        
        # Teste com prediction_horizon extremo
        result_extreme_horizon = self.insights_engine._run(
            analysis_type="churn_prediction",
            data_csv=self.real_data_path,
            prediction_horizon=365  # 1 ano
        )
        
        assert isinstance(result_extreme_horizon, str), "Resultado deve ser string"
        
        self.log_test("SUCCESS", "Casos extremos testados")
    
    def test_output_quality_and_formatting(self):
        """Teste de qualidade e formatação da saída."""
        self.log_test("INFO", "Testando qualidade da saída")
        
        result = self.insights_engine._run(
            analysis_type="behavioral_segmentation",
            data_csv=self.real_data_path,
            customer_id_column="Codigo_Cliente"
        )
        
        # Verificar estrutura JSON se aplicável
        try:
            parsed_result = json.loads(result)
            
            # Verificar campos obrigatórios se for JSON
            expected_top_level_keys = ["metadata", "analysis_results", "key_insights"]
            found_keys = [key for key in expected_top_level_keys if key in parsed_result]
            
            if len(found_keys) >= 2:
                self.log_test("SUCCESS", "Saída JSON bem estruturada",
                             found_keys=len(found_keys),
                             total_keys=len(parsed_result.keys()))
            else:
                self.log_test("WARNING", "Saída JSON com estrutura diferente do esperado")
                
        except json.JSONDecodeError:
            # Se não for JSON, verificar qualidade textual
            lines = result.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            
            self.log_test("INFO", "Saída em formato texto",
                         total_lines=len(lines),
                         non_empty_lines=len(non_empty_lines))
        
        # Verificar se há informações úteis
        useful_terms = ["cliente", "customer", "análise", "analysis", "insight", "recomendação"]
        found_useful = [term for term in useful_terms if term.lower() in result.lower()]
        
        assert len(found_useful) >= 3, f"Saída deve conter informações úteis: {found_useful}"
        
        self.log_test("SUCCESS", "Qualidade da saída verificada",
                     useful_terms_found=len(found_useful))
    
    def test_performance_benchmarks(self):
        """Teste de benchmarks de performance."""
        self.log_test("INFO", "Testando performance")
        
        # Benchmark da análise mais comum
        start_time = time.time()
        memory_before = tracemalloc.start()
        
        result = self.insights_engine._run(
            analysis_type="behavioral_segmentation",
            data_csv=self.real_data_path,
            customer_id_column="Codigo_Cliente"
        )
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Benchmarks aceitáveis (ajustar conforme necessário)
        max_execution_time = 30.0  # 30 segundos máximo
        max_memory_mb = 500  # 500MB máximo
        
        actual_memory_mb = peak / 1024 / 1024
        
        assert execution_time <= max_execution_time, \
            f"Tempo de execução muito alto: {execution_time:.2f}s > {max_execution_time}s"
        
        assert actual_memory_mb <= max_memory_mb, \
            f"Uso de memória muito alto: {actual_memory_mb:.2f}MB > {max_memory_mb}MB"
        
        self.log_test("SUCCESS", "Performance dentro dos limites",
                     execution_time=round(execution_time, 2),
                     memory_mb=round(actual_memory_mb, 2),
                     max_time=max_execution_time,
                     max_memory_mb=max_memory_mb)
    
    def teardown_method(self, method):
        """Cleanup após cada teste."""
        total_time = time.time() - self.start_time
        
        print(f"\n📊 RESUMO DO TESTE: {method.__name__}")
        print(f"    ⏱️  Tempo total: {total_time:.2f}s")
        print(f"    📝 Logs gerados: {len(self.test_logs)}")
        
        if hasattr(self, 'test_logs') and self.test_logs:
            success_logs = [log for log in self.test_logs if log['level'] == 'SUCCESS']
            error_logs = [log for log in self.test_logs if log['level'] == 'ERROR']
            
            print(f"    ✅ Sucessos: {len(success_logs)}")
            if error_logs:
                print(f"    ❌ Erros: {len(error_logs)}")
                for error_log in error_logs:
                    print(f"        - {error_log['message']}")
        
        print("─" * 50)


# ==========================================
# FIXTURE PARA DADOS DE TESTE
# ==========================================

@pytest.fixture
def real_vendas_data():
    """Fixture para dados reais de vendas."""
    # Procurar arquivo de dados reais
    possible_paths = [
        "data/vendas.csv",
        "../data/vendas.csv", 
        "../../data/vendas.csv",
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "vendas.csv")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Se não encontrar, criar dados mínimos para teste
    import tempfile
    
    minimal_data = {
        'Data': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-02-01', '2023-02-02'],
        'Total_Liquido': [1000, 1500, 800, 2000, 1200],
        'Quantidade': [1, 2, 1, 3, 1],
        'Codigo_Cliente': ['C001', 'C002', 'C001', 'C003', 'C002'],
        'Nome_Cliente': ['João Silva', 'Maria Santos', 'João Silva', 'Ana Costa', 'Maria Santos']
    }
    
    df = pd.DataFrame(minimal_data)
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        df.to_csv(f.name, sep=';', index=False)
        return f.name


# ==========================================
# EXECUÇÃO STANDALONE PARA TESTES
# ==========================================

if __name__ == "__main__":
    print("🧪 EXECUTANDO TESTES CUSTOMER INSIGHTS ENGINE - MODO STANDALONE")
    print("=" * 60)
    
    # Criar instância de teste
    test_instance = TestCustomerInsightsEngine()
    
    # Configurar dados de teste
    data_path = None
    for possible_path in ["data/vendas.csv", "../data/vendas.csv", "../../data/vendas.csv"]:
        if os.path.exists(possible_path):
            data_path = possible_path
            break
    
    if not data_path:
        print("❌ Arquivo de dados não encontrado. Criando dados mínimos...")
        # Criar dados mínimos como fallback
        minimal_data = {
            'Data': ['2023-01-01', '2023-01-02', '2023-01-03'] * 10,
            'Total_Liquido': [1000, 1500, 800] * 10,
            'Quantidade': [1, 2, 1] * 10,
            'Codigo_Cliente': ['C001', 'C002', 'C003'] * 10
        }
        
        import tempfile
        df = pd.DataFrame(minimal_data)
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            df.to_csv(f.name, sep=';', index=False)
            data_path = f.name
    
    test_instance.setup_standalone(data_path)
    
    # Lista de testes para executar
    tests_to_run = [
        "test_data_loading_and_validation",
        "test_customer_data_preparation", 
        "test_behavioral_segmentation_analysis",
        "test_lifecycle_analysis",
        "test_churn_prediction_analysis",
        "test_value_analysis",
        "test_preference_mining_analysis",
        "test_journey_mapping_analysis",
        "test_error_handling_comprehensive",
        "test_edge_cases",
        "test_output_quality_and_formatting",
        "test_performance_benchmarks"
    ]
    
    successful_tests = 0
    failed_tests = 0
    
    for test_name in tests_to_run:
        print(f"\n🔄 Executando: {test_name}")
        print("-" * 40)
        
        try:
            test_method = getattr(test_instance, test_name)
            test_method()
            successful_tests += 1
            print(f"✅ {test_name} - PASSOU")
            
        except Exception as e:
            failed_tests += 1
            print(f"❌ {test_name} - FALHOU")
            print(f"   Erro: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            if hasattr(test_instance, 'teardown_method'):
                try:
                    method_mock = type('Method', (), {'__name__': test_name})()
                    test_instance.teardown_method(method_mock)
                except:
                    pass
    
    # Resumo final
    print(f"\n🏁 RESUMO FINAL DOS TESTES")
    print("=" * 60)
    print(f"✅ Testes bem-sucedidos: {successful_tests}")
    print(f"❌ Testes falharam: {failed_tests}")
    print(f"📊 Taxa de sucesso: {(successful_tests/(successful_tests+failed_tests)*100):.1f}%")
    
    if failed_tests == 0:
        print("🎉 TODOS OS TESTES PASSARAM!")
    else:
        print(f"⚠️  {failed_tests} testes falharam - revisar implementação") 