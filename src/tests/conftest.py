"""
Configurações e fixtures para os testes do Insights-AI
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path
import json
from unittest.mock import Mock, patch

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent.parent))

# Configuração global para testes
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

@pytest.fixture(scope="session")
def test_config():
    """Configurações globais para toda a sessão de testes"""
    return {
        'test_data_rows': {
            'small': 100,
            'medium': 1000, 
            'large': 5000,
            'xlarge': 10000
        },
        'timeout_seconds': 60,
        'acceptable_error_rate': 0.05,
        'performance_thresholds': {
            'kpi_calculation': 10,  # segundos
            'prophet_forecast': 30,
            'statistical_analysis': 15,
            'visualization': 20
        },
        'quality_gates': {
            'min_success_rate': 80,
            'max_memory_usage_mb': 500,
            'max_execution_time': 300
        }
    }

@pytest.fixture(scope="session")
def temp_data_dir():
    """Diretório temporário para dados de teste"""
    temp_dir = tempfile.mkdtemp(prefix="insights_ai_test_")
    yield temp_dir
    # Cleanup
    try:
        shutil.rmtree(temp_dir)
    except:
        pass

@pytest.fixture
def sample_vendas_data():
    """Fixture com dados de vendas simulados para testes"""
    np.random.seed(42)  # Para reprodutibilidade
    
    # Gerar dados de vendas realistas
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    customer_ids = [f"CLI_{i:04d}" for i in range(1, 501)]  # 500 clientes
    product_codes = [f"PROD_{i:04d}" for i in range(1, 201)]  # 200 produtos
    
    grupos_produto = ['Anéis', 'Brincos', 'Colares', 'Pulseiras', 'Alianças', 'Pingentes']
    metals = ['Ouro', 'Prata', 'Ouro Branco', 'Ouro Rosé', 'Platina']
    colecoes = ['Clássica', 'Moderna', 'Vintage', 'Exclusiva', 'Sazonal']
    
    for date in date_range:
        # Número de transações por dia (variação sazonal)
        base_transactions = 50
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
        daily_transactions = int(base_transactions * seasonal_factor * np.random.uniform(0.5, 1.5))
        
        for _ in range(daily_transactions):
            customer_id = np.random.choice(customer_ids)
            product_code = np.random.choice(product_codes)
            grupo = np.random.choice(grupos_produto)
            metal = np.random.choice(metals)
            colecao = np.random.choice(colecoes)
            
            # Preços baseados no tipo de produto e metal
            base_price = {
                'Anéis': 1500, 'Brincos': 800, 'Colares': 2000,
                'Pulseiras': 1200, 'Alianças': 2500, 'Pingentes': 600
            }[grupo]
            
            metal_multiplier = {
                'Ouro': 1.0, 'Prata': 0.4, 'Ouro Branco': 1.2,
                'Ouro Rosé': 1.1, 'Platina': 1.8
            }[metal]
            
            preco_unitario = base_price * metal_multiplier * np.random.uniform(0.7, 1.3)
            quantidade = np.random.choice([1, 1, 1, 1, 2], p=[0.7, 0.15, 0.1, 0.03, 0.02])
            total_liquido = preco_unitario * quantidade
            
            # Adicionar custo e desconto
            custo_produto = total_liquido * 0.4  # 40% de custo
            desconto = total_liquido * np.random.uniform(0, 0.15)  # 0-15% desconto
            preco_tabela = total_liquido + desconto
            
            data.append({
                'Data': date.strftime('%Y-%m-%d'),
                'Ano': date.year,
                'Mes': date.month,
                'Codigo_Cliente': customer_id,
                'Nome_Cliente': f"Cliente {customer_id}",
                'Sexo': np.random.choice(['M', 'F']),
                'Estado_Civil': np.random.choice(['Solteiro', 'Casado', 'Divorciado']),
                'Idade': np.random.randint(18, 70),
                'Cidade': np.random.choice(['São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Brasília']),
                'Estado': np.random.choice(['SP', 'RJ', 'MG', 'DF']),
                'Codigo_Vendedor': f"VEND_{np.random.randint(1, 21):02d}",
                'Nome_Vendedor': f"Vendedor {np.random.randint(1, 21):02d}",
                'Codigo_Produto': product_code,
                'Descricao_Produto': f"{grupo} {metal} {colecao}",
                'Estoque_Atual': np.random.randint(0, 50),
                'Colecao': colecao,
                'Grupo_Produto': grupo,
                'Subgrupo_Produto': f"Sub{grupo}",
                'Metal': metal,
                'Quantidade': quantidade,
                'Custo_Produto': custo_produto,
                'Preco_Tabela': preco_tabela,
                'Desconto_Aplicado': desconto,
                'Total_Liquido': total_liquido
            })
    
    return pd.DataFrame(data)

@pytest.fixture
def small_sample_data():
    """Fixture com dados pequenos para testes rápidos"""
    np.random.seed(42)
    n_records = 50
    
    data = []
    for i in range(n_records):
        date = datetime(2024, 1, 1) + timedelta(days=i)
        data.append({
            'Data': date.strftime('%Y-%m-%d'),
            'Ano': date.year,
            'Mes': date.month,
            'Codigo_Cliente': f"CLI_{i % 10:03d}",
            'Codigo_Produto': f"PROD_{i % 5:03d}",
            'Grupo_Produto': np.random.choice(['Anéis', 'Brincos', 'Colares']),
            'Metal': np.random.choice(['Ouro', 'Prata']),
            'Quantidade': np.random.randint(1, 3),
            'Total_Liquido': np.random.uniform(500, 3000),
            'Preco_Unitario': np.random.uniform(500, 1500),
            'Custo_Produto': np.random.uniform(200, 600)
        })
    
    return pd.DataFrame(data)

@pytest.fixture
def large_sample_data(test_config):
    """Fixture com dados grandes para testes de performance"""
    np.random.seed(42)
    n_records = test_config['test_data_rows']['large']
    
    # Geração otimizada para dados grandes
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    customers = [f"CLI_{i:04d}" for i in range(100)]
    products = [f"PROD_{i:03d}" for i in range(50)]
    
    data = {
        'Data': np.random.choice(dates, n_records).strftime('%Y-%m-%d'),
        'Codigo_Cliente': np.random.choice(customers, n_records),
        'Codigo_Produto': np.random.choice(products, n_records),
        'Grupo_Produto': np.random.choice(['Anéis', 'Brincos', 'Colares', 'Pulseiras'], n_records),
        'Metal': np.random.choice(['Ouro', 'Prata', 'Ouro Branco'], n_records),
        'Quantidade': np.random.randint(1, 4, n_records),
        'Total_Liquido': np.random.normal(1500, 500, n_records),
        'Preco_Unitario': np.random.normal(800, 200, n_records),
        'Custo_Produto': np.random.normal(400, 100, n_records)
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_csv_file(tmp_path, sample_vendas_data):
    """Fixture que cria um arquivo CSV temporário com dados de teste"""
    csv_file = tmp_path / "test_vendas.csv"
    sample_vendas_data.to_csv(csv_file, sep=';', index=False, encoding='utf-8')
    return str(csv_file)

@pytest.fixture
def real_vendas_data():
    """Fixture com dados reais de vendas para testes completos"""
    real_data_path = "data/vendas.csv"
    if os.path.exists(real_data_path):
        return real_data_path
    else:
        # Fallback para dados simulados se arquivo real não existir
        pytest.skip(f"Arquivo de dados reais não encontrado: {real_data_path}")

@pytest.fixture
def empty_csv_file(tmp_path):
    """Fixture com arquivo CSV vazio para testes de erro"""
    csv_file = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(csv_file, index=False)
    return str(csv_file)

@pytest.fixture
def invalid_csv_file(tmp_path):
    """Fixture com CSV inválido para testes de erro"""
    csv_file = tmp_path / "invalid.csv"
    with open(csv_file, 'w') as f:
        f.write("invalid,csv,data\nwith,incomplete\n")
    return str(csv_file)

@pytest.fixture
def corrupted_csv_file(tmp_path):
    """Fixture com CSV corrompido para testes de robustez"""
    csv_file = tmp_path / "corrupted.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("Data;Total_Liquido;Quantidade\n")
        f.write("2024-01-01;abc;def\n")  # Dados inválidos
        f.write("invalid_date;1000;2\n")
        f.write("2024-01-02;;\n")  # Valores vazios
    return str(csv_file)

@pytest.fixture
def prophet_data():
    """Dados formatados para Prophet"""
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    values = np.random.normal(1000, 200, len(dates)) + 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    
    return pd.DataFrame({
        'ds': dates,
        'y': np.maximum(values, 0)  # Garantir valores positivos
    })

@pytest.fixture
def prophet_json_data(prophet_data):
    """Dados Prophet em formato JSON"""
    return prophet_data.to_json(orient='records', date_format='iso')

@pytest.fixture
def mock_sql_connection():
    """Mock de conexão SQL para testes"""
    with patch('pyodbc.connect') as mock_conn:
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            ('2024-01-01', 'CLI_001', 'PROD_001', 'Anéis', 'Ouro', 1, 1500.00),
            ('2024-01-02', 'CLI_002', 'PROD_002', 'Brincos', 'Prata', 2, 800.00)
        ]
        mock_cursor.description = [
            ('Data',), ('Codigo_Cliente',), ('Codigo_Produto',),
            ('Grupo_Produto',), ('Metal',), ('Quantidade',), ('Total_Liquido',)
        ]
        mock_conn.return_value.cursor.return_value = mock_cursor
        yield mock_conn

@pytest.fixture
def performance_monitor():
    """Monitor de performance para testes"""
    import time
    import psutil
    import os
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.process = psutil.Process(os.getpid()) if hasattr(psutil, 'Process') else None
        
        def start(self):
            self.start_time = time.time()
            if self.process:
                self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        def stop(self):
            if not self.start_time:
                return {'error': 'Monitor not started'}
            
            duration = time.time() - self.start_time
            result = {'duration': duration}
            
            if self.process:
                end_memory = self.process.memory_info().rss / 1024 / 1024
                result['memory_used'] = end_memory - (self.start_memory or 0)
                result['memory_peak'] = end_memory
            
            return result
    
    return PerformanceMonitor()

@pytest.fixture
def test_environment_variables():
    """Variáveis de ambiente para testes"""
    env_vars = {
        'DB_DRIVER': 'ODBC Driver 17 for SQL Server',
        'DB_SERVER': 'localhost',
        'DB_DATABASE': 'test_db',
        'DB_UID': 'test_user',
        'DB_PWD': 'test_password',
        'DB_PORT': '1433'
    }
    
    # Salvar valores originais
    original_values = {}
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield env_vars
    
    # Restaurar valores originais
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

@pytest.fixture
def error_tracker():
    """Rastreador de erros para análise de qualidade"""
    class ErrorTracker:
        def __init__(self):
            self.errors = []
            self.warnings = []
        
        def add_error(self, error, context=None):
            self.errors.append({
                'error': str(error),
                'context': context,
                'timestamp': datetime.now()
            })
        
        def add_warning(self, warning, context=None):
            self.warnings.append({
                'warning': str(warning),
                'context': context,
                'timestamp': datetime.now()
            })
        
        def get_summary(self):
            return {
                'error_count': len(self.errors),
                'warning_count': len(self.warnings),
                'errors': self.errors,
                'warnings': self.warnings
            }
    
    return ErrorTracker()

@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Limpeza automática de arquivos temporários após cada teste"""
    yield
    
    # Limpeza após o teste
    temp_patterns = ['temp_*.csv', 'test_*.csv', '*.tmp']
    
    for pattern in temp_patterns:
        for file_path in Path('.').glob(pattern):
            try:
                file_path.unlink()
            except:
                pass

# Configurações do pytest
def pytest_configure(config):
    """Configuração do pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "advanced: marks tests for advanced tools"
    )

def pytest_collection_modifyitems(config, items):
    """Modificar itens de teste coletados"""
    for item in items:
        # Marcar testes que demoram como slow
        if "prophet" in item.nodeid or "advanced" in item.nodeid or "performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Marcar testes de integração
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Marcar testes unitários
        if any(tool in item.nodeid for tool in ["kpi", "sql", "prophet", "stats", "viz"]):
            item.add_marker(pytest.mark.unit)

# Hook para capturar resultados de teste
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capturar resultados de teste para relatórios"""
    outcome = yield
    rep = outcome.get_result()
    
    # Adicionar informações extras para relatórios
    if rep.when == "call":
        item.test_result = {
            'name': item.name,
            'outcome': rep.outcome,
            'duration': rep.duration,
            'error': str(rep.longrepr) if rep.longrepr else None
        }
