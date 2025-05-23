#!/usr/bin/env python3
"""
🔧 SETUP E CONFIGURAÇÃO DOS TESTES
==================================

Script para configurar o ambiente de testes do Insights-AI.
Verifica dependências, cria dados de exemplo e valida configuração.
"""

import sys
import os
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def check_dependencies():
    """Verificar dependências necessárias"""
    print("🔍 Verificando dependências...")
    
    required_packages = [
        'pandas', 'numpy', 'scipy', 'scikit-learn', 
        'matplotlib', 'seaborn'
    ]
    
    optional_packages = [
        'prophet', 'plotly', 'psutil'
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"  ❌ {package} - OBRIGATÓRIO")
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_optional.append(package)
            print(f"  ⚠️ {package} - OPCIONAL")
    
    if missing_required:
        print(f"\n❌ Pacotes obrigatórios ausentes: {missing_required}")
        print("Execute: pip install " + " ".join(missing_required))
        return False
    
    if missing_optional:
        print(f"\n⚠️ Pacotes opcionais ausentes: {missing_optional}")
        print("Para testes completos, execute: pip install " + " ".join(missing_optional))
    
    print("\n✅ Dependências verificadas!")
    return True

def create_sample_data():
    """Criar dados de exemplo para testes"""
    print("\n📊 Criando dados de exemplo...")
    
    # Verificar se data/vendas.csv já existe
    data_dir = Path("data")
    data_file = data_dir / "vendas.csv"
    
    if data_file.exists():
        print(f"  ℹ️ Arquivo {data_file} já existe")
        
        # Verificar se é válido
        try:
            df = pd.read_csv(data_file, sep=';', encoding='utf-8')
            if len(df) > 0 and 'Total_Liquido' in df.columns:
                print(f"  ✅ Arquivo válido com {len(df)} registros")
                return True
            else:
                print("  ⚠️ Arquivo existe mas parece inválido, criando novo...")
        except Exception as e:
            print(f"  ⚠️ Erro ao ler arquivo existente: {e}")
            print("  🔄 Criando novo arquivo...")
    
    # Criar diretório se não existir
    data_dir.mkdir(exist_ok=True)
    
    # Gerar dados de exemplo
    np.random.seed(42)
    
    print("  🏗️ Gerando dados de vendas...")
    
    # 6 meses de dados
    start_date = datetime(2024, 6, 1)
    end_date = datetime(2024, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    
    # Produtos e categorias realistas para joalheria
    categories = ['Anéis', 'Brincos', 'Colares', 'Pulseiras', 'Alianças', 'Pingentes']
    metals = ['Ouro', 'Prata', 'Ouro Branco', 'Ouro Rosé', 'Platina']
    collections = ['Clássica', 'Moderna', 'Vintage', 'Exclusiva', 'Sazonal']
    
    customers = [f"CLI_{i:04d}" for i in range(1, 301)]  # 300 clientes
    sellers = [f"VEND_{i:02d}" for i in range(1, 21)]     # 20 vendedores
    
    transaction_id = 1
    
    for date in date_range:
        # Sazonalidade (mais vendas próximo ao fim do ano)
        seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365 + np.pi/2)
        
        # Padrão semanal (menos vendas no fim de semana)
        weekday_factor = 1.3 if date.weekday() < 5 else 0.4
        
        # Número de transações por dia
        daily_transactions = max(1, int(20 * seasonal_factor * weekday_factor * np.random.uniform(0.6, 1.4)))
        
        for _ in range(daily_transactions):
            customer = np.random.choice(customers)
            seller = np.random.choice(sellers)
            category = np.random.choice(categories)
            metal = np.random.choice(metals)
            collection = np.random.choice(collections)
            
            # Preços baseados na categoria e metal
            base_prices = {
                'Anéis': 1200, 'Brincos': 650, 'Colares': 1800,
                'Pulseiras': 950, 'Alianças': 2400, 'Pingentes': 450
            }
            
            metal_multipliers = {
                'Ouro': 1.0, 'Prata': 0.35, 'Ouro Branco': 1.15,
                'Ouro Rosé': 1.08, 'Platina': 1.75
            }
            
            collection_multipliers = {
                'Clássica': 0.9, 'Moderna': 1.0, 'Vintage': 1.2,
                'Exclusiva': 1.8, 'Sazonal': 0.85
            }
            
            # Calcular preços
            base_price = base_prices[category]
            metal_mult = metal_multipliers[metal]
            collection_mult = collection_multipliers[collection]
            
            quantidade = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.6, 0.2, 0.1, 0.06, 0.03, 0.01])
            preco_unitario = base_price * metal_mult * collection_mult * np.random.uniform(0.8, 1.2)
            preco_tabela = preco_unitario * 1.3  # 30% markup
            desconto = preco_tabela * np.random.uniform(0, 0.2)  # 0-20% desconto
            total_liquido = (preco_tabela - desconto) * quantidade
            custo_produto = preco_unitario * 0.4 * quantidade  # 40% do preço final
            
            # Produto único baseado na combinação
            product_hash = hash(f"{category}_{metal}_{collection}") % 1000
            codigo_produto = f"PROD_{product_hash:04d}"
            
            data.append({
                'Data': date.strftime('%Y-%m-%d'),
                'Ano': date.year,
                'Mes': date.month,
                'Codigo_Cliente': customer,
                'Nome_Cliente': f"Cliente {customer.split('_')[1]}",
                'Sexo': np.random.choice(['M', 'F']),
                'Estado_Civil': np.random.choice(['Solteiro', 'Casado', 'Divorciado', 'Viúvo'], p=[0.3, 0.5, 0.15, 0.05]),
                'Idade': np.random.randint(18, 75),
                'Cidade': np.random.choice([
                    'São Paulo', 'Rio de Janeiro', 'Belo Horizonte', 'Brasília', 
                    'Salvador', 'Fortaleza', 'Curitiba', 'Porto Alegre'
                ], p=[0.25, 0.2, 0.15, 0.1, 0.1, 0.05, 0.08, 0.07]),
                'Estado': np.random.choice(['SP', 'RJ', 'MG', 'DF', 'BA', 'CE', 'PR', 'RS'], 
                                         p=[0.25, 0.2, 0.15, 0.1, 0.1, 0.05, 0.08, 0.07]),
                'Codigo_Vendedor': seller,
                'Nome_Vendedor': f"Vendedor {seller.split('_')[1]}",
                'Codigo_Produto': codigo_produto,
                'Descricao_Produto': f"{category} {metal} {collection}",
                'Estoque_Atual': np.random.randint(0, 50),
                'Colecao': collection,
                'Grupo_Produto': category,
                'Subgrupo_Produto': f"Sub{category}",
                'Metal': metal,
                'Quantidade': quantidade,
                'Custo_Produto': round(custo_produto, 2),
                'Preco_Tabela': round(preco_tabela, 2),
                'Desconto_Aplicado': round(desconto, 2),
                'Total_Liquido': round(total_liquido, 2)
            })
            
            transaction_id += 1
    
    # Criar DataFrame
    df = pd.DataFrame(data)
    
    # Salvar arquivo
    df.to_csv(data_file, sep=';', index=False, encoding='utf-8')
    
    print(f"  ✅ Arquivo criado: {data_file}")
    print(f"  📊 {len(df)} registros")
    print(f"  📅 Período: {df['Data'].min()} até {df['Data'].max()}")
    print(f"  💰 Vendas totais: R$ {df['Total_Liquido'].sum():,.2f}")
    print(f"  🏆 Produtos únicos: {df['Codigo_Produto'].nunique()}")
    print(f"  👥 Clientes únicos: {df['Codigo_Cliente'].nunique()}")
    
    return True

def validate_environment():
    """Validar ambiente de testes"""
    print("\n🔍 Validando ambiente...")
    
    # Verificar estrutura de diretórios
    required_dirs = ['src', 'src/insights', 'src/tests', 'data']
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✅ Diretório {dir_path}")
        else:
            print(f"  ❌ Diretório {dir_path} ausente")
            return False
    
    # Verificar arquivos principais de teste
    test_files = [
        'src/tests/test_main.py',
        'src/tests/conftest.py',
        'src/tests/test_kpi_calculator_tool.py'
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            print(f"  ✅ Arquivo {file_path}")
        else:
            print(f"  ❌ Arquivo {file_path} ausente")
            return False
    
    print("\n✅ Ambiente validado!")
    return True

def run_quick_test():
    """Executar teste rápido para verificar funcionamento"""
    print("\n🚀 Executando teste rápido...")
    
    try:
        # Tentar importar e executar teste básico
        sys.path.append('src')
        from tests.test_data_validation import test_data_validation
        
        result = test_data_validation(verbose=True, quick=True)
        
        if result['success']:
            print("  ✅ Teste rápido passou!")
            return True
        else:
            print("  ❌ Teste rápido falhou:")
            for error in result['errors']:
                print(f"    - {error}")
            return False
            
    except Exception as e:
        print(f"  ❌ Erro no teste rápido: {str(e)}")
        return False

def main():
    """Função principal do setup"""
    print("🔧 SETUP DOS TESTES INSIGHTS-AI")
    print("=" * 50)
    
    success = True
    
    # 1. Verificar dependências
    if not check_dependencies():
        success = False
    
    # 2. Validar ambiente
    if success and not validate_environment():
        success = False
    
    # 3. Criar dados de exemplo
    if success and not create_sample_data():
        success = False
    
    # 4. Teste rápido
    if success and not run_quick_test():
        success = False
    
    # Resultado final
    print("\n" + "=" * 50)
    if success:
        print("✅ SETUP CONCLUÍDO COM SUCESSO!")
        print("\n🎯 Próximos passos:")
        print("1. Execute: python src/tests/test_main.py --verbose")
        print("2. Para testes rápidos: python src/tests/test_main.py --quick --verbose")
        print("3. Para ferramentas específicas: python src/tests/test_main.py --tool kpi --verbose")
        print("\n📖 Consulte src/tests/README.md para mais informações")
    else:
        print("❌ SETUP FALHOU!")
        print("\n🔧 Ações necessárias:")
        print("1. Instale as dependências obrigatórias")
        print("2. Verifique a estrutura de diretórios")
        print("3. Execute este script novamente")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
