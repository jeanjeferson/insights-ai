#!/usr/bin/env python3
"""
🔧 SETUP SIMPLES DOS TESTES
===========================

Script simplificado para configurar o ambiente de testes do Insights-AI.
Verifica dependências básicas e cria dados de exemplo simples.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def check_basic_dependencies():
    """Verificar dependências básicas necessárias"""
    print("🔍 Verificando dependências básicas...")
    
    required_packages = ['pandas', 'numpy']
    optional_packages = ['prophet', 'plotly']
    
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
        print(f"\n❌ Instale: pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"\n⚠️ Para testes completos: pip install {' '.join(missing_optional)}")
    
    print("\n✅ Dependências verificadas!")
    return True

def create_simple_sample_data():
    """Criar dados de exemplo simples para testes"""
    print("\n📊 Criando dados de exemplo...")
    
    data_dir = Path("data")
    data_file = data_dir / "vendas.csv"
    
    # Verificar se já existe
    if data_file.exists():
        try:
            df = pd.read_csv(data_file, sep=';', encoding='utf-8')
            if len(df) > 0 and 'Total_Liquido' in df.columns:
                print(f"  ✅ Arquivo válido existente com {len(df)} registros")
                return True
        except:
            print("  ⚠️ Arquivo inválido, criando novo...")
    
    # Criar diretório
    data_dir.mkdir(exist_ok=True)
    
    # Gerar dados simples
    np.random.seed(42)
    
    print("  🏗️ Gerando dados simples...")
    
    # 3 meses de dados simples
    dates = pd.date_range('2024-01-01', periods=90, freq='D')
    data = []
    
    categories = ['Anéis', 'Brincos', 'Colares', 'Pulseiras']
    customers = [f"CLI_{i:03d}" for i in range(1, 51)]  # 50 clientes
    
    for i, date in enumerate(dates):
        # 5-15 transações por dia
        daily_transactions = np.random.randint(5, 16)
        
        for _ in range(daily_transactions):
            data.append({
                'Data': date.strftime('%Y-%m-%d'),
                'Codigo_Cliente': np.random.choice(customers),
                'Codigo_Produto': f"PROD_{np.random.randint(1, 21):03d}",
                'Categoria': np.random.choice(categories),
                'Quantidade': np.random.randint(1, 4),
                'Total_Liquido': np.random.uniform(100, 2000),
                'Preco_Unitario': np.random.uniform(50, 500)
            })
    
    # Criar e salvar DataFrame
    df = pd.DataFrame(data)
    df.to_csv(data_file, sep=';', index=False, encoding='utf-8')
    
    print(f"  ✅ Arquivo criado: {data_file}")
    print(f"  📊 {len(df)} registros")
    print(f"  📅 Período: {df['Data'].min()} até {df['Data'].max()}")
    print(f"  💰 Total: R$ {df['Total_Liquido'].sum():,.2f}")
    
    return True

def run_quick_validation():
    """Executar validação rápida"""
    print("\n🚀 Executando validação rápida...")
    
    try:
        # Testar import das ferramentas principais
        sys.path.append(str(Path(__file__).parent.parent))
        
        from insights.tools.kpi_calculator_tool import KPICalculatorTool
        from insights.tools.statistical_analysis_tool import StatisticalAnalysisTool
        from insights.tools.business_intelligence_tool import BusinessIntelligenceTool
        
        print("  ✅ Ferramentas v3.0 importadas com sucesso")
        
        # Testar dados
        data_file = Path("data/vendas.csv")
        if data_file.exists():
            df = pd.read_csv(data_file, sep=';', encoding='utf-8')
            if len(df) > 0:
                print(f"  ✅ Dados válidos: {len(df)} registros")
            else:
                print("  ❌ Dados vazios")
                return False
        else:
            print("  ❌ Arquivo de dados não encontrado")
            return False
        
        print("  ✅ Validação concluída com sucesso!")
        return True
        
    except Exception as e:
        print(f"  ❌ Erro na validação: {str(e)}")
        return False

def main():
    """Função principal do setup simplificado"""
    print("🔧 SETUP SIMPLES - INSIGHTS-AI")
    print("=" * 35)
    
    success = True
    
    # 1. Verificar dependências
    if not check_basic_dependencies():
        success = False
    
    # 2. Criar dados de exemplo
    if success and not create_simple_sample_data():
        success = False
    
    # 3. Validação rápida
    if success and not run_quick_validation():
        success = False
    
    # Resultado final
    print("\n" + "=" * 35)
    if success:
        print("✅ SETUP CONCLUÍDO COM SUCESSO!")
        print("\n🎯 Próximos passos:")
        print("1. Execute: python test_performance.py")
        print("2. Execute: python test_integration.py")
        print("3. Execute: python test_advanced_tools.py")
    else:
        print("❌ SETUP FALHOU!")
        print("\n🔧 Instale as dependências e execute novamente")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
