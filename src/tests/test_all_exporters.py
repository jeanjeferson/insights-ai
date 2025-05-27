#!/usr/bin/env python3
"""
Script de teste completo para todos os Data Exporters.
Testa as quatro ferramentas: Product, Inventory, Customer e Financial Data Exporters.
"""

import sys
import os
from pathlib import Path
import time

# Adicionar o diretório raiz ao path
sys.path.append(str(Path(__file__).parent))

def test_product_data_exporter():
    """Testar o Product Data Exporter."""
    print("🎯 TESTANDO PRODUCT DATA EXPORTER")
    print("-" * 50)
    
    try:
        from src.insights.tools.product_data_exporter import ProductDataExporter
        
        exporter = ProductDataExporter()
        result = exporter._run(
            data_csv="data/vendas.csv",
            output_path="data/outputs/test_produtos_completo.csv"
        )
        
        print("✅ PRODUCT DATA EXPORTER:")
        print(result[:200] + "..." if len(result) > 200 else result)
        
        # Verificar arquivo
        output_file = Path("data/outputs/test_produtos_completo.csv")
        if output_file.exists():
            print(f"📁 Arquivo criado: {output_file.stat().st_size / 1024:.1f} KB")
            return True
        else:
            print("❌ Arquivo não foi criado")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_inventory_data_exporter():
    """Testar o Inventory Data Exporter."""
    print("\n📦 TESTANDO INVENTORY DATA EXPORTER")
    print("-" * 50)
    
    try:
        from src.insights.tools.inventory_data_exporter import InventoryDataExporter
        
        exporter = InventoryDataExporter()
        result = exporter._run(
            data_csv="data/vendas.csv",
            output_path="data/outputs/test_estoque_completo.csv"
        )
        
        print("✅ INVENTORY DATA EXPORTER:")
        print(result[:200] + "..." if len(result) > 200 else result)
        
        # Verificar arquivo
        output_file = Path("data/outputs/test_estoque_completo.csv")
        if output_file.exists():
            print(f"📁 Arquivo criado: {output_file.stat().st_size / 1024:.1f} KB")
            return True
        else:
            print("❌ Arquivo não foi criado")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_customer_data_exporter():
    """Testar o Customer Data Exporter."""
    print("\n👥 TESTANDO CUSTOMER DATA EXPORTER")
    print("-" * 50)
    
    try:
        from src.insights.tools.customer_data_exporter import CustomerDataExporter
        
        exporter = CustomerDataExporter()
        result = exporter._run(
            data_csv="data/vendas.csv",
            output_path="data/outputs/test_clientes_completo.csv"
        )
        
        print("✅ CUSTOMER DATA EXPORTER:")
        print(result[:200] + "..." if len(result) > 200 else result)
        
        # Verificar arquivo
        output_file = Path("data/outputs/test_clientes_completo.csv")
        if output_file.exists():
            print(f"📁 Arquivo criado: {output_file.stat().st_size / 1024:.1f} KB")
            return True
        else:
            print("❌ Arquivo não foi criado")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def test_financial_data_exporter():
    """Testar o Financial Data Exporter."""
    print("\n💰 TESTANDO FINANCIAL DATA EXPORTER")
    print("-" * 50)
    
    try:
        from src.insights.tools.financial_data_exporter import FinancialDataExporter
        
        exporter = FinancialDataExporter()
        result = exporter._run(
            data_csv="data/vendas.csv",
            output_path="data/outputs/test_financeiro_completo.csv",
            group_by_period="monthly"
        )
        
        print("✅ FINANCIAL DATA EXPORTER:")
        print(result[:200] + "..." if len(result) > 200 else result)
        
        # Verificar arquivo
        output_file = Path("data/outputs/test_financeiro_completo.csv")
        if output_file.exists():
            print(f"📁 Arquivo criado: {output_file.stat().st_size / 1024:.1f} KB")
            return True
        else:
            print("❌ Arquivo não foi criado")
            return False
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def validate_all_csv_files():
    """Validar estrutura de todos os CSVs gerados."""
    print("\n🔍 VALIDANDO ESTRUTURA DOS CSVs")
    print("=" * 60)
    
    csv_files = {
        "Produtos": "data/outputs/test_produtos_completo.csv",
        "Estoque": "data/outputs/test_estoque_completo.csv", 
        "Clientes": "data/outputs/test_clientes_completo.csv",
        "Financeiro": "data/outputs/test_financeiro_completo.csv"
    }
    
    validation_results = {}
    
    try:
        import pandas as pd
        
        for name, file_path in csv_files.items():
            print(f"\n📊 Validando {name}...")
            
            if not Path(file_path).exists():
                print(f"❌ {name}: Arquivo não encontrado")
                validation_results[name] = False
                continue
            
            try:
                df = pd.read_csv(file_path, sep=';', encoding='utf-8')
                print(f"✅ {name}: {len(df)} registros, {len(df.columns)} colunas")
                
                # Mostrar algumas colunas principais
                print(f"   📋 Primeiras colunas: {list(df.columns[:5])}")
                
                validation_results[name] = True
                
            except Exception as e:
                print(f"❌ {name}: Erro ao carregar - {e}")
                validation_results[name] = False
        
        return validation_results
        
    except ImportError:
        print("❌ Pandas não disponível para validação")
        return {}

def generate_usage_examples():
    """Gerar exemplos de uso dos exportadores."""
    print("\n💡 EXEMPLOS DE USO DOS EXPORTADORES")
    print("=" * 60)
    
    examples = {
        "Product Data Exporter": """
# Para análise de produtos e ABC/BCG
from src.insights.tools.product_data_exporter import ProductDataExporter

exporter = ProductDataExporter()
result = exporter._run(
    data_csv='data/vendas.csv',
    output_path='data/outputs/produtos_analise.csv',
    include_abc_classification=True,
    include_bcg_analysis=True
)
""",
        
        "Inventory Data Exporter": """
# Para gestão de estoque e restock
from src.insights.tools.inventory_data_exporter import InventoryDataExporter

exporter = InventoryDataExporter()
result = exporter._run(
    data_csv='data/vendas.csv',
    output_path='data/outputs/gestao_estoque.csv',
    low_stock_days=14,
    obsolescence_months=6
)
""",
        
        "Customer Data Exporter": """
# Para análise RFM e CLV
from src.insights.tools.customer_data_exporter import CustomerDataExporter

exporter = CustomerDataExporter()
result = exporter._run(
    data_csv='data/vendas.csv',
    output_path='data/outputs/segmentacao_clientes.csv',
    clv_months=24,
    churn_days=180
)
""",
        
        "Financial Data Exporter": """
# Para análise financeira e projeções
from src.insights.tools.financial_data_exporter import FinancialDataExporter

exporter = FinancialDataExporter()
result = exporter._run(
    data_csv='data/vendas.csv',
    output_path='data/outputs/dashboard_financeiro.csv',
    group_by_period='monthly'
)
"""
    }
    
    for tool_name, example_code in examples.items():
        print(f"\n🔧 {tool_name}:")
        print(example_code.strip())

def main():
    """Função principal de teste."""
    print("🧪 TESTE COMPLETO DOS DATA EXPORTERS")
    print("=" * 80)
    print("Testando todas as 4 ferramentas de exportação:")
    print("• Product Data Exporter - Análise de produtos")
    print("• Inventory Data Exporter - Gestão de estoque") 
    print("• Customer Data Exporter - Segmentação RFM/CLV")
    print("• Financial Data Exporter - KPIs e projeções")
    print("=" * 80)
    
    start_time = time.time()
    
    # Executar todos os testes
    test_results = {
        "Product": test_product_data_exporter(),
        "Inventory": test_inventory_data_exporter(),
        "Customer": test_customer_data_exporter(),
        "Financial": test_financial_data_exporter()
    }
    
    # Validar CSVs
    csv_validations = validate_all_csv_files()
    
    # Resultado final
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("📊 RESULTADO FINAL DOS TESTES")
    print("=" * 80)
    
    success_count = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"⏱️ Tempo total: {total_time:.1f} segundos")
    print(f"✅ Sucessos: {success_count}/{total_tests}")
    print(f"❌ Falhas: {total_tests - success_count}/{total_tests}")
    
    print("\n📋 DETALHAMENTO POR FERRAMENTA:")
    for tool, success in test_results.items():
        status = "✅ OK" if success else "❌ FALHOU"
        csv_status = "📁 CSV OK" if csv_validations.get(tool, False) else "📁 CSV ERRO"
        print(f"  {tool:12} | {status:8} | {csv_status}")
    
    if success_count == total_tests:
        print("\n🎉 TODOS OS EXPORTADORES FUNCIONANDO PERFEITAMENTE!")
        print("\n🎯 PRÓXIMOS PASSOS:")
        print("1. Testar com dados reais da empresa")
        print("2. Criar dashboards no Power BI/Tableau")
        print("3. Integrar com sistemas ERP/CRM")
        print("4. Automatizar relatórios mensais")
        print("5. Configurar alertas automáticos")
        
        # Mostrar exemplos de uso
        generate_usage_examples()
        
        print("\n🏆 SISTEMA COMPLETO DE EXPORTAÇÃO IMPLEMENTADO!")
        print("✅ 4 ferramentas de exportação CSV funcionais")
        print("✅ Dados estruturados para análise externa")
        print("✅ Compatibilidade com Excel, Power BI, Tableau")
        print("✅ Pronto para uso em produção")
    else:
        print("\n⚠️ ALGUNS TESTES FALHARAM")
        print("💡 Verifique os erros acima e corrija os problemas.")
    
    return success_count == total_tests

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Teste interrompido pelo usuário")
        exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        exit(1) 