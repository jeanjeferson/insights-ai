"""
EXEMPLO: Comportamento Padrão YTD no FinancialDataExporter
=========================================================

Este exemplo demonstra como o YTD (Year-to-Date) funciona como padrão.
"""

from src.insights.tools.financial_data_exporter import FinancialDataExporter

def demonstrar_ytd_padrao():
    """Demonstra o comportamento padrão YTD"""
    
    exporter = FinancialDataExporter()
    
    print("🎯 COMPORTAMENTO PADRÃO: YTD (Year-to-Date)")
    print("=" * 60)
    
    print("\n📅 QUANDO USAR YTD (PADRÃO):")
    print("   • Sem parâmetros de período")
    print("   • Analisa do início do ano atual até hoje")
    print("   • Compara com YTD do ano anterior")
    print("   • Ideal para relatórios mensais e acompanhamento")
    
    print("\n🔧 CÓDIGO:")
    print("   from src.insights.tools.financial_data_exporter import FinancialDataExporter")
    print("   exporter = FinancialDataExporter()")
    print("   # SEM parâmetros = YTD automático")
    print("   result = exporter._run(data_csv='data/vendas.csv')")
    
    print("\n🚀 EXECUTANDO EXEMPLO YTD...")
    result = exporter._run(
        data_csv='data/vendas.csv',
        output_path='test_results/exemplo_ytd_padrao.csv'
    )
    
    print("\n✅ RESULTADO:")
    print(result[:800])
    
    print("\n📊 PRINCIPAIS CARACTERÍSTICAS DO YTD:")
    print("   ✅ Tipo de Análise: 'ytd'")
    print("   ✅ Período: 'YTD 2025 (Jan-May)'")
    print("   ✅ Comparação: YTD 2025 vs YTD 2024")
    print("   ✅ Projeção: Baseada no progresso do ano (41% do ano)")
    print("   ✅ Ideal para: Dashboards, relatórios gerenciais, KPIs")
    
    print(f"\n📁 Arquivo YTD gerado: test_results/exemplo_ytd_padrao.csv")

if __name__ == "__main__":
    demonstrar_ytd_padrao() 