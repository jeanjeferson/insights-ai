"""
GUIA COMPLETO: Como Usar Filtros no FinancialDataExporter
=========================================================

Este arquivo demonstra todas as formas de filtrar períodos na análise financeira.
"""

from src.insights.tools.financial_data_exporter import FinancialDataExporter

def demonstrar_tipos_de_filtros():
    """Demonstra todos os tipos de filtros disponíveis"""
    
    exporter = FinancialDataExporter()
    
    print("🎯 GUIA DE FILTROS - FinancialDataExporter")
    print("=" * 60)
    
    # 1. FILTRO POR ANO ESPECÍFICO
    print("\n1️⃣ FILTRO POR ANO ESPECÍFICO:")
    print("   Analisa apenas o ano especificado")
    print("   Código:")
    print("   exporter._run(")
    print("       data_csv='data/vendas.csv',")
    print("       current_year=2024,  # ← ANO ESPECÍFICO")
    print("       output_path='resultado_2024.csv'")
    print("   )")
    
    # 2. FILTRO POR ÚLTIMOS X DIAS
    print("\n2️⃣ FILTRO POR ÚLTIMOS X DIAS:")
    print("   Analisa os últimos X dias a partir da data mais recente")
    print("   Código:")
    print("   exporter._run(")
    print("       data_csv='data/vendas.csv',")
    print("       last_x_days=90,  # ← ÚLTIMOS 90 DIAS")
    print("       output_path='resultado_90dias.csv'")
    print("   )")
    
    # 3. FILTRO POR PERÍODO CUSTOMIZADO
    print("\n3️⃣ FILTRO POR PERÍODO CUSTOMIZADO:")
    print("   Analisa um período específico definido por você")
    print("   Código:")
    print("   exporter._run(")
    print("       data_csv='data/vendas.csv',")
    print("       period_start_date='2024-01-01',  # ← DATA INÍCIO")
    print("       period_end_date='2024-06-30',    # ← DATA FIM")
    print("       output_path='resultado_semestre.csv'")
    print("   )")
    
    # 4. FILTRO PADRÃO (ANO ATUAL ATÉ HOJE)
    print("\n4️⃣ FILTRO PADRÃO (YTD - Year to Date):")
    print("   Analisa do início do ano atual até hoje")
    print("   Código:")
    print("   exporter._run(")
    print("       data_csv='data/vendas.csv'")
    print("       # SEM PARÂMETROS DE PERÍODO = YTD")
    print("   )")
    
    print("\n🔄 PRIORIDADE DOS FILTROS:")
    print("1. Período Customizado (start_date + end_date)")
    print("2. Últimos X dias (last_x_days)")
    print("3. Ano específico (current_year)")
    print("4. Ano atual até hoje (padrão)")
    
    print("\n📊 AGRUPAMENTO DOS DADOS:")
    print("Você também pode especificar como agrupar os dados:")
    print("- group_by_period='daily'     # Diário")
    print("- group_by_period='weekly'    # Semanal")
    print("- group_by_period='monthly'   # Mensal (padrão)")
    print("- group_by_period='quarterly' # Trimestral")

if __name__ == "__main__":
    demonstrar_tipos_de_filtros() 