"""
Exemplo de uso da classe FinancialDataExporter MELHORADA
========================================================

Este exemplo demonstra as novas funcionalidades implementadas:
- Filtros de período flexíveis
- Comparação YoY inteligente
- Projeções baseadas em histórico de 2 anos
- Segregação de KPIs entre base filtrada e completa
"""

from src.insights.tools.financial_data_exporter import FinancialDataExporter
from datetime import datetime, timedelta
import os

def main():
    """Demonstrar as novas funcionalidades da classe melhorada"""
    
    print("=" * 80)
    print("🚀 DEMONSTRAÇÃO: FinancialDataExporter MELHORADO")
    print("=" * 80)
    
    # Instanciar a ferramenta
    exporter = FinancialDataExporter()
    
    # Criar diretório de exemplos
    os.makedirs("exemplos_output", exist_ok=True)
    
    # ========================
    # EXEMPLO 1: ANO ESPECÍFICO
    # ========================
    print("\n📅 EXEMPLO 1: ANÁLISE DE ANO ESPECÍFICO (2024)")
    print("-" * 60)
    
    try:
        result1 = exporter._run(
            data_csv="data/vendas.csv",
            current_year=2024,
            output_path="exemplos_output/financeiro_2024.csv",
            group_by_period="monthly"
        )
        
        print("✅ SUCESSO - Análise do ano 2024 concluída")
        print(f"📁 Arquivo gerado: exemplos_output/financeiro_2024.csv")
        print("\n💡 PRINCIPAIS MELHORIAS DEMONSTRADAS:")
        print("   • Receita calculada apenas para 2024")
        print("   • Crescimento YoY: 2024 vs 2023 (mesmo período)")
        print("   • Projeção anual baseada no progresso até agora")
        print("   • Sazonalidade calculada com histórico completo")
        
    except Exception as e:
        print(f"❌ ERRO no Exemplo 1: {e}")
    
    # ========================
    # EXEMPLO 2: ÚLTIMOS 90 DIAS
    # ========================
    print("\n📅 EXEMPLO 2: ANÁLISE DOS ÚLTIMOS 90 DIAS")
    print("-" * 60)
    
    try:
        result2 = exporter._run(
            data_csv="data/vendas.csv",
            last_x_days=90,
            output_path="exemplos_output/financeiro_90dias.csv",
            group_by_period="weekly"
        )
        
        print("✅ SUCESSO - Análise dos últimos 90 dias concluída")
        print(f"📁 Arquivo gerado: exemplos_output/financeiro_90dias.csv")
        print("\n💡 PRINCIPAIS MELHORIAS DEMONSTRADAS:")
        print("   • Receita dos últimos 90 dias apenas")
        print("   • Comparação com mesmo período do ano anterior")
        print("   • Tendência semanal baseada no período")
        print("   • Projeções usando dados dos últimos 2 anos")
        
    except Exception as e:
        print(f"❌ ERRO no Exemplo 2: {e}")
    
    # ========================
    # EXEMPLO 3: PERÍODO CUSTOMIZADO
    # ========================
    print("\n📅 EXEMPLO 3: PERÍODO CUSTOMIZADO (JAN-JUN 2024)")
    print("-" * 60)
    
    try:
        result3 = exporter._run(
            data_csv="data/vendas.csv",
            period_start_date="2024-01-01",
            period_end_date="2024-06-30",
            output_path="exemplos_output/financeiro_semestre.csv",
            group_by_period="monthly"
        )
        
        print("✅ SUCESSO - Análise do semestre concluída")
        print(f"📁 Arquivo gerado: exemplos_output/financeiro_semestre.csv")
        print("\n💡 PRINCIPAIS MELHORIAS DEMONSTRADAS:")
        print("   • Receita do 1º semestre 2024")
        print("   • Comparação com 1º semestre 2023")
        print("   • Análise mensal detalhada do período")
        print("   • KPIs específicos para o semestre")
        
    except Exception as e:
        print(f"❌ ERRO no Exemplo 3: {e}")
    
    # ========================
    # EXEMPLO 4: ANO ATUAL (PADRÃO)
    # ========================
    print("\n📅 EXEMPLO 4: ANO ATUAL ATÉ HOJE (COMPORTAMENTO PADRÃO)")
    print("-" * 60)
    
    try:
        result4 = exporter._run(
            data_csv="data/vendas.csv",
            # Sem parâmetros de período = ano atual até hoje
            output_path="exemplos_output/financeiro_ytd.csv",
            group_by_period="monthly"
        )
        
        print("✅ SUCESSO - Análise YTD concluída")
        print(f"📁 Arquivo gerado: exemplos_output/financeiro_ytd.csv")
        print("\n💡 PRINCIPAIS MELHORIAS DEMONSTRADAS:")
        print("   • Receita do ano atual até hoje")
        print("   • Comparação YTD vs ano anterior")
        print("   • Projeção para fim do ano")
        print("   • Performance acumulada do ano")
        
    except Exception as e:
        print(f"❌ ERRO no Exemplo 4: {e}")
    
    # ========================
    # RESUMO DAS MELHORIAS
    # ========================
    print("\n" + "=" * 80)
    print("📊 RESUMO DAS PRINCIPAIS MELHORIAS IMPLEMENTADAS")
    print("=" * 80)
    
    melhorias = [
        "🎯 FILTROS DE PERÍODO FLEXÍVEIS:",
        "   • current_year: Ano específico (ex: 2024)",
        "   • last_x_days: Últimos X dias (ex: 90)",  
        "   • period_start_date + period_end_date: Período customizado",
        "   • Padrão: Ano atual até hoje",
        "",
        "📈 COMPARAÇÃO YoY INTELIGENTE:",
        "   • Compara períodos equivalentes (jan-jun 2024 vs jan-jun 2023)",
        "   • YoY_Receita_Growth_Pct: Crescimento ano sobre ano",
        "   • Confiabilidade baseada na disponibilidade de dados",
        "",
        "🔮 PROJEÇÕES MELHORADAS:",
        "   • Baseadas nos últimos 2 anos completos",
        "   • Sazonalidade robusta com histórico longo",
        "   • Confiabilidade de 95% com dados suficientes",
        "",
        "⚖️ SEGREGAÇÃO DE KPIS:",
        "   • Base Filtrada: Receita, crescimento, KPIs operacionais",
        "   • Base Completa: Sazonalidade, benchmarks históricos",
        "   • Base Híbrida: Projeções (2 anos) + análise atual (filtrado)",
        "",
        "📋 NOVAS COLUNAS PRINCIPAIS:",
        "   • YoY_Receita_Growth_Pct: Crescimento vs ano anterior",
        "   • Receita_Periodo_Atual/Anterior: Valores comparativos",
        "   • Periodo_Analisado: Descrição do filtro aplicado",
        "   • Confiabilidade_Comparacao: Qualidade da comparação",
        "   • Dias_No_Periodo: Duração do período analisado"
    ]
    
    for melhoria in melhorias:
        print(melhoria)
    
    print("\n🎯 CASOS DE USO RECOMENDADOS:")
    casos = [
        "• Relatórios mensais: last_x_days=30",
        "• Análise trimestral: last_x_days=90", 
        "• Performance anual: current_year=2024",
        "• Comparativo semestral: period_start_date + period_end_date",
        "• Dashboard YTD: sem parâmetros (padrão)"
    ]
    
    for caso in casos:
        print(caso)
    
    print("\n✅ Demonstração concluída! Arquivos gerados em 'exemplos_output/'")

if __name__ == "__main__":
    main() 