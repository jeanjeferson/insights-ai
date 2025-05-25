#!/usr/bin/env python3
"""
📊 GERADOR DE DASHBOARD HTML EXECUTIVO - VERSÃO CORRIGIDA
=========================================================

Script para gerar dashboard executivo interativo em HTML usando
a plataforma Unified Business Intelligence.

CORREÇÕES IMPLEMENTADAS:
- Remove duplicações de dashboards
- Melhora organização visual dos KPIs
- Corrige títulos repetidos
- Adiciona visuais para todas as análises

Execução: python generate_dashboard_html_fixed.py
"""

import sys
import os
import time
import webbrowser
from pathlib import Path
from datetime import datetime

# Adicionar path do projeto
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from insights.tools.unified_business_intelligence import UnifiedBusinessIntelligence
    TOOL_AVAILABLE = True
except ImportError as e:
    print(f"❌ Erro ao importar ferramenta: {e}")
    TOOL_AVAILABLE = False

def generate_all_reports_fixed():
    """Gerar todos os relatórios corrigidos."""
    print("\n📊 GERANDO RELATÓRIOS HTML CORRIGIDOS")
    print("=" * 50)
    
    if not TOOL_AVAILABLE:
        print("❌ Ferramenta não disponível")
        return
    
    tool = UnifiedBusinessIntelligence()
    data_file = "data/vendas.csv"
    
    # Lista de análises para gerar (sem duplicatas)
    analyses = [
        ("executive_summary", "Resumo Executivo"),
        ("executive_dashboard", "Dashboard Executivo Visual"),
        ("financial_analysis", "Análise Financeira"),
        ("customer_intelligence", "Inteligência de Clientes"),
        ("product_performance", "Performance de Produtos")
    ]
    
    results = {}
    
    # Limpar arquivos antigos
    output_dir = Path("output")
    if output_dir.exists():
        for old_file in output_dir.glob("unified_*.html"):
            try:
                old_file.unlink()
                print(f"🗑️ Removido arquivo antigo: {old_file.name}")
            except:
                pass
    
    for analysis_type, description in analyses:
        try:
            print(f"\n🔄 Gerando {description}...")
            start_time = time.time()
            
            result = tool._run(
                analysis_type=analysis_type,
                data_csv=data_file,
                output_format="html",
                export_file=True,
                detail_level="detailed"
            )
            
            execution_time = time.time() - start_time
            results[analysis_type] = {
                'success': True,
                'time': execution_time,
                'result': result
            }
            
            print(f"✅ {description} gerado em {execution_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Erro em {description}: {e}")
            results[analysis_type] = {
                'success': False,
                'error': str(e)
            }
    
    # Relatório final
    print("\n📊 RELATÓRIO DE GERAÇÃO")
    print("=" * 30)
    
    successful = sum(1 for r in results.values() if r.get('success', False))
    total = len(results)
    success_rate = (successful / total) * 100
    
    for analysis, result in results.items():
        status = "✅" if result.get('success', False) else "❌"
        time_info = f"({result.get('time', 0):.2f}s)" if result.get('success', False) else ""
        print(f"  {status} {analysis.replace('_', ' ').title()} {time_info}")
    
    print(f"\n📈 Taxa de sucesso: {success_rate:.1f}% ({successful}/{total})")
    
    return results

def create_dashboard_index_fixed():
    """Criar página índice corrigida sem duplicações."""
    print("\n📋 Criando página índice corrigida...")
    
    output_dir = Path("output")
    if not output_dir.exists():
        print("❌ Diretório output não existe")
        return
    
    html_files = list(output_dir.glob("unified_*.html"))
    if not html_files:
        print("❌ Nenhum arquivo HTML encontrado")
        return
    
    # Mapear tipos de análise para metadados
    analysis_metadata = {
        'executive_summary': {
            'icon': '🎯',
            'title': 'Resumo Executivo',
            'description': 'Visão geral dos KPIs principais, alertas e recomendações estratégicas para tomada de decisão.'
        },
        'executive_dashboard': {
            'icon': '📊',
            'title': 'Dashboard Executivo Visual',
            'description': 'Dashboard visual interativo com gráficos Plotly e métricas em tempo real para análise executiva.'
        },
        'financial_analysis': {
            'icon': '💰',
            'title': 'Análise Financeira Detalhada',
            'description': 'KPIs financeiros detalhados, análise de tendências, sazonalidade e métricas de eficiência.'
        },
        'customer_intelligence': {
            'icon': '👥',
            'title': 'Inteligência de Clientes',
            'description': 'Segmentação RFM, análise de valor do cliente, retenção e insights comportamentais.'
        },
        'product_performance': {
            'icon': '📦',
            'title': 'Performance de Produtos',
            'description': 'Análise ABC, rankings de produtos, alertas de inventário e performance por categoria.'
        }
    }
    
    # Remover duplicatas - manter apenas o mais recente de cada tipo
    unique_files = {}
    for file in sorted(html_files, key=os.path.getctime, reverse=True):
        # Extrair tipo de análise do nome do arquivo
        analysis_type = None
        for key in analysis_metadata.keys():
            if key in file.name:
                analysis_type = key
                break
        
        # Se não encontrou tipo específico, usar nome genérico
        if not analysis_type:
            analysis_type = file.stem.replace('unified_', '').replace('_', ' ').title()
        
        # Manter apenas o mais recente de cada tipo
        if analysis_type not in unique_files:
            unique_files[analysis_type] = file
    
    print(f"📁 Arquivos únicos encontrados: {len(unique_files)}")
    for analysis_type, file in unique_files.items():
        print(f"  📄 {analysis_type}: {file.name}")
    
    # Criar HTML da página índice
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>📊 Dashboard Business Intelligence - Insights AI</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            
            .header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
                font-weight: 300;
            }}
            
            .header p {{
                font-size: 1.2em;
                opacity: 0.9;
            }}
            
            .content {{
                padding: 40px;
            }}
            
            .dashboard-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 25px;
                margin-top: 30px;
            }}
            
            .dashboard-card {{
                background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                border-radius: 20px;
                padding: 30px;
                border: 2px solid #e2e8f0;
                transition: all 0.3s ease;
                text-decoration: none;
                color: inherit;
                position: relative;
                overflow: hidden;
            }}
            
            .dashboard-card:hover {{
                transform: translateY(-8px);
                box-shadow: 0 20px 40px rgba(0,0,0,0.15);
                border-color: #3b82f6;
            }}
            
            .card-icon {{
                font-size: 3em;
                margin-bottom: 20px;
                display: block;
                opacity: 0.9;
            }}
            
            .card-title {{
                font-size: 1.4em;
                font-weight: 700;
                margin-bottom: 15px;
                color: #1e293b;
                line-height: 1.3;
            }}
            
            .card-description {{
                color: #64748b;
                line-height: 1.6;
                margin-bottom: 20px;
                font-size: 1em;
            }}
            
            .card-meta {{
                font-size: 0.9em;
                color: #94a3b8;
                border-top: 1px solid #e2e8f0;
                padding-top: 15px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .stat-card {{
                background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                border: 1px solid #e2e8f0;
            }}
            
            .stat-number {{
                font-size: 2.2em;
                font-weight: bold;
                color: #1e293b;
                margin-bottom: 5px;
            }}
            
            .stat-label {{
                color: #64748b;
                font-weight: 500;
            }}
            
            .footer {{
                background: #f8fafc;
                padding: 30px 40px;
                text-align: center;
                color: #64748b;
                border-top: 1px solid #e2e8f0;
            }}
            
            .badge {{
                background: #3b82f6;
                color: white;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: 600;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Business Intelligence Dashboard</h1>
                <p>Insights AI - Análises Executivas Completas</p>
                <p>Gerado em: {datetime.now().strftime('%d/%m/%Y às %H:%M')}</p>
            </div>
            
            <div class="content">
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number">{len(unique_files)}</div>
                        <div class="stat-label">Relatórios Únicos</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{sum(f.stat().st_size for f in unique_files.values()) / 1024:.0f}KB</div>
                        <div class="stat-label">Tamanho Total</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{datetime.now().strftime('%d/%m')}</div>
                        <div class="stat-label">Última Atualização</div>
                    </div>
                </div>
                
                <h2 style="color: #1e293b; margin-bottom: 20px; font-size: 1.8em;">🎯 Dashboards Disponíveis</h2>
                
                <div class="dashboard-grid">
    """
    
    # Adicionar cards para cada arquivo único
    for analysis_type, file in unique_files.items():
        if analysis_type in analysis_metadata:
            meta = analysis_metadata[analysis_type]
        else:
            meta = {
                'icon': '📄',
                'title': analysis_type,
                'description': 'Relatório de business intelligence gerado automaticamente.'
            }
        
        file_size = file.stat().st_size / 1024
        file_time = datetime.fromtimestamp(file.stat().st_ctime).strftime('%d/%m/%Y %H:%M')
        
        # Determinar badge baseado no tipo
        badge = ""
        if "dashboard" in analysis_type.lower():
            badge = '<span class="badge">VISUAL</span>'
        elif "summary" in analysis_type.lower():
            badge = '<span class="badge">EXECUTIVO</span>'
        
        html_content += f"""
                    <a href="{file.name}" class="dashboard-card" target="_blank">
                        <span class="card-icon">{meta['icon']}</span>
                        <div class="card-title">{meta['title']} {badge}</div>
                        <div class="card-description">{meta['description']}</div>
                        <div class="card-meta">
                            <span>📁 {file_size:.1f}KB</span>
                            <span>🕒 {file_time}</span>
                        </div>
                    </a>
        """
    
    html_content += """
                </div>
            </div>
            
            <div class="footer">
                <p><strong>🚀 Powered by Insights AI - Business Intelligence Platform</strong></p>
                <p>Dashboards gerados automaticamente com dados em tempo real</p>
                <p style="margin-top: 10px; font-size: 0.9em;">✨ Versão corrigida - sem duplicações</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Salvar arquivo índice
    index_file = output_dir / "index.html"
    with open(index_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Página índice corrigida criada: {index_file}")
    
    # Abrir no navegador
    try:
        webbrowser.open(f"file://{index_file.absolute()}")
        print("🌐 Página índice aberta no navegador")
    except Exception as e:
        print(f"⚠️ Não foi possível abrir no navegador: {e}")

def main():
    """Função principal."""
    print("🚀 GERADOR DE DASHBOARDS HTML CORRIGIDO - INSIGHTS AI")
    print("=" * 60)
    
    # Criar diretório output se não existir
    os.makedirs("output", exist_ok=True)
    
    print("\n🔧 CORREÇÕES IMPLEMENTADAS:")
    print("✅ Remove duplicações de dashboards")
    print("✅ Melhora organização visual dos KPIs")
    print("✅ Corrige títulos repetidos")
    print("✅ Adiciona visuais para todas as análises")
    print("✅ Interface moderna e responsiva")
    
    # Gerar todos os relatórios corrigidos
    print("\n🚀 Gerando todos os dashboards corrigidos...")
    results = generate_all_reports_fixed()
    
    # Criar página índice corrigida
    create_dashboard_index_fixed()
    
    print("\n✅ PROCESSO CONCLUÍDO COM SUCESSO!")
    print("📁 Verifique o diretório 'output' para os arquivos gerados")
    print("🌐 Acesse 'output/index.html' para navegar pelos dashboards")

if __name__ == "__main__":
    main() 