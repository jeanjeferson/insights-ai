from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any, List
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import json
import base64
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

class AdvancedVisualizationInput(BaseModel):
    """Schema de entrada para visualizações avançadas."""
    chart_type: str = Field(..., description="Tipo: 'executive_dashboard', 'sales_trends', 'product_analysis', 'seasonal_heatmap', 'category_performance', 'inventory_matrix', 'customer_segments', 'financial_overview'")
    data_csv: str = Field(default="data/vendas.csv", description="Caminho para o arquivo CSV")
    title: str = Field(default="Análise de Vendas - Joalheria", description="Título do gráfico")
    export_format: str = Field(default="html", description="Formato: 'html', 'json', 'png'")
    save_output: bool = Field(default=True, description="Salvar arquivo de saída")

class AdvancedVisualizationTool(BaseTool):
    name: str = "Advanced Visualization Tool"
    description: str = """
    Cria visualizações interativas sofisticadas especializadas para joalherias:
    - executive_dashboard: Dashboard executivo completo com múltiplos painéis
    - sales_trends: Análise detalhada de tendências de vendas
    - product_analysis: Análise de performance de produtos e categorias
    - seasonal_heatmap: Mapa de calor de sazonalidade por categoria
    - category_performance: Performance comparativa detalhada por categoria
    - inventory_matrix: Matriz de inventário (ABC + giro + sazonalidade)
    - customer_segments: Análise de segmentação de clientes estimada
    - financial_overview: Visão geral financeira com KPIs principais
    
    Todas as visualizações são otimizadas para dados de joalherias com:
    - Design profissional e cores adequadas ao setor de luxo
    - Interatividade completa (hover, zoom, filtros)
    - Responsividade para diferentes dispositivos
    - Métricas especializadas do setor
    """
    args_schema: Type[BaseModel] = AdvancedVisualizationInput
    
    def _run(self, chart_type: str, data_csv: str = "data/vendas.csv", 
             title: str = "Análise de Vendas - Joalheria", export_format: str = "html",
             save_output: bool = True) -> str:
        try:
            # Carregar e validar dados
            df = pd.read_csv(data_csv, sep=';', encoding='utf-8')
            df = self._validate_and_prepare_data(df)
            
            if df is None or len(df) == 0:
                return "Erro: Não foi possível carregar os dados ou dataset vazio"
            
            # Dicionário de visualizações
            charts = {
                'executive_dashboard': self._create_executive_dashboard,
                'sales_trends': self._create_sales_trends,
                'product_analysis': self._create_product_analysis,
                'seasonal_heatmap': self._create_seasonal_heatmap,
                'category_performance': self._create_category_performance,
                'inventory_matrix': self._create_inventory_matrix,
                'customer_segments': self._create_customer_segments,
                'financial_overview': self._create_financial_overview
            }
            
            if chart_type not in charts:
                return f"Tipo '{chart_type}' não suportado. Opções: {list(charts.keys())}"
            
            # Criar visualização
            fig = charts[chart_type](df, title)
            
            # Aplicar tema profissional
            fig = self._apply_professional_theme(fig, chart_type)
            
            # Exportar conforme formato solicitado
            result = self._export_visualization(fig, chart_type, export_format, save_output)
            
            return result
            
        except Exception as e:
            return f"Erro na criação da visualização: {str(e)}"
    
    def _validate_and_prepare_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Validar e preparar dados para visualização."""
        try:
            # Verificar colunas essenciais
            required_cols = ['Data', 'Total_Liquido']
            if not all(col in df.columns for col in required_cols):
                return None
            
            # Converter tipos
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
            df['Total_Liquido'] = pd.to_numeric(df['Total_Liquido'], errors='coerce')
            df['Quantidade'] = pd.to_numeric(df['Quantidade'], errors='coerce') if 'Quantidade' in df.columns else 1
            
            # Remover registros inválidos
            df = df.dropna(subset=['Data', 'Total_Liquido'])
            df = df[df['Total_Liquido'] > 0]
            
            # Adicionar colunas derivadas para análises
            df['Ano'] = df['Data'].dt.year
            df['Mes'] = df['Data'].dt.month
            df['Trimestre'] = df['Data'].dt.quarter
            df['Dia_Semana'] = df['Data'].dt.dayofweek
            df['Semana_Ano'] = df['Data'].dt.isocalendar().week
            df['Mes_Nome'] = df['Data'].dt.strftime('%b')
            df['Ano_Mes'] = df['Data'].dt.to_period('M').astype(str)
            
            # Preencher colunas opcionais se não existirem
            optional_cols = ['Codigo_Produto', 'Descricao_Produto', 'Grupo_Produto', 'Metal', 'Colecao']
            for col in optional_cols:
                if col not in df.columns:
                    df[col] = 'N/A'
            
            return df
            
        except Exception as e:
            print(f"Erro na validação: {str(e)}")
            return None
    
    def _create_executive_dashboard(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Criar dashboard executivo completo com múltiplos painéis."""
        
        # Criar layout de subplots
        fig = make_subplots(
            rows=4, cols=3,
            subplot_titles=[
                '📈 Evolução Mensal de Vendas', '🏆 Top 10 Produtos', '💎 Vendas por Metal',
                '📊 Performance por Categoria', '📅 Padrão Sazonal', '💰 Ticket Médio Mensal',
                '⚖️ Concentração ABC', '📆 Performance por Dia da Semana', '🎯 KPIs Principais'
            ],
            specs=[
                [{"secondary_y": True}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}, {"secondary_y": True}],
                [{"type": "pie"}, {"type": "bar"}, {"type": "table"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Evolução Mensal com Tendência
        monthly_sales = df.groupby('Ano_Mes').agg({
            'Total_Liquido': 'sum',
            'Quantidade': 'sum'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=monthly_sales['Ano_Mes'],
                y=monthly_sales['Total_Liquido'],
                mode='lines+markers',
                name='Receita Mensal',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Linha de tendência
        x_numeric = np.arange(len(monthly_sales))
        z = np.polyfit(x_numeric, monthly_sales['Total_Liquido'], 1)
        p = np.poly1d(z)
        
        fig.add_trace(
            go.Scatter(
                x=monthly_sales['Ano_Mes'],
                y=p(x_numeric),
                mode='lines',
                name='Tendência',
                line=dict(color='red', width=2, dash='dash'),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # 2. Top 10 Produtos
        if 'Descricao_Produto' in df.columns and df['Descricao_Produto'].notna().any():
            top_products = df.groupby('Descricao_Produto')['Total_Liquido'].sum().nlargest(10)
            
            fig.add_trace(
                go.Bar(
                    y=[prod[:30] + '...' if len(str(prod)) > 30 else str(prod) for prod in top_products.index],
                    x=top_products.values,
                    orientation='h',
                    name='Top Produtos',
                    marker_color='lightblue',
                    text=[f'R$ {v:,.0f}' for v in top_products.values],
                    textposition='inside'
                ),
                row=1, col=2
            )
        
        # 3. Vendas por Metal
        if 'Metal' in df.columns and df['Metal'].notna().any():
            metal_sales = df.groupby('Metal')['Total_Liquido'].sum()
            colors = ['gold', 'silver', '#CD7F32', 'lightcoral', 'lightgreen'][:len(metal_sales)]
            
            fig.add_trace(
                go.Bar(
                    x=metal_sales.index,
                    y=metal_sales.values,
                    name='Vendas por Metal',
                    marker_color=colors,
                    text=[f'R$ {v:,.0f}' for v in metal_sales.values],
                    textposition='outside'
                ),
                row=1, col=3
            )
        
        # 4. Performance por Categoria
        if 'Grupo_Produto' in df.columns and df['Grupo_Produto'].notna().any():
            category_perf = df.groupby('Grupo_Produto')['Total_Liquido'].sum()
            
            fig.add_trace(
                go.Bar(
                    x=category_perf.index,
                    y=category_perf.values,
                    name='Receita por Categoria',
                    marker_color='lightcoral',
                    text=[f'R$ {v:,.0f}' for v in category_perf.values],
                    textposition='outside'
                ),
                row=2, col=1
            )
        
        # 5. Padrão Sazonal
        monthly_pattern = df.groupby('Mes')['Total_Liquido'].mean()
        months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        
        fig.add_trace(
            go.Scatter(
                x=[months[i-1] if i <= 12 else f'M{i}' for i in monthly_pattern.index],
                y=monthly_pattern.values,
                mode='lines+markers',
                name='Padrão Sazonal',
                line=dict(color='green', width=3),
                fill='tozeroy',
                fillcolor='rgba(0,128,0,0.1)'
            ),
            row=2, col=2
        )
        
        # 6. Ticket Médio Mensal
        monthly_ticket = df.groupby('Ano_Mes')['Total_Liquido'].mean()
        
        fig.add_trace(
            go.Scatter(
                x=monthly_ticket.index,
                y=monthly_ticket.values,
                mode='lines+markers',
                name='Ticket Médio',
                line=dict(color='purple', width=2),
                marker=dict(size=6)
            ),
            row=2, col=3
        )
        
        # 7. Concentração ABC (Pie Chart)
        if 'Codigo_Produto' in df.columns:
            product_sales = df.groupby('Codigo_Produto')['Total_Liquido'].sum().sort_values(ascending=False)
            total_products = len(product_sales)
            
            # Classificação ABC simplificada
            class_a = int(total_products * 0.2)  # 20%
            class_b = int(total_products * 0.3)  # 30% 
            class_c = total_products - class_a - class_b  # 50%
            
            abc_revenue = [
                product_sales.head(class_a).sum(),
                product_sales.iloc[class_a:class_a+class_b].sum(),
                product_sales.tail(class_c).sum()
            ]
            
            fig.add_trace(
                go.Pie(
                    labels=['Classe A (20%)', 'Classe B (30%)', 'Classe C (50%)'],
                    values=abc_revenue,
                    hole=0.4,
                    marker_colors=['#ff7f0e', '#2ca02c', '#d62728']
                ),
                row=3, col=1
            )
        
        # 8. Performance por Dia da Semana
        weekday_names = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        weekday_sales = df.groupby('Dia_Semana')['Total_Liquido'].sum()
        
        fig.add_trace(
            go.Bar(
                x=[weekday_names[i] if i < 7 else f'Dia {i}' for i in weekday_sales.index],
                y=weekday_sales.values,
                name='Vendas por Dia',
                marker_color='lightsteelblue',
                text=[f'R$ {v:,.0f}' for v in weekday_sales.values],
                textposition='outside'
            ),
            row=3, col=2
        )
        
        # 9. Tabela de KPIs Principais
        total_revenue = df['Total_Liquido'].sum()
        avg_ticket = df['Total_Liquido'].mean()
        total_transactions = len(df)
        total_products = df['Codigo_Produto'].nunique() if 'Codigo_Produto' in df.columns else len(df)
        
        kpi_data = [
            ['Receita Total', f'R$ {total_revenue:,.2f}'],
            ['Ticket Médio', f'R$ {avg_ticket:,.2f}'],
            ['Total Transações', f'{total_transactions:,}'],
            ['Produtos Ativos', f'{total_products:,}'],
            ['Período', f"{df['Data'].min().strftime('%m/%Y')} - {df['Data'].max().strftime('%m/%Y')}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>KPI</b>', '<b>Valor</b>'],
                    fill_color='lightblue',
                    align='left',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=list(zip(*kpi_data)),
                    fill_color='white',
                    align='left',
                    font=dict(size=11)
                )
            ),
            row=3, col=3
        )
        
        # Atualizar layout geral
        fig.update_layout(
            title=dict(
                text=f"<b>{title} - Dashboard Executivo</b>",
                x=0.5,
                font=dict(size=20)
            ),
            height=1400,
            showlegend=False
        )
        
        return fig
    
    def _create_sales_trends(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Criar análise detalhada de tendências de vendas."""
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Tendência Geral com Projeção', 'Crescimento Mês a Mês (%)',
                'Decomposição: Tendência vs Sazonalidade', 'Comparação Anual',
                'Análise de Volatilidade', 'Ciclos de Vendas'
            ],
            specs=[
                [{"secondary_y": True}, {}],
                [{}, {}],
                [{}, {}]
            ],
            vertical_spacing=0.12
        )
        
        # 1. Tendência Geral com Projeção
        monthly_data = df.groupby('Ano_Mes')['Total_Liquido'].sum().reset_index()
        
        # Dados históricos
        fig.add_trace(
            go.Scatter(
                x=monthly_data['Ano_Mes'],
                y=monthly_data['Total_Liquido'],
                mode='lines+markers',
                name='Vendas Reais',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        # Linha de tendência e projeção
        x_numeric = np.arange(len(monthly_data))
        z = np.polyfit(x_numeric, monthly_data['Total_Liquido'], 1)
        p = np.poly1d(z)
        
        # Tendência histórica
        fig.add_trace(
            go.Scatter(
                x=monthly_data['Ano_Mes'],
                y=p(x_numeric),
                mode='lines',
                name='Tendência',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Projeção (6 meses futuros)
        last_date = pd.to_datetime(monthly_data['Ano_Mes'].iloc[-1])
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='M')
        future_x = np.arange(len(monthly_data), len(monthly_data) + 6)
        future_projection = p(future_x)
        
        fig.add_trace(
            go.Scatter(
                x=[d.strftime('%Y-%m') for d in future_dates],
                y=future_projection,
                mode='lines+markers',
                name='Projeção',
                line=dict(color='orange', width=2, dash='dot'),
                marker=dict(symbol='diamond', size=8)
            ),
            row=1, col=1
        )
        
        # 2. Crescimento Mês a Mês
        monthly_growth = monthly_data['Total_Liquido'].pct_change() * 100
        colors = ['green' if x > 0 else 'red' for x in monthly_growth.dropna()]
        
        fig.add_trace(
            go.Bar(
                x=monthly_data['Ano_Mes'][1:],
                y=monthly_growth.dropna(),
                name='Crescimento %',
                marker_color=colors,
                text=[f'{x:.1f}%' for x in monthly_growth.dropna()],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # Linha de referência em 0%
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=2)
        
        # 3. Decomposição Tendência vs Sazonalidade
        monthly_avg_by_month = df.groupby('Mes')['Total_Liquido'].mean()
        overall_avg = df['Total_Liquido'].mean()
        seasonal_index = monthly_avg_by_month / overall_avg
        
        months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        
        fig.add_trace(
            go.Scatter(
                x=[months[i-1] if i <= 12 else f'M{i}' for i in seasonal_index.index],
                y=seasonal_index.values,
                mode='lines+markers',
                name='Índice Sazonal',
                line=dict(color='green', width=3),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=1, line_dash="dash", line_color="gray", opacity=0.7, row=2, col=1)
        
        # 4. Comparação Anual (se há múltiplos anos)
        years = df['Ano'].unique()
        if len(years) > 1:
            for year in sorted(years):
                year_data = df[df['Ano'] == year].groupby('Mes')['Total_Liquido'].sum()
                fig.add_trace(
                    go.Scatter(
                        x=[months[i-1] if i <= 12 else f'M{i}' for i in year_data.index],
                        y=year_data.values,
                        mode='lines+markers',
                        name=f'Ano {year}',
                        line=dict(width=2)
                    ),
                    row=2, col=2
                )
        else:
            # Se só há um ano, mostrar comparação trimestral
            quarterly = df.groupby('Trimestre')['Total_Liquido'].sum()
            fig.add_trace(
                go.Bar(
                    x=[f'Q{q}' for q in quarterly.index],
                    y=quarterly.values,
                    name='Vendas Trimestrais',
                    marker_color='lightcoral'
                ),
                row=2, col=2
            )
        
        # 5. Análise de Volatilidade
        monthly_data['Rolling_Std'] = monthly_data['Total_Liquido'].rolling(window=3).std()
        monthly_data['CV'] = monthly_data['Rolling_Std'] / monthly_data['Total_Liquido'].rolling(window=3).mean()
        
        fig.add_trace(
            go.Scatter(
                x=monthly_data['Ano_Mes'],
                y=monthly_data['CV'] * 100,
                mode='lines+markers',
                name='Coeficiente de Variação (%)',
                line=dict(color='purple', width=2)
            ),
            row=3, col=1
        )
        
        # 6. Ciclos de Vendas (Weekly pattern)
        weekly_pattern = df.groupby('Dia_Semana')['Total_Liquido'].mean()
        weekday_names = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
        
        fig.add_trace(
            go.Scatter(
                x=[weekday_names[i] if i < 7 else f'D{i}' for i in weekly_pattern.index],
                y=weekly_pattern.values,
                mode='lines+markers',
                name='Padrão Semanal',
                line=dict(color='orange', width=3),
                fill='tozeroy'
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            title=f"<b>{title} - Análise de Tendências</b>",
            height=1200,
            showlegend=True
        )
        
        return fig
    
    def _create_product_analysis(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Criar análise detalhada de produtos."""
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Top & Bottom Performers', 'Performance por Metal',
                'Análise de Concentração', 'Ciclo de Vida dos Produtos',
                'Performance por Categoria', 'Cross-Selling Patterns'
            ],
            specs=[
                [{}, {}, {"type": "pie"}],
                [{}, {}, {}]
            ]
        )
        
        # 1. Top & Bottom Performers
        if 'Descricao_Produto' in df.columns and df['Descricao_Produto'].notna().any():
            product_sales = df.groupby('Descricao_Produto')['Total_Liquido'].sum()
            top_5 = product_sales.nlargest(5)
            bottom_5 = product_sales.nsmallest(5)
            
            # Top performers
            fig.add_trace(
                go.Bar(
                    y=[p[:25] + '...' if len(str(p)) > 25 else str(p) for p in top_5.index],
                    x=top_5.values,
                    orientation='h',
                    name='Top 5',
                    marker_color='green',
                    text=[f'R$ {v:,.0f}' for v in top_5.values],
                    textposition='inside'
                ),
                row=1, col=1
            )
            
            # Bottom performers  
            fig.add_trace(
                go.Bar(
                    y=[p[:25] + '...' if len(str(p)) > 25 else str(p) for p in bottom_5.index],
                    x=bottom_5.values,
                    orientation='h',
                    name='Bottom 5',
                    marker_color='red',
                    text=[f'R$ {v:,.0f}' for v in bottom_5.values],
                    textposition='inside'
                ),
                row=1, col=1
            )
        
        # 2. Performance por Metal
        if 'Metal' in df.columns and df['Metal'].notna().any():
            metal_perf = df.groupby('Metal').agg({
                'Total_Liquido': ['sum', 'mean', 'count']
            })
            metal_perf.columns = ['Total', 'Média', 'Count']
            
            # Gráfico de barras agrupadas
            fig.add_trace(
                go.Bar(
                    x=metal_perf.index,
                    y=metal_perf['Total'],
                    name='Receita Total',
                    marker_color='lightblue',
                    offsetgroup=1
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=metal_perf.index,
                    y=metal_perf['Média'],
                    name='Ticket Médio',
                    marker_color='lightcoral',
                    offsetgroup=2
                ),
                row=1, col=2
            )
        
        # 3. Análise de Concentração (Pareto)
        if 'Codigo_Produto' in df.columns:
            product_sales = df.groupby('Codigo_Produto')['Total_Liquido'].sum().sort_values(ascending=False)
            cumsum_pct = (product_sales.cumsum() / product_sales.sum() * 100)
            
            # Classificação ABC
            class_a = len(cumsum_pct[cumsum_pct <= 80])
            class_b = len(cumsum_pct[(cumsum_pct > 80) & (cumsum_pct <= 95)])
            class_c = len(cumsum_pct[cumsum_pct > 95])
            
            fig.add_trace(
                go.Pie(
                    labels=['Classe A (80%)', 'Classe B (15%)', 'Classe C (5%)'],
                    values=[class_a, class_b, class_c],
                    hole=0.4,
                    marker_colors=['#ff7f0e', '#2ca02c', '#d62728']
                ),
                row=1, col=3
            )
        
        # 4. Ciclo de Vida dos Produtos
        if 'Codigo_Produto' in df.columns:
            current_date = df['Data'].max()
            product_age = df.groupby('Codigo_Produto')['Data'].min()
            product_age_days = (current_date - product_age).dt.days
            
            # Categorizar por idade
            age_categories = []
            for days in product_age_days:
                if days <= 90:
                    age_categories.append('Novos (0-3m)')
                elif days <= 365:
                    age_categories.append('Crescimento (3m-1a)')
                elif days <= 730:
                    age_categories.append('Maturidade (1-2a)')
                else:
                    age_categories.append('Declínio (>2a)')
            
            age_dist = pd.Series(age_categories).value_counts()
            
            fig.add_trace(
                go.Bar(
                    x=age_dist.index,
                    y=age_dist.values,
                    name='Produtos por Idade',
                    marker_color='lightgreen'
                ),
                row=2, col=1
            )
        
        # 5. Performance por Categoria
        if 'Grupo_Produto' in df.columns and df['Grupo_Produto'].notna().any():
            category_perf = df.groupby('Grupo_Produto')['Total_Liquido'].sum()
            
            fig.add_trace(
                go.Bar(
                    x=category_perf.index,
                    y=category_perf.values,
                    name='Receita por Categoria',
                    marker_color='lightcoral'
                ),
                row=2, col=2
            )
        
        # 6. Cross-Selling Patterns (Heatmap de categorias por dia)
        if 'Grupo_Produto' in df.columns and df['Grupo_Produto'].notna().any():
            daily_categories = df.groupby([df['Data'].dt.date, 'Grupo_Produto']).size().unstack(fill_value=0)
            
            # Calcular correlação entre categorias
            if daily_categories.shape[1] > 1:
                corr_matrix = daily_categories.corr()
                
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0,
                        text=corr_matrix.round(2).values,
                        texttemplate="%{text}",
                        showscale=True
                    ),
                    row=2, col=3
                )
        
        fig.update_layout(
            title=f"<b>{title} - Análise de Produtos</b>",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def _create_seasonal_heatmap(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Criar mapa de calor de sazonalidade."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Heatmap: Mês vs Dia da Semana', 'Intensidade Sazonal por Categoria',
                'Padrão Semanal por Categoria', 'Calendário de Vendas'
            ],
            specs=[
                [{"type": "heatmap"}, {}],
                [{}, {"type": "heatmap"}]
            ]
        )
        
        # 1. Heatmap Principal: Mês vs Dia da Semana
        df_pivot = df.groupby([df['Data'].dt.month, df['Data'].dt.dayofweek])['Total_Liquido'].sum().unstack(fill_value=0)
        
        months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        weekdays = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
        
        fig.add_trace(
            go.Heatmap(
                z=df_pivot.values,
                x=[weekdays[i] if i < 7 else f'D{i}' for i in df_pivot.columns],
                y=[months[i-1] if i <= 12 else f'M{i}' for i in df_pivot.index],
                colorscale='Viridis',
                showscale=True,
                text=[[f'R$ {val:,.0f}' for val in row] for row in df_pivot.values],
                texttemplate="%{text}",
                hoverongaps=False
            ),
            row=1, col=1
        )
        
        # 2. Intensidade Sazonal por Categoria
        if 'Grupo_Produto' in df.columns and df['Grupo_Produto'].notna().any():
            category_seasonal = df.groupby(['Grupo_Produto', df['Data'].dt.month])['Total_Liquido'].sum().unstack(fill_value=0)
            
            # Normalizar por categoria para mostrar sazonalidade relativa
            category_seasonal_norm = category_seasonal.div(category_seasonal.sum(axis=1), axis=0)
            
            # Plotar apenas top 5 categorias
            top_categories = df.groupby('Grupo_Produto')['Total_Liquido'].sum().nlargest(5).index
            category_seasonal_top = category_seasonal_norm.loc[top_categories]
            
            for i, category in enumerate(category_seasonal_top.index):
                fig.add_trace(
                    go.Scatter(
                        x=[months[j-1] if j <= 12 else f'M{j}' for j in category_seasonal_top.columns],
                        y=category_seasonal_top.loc[category],
                        mode='lines+markers',
                        name=str(category),
                        line=dict(width=2)
                    ),
                    row=1, col=2
                )
        
        # 3. Padrão Semanal por Categoria
        if 'Grupo_Produto' in df.columns and df['Grupo_Produto'].notna().any():
            weekly_category = df.groupby(['Grupo_Produto', df['Data'].dt.dayofweek])['Total_Liquido'].mean().unstack(fill_value=0)
            
            for i, category in enumerate(weekly_category.index[:5]):  # Top 5
                fig.add_trace(
                    go.Bar(
                        x=[weekdays[j] if j < 7 else f'D{j}' for j in weekly_category.columns],
                        y=weekly_category.loc[category],
                        name=str(category),
                        opacity=0.7
                    ),
                    row=2, col=1
                )
        
        # 4. Calendário de Vendas (Heatmap de dias do ano)
        df['Day_of_Year'] = df['Data'].dt.dayofyear
        df['Week_of_Year'] = df['Data'].dt.isocalendar().week
        
        daily_sales = df.groupby([df['Data'].dt.date])['Total_Liquido'].sum().reset_index()
        daily_sales['Day_of_Year'] = daily_sales['Data'].dt.dayofyear
        daily_sales['Week_of_Year'] = daily_sales['Data'].dt.isocalendar().week
        daily_sales['Weekday'] = daily_sales['Data'].dt.dayofweek
        
        # Criar matriz de calendário
        calendar_matrix = daily_sales.pivot_table(
            index='Week_of_Year', 
            columns='Weekday', 
            values='Total_Liquido', 
            fill_value=0
        )
        
        fig.add_trace(
            go.Heatmap(
                z=calendar_matrix.values,
                x=[weekdays[i] if i < 7 else f'D{i}' for i in calendar_matrix.columns],
                y=calendar_matrix.index,
                colorscale='Blues',
                showscale=True,
                name='Calendário'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"<b>{title} - Análise de Sazonalidade</b>",
            height=1000,
            showlegend=True
        )
        
        return fig
    
    def _create_category_performance(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Criar análise de performance por categoria."""
        
        if 'Grupo_Produto' not in df.columns or not df['Grupo_Produto'].notna().any():
            # Criar gráfico de erro se não há dados de categoria
            fig = go.Figure()
            fig.add_annotation(
                text="Dados de categoria de produto não disponíveis para análise",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Receita por Categoria', 'Ticket Médio por Categoria',
                'Volume vs Valor (Scatter)', 'Margem Estimada por Categoria',
                'Market Share Evolution', 'Performance Relativa'
            ],
            specs=[
                [{}, {}, {"type": "scatter"}],
                [{}, {}, {}]
            ]
        )
        
        # Métricas por categoria
        category_metrics = df.groupby('Grupo_Produto').agg({
            'Total_Liquido': ['sum', 'mean', 'count'],
            'Quantidade': 'sum'
        })
        category_metrics.columns = ['Receita_Total', 'Ticket_Medio', 'Num_Transacoes', 'Volume_Total']
        
        # 1. Receita por Categoria
        fig.add_trace(
            go.Bar(
                x=category_metrics.index,
                y=category_metrics['Receita_Total'],
                name='Receita Total',
                marker_color='lightblue',
                text=[f'R$ {v:,.0f}' for v in category_metrics['Receita_Total']],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. Ticket Médio por Categoria
        fig.add_trace(
            go.Bar(
                x=category_metrics.index,
                y=category_metrics['Ticket_Medio'],
                name='Ticket Médio',
                marker_color='lightcoral',
                text=[f'R$ {v:,.0f}' for v in category_metrics['Ticket_Medio']],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        # 3. Scatter: Volume vs Valor
        fig.add_trace(
            go.Scatter(
                x=category_metrics['Num_Transacoes'],
                y=category_metrics['Ticket_Medio'],
                mode='markers+text',
                text=category_metrics.index,
                textposition='top center',
                marker=dict(
                    size=np.sqrt(category_metrics['Receita_Total']) / 100,
                    color=category_metrics['Receita_Total'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Receita Total"),
                    line=dict(width=1, color='black')
                ),
                name='Volume vs Valor'
            ),
            row=1, col=3
        )
        
        # 4. Margem Estimada por Categoria (baseado em benchmarks)
        margin_estimates = {
            'Anéis': 0.62, 'Brincos': 0.58, 'Colares': 0.55, 'Pulseiras': 0.60,
            'Alianças': 0.45, 'Pingentes': 0.65, 'Correntes': 0.52, 'Outros': 0.50
        }
        
        estimated_margins = []
        for category in category_metrics.index:
            margin = margin_estimates.get(str(category), 0.55)
            estimated_margins.append(category_metrics.loc[category, 'Receita_Total'] * margin)
        
        fig.add_trace(
            go.Bar(
                x=category_metrics.index,
                y=estimated_margins,
                name='Margem Estimada',
                marker_color='lightgreen',
                text=[f'R$ {v:,.0f}' for v in estimated_margins],
                textposition='outside'
            ),
            row=2, col=1
        )
        
        # 5. Market Share Evolution
        total_revenue = df['Total_Liquido'].sum()
        market_share = (category_metrics['Receita_Total'] / total_revenue * 100).sort_values(ascending=True)
        
        fig.add_trace(
            go.Bar(
                y=market_share.index,
                x=market_share.values,
                orientation='h',
                name='Market Share (%)',
                marker_color='gold',
                text=[f'{v:.1f}%' for v in market_share.values],
                textposition='inside'
            ),
            row=2, col=2
        )
        
        # 6. Performance Relativa
        performance_score = (
            (category_metrics['Receita_Total'] / category_metrics['Receita_Total'].max()) * 0.4 +
            (category_metrics['Ticket_Medio'] / category_metrics['Ticket_Medio'].max()) * 0.3 +
            (category_metrics['Num_Transacoes'] / category_metrics['Num_Transacoes'].max()) * 0.3
        ) * 100
        
        colors = ['green' if x > 70 else 'orange' if x > 50 else 'red' for x in performance_score]
        
        fig.add_trace(
            go.Bar(
                x=performance_score.index,
                y=performance_score.values,
                name='Score Performance',
                marker_color=colors,
                text=[f'{v:.0f}' for v in performance_score.values],
                textposition='outside'
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title=f"<b>{title} - Performance por Categoria</b>",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def _create_inventory_matrix(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Criar matriz de análise de inventário."""
        
        if 'Codigo_Produto' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="Dados de código de produto necessários para análise de inventário",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Matriz ABC', 'Giro vs Volume', 'Produtos Slow-Moving',
                'Análise de Pareto', 'Idade dos Produtos', 'Recomendações'
            ],
            specs=[
                [{"type": "pie"}, {"type": "scatter"}, {}],
                [{}, {}, {"type": "table"}]
            ]
        )
        
        # Calcular métricas por produto
        product_metrics = df.groupby('Codigo_Produto').agg({
            'Total_Liquido': 'sum',
            'Quantidade': 'sum',
            'Data': ['min', 'max', 'count']
        })
        
        product_metrics.columns = ['Revenue', 'Quantity', 'First_Sale', 'Last_Sale', 'Frequency']
        
        # 1. Análise ABC
        product_metrics_sorted = product_metrics.sort_values('Revenue', ascending=False)
        total_revenue = product_metrics_sorted['Revenue'].sum()
        cumsum_pct = (product_metrics_sorted['Revenue'].cumsum() / total_revenue * 100)
        
        class_a = len(cumsum_pct[cumsum_pct <= 80])
        class_b = len(cumsum_pct[(cumsum_pct > 80) & (cumsum_pct <= 95)])
        class_c = len(cumsum_pct[cumsum_pct > 95])
        
        fig.add_trace(
            go.Pie(
                labels=['Classe A (80% Rev)', 'Classe B (15% Rev)', 'Classe C (5% Rev)'],
                values=[class_a, class_b, class_c],
                hole=0.4,
                marker_colors=['#ff7f0e', '#2ca02c', '#d62728'],
                textinfo='label+percent'
            ),
            row=1, col=1
        )
        
        # 2. Giro vs Volume
        current_date = df['Data'].max()
        product_metrics['Days_Active'] = (current_date - pd.to_datetime(product_metrics['Last_Sale'])).dt.days
        product_metrics['Turnover_Rate'] = product_metrics['Frequency'] / (
            (pd.to_datetime(product_metrics['Last_Sale']) - pd.to_datetime(product_metrics['First_Sale'])).dt.days + 1
        ) * 30  # Monthly turnover
        
        # Classificar produtos por performance
        turnover_median = product_metrics['Turnover_Rate'].median()
        revenue_median = product_metrics['Revenue'].median()
        
        colors = []
        for _, row in product_metrics.iterrows():
            if row['Turnover_Rate'] > turnover_median and row['Revenue'] > revenue_median:
                colors.append('gold')  # Stars
            elif row['Revenue'] > revenue_median:
                colors.append('green')  # Cash Cows
            elif row['Turnover_Rate'] > turnover_median:
                colors.append('blue')  # Fast Movers
            else:
                colors.append('red')  # Slow Movers
        
        fig.add_trace(
            go.Scatter(
                x=product_metrics['Frequency'],
                y=product_metrics['Revenue'],
                mode='markers',
                marker=dict(
                    color=colors,
                    size=np.sqrt(product_metrics['Quantity']) / 5,
                    opacity=0.7,
                    line=dict(width=1, color='black')
                ),
                text=[f'Produto: {idx}<br>Revenue: R$ {row["Revenue"]:,.0f}' 
                      for idx, row in product_metrics.iterrows()],
                hovertemplate='%{text}<extra></extra>',
                name='Produtos'
            ),
            row=1, col=2
        )
        
        # 3. Produtos Slow-Moving
        slow_moving = product_metrics[product_metrics['Days_Active'] > 60].sort_values('Days_Active', ascending=False)
        
        if len(slow_moving) > 0:
            top_slow = slow_moving.head(10)
            fig.add_trace(
                go.Bar(
                    y=top_slow.index,
                    x=top_slow['Days_Active'],
                    orientation='h',
                    name='Dias sem Venda',
                    marker_color='red'
                ),
                row=1, col=3
            )
        
        # 4. Análise de Pareto
        pareto_x = list(range(1, len(product_metrics_sorted) + 1))
        pareto_y = cumsum_pct.values
        
        fig.add_trace(
            go.Scatter(
                x=pareto_x,
                y=pareto_y,
                mode='lines',
                name='Curva de Pareto',
                line=dict(color='blue', width=3)
            ),
            row=2, col=1
        )
        
        # Linhas de referência 80/20
        fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
        fig.add_vline(x=class_a, line_dash="dash", line_color="red", opacity=0.7, row=2, col=1)
        
        # 5. Idade dos Produtos
        product_metrics['Age_Days'] = (current_date - pd.to_datetime(product_metrics['First_Sale'])).dt.days
        
        age_categories = []
        for age in product_metrics['Age_Days']:
            if age <= 90:
                age_categories.append('Novos')
            elif age <= 365:
                age_categories.append('Jovens')
            elif age <= 730:
                age_categories.append('Maduros')
            else:
                age_categories.append('Veteranos')
        
        age_dist = pd.Series(age_categories).value_counts()
        
        fig.add_trace(
            go.Bar(
                x=age_dist.index,
                y=age_dist.values,
                name='Distribuição por Idade',
                marker_color='lightgreen'
            ),
            row=2, col=2
        )
        
        # 6. Tabela de Recomendações
        recommendations = [
            ['Stars', f'{sum([1 for c in colors if c == "gold"])} produtos', 'Manter investimento'],
            ['Slow Movers', f'{sum([1 for c in colors if c == "red"])} produtos', 'Liquidação'],
            ['Dead Stock', f'{len(product_metrics[product_metrics["Days_Active"] > 90])} produtos', 'Descontinuar'],
            ['Fast Movers', f'{sum([1 for c in colors if c == "blue"])} produtos', 'Aumentar estoque']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Categoria</b>', '<b>Quantidade</b>', '<b>Ação</b>'],
                    fill_color='lightblue',
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*recommendations)),
                    fill_color='white',
                    align='left'
                )
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title=f"<b>{title} - Matriz de Inventário</b>",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def _create_customer_segments(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Criar análise de segmentação de clientes estimada."""
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Segmentação por Valor', 'Distribuição de Tickets', 'Padrão Temporal',
                'RFM Simplificado', 'Análise de Oportunidades', 'Métricas de Cliente'
            ],
            specs=[
                [{"type": "pie"}, {}, {}],
                [{"type": "scatter"}, {"type": "table"}, {"type": "table"}]
            ]
        )
        
        # 1. Segmentação por Valor de Compra
        value_segments = {
            'Premium (>R$5K)': len(df[df['Total_Liquido'] > 5000]),
            'Alto Valor (R$2K-5K)': len(df[(df['Total_Liquido'] >= 2000) & (df['Total_Liquido'] <= 5000)]),
            'Médio (R$1K-2K)': len(df[(df['Total_Liquido'] >= 1000) & (df['Total_Liquido'] < 2000)]),
            'Entry (< R$1K)': len(df[df['Total_Liquido'] < 1000])
        }
        
        fig.add_trace(
            go.Pie(
                labels=list(value_segments.keys()),
                values=list(value_segments.values()),
                hole=0.4,
                marker_colors=['gold', 'orange', 'lightblue', 'lightcoral']
            ),
            row=1, col=1
        )
        
        # 2. Distribuição de Tickets
        fig.add_trace(
            go.Histogram(
                x=df['Total_Liquido'],
                nbinsx=30,
                name='Distribuição de Tickets',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # 3. Padrão Temporal de Clientes
        monthly_customers = df.groupby('Ano_Mes').agg({
            'Total_Liquido': ['count', 'sum', 'mean']
        })
        monthly_customers.columns = ['Transactions', 'Revenue', 'Avg_Ticket']
        
        fig.add_trace(
            go.Scatter(
                x=monthly_customers.index,
                y=monthly_customers['Transactions'],
                mode='lines+markers',
                name='Transações/Mês',
                line=dict(color='blue', width=2)
            ),
            row=1, col=3
        )
        
        # 4. RFM Simplificado
        current_date = df['Data'].max()
        
        # Criar clusters baseados em valor e frequência temporal
        rfm_data = df.groupby(df['Total_Liquido'].round(-2)).agg({
            'Data': lambda x: (current_date - x.max()).days,  # Recency
            'Total_Liquido': ['count', 'sum']  # Frequency, Monetary
        })
        rfm_data.columns = ['Recency', 'Frequency', 'Monetary']
        
        # Classificar em quartis
        rfm_data['R_Score'] = pd.qcut(rfm_data['Recency'], 4, labels=[4,3,2,1]).astype(int)
        rfm_data['F_Score'] = pd.qcut(rfm_data['Frequency'], 4, labels=[1,2,3,4]).astype(int)
        rfm_data['M_Score'] = pd.qcut(rfm_data['Monetary'], 4, labels=[1,2,3,4]).astype(int)
        
        # Score total
        rfm_data['Total_Score'] = rfm_data['R_Score'] + rfm_data['F_Score'] + rfm_data['M_Score']
        
        fig.add_trace(
            go.Scatter(
                x=rfm_data['Frequency'],
                y=rfm_data['Monetary'],
                mode='markers',
                marker=dict(
                    color=rfm_data['Total_Score'],
                    colorscale='Viridis',
                    size=np.sqrt(rfm_data['Monetary']) / 100,
                    showscale=True,
                    colorbar=dict(title="RFM Score")
                ),
                name='Segments RFM'
            ),
            row=2, col=1
        )
        
        # 5. Análise de Oportunidades
        opportunities = []
        
        premium_pct = value_segments['Premium (>R$5K)'] / sum(value_segments.values()) * 100
        entry_pct = value_segments['Entry (< R$1K)'] / sum(value_segments.values()) * 100
        
        if premium_pct < 10:
            opportunities.append(['Up-sell', 'Produtos Premium', f'{premium_pct:.1f}% clientes premium'])
        
        if entry_pct > 40:
            opportunities.append(['Retenção', 'Programa Fidelidade', f'{entry_pct:.1f}% entry-level'])
        
        opportunities.append(['Cross-sell', 'Categorias Complementares', 'Analisar padrões'])
        opportunities.append(['Segmentação', 'Customer ID', 'Implementar tracking'])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Estratégia</b>', '<b>Foco</b>', '<b>Observação</b>'],
                    fill_color='lightblue',
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*opportunities)),
                    fill_color='white',
                    align='left'
                )
            ),
            row=2, col=2
        )
        
        # 6. Métricas de Cliente
        total_revenue = df['Total_Liquido'].sum()
        avg_ticket = df['Total_Liquido'].mean()
        estimated_customers = len(df[df['Total_Liquido'] > 2000]) + (len(df[df['Total_Liquido'] <= 2000]) * 0.7)
        estimated_clv = avg_ticket * 2.3 * 3.5  # Benchmark joalherias
        
        metrics = [
            ['Total Transações', f'{len(df):,}'],
            ['Receita Total', f'R$ {total_revenue:,.0f}'],
            ['Ticket Médio', f'R$ {avg_ticket:,.0f}'],
            ['CLV Estimado', f'R$ {estimated_clv:,.0f}'],
            ['Clientes Estimados', f'{int(estimated_customers):,}']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Métrica</b>', '<b>Valor</b>'],
                    fill_color='lightgreen',
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*metrics)),
                    fill_color='white',
                    align='left'
                )
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title=f"<b>{title} - Segmentação de Clientes</b>",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def _create_financial_overview(self, df: pd.DataFrame, title: str) -> go.Figure:
        """Criar visão geral financeira com KPIs principais."""
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Evolução da Receita', 'Composição da Receita',
                'Performance vs Benchmark', 'Projeção Financeira',
                'Margem Estimada', 'KPIs Críticos'
            ],
            specs=[
                [{"secondary_y": True}, {"type": "pie"}, {}],
                [{}, {}, {"type": "table"}]
            ]
        )
        
        # 1. Evolução da Receita com Crescimento
        monthly_revenue = df.groupby('Ano_Mes')['Total_Liquido'].sum()
        monthly_growth = monthly_revenue.pct_change() * 100
        
        fig.add_trace(
            go.Scatter(
                x=monthly_revenue.index,
                y=monthly_revenue.values,
                mode='lines+markers',
                name='Receita Mensal',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=monthly_growth.index[1:],
                y=monthly_growth.dropna(),
                name='Crescimento %',
                marker_color=['green' if x > 0 else 'red' for x in monthly_growth.dropna()],
                yaxis='y2',
                opacity=0.6
            ),
            row=1, col=1, secondary_y=True
        )
        
        # 2. Composição da Receita por Categoria
        if 'Grupo_Produto' in df.columns and df['Grupo_Produto'].notna().any():
            category_revenue = df.groupby('Grupo_Produto')['Total_Liquido'].sum()
            
            fig.add_trace(
                go.Pie(
                    labels=category_revenue.index,
                    values=category_revenue.values,
                    hole=0.4,
                    textinfo='label+percent'
                ),
                row=1, col=2
            )
        
        # 3. Performance vs Benchmark
        current_aov = df['Total_Liquido'].mean()
        benchmark_aov = 1500
        
        fig.add_trace(
            go.Bar(
                x=['AOV Atual', 'Benchmark'],
                y=[current_aov, benchmark_aov],
                name='Comparação AOV',
                marker_color=['blue', 'gray'],
                text=[f'R$ {current_aov:,.0f}', f'R$ {benchmark_aov:,.0f}'],
                textposition='outside'
            ),
            row=1, col=3
        )
        
        # 4. Projeção Financeira
        x_numeric = np.arange(len(monthly_revenue))
        z = np.polyfit(x_numeric, monthly_revenue.values, 1)
        p = np.poly1d(z)
        
        # Projeção
        future_periods = 6
        future_x = np.arange(len(monthly_revenue), len(monthly_revenue) + future_periods)
        future_projection = p(future_x)
        
        # Histórico
        fig.add_trace(
            go.Scatter(
                x=monthly_revenue.index,
                y=monthly_revenue.values,
                mode='lines+markers',
                name='Histórico',
                line=dict(color='blue', width=2)
            ),
            row=2, col=1
        )
        
        # Projeção
        last_date = pd.to_datetime(monthly_revenue.index[-1])
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_periods, freq='M')
        
        fig.add_trace(
            go.Scatter(
                x=[d.strftime('%Y-%m') for d in future_dates],
                y=future_projection,
                mode='lines+markers',
                name='Projeção',
                line=dict(color='orange', width=2, dash='dot'),
                marker=dict(symbol='diamond')
            ),
            row=2, col=1
        )
        
        # 5. Margem Estimada por Categoria
        if 'Grupo_Produto' in df.columns and df['Grupo_Produto'].notna().any():
            margin_estimates = {
                'Anéis': 0.62, 'Brincos': 0.58, 'Colares': 0.55, 'Pulseiras': 0.60,
                'Alianças': 0.45, 'Pingentes': 0.65, 'Correntes': 0.52, 'Outros': 0.50
            }
            
            category_margins = []
            category_names = []
            for category in category_revenue.index:
                margin = margin_estimates.get(str(category), 0.55)
                estimated_profit = category_revenue[category] * margin
                category_margins.append(estimated_profit)
                category_names.append(str(category))
            
            fig.add_trace(
                go.Bar(
                    x=category_names,
                    y=category_margins,
                    name='Margem Estimada',
                    marker_color='lightgreen',
                    text=[f'R$ {v:,.0f}' for v in category_margins],
                    textposition='outside'
                ),
                row=2, col=2
            )
        
        # 6. Tabela de KPIs Críticos
        total_revenue = df['Total_Liquido'].sum()
        total_transactions = len(df)
        avg_ticket = df['Total_Liquido'].mean()
        days_in_period = (df['Data'].max() - df['Data'].min()).days + 1
        daily_revenue = total_revenue / days_in_period
        
        # Calcular crescimento se possível
        if len(monthly_revenue) > 1:
            latest_growth = monthly_growth.iloc[-1]
        else:
            latest_growth = 0
        
        kpis = [
            ['Receita Total', f'R$ {total_revenue:,.2f}'],
            ['Ticket Médio', f'R$ {avg_ticket:,.2f}'],
            ['Transações', f'{total_transactions:,}'],
            ['Receita/Dia', f'R$ {daily_revenue:,.2f}'],
            ['Crescimento M/M', f'{latest_growth:.1f}%'],
            ['Margem Estimada', '55%']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>KPI</b>', '<b>Valor</b>'],
                    fill_color='lightblue',
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*kpis)),
                    fill_color='white',
                    align='left'
                )
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title=f"<b>{title} - Visão Geral Financeira</b>",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def _apply_professional_theme(self, fig: go.Figure, chart_type: str) -> go.Figure:
        """Aplicar tema profissional adequado para joalherias."""
        
        # Cores do tema joalheria (elegante e luxuoso)
        color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'accent': '#2ca02c',
            'gold': '#FFD700',
            'silver': '#C0C0C0',
            'bronze': '#CD7F32',
            'dark': '#2F4F4F',
            'light': '#F8F8FF'
        }
        
        fig.update_layout(
            # Tema geral
            template='plotly_white',
            
            # Fonte profissional
            font=dict(
                family="Arial, sans-serif",
                size=11,
                color=color_palette['dark']
            ),
            
            # Título principal
            title=dict(
                font=dict(size=18, color=color_palette['dark']),
                x=0.5,
                xanchor='center'
            ),
            
            # Background
            paper_bgcolor='white',
            plot_bgcolor='rgba(248,248,255,0.8)',
            
            # Margins
            margin=dict(l=80, r=80, t=100, b=80),
            
            # Hover
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial",
                bordercolor=color_palette['primary']
            ),
            
            # Legend
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=color_palette['dark'],
                borderwidth=1
            )
        )
        
        return fig
    
    def _export_visualization(self, fig: go.Figure, chart_type: str, 
                            export_format: str, save_output: bool) -> str:
        """Exportar visualização no formato especificado."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_filename = f"{chart_type}_{timestamp}"
            
            if not save_output:
                if export_format == "json":
                    return fig.to_json()
                elif export_format == "html":
                    return fig.to_html(full_html=False, include_plotlyjs='cdn')
                else:
                    return "Formato não suportado para preview sem salvar"
            
            # Criar diretório output se não existir
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            if export_format == "html":
                output_path = f"{output_dir}/{base_filename}.html"
                
                # HTML customizado com tema profissional
                html_template = f"""
                            <!DOCTYPE html>
                            <html>
                            <head>
                                <meta charset="utf-8">
                                <title>{chart_type.replace('_', ' ').title()} - Análise de Joalheria</title>
                                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                                <style>
                                    body {{
                                        font-family: Arial, sans-serif;
                                        margin: 0;
                                        padding: 20px;
                                        background-color: #f8f9fa;
                                    }}
                                    .header {{
                                        text-align: center;
                                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                        color: white;
                                        padding: 20px;
                                        border-radius: 10px;
                                        margin-bottom: 20px;
                                    }}
                                    .chart-container {{
                                        background: white;
                                        border-radius: 10px;
                                        padding: 20px;
                                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                                    }}
                                    .footer {{
                                        text-align: center;
                                        margin-top: 20px;
                                        color: #666;
                                        font-size: 12px;
                                    }}
                                </style>
                            </head>
                            <body>
                                <div class="header">
                                    <h1>{chart_type.replace('_', ' ').title()}</h1>
                                    <p>Análise Profissional para Joalherias - {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
                                </div>
                                <div class="chart-container">
                                    {fig.to_html(full_html=False, include_plotlyjs=False)}
                                </div>
                                <div class="footer">
                                    <p>Gerado por Insights AI - Sistema de Business Intelligence</p>
                                </div>
                            </body>
                            </html>
                            """
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html_template)
                
                return f"✅ Visualização salva em: {output_path}\n\n📊 Gráfico interativo criado com sucesso!\n\n🔗 Abra o arquivo HTML no navegador para visualização completa."
                
            elif export_format == "json":
                output_path = f"{output_dir}/{base_filename}.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(fig.to_json())
                return f"✅ Dados JSON salvos em: {output_path}"
                
            elif export_format == "png":
                output_path = f"{output_dir}/{base_filename}.png"
                try:
                    fig.write_image(output_path, width=1400, height=1000, scale=2)
                    return f"✅ Imagem PNG salva em: {output_path}"
                except Exception as e:
                    return f"❌ Erro ao salvar PNG (instale kaleido): {str(e)}\n\n💡 Use formato 'html' para visualização interativa."
            
            else:
                return f"❌ Formato '{export_format}' não suportado. Use: html, json, png"
                
        except Exception as e:
            return f"❌ Erro na exportação: {str(e)}"
