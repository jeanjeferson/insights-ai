from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import base64
warnings.filterwarnings('ignore')

class BusinessIntelligenceToolInput(BaseModel):
    """Schema otimizado para Business Intelligence com validações robustas."""
    
    analysis_type: str = Field(
        ..., 
        description="""Tipos de análise de Business Intelligence especializados:
        
        🎯 ANÁLISES EXECUTIVAS:
        - 'executive_summary': Resumo C-level com KPIs críticos e insights acionáveis
        - 'executive_dashboard': Dashboard visual executivo interativo completo
        
        💰 ANÁLISES FINANCEIRAS:
        - 'financial_analysis': KPIs financeiros + visualizações + benchmarks + forecasting
        - 'profitability_analysis': Rentabilidade real com análise de custo/preço/desconto
        
        👥 ANÁLISES DE CLIENTES:
        - 'customer_intelligence': RFM + segmentação + retenção + visualizações completas
        
        📦 ANÁLISES DE PRODUTOS:
        - 'product_performance': ABC analysis + rankings + categorias + alertas de estoque
        
        🆕 ANÁLISES ESPECIALIZADAS:
        - 'demographic_analysis': Comportamento por idade/sexo/estado civil
        - 'geographic_analysis': Performance por estado/cidade com mapas interativos
        - 'sales_team_analysis': Performance individual e coletiva de vendedores
        - 'comprehensive_report': Relatório executivo integrado completo
        """,
        json_schema_extra={"example": "executive_summary"}
    )
    
    data_csv: str = Field(
        default="data/vendas.csv", 
        description="Caminho para arquivo CSV de vendas. Use 'data/vendas.csv' para dados principais.",
        json_schema_extra={"example": "data/vendas.csv"}
    )
    
    time_period: str = Field(
        default="last_12_months", 
        description="Período de análise: 'last_month' (30 dias), 'last_quarter' (90 dias), 'last_12_months' (365 dias), 'ytd' (ano atual).",
        json_schema_extra={
            "pattern": "^(last_month|last_quarter|last_12_months|ytd)$"
        }
    )
    
    output_format: str = Field(
        default="interactive", 
        description="Formato de saída: 'text' (texto), 'interactive' (JSON estruturado), 'html' (relatório web).",
        json_schema_extra={
            "pattern": "^(text|interactive|html)$"
        }
    )
    
    include_forecasts: bool = Field(
        default=True, 
        description="Incluir projeções e previsões. Recomendado: True para análises estratégicas."
    )
    
    detail_level: str = Field(
        default="summary", 
        description="Nível de detalhamento: 'summary' (resumido), 'detailed' (completo).",
        json_schema_extra={
            "pattern": "^(summary|detailed)$"
        }
    )
    
    export_file: bool = Field(
        default=True, 
        description="Salvar arquivo de saída. Recomendado: True para relatórios executivos."
    )
    
    @field_validator('analysis_type')
    @classmethod
    def validate_analysis_type(cls, v):
        valid_types = [
            'executive_summary', 'executive_dashboard', 'financial_analysis', 
            'profitability_analysis', 'customer_intelligence', 'product_performance',
            'demographic_analysis', 'geographic_analysis', 'sales_team_analysis', 
            'comprehensive_report'
        ]
        if v not in valid_types:
            raise ValueError(f"analysis_type deve ser um de: {valid_types}")
        return v

class BusinessIntelligenceTool(BaseTool):
    """
    📊 PLATAFORMA UNIFICADA DE BUSINESS INTELLIGENCE PARA JOALHERIAS
    
    QUANDO USAR:
    - Criar relatórios executivos completos para tomada de decisão
    - Gerar dashboards visuais interativos para apresentações
    - Analisar performance financeira com visualizações profissionais
    - Segmentar clientes com análises RFM visuais
    - Avaliar performance de produtos com gráficos ABC
    - Criar análises demográficas e geográficas detalhadas
    
    CASOS DE USO ESPECÍFICOS:
    - analysis_type='executive_summary': Relatório C-level com KPIs críticos
    - analysis_type='executive_dashboard': Dashboard interativo para apresentações
    - analysis_type='financial_analysis': Análise financeira com forecasting
    - analysis_type='customer_intelligence': Segmentação de clientes com RFM
    - analysis_type='product_performance': Performance de produtos com ABC
    - analysis_type='comprehensive_report': Relatório executivo completo
    
    RESULTADOS ENTREGUES:
    - Relatórios executivos profissionais em HTML/JSON
    - Dashboards interativos com visualizações Plotly
    - Análises com benchmarks do setor joalheiro
    - Forecasting baseado em tendências históricas
    - Alertas inteligentes automatizados
    - Scores de saúde do negócio
    - Recomendações estratégicas acionáveis
    """
    
    name: str = "Business Intelligence Platform"
    description: str = (
        "Plataforma unificada de Business Intelligence para joalherias com relatórios executivos e dashboards interativos. "
        "Cria análises visuais profissionais, forecasting, segmentação de clientes e benchmarks do setor. "
        "Ideal para apresentações executivas, tomada de decisão estratégica e monitoramento de performance."
    )
    args_schema: Type[BaseModel] = BusinessIntelligenceToolInput
    
    def _run(self, analysis_type: str, data_csv: str = "data/vendas.csv",
             time_period: str = "last_12_months", output_format: str = "interactive",
             include_forecasts: bool = True, detail_level: str = "summary", 
             export_file: bool = True) -> str:
        try:
            print(f"🚀 Iniciando análise unificada: {analysis_type}")
            
            # 1. CAMADA DE DADOS - Preparação unificada
            df = self._prepare_data_unified(data_csv, time_period)
            if df is None or len(df) < 5:
                return "❌ Erro: Dados insuficientes (mínimo 5 registros)"
            
            # 2. CAMADA DE ANÁLISE - Roteamento unificado
            analysis_router = {
                'executive_summary': self._create_executive_summary,
                'executive_dashboard': self._create_executive_dashboard,
                'financial_analysis': self._create_financial_analysis,
                'profitability_analysis': self._create_profitability_analysis,
                'customer_intelligence': self._create_customer_intelligence,
                'product_performance': self._create_product_performance,
                'demographic_analysis': self._create_demographic_analysis,
                'geographic_analysis': self._create_geographic_analysis,
                'sales_team_analysis': self._create_sales_team_analysis,
                'comprehensive_report': self._create_comprehensive_report
            }
            
            if analysis_type not in analysis_router:
                return f"❌ Análise '{analysis_type}' não suportada. Opções: {list(analysis_router.keys())}"
            
            # 3. EXECUÇÃO DA ANÁLISE
            result = analysis_router[analysis_type](df, include_forecasts, detail_level)
            
            # 4. CAMADA DE SAÍDA - Formatação unificada
            return self._format_output_unified(analysis_type, result, output_format, export_file)
            
        except Exception as e:
            return f"❌ Erro na análise unificada: {str(e)}"
    
    # ========================================
    # CAMADA DE DADOS - Preparação Unificada
    # ========================================
    
    def _prepare_data_unified(self, data_csv: str, time_period: str) -> Optional[pd.DataFrame]:
        """Preparação UNIFICADA de dados - elimina duplicação."""
        try:
            print("📊 Preparando dados unificados...")
            
            # Carregar dados
            df = pd.read_csv(data_csv, sep=';', encoding='utf-8')
            
            # Validações essenciais
            required_cols = ['Data', 'Total_Liquido']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"❌ Colunas essenciais faltando: {missing_cols}")
                return None
            
            # Conversões de tipos
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
            df['Total_Liquido'] = pd.to_numeric(df['Total_Liquido'], errors='coerce')
            df['Quantidade'] = pd.to_numeric(df.get('Quantidade', 1), errors='coerce').fillna(1)
            
            # Filtrar dados válidos
            df = df.dropna(subset=['Data', 'Total_Liquido'])
            df = df[df['Total_Liquido'] > 0]
            
            # Filtrar por período
            df = self._filter_by_period(df, time_period)
            
            # Adicionar métricas derivadas UNIFICADAS
            df = self._add_derived_metrics(df)
            
            # Simular dados faltantes se necessário
            df = self._simulate_missing_data(df)
            
            print(f"✅ Dados preparados: {len(df)} registros, {len(df.columns)} colunas")
            return df
            
        except Exception as e:
            print(f"❌ Erro na preparação: {str(e)}")
            return None
    
    def _filter_by_period(self, df: pd.DataFrame, time_period: str) -> pd.DataFrame:
        """Filtrar dados por período."""
        current_date = df['Data'].max()
        
        period_map = {
            'last_month': 30,
            'last_quarter': 90,
            'last_12_months': 365,
            'ytd': None  # Ano atual
        }
        
        if time_period == 'ytd':
            start_date = datetime(current_date.year, 1, 1)
        else:
            days = period_map.get(time_period, 365)
            start_date = current_date - timedelta(days=days)
        
        return df[df['Data'] >= start_date]
    
    def _add_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adicionar métricas derivadas UNIFICADAS."""
        # Métricas temporais
        df['Ano'] = df['Data'].dt.year
        df['Mes'] = df['Data'].dt.month
        df['Trimestre'] = df['Data'].dt.quarter
        df['Dia_Semana'] = df['Data'].dt.dayofweek
        df['Mes_Nome'] = df['Data'].dt.strftime('%b')
        df['Ano_Mes'] = df['Data'].dt.to_period('M').astype(str)
        df['Semana'] = df['Data'].dt.isocalendar().week
        
        # Métricas financeiras
        df['Preco_Unitario'] = df['Total_Liquido'] / df['Quantidade']
        
        # Métricas demográficas (se disponíveis)
        if 'Idade' in df.columns:
            df['Idade'] = pd.to_numeric(df['Idade'], errors='coerce')
            df['Faixa_Etaria'] = pd.cut(df['Idade'], 
                                       bins=[0, 25, 35, 45, 55, 100],
                                       labels=['18-25', '26-35', '36-45', '46-55', '55+'],
                                       include_lowest=True)
        
        # Métricas de rentabilidade (se disponíveis)
        if 'Custo_Produto' in df.columns and 'Total_Liquido' in df.columns:
            df['Custo_Produto'] = pd.to_numeric(df['Custo_Produto'], errors='coerce')
            df['Margem_Real'] = df['Total_Liquido'] - df['Custo_Produto']
            df['Margem_Percentual'] = (df['Margem_Real'] / df['Total_Liquido'] * 100).fillna(0)
        
        return df
    
    def _simulate_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simular dados faltantes para análises completas."""
        # Customer ID
        if 'Codigo_Cliente' not in df.columns:
            np.random.seed(42)
            n_customers = max(len(df) // 4, 1)
            df['Codigo_Cliente'] = 'CUST_' + np.random.choice(range(1, n_customers + 1), len(df)).astype(str)
        
        # Dados de produto
        if 'Codigo_Produto' not in df.columns:
            df['Codigo_Produto'] = 'PROD_' + (df.index + 1).astype(str)
        
        # Preencher campos opcionais
        optional_fields = {
            'Descricao_Produto': 'Produto Genérico',
            'Grupo_Produto': 'Categoria Geral',
            'Metal': 'Ouro',
            'Sexo': 'F',
            'Estado': 'SP',
            'Nome_Vendedor': 'Vendedor Padrão'
        }
        
        for field, default_value in optional_fields.items():
            if field not in df.columns:
                df[field] = default_value
        
        return df
    
    # ========================================
    # CAMADA DE ANÁLISE - Métodos Unificados
    # ========================================
    
    def _calculate_kpis_unified(self, df: pd.DataFrame) -> Dict[str, Any]:
        """KPIs UNIFICADOS - base para todas as análises."""
        # KPIs de Receita
        total_revenue = df['Total_Liquido'].sum()
        avg_ticket = df['Total_Liquido'].mean()
        total_transactions = len(df)
        total_customers = df['Codigo_Cliente'].nunique()
        
        # KPIs de Crescimento
        monthly_revenue = df.groupby('Ano_Mes')['Total_Liquido'].sum()
        if len(monthly_revenue) >= 2:
            current_month = monthly_revenue.iloc[-1]
            previous_month = monthly_revenue.iloc[-2]
            mom_growth = ((current_month - previous_month) / previous_month * 100) if previous_month > 0 else 0
        else:
            mom_growth = 0
        
        # KPIs de Produtos
        total_products = df['Codigo_Produto'].nunique()
        avg_items_per_transaction = df['Quantidade'].mean()
        
        # Período de análise
        period_days = (df['Data'].max() - df['Data'].min()).days + 1
        daily_revenue = total_revenue / period_days
        
        return {
            'revenue': {
                'total': total_revenue,
                'daily': daily_revenue,
                'avg_ticket': avg_ticket,
                'mom_growth': mom_growth
            },
            'customers': {
                'total': total_customers,
                'avg_per_day': total_customers / period_days,
                'new_estimate': max(total_customers // 4, 1)  # Estimativa
            },
            'products': {
                'total': total_products,
                'avg_items_per_transaction': avg_items_per_transaction
            },
            'transactions': {
                'total': total_transactions,
                'per_day': total_transactions / period_days
            },
            'period': {
                'days': period_days,
                'start': df['Data'].min().strftime('%d/%m/%Y'),
                'end': df['Data'].max().strftime('%d/%m/%Y')
            }
        }
    
    def _calculate_customer_segments_unified(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Segmentação de clientes UNIFICADA."""
        # RFM básico
        current_date = df['Data'].max()
        customer_rfm = df.groupby('Codigo_Cliente').agg({
            'Data': lambda x: (current_date - x.max()).days,  # Recency
            'Total_Liquido': ['count', 'sum'],  # Frequency, Monetary
        })
        customer_rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        # Classificação simples
        def classify_customer(row):
            if row['Monetary'] > customer_rfm['Monetary'].quantile(0.8):
                return 'VIP'
            elif row['Frequency'] > 1 and row['Recency'] <= 60:
                return 'Leal'
            elif row['Recency'] <= 30:
                return 'Ativo'
            elif row['Recency'] <= 90:
                return 'Em Risco'
            else:
                return 'Perdido'
        
        customer_rfm['Segment'] = customer_rfm.apply(classify_customer, axis=1)
        segments = customer_rfm['Segment'].value_counts().to_dict()
        
        return {
            'distribution': segments,
            'total_customers': len(customer_rfm),
            'avg_monetary': customer_rfm['Monetary'].mean(),
            'avg_frequency': customer_rfm['Frequency'].mean(),
            'customer_data': customer_rfm
        }
    
    def _calculate_abc_analysis_unified(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise ABC UNIFICADA."""
        product_revenue = df.groupby('Codigo_Produto')['Total_Liquido'].sum().sort_values(ascending=False)
        total_revenue = product_revenue.sum()
        
        cumsum_pct = product_revenue.cumsum() / total_revenue
        
        class_a = (cumsum_pct <= 0.8).sum()
        class_b = ((cumsum_pct > 0.8) & (cumsum_pct <= 0.95)).sum()
        class_c = len(product_revenue) - class_a - class_b
        
        return {
            'classes': {'A': class_a, 'B': class_b, 'C': class_c},
            'class_a_revenue_share': (product_revenue.head(class_a).sum() / total_revenue * 100),
            'total_products': len(product_revenue),
            'top_products': product_revenue.head(10).to_dict()
        }
    
    # ========================================
    # ANÁLISES ESPECÍFICAS - Métodos Únicos
    # ========================================
    
    def _create_executive_summary(self, df: pd.DataFrame, include_forecasts: bool, detail_level: str) -> Dict[str, Any]:
        """Resumo executivo textual."""
        kpis = self._calculate_kpis_unified(df)
        segments = self._calculate_customer_segments_unified(df)
        abc = self._calculate_abc_analysis_unified(df)
        
        # Alertas executivos
        alerts = []
        if kpis['revenue']['mom_growth'] < -10:
            alerts.append(f"🚨 ALERTA: Queda de receita de {kpis['revenue']['mom_growth']:.1f}% no último mês")
        
        vip_pct = (segments['distribution'].get('VIP', 0) / segments['total_customers']) * 100
        if vip_pct < 10:
            alerts.append(f"⚠️ Baixa % de clientes VIP ({vip_pct:.1f}%) - focar em up-selling")
        
        # Insights executivos
        insights = [
            f"Receita total: R$ {kpis['revenue']['total']:,.2f} no período",
            f"Crescimento mensal: {kpis['revenue']['mom_growth']:+.1f}%",
            f"Base de clientes: {kpis['customers']['total']} únicos",
            f"Ticket médio: R$ {kpis['revenue']['avg_ticket']:,.2f}",
            f"Produtos ativos: {kpis['products']['total']} itens",
            f"Concentração: Top 20% produtos = {abc['class_a_revenue_share']:.1f}% receita"
        ]
        
        # Recomendações
        recommendations = []
        if kpis['revenue']['mom_growth'] > 5:
            recommendations.append("✅ Crescimento positivo - manter estratégia atual")
        else:
            recommendations.append("📈 Implementar campanhas de reativação urgentes")
        
        if vip_pct < 15:
            recommendations.append("🎯 Desenvolver programa de fidelidade VIP")
        
        if abc['classes']['A'] < abc['total_products'] * 0.1:
            recommendations.append("📦 Revisar mix de produtos - muitos itens de baixo giro")
        
        return {
            'kpis': kpis,
            'segments': segments,
            'abc_analysis': abc,
            'alerts': alerts,
            'insights': insights,
            'recommendations': recommendations,
            'health_score': self._calculate_health_score(kpis, segments)
        }
    
    def _create_executive_dashboard(self, df: pd.DataFrame, include_forecasts: bool, detail_level: str) -> go.Figure:
        """Dashboard executivo visual unificado com layout otimizado."""
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                '📈 Evolução da Receita Mensal', '',  # Linha 1: Gráfico temporal ocupa toda a linha
                '🏆 Top Produtos por Receita', '👥 Segmentação de Clientes',  # Linha 2: 2 gráficos pequenos
                '📊 Performance Mensal (Barras)', '',  # Linha 3: Gráfico temporal ocupa toda a linha
                '🎯 Análise ABC de Produtos', '📋 KPIs Principais'  # Linha 4: 2 gráficos pequenos
            ],
            specs=[
                [{"colspan": 2, "secondary_y": True}, None],  # Linha 1: gráfico temporal full width
                [{"type": "bar"}, {"type": "pie"}],  # Linha 2: 2 colunas
                [{"colspan": 2}, None],  # Linha 3: gráfico temporal full width
                [{"type": "bar"}, {"type": "table"}]  # Linha 4: 2 colunas
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        # 1. Evolução da Receita (Linha 1 - Full Width)
        monthly_revenue = df.groupby('Ano_Mes')['Total_Liquido'].sum()
        fig.add_trace(
            go.Scatter(
                x=monthly_revenue.index,
                y=monthly_revenue.values,
                mode='lines+markers',
                name='Receita Mensal',
                line=dict(color='#1f77b4', width=4),
                marker=dict(size=8, color='#1f77b4'),
                hovertemplate='<b>%{x}</b><br>Receita: R$ %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Forecast se habilitado
        if include_forecasts and len(monthly_revenue) >= 3:
            x_numeric = np.arange(len(monthly_revenue))
            z = np.polyfit(x_numeric, monthly_revenue.values, 1)
            p = np.poly1d(z)
            
            fig.add_trace(
                go.Scatter(
                    x=monthly_revenue.index,
                    y=p(x_numeric),
                    mode='lines',
                    name='Tendência',
                    line=dict(color='#ff7f0e', width=3, dash='dash'),
                    hovertemplate='<b>Tendência</b><br>%{x}: R$ %{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Top Produtos (Linha 2 - Coluna 1)
        if 'Descricao_Produto' in df.columns:
            top_products = df.groupby('Descricao_Produto')['Total_Liquido'].sum().nlargest(6)
            fig.add_trace(
                go.Bar(
                    y=[str(prod)[:30] + '...' if len(str(prod)) > 30 else str(prod) for prod in top_products.index],
                    x=top_products.values,
                    orientation='h',
                    name='Top Produtos',
                    marker_color='#2ca02c',
                    hovertemplate='<b>%{y}</b><br>Receita: R$ %{x:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 3. Segmentos de Clientes (Linha 2 - Coluna 2)
        segments = self._calculate_customer_segments_unified(df)
        colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        fig.add_trace(
            go.Pie(
                labels=list(segments['distribution'].keys()),
                values=list(segments['distribution'].values()),
                hole=0.4,
                name='Segmentos',
                marker_colors=colors[:len(segments['distribution'])],
                hovertemplate='<b>%{label}</b><br>Clientes: %{value}<br>Percentual: %{percent}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 4. Performance Mensal em Barras (Linha 3 - Full Width)
        fig.add_trace(
            go.Bar(
                x=monthly_revenue.index,
                y=monthly_revenue.values,
                name='Receita Mensal',
                marker_color='#17becf',
                hovertemplate='<b>%{x}</b><br>Receita: R$ %{y:,.0f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # 5. Análise ABC (Linha 4 - Coluna 1)
        abc = self._calculate_abc_analysis_unified(df)
        fig.add_trace(
            go.Bar(
                x=list(abc['classes'].keys()),
                y=list(abc['classes'].values()),
                name='Produtos por Classe',
                marker_color=['#FFD700', '#C0C0C0', '#CD7F32'],  # Ouro, Prata, Bronze
                hovertemplate='<b>Classe %{x}</b><br>Produtos: %{y}<extra></extra>'
            ),
            row=4, col=1
        )
        
        # 6. KPIs (Tabela) (Linha 4 - Coluna 2)
        kpis = self._calculate_kpis_unified(df)
        kpi_data = [
            ['💰 Receita Total', f"R$ {kpis['revenue']['total']:,.0f}"],
            ['🎫 Ticket Médio', f"R$ {kpis['revenue']['avg_ticket']:,.0f}"],
            ['📈 Crescimento M/M', f"{kpis['revenue']['mom_growth']:+.1f}%"],
            ['👥 Total Clientes', f"{kpis['customers']['total']:,}"],
            ['📦 Produtos Ativos', f"{kpis['products']['total']:,}"],
            ['📅 Período', f"{kpis['period']['days']} dias"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>KPI</b>', '<b>Valor</b>'], 
                    fill_color='#1f77b4',
                    font=dict(color='white', size=12)
                ),
                cells=dict(
                    values=list(zip(*kpi_data)), 
                    fill_color=['#f8f9fa', '#ffffff'],
                    font=dict(size=11),
                    height=30
                )
            ),
            row=4, col=2
        )
        
        # Configurações finais do layout
        fig.update_layout(
            title={
                'text': "<b>📊 Dashboard Executivo Unificado</b>",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#1f77b4'}
            },
            height=1200,  # Aumentado para acomodar 4 linhas
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='#f8f9fa',
            margin=dict(t=80, b=50, l=50, r=50)
        )
        
        # Atualizar eixos para melhor visualização
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e0e0e0')
        
        # Configurações específicas para gráficos temporais
        fig.update_xaxes(tickangle=45, row=1, col=1)  # Evolução da receita
        fig.update_xaxes(tickangle=45, row=3, col=1)  # Performance mensal
        
        return fig
    
    def _create_financial_analysis(self, df: pd.DataFrame, include_forecasts: bool, detail_level: str) -> Dict[str, Any]:
        """Análise financeira consolidada."""
        kpis = self._calculate_kpis_unified(df)
        
        # Análise de tendência
        monthly_data = df.groupby('Ano_Mes')['Total_Liquido'].sum()
        if len(monthly_data) >= 2:
            trend = "Crescimento" if monthly_data.iloc[-1] > monthly_data.iloc[-2] else "Declínio"
        else:
            trend = "Estável"
        
        # Análise de sazonalidade
        monthly_avg = df.groupby('Mes')['Total_Liquido'].mean()
        peak_month = monthly_avg.idxmax()
        low_month = monthly_avg.idxmin()
        
        # Métricas de eficiência
        efficiency_metrics = {
            'revenue_per_customer': kpis['revenue']['total'] / kpis['customers']['total'],
            'revenue_per_transaction': kpis['revenue']['avg_ticket'],
            'transactions_per_customer': kpis['transactions']['total'] / kpis['customers']['total']
        }
        
        return {
            'kpis': kpis,
            'trend_analysis': {
                'direction': trend,
                'monthly_data': monthly_data.to_dict()
            },
            'seasonality': {
                'peak_month': peak_month,
                'low_month': low_month,
                'seasonal_index': monthly_avg.to_dict()
            },
            'efficiency_metrics': efficiency_metrics
        }
    
    def _create_profitability_analysis(self, df: pd.DataFrame, include_forecasts: bool, detail_level: str) -> Dict[str, Any]:
        """Análise de rentabilidade com dados reais."""
        # Verificar se temos dados de custo
        has_cost_data = 'Custo_Produto' in df.columns and df['Custo_Produto'].notna().any()
        
        if has_cost_data:
            # Análise real com dados de custo
            df['Margem_Real'] = df['Total_Liquido'] - df['Custo_Produto']
            df['Margem_Percentual'] = (df['Margem_Real'] / df['Total_Liquido'] * 100).fillna(0)
            
            profitability_metrics = {
                'total_margin': df['Margem_Real'].sum(),
                'avg_margin_pct': df['Margem_Percentual'].mean(),
                'total_cost': df['Custo_Produto'].sum(),
                'markup_avg': ((df['Total_Liquido'].sum() / df['Custo_Produto'].sum()) - 1) * 100
            }
            
            # Análise por categoria
            if 'Grupo_Produto' in df.columns:
                category_profitability = df.groupby('Grupo_Produto').agg({
                    'Margem_Real': 'sum',
                    'Margem_Percentual': 'mean'
                }).to_dict()
            else:
                category_profitability = {}
        else:
            # Estimativa baseada em benchmarks do setor
            estimated_margin_pct = 55  # Benchmark joalherias
            estimated_total_margin = df['Total_Liquido'].sum() * (estimated_margin_pct / 100)
            
            profitability_metrics = {
                'estimated_margin': estimated_total_margin,
                'estimated_margin_pct': estimated_margin_pct,
                'note': 'Estimativa baseada em benchmark do setor (55%)'
            }
            category_profitability = {}
        
        return {
            'has_real_data': has_cost_data,
            'profitability_metrics': profitability_metrics,
            'category_profitability': category_profitability
        }
    
    def _create_customer_intelligence(self, df: pd.DataFrame, include_forecasts: bool, detail_level: str) -> Dict[str, Any]:
        """Inteligência de clientes unificada."""
        segments = self._calculate_customer_segments_unified(df)
        
        # Análise de valor do cliente
        customer_value = df.groupby('Codigo_Cliente').agg({
            'Total_Liquido': 'sum',
            'Data': ['min', 'max', 'count']
        })
        customer_value.columns = ['Total_Spent', 'First_Purchase', 'Last_Purchase', 'Purchase_Count']
        
        # CLV estimado
        avg_lifespan_months = 24  # Estimativa para joalherias
        customer_value['CLV_Estimate'] = (customer_value['Total_Spent'] / customer_value['Purchase_Count']) * avg_lifespan_months
        
        # Análise de retenção
        current_date = df['Data'].max()
        customer_value['Days_Since_Last_Purchase'] = (current_date - pd.to_datetime(customer_value['Last_Purchase'])).dt.days
        
        retention_analysis = {
            'active_30d': len(customer_value[customer_value['Days_Since_Last_Purchase'] <= 30]),
            'at_risk_90d': len(customer_value[customer_value['Days_Since_Last_Purchase'] > 90]),
            'lost_180d': len(customer_value[customer_value['Days_Since_Last_Purchase'] > 180])
        }
        
        return {
            'segments': segments,
            'customer_value_analysis': {
                'avg_clv': customer_value['CLV_Estimate'].mean(),
                'top_customers': customer_value.nlargest(10, 'Total_Spent')[['Total_Spent', 'Purchase_Count']].to_dict('index')
            },
            'retention_analysis': retention_analysis
        }
    
    def _create_product_performance(self, df: pd.DataFrame, include_forecasts: bool, detail_level: str) -> Dict[str, Any]:
        """Performance de produtos unificada."""
        abc = self._calculate_abc_analysis_unified(df)
        
        # Performance por produto
        product_performance = df.groupby('Codigo_Produto').agg({
            'Total_Liquido': ['sum', 'mean', 'count'],
            'Quantidade': 'sum',
            'Data': ['min', 'max']
        })
        product_performance.columns = ['Total_Revenue', 'Avg_Price', 'Transaction_Count', 'Total_Quantity', 'First_Sale', 'Last_Sale']
        
        # Alertas de inventário
        current_date = df['Data'].max()
        product_performance['Days_Since_Last_Sale'] = (current_date - pd.to_datetime(product_performance['Last_Sale'])).dt.days
        
        inventory_alerts = {
            'slow_movers_60d': len(product_performance[product_performance['Days_Since_Last_Sale'] > 60]),
            'dead_stock_120d': len(product_performance[product_performance['Days_Since_Last_Sale'] > 120])
        }
        
        # Performance por categoria
        category_performance = {}
        if 'Grupo_Produto' in df.columns:
            category_performance = df.groupby('Grupo_Produto').agg({
                'Total_Liquido': 'sum',
                'Quantidade': 'sum',
                'Codigo_Produto': 'nunique'
            }).to_dict()
        
        return {
            'abc_analysis': abc,
            'product_rankings': {
                'top_by_revenue': product_performance.nlargest(10, 'Total_Revenue')[['Total_Revenue', 'Transaction_Count']].to_dict('index'),
                'top_by_volume': product_performance.nlargest(10, 'Total_Quantity')[['Total_Quantity', 'Total_Revenue']].to_dict('index')
            },
            'inventory_alerts': inventory_alerts,
            'category_performance': category_performance
        }
    
    def _create_demographic_analysis(self, df: pd.DataFrame, include_forecasts: bool, detail_level: str) -> Dict[str, Any]:
        """Análise demográfica especializada."""
        demographic_data = {}
        
        # Análise por gênero
        if 'Sexo' in df.columns and df['Sexo'].notna().any():
            gender_analysis = df.groupby('Sexo').agg({
                'Total_Liquido': ['sum', 'mean', 'count'],
                'Codigo_Cliente': 'nunique'
            })
            gender_analysis.columns = ['Total_Revenue', 'Avg_Ticket', 'Transactions', 'Customers']
            demographic_data['gender'] = gender_analysis.to_dict()
        
        # Análise por faixa etária
        if 'Faixa_Etaria' in df.columns and df['Faixa_Etaria'].notna().any():
            age_analysis = df.groupby('Faixa_Etaria').agg({
                'Total_Liquido': ['sum', 'mean'],
                'Codigo_Cliente': 'nunique'
            })
            age_analysis.columns = ['Total_Revenue', 'Avg_Ticket', 'Customers']
            demographic_data['age'] = age_analysis.to_dict()
        
        # Análise por estado civil
        if 'Estado_Civil' in df.columns and df['Estado_Civil'].notna().any():
            marital_analysis = df.groupby('Estado_Civil').agg({
                'Total_Liquido': ['sum', 'mean'],
                'Codigo_Cliente': 'nunique'
            })
            marital_analysis.columns = ['Total_Revenue', 'Avg_Ticket', 'Customers']
            demographic_data['marital_status'] = marital_analysis.to_dict()
        
        return {
            'demographic_data': demographic_data,
            'has_demographic_data': len(demographic_data) > 0
        }
    
    def _create_geographic_analysis(self, df: pd.DataFrame, include_forecasts: bool, detail_level: str) -> Dict[str, Any]:
        """Análise geográfica especializada."""
        geographic_data = {}
        
        # Análise por estado
        if 'Estado' in df.columns and df['Estado'].notna().any():
            state_analysis = df.groupby('Estado').agg({
                'Total_Liquido': ['sum', 'mean'],
                'Codigo_Cliente': 'nunique'
            })
            state_analysis.columns = ['Total_Revenue', 'Avg_Ticket', 'Customers']
            geographic_data['states'] = state_analysis.to_dict()
        
        # Análise por cidade
        if 'Cidade' in df.columns and df['Cidade'].notna().any():
            city_analysis = df.groupby('Cidade')['Total_Liquido'].sum().nlargest(10)
            geographic_data['top_cities'] = city_analysis.to_dict()
        
        return {
            'geographic_data': geographic_data,
            'has_geographic_data': len(geographic_data) > 0
        }
    
    def _create_sales_team_analysis(self, df: pd.DataFrame, include_forecasts: bool, detail_level: str) -> Dict[str, Any]:
        """Análise da equipe de vendas."""
        sales_data = {}
        
        if 'Nome_Vendedor' in df.columns and df['Nome_Vendedor'].notna().any():
            vendor_analysis = df.groupby('Nome_Vendedor').agg({
                'Total_Liquido': ['sum', 'mean', 'count'],
                'Codigo_Cliente': 'nunique'
            })
            vendor_analysis.columns = ['Total_Revenue', 'Avg_Ticket', 'Transactions', 'Customers']
            
            # Rankings
            sales_data = {
                'top_by_revenue': vendor_analysis.nlargest(10, 'Total_Revenue')[['Total_Revenue', 'Transactions']].to_dict('index'),
                'top_by_volume': vendor_analysis.nlargest(10, 'Transactions')[['Transactions', 'Total_Revenue']].to_dict('index'),
                'performance_matrix': vendor_analysis.to_dict()
            }
        
        return {
            'sales_data': sales_data,
            'has_sales_team_data': len(sales_data) > 0
        }
    
    def _create_comprehensive_report(self, df: pd.DataFrame, include_forecasts: bool, detail_level: str) -> Dict[str, Any]:
        """Relatório executivo completo integrado."""
        # Compilar todas as análises
        executive_summary = self._create_executive_summary(df, include_forecasts, detail_level)
        financial_analysis = self._create_financial_analysis(df, include_forecasts, detail_level)
        customer_intelligence = self._create_customer_intelligence(df, include_forecasts, detail_level)
        product_performance = self._create_product_performance(df, include_forecasts, detail_level)
        
        # Score geral do negócio
        overall_score = self._calculate_health_score(
            executive_summary['kpis'],
            customer_intelligence['segments']
        )
        
        return {
            'executive_summary': executive_summary,
            'financial_analysis': financial_analysis,
            'customer_intelligence': customer_intelligence,
            'product_performance': product_performance,
            'overall_health_score': overall_score,
            'timestamp': datetime.now().strftime('%d/%m/%Y %H:%M')
        }
    
    # ========================================
    # MÉTODOS AUXILIARES UNIFICADOS
    # ========================================
    
    def _calculate_health_score(self, kpis: Dict, segments: Dict) -> Dict[str, Any]:
        """Calcular score de saúde do negócio."""
        # Componentes do score (0-100)
        revenue_score = min(100, max(0, (kpis['revenue']['mom_growth'] + 20) * 2.5))  # -20% a +20% = 0 a 100
        
        vip_pct = (segments['distribution'].get('VIP', 0) / segments['total_customers']) * 100
        customer_score = min(100, vip_pct * 5)  # 20% VIP = 100 pontos
        
        efficiency_score = min(100, (kpis['revenue']['avg_ticket'] / 1000) * 50)  # R$ 2000 ticket = 100 pontos
        
        overall_score = (revenue_score + customer_score + efficiency_score) / 3
        
        # Classificação
        if overall_score >= 80:
            classification = "Excelente"
            color = "green"
        elif overall_score >= 60:
            classification = "Bom"
            color = "blue"
        elif overall_score >= 40:
            classification = "Regular"
            color = "orange"
        else:
            classification = "Crítico"
            color = "red"
        
        return {
            'overall_score': round(overall_score, 1),
            'classification': classification,
            'color': color,
            'components': {
                'revenue': round(revenue_score, 1),
                'customer': round(customer_score, 1),
                'efficiency': round(efficiency_score, 1)
            }
        }
    
    # ========================================
    # CAMADA DE SAÍDA - Formatação Unificada
    # ========================================
    
    def _format_output_unified(self, analysis_type: str, result: Union[Dict, go.Figure], 
                              output_format: str, export_file: bool) -> str:
        """Formatação de saída UNIFICADA."""
        
        if isinstance(result, go.Figure):
            # Resultado é uma figura Plotly
            if export_file:
                return self._export_plotly_figure(result, analysis_type, output_format)
            else:
                return result.to_html(full_html=False, include_plotlyjs='cdn')
        
        else:
            # Resultado é um dicionário de dados
            if output_format == "interactive":
                return self._format_interactive_result(analysis_type, result)
            elif output_format == "html" and export_file:
                return self._export_html_report(analysis_type, result)
            else:
                return self._format_text_result(analysis_type, result)
    
    def _format_interactive_result(self, analysis_type: str, result: Dict) -> str:
        """Formatação interativa rica."""
        output = [
            f"🎯 **{analysis_type.replace('_', ' ').title()}**",
            "=" * 50
        ]
        
        if analysis_type == "executive_summary":
            kpis = result['kpis']
            output.extend([
                "📊 **KPIs PRINCIPAIS:**",
                f"💰 Receita Total: R$ {kpis['revenue']['total']:,.2f}",
                f"📈 Crescimento M/M: {kpis['revenue']['mom_growth']:+.1f}%",
                f"🎫 Ticket Médio: R$ {kpis['revenue']['avg_ticket']:,.2f}",
                f"👥 Total Clientes: {kpis['customers']['total']:,}",
                "",
                "🚨 **ALERTAS:**"
            ])
            
            for alert in result['alerts']:
                output.append(f"  • {alert}")
            
            output.extend(["", "💡 **INSIGHTS:**"])
            for insight in result['insights']:
                output.append(f"  • {insight}")
            
            output.extend(["", "🎯 **RECOMENDAÇÕES:**"])
            for rec in result['recommendations']:
                output.append(f"  • {rec}")
            
            # Score de saúde
            health = result['health_score']
            output.extend([
                "",
                f"🏥 **SCORE DE SAÚDE: {health['overall_score']}/100 ({health['classification']})**",
                f"  📊 Receita: {health['components']['revenue']}/100",
                f"  👥 Clientes: {health['components']['customer']}/100",
                f"  ⚡ Eficiência: {health['components']['efficiency']}/100"
            ])
        
        elif analysis_type == "customer_intelligence":
            segments = result['segments']
            output.extend([
                "👥 **SEGMENTAÇÃO DE CLIENTES:**",
                f"Total de clientes: {segments['total_customers']:,}",
                "",
                "📊 **DISTRIBUIÇÃO POR SEGMENTO:**"
            ])
            
            for segment, count in segments['distribution'].items():
                pct = (count / segments['total_customers']) * 100
                output.append(f"  • {segment}: {count:,} ({pct:.1f}%)")
            
            output.extend([
                "",
                f"💰 Valor médio por cliente: R$ {segments['avg_monetary']:,.2f}",
                f"🔄 Frequência média: {segments['avg_frequency']:.1f} compras"
            ])
        
        elif analysis_type == "product_performance":
            abc = result['abc_analysis']
            output.extend([
                "📦 **ANÁLISE ABC DE PRODUTOS:**",
                f"  🥇 Classe A: {abc['classes']['A']} produtos ({abc['class_a_revenue_share']:.1f}% da receita)",
                f"  🥈 Classe B: {abc['classes']['B']} produtos",
                f"  🥉 Classe C: {abc['classes']['C']} produtos",
                "",
                "🏆 **TOP PRODUTOS POR RECEITA:**"
            ])
            
            for i, (product, data) in enumerate(list(result['product_rankings']['top_by_revenue'].items())[:5], 1):
                output.append(f"  {i}. {product}: R$ {data['Total_Revenue']:,.2f}")
            
            alerts = result['inventory_alerts']
            if alerts['slow_movers_60d'] > 0 or alerts['dead_stock_120d'] > 0:
                output.extend([
                    "",
                    "⚠️ **ALERTAS DE INVENTÁRIO:**",
                    f"  📉 Slow movers (60+ dias): {alerts['slow_movers_60d']} produtos",
                    f"  💀 Dead stock (120+ dias): {alerts['dead_stock_120d']} produtos"
                ])
        
        elif analysis_type == "financial_analysis":
            kpis = result['kpis']
            output.extend([
                "💰 **KPIs FINANCEIROS:**",
                f"💰 Receita Total: R$ {kpis['revenue']['total']:,.2f}",
                f"📊 Receita Diária Média: R$ {kpis['revenue']['daily']:,.2f}",
                f"🎫 Ticket Médio: R$ {kpis['revenue']['avg_ticket']:,.2f}",
                f"📈 Crescimento M/M: {kpis['revenue']['mom_growth']:+.1f}%",
                "",
                "📊 **ANÁLISE DE TENDÊNCIA:**"
            ])
            
            trend = result['trend_analysis']
            output.append(f"  📈 Direção: {trend['direction']}")
            
            seasonality = result['seasonality']
            output.extend([
                "",
                "🌐 **ANÁLISE SAZONAL:**",
                f"  🏆 Melhor mês: {seasonality['peak_month']}",
                f"  📉 Menor mês: {seasonality['low_month']}",
                "",
                "⚡ **MÉTRICAS DE EFICIÊNCIA:**"
            ])
            
            efficiency = result['efficiency_metrics']
            output.extend([
                f"  💰 Receita por Cliente: R$ {efficiency['revenue_per_customer']:,.2f}",
                f"  🎫 Receita por Transação: R$ {efficiency['revenue_per_transaction']:,.2f}",
                f"  🔄 Transações por Cliente: {efficiency['transactions_per_customer']:.1f}"
            ])
        
        return "\n".join(output)
    
    def _format_text_result(self, analysis_type: str, result: Dict) -> str:
        """Formatação de texto simples."""
        return f"""
                {analysis_type.replace('_', ' ').title()}
                {'='*50}

                Análise gerada em: {datetime.now().strftime('%d/%m/%Y %H:%M')}

                Resumo dos principais resultados:
                {json.dumps(result, indent=2, default=str, ensure_ascii=False)}
                        """
    
    def _export_plotly_figure(self, fig: go.Figure, analysis_type: str, output_format: str) -> str:
        """Exportar figura Plotly."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"unified_{analysis_type}_{timestamp}"
            
            # Criar diretório se não existir
            os.makedirs("output", exist_ok=True)
            
            if output_format == "html":
                filepath = f"output/{filename}.html"
                fig.write_html(filepath, include_plotlyjs=True)
                return f"✅ Dashboard salvo em: {filepath}"
            else:
                filepath = f"output/{filename}.json"
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(fig.to_json())
                return f"✅ Dados salvos em: {filepath}"
                
        except Exception as e:
            return f"❌ Erro na exportação: {str(e)}"
    
    def _export_html_report(self, analysis_type: str, result: Dict) -> str:
        """Exportar relatório HTML profissional."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"unified_report_{analysis_type}_{timestamp}.html"
            
            # Criar diretório se não existir
            os.makedirs("output", exist_ok=True)
            filepath = f"output/{filename}"
            
            html_content = f"""
            <!DOCTYPE html>
            <html lang="pt-BR">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>📊 {analysis_type.replace('_', ' ').title()} - Insights AI</title>
                <style>
                    * {{
                        margin: 0;
                        padding: 0;
                        box-sizing: border-box;
                    }}
                    
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                        min-height: 100vh;
                        padding: 20px;
                        line-height: 1.6;
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
                    
                    .kpi-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 20px;
                        margin-bottom: 30px;
                    }}
                    
                    .kpi-card {{
                        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
                        padding: 25px;
                        border-radius: 15px;
                        border: 1px solid #e2e8f0;
                        text-align: center;
                        transition: transform 0.3s ease;
                    }}
                    
                    .kpi-card:hover {{
                        transform: translateY(-5px);
                        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                    }}
                    
                    .kpi-value {{
                        font-size: 2.5em;
                        font-weight: bold;
                        color: #1e293b;
                        margin-bottom: 10px;
                    }}
                    
                    .kpi-icon {{
                        font-size: 2.5em;
                        margin-bottom: 15px;
                        opacity: 0.8;
                    }}
                    
                    .kpi-label {{
                        color: #374151;
                        font-size: 1.1em;
                        font-weight: 600;
                        margin-bottom: 5px;
                    }}
                    
                    .kpi-meta {{
                        color: #6b7280;
                        font-size: 0.9em;
                        font-style: italic;
                        margin-top: 5px;
                    }}
                    
                    .kpi-change {{
                        font-size: 0.9em;
                        margin-top: 5px;
                        padding: 5px 10px;
                        border-radius: 20px;
                        display: inline-block;
                    }}
                    
                    .kpi-change.positive {{
                        background: #dcfce7;
                        color: #166534;
                    }}
                    
                    .kpi-change.negative {{
                        background: #fef2f2;
                        color: #dc2626;
                    }}
                    
                    .section {{
                        margin-bottom: 40px;
                    }}
                    
                    .section-title {{
                        font-size: 1.8em;
                        color: #1e293b;
                        margin-bottom: 20px;
                        padding-bottom: 10px;
                        border-bottom: 3px solid #3b82f6;
                    }}
                    
                    .alert {{
                        background: #fef2f2;
                        border-left: 4px solid #ef4444;
                        padding: 15px 20px;
                        margin: 15px 0;
                        border-radius: 0 8px 8px 0;
                    }}
                    
                    .alert-title {{
                        font-weight: bold;
                        color: #dc2626;
                        margin-bottom: 5px;
                    }}
                    
                    .insight {{
                        background: #f0f9ff;
                        border-left: 4px solid #3b82f6;
                        padding: 15px 20px;
                        margin: 15px 0;
                        border-radius: 0 8px 8px 0;
                    }}
                    
                    .insight-title {{
                        font-weight: bold;
                        color: #1e40af;
                        margin-bottom: 5px;
                    }}
                    
                    .recommendation {{
                        background: #f0fdf4;
                        border-left: 4px solid #22c55e;
                        padding: 15px 20px;
                        margin: 15px 0;
                        border-radius: 0 8px 8px 0;
                    }}
                    
                    .recommendation-title {{
                        font-weight: bold;
                        color: #166534;
                        margin-bottom: 5px;
                    }}
                    
                    .health-score {{
                        background: linear-gradient(135deg, #fef3c7 0%, #fbbf24 100%);
                        padding: 30px;
                        border-radius: 15px;
                        text-align: center;
                        margin: 30px 0;
                    }}
                    
                    .health-score-value {{
                        font-size: 4em;
                        font-weight: bold;
                        color: #92400e;
                    }}
                    
                    .health-score-label {{
                        font-size: 1.5em;
                        color: #92400e;
                        margin-top: 10px;
                    }}
                    
                    .data-table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                        background: white;
                        border-radius: 10px;
                        overflow: hidden;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    }}
                    
                    .data-table th {{
                        background: #f8fafc;
                        padding: 15px;
                        text-align: left;
                        font-weight: 600;
                        color: #374151;
                        border-bottom: 1px solid #e5e7eb;
                    }}
                    
                    .data-table td {{
                        padding: 12px 15px;
                        border-bottom: 1px solid #f3f4f6;
                    }}
                    
                    .data-table tr:hover {{
                        background: #f9fafb;
                    }}
                    
                    .footer {{
                        background: #f8fafc;
                        padding: 30px;
                        text-align: center;
                        color: #64748b;
                        border-top: 1px solid #e2e8f0;
                    }}
                    
                    .footer-logo {{
                        font-size: 1.5em;
                        margin-bottom: 10px;
                    }}
                    
                    @media (max-width: 768px) {{
                        .container {{
                            margin: 10px;
                            border-radius: 10px;
                        }}
                        
                        .header {{
                            padding: 20px;
                        }}
                        
                        .header h1 {{
                            font-size: 2em;
                        }}
                        
                        .content {{
                            padding: 20px;
                        }}
                        
                        .kpi-grid {{
                            grid-template-columns: 1fr;
                        }}
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>📊 {analysis_type.replace('_', ' ').title()}</h1>
                        <p>Relatório Executivo de Business Intelligence</p>
                        <p>Gerado em: {datetime.now().strftime('%d/%m/%Y às %H:%M')}</p>
                    </div>
                    
                    <div class="content">
                        {self._format_html_content(analysis_type, result)}
                    </div>
                    
                    <div class="footer">
                        <div class="footer-logo">🚀 Insights AI</div>
                        <p>Business Intelligence Platform</p>
                        <p>Relatório gerado automaticamente com dados em tempo real</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Salvar arquivo
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return f"✅ Relatório HTML salvo em: {filepath}"
            
        except Exception as e:
            return f"❌ Erro na exportação HTML: {str(e)}"
    
    def _format_html_content(self, analysis_type: str, result: Dict) -> str:
        """Formatar conteúdo específico para HTML."""
        if analysis_type == "executive_summary":
            return self._format_executive_summary_html(result)
        elif analysis_type == "financial_analysis":
            return self._format_financial_analysis_html(result)
        elif analysis_type == "customer_intelligence":
            return self._format_customer_intelligence_html(result)
        elif analysis_type == "product_performance":
            return self._format_product_performance_html(result)
        else:
            # Formato genérico para outros tipos
            return f"""
                        <div class="section">
                            <h2 class="section-title">📊 Dados da Análise</h2>
                            <pre style="background: #f8fafc; padding: 20px; border-radius: 10px; overflow-x: auto;">
            {json.dumps(result, indent=2, default=str, ensure_ascii=False)}
                            </pre>
                        </div>
                        """
                
    def _format_executive_summary_html(self, result: Dict) -> str:
        """Formatar resumo executivo em HTML."""
        kpis = result.get('kpis', {})
        revenue = kpis.get('revenue', {})
        customers = kpis.get('customers', {})
        products = kpis.get('products', {})
        transactions = kpis.get('transactions', {})
        
        html = f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-icon">💰</div>
                <div class="kpi-value">R$ {revenue.get('total', 0):,.0f}</div>
                <div class="kpi-label">Receita Total</div>
                <div class="kpi-change {'positive' if revenue.get('mom_growth', 0) > 0 else 'negative'}">
                    {revenue.get('mom_growth', 0):+.1f}% vs mês anterior
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🎫</div>
                <div class="kpi-value">R$ {revenue.get('avg_ticket', 0):,.0f}</div>
                <div class="kpi-label">Ticket Médio</div>
                <div class="kpi-meta">Receita por transação</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">👥</div>
                <div class="kpi-value">{customers.get('total', 0):,}</div>
                <div class="kpi-label">Total de Clientes</div>
                <div class="kpi-meta">Base ativa no período</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">📦</div>
                <div class="kpi-value">{products.get('total', 0):,}</div>
                <div class="kpi-label">Produtos Ativos</div>
                <div class="kpi-meta">Itens com vendas</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🛒</div>
                <div class="kpi-value">{transactions.get('total', 0):,}</div>
                <div class="kpi-label">Total de Transações</div>
                <div class="kpi-meta">{transactions.get('per_day', 0):.1f} por dia</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">📈</div>
                <div class="kpi-value">R$ {revenue.get('daily', 0):,.0f}</div>
                <div class="kpi-label">Receita Diária</div>
                <div class="kpi-meta">Média do período</div>
            </div>
        </div>
        """
        
        # Score de saúde
        health = result.get('health_score', {})
        if health:
            html += f"""
            <div class="health-score">
                <div class="health-score-value">{health.get('overall_score', 0)}/100</div>
                <div class="health-score-label">🏥 Score de Saúde do Negócio</div>
                <p style="margin-top: 10px; color: #92400e;">
                    Classificação: <strong>{health.get('classification', 'N/A')}</strong>
                </p>
            </div>
            """
        
        # Alertas
        alerts = result.get('alerts', [])
        if alerts:
            html += """
            <div class="section">
                <h2 class="section-title">🚨 Alertas Importantes</h2>
            """
            for alert in alerts:
                html += f"""
                <div class="alert">
                    <div class="alert-title">⚠️ Atenção</div>
                    <div>{alert}</div>
                </div>
                """
            html += "</div>"
        
        # Insights
        insights = result.get('insights', [])
        if insights:
            html += """
            <div class="section">
                <h2 class="section-title">💡 Insights Estratégicos</h2>
            """
            for insight in insights:
                html += f"""
                <div class="insight">
                    <div class="insight-title">💡 Insight</div>
                    <div>{insight}</div>
                </div>
                """
            html += "</div>"
        
        # Recomendações
        recommendations = result.get('recommendations', [])
        if recommendations:
            html += """
            <div class="section">
                <h2 class="section-title">🎯 Recomendações</h2>
            """
            for rec in recommendations:
                html += f"""
                <div class="recommendation">
                    <div class="recommendation-title">🎯 Recomendação</div>
                    <div>{rec}</div>
                </div>
                """
            html += "</div>"
        
        return html
    
    def _format_financial_analysis_html(self, result: Dict) -> str:
        """Formatar análise financeira em HTML."""
        kpis = result.get('kpis', {})
        revenue = kpis.get('revenue', {})
        customers = kpis.get('customers', {})
        transactions = kpis.get('transactions', {})
        products = kpis.get('products', {})
        period = kpis.get('period', {})
        
        trend = result.get('trend_analysis', {})
        seasonality = result.get('seasonality', {})
        efficiency = result.get('efficiency_metrics', {})
        
        html = f"""
        <!-- KPIs Financeiros -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-icon">💰</div>
                <div class="kpi-value">R$ {revenue.get('total', 0):,.0f}</div>
                <div class="kpi-label">Receita Total</div>
                <div class="kpi-meta">Período de {period.get('days', 0)} dias</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">📈</div>
                <div class="kpi-value">{revenue.get('mom_growth', 0):+.1f}%</div>
                <div class="kpi-label">Crescimento M/M</div>
                <div class="kpi-meta">{'📈 Positivo' if revenue.get('mom_growth', 0) > 0 else '📉 Negativo'}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">📊</div>
                <div class="kpi-value">R$ {revenue.get('daily', 0):,.0f}</div>
                <div class="kpi-label">Receita Diária</div>
                <div class="kpi-meta">Média do período</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🎫</div>
                <div class="kpi-value">R$ {revenue.get('avg_ticket', 0):,.0f}</div>
                <div class="kpi-label">Ticket Médio</div>
                <div class="kpi-meta">Por transação</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🛒</div>
                <div class="kpi-value">{transactions.get('total', 0):,}</div>
                <div class="kpi-label">Total Transações</div>
                <div class="kpi-meta">{transactions.get('per_day', 0):.1f} por dia</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">👥</div>
                <div class="kpi-value">{customers.get('total', 0):,}</div>
                <div class="kpi-label">Base de Clientes</div>
                <div class="kpi-meta">{customers.get('avg_per_day', 0):.1f} novos/dia</div>
            </div>
        </div>
        
        <!-- Análise de Tendência -->
        <div class="section">
            <h2 class="section-title">📈 Análise de Tendência</h2>
            <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 30px; border-radius: 15px; margin-bottom: 30px;">
                <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
                    <div style="font-size: 3em; margin-right: 20px;">
                        {'📈' if trend.get('direction') == 'Crescimento' else '📉' if trend.get('direction') == 'Declínio' else '➡️'}
                    </div>
                    <div>
                        <div style="font-size: 2em; font-weight: bold; color: #1e40af;">{trend.get('direction', 'Estável')}</div>
                        <div style="color: #3730a3; font-size: 1.2em;">Tendência geral do período</div>
                    </div>
                </div>
            </div>
            
            <!-- Dados Mensais -->
            <h3 style="color: #1e293b; margin-bottom: 15px;">📅 Performance Mensal</h3>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Mês</th>
                        <th>Receita</th>
                        <th>% do Total</th>
                        <th>Variação</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Adicionar dados mensais
        monthly_data = trend.get('monthly_data', {})
        total_revenue = sum(monthly_data.values()) if monthly_data else 1
        previous_value = None
        
        for month, value in monthly_data.items():
            percentage = (value / total_revenue * 100) if total_revenue > 0 else 0
            
            if previous_value is not None:
                variation = ((value - previous_value) / previous_value * 100) if previous_value > 0 else 0
                variation_text = f"{variation:+.1f}%"
                variation_color = "color: #22c55e;" if variation > 0 else "color: #ef4444;" if variation < 0 else "color: #6b7280;"
            else:
                variation_text = "-"
                variation_color = "color: #6b7280;"
            
            html += f"""
                    <tr>
                        <td><strong>{month}</strong></td>
                        <td>R$ {value:,.2f}</td>
                        <td>{percentage:.1f}%</td>
                        <td style="{variation_color}">{variation_text}</td>
                    </tr>
            """
            previous_value = value
        
        html += """
                </tbody>
            </table>
        </div>
        
        <!-- Análise Sazonal -->
        <div class="section">
            <h2 class="section-title">🌐 Análise de Sazonalidade</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px;">
                <div style="background: linear-gradient(135deg, #dcfce7 0%, #22c55e 100%); padding: 25px; border-radius: 15px; text-align: center;">
                    <div style="font-size: 2em; margin-bottom: 10px;">🏆</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: #166534;">Mês """ + str(seasonality.get('peak_month', 'N/A')) + """</div>
                    <div style="color: #166534; font-weight: 600;">Melhor Performance</div>
                    <div style="color: #166534; font-size: 0.9em;">Pico de vendas</div>
                </div>
                <div style="background: linear-gradient(135deg, #fecaca 0%, #ef4444 100%); padding: 25px; border-radius: 15px; text-align: center;">
                    <div style="font-size: 2em; margin-bottom: 10px;">📉</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: #7f1d1d;">Mês """ + str(seasonality.get('low_month', 'N/A')) + """</div>
                    <div style="color: #7f1d1d; font-weight: 600;">Menor Performance</div>
                    <div style="color: #7f1d1d; font-size: 0.9em;">Período mais fraco</div>
                </div>
            </div>
        </div>
        
        <!-- Métricas de Eficiência -->
        <div class="section">
            <h2 class="section-title">⚡ Métricas de Eficiência</h2>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-icon">💰</div>
                    <div class="kpi-value">R$ """ + f"{efficiency.get('revenue_per_customer', 0):,.0f}" + """</div>
                    <div class="kpi-label">Receita por Cliente</div>
                    <div class="kpi-meta">Valor médio por cliente</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-icon">🎫</div>
                    <div class="kpi-value">R$ """ + f"{efficiency.get('revenue_per_transaction', 0):,.0f}" + """</div>
                    <div class="kpi-label">Receita por Transação</div>
                    <div class="kpi-meta">Ticket médio</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-icon">🔄</div>
                    <div class="kpi-value">""" + f"{efficiency.get('transactions_per_customer', 0):.1f}" + """</div>
                    <div class="kpi-label">Transações por Cliente</div>
                    <div class="kpi-meta">Frequência de compra</div>
                </div>
            </div>
        </div>
        
        <!-- Insights Financeiros -->
        <div class="section">
            <h2 class="section-title">💡 Insights Financeiros</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
        """
        
        # Gerar insights baseados nos dados
        growth = revenue.get('mom_growth', 0)
        if growth > 10:
            html += f"""
                <div class="insight">
                    <div class="insight-title">🚀 Crescimento Acelerado</div>
                    <div>Crescimento de {growth:+.1f}% indica momentum positivo. Considere expandir investimentos em marketing.</div>
                </div>
            """
        elif growth < -10:
            html += f"""
                <div class="alert">
                    <div class="alert-title">⚠️ Declínio Significativo</div>
                    <div>Queda de {growth:.1f}% requer ação imediata. Revise estratégias de preço e promoções.</div>
                </div>
            """
        
        avg_ticket = revenue.get('avg_ticket', 0)
        if avg_ticket > 3000:
            html += f"""
                <div class="insight">
                    <div class="insight-title">💎 Alto Valor por Transação</div>
                    <div>Ticket médio de R$ {avg_ticket:,.0f} indica produtos de alto valor. Foque em qualidade e experiência premium.</div>
                </div>
            """
        
        transactions_per_customer = efficiency.get('transactions_per_customer', 0)
        if transactions_per_customer < 2:
            html += f"""
                <div class="recommendation">
                    <div class="recommendation-title">🎯 Oportunidade de Retenção</div>
                    <div>Média de {transactions_per_customer:.1f} compras por cliente. Implemente programas de fidelidade para aumentar frequência.</div>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _format_customer_intelligence_html(self, result: Dict) -> str:
        """Formatar inteligência de clientes em HTML."""
        segments = result.get('segments', {})
        customer_value = result.get('customer_value_analysis', {})
        retention = result.get('retention_analysis', {})
        
        total_customers = segments.get('total_customers', 0)
        distribution = segments.get('distribution', {})
        
        html = f"""
        <!-- KPIs de Clientes -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-icon">👥</div>
                <div class="kpi-value">{total_customers:,}</div>
                <div class="kpi-label">Total de Clientes</div>
                <div class="kpi-meta">Base ativa no período</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">💎</div>
                <div class="kpi-value">{distribution.get('VIP', 0):,}</div>
                <div class="kpi-label">Clientes VIP</div>
                <div class="kpi-meta">{(distribution.get('VIP', 0) / total_customers * 100) if total_customers > 0 else 0:.1f}% da base</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">💰</div>
                <div class="kpi-value">R$ {customer_value.get('avg_clv', 0):,.0f}</div>
                <div class="kpi-label">CLV Médio</div>
                <div class="kpi-meta">Customer Lifetime Value</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🔄</div>
                <div class="kpi-value">{segments.get('avg_frequency', 0):.1f}</div>
                <div class="kpi-label">Frequência Média</div>
                <div class="kpi-meta">Compras por cliente</div>
            </div>
        </div>
        
        <!-- Segmentação RFM -->
        <div class="section">
            <h2 class="section-title">🎯 Segmentação RFM de Clientes</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px;">
        """
        
        # Cores para cada segmento
        segment_colors = {
            'VIP': 'linear-gradient(135deg, #fef3c7 0%, #fbbf24 100%)',
            'Leal': 'linear-gradient(135deg, #dcfce7 0%, #22c55e 100%)',
            'Ativo': 'linear-gradient(135deg, #dbeafe 0%, #3b82f6 100%)',
            'Em Risco': 'linear-gradient(135deg, #fed7aa 0%, #f97316 100%)',
            'Perdido': 'linear-gradient(135deg, #fecaca 0%, #ef4444 100%)'
        }
        
        segment_icons = {
            'VIP': '💎',
            'Leal': '💚',
            'Ativo': '🟢',
            'Em Risco': '⚠️',
            'Perdido': '🔴'
        }
        
        for segment, count in distribution.items():
            percentage = (count / total_customers * 100) if total_customers > 0 else 0
            color = segment_colors.get(segment, 'linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%)')
            icon = segment_icons.get(segment, '👤')
            
            html += f"""
                <div style="background: {color}; padding: 20px; border-radius: 15px; text-align: center;">
                    <div style="font-size: 2em; margin-bottom: 10px;">{icon}</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: #1f2937;">{count:,}</div>
                    <div style="color: #374151; font-weight: 600; margin-bottom: 5px;">{segment}</div>
                    <div style="color: #6b7280; font-size: 0.9em;">{percentage:.1f}% da base</div>
                </div>
            """
        
        html += """
            </div>
        </div>
        
        <!-- Top Clientes por Valor -->
        <div class="section">
            <h2 class="section-title">🏆 Top 10 Clientes por Valor</h2>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Posição</th>
                        <th>Código do Cliente</th>
                        <th>Valor Total Gasto</th>
                        <th>Número de Compras</th>
                        <th>Ticket Médio</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Adicionar top clientes
        top_customers = customer_value.get('top_customers', {})
        for i, (customer, data) in enumerate(list(top_customers.items())[:10], 1):
            total_spent = data.get('Total_Spent', 0)
            purchase_count = data.get('Purchase_Count', 0)
            avg_ticket = total_spent / purchase_count if purchase_count > 0 else 0
            
            html += f"""
                    <tr>
                        <td><strong>#{i}</strong></td>
                        <td><code>{customer}</code></td>
                        <td><strong>R$ {total_spent:,.2f}</strong></td>
                        <td>{purchase_count:,}</td>
                        <td>R$ {avg_ticket:,.2f}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        # Análise de Retenção com valores corretos
        html += f"""
        <!-- Análise de Retenção -->
        <div class="section">
            <h2 class="section-title">📊 Análise de Retenção</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
                <div style="background: linear-gradient(135deg, #dcfce7 0%, #22c55e 100%); padding: 25px; border-radius: 15px; text-align: center;">
                    <div style="font-size: 2em; margin-bottom: 10px;">✅</div>
                    <div style="font-size: 2em; font-weight: bold; color: #166534;">{retention.get('active_30d', 0):,}</div>
                    <div style="color: #166534; font-weight: 600; margin-bottom: 5px;">Ativos (30 dias)</div>
                    <div style="color: #166534; font-size: 0.9em;">Compraram recentemente</div>
                </div>
                <div style="background: linear-gradient(135deg, #fed7aa 0%, #f97316 100%); padding: 25px; border-radius: 15px; text-align: center;">
                    <div style="font-size: 2em; margin-bottom: 10px;">⚠️</div>
                    <div style="font-size: 2em; font-weight: bold; color: #9a3412;">{retention.get('at_risk_90d', 0):,}</div>
                    <div style="color: #9a3412; font-weight: 600; margin-bottom: 5px;">Em Risco (90+ dias)</div>
                    <div style="color: #9a3412; font-size: 0.9em;">Precisam de reativação</div>
                </div>
                <div style="background: linear-gradient(135deg, #fecaca 0%, #ef4444 100%); padding: 25px; border-radius: 15px; text-align: center;">
                    <div style="font-size: 2em; margin-bottom: 10px;">❌</div>
                    <div style="font-size: 2em; font-weight: bold; color: #7f1d1d;">{retention.get('lost_180d', 0):,}</div>
                    <div style="color: #7f1d1d; font-weight: 600; margin-bottom: 5px;">Perdidos (180+ dias)</div>
                    <div style="color: #7f1d1d; font-size: 0.9em;">Sem compras há muito tempo</div>
                </div>
            </div>
        </div>
        
        <!-- Insights e Recomendações -->
        <div class="section">
            <h2 class="section-title">💡 Insights e Recomendações</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
        """
        
        # Gerar insights baseados nos dados
        vip_percentage = (distribution.get('VIP', 0) / total_customers * 100) if total_customers > 0 else 0
        lost_percentage = (distribution.get('Perdido', 0) / total_customers * 100) if total_customers > 0 else 0
        
        if vip_percentage < 10:
            html += f"""
                <div class="recommendation">
                    <div class="recommendation-title">🎯 Oportunidade VIP</div>
                    <div>Apenas {vip_percentage:.1f}% dos clientes são VIP. Considere criar um programa de fidelidade para aumentar o valor dos clientes leais.</div>
                </div>
            """
        
        if lost_percentage > 30:
            html += f"""
                <div class="alert">
                    <div class="alert-title">⚠️ Alta Taxa de Perda</div>
                    <div>{lost_percentage:.1f}% dos clientes estão perdidos. Implemente campanhas de reativação urgentes.</div>
                </div>
            """
        
        active_percentage = (retention.get('active_30d', 0) / total_customers * 100) if total_customers > 0 else 0
        if active_percentage > 20:
            html += f"""
                <div class="insight">
                    <div class="insight-title">✅ Base Ativa Saudável</div>
                    <div>{active_percentage:.1f}% dos clientes compraram nos últimos 30 dias, indicando uma base ativa saudável.</div>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _format_product_performance_html(self, result: Dict) -> str:
        """Formatar performance de produtos em HTML."""
        abc = result.get('abc_analysis', {})
        rankings = result.get('product_rankings', {})
        alerts = result.get('inventory_alerts', {})
        
        html = f"""
        <!-- KPIs de Produtos -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-icon">📦</div>
                <div class="kpi-value">{abc.get('total_products', 0):,}</div>
                <div class="kpi-label">Total de Produtos</div>
                <div class="kpi-meta">Itens únicos no catálogo</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">🥇</div>
                <div class="kpi-value">{abc.get('classes', {}).get('A', 0)}</div>
                <div class="kpi-label">Produtos Classe A</div>
                <div class="kpi-meta">{abc.get('class_a_revenue_share', 0):.1f}% da receita</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">⚠️</div>
                <div class="kpi-value">{alerts.get('slow_movers_60d', 0):,}</div>
                <div class="kpi-label">Slow Movers</div>
                <div class="kpi-meta">Sem venda há 60+ dias</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-icon">💀</div>
                <div class="kpi-value">{alerts.get('dead_stock_120d', 0):,}</div>
                <div class="kpi-label">Dead Stock</div>
                <div class="kpi-meta">Sem venda há 120+ dias</div>
            </div>
        </div>
        
        <!-- Análise ABC -->
        <div class="section">
            <h2 class="section-title">🏆 Análise ABC de Produtos</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px;">
                <div style="background: linear-gradient(135deg, #fef3c7 0%, #fbbf24 100%); padding: 20px; border-radius: 15px; text-align: center;">
                    <div style="font-size: 2em; font-weight: bold; color: #92400e;">🥇</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: #92400e;">{abc.get('classes', {}).get('A', 0)}</div>
                    <div style="color: #92400e; font-weight: 600;">Classe A</div>
                    <div style="color: #92400e; font-size: 0.9em;">{abc.get('class_a_revenue_share', 0):.1f}% receita</div>
                </div>
                <div style="background: linear-gradient(135deg, #e5e7eb 0%, #9ca3af 100%); padding: 20px; border-radius: 15px; text-align: center;">
                    <div style="font-size: 2em; font-weight: bold; color: #374151;">🥈</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: #374151;">{abc.get('classes', {}).get('B', 0)}</div>
                    <div style="color: #374151; font-weight: 600;">Classe B</div>
                    <div style="color: #374151; font-size: 0.9em;">Produtos intermediários</div>
                </div>
                <div style="background: linear-gradient(135deg, #fecaca 0%, #ef4444 100%); padding: 20px; border-radius: 15px; text-align: center;">
                    <div style="font-size: 2em; font-weight: bold; color: #7f1d1d;">🥉</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: #7f1d1d;">{abc.get('classes', {}).get('C', 0)}</div>
                    <div style="color: #7f1d1d; font-weight: 600;">Classe C</div>
                    <div style="color: #7f1d1d; font-size: 0.9em;">Baixo giro</div>
                </div>
            </div>
        </div>
        
        <!-- Top Produtos por Receita -->
        <div class="section">
            <h2 class="section-title">💰 Top 10 Produtos por Receita</h2>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Posição</th>
                        <th>Código do Produto</th>
                        <th>Receita Total</th>
                        <th>Transações</th>
                        <th>Receita por Transação</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Adicionar top produtos
        top_revenue = rankings.get('top_by_revenue', {})
        for i, (product, data) in enumerate(list(top_revenue.items())[:10], 1):
            revenue = data.get('Total_Revenue', 0)
            transactions = data.get('Transaction_Count', 0)
            avg_per_transaction = revenue / transactions if transactions > 0 else 0
            
            html += f"""
                    <tr>
                        <td><strong>#{i}</strong></td>
                        <td><code>{product}</code></td>
                        <td><strong>R$ {revenue:,.2f}</strong></td>
                        <td>{transactions:,}</td>
                        <td>R$ {avg_per_transaction:,.2f}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        
        <!-- Top Produtos por Volume -->
        <div class="section">
            <h2 class="section-title">📊 Top 10 Produtos por Volume</h2>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Posição</th>
                        <th>Código do Produto</th>
                        <th>Quantidade Total</th>
                        <th>Receita Total</th>
                        <th>Preço Médio</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Adicionar top produtos por volume
        top_volume = rankings.get('top_by_volume', {})
        for i, (product, data) in enumerate(list(top_volume.items())[:10], 1):
            quantity = data.get('Total_Quantity', 0)
            revenue = data.get('Total_Revenue', 0)
            avg_price = revenue / quantity if quantity > 0 else 0
            
            html += f"""
                    <tr>
                        <td><strong>#{i}</strong></td>
                        <td><code>{product}</code></td>
                        <td><strong>{quantity:,}</strong></td>
                        <td>R$ {revenue:,.2f}</td>
                        <td>R$ {avg_price:,.2f}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        # Alertas de inventário se existirem
        if alerts.get('slow_movers_60d', 0) > 0 or alerts.get('dead_stock_120d', 0) > 0:
            html += f"""
            <div class="section">
                <h2 class="section-title">⚠️ Alertas de Inventário</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                    <div class="alert">
                        <div class="alert-title">📉 Slow Movers (60+ dias)</div>
                        <div><strong>{alerts.get('slow_movers_60d', 0):,} produtos</strong> sem vendas há mais de 60 dias</div>
                        <div style="margin-top: 10px; font-size: 0.9em;">Recomendação: Revisar preços ou promoções</div>
                    </div>
                    <div class="alert">
                        <div class="alert-title">💀 Dead Stock (120+ dias)</div>
                        <div><strong>{alerts.get('dead_stock_120d', 0):,} produtos</strong> sem vendas há mais de 120 dias</div>
                        <div style="margin-top: 10px; font-size: 0.9em;">Recomendação: Considerar liquidação</div>
                    </div>
                </div>
            </div>
            """
        
        return html 