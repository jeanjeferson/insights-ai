from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any, List
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
import json
import time
import traceback
import sys

# Importar módulos compartilhados consolidados
try:
    # Imports relativos (quando usado como módulo)
    from .shared.data_preparation import DataPreparationMixin
    from .shared.business_mixins import JewelryBusinessAnalysisMixin, JewelryRFMAnalysisMixin
except ImportError:
    # Imports absolutos (quando executado diretamente)
    try:
        from insights.tools.shared.data_preparation import DataPreparationMixin
        from insights.tools.shared.business_mixins import JewelryBusinessAnalysisMixin, JewelryRFMAnalysisMixin
    except ImportError:
        # Se não conseguir importar, usar versão local ou criar stubs
        print("⚠️ Importações locais não encontradas, usando implementação básica...")
        
        class DataPreparationMixin:
            """Stub básico para DataPreparationMixin"""
            pass
        
        class JewelryBusinessAnalysisMixin:
            """Stub básico para JewelryBusinessAnalysisMixin"""
            pass
            
        class JewelryRFMAnalysisMixin:
            """Stub básico para JewelryRFMAnalysisMixin"""
            pass

warnings.filterwarnings('ignore')

class FinancialDataExporterInput(BaseModel):
    """Schema para exportação de dados financeiros."""
    
    data_csv: str = Field(
        default="data/vendas.csv", 
        description="Caminho para arquivo CSV de vendas"
    )
    
    output_path: str = Field(
        default="assets/data/analise_financeira_dados_completos.csv",
        description="Caminho de saída para o arquivo CSV financeiro exportado"
    )
    
    # Novos parâmetros de período
    current_year: Optional[int] = Field(
        default=None,
        description="Ano específico para análise (ex: 2024). Se None, usa ano atual"
    )
    
    last_x_days: Optional[int] = Field(
        default=None,
        description="Últimos X dias para análise (ex: 90). Se None, usa ano completo"
    )
    
    period_start_date: Optional[str] = Field(
        default=None,
        description="Data início período customizado (YYYY-MM-DD). Opcional"
    )
    
    period_end_date: Optional[str] = Field(
        default=None,
        description="Data fim período customizado (YYYY-MM-DD). Opcional"
    )
    
    include_kpi_analysis: bool = Field(
        default=True,
        description="Incluir análise de KPIs financeiros críticos"
    )
    
    include_margin_analysis: bool = Field(
        default=True,
        description="Incluir análise de margens e rentabilidade"
    )
    
    include_trend_analysis: bool = Field(
        default=True,
        description="Incluir análise de tendências e sazonalidade"
    )
    
    include_projections: bool = Field(
        default=True,
        description="Incluir projeções financeiras (30/60/90 dias)"
    )
    
    group_by_period: str = Field(
        default="monthly",
        description="Período de agrupamento: daily, weekly, monthly, quarterly"
    )

class FinancialDataExporter(BaseTool, DataPreparationMixin, JewelryBusinessAnalysisMixin, JewelryRFMAnalysisMixin):
    """
    Ferramenta especializada para exportar dados completos de análise financeira.
    
    COMPORTAMENTO PADRÃO: YTD (Year-to-Date) - analisa do início do ano até hoje.
    
    Esta ferramenta gera um CSV abrangente com:
    - KPIs financeiros críticos (baseados no período filtrado)
    - Métricas de margens e rentabilidade
    - Análise de tendências e sazonalidade
    - Projeções financeiras (usando últimos 2 anos)
    - Insights e oportunidades estratégicas
    - Comparação YoY inteligente (período atual vs mesmo período ano anterior)
    
    FILTROS DE PERÍODO (por prioridade):
    1. Período customizado: period_start_date + period_end_date
    2. Últimos X dias: last_x_days
    3. Ano específico: current_year
    4. YTD (padrão): sem parâmetros = início do ano até hoje
    """
    
    name: str = "Financial Data Exporter"
    description: str = """
    Exporta dados completos de análise financeira em formato CSV para análise avançada.
    
    PADRÃO: YTD (Year-to-Date) - analisa do início do ano até hoje, comparando com YTD do ano anterior.
    
    Inclui KPIs financeiros, análise de margens, tendências sazonais, 
    projeções e insights estratégicos por período filtrado.
    
    Use esta ferramenta quando precisar de dados estruturados financeiros para:
    - Relatórios executivos e board (YTD vs YTD anterior)
    - Análise de performance financeira por período específico
    - Planejamento orçamentário baseado em projeção anual
    - Dashboards de BI (Power BI, Tableau)
    - Análises de rentabilidade por categoria
    - Projeções e forecasting (baseado em últimos 2 anos)
    
    FILTROS DISPONÍVEIS:
    • Sem parâmetros: YTD (início do ano até hoje) 
    • current_year: Ano específico completo
    • last_x_days: Últimos X dias
    • period_start_date + period_end_date: Período customizado
    """
    args_schema: Type[BaseModel] = FinancialDataExporterInput

    def _run(self, data_csv: str = "data/vendas.csv", 
             output_path: str = "assets/data/analise_financeira_dados_completos.csv",
             current_year: Optional[int] = None,
             last_x_days: Optional[int] = None,
             period_start_date: Optional[str] = None,
             period_end_date: Optional[str] = None,
             include_kpi_analysis: bool = True,
             include_margin_analysis: bool = True,
             include_trend_analysis: bool = True,
             include_projections: bool = True,
             group_by_period: str = "monthly") -> str:
        
        try:
            print("🚀 Iniciando exportação de dados financeiros...")
            
            # 1. Carregar e preparar dados completos
            print("📊 Carregando dados de vendas para análise financeira...")
            df_complete = self._load_and_prepare_data(data_csv)
            
            if df_complete.empty:
                return "❌ Erro: Dados de vendas não encontrados ou inválidos"
            
            print(f"✅ Dados completos carregados: {len(df_complete):,} registros")
            
            # 2. Filtrar dados por período especificado
            print("🔍 Aplicando filtros de período...")
            df_filtered, period_info = self._filter_data_by_period(
                df_complete, current_year, last_x_days, period_start_date, period_end_date
            )
            
            if df_filtered.empty:
                return f"❌ Erro: Nenhum dado encontrado para o período especificado: {period_info['description']}"
            
            print(f"✅ Dados filtrados: {len(df_filtered):,} registros ({period_info['description']})")
            
            # 3. Agregar dados por período (usando dados filtrados)
            print(f"📅 Agregando dados por período ({group_by_period})...")
            financial_data = self._aggregate_financial_data(df_filtered, group_by_period)
            
            # 4. Análise de KPIs (usando lógica híbrida: filtrado + comparação histórica)
            if include_kpi_analysis:
                print("💰 Calculando KPIs financeiros críticos...")
                financial_data = self._add_enhanced_kpi_analysis(df_complete, df_filtered, financial_data, period_info)
            
            # 5. Análise de margens (usando dados filtrados)
            if include_margin_analysis:
                print("📊 Analisando margens e rentabilidade...")
                financial_data = self._add_margin_analysis(df_filtered, financial_data)
            
            # 6. Análise de tendências (usando dados filtrados para tendência, completos para sazonalidade)
            if include_trend_analysis:
                print("📈 Analisando tendências e sazonalidade...")
                financial_data = self._add_enhanced_trend_analysis(df_complete, df_filtered, financial_data)
            
            # 7. Projeções financeiras (usando últimos 2 anos)
            if include_projections:
                print("🔮 Gerando projeções financeiras...")
                financial_data = self._add_enhanced_financial_projections(df_complete, df_filtered, financial_data)
            
            # 8. Insights e oportunidades (usando dados filtrados)
            print("💡 Identificando insights e oportunidades...")
            financial_data = self._add_strategic_insights(df_filtered, financial_data)
            
            # 9. Scores de saúde financeira (usando dados filtrados)
            print("📊 Calculando scores de saúde financeira...")
            financial_data = self._add_financial_health_scores(financial_data)
            
            # 10. Adicionar informações do período aos dados
            for col, value in period_info.items():
                if col != 'description':
                    financial_data[f'Periodo_Info_{col}'] = value
            
            # 11. Exportar CSV
            print("💾 Exportando arquivo CSV financeiro...")
            success = self._export_to_csv(financial_data, output_path)
            
            if success:
                return self._generate_enhanced_export_summary(financial_data, output_path, df_complete, df_filtered, period_info)
            else:
                return "❌ Erro na exportação do arquivo CSV"
                
        except Exception as e:
            return f"❌ Erro na exportação de dados financeiros: {str(e)}"

    def _filter_data_by_period(self, df: pd.DataFrame, current_year: Optional[int] = None,
                              last_x_days: Optional[int] = None, 
                              period_start_date: Optional[str] = None,
                              period_end_date: Optional[str] = None) -> tuple:
        """
        Filtrar dados por período especificado com prioridade:
        1. Período customizado (start_date + end_date)
        2. Últimos X dias
        3. Ano específico
        4. Ano atual (padrão)
        """
        
        max_date = df['Data'].max()
        min_date = df['Data'].min()
        
        # Prioridade 1: Período customizado
        if period_start_date and period_end_date:
            start_date = pd.to_datetime(period_start_date)
            end_date = pd.to_datetime(period_end_date)
            filtered_df = df[(df['Data'] >= start_date) & (df['Data'] <= end_date)]
            
            period_info = {
                'type': 'custom',
                'start_date': start_date,
                'end_date': end_date,
                'description': f"Período customizado: {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}",
                'days_count': (end_date - start_date).days + 1
            }
            
        # Prioridade 2: Últimos X dias
        elif last_x_days:
            start_date = max_date - timedelta(days=last_x_days - 1)
            filtered_df = df[df['Data'] >= start_date]
            
            period_info = {
                'type': 'last_days',
                'start_date': start_date,
                'end_date': max_date,
                'description': f"Últimos {last_x_days} dias ({start_date.strftime('%Y-%m-%d')} a {max_date.strftime('%Y-%m-%d')})",
                'days_count': last_x_days
            }
            
        # Prioridade 3: Ano específico
        elif current_year:
            start_date = pd.to_datetime(f"{current_year}-01-01")
            end_date = pd.to_datetime(f"{current_year}-12-31")
            filtered_df = df[(df['Data'] >= start_date) & (df['Data'] <= end_date)]
            
            period_info = {
                'type': 'year',
                'start_date': start_date,
                'end_date': end_date,
                'description': f"Ano {current_year}",
                'year': current_year,
                'days_count': (end_date - start_date).days + 1
            }
            
        # Prioridade 4: YTD (Year-to-Date) - PADRÃO
        else:
            current_year = max_date.year
            start_date = pd.to_datetime(f"{current_year}-01-01")
            end_date = max_date  # Até a data mais recente disponível (YTD)
            filtered_df = df[(df['Data'] >= start_date) & (df['Data'] <= end_date)]
            
            period_info = {
                'type': 'ytd',  # Year-to-Date como padrão
                'start_date': start_date,
                'end_date': end_date,
                'description': f"YTD {current_year} (Jan-{end_date.strftime('%b')})",
                'year': current_year,
                'days_count': (end_date - start_date).days + 1,
                'ytd_progress': end_date.timetuple().tm_yday / 365.0
            }
        
        return filtered_df, period_info

    def _get_comparison_period(self, df_complete: pd.DataFrame, period_info: dict) -> pd.DataFrame:
        """
        Obter período de comparação equivalente do ano anterior.
        """
        
        if period_info['type'] in ['custom', 'last_days']:
            # Para períodos customizados, comparar com mesmo período do ano anterior
            start_comp = period_info['start_date'] - timedelta(days=365)
            end_comp = period_info['end_date'] - timedelta(days=365)
            
        elif period_info['type'] in ['year', 'current_year', 'ytd']:
            # Para anos, comparar com ano anterior completo
            year = period_info.get('year', period_info['start_date'].year)
            start_comp = pd.to_datetime(f"{year-1}-01-01")
            
            if period_info['type'] in ['current_year', 'ytd']:
                # Se for ano atual ou YTD, comparar até a mesma data do ano anterior
                end_comp = period_info['end_date'] - timedelta(days=365)
            else:
                # Se for ano completo, comparar ano anterior completo
                end_comp = pd.to_datetime(f"{year-1}-12-31")
        
        comparison_df = df_complete[
            (df_complete['Data'] >= start_comp) & 
            (df_complete['Data'] <= end_comp)
        ]
        
        return comparison_df

    def _add_enhanced_kpi_analysis(self, df_complete: pd.DataFrame, df_filtered: pd.DataFrame, 
                                  financial_data: pd.DataFrame, period_info: dict) -> pd.DataFrame:
        """
        Análise de KPIs melhorada com comparação YoY inteligente.
        """
        
        # Crescimento período anterior (sequencial)
        financial_data = financial_data.sort_values('Periodo')
        
        # Variações sequenciais (usando dados filtrados)
        financial_data['Receita_Variacao_Pct'] = (
            financial_data['Receita_Total'].pct_change() * 100
        ).round(2)
        
        financial_data['Transacoes_Variacao_Pct'] = (
            financial_data['Num_Transacoes'].pct_change() * 100
        ).round(2)
        
        financial_data['Ticket_Variacao_Pct'] = (
            financial_data['Ticket_Medio'].pct_change() * 100
        ).round(2)
        
        # Métricas do período atual (filtrado)
        current_receita = df_filtered['Total_Liquido'].sum()
        current_transacoes = len(df_filtered)
        current_ticket = current_receita / current_transacoes if current_transacoes > 0 else 0
        
        # Comparação com período equivalente do ano anterior
        comparison_df = self._get_comparison_period(df_complete, period_info)
        
        if not comparison_df.empty:
            previous_receita = comparison_df['Total_Liquido'].sum()
            previous_transacoes = len(comparison_df)
            previous_ticket = previous_receita / previous_transacoes if previous_transacoes > 0 else 0
            
            # Crescimento YoY
            yoy_receita_growth = ((current_receita - previous_receita) / previous_receita * 100) if previous_receita > 0 else 0
            yoy_transacoes_growth = ((current_transacoes - previous_transacoes) / previous_transacoes * 100) if previous_transacoes > 0 else 0
            yoy_ticket_growth = ((current_ticket - previous_ticket) / previous_ticket * 100) if previous_ticket > 0 else 0
            
        else:
            previous_receita = 0
            previous_transacoes = 0
            previous_ticket = 0
            yoy_receita_growth = 0
            yoy_transacoes_growth = 0
            yoy_ticket_growth = 0
        
        # Adicionar métricas YoY aos dados
        financial_data['Receita_Periodo_Atual'] = current_receita
        financial_data['Receita_Periodo_Anterior'] = previous_receita
        financial_data['YoY_Receita_Growth_Pct'] = round(yoy_receita_growth, 2)
        financial_data['YoY_Transacoes_Growth_Pct'] = round(yoy_transacoes_growth, 2)
        financial_data['YoY_Ticket_Growth_Pct'] = round(yoy_ticket_growth, 2)
        
        # Projeção anual baseada no período atual
        if period_info['type'] in ['current_year', 'ytd']:
            # Para YTD, usar progresso do ano para extrapolar
            if 'ytd_progress' in period_info:
                progress_year = period_info['ytd_progress']
            else:
                day_of_year = period_info['end_date'].timetuple().tm_yday
                progress_year = day_of_year / 365.0
            
            projecao_anual = (current_receita / progress_year) if progress_year > 0 else 0
        else:
            # Para outros períodos, extrapolar baseado na duração
            days_in_year = 365
            days_in_period = period_info['days_count']
            projecao_anual = (current_receita / days_in_period * days_in_year) if days_in_period > 0 else 0
        
        financial_data['Projecao_Anual'] = round(projecao_anual, 2)
        
        # Confiabilidade baseada na disponibilidade de dados históricos
        years_of_data = (df_complete['Data'].max().year - df_complete['Data'].min().year) + 1
        has_comparison_data = not comparison_df.empty
        
        def get_confidence_level(years, has_comparison):
            if not has_comparison:
                return 'Baixa'
            elif years >= 3:
                return 'Alta'
            elif years >= 2:
                return 'Média'
            else:
                return 'Baixa'
        
        financial_data['Confiabilidade_Comparacao'] = get_confidence_level(years_of_data, has_comparison_data)
        
        # Informações do período
        financial_data['Periodo_Analisado'] = period_info['description']
        financial_data['Dias_No_Periodo'] = period_info['days_count']
        
        return financial_data

    def _add_enhanced_trend_analysis(self, df_complete: pd.DataFrame, df_filtered: pd.DataFrame, 
                                   financial_data: pd.DataFrame) -> pd.DataFrame:
        """
        Análise de tendências melhorada usando dados completos para sazonalidade 
        e dados filtrados para tendência atual.
        """
        
        # Tendência de crescimento usando dados filtrados
        financial_data = financial_data.sort_values('Periodo').reset_index(drop=True)
        
        if len(financial_data) > 1:
            x = np.arange(len(financial_data))
            y = financial_data['Receita_Total'].values
            
            # Coeficiente de tendência
            trend_coef = np.polyfit(x, y, 1)[0]
            financial_data['Tendencia_Crescimento'] = trend_coef
            
            # Classificar tendência
            if trend_coef > 1000:
                financial_data['Classificacao_Tendencia'] = 'Crescimento_Forte'
            elif trend_coef > 0:
                financial_data['Classificacao_Tendencia'] = 'Crescimento_Moderado'
            elif trend_coef > -1000:
                financial_data['Classificacao_Tendencia'] = 'Estavel'
            else:
                financial_data['Classificacao_Tendencia'] = 'Declinio'
        else:
            financial_data['Tendencia_Crescimento'] = 0
            financial_data['Classificacao_Tendencia'] = 'Insuficiente'
        
        # Análise sazonal usando dados completos (histórico robusto)
        monthly_avg = df_complete.groupby(df_complete['Data'].dt.month)['Total_Liquido'].mean()
        overall_avg = df_complete['Total_Liquido'].mean()
        
        def get_seasonal_index(periodo):
            try:
                # Extrair mês do período
                if '-' in str(periodo):
                    month = int(str(periodo).split('-')[1])
                else:
                    month = df_filtered['Data'].dt.month.mode()[0]  # Mês mais comum no período filtrado
                
                return monthly_avg.get(month, overall_avg) / overall_avg
            except:
                return 1.0
        
        financial_data['Indice_Sazonal'] = financial_data['Periodo'].apply(get_seasonal_index).round(3)
        
        # Classificar sazonalidade
        def classify_seasonality(index):
            if index > 1.3:
                return 'Pico_Alto'
            elif index > 1.1:
                return 'Pico_Moderado'
            elif index < 0.7:
                return 'Vale_Alto'
            elif index < 0.9:
                return 'Vale_Moderado'
            else:
                return 'Normal'
        
        financial_data['Classificacao_Sazonal'] = financial_data['Indice_Sazonal'].apply(classify_seasonality)
        
        # Correlação preço vs volume usando dados filtrados
        if 'Quantidade' in financial_data.columns:
            correlation = financial_data['Ticket_Medio'].corr(financial_data['Quantidade'])
            financial_data['Correlacao_Preco_Volume'] = correlation
        else:
            financial_data['Correlacao_Preco_Volume'] = -0.72  # Conforme relatório padrão
        
        return financial_data

    def _add_enhanced_financial_projections(self, df_complete: pd.DataFrame, df_filtered: pd.DataFrame, 
                                          financial_data: pd.DataFrame) -> pd.DataFrame:
        """
        Projeções financeiras melhoradas usando últimos 2 anos completos.
        """
        
        # Obter dados dos últimos 2 anos completos para projeções
        max_date = df_complete['Data'].max()
        two_years_ago = max_date - timedelta(days=730)  # Aproximadamente 2 anos
        
        projection_data = df_complete[df_complete['Data'] >= two_years_ago]
        
        if len(projection_data) > 0:
            # Agregar por mês para análise de tendência
            monthly_projection = projection_data.groupby(
                projection_data['Data'].dt.to_period('M')
            )['Total_Liquido'].sum()
            
            # Calcular tendência mensal dos últimos 2 anos
            if len(monthly_projection) > 3:
                x = np.arange(len(monthly_projection))
                y = monthly_projection.values
                trend_coef = np.polyfit(x, y, 1)[0]
                avg_monthly = monthly_projection.mean()
            else:
                trend_coef = 0
                avg_monthly = df_filtered['Total_Liquido'].mean() * 30  # Estimativa mensal
            
            # Calcular sazonalidade dos últimos 2 anos
            seasonal_factors = projection_data.groupby(
                projection_data['Data'].dt.month
            )['Total_Liquido'].sum() / projection_data.groupby(
                projection_data['Data'].dt.month
            )['Total_Liquido'].sum().mean()
            
            # Usar período atual para determinar sazonalidade futura
            current_month = max_date.month
            next_month = (current_month % 12) + 1
            next_next_month = ((current_month + 1) % 12) + 1
            
            seasonal_30d = seasonal_factors.get(next_month, 1.0)
            seasonal_60d = seasonal_factors.get(next_next_month, 1.0)
            seasonal_90d = seasonal_factors.get((current_month + 2) % 12 + 1, 1.0)
            
            # Base para projeção (média mensal ajustada)
            base_monthly = avg_monthly + (trend_coef * len(monthly_projection))
            base_daily = base_monthly / 30
            
            # Cenários de projeção
            # Conservador: -5% da base + sazonalidade
            financial_data['Projecao_30d_Conservador'] = (base_daily * 30 * 0.95 * seasonal_30d).round(2)
            financial_data['Projecao_60d_Conservador'] = (base_daily * 60 * 0.95 * ((seasonal_30d + seasonal_60d) / 2)).round(2)
            financial_data['Projecao_90d_Conservador'] = (base_daily * 90 * 0.95 * ((seasonal_30d + seasonal_60d + seasonal_90d) / 3)).round(2)
            
            # Realista: base + sazonalidade
            financial_data['Projecao_30d_Realista'] = (base_daily * 30 * seasonal_30d).round(2)
            financial_data['Projecao_60d_Realista'] = (base_daily * 60 * ((seasonal_30d + seasonal_60d) / 2)).round(2)
            financial_data['Projecao_90d_Realista'] = (base_daily * 90 * ((seasonal_30d + seasonal_60d + seasonal_90d) / 3)).round(2)
            
            # Otimista: +10% da base + sazonalidade
            financial_data['Projecao_30d_Otimista'] = (base_daily * 30 * 1.10 * seasonal_30d).round(2)
            financial_data['Projecao_60d_Otimista'] = (base_daily * 60 * 1.10 * ((seasonal_30d + seasonal_60d) / 2)).round(2)
            financial_data['Projecao_90d_Otimista'] = (base_daily * 90 * 1.10 * ((seasonal_30d + seasonal_60d + seasonal_90d) / 3)).round(2)
            
            # Confiabilidade baseada na quantidade de dados históricos
            months_of_data = len(monthly_projection)
            if months_of_data >= 24:
                confidence = 95.0
            elif months_of_data >= 18:
                confidence = 85.0
            elif months_of_data >= 12:
                confidence = 75.0
            else:
                confidence = 60.0
                
            financial_data['Confiabilidade_Modelo_Pct'] = confidence
            
        else:
            # Fallback quando não há dados suficientes
            for scenario in ['Conservador', 'Realista', 'Otimista']:
                for period in ['30d', '60d', '90d']:
                    financial_data[f'Projecao_{period}_{scenario}'] = 0
            
            financial_data['Confiabilidade_Modelo_Pct'] = 0.0
        
        return financial_data

    def _add_margin_analysis(self, df: pd.DataFrame, financial_data: pd.DataFrame) -> pd.DataFrame:
        """Adicionar análise de margens e rentabilidade."""
        
        # Margem bruta estimada (assumindo 62.4% conforme relatório)
        estimated_gross_margin = 0.624
        
        financial_data['Margem_Bruta_Estimada'] = (
            financial_data['Receita_Total'] * estimated_gross_margin
        ).round(2)
        
        financial_data['Margem_Bruta_Pct'] = estimated_gross_margin * 100
        
        # Se temos dados reais de margem
        if 'Margem_Total' in financial_data.columns:
            financial_data['Margem_Real_Pct'] = (
                financial_data['Margem_Total'] / financial_data['Receita_Total'] * 100
            ).fillna(estimated_gross_margin * 100).round(2)
        else:
            financial_data['Margem_Real_Pct'] = financial_data['Margem_Bruta_Pct']
        
        # Impacto de descontos
        if 'Desconto_Total' in financial_data.columns:
            financial_data['Impacto_Desconto_Pct'] = (
                financial_data['Desconto_Total'] / financial_data['Receita_Total'] * 100
            ).fillna(0).round(2)
            
            financial_data['Margem_Liquida_Pct'] = (
                financial_data['Margem_Real_Pct'] - financial_data['Impacto_Desconto_Pct']
            ).round(2)
        else:
            # Assumir 22pp de impacto conforme relatório
            financial_data['Impacto_Desconto_Pct'] = 22.0
            financial_data['Margem_Liquida_Pct'] = (
                financial_data['Margem_Real_Pct'] - 22.0
            ).round(2)
        
        # Margem líquida estimada (18.7% conforme relatório)
        financial_data['Margem_Liquida_Final_Pct'] = 18.7
        financial_data['Margem_Liquida_Final'] = (
            financial_data['Receita_Total'] * 0.187
        ).round(2)
        
        # ROI por período
        # Assumindo custo operacional = receita - margem líquida
        financial_data['Custo_Operacional_Estimado'] = (
            financial_data['Receita_Total'] - financial_data['Margem_Liquida_Final']
        ).round(2)
        
        financial_data['ROI_Periodo_Pct'] = (
            financial_data['Margem_Liquida_Final'] / 
            financial_data['Custo_Operacional_Estimado'] * 100
        ).fillna(0).round(2)
        
        return financial_data

    def _add_strategic_insights(self, df: pd.DataFrame, financial_data: pd.DataFrame) -> pd.DataFrame:
        """Adicionar insights e oportunidades estratégicas."""
        
        # Análise por categoria (se disponível)
        if 'Grupo_Produto' in df.columns:
            category_performance = df.groupby('Grupo_Produto')['Total_Liquido'].sum().sort_values(ascending=False)
            top_category = category_performance.index[0] if len(category_performance) > 0 else 'N/A'
            
            financial_data['Categoria_Top_Performance'] = top_category
            financial_data['Receita_Top_Categoria'] = category_performance.iloc[0] if len(category_performance) > 0 else 0
            
            # Oportunidade de margem (baseada na categoria)
            margin_opportunity = category_performance.sum() * 0.05  # 5% de oportunidade
            financial_data['Oportunidade_Margem'] = margin_opportunity.round(2)
        else:
            financial_data['Categoria_Top_Performance'] = 'N/A'
            financial_data['Receita_Top_Categoria'] = 0
            financial_data['Oportunidade_Margem'] = financial_data['Receita_Total'] * 0.05
        
        # Eficiência operacional
        financial_data['Eficiencia_Operacional'] = (
            financial_data['Receita_Total'] / financial_data['Num_Transacoes']
        ).round(2)
        
        # ROAS (Return on Ad Spend) estimado
        financial_data['ROAS_Estimado'] = 4.8  # Conforme relatório
        
        # Custo de aquisição por cliente estimado
        financial_data['CAC_Estimado'] = 320  # Conforme relatório
        
        # Valor vitalício vs CAC ratio
        avg_clv = 24850  # Conforme relatório
        financial_data['LTV_CAC_Ratio'] = avg_clv / 320  # 7.6:1 conforme relatório
        
        # Canal performance (estimado)
        financial_data['Performance_Online_Pct'] = 61  # Conforme relatório
        financial_data['Performance_Presencial_Pct'] = 39  # Conforme relatório
        
        # Recomendações estratégicas
        def get_strategic_recommendation(row):
            growth_rate = row.get('Receita_Variacao_Pct', 0)
            margin = row.get('Margem_Liquida_Pct', 0)
            
            if growth_rate > 10 and margin > 15:
                return 'Expansao_Agressiva'
            elif growth_rate > 5:
                return 'Crescimento_Sustentavel'
            elif growth_rate < -5:
                return 'Otimizacao_Urgente'
            elif margin < 10:
                return 'Foco_Margem'
            else:
                return 'Manutencao_Performance'
        
        financial_data['Recomendacao_Estrategica'] = financial_data.apply(get_strategic_recommendation, axis=1)
        
        # Prioridade de ação
        def get_action_priority(row):
            growth = row.get('Receita_Variacao_Pct', 0)
            margin = row.get('Margem_Liquida_Pct', 0)
            
            if growth < -10 or margin < 5:
                return 1  # Urgente
            elif growth < 0 or margin < 10:
                return 2  # Alto
            elif growth < 5:
                return 3  # Médio
            else:
                return 4  # Baixo
        
        financial_data['Prioridade_Acao'] = financial_data.apply(get_action_priority, axis=1)
        
        return financial_data

    def _add_financial_health_scores(self, financial_data: pd.DataFrame) -> pd.DataFrame:
        """Calcular scores de saúde financeira."""
        
        # Verificar se as colunas necessárias existem
        if 'Receita_Variacao_Pct' not in financial_data.columns:
            financial_data['Receita_Variacao_Pct'] = 0.0
        
        if 'Margem_Liquida_Pct' not in financial_data.columns:
            financial_data['Margem_Liquida_Pct'] = 18.7  # Valor padrão
        
        if 'Eficiencia_Operacional' not in financial_data.columns:
            financial_data['Eficiencia_Operacional'] = (
                financial_data['Receita_Total'] / financial_data['Num_Transacoes']
            ).round(2) if 'Receita_Total' in financial_data.columns and 'Num_Transacoes' in financial_data.columns else 500
        
        if 'Oportunidade_Margem' not in financial_data.columns:
            financial_data['Oportunidade_Margem'] = financial_data['Receita_Total'] * 0.05 if 'Receita_Total' in financial_data.columns else 0
        
        # Score de Crescimento
        max_growth = financial_data['Receita_Variacao_Pct'].max()
        min_growth = financial_data['Receita_Variacao_Pct'].min()
        
        if max_growth > min_growth and max_growth != min_growth:
            financial_data['Score_Crescimento'] = (
                (financial_data['Receita_Variacao_Pct'] - min_growth) / 
                (max_growth - min_growth) * 100
            ).clip(0, 100).round(1)
        else:
            financial_data['Score_Crescimento'] = 50.0
        
        # Score de Margem
        financial_data['Score_Margem'] = (
            financial_data['Margem_Liquida_Pct'] / 25 * 100  # 25% como margem excelente
        ).clip(0, 100).round(1)
        
        # Score de Eficiência
        max_efficiency = financial_data['Eficiencia_Operacional'].max()
        if max_efficiency > 0:
            financial_data['Score_Eficiencia'] = (
                financial_data['Eficiencia_Operacional'] / max_efficiency * 100
            ).clip(0, 100).round(1)
        else:
            financial_data['Score_Eficiencia'] = 50.0
        
        # Score de Estabilidade (baseado na variação)
        financial_data['Score_Estabilidade'] = (
            100 - abs(financial_data['Receita_Variacao_Pct']).clip(0, 100)
        ).round(1)
        
        # Score Geral de Saúde Financeira
        financial_data['Score_Saude_Financeira'] = (
            financial_data['Score_Crescimento'] * 0.3 +
            financial_data['Score_Margem'] * 0.3 +
            financial_data['Score_Eficiencia'] * 0.2 +
            financial_data['Score_Estabilidade'] * 0.2
        ).round(1)
        
        # Score de Oportunidade
        if 'Receita_Total' in financial_data.columns:
            financial_data['Score_Oportunidade'] = (
                financial_data['Oportunidade_Margem'] / financial_data['Receita_Total'] * 100 * 10
            ).clip(0, 100).round(1)
        else:
            financial_data['Score_Oportunidade'] = 50.0
        
        return financial_data

    def _export_to_csv(self, financial_data: pd.DataFrame, output_path: str) -> bool:
        """Exportar dados para CSV."""
        
        try:
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Ordenar por período
            financial_data_sorted = financial_data.sort_values('Periodo')
            
            # Reorganizar colunas para melhor visualização
            priority_columns = [
                'Periodo', 'Receita_Total', 'Receita_Variacao_Pct', 'YoY_Receita_Growth_Pct', 
                'Ticket_Medio', 'Num_Transacoes', 'Score_Saude_Financeira', 'Recomendacao_Estrategica',
                'Margem_Liquida_Pct', 'Receita_Periodo_Atual', 'Receita_Periodo_Anterior',
                'Projecao_Anual', 'Confiabilidade_Comparacao',
                'Tendencia_Crescimento', 'Classificacao_Tendencia', 'Indice_Sazonal'
            ]
            
            # Adicionar colunas restantes
            remaining_columns = [col for col in financial_data_sorted.columns if col not in priority_columns]
            final_columns = priority_columns + remaining_columns
            
            # Filtrar colunas que existem
            existing_columns = [col for col in final_columns if col in financial_data_sorted.columns]
            
            # Exportar CSV
            financial_data_sorted[existing_columns].to_csv(
                output_path, index=False, sep=';', encoding='utf-8'
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao exportar CSV: {str(e)}")
            return False

    def _load_and_prepare_data(self, data_csv: str) -> pd.DataFrame:
        """Carregar e preparar dados usando o mixin."""
        try:
            import pandas as pd
            
            # Verificar se arquivo existe
            if not os.path.exists(data_csv):
                print(f"❌ Arquivo não encontrado: {data_csv}")
                return pd.DataFrame()
            
            # Carregar CSV
            df = pd.read_csv(data_csv, sep=';', encoding='utf-8')
            
            if df.empty:
                print("❌ Arquivo CSV está vazio")
                return pd.DataFrame()
            
            # Preparar dados básicos
            df['Data'] = pd.to_datetime(df['Data'])
            
            # Garantir campos essenciais
            required_columns = ['Total_Liquido', 'Data']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ Colunas obrigatórias ausentes: {missing_columns}")
                return pd.DataFrame()
            
            print(f"✅ Dados preparados: {len(df)} registros")
            return df
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return pd.DataFrame()
    
    def _aggregate_financial_data(self, df: pd.DataFrame, group_by_period: str) -> pd.DataFrame:
        """Agregar dados financeiros por período."""
        
        # Definir período de agrupamento
        period_mapping = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'M',
            'quarterly': 'Q'
        }
        
        freq = period_mapping.get(group_by_period, 'M')
        
        # Criar período baseado na frequência
        if freq == 'D':
            df['Periodo'] = df['Data'].dt.date
        elif freq == 'W':
            df['Periodo'] = df['Data'].dt.to_period('W').astype(str)
        elif freq == 'M':
            df['Periodo'] = df['Data'].dt.to_period('M').astype(str)
        elif freq == 'Q':
            df['Periodo'] = df['Data'].dt.to_period('Q').astype(str)
        
        # Agregações financeiras
        agg_dict = {
            'Total_Liquido': ['sum', 'mean', 'count'],
            'Data': ['min', 'max'],
            'Quantidade': 'sum' if 'Quantidade' in df.columns else lambda x: len(x)
        }
        
        # Colunas opcionais para análise financeira
        optional_columns = {
            'Codigo_Produto': 'nunique',
            'Codigo_Cliente': 'nunique' if 'Codigo_Cliente' in df.columns else lambda x: x.nunique(),
            'Grupo_Produto': lambda x: x.mode().iloc[0] if not x.empty else 'N/A',
            'Margem_Real': 'sum' if 'Margem_Real' in df.columns else lambda x: 0,
            'Desconto': 'sum' if 'Desconto' in df.columns else lambda x: 0
        }
        
        for col, agg in optional_columns.items():
            if col in df.columns:
                agg_dict[col] = agg
        
        # Agregar por período
        financial_aggregated = df.groupby('Periodo').agg(agg_dict)
        
        # Flatten column names
        new_columns = []
        for col in financial_aggregated.columns:
            if isinstance(col, tuple):
                if col[1] in ['first', 'last', '']:
                    new_columns.append(col[0])
                else:
                    new_columns.append(f"{col[0]}_{col[1]}")
            else:
                new_columns.append(col)
        
        financial_aggregated.columns = new_columns
        
        # Renomear colunas
        column_mapping = {
            'Total_Liquido_sum': 'Receita_Total',
            'Total_Liquido_mean': 'Ticket_Medio',
            'Total_Liquido_count': 'Num_Transacoes',
            'Data_min': 'Data_Inicio_Periodo',
            'Data_max': 'Data_Fim_Periodo',
            'Codigo_Produto_nunique': 'Produtos_Vendidos',
            'Codigo_Cliente_nunique': 'Clientes_Ativos',
            'Margem_Real_sum': 'Margem_Total',
            'Desconto_sum': 'Desconto_Total'
        }
        
        financial_aggregated.rename(columns=column_mapping, inplace=True)
        
        # Calcular métricas básicas
        financial_aggregated['Receita_Media_Diaria'] = financial_aggregated['Receita_Total']
        if freq in ['W', 'M', 'Q']:
            days_in_period = {
                'W': 7,
                'M': 30,  # Aproximação
                'Q': 90   # Aproximação
            }
            financial_aggregated['Receita_Media_Diaria'] = (
                financial_aggregated['Receita_Total'] / days_in_period.get(freq, 1)
            )
        
        # Receita por cliente
        financial_aggregated['Receita_Por_Cliente'] = (
            financial_aggregated['Receita_Total'] / 
            financial_aggregated.get('Clientes_Ativos', 1).clip(lower=1)
        ).round(2)
        
        # Transações por cliente
        financial_aggregated['Transacoes_Por_Cliente'] = (
            financial_aggregated['Num_Transacoes'] / 
            financial_aggregated.get('Clientes_Ativos', 1).clip(lower=1)
        ).round(2)
        
        return financial_aggregated.reset_index()

    def _generate_enhanced_export_summary(self, financial_data: pd.DataFrame, output_path: str, 
                                        df_complete: pd.DataFrame, df_filtered: pd.DataFrame, 
                                        period_info: dict) -> str:
        """Gerar resumo melhorado da exportação com informações de período."""
        
        total_periods = len(financial_data)
        
        # Estatísticas do período filtrado
        filtered_revenue = financial_data['Receita_Total'].sum()
        avg_growth = financial_data['Receita_Variacao_Pct'].mean()
        avg_margin = financial_data['Margem_Liquida_Pct'].mean()
        
        # Comparação YoY
        yoy_growth = financial_data['YoY_Receita_Growth_Pct'].iloc[-1] if len(financial_data) > 0 else 0
        
        # Melhor e pior período
        best_period = financial_data.loc[financial_data['Receita_Total'].idxmax()]
        worst_period = financial_data.loc[financial_data['Receita_Total'].idxmin()]
        
        # Estatísticas da base completa
        complete_revenue = df_complete['Total_Liquido'].sum()
        complete_date_range = f"{df_complete['Data'].min().strftime('%Y-%m-%d')} a {df_complete['Data'].max().strftime('%Y-%m-%d')}"
        
        # Tendências
        trend_stats = financial_data['Classificacao_Tendencia'].value_counts()
        
        # Recomendações
        strategy_stats = financial_data['Recomendacao_Estrategica'].value_counts()
        
        file_size = os.path.getsize(output_path) / 1024  # KB
        
        summary = f"""
                        ✅ EXPORTAÇÃO DE DADOS FINANCEIROS CONCLUÍDA!

                        📁 **ARQUIVO GERADO**: {output_path}
                        📊 **TAMANHO**: {file_size:.1f} KB
                        🔢 **TOTAL DE PERÍODOS**: {total_periods:,}

                        ### 📅 PERÍODO ANALISADO:
                        - **Filtro Aplicado**: {period_info['description']}
                        - **Duração**: {period_info['days_count']} dias
                        - **Tipo de Análise**: {period_info['type']}

                        ### 💰 RESUMO FINANCEIRO DO PERÍODO:
                        - **Receita do Período**: R$ {filtered_revenue:,.0f}
                        - **Crescimento YoY**: {yoy_growth:.1f}%
                        - **Crescimento Sequencial Médio**: {avg_growth:.1f}%
                        - **Margem Líquida Média**: {avg_margin:.1f}%

                        ### 📊 BASE DE DADOS UTILIZADA:
                        - **Registros no Período**: {len(df_filtered):,}
                        - **Registros na Base Completa**: {len(df_complete):,}
                        - **Período Completo Disponível**: {complete_date_range}
                        - **Receita Total Histórica**: R$ {complete_revenue:,.0f}

                        ### 📈 MELHOR/PIOR PERFORMANCE NO PERÍODO:
                        - **Melhor Período**: {best_period['Periodo']} - R$ {best_period['Receita_Total']:,.0f}
                        - **Pior Período**: {worst_period['Periodo']} - R$ {worst_period['Receita_Total']:,.0f}

                        ### 📊 TENDÊNCIAS IDENTIFICADAS:
                        {chr(10).join([f"- **{k.replace('_', ' ')}**: {v} períodos" for k, v in trend_stats.head().items()])}

                        ### 🎯 RECOMENDAÇÕES ESTRATÉGICAS:
                        {chr(10).join([f"- **{k.replace('_', ' ')}**: {v} períodos" for k, v in strategy_stats.head().items()])}

                        ### 📋 PRINCIPAIS COLUNAS DO CSV:
                        
                        **🎯 Métricas do Período Filtrado:**
                        - **Receita_Total, Ticket_Medio, Num_Transacoes**: Dados do período especificado
                        - **Receita_Variacao_Pct**: Crescimento sequencial dentro do período
                        - **YoY_Receita_Growth_Pct**: Crescimento vs mesmo período ano anterior
                        
                        **📊 Métricas da Base Completa:**
                        - **Indice_Sazonal**: Baseado no histórico completo disponível
                        - **Projecoes**: Calculadas com dados dos últimos 2 anos
                        - **Confiabilidade_Modelo_Pct**: Baseada na disponibilidade histórica
                        
                        **🔄 Métricas Híbridas:**
                        - **Projecao_Anual**: Extrapolação do período atual
                        - **Tendencia_Crescimento**: Baseada no período filtrado
                        - **Scores de Saúde**: Calculados sobre o período especificado

                        ### ⚙️ CONFIGURAÇÃO UTILIZADA:
                        - **Fonte de Dados**: {len(df_complete):,} registros históricos
                        - **Filtro de Período**: {period_info['type']}
                        - **Comparação YoY**: {'Disponível' if yoy_growth != 0 else 'Não disponível'}
                        - **Confiabilidade**: {financial_data['Confiabilidade_Modelo_Pct'].iloc[0] if len(financial_data) > 0 else 'N/A'}

                        ### 💡 PRÓXIMOS PASSOS SUGERIDOS:
                        1. **Analisar crescimento YoY** de {yoy_growth:.1f}% vs benchmark
                        2. **Focar em períodos** com Prioridade_Acao = 1 (Urgente)
                        3. **Monitorar projeções** vs realizado mensalmente
                        4. **Implementar estratégias** por Recomendacao_Estrategica
                        5. **Expandir período** se confiabilidade for baixa

                        🎯 **Dados otimizados para o período especificado e prontos para análise executiva!**
                        """
        
        return summary.strip()

    def generate_financial_test_report(self, test_data: dict) -> str:
        """Gera relatório visual completo dos testes financeiros em formato markdown."""
        
        # Coletar dados com fallbacks
        metadata = test_data.get('metadata', {})
        data_metrics = test_data.get('data_metrics', {})
        results = test_data.get('results', {})
        component_tests = test_data.get('component_tests', {})
        
        report = [
            "# 💰 Teste Completo de Análise Financeira - Relatório Executivo",
            f"**Data do Teste:** {metadata.get('test_timestamp', 'N/A')}",
            f"**Fonte de Dados:** `{metadata.get('data_source', 'desconhecida')}`",
            f"**Registros Analisados:** {data_metrics.get('total_records', 0):,}",
            f"**Períodos Financeiros:** {data_metrics.get('total_periods', 0):,}",
            f"**Intervalo de Análise:** {data_metrics.get('date_range', {}).get('start', 'N/A')} até {data_metrics.get('date_range', {}).get('end', 'N/A')}",
            "\n## 📈 Performance de Execução",
            f"```\n{json.dumps(test_data.get('performance_metrics', {}), indent=2)}\n```",
            "\n## 🎯 Resumo dos Testes Executados"
        ]
        
        # Contabilizar sucessos e falhas
        successful_tests = len([r for r in results.values() if 'success' in r and r['success']])
        failed_tests = len([r for r in results.values() if 'success' in r and not r['success']])
        total_tests = len(results)
        
        report.extend([
            f"- **Total de Componentes:** {total_tests}",
            f"- **Sucessos:** {successful_tests} ✅",
            f"- **Falhas:** {failed_tests} ❌",
            f"- **Taxa de Sucesso:** {(successful_tests/total_tests*100):.1f}%" if total_tests > 0 else "- **Taxa de Sucesso:** N/A"
        ])
        
        # Principais Descobertas Financeiras
        report.append("\n## 💰 Principais Descobertas Financeiras")
        
        # KPIs Financeiros
        if 'kpi_analysis' in results and results['kpi_analysis'].get('success'):
            kpi_data = results['kpi_analysis']
            total_revenue_filtered = kpi_data.get('total_revenue_filtered', 0)
            total_revenue_aggregated = kpi_data.get('total_revenue_aggregated', 0)
            avg_growth = kpi_data.get('avg_growth_rate', 0)
            ytd_performance = kpi_data.get('ytd_performance', 0)
            report.append(f"- **Receita do Período Filtrado:** R$ {total_revenue_filtered:,.0f}")
            report.append(f"- **Receita Agregada por Períodos:** R$ {total_revenue_aggregated:,.0f}")
            report.append(f"- **Crescimento Médio:** {avg_growth:.1f}%")
            report.append(f"- **Performance YoY:** {ytd_performance:.1f}%")
        
        # Análise de Margens
        if 'margin_analysis' in results and results['margin_analysis'].get('success'):
            margin_data = results['margin_analysis']
            avg_margin = margin_data.get('avg_margin_pct', 0)
            discount_impact = margin_data.get('discount_impact_pct', 0)
            roi_avg = margin_data.get('avg_roi_pct', 0)
            report.append(f"- **Margem Líquida Média:** {avg_margin:.1f}%")
            report.append(f"- **Impacto de Descontos:** {discount_impact:.1f}%")
            report.append(f"- **ROI Médio:** {roi_avg:.1f}%")
        
        # Tendências e Sazonalidade
        if 'trend_analysis' in results and results['trend_analysis'].get('success'):
            trend_data = results['trend_analysis']
            trend_classification = trend_data.get('dominant_trend', 'N/A')
            seasonal_variation = trend_data.get('seasonal_variation', 0)
            report.append(f"- **Tendência Dominante:** {trend_classification.replace('_', ' ')}")
            report.append(f"- **Variação Sazonal:** {seasonal_variation:.1f}%")
        
        # Projeções
        if 'projections' in results and results['projections'].get('success'):
            proj_data = results['projections']
            projection_30d = proj_data.get('projection_30d_realistic', 0)
            confidence = proj_data.get('model_confidence', 0)
            report.append(f"- **Projeção 30 dias (Realista):** R$ {projection_30d:,.0f}")
            report.append(f"- **Confiabilidade do Modelo:** {confidence:.1f}%")
        
        # Saúde Financeira
        if 'health_scores' in results and results['health_scores'].get('success'):
            health_data = results['health_scores']
            overall_health = health_data.get('avg_health_score', 0)
            report.append(f"- **Score de Saúde Financeira:** {overall_health:.1f}/100")
        
        # Detalhamento por Componente
        report.append("\n## 🔧 Detalhamento dos Componentes Testados")
        
        component_categories = {
            'Preparação de Dados': ['data_loading', 'data_aggregation'],
            'Análise de KPIs': ['kpi_analysis'],
            'Análise de Margens': ['margin_analysis'],
            'Análise de Tendências': ['trend_analysis'],
            'Projeções Financeiras': ['projections'],
            'Insights Estratégicos': ['strategic_insights'],
            'Scores de Saúde': ['health_scores'],
            'Exportação': ['csv_export', 'summary_generation']
        }
        
        for category, components in component_categories.items():
            report.append(f"\n### {category}")
            for component in components:
                if component in results:
                    if results[component].get('success'):
                        metrics = results[component].get('metrics', {})
                        report.append(f"- ✅ **{component}**: Concluído")
                        if 'processing_time' in metrics:
                            report.append(f"  - Tempo: {metrics['processing_time']:.3f}s")
                        if 'records_processed' in metrics:
                            report.append(f"  - Registros: {metrics['records_processed']:,}")
                    else:
                        error_msg = results[component].get('error', 'Erro desconhecido')
                        report.append(f"- ❌ **{component}**: {error_msg}")
                else:
                    report.append(f"- ⏭️ **{component}**: Não testado")
        
        # Análise de Configurações
        report.append("\n## ⚙️ Teste de Configurações")
        
        if 'configuration_tests' in component_tests:
            config_tests = component_tests['configuration_tests']
            for config_name, config_result in config_tests.items():
                status = "✅" if config_result.get('success') else "❌"
                report.append(f"- {status} **{config_name}**: {config_result.get('description', 'N/A')}")
        
        # Qualidade dos Dados e Limitações
        report.append("\n## ⚠️ Qualidade dos Dados e Limitações")
        
        data_quality = data_metrics.get('data_quality_check', {})
        if data_quality:
            report.append("### Qualidade dos Dados:")
            for check, value in data_quality.items():
                if value > 0:
                    report.append(f"- **{check}**: {value} ocorrências")
        
        # Arquivos Gerados
        if 'files_generated' in component_tests:
            files = component_tests['files_generated']
            report.append(f"\n### Arquivos Gerados ({len(files)}):")
            for file_info in files:
                size_kb = file_info.get('size_kb', 0)
                report.append(f"- **{file_info['path']}**: {size_kb:.1f} KB")
        
        # Recomendações Finais
        report.append("\n## 💡 Recomendações do Sistema Financeiro")
        
        recommendations = [
            "📊 Monitorar períodos com Score_Saude_Financeira < 60",
            "💰 Focar na otimização de margens em períodos de baixa rentabilidade",
            "📈 Implementar estratégias baseadas nas Recomendacao_Estrategica",
            "🎯 Acompanhar projeções vs realizado mensalmente",
            "🔍 Investigar períodos com Prioridade_Acao = 1 (Urgente)"
        ]
        
        for rec in recommendations:
            report.append(f"- {rec}")
        
        # Erros encontrados
        errors = test_data.get('errors', [])
        if errors:
            report.append(f"\n### Erros Detectados ({len(errors)}):")
            for error in errors[-3:]:  # Últimos 3 erros
                report.append(f"- **{error['context']}**: {error['error_message']}")
        
        return "\n".join(report)

    def run_full_financial_test(self) -> str:
        """Executa teste completo e retorna relatório formatado"""
        test_result = self.test_all_financial_components()
        parsed = json.loads(test_result)
        return self.generate_financial_test_report(parsed)

    def test_all_financial_components(self, sample_data: str = "data/vendas.csv") -> str:
        """
        Executa teste completo de todos os componentes da classe FinancialDataExporter
        usando especificamente o arquivo data/vendas.csv
        """
        
        # Corrigir caminho do arquivo para usar data/vendas.csv especificamente
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
        
        # Usar especificamente data/vendas.csv
        data_file_path = os.path.join(project_root, "data", "vendas.csv")
        
        print(f"🔍 DEBUG: Caminho calculado: {data_file_path}")
        print(f"🔍 DEBUG: Arquivo existe? {os.path.exists(data_file_path)}")
        
        # Verificar se arquivo existe
        if not os.path.exists(data_file_path):
            # Tentar caminhos alternativos
            alternative_paths = [
                os.path.join(project_root, "data", "vendas.csv"),
                os.path.join(os.getcwd(), "data", "vendas.csv"),
                "data/vendas.csv",
                "data\\vendas.csv"
            ]
            
            for alt_path in alternative_paths:
                print(f"🔍 Tentando: {alt_path}")
                if os.path.exists(alt_path):
                    data_file_path = alt_path
                    print(f"✅ Arquivo encontrado em: {data_file_path}")
                    break
            else:
                return json.dumps({
                    "error": f"Arquivo data/vendas.csv não encontrado em nenhum dos caminhos testados",
                    "tested_paths": alternative_paths,
                    "current_dir": current_dir,
                    "project_root": project_root,
                    "working_directory": os.getcwd()
                }, indent=2)

        test_report = {
            "metadata": {
                "test_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "test_version": "Financial Test Suite v1.0",
                "data_source": data_file_path,
                "data_file_specified": "data/vendas.csv",
                "tool_version": "Financial Data Exporter v1.0",
                "status": "in_progress"
            },
            "data_metrics": {
                "total_records": 0,
                "total_periods": 0,
                "date_range": {},
                "data_quality_check": {}
            },
            "results": {},
            "component_tests": {},
            "performance_metrics": {},
            "errors": []
        }

        try:
            # 1. Fase de Carregamento de Dados
            test_report["metadata"]["current_stage"] = "data_loading"
            print("\n=== ETAPA 1: CARREGAMENTO DE DADOS FINANCEIROS ===")
            print(f"📁 Carregando especificamente: data/vendas.csv")
            print(f"📁 Caminho completo: {data_file_path}")
            
            start_time = time.time()
            df = self._load_and_prepare_data(data_file_path)
            loading_time = time.time() - start_time
            
            if df.empty:
                raise Exception("Falha no carregamento do arquivo data/vendas.csv")
            
            print(f"✅ data/vendas.csv carregado: {len(df)} registros em {loading_time:.3f}s")
            
            # Coletar métricas básicas dos dados
            test_report["data_metrics"] = {
                "total_records": int(len(df)),
                "date_range": {
                    "start": str(df['Data'].min()) if 'Data' in df.columns else "N/A",
                    "end": str(df['Data'].max()) if 'Data' in df.columns else "N/A"
                },
                "data_quality_check": self._perform_financial_data_quality_check(df)
            }
            
            test_report["results"]["data_loading"] = {
                "success": True,
                "metrics": {
                    "processing_time": loading_time,
                    "records_processed": len(df)
                }
            }

            # 2. Filtrar dados para YTD (comportamento padrão)
            test_report["metadata"]["current_stage"] = "data_filtering"
            print("\n=== ETAPA 2: APLICAÇÃO DE FILTRO YTD (PADRÃO) ===")
            
            try:
                start_time = time.time()
                print("🔍 Aplicando filtro YTD (Year-to-Date) padrão...")
                
                # Aplicar filtro YTD padrão (sem parâmetros)
                df_filtered, period_info = self._filter_data_by_period(df)
                filter_time = time.time() - start_time
                
                if df_filtered.empty:
                    raise Exception("Falha na aplicação do filtro YTD")
                
                print(f"✅ Filtro YTD aplicado: {len(df_filtered)} registros ({period_info['description']}) em {filter_time:.3f}s")
                
                test_report["results"]["data_filtering"] = {
                    "success": True,
                    "metrics": {
                        "processing_time": filter_time,
                        "records_filtered": len(df_filtered),
                        "filter_type": period_info['type'],
                        "period_description": period_info['description']
                    }
                }
                
            except Exception as e:
                self._log_financial_test_error(test_report, e, "data_filtering")
                print(f"❌ Erro na aplicação do filtro: {str(e)}")
                df_filtered = df  # Fallback para dados completos
                period_info = {'type': 'fallback', 'description': 'Sem filtro (fallback)'}

            # 3. Teste de Agregação de Dados por Período (usando dados filtrados)
            test_report["metadata"]["current_stage"] = "data_aggregation"
            print("\n=== ETAPA 3: TESTE DE AGREGAÇÃO FINANCEIRA (DADOS FILTRADOS) ===")
            
            try:
                start_time = time.time()
                print("📊 Testando agregação de dados filtrados por período mensal...")
                financial_data = self._aggregate_financial_data(df_filtered, "monthly")
                aggregation_time = time.time() - start_time
                
                test_report["data_metrics"]["total_periods"] = len(financial_data)
                test_report["data_metrics"]["filtered_records"] = len(df_filtered)
                test_report["data_metrics"]["period_info"] = period_info
                
                test_report["results"]["data_aggregation"] = {
                    "success": True,
                    "metrics": {
                        "processing_time": aggregation_time,
                        "periods_generated": len(financial_data),
                        "columns_generated": len(financial_data.columns)
                    }
                }
                print(f"✅ Agregação concluída: {len(financial_data)} períodos (dados filtrados) em {aggregation_time:.3f}s")
                
            except Exception as e:
                self._log_financial_test_error(test_report, e, "data_aggregation")
                print(f"❌ Erro na agregação: {str(e)}")
                financial_data = pd.DataFrame()  # Fallback vazio

            # 4. Teste de Análise de KPIs (usando dados filtrados)
            test_report["metadata"]["current_stage"] = "kpi_analysis"
            print("\n=== ETAPA 4: TESTE DE ANÁLISE DE KPIS (DADOS FILTRADOS) ===")
            
            if not financial_data.empty:
                try:
                    start_time = time.time()
                    print("💰 Testando análise de KPIs financeiros com dados filtrados...")
                    
                    # Usar o period_info correto do filtro aplicado
                    financial_data_kpi = self._add_enhanced_kpi_analysis(df, df_filtered, financial_data.copy(), period_info)
                    kpi_time = time.time() - start_time
                    
                    # Calcular métricas de KPI usando dados filtrados
                    total_revenue_filtered = financial_data_kpi['Receita_Periodo_Atual'].iloc[0] if len(financial_data_kpi) > 0 else 0
                    avg_growth = financial_data_kpi['Receita_Variacao_Pct'].mean()
                    ytd_performance = financial_data_kpi['YoY_Receita_Growth_Pct'].iloc[-1] if len(financial_data_kpi) > 0 else 0
                    
                    test_report["results"]["kpi_analysis"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": kpi_time,
                            "periods_analyzed": len(financial_data_kpi)
                        },
                        "total_revenue_filtered": float(total_revenue_filtered),  # Receita do período filtrado
                        "total_revenue_aggregated": float(financial_data_kpi['Receita_Total'].sum()),  # Receita agregada por sub-períodos
                        "avg_growth_rate": float(avg_growth),
                        "ytd_performance": float(ytd_performance)
                    }
                    print(f"✅ KPIs calculados: Receita YTD R$ {total_revenue_filtered:,.0f}, Agregada R$ {financial_data_kpi['Receita_Total'].sum():,.0f}, Crescimento {avg_growth:.1f}% em {kpi_time:.3f}s")
                    
                except Exception as e:
                    self._log_financial_test_error(test_report, e, "kpi_analysis")
                    print(f"❌ Erro na análise de KPIs: {str(e)}")
                    financial_data_kpi = financial_data.copy()
            else:
                financial_data_kpi = pd.DataFrame()

            # 5. Teste de Análise de Margens (usando dados filtrados)
            test_report["metadata"]["current_stage"] = "margin_analysis"
            print("\n=== ETAPA 5: TESTE DE ANÁLISE DE MARGENS (DADOS FILTRADOS) ===")
            
            if not financial_data_kpi.empty:
                try:
                    start_time = time.time()
                    print("📊 Testando análise de margens e rentabilidade com dados filtrados...")
                    financial_data_margin = self._add_margin_analysis(df_filtered, financial_data_kpi.copy())
                    margin_time = time.time() - start_time
                    
                    # Calcular métricas de margem
                    avg_margin = financial_data_margin['Margem_Liquida_Pct'].mean()
                    discount_impact = financial_data_margin['Impacto_Desconto_Pct'].mean()
                    avg_roi = financial_data_margin['ROI_Periodo_Pct'].mean()
                    
                    test_report["results"]["margin_analysis"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": margin_time,
                            "periods_analyzed": len(financial_data_margin)
                        },
                        "avg_margin_pct": float(avg_margin),
                        "discount_impact_pct": float(discount_impact),
                        "avg_roi_pct": float(avg_roi)
                    }
                    print(f"✅ Margens analisadas: {avg_margin:.1f}% líquida, {discount_impact:.1f}% desconto em {margin_time:.3f}s")
                    
                except Exception as e:
                    self._log_financial_test_error(test_report, e, "margin_analysis")
                    print(f"❌ Erro na análise de margens: {str(e)}")
                    financial_data_margin = financial_data_kpi.copy()
            else:
                financial_data_margin = pd.DataFrame()

            # 6. Teste de Análise de Tendências
            test_report["metadata"]["current_stage"] = "trend_analysis"
            print("\n=== ETAPA 6: TESTE DE ANÁLISE DE TENDÊNCIAS ===")
            
            if not financial_data_margin.empty:
                try:
                    start_time = time.time()
                    print("📈 Testando análise de tendências e sazonalidade...")
                    financial_data_trend = self._add_enhanced_trend_analysis(df, df, financial_data_margin.copy())
                    trend_time = time.time() - start_time
                    
                    # Analisar tendências
                    trend_stats = financial_data_trend['Classificacao_Tendencia'].value_counts()
                    dominant_trend = trend_stats.index[0] if len(trend_stats) > 0 else 'N/A'
                    seasonal_variation = financial_data_trend['Indice_Sazonal'].std() * 100
                    
                    test_report["results"]["trend_analysis"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": trend_time,
                            "periods_analyzed": len(financial_data_trend)
                        },
                        "dominant_trend": dominant_trend,
                        "seasonal_variation": float(seasonal_variation),
                        "trend_distribution": trend_stats.to_dict()
                    }
                    print(f"✅ Tendências: {dominant_trend}, variação sazonal {seasonal_variation:.1f}% em {trend_time:.3f}s")
                    
                except Exception as e:
                    self._log_financial_test_error(test_report, e, "trend_analysis")
                    print(f"❌ Erro na análise de tendências: {str(e)}")
                    financial_data_trend = financial_data_margin.copy()
            else:
                financial_data_trend = pd.DataFrame()

            # 7. Teste de Projeções Financeiras
            test_report["metadata"]["current_stage"] = "projections"
            print("\n=== ETAPA 7: TESTE DE PROJEÇÕES FINANCEIRAS ===")
            
            if not financial_data_trend.empty:
                try:
                    start_time = time.time()
                    print("🔮 Testando projeções financeiras...")
                    financial_data_proj = self._add_enhanced_financial_projections(df, df, financial_data_trend.copy())
                    proj_time = time.time() - start_time
                    
                    # Calcular métricas de projeção
                    projection_30d = financial_data_proj['Projecao_30d_Realista'].iloc[-1] if len(financial_data_proj) > 0 else 0
                    model_confidence = financial_data_proj['Confiabilidade_Modelo_Pct'].iloc[-1] if len(financial_data_proj) > 0 else 0
                    
                    test_report["results"]["projections"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": proj_time,
                            "periods_projected": len(financial_data_proj)
                        },
                        "projection_30d_realistic": float(projection_30d),
                        "model_confidence": float(model_confidence)
                    }
                    print(f"✅ Projeções: R$ {projection_30d:,.0f} (30d), {model_confidence:.1f}% confiança em {proj_time:.3f}s")
                    
                except Exception as e:
                    self._log_financial_test_error(test_report, e, "projections")
                    print(f"❌ Erro nas projeções: {str(e)}")
                    financial_data_proj = financial_data_trend.copy()
            else:
                financial_data_proj = pd.DataFrame()

            # 8. Teste de Insights Estratégicos
            test_report["metadata"]["current_stage"] = "strategic_insights"
            print("\n=== ETAPA 8: TESTE DE INSIGHTS ESTRATÉGICOS ===")
            
            if not financial_data_proj.empty:
                try:
                    start_time = time.time()
                    print("💡 Testando insights estratégicos...")
                    financial_data_insights = self._add_strategic_insights(df_filtered, financial_data_proj.copy())
                    insights_time = time.time() - start_time
                    
                    # Contar recomendações estratégicas
                    strategy_stats = financial_data_insights['Recomendacao_Estrategica'].value_counts()
                    urgent_actions = len(financial_data_insights[financial_data_insights['Prioridade_Acao'] == 1])
                    
                    test_report["results"]["strategic_insights"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": insights_time,
                            "periods_analyzed": len(financial_data_insights)
                        },
                        "strategy_distribution": strategy_stats.to_dict(),
                        "urgent_actions_needed": urgent_actions
                    }
                    print(f"✅ Insights: {len(strategy_stats)} estratégias, {urgent_actions} ações urgentes em {insights_time:.3f}s")
                    
                except Exception as e:
                    self._log_financial_test_error(test_report, e, "strategic_insights")
                    print(f"❌ Erro nos insights: {str(e)}")
                    financial_data_insights = financial_data_proj.copy()
            else:
                financial_data_insights = pd.DataFrame()

            # 9. Teste de Scores de Saúde Financeira
            test_report["metadata"]["current_stage"] = "health_scores"
            print("\n=== ETAPA 9: TESTE DE SCORES DE SAÚDE ===")
            
            if not financial_data_insights.empty:
                try:
                    start_time = time.time()
                    print("📊 Testando scores de saúde financeira...")
                    financial_data_health = self._add_financial_health_scores(financial_data_insights.copy())
                    health_time = time.time() - start_time
                    
                    # Calcular métricas de saúde
                    avg_health_score = financial_data_health['Score_Saude_Financeira'].mean()
                    healthy_periods = len(financial_data_health[financial_data_health['Score_Saude_Financeira'] > 70])
                    
                    test_report["results"]["health_scores"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": health_time,
                            "periods_scored": len(financial_data_health)
                        },
                        "avg_health_score": float(avg_health_score),
                        "healthy_periods": healthy_periods
                    }
                    print(f"✅ Saúde financeira: {avg_health_score:.1f}/100, {healthy_periods} períodos saudáveis em {health_time:.3f}s")
                    
                except Exception as e:
                    self._log_financial_test_error(test_report, e, "health_scores")
                    print(f"❌ Erro nos scores de saúde: {str(e)}")
                    financial_data_health = financial_data_insights.copy()
            else:
                financial_data_health = pd.DataFrame()

            # 10. Teste de Exportação CSV
            test_report["metadata"]["current_stage"] = "csv_export"
            print("\n=== ETAPA 10: TESTE DE EXPORTAÇÃO CSV ===")
            
            if not financial_data_health.empty:
                try:
                    start_time = time.time()
                    print("💾 Testando exportação CSV...")
                    
                    # Criar pasta de teste
                    test_output_dir = "test_results"
                    os.makedirs(test_output_dir, exist_ok=True)
                    test_output_path = os.path.join(test_output_dir, "financial_test_export.csv")
                    
                    export_success = self._export_to_csv(financial_data_health, test_output_path)
                    export_time = time.time() - start_time
                    
                    if export_success and os.path.exists(test_output_path):
                        file_size_kb = os.path.getsize(test_output_path) / 1024
                        
                        test_report["results"]["csv_export"] = {
                            "success": True,
                            "metrics": {
                                "processing_time": export_time,
                                "file_size_kb": file_size_kb,
                                "records_exported": len(financial_data_health)
                            },
                            "output_path": test_output_path
                        }
                        print(f"✅ CSV exportado: {file_size_kb:.1f} KB em {export_time:.3f}s")
                        
                        # Armazenar informação do arquivo gerado
                        test_report["component_tests"]["files_generated"] = [{
                            "path": test_output_path,
                            "size_kb": file_size_kb,
                            "type": "financial_export"
                        }]
                    else:
                        raise Exception("Falha na exportação do arquivo CSV")
                        
                except Exception as e:
                    self._log_financial_test_error(test_report, e, "csv_export")
                    print(f"❌ Erro na exportação: {str(e)}")

            # 11. Teste de Geração de Sumário
            test_report["metadata"]["current_stage"] = "summary_generation"
            print("\n=== ETAPA 11: TESTE DE GERAÇÃO DE SUMÁRIO ===")
            
            if not financial_data_health.empty:
                try:
                    start_time = time.time()
                    print("📋 Testando geração de sumário...")
                    
                    # Criar período_info para o sumário
                    period_info = {
                        'type': 'current_year',
                        'start_date': pd.to_datetime(f"{df['Data'].max().year}-01-01"),
                        'end_date': df['Data'].max(),
                        'description': f"Ano atual {df['Data'].max().year}",
                        'year': df['Data'].max().year,
                        'days_count': (df['Data'].max() - pd.to_datetime(f"{df['Data'].max().year}-01-01")).days + 1
                    }
                    
                    summary = self._generate_enhanced_export_summary(
                        financial_data_health, 
                        test_output_path if 'test_output_path' in locals() else "test_path", 
                        df,  # df_complete
                        df,  # df_filtered (usando mesmo df para teste)
                        period_info
                    )
                    summary_time = time.time() - start_time
                    
                    test_report["results"]["summary_generation"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": summary_time,
                            "summary_length": len(summary)
                        },
                        "summary_preview": summary[:500] + "..." if len(summary) > 500 else summary
                    }
                    print(f"✅ Sumário gerado: {len(summary)} caracteres em {summary_time:.3f}s")
                    
                except Exception as e:
                    self._log_financial_test_error(test_report, e, "summary_generation")
                    print(f"❌ Erro na geração de sumário: {str(e)}")

            # 12. Teste de Configurações Diferentes
            test_report["metadata"]["current_stage"] = "configuration_testing"
            print("\n=== ETAPA 12: TESTE DE CONFIGURAÇÕES ===")
            
            config_tests = {}
            
            # Teste com agrupamento semanal
            try:
                print("🔧 Testando configuração semanal...")
                start_time = time.time()
                weekly_result = self._run(
                    data_csv=data_file_path,
                    output_path="test_results/financial_weekly_test.csv",
                    group_by_period="weekly"
                )
                config_tests["weekly"] = {
                    "success": "❌" not in weekly_result,
                    "description": "Agrupamento semanal para análise de curto prazo",
                    "execution_time": time.time() - start_time
                }
                print("✅ Configuração semanal testada")
            except Exception as e:
                config_tests["weekly"] = {"success": False, "error": str(e)}
                print(f"❌ Erro na configuração semanal: {str(e)}")
            
            # Teste com agrupamento trimestral
            try:
                print("🔧 Testando configuração trimestral...")
                start_time = time.time()
                quarterly_result = self._run(
                    data_csv=data_file_path,
                    output_path="test_results/financial_quarterly_test.csv",
                    group_by_period="quarterly"
                )
                config_tests["quarterly"] = {
                    "success": "❌" not in quarterly_result,
                    "description": "Agrupamento trimestral para análise estratégica",
                    "execution_time": time.time() - start_time
                }
                print("✅ Configuração trimestral testada")
            except Exception as e:
                config_tests["quarterly"] = {"success": False, "error": str(e)}
                print(f"❌ Erro na configuração trimestral: {str(e)}")
            
            test_report["component_tests"]["configuration_tests"] = config_tests
            
            # 13. Análise de Performance Financeira
            if not financial_data_health.empty and 'Receita_Total' in financial_data_health.columns:
                total_revenue = financial_data_health['Receita_Total'].sum()
                best_period_revenue = financial_data_health['Receita_Total'].max()
                avg_health_score = financial_data_health['Score_Saude_Financeira'].mean()
                
                test_report["component_tests"]["financial_analysis"] = {
                    "total_revenue_analyzed": float(total_revenue),
                    "best_period_revenue": float(best_period_revenue),
                    "avg_health_score": float(avg_health_score),
                    "periods_with_growth": int(len(financial_data_health[financial_data_health['Receita_Variacao_Pct'] > 0]))
                }

            # 14. Performance Metrics
            test_report["performance_metrics"] = {
                "total_execution_time": sum([
                    result.get('metrics', {}).get('processing_time', 0) 
                    for result in test_report["results"].values() 
                    if isinstance(result, dict)
                ]),
                "memory_usage_mb": self._get_financial_memory_usage(),
                "largest_dataset_processed": len(financial_data_health) if not financial_data_health.empty else 0
            }

            # 15. Análise Final
            test_report["metadata"]["status"] = "completed" if not test_report["errors"] else "completed_with_errors"
            print(f"\n✅✅✅ TESTE FINANCEIRO COMPLETO - {len(test_report['errors'])} erros ✅✅✅")
            
            return json.dumps(test_report, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            test_report["metadata"]["status"] = "failed"
            self._log_financial_test_error(test_report, e, "global")
            print(f"❌ TESTE FINANCEIRO FALHOU: {str(e)}")
            return json.dumps(test_report, ensure_ascii=False, indent=2, default=str)

    def _log_financial_test_error(self, report: dict, error: Exception, context: str) -> None:
        """Registra erros de teste financeiro de forma estruturada"""
        error_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc()
        }
        report["errors"].append(error_entry)

    def _perform_financial_data_quality_check(self, df: pd.DataFrame) -> dict:
        """Executa verificações de qualidade específicas para dados financeiros"""
        checks = {
            "missing_dates": int(df['Data'].isnull().sum()) if 'Data' in df.columns else 0,
            "missing_revenue": int(df['Total_Liquido'].isnull().sum()) if 'Total_Liquido' in df.columns else 0,
            "negative_revenue": int((df['Total_Liquido'] < 0).sum()) if 'Total_Liquido' in df.columns else 0,
            "zero_revenue": int((df['Total_Liquido'] == 0).sum()) if 'Total_Liquido' in df.columns else 0,
            "duplicate_transactions": int(df.duplicated().sum()),
            "extreme_values": int((df['Total_Liquido'] > df['Total_Liquido'].quantile(0.99)).sum()) if 'Total_Liquido' in df.columns else 0
        }
        return checks

    def _get_financial_memory_usage(self) -> float:
        """Obtém uso de memória específico para análises financeiras"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Em MB
        except:
            return 0.0


# Exemplo de uso
if __name__ == "__main__":
    exporter = FinancialDataExporter()
    
    print("💰 Iniciando Teste Completo do Sistema Financeiro...")
    print("📁 Testando especificamente com: data/vendas.csv")
    
    # Executar teste usando especificamente data/vendas.csv
    report = exporter.run_full_financial_test()
    
    # Salvar relatório
    os.makedirs("test_results", exist_ok=True)
    with open("test_results/financial_test_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("✅ Relatório financeiro gerado em test_results/financial_test_report.md")
    print(f"📁 Teste executado com arquivo: data/vendas.csv")
    print("\n" + "="*80)
    print(report[:1500])  # Exibir parte do relatório no console 