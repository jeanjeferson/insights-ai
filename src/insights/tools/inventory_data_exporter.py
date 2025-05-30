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
    from .shared.report_formatter import ReportFormatterMixin
    from .shared.business_mixins import JewelryRFMAnalysisMixin, JewelryBusinessAnalysisMixin
except ImportError:
    # Imports absolutos (quando executado diretamente)
    from insights.tools.shared.data_preparation import DataPreparationMixin
    from insights.tools.shared.report_formatter import ReportFormatterMixin
    from insights.tools.shared.business_mixins import JewelryRFMAnalysisMixin, JewelryBusinessAnalysisMixin

warnings.filterwarnings('ignore')

class InventoryDataExporterInput(BaseModel):
    """Schema para exportação de dados de estoque."""
    
    data_csv: str = Field(
        default="data/vendas.csv", 
        description="Caminho para arquivo CSV de vendas"
    )
    
    output_path: str = Field(
        default="assets/data/analise_estoque_dados_completos.csv",
        description="Caminho de saída para o arquivo CSV de estoque exportado"
    )
    
    include_abc_classification: bool = Field(
        default=True,
        description="Incluir classificação ABC baseada em capital investido"
    )
    
    include_risk_analysis: bool = Field(
        default=True,
        description="Incluir análise de riscos (ruptura/obsolescência)"
    )
    
    include_ml_recommendations: bool = Field(
        default=True,
        description="Incluir recomendações ML para restock/liquidação"
    )
    
    low_stock_days: int = Field(
        default=7,
        description="Dias de estoque para considerar risco de ruptura"
    )
    
    obsolescence_months: int = Field(
        default=9,
        description="Meses sem venda para considerar risco de obsolescência"
    )
    
    min_turnover_rate: float = Field(
        default=1.0,
        description="Taxa mínima de giro anual para não ser slow mover"
    )

class InventoryDataExporter(BaseTool, DataPreparationMixin, JewelryBusinessAnalysisMixin, JewelryRFMAnalysisMixin):
    """
    Ferramenta especializada para exportar dados completos de gestão de estoque.
    
    Esta ferramenta gera um CSV abrangente com:
    - Classificação ABC baseada em capital investido
    - Análise de giro e turnover de estoque
    - Identificação de riscos (ruptura/obsolescência)
    - Recomendações ML para restock e liquidação
    - Scores de saúde de estoque
    - Flags de alertas para tomada de decisão
    """
    
    name: str = "Inventory Data Exporter"
    description: str = """
    Exporta dados completos de gestão de estoque em formato CSV para análise avançada.
    
    Inclui classificações ABC por capital, análise de giro, identificação de riscos,
    recomendações ML para restock/liquidação e scores de saúde por produto.
    
    Use esta ferramenta quando precisar de dados estruturados de estoque para:
    - Gestão de reposição e compras
    - Identificação de produtos críticos (ruptura/obsolescência)
    - Análises de capital investido e liberação de caixa
    - Dashboards de gestão de estoque (Power BI, Tableau)
    - Planejamento de liquidações e promoções
    - Otimização de níveis de estoque
    """
    args_schema: Type[BaseModel] = InventoryDataExporterInput

    def _run(self, data_csv: str = "data/vendas.csv", 
             output_path: str = "assets/data/analise_estoque_dados_completos.csv",
             include_abc_classification: bool = True,
             include_risk_analysis: bool = True,
             include_ml_recommendations: bool = True,
             low_stock_days: int = 7,
             obsolescence_months: int = 9,
             min_turnover_rate: float = 1.0) -> str:
        
        try:
            print("🚀 Iniciando exportação de dados de estoque...")
            
            # 1. Carregar e preparar dados
            print("📊 Carregando dados de vendas para análise de estoque...")
            df = self._load_and_prepare_data(data_csv)
            
            if df.empty:
                return "❌ Erro: Dados de vendas não encontrados ou inválidos"
            
            print(f"✅ Dados carregados: {len(df):,} registros")
            
            # 2. Agregar dados por produto para análise de estoque
            print("📦 Agregando dados de estoque por produto...")
            inventory_data = self._aggregate_inventory_data(df)
            
            # 3. Aplicar classificações e análises
            if include_abc_classification:
                print("💰 Aplicando classificação ABC por capital investido...")
                inventory_data = self._add_abc_capital_classification(inventory_data)
            
            # 4. Calcular métricas de giro e turnover
            print("🔄 Calculando métricas de giro e turnover...")
            inventory_data = self._add_turnover_metrics(df, inventory_data)
            
            # 5. Análise de riscos
            if include_risk_analysis:
                print("⚠️ Analisando riscos de ruptura e obsolescência...")
                inventory_data = self._add_risk_analysis(df, inventory_data, low_stock_days, obsolescence_months, min_turnover_rate)
            
            # 6. Recomendações ML
            if include_ml_recommendations:
                print("🤖 Gerando recomendações ML...")
                inventory_data = self._add_ml_recommendations(df, inventory_data)
            
            # 7. Adicionar scores e flags
            print("📊 Calculando scores de saúde de estoque...")
            inventory_data = self._add_health_scores(inventory_data)
            
            # 8. Exportar CSV
            print("💾 Exportando arquivo CSV de estoque...")
            success = self._export_to_csv(inventory_data, output_path)
            
            if success:
                return self._generate_export_summary(inventory_data, output_path)
            else:
                return "❌ Erro na exportação do arquivo CSV"
                
        except Exception as e:
            return f"❌ Erro na exportação de dados de estoque: {str(e)}"
    
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
            required_columns = ['Codigo_Produto', 'Total_Liquido', 'Quantidade', 'Data']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ Colunas obrigatórias ausentes: {missing_columns}")
                return pd.DataFrame()
            
            print(f"✅ Dados preparados: {len(df)} registros")
            return df
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return pd.DataFrame()
    
    def _aggregate_inventory_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agregar dados por produto com foco em métricas de estoque."""
        
        # Agregações específicas para estoque
        agg_dict = {
            'Total_Liquido': ['sum', 'mean'],
            'Quantidade': 'sum',
            'Data': ['min', 'max'],
            'Descricao_Produto': 'first',
        }
        
        # Adicionar colunas opcionais se existirem
        optional_columns = {
            'Grupo_Produto': 'first',
            'Preco_Unitario': 'mean',
            'Custo_Produto': 'mean',
            'Margem_Real': 'sum',
            'Margem_Percentual': 'mean'
        }
        
        for col, agg in optional_columns.items():
            if col in df.columns:
                agg_dict[col] = agg
        
        # Agregar dados
        inventory_aggregated = df.groupby('Codigo_Produto').agg(agg_dict)
        
        # Flatten column names
        new_columns = []
        for col in inventory_aggregated.columns:
            if isinstance(col, tuple):
                if col[1] == 'first' or col[1] == '':
                    new_columns.append(col[0])
                else:
                    new_columns.append(f"{col[0]}_{col[1]}")
            else:
                new_columns.append(col)
        
        inventory_aggregated.columns = new_columns
        
        # Renomear colunas para formato mais limpo
        column_mapping = {
            'Total_Liquido_sum': 'Receita_Total_Historica',
            'Total_Liquido_mean': 'Ticket_Medio',
            'Quantidade_sum': 'Volume_Vendido_Total',
            'Data_min': 'Primeira_Venda',
            'Data_max': 'Ultima_Venda',
            'Margem_Real_sum': 'Margem_Total_Historica',
            'Margem_Percentual_mean': 'Margem_Percentual_Media'
        }
        
        # Renomear apenas colunas que existem
        for old_name, new_name in column_mapping.items():
            if old_name in inventory_aggregated.columns:
                inventory_aggregated.rename(columns={old_name: new_name}, inplace=True)
        
        # Criar Preco_Medio_Venda se Preco_Unitario existe
        if 'Preco_Unitario_mean' in inventory_aggregated.columns:
            inventory_aggregated['Preco_Medio_Venda'] = inventory_aggregated['Preco_Unitario_mean']
        elif 'Preco_Unitario' in inventory_aggregated.columns:
            inventory_aggregated['Preco_Medio_Venda'] = inventory_aggregated['Preco_Unitario']
        else:
            # Calcular preço médio baseado em Total_Liquido / Quantidade
            inventory_aggregated['Preco_Medio_Venda'] = (
                inventory_aggregated['Receita_Total_Historica'] / 
                inventory_aggregated['Volume_Vendido_Total']
            ).fillna(0)
        
        # Criar Custo_Unitario se não existir
        if 'Custo_Produto_mean' in inventory_aggregated.columns:
            inventory_aggregated['Custo_Unitario'] = inventory_aggregated['Custo_Produto_mean']
        elif 'Custo_Produto' in inventory_aggregated.columns:
            inventory_aggregated['Custo_Unitario'] = inventory_aggregated['Custo_Produto']
        else:
            # Estimar custo como 40% do preço de venda
            inventory_aggregated['Custo_Unitario'] = (
                inventory_aggregated['Preco_Medio_Venda'] * 0.4
            ).fillna(0)
        
        # Calcular métricas básicas de estoque
        current_date = df['Data'].max()
        inventory_aggregated['Days_Since_Last_Sale'] = (
            current_date - pd.to_datetime(inventory_aggregated['Ultima_Venda'])
        ).dt.days
        
        # Período de vida do produto
        inventory_aggregated['Lifecycle_Days'] = (
            pd.to_datetime(inventory_aggregated['Ultima_Venda']) - 
            pd.to_datetime(inventory_aggregated['Primeira_Venda'])
        ).dt.days + 1
        
        # Estimar estoque atual (usando dados de venda para proxy)
        # Método conservador: baseado na média de vendas dos últimos 30 dias
        last_30_days = current_date - timedelta(days=30)
        recent_sales = df[df['Data'] >= last_30_days].groupby('Codigo_Produto')['Quantidade'].sum()
        
        # Estimar estoque como 45 dias de vendas (conservador para joalherias)
        inventory_aggregated['Estoque_Estimado'] = inventory_aggregated.index.map(
            lambda x: recent_sales.get(x, 0) * 1.5  # 45 dias = 1.5 * 30 dias
        ).fillna(0)
        
        # Capital investido estimado
        inventory_aggregated['Capital_Investido'] = (
            inventory_aggregated['Estoque_Estimado'] * inventory_aggregated['Custo_Unitario']
        ).fillna(0)
        
        return inventory_aggregated.reset_index()
    
    def _add_abc_capital_classification(self, inventory_data: pd.DataFrame) -> pd.DataFrame:
        """Adicionar classificação ABC baseada em capital investido."""
        
        # Classificação ABC por capital investido (não por receita)
        inventory_sorted = inventory_data.sort_values('Capital_Investido', ascending=False)
        
        total_capital = inventory_sorted['Capital_Investido'].sum()
        inventory_sorted['Capital_Cumsum'] = inventory_sorted['Capital_Investido'].cumsum()
        inventory_sorted['Capital_Cumsum_Pct'] = (
            inventory_sorted['Capital_Cumsum'] / total_capital * 100
        )
        
        # Classificação ABC para estoque (70% - 90% - 100%)
        def classify_abc_capital(row):
            pct = row['Capital_Cumsum_Pct']
            if pct <= 70:
                return 'A'
            elif pct <= 90:
                return 'B'
            else:
                return 'C'
        
        inventory_sorted['Classificacao_ABC_Estoque'] = inventory_sorted.apply(classify_abc_capital, axis=1)
        
        # Market share por capital
        inventory_sorted['Capital_Share_Pct'] = (
            inventory_sorted['Capital_Investido'] / total_capital * 100
        ).round(3)
        
        # Posição no ranking
        inventory_sorted['Posicao_ABC_Ranking'] = range(1, len(inventory_sorted) + 1)
        
        # Merge de volta com o índice original
        inventory_data = inventory_data.merge(
            inventory_sorted[['Codigo_Produto', 'Classificacao_ABC_Estoque', 'Capital_Share_Pct', 'Posicao_ABC_Ranking']],
            on='Codigo_Produto',
            how='left'
        )
        
        return inventory_data
    
    def _add_turnover_metrics(self, df: pd.DataFrame, inventory_data: pd.DataFrame) -> pd.DataFrame:
        """Adicionar métricas de giro e turnover."""
        
        # Calcular giro anual baseado em vendas históricas
        days_of_data = (df['Data'].max() - df['Data'].min()).days + 1
        
        # Giro anual = (volume vendido total / período) * 365 / estoque médio
        for idx, row in inventory_data.iterrows():
            volume_total = row['Volume_Vendido_Total']
            estoque_atual = row['Estoque_Estimado']
            
            if estoque_atual > 0 and days_of_data > 0:
                # Giro = vendas anualizadas / estoque médio
                vendas_anualizadas = (volume_total / days_of_data) * 365
                giro_anual = vendas_anualizadas / estoque_atual
                
                # DSI (Days Sales Inventory) = 365 / giro
                dsi_dias = 365 / giro_anual if giro_anual > 0 else 999
                
                inventory_data.at[idx, 'Giro_Anual'] = round(giro_anual, 2)
                inventory_data.at[idx, 'DSI_Dias'] = round(dsi_dias, 1)
                inventory_data.at[idx, 'Cobertura_Dias'] = round(dsi_dias, 1)  # Mesmo que DSI para simplificar
            else:
                inventory_data.at[idx, 'Giro_Anual'] = 0
                inventory_data.at[idx, 'DSI_Dias'] = 999
                inventory_data.at[idx, 'Cobertura_Dias'] = 999
        
        # Fill Rate estimado (assumindo 95% como padrão para produtos ativos)
        inventory_data['Fill_Rate_Pct'] = inventory_data.apply(
            lambda row: 95.0 if row['Days_Since_Last_Sale'] <= 60 else 
                       85.0 if row['Days_Since_Last_Sale'] <= 120 else 70.0, 
            axis=1
        )
        
        # Turnover Rate (velocidade de rotação)
        inventory_data['Turnover_Rate'] = inventory_data['Giro_Anual']
        
        return inventory_data
    
    def _add_risk_analysis(self, df: pd.DataFrame, inventory_data: pd.DataFrame, 
                          low_stock_days: int, obsolescence_months: int, min_turnover_rate: float) -> pd.DataFrame:
        """Adicionar análise de riscos de ruptura e obsolescência."""
        
        # Risco de ruptura (estoque baixo)
        inventory_data['Risco_Ruptura_Flag'] = (
            inventory_data['Cobertura_Dias'] <= low_stock_days
        ).astype(int)
        
        # Risco de obsolescência (sem venda há X meses)
        obsolescence_days = obsolescence_months * 30
        inventory_data['Risco_Obsolescencia_Flag'] = (
            inventory_data['Days_Since_Last_Sale'] >= obsolescence_days
        ).astype(int)
        
        # Slow Mover (giro baixo)
        inventory_data['Slow_Mover_Flag'] = (
            inventory_data['Giro_Anual'] < min_turnover_rate
        ).astype(int)
        
        # Dead Stock (sem movimento há muito tempo)
        inventory_data['Dead_Stock_Flag'] = (
            inventory_data['Days_Since_Last_Sale'] >= (obsolescence_days * 1.5)
        ).astype(int)
        
        # Status consolidado de estoque
        def get_status_estoque(row):
            if row['Dead_Stock_Flag'] == 1:
                return 'Dead_Stock'
            elif row['Risco_Ruptura_Flag'] == 1:
                return 'Critico'
            elif row['Risco_Obsolescencia_Flag'] == 1:
                return 'Obsolescencia'
            elif row['Slow_Mover_Flag'] == 1:
                return 'Slow_Mover'
            elif row['Giro_Anual'] > 6:  # Mais de 6x por ano
                return 'Alto_Giro'
            else:
                return 'Normal'
        
        inventory_data['Status_Estoque'] = inventory_data.apply(get_status_estoque, axis=1)
        
        # Calcular impacto mensal estimado para produtos críticos
        inventory_data['Impacto_Mensal_Estimado'] = (
            inventory_data['Receita_Total_Historica'] / inventory_data['Lifecycle_Days'] * 30
        ).round(2)
        
        return inventory_data
    
    def _add_ml_recommendations(self, df: pd.DataFrame, inventory_data: pd.DataFrame) -> pd.DataFrame:
        """Adicionar recomendações ML para restock e liquidação."""
        
        # Algoritmo simples de recomendações baseado em regras
        def get_recomendacao_acao(row):
            # Produtos críticos (risco de ruptura) = Restock
            if row['Risco_Ruptura_Flag'] == 1 and row['Giro_Anual'] > 2:
                return 'Restock_Urgente'
            elif row['Cobertura_Dias'] <= 15 and row['Giro_Anual'] > 1:
                return 'Restock'
            # Produtos obsoletos = Liquidação
            elif row['Dead_Stock_Flag'] == 1 or (row['Slow_Mover_Flag'] == 1 and row['Days_Since_Last_Sale'] > 180):
                return 'Liquidacao'
            # Produtos em excesso = Promoção
            elif row['Cobertura_Dias'] > 180 and row['Giro_Anual'] < 1:
                return 'Promocao'
            else:
                return 'Manter'
        
        inventory_data['Recomendacao_Acao'] = inventory_data.apply(get_recomendacao_acao, axis=1)
        
        # ROI de restock (baseado em giro histórico)
        def calculate_restock_roi(row):
            if row['Recomendacao_Acao'] in ['Restock_Urgente', 'Restock']:
                # ROI = (margem unitária * giro esperado) / custo unitário
                margem_unitaria = row['Preco_Medio_Venda'] - row['Custo_Unitario']
                if row['Custo_Unitario'] > 0 and margem_unitaria > 0:
                    roi = (margem_unitaria * row['Giro_Anual']) / row['Custo_Unitario'] * 100
                    return round(min(roi, 200), 1)  # Cap em 200%
            return 0
        
        inventory_data['ROI_Restock_Pct'] = inventory_data.apply(calculate_restock_roi, axis=1)
        
        # Quantidade sugerida para restock
        def calculate_quantidade_restock(row):
            if row['Recomendacao_Acao'] in ['Restock_Urgente', 'Restock']:
                # Sugerir 60 dias de cobertura baseado no giro
                if row['Giro_Anual'] > 0:
                    vendas_diarias = row['Volume_Vendido_Total'] / row['Lifecycle_Days']
                    return round(vendas_diarias * 60, 0)
            return 0
        
        inventory_data['Quantidade_Sugerida_Restock'] = inventory_data.apply(calculate_quantidade_restock, axis=1)
        
        # Desconto sugerido para liquidação
        def calculate_desconto_liquidacao(row):
            if row['Recomendacao_Acao'] == 'Liquidacao':
                # Desconto baseado no tempo sem venda
                days_since_sale = row['Days_Since_Last_Sale']
                if days_since_sale > 365:
                    return 50  # 50% para produtos muito parados
                elif days_since_sale > 270:
                    return 40  # 40% para produtos parados
                elif days_since_sale > 180:
                    return 30  # 30% para slow movers
                else:
                    return 20  # 20% desconto básico
            elif row['Recomendacao_Acao'] == 'Promocao':
                return 15  # 15% para promoções
            return 0
        
        inventory_data['Desconto_Liquidacao_Pct'] = inventory_data.apply(calculate_desconto_liquidacao, axis=1)
        
        # Liberação de capital estimada
        inventory_data['Liberacao_Capital_Estimada'] = (
            inventory_data['Capital_Investido'] * 
            (inventory_data['Desconto_Liquidacao_Pct'] / 100)
        ).round(2)
        
        # Prioridade de ação (1 = mais urgente, 5 = menos urgente)
        def get_prioridade_acao(row):
            if row['Recomendacao_Acao'] == 'Restock_Urgente':
                return 1
            elif row['Recomendacao_Acao'] == 'Liquidacao' and row['Capital_Investido'] > 10000:
                return 2
            elif row['Recomendacao_Acao'] == 'Restock':
                return 3
            elif row['Recomendacao_Acao'] == 'Liquidacao':
                return 4
            else:
                return 5
        
        inventory_data['Prioridade_Acao'] = inventory_data.apply(get_prioridade_acao, axis=1)
        
        # 🔥 GERAR CSV DE RECOMENDAÇÕES
        self._generate_recommendations_csv(inventory_data)
        
        return inventory_data

    def _generate_recommendations_csv(self, inventory_data: pd.DataFrame) -> None:
        """Gera CSV específico com recomendações ML organizadas por prioridade"""
        
        try:
            # Criar diretório de saída
            output_dir = "test_results"
            os.makedirs(output_dir, exist_ok=True)
            
            # Filtrar apenas produtos que precisam de ação (não 'Manter')
            actionable_recommendations = inventory_data[
                inventory_data['Recomendacao_Acao'] != 'Manter'
            ].copy()
            
            if actionable_recommendations.empty:
                print("⚠️ Nenhuma recomendação acionável encontrada")
                return
            
            # Organizar por prioridade e impacto
            actionable_recommendations = actionable_recommendations.sort_values([
                'Prioridade_Acao', 
                'Capital_Investido'
            ], ascending=[True, False])
            
            # Selecionar colunas mais relevantes para recomendações
            recommendations_columns = [
                'Codigo_Produto',
                'Descricao_Produto', 
                'Recomendacao_Acao',
                'Prioridade_Acao',
                'Capital_Investido',
                'Giro_Anual',
                'Days_Since_Last_Sale',
                'Cobertura_Dias',
                'ROI_Restock_Pct',
                'Quantidade_Sugerida_Restock',
                'Desconto_Liquidacao_Pct',
                'Liberacao_Capital_Estimada',
                'Classificacao_ABC_Estoque',
                'Status_Estoque'
            ]
            
            # Filtrar colunas que existem
            existing_columns = [col for col in recommendations_columns if col in actionable_recommendations.columns]
            recommendations_df = actionable_recommendations[existing_columns].copy()
            
            # Criar colunas adicionais para ação
            recommendations_df['Data_Recomendacao'] = datetime.now().strftime("%Y-%m-%d")
            recommendations_df['Status_Execucao'] = 'Pendente'
            recommendations_df['Observacoes'] = ''
            
            # Separar por tipo de recomendação
            restock_df = recommendations_df[recommendations_df['Recomendacao_Acao'].isin(['Restock', 'Restock_Urgente'])]
            liquidacao_df = recommendations_df[recommendations_df['Recomendacao_Acao'].isin(['Liquidacao', 'Promocao'])]
            
            # Salvar CSV principal com todas as recomendações
            main_csv_path = os.path.join(output_dir, "ml_recommendations_all.csv")
            recommendations_df.to_csv(main_csv_path, index=False, sep=';', encoding='utf-8')
            
            # Salvar CSV específico para Restock
            if not restock_df.empty:
                restock_csv_path = os.path.join(output_dir, "ml_recommendations_restock.csv")
                restock_df.to_csv(restock_csv_path, index=False, sep=';', encoding='utf-8')
                print(f"✅ CSV Restock gerado: {restock_csv_path} ({len(restock_df)} produtos)")
            
            # Salvar CSV específico para Liquidação
            if not liquidacao_df.empty:
                liquidacao_csv_path = os.path.join(output_dir, "ml_recommendations_liquidacao.csv")
                liquidacao_df.to_csv(liquidacao_csv_path, index=False, sep=';', encoding='utf-8')
                print(f"✅ CSV Liquidação gerado: {liquidacao_csv_path} ({len(liquidacao_df)} produtos)")
            
            # Gerar relatório resumo das recomendações
            self._generate_recommendations_summary(recommendations_df, output_dir)
            
            print(f"✅ CSV Principal gerado: {main_csv_path} ({len(recommendations_df)} recomendações)")
            
            # Estatísticas das recomendações
            stats = recommendations_df['Recomendacao_Acao'].value_counts()
            total_capital = recommendations_df['Capital_Investido'].sum()
            
            print(f"📊 RESUMO DAS RECOMENDAÇÕES:")
            for acao, count in stats.items():
                print(f"   - {acao}: {count} produtos")
            print(f"💰 Capital Total Envolvido: R$ {total_capital:,.2f}")
            
        except Exception as e:
            print(f"❌ Erro ao gerar CSV de recomendações: {str(e)}")

    def _generate_recommendations_summary(self, recommendations_df: pd.DataFrame, output_dir: str) -> None:
        """Gera relatório resumo das recomendações em markdown"""
        
        try:
            summary_path = os.path.join(output_dir, "ml_recommendations_summary.md")
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("# 🤖 Relatório de Recomendações ML - Sistema de Inventário\n\n")
                f.write(f"**Data de Geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Total de Recomendações:** {len(recommendations_df)}\n\n")
                
                # Estatísticas por tipo de recomendação
                f.write("## 📊 Distribuição de Recomendações\n\n")
                stats = recommendations_df['Recomendacao_Acao'].value_counts()
                for acao, count in stats.items():
                    pct = (count / len(recommendations_df)) * 100
                    f.write(f"- **{acao}**: {count} produtos ({pct:.1f}%)\n")
                
                # Top 10 por capital investido
                f.write("\n## 💰 Top 10 por Capital Investido\n\n")
                top_capital = recommendations_df.nlargest(10, 'Capital_Investido')
                f.write("| Produto | Recomendação | Capital | Prioridade |\n")
                f.write("|---------|--------------|---------|------------|\n")
                for _, row in top_capital.iterrows():
                    f.write(f"| {row['Codigo_Produto']} | {row['Recomendacao_Acao']} | R$ {row['Capital_Investido']:,.2f} | {row['Prioridade_Acao']} |\n")
                
                # Recomendações urgentes
                urgent = recommendations_df[recommendations_df['Prioridade_Acao'] <= 2]
                if not urgent.empty:
                    f.write(f"\n## 🚨 Recomendações Urgentes ({len(urgent)} produtos)\n\n")
                    f.write("| Produto | Ação | Motivo | Capital |\n")
                    f.write("|---------|------|--------|----------|\n")
                    for _, row in urgent.iterrows():
                        f.write(f"| {row['Codigo_Produto']} | {row['Recomendacao_Acao']} | Prioridade {row['Prioridade_Acao']} | R$ {row['Capital_Investido']:,.2f} |\n")
            
                # Potencial de liberação de capital
                total_liberacao = recommendations_df['Liberacao_Capital_Estimada'].sum()
                if total_liberacao > 0:
                    f.write(f"\n## 💸 Potencial de Liberação de Capital\n\n")
                    f.write(f"**Total Estimado:** R$ {total_liberacao:,.2f}\n\n")
                    
                    liquidacao_products = recommendations_df[recommendations_df['Recomendacao_Acao'].isin(['Liquidacao', 'Promocao'])]
                    if not liquidacao_products.empty:
                        f.write("### Top 5 Liberação de Capital:\n")
                        top_liberacao = liquidacao_products.nlargest(5, 'Liberacao_Capital_Estimada')
                        for _, row in top_liberacao.iterrows():
                            f.write(f"- **{row['Codigo_Produto']}**: R$ {row['Liberacao_Capital_Estimada']:,.2f} (desconto {row['Desconto_Liquidacao_Pct']}%)\n")
            
            print(f"✅ Relatório resumo gerado: {summary_path}")
            
        except Exception as e:
            print(f"❌ Erro ao gerar relatório resumo: {str(e)}")
    
    def _add_health_scores(self, inventory_data: pd.DataFrame) -> pd.DataFrame:
        """Adicionar scores de saúde de estoque."""
        
        # Score de Giro (normalizado 0-100)
        max_giro = inventory_data['Giro_Anual'].quantile(0.95)  # 95th percentile como máximo
        inventory_data['Score_Giro'] = (
            inventory_data['Giro_Anual'] / max_giro * 100
        ).clip(0, 100).round(1)
        
        # Score de Cobertura (inverso - menos dias é melhor)
        # Ideal: 30-60 dias de cobertura
        def calculate_coverage_score(days):
            if 30 <= days <= 60:
                return 100
            elif days < 30:
                return max(50, 100 - (30 - days) * 2)  # Penalizar estoque muito baixo
            else:
                return max(0, 100 - (days - 60) * 0.5)  # Penalizar estoque excessivo
        
        inventory_data['Score_Cobertura'] = inventory_data['Cobertura_Dias'].apply(calculate_coverage_score).round(1)
        
        # Score de Demanda (baseado em recência)
        max_days_since_sale = inventory_data['Days_Since_Last_Sale'].max()
        inventory_data['Score_Demanda'] = (
            (max_days_since_sale - inventory_data['Days_Since_Last_Sale']) / max_days_since_sale * 100
        ).round(1)
        
        # Score Geral de Saúde (média ponderada)
        inventory_data['Score_Saude_Estoque'] = (
            inventory_data['Score_Giro'] * 0.4 +
            inventory_data['Score_Cobertura'] * 0.35 +
            inventory_data['Score_Demanda'] * 0.25
        ).round(1)
        
        # Score de Urgência (baseado em flags)
        def calculate_urgency_score(row):
            score = 0
            if row['Risco_Ruptura_Flag'] == 1:
                score += 50
            if row['Dead_Stock_Flag'] == 1:
                score += 40
            elif row['Risco_Obsolescencia_Flag'] == 1:
                score += 30
            elif row['Slow_Mover_Flag'] == 1:
                score += 20
            
            # Adicionar urgência baseada em capital
            if row['Capital_Investido'] > 50000:
                score += 20
            elif row['Capital_Investido'] > 20000:
                score += 10
            
            return min(score, 100)
        
        inventory_data['Urgencia_Score'] = inventory_data.apply(calculate_urgency_score, axis=1)
        
        # Score de Risco (combinação de fatores)
        inventory_data['Risco_Score'] = (
            inventory_data['Risco_Ruptura_Flag'] * 30 +
            inventory_data['Risco_Obsolescencia_Flag'] * 25 +
            inventory_data['Dead_Stock_Flag'] * 35 +
            inventory_data['Slow_Mover_Flag'] * 10
        ).clip(0, 100)
        
        return inventory_data
    
    def _export_to_csv(self, inventory_data: pd.DataFrame, output_path: str) -> bool:
        """Exportar dados para CSV."""
        
        try:
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Ordenar por Score de Saúde e Prioridade de Ação
            inventory_data_sorted = inventory_data.sort_values(
                ['Prioridade_Acao', 'Score_Saude_Estoque'], 
                ascending=[True, False]
            )
            
            # Reorganizar colunas para melhor visualização
            priority_columns = [
                'Codigo_Produto', 'Descricao_Produto', 'Grupo_Produto',
                'Status_Estoque', 'Recomendacao_Acao', 'Prioridade_Acao',
                'Classificacao_ABC_Estoque', 'Capital_Investido', 'Capital_Share_Pct',
                'Estoque_Estimado', 'Giro_Anual', 'DSI_Dias', 'Cobertura_Dias',
                'Score_Saude_Estoque', 'Urgencia_Score', 'Risco_Score',
                'ROI_Restock_Pct', 'Quantidade_Sugerida_Restock',
                'Desconto_Liquidacao_Pct', 'Liberacao_Capital_Estimada',
                'Risco_Ruptura_Flag', 'Risco_Obsolescencia_Flag', 'Slow_Mover_Flag', 'Dead_Stock_Flag'
            ]
            
            # Adicionar colunas restantes
            remaining_columns = [col for col in inventory_data_sorted.columns if col not in priority_columns]
            final_columns = priority_columns + remaining_columns
            
            # Filtrar colunas que existem
            existing_columns = [col for col in final_columns if col in inventory_data_sorted.columns]
            
            # Exportar CSV
            inventory_data_sorted[existing_columns].to_csv(
                output_path, index=False, sep=';', encoding='utf-8'
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao exportar CSV: {str(e)}")
            return False
    
    def _generate_export_summary(self, inventory_data: pd.DataFrame, output_path: str) -> str:
        """Gerar resumo da exportação."""
        
        total_products = len(inventory_data)
        
        # Estatísticas por classificação ABC
        abc_stats = inventory_data['Classificacao_ABC_Estoque'].value_counts()
        abc_capital = inventory_data.groupby('Classificacao_ABC_Estoque')['Capital_Investido'].sum()
        
        # Estatísticas de status
        status_stats = inventory_data['Status_Estoque'].value_counts()
        
        # Estatísticas de recomendações
        action_stats = inventory_data['Recomendacao_Acao'].value_counts()
        
        # Alertas críticos
        criticos = inventory_data[inventory_data['Risco_Ruptura_Flag'] == 1]
        obsoletos = inventory_data[inventory_data['Risco_Obsolescencia_Flag'] == 1]
        dead_stock = inventory_data[inventory_data['Dead_Stock_Flag'] == 1]
        
        # Capital total e oportunidades
        capital_total = inventory_data['Capital_Investido'].sum()
        liberacao_potencial = inventory_data['Liberacao_Capital_Estimada'].sum()
        
        file_size = os.path.getsize(output_path) / 1024  # KB
        
        summary = f"""
                        ✅ EXPORTAÇÃO DE DADOS DE ESTOQUE CONCLUÍDA!

                        📁 **ARQUIVO GERADO**: {output_path}
                        📊 **TAMANHO**: {file_size:.1f} KB
                        🔢 **TOTAL DE PRODUTOS**: {total_products:,}

                        ### 💰 CLASSIFICAÇÃO ABC DE ESTOQUE (POR CAPITAL):
                        {chr(10).join([f"- **Classe {k}**: {v} produtos - R$ {abc_capital.get(k, 0):,.0f} ({v/total_products*100:.1f}%)" for k, v in abc_stats.items()])}

                        ### 📦 STATUS DE ESTOQUE:
                        {chr(10).join([f"- **{k.replace('_', ' ')}**: {v} produtos ({v/total_products*100:.1f}%)" for k, v in status_stats.head().items()])}

                        ### 🚨 ALERTAS CRÍTICOS:
                        - **Risco Ruptura**: {len(criticos)} produtos (R$ {criticos['Capital_Investido'].sum():,.0f} em risco)
                        - **Obsolescência**: {len(obsoletos)} produtos (R$ {obsoletos['Capital_Investido'].sum():,.0f} em risco)
                        - **Dead Stock**: {len(dead_stock)} produtos (R$ {dead_stock['Capital_Investido'].sum():,.0f} parado)

                        ### 🤖 RECOMENDAÇÕES ML:
                        {chr(10).join([f"- **{k.replace('_', ' ')}**: {v} produtos" for k, v in action_stats.head().items()])}

                        ### 💵 ANÁLISE FINANCEIRA:
                        - **Capital Total Investido**: R$ {capital_total:,.0f}
                        - **Liberação Potencial** (liquidações): R$ {liberacao_potencial:,.0f}
                        - **ROI Médio Restock**: {inventory_data[inventory_data['ROI_Restock_Pct'] > 0]['ROI_Restock_Pct'].mean():.1f}%

                        ### 📋 PRINCIPAIS COLUNAS DO CSV:
                        - **Identificação**: Codigo_Produto, Descricao_Produto, Grupo_Produto
                        - **Classificação**: Classificacao_ABC_Estoque, Status_Estoque
                        - **Métricas**: Giro_Anual, DSI_Dias, Capital_Investido, Score_Saude_Estoque
                        - **Alertas**: Risco_Ruptura_Flag, Risco_Obsolescencia_Flag, Dead_Stock_Flag
                        - **Recomendações**: Recomendacao_Acao, ROI_Restock_Pct, Desconto_Liquidacao_Pct

                        ### 💡 PRÓXIMOS PASSOS SUGERIDOS:
                        1. **Filtrar Prioridade_Acao = 1** para ações urgentes
                        2. **Analisar Classe A** com Score_Saude_Estoque < 70
                        3. **Revisar produtos** com Liberacao_Capital_Estimada > R$ 10K
                        4. **Implementar restock** para produtos com ROI_Restock_Pct > 30%
                        5. **Planejar liquidação** para Dead_Stock_Flag = 1

                        🎯 **Dados prontos para gestão de estoque em Excel, Power BI ou ERP!**
                        """
        
        return summary.strip()

    def generate_inventory_test_report(self, test_data: dict) -> str:
        """Gera relatório visual completo dos testes de inventário em formato markdown."""
        
        # Coletar dados com fallbacks
        metadata = test_data.get('metadata', {})
        data_metrics = test_data.get('data_metrics', {})
        results = test_data.get('results', {})
        component_tests = test_data.get('component_tests', {})
        
        report = [
            "# 📦 Teste Completo de Inventário - Relatório Executivo",
            f"**Data do Teste:** {metadata.get('test_timestamp', 'N/A')}",
            f"**Fonte de Dados:** `{metadata.get('data_source', 'desconhecida')}`",
            f"**Registros Analisados:** {data_metrics.get('total_records', 0):,}",
            f"**Produtos Únicos:** {data_metrics.get('unique_products', 0):,}",
            f"**Período de Análise:** {data_metrics.get('date_range', {}).get('start', 'N/A')} até {data_metrics.get('date_range', {}).get('end', 'N/A')}",
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
        
        # Principais Descobertas do Inventário
        report.append("\n## 📊 Principais Descobertas do Inventário")
        
        # Estatísticas ABC
        if 'abc_classification' in results and results['abc_classification'].get('success'):
            abc_data = results['abc_classification']
            if 'abc_distribution' in abc_data:
                abc_dist = abc_data['abc_distribution']
                report.append(f"- **Classificação ABC:** Classe A: {abc_dist.get('A', 0)} produtos, Classe B: {abc_dist.get('B', 0)} produtos, Classe C: {abc_dist.get('C', 0)} produtos")
        
        # Capital Investido
        if 'capital_analysis' in component_tests:
            capital_data = component_tests['capital_analysis']
            total_capital = capital_data.get('total_capital', 0)
            report.append(f"- **Capital Total Investido:** R$ {total_capital:,.2f}")
        
        # Riscos Identificados
        if 'risk_analysis' in results and results['risk_analysis'].get('success'):
            risk_data = results['risk_analysis']
            ruptura = risk_data.get('ruptura_count', 0)
            obsolescencia = risk_data.get('obsolescencia_count', 0)
            dead_stock = risk_data.get('dead_stock_count', 0)
            report.append(f"- **Riscos Identificados:** {ruptura} rupturas, {obsolescencia} obsolescências, {dead_stock} dead stock")
        
        # Recomendações ML
        if 'ml_recommendations' in results and results['ml_recommendations'].get('success'):
            ml_data = results['ml_recommendations']
            restock = ml_data.get('restock_count', 0)
            liquidacao = ml_data.get('liquidacao_count', 0)
            report.append(f"- **Recomendações ML:** {restock} para restock, {liquidacao} para liquidação")
        
        # Giro de Estoque
        if 'turnover_metrics' in results and results['turnover_metrics'].get('success'):
            turnover_data = results['turnover_metrics']
            avg_turnover = turnover_data.get('avg_turnover', 0)
            report.append(f"- **Giro Médio Anual:** {avg_turnover:.2f}x")
        
        # Detalhamento por Componente
        report.append("\n## 🔧 Detalhamento dos Componentes Testados")
        
        component_categories = {
            'Preparação de Dados': ['data_loading', 'data_aggregation'],
            'Classificações': ['abc_classification', 'turnover_metrics'],
            'Análise de Riscos': ['risk_analysis', 'health_scores'],
            'Machine Learning': ['ml_recommendations'],
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
        report.append("\n## 💡 Recomendações do Sistema de Inventário")
        
        recommendations = [
            "📊 Implementar monitoramento automático de produtos Classe A",
            "🚨 Configurar alertas para produtos em risco de ruptura",
            "💰 Priorizar liquidação de dead stock para liberação de capital",
            "🔄 Revisar políticas de restock baseadas no giro calculado",
            "📈 Utilizar scores de saúde para dashboards executivos"
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

    def run_full_inventory_test(self) -> str:
        """Executa teste completo e retorna relatório formatado"""
        test_result = self.test_all_inventory_components()
        parsed = json.loads(test_result)
        return self.generate_inventory_test_report(parsed)

    def test_all_inventory_components(self, sample_data: str = "data/vendas.csv") -> str:
        """
        Executa teste completo de todos os componentes da classe InventoryDataExporter
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
                "test_version": "Inventory Test Suite v1.0",
                "data_source": data_file_path,
                "data_file_specified": "data/vendas.csv",
                "tool_version": "Inventory Data Exporter v1.0",
                "status": "in_progress"
            },
            "data_metrics": {
                "total_records": 0,
                "unique_products": 0,
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
            print("\n=== ETAPA 1: CARREGAMENTO DE DADOS DE INVENTÁRIO ===")
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
                "unique_products": int(df['Codigo_Produto'].nunique()) if 'Codigo_Produto' in df.columns else 0,
                "date_range": {
                    "start": str(df['Data'].min()) if 'Data' in df.columns else "N/A",
                    "end": str(df['Data'].max()) if 'Data' in df.columns else "N/A"
                },
                "data_quality_check": self._perform_inventory_data_quality_check(df)
            }
            
            test_report["results"]["data_loading"] = {
                "success": True,
                "metrics": {
                    "processing_time": loading_time,
                    "records_processed": len(df)
                }
            }

            # 2. Teste de Agregação de Dados
            test_report["metadata"]["current_stage"] = "data_aggregation"
            print("\n=== ETAPA 2: TESTE DE AGREGAÇÃO DE DADOS ===")
            
            try:
                start_time = time.time()
                print("📊 Testando agregação de dados por produto...")
                inventory_data = self._aggregate_inventory_data(df)
                aggregation_time = time.time() - start_time
                
                test_report["results"]["data_aggregation"] = {
                    "success": True,
                    "metrics": {
                        "processing_time": aggregation_time,
                        "products_aggregated": len(inventory_data),
                        "columns_generated": len(inventory_data.columns)
                    }
                }
                print(f"✅ Agregação concluída: {len(inventory_data)} produtos em {aggregation_time:.3f}s")
                
            except Exception as e:
                self._log_inventory_test_error(test_report, e, "data_aggregation")
                print(f"❌ Erro na agregação: {str(e)}")
                inventory_data = pd.DataFrame()  # Fallback vazio

            # 3. Teste de Classificação ABC
            test_report["metadata"]["current_stage"] = "abc_classification"
            print("\n=== ETAPA 3: TESTE DE CLASSIFICAÇÃO ABC ===")
            
            if not inventory_data.empty:
                try:
                    start_time = time.time()
                    print("💰 Testando classificação ABC por capital...")
                    inventory_data_abc = self._add_abc_capital_classification(inventory_data.copy())
                    abc_time = time.time() - start_time
                    
                    # Analisar distribuição ABC
                    abc_distribution = inventory_data_abc['Classificacao_ABC_Estoque'].value_counts().to_dict()
                    
                    test_report["results"]["abc_classification"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": abc_time,
                            "products_classified": len(inventory_data_abc)
                        },
                        "abc_distribution": abc_distribution
                    }
                    print(f"✅ Classificação ABC: {abc_distribution} em {abc_time:.3f}s")
                    
                except Exception as e:
                    self._log_inventory_test_error(test_report, e, "abc_classification")
                    print(f"❌ Erro na classificação ABC: {str(e)}")
                    inventory_data_abc = inventory_data.copy()
            else:
                inventory_data_abc = pd.DataFrame()

            # 4. Teste de Métricas de Turnover
            test_report["metadata"]["current_stage"] = "turnover_metrics"
            print("\n=== ETAPA 4: TESTE DE MÉTRICAS DE TURNOVER ===")
            
            if not inventory_data_abc.empty:
                try:
                    start_time = time.time()
                    print("🔄 Testando cálculo de giro e turnover...")
                    inventory_data_turnover = self._add_turnover_metrics(df, inventory_data_abc.copy())
                    turnover_time = time.time() - start_time
                    
                    # Calcular estatísticas de giro
                    avg_turnover = inventory_data_turnover['Giro_Anual'].mean()
                    high_turnover = len(inventory_data_turnover[inventory_data_turnover['Giro_Anual'] > 6])
                    
                    test_report["results"]["turnover_metrics"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": turnover_time,
                            "products_analyzed": len(inventory_data_turnover)
                        },
                        "avg_turnover": avg_turnover,
                        "high_turnover_count": high_turnover
                    }
                    print(f"✅ Turnover calculado: giro médio {avg_turnover:.2f}x em {turnover_time:.3f}s")
                    
                except Exception as e:
                    self._log_inventory_test_error(test_report, e, "turnover_metrics")
                    print(f"❌ Erro no turnover: {str(e)}")
                    inventory_data_turnover = inventory_data_abc.copy()
            else:
                inventory_data_turnover = pd.DataFrame()

            # 5. Teste de Análise de Riscos
            test_report["metadata"]["current_stage"] = "risk_analysis"
            print("\n=== ETAPA 5: TESTE DE ANÁLISE DE RISCOS ===")
            
            if not inventory_data_turnover.empty:
                try:
                    start_time = time.time()
                    print("⚠️ Testando análise de riscos...")
                    inventory_data_risk = self._add_risk_analysis(df, inventory_data_turnover.copy(), 7, 9, 1.0)
                    risk_time = time.time() - start_time
                    
                    # Contar riscos identificados
                    ruptura_count = inventory_data_risk['Risco_Ruptura_Flag'].sum()
                    obsolescencia_count = inventory_data_risk['Risco_Obsolescencia_Flag'].sum()
                    dead_stock_count = inventory_data_risk['Dead_Stock_Flag'].sum()
                    
                    test_report["results"]["risk_analysis"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": risk_time,
                            "products_analyzed": len(inventory_data_risk)
                        },
                        "ruptura_count": int(ruptura_count),
                        "obsolescencia_count": int(obsolescencia_count),
                        "dead_stock_count": int(dead_stock_count)
                    }
                    print(f"✅ Riscos analisados: {ruptura_count} rupturas, {obsolescencia_count} obsolescências em {risk_time:.3f}s")
                    
                except Exception as e:
                    self._log_inventory_test_error(test_report, e, "risk_analysis")
                    print(f"❌ Erro na análise de riscos: {str(e)}")
                    inventory_data_risk = inventory_data_turnover.copy()
            else:
                inventory_data_risk = pd.DataFrame()

            # 6. Teste de Recomendações ML
            test_report["metadata"]["current_stage"] = "ml_recommendations"
            print("\n=== ETAPA 6: TESTE DE RECOMENDAÇÕES ML ===")
            
            if not inventory_data_risk.empty:
                try:
                    start_time = time.time()
                    print("🤖 Testando recomendações ML...")
                    inventory_data_ml = self._add_ml_recommendations(df, inventory_data_risk.copy())
                    ml_time = time.time() - start_time
                    
                    # Contar recomendações
                    recomendacoes = inventory_data_ml['Recomendacao_Acao'].value_counts().to_dict()
                    restock_count = recomendacoes.get('Restock', 0) + recomendacoes.get('Restock_Urgente', 0)
                    liquidacao_count = recomendacoes.get('Liquidacao', 0)
                    
                    test_report["results"]["ml_recommendations"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": ml_time,
                            "products_analyzed": len(inventory_data_ml)
                        },
                        "restock_count": int(restock_count),
                        "liquidacao_count": int(liquidacao_count),
                        "recommendations_distribution": recomendacoes
                    }
                    print(f"✅ ML concluído: {restock_count} restock, {liquidacao_count} liquidações em {ml_time:.3f}s")
                    
                except Exception as e:
                    self._log_inventory_test_error(test_report, e, "ml_recommendations")
                    print(f"❌ Erro nas recomendações ML: {str(e)}")
                    inventory_data_ml = inventory_data_risk.copy()
            else:
                inventory_data_ml = pd.DataFrame()

            # 7. Teste de Health Scores
            test_report["metadata"]["current_stage"] = "health_scores"
            print("\n=== ETAPA 7: TESTE DE HEALTH SCORES ===")
            
            if not inventory_data_ml.empty:
                try:
                    start_time = time.time()
                    print("🩺 Testando cálculo de health scores...")
                    inventory_data_health = self._add_health_scores(inventory_data_ml.copy())
                    health_time = time.time() - start_time
                    
                    # Calcular estatísticas de saúde
                    avg_health_score = inventory_data_health['Score_Saude_Estoque'].mean()
                    healthy_products = len(inventory_data_health[inventory_data_health['Score_Saude_Estoque'] > 70])
                    
                    test_report["results"]["health_scores"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": health_time,
                            "products_scored": len(inventory_data_health)
                        },
                        "avg_health_score": avg_health_score,
                        "healthy_products_count": healthy_products
                    }
                    print(f"✅ Health scores: média {avg_health_score:.1f}, {healthy_products} produtos saudáveis em {health_time:.3f}s")
                    
                except Exception as e:
                    self._log_inventory_test_error(test_report, e, "health_scores")
                    print(f"❌ Erro nos health scores: {str(e)}")
                    inventory_data_health = inventory_data_ml.copy()
            else:
                inventory_data_health = pd.DataFrame()

            # 8. Teste de Exportação CSV
            test_report["metadata"]["current_stage"] = "csv_export"
            print("\n=== ETAPA 8: TESTE DE EXPORTAÇÃO CSV ===")
            
            if not inventory_data_health.empty:
                try:
                    start_time = time.time()
                    print("💾 Testando exportação CSV...")
                    
                    # Criar pasta de teste
                    test_output_dir = "test_results"
                    os.makedirs(test_output_dir, exist_ok=True)
                    test_output_path = os.path.join(test_output_dir, "inventory_test_export.csv")
                    
                    export_success = self._export_to_csv(inventory_data_health, test_output_path)
                    export_time = time.time() - start_time
                    
                    if export_success and os.path.exists(test_output_path):
                        file_size_kb = os.path.getsize(test_output_path) / 1024
                        
                        test_report["results"]["csv_export"] = {
                            "success": True,
                            "metrics": {
                                "processing_time": export_time,
                                "file_size_kb": file_size_kb,
                                "records_exported": len(inventory_data_health)
                            },
                            "output_path": test_output_path
                        }
                        print(f"✅ CSV exportado: {file_size_kb:.1f} KB em {export_time:.3f}s")
                        
                        # Armazenar informação do arquivo gerado
                        test_report["component_tests"]["files_generated"] = [{
                            "path": test_output_path,
                            "size_kb": file_size_kb,
                            "type": "inventory_export"
                        }]
                    else:
                        raise Exception("Falha na exportação do arquivo CSV")
                        
                except Exception as e:
                    self._log_inventory_test_error(test_report, e, "csv_export")
                    print(f"❌ Erro na exportação: {str(e)}")

            # 9. Teste de Geração de Sumário
            test_report["metadata"]["current_stage"] = "summary_generation"
            print("\n=== ETAPA 9: TESTE DE GERAÇÃO DE SUMÁRIO ===")
            
            if not inventory_data_health.empty:
                try:
                    start_time = time.time()
                    print("📋 Testando geração de sumário...")
                    
                    summary = self._generate_export_summary(inventory_data_health, test_output_path if 'test_output_path' in locals() else "test_path")
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
                    self._log_inventory_test_error(test_report, e, "summary_generation")
                    print(f"❌ Erro na geração de sumário: {str(e)}")

            # 10. Teste de Configurações Diferentes
            test_report["metadata"]["current_stage"] = "configuration_testing"
            print("\n=== ETAPA 10: TESTE DE CONFIGURAÇÕES ===")
            
            config_tests = {}
            
            # Teste com configurações conservadoras
            try:
                print("🔧 Testando configuração conservadora...")
                start_time = time.time()
                conservative_result = self._run(
                    data_csv=data_file_path,
                    output_path="test_results/conservative_test.csv",
                    low_stock_days=14,
                    obsolescence_months=6,
                    min_turnover_rate=1.5
                )
                config_tests["conservative"] = {
                    "success": "❌" not in conservative_result,
                    "description": "Configuração conservadora (14 dias, 6 meses, 1.5x turnover)",
                    "execution_time": time.time() - start_time
                }
                print("✅ Configuração conservadora testada")
            except Exception as e:
                config_tests["conservative"] = {"success": False, "error": str(e)}
                print(f"❌ Erro na configuração conservadora: {str(e)}")
            
            # Teste com configurações agressivas
            try:
                print("🔧 Testando configuração agressiva...")
                start_time = time.time()
                aggressive_result = self._run(
                    data_csv=data_file_path,
                    output_path="test_results/aggressive_test.csv",
                    low_stock_days=3,
                    obsolescence_months=12,
                    min_turnover_rate=0.5
                )
                config_tests["aggressive"] = {
                    "success": "❌" not in aggressive_result,
                    "description": "Configuração agressiva (3 dias, 12 meses, 0.5x turnover)",
                    "execution_time": time.time() - start_time
                }
                print("✅ Configuração agressiva testada")
            except Exception as e:
                config_tests["aggressive"] = {"success": False, "error": str(e)}
                print(f"❌ Erro na configuração agressiva: {str(e)}")
            
            test_report["component_tests"]["configuration_tests"] = config_tests
            
            # 11. Análise de Capital
            if not inventory_data_health.empty and 'Capital_Investido' in inventory_data_health.columns:
                total_capital = inventory_data_health['Capital_Investido'].sum()
                avg_capital_per_product = inventory_data_health['Capital_Investido'].mean()
                
                test_report["component_tests"]["capital_analysis"] = {
                    "total_capital": float(total_capital),
                    "avg_capital_per_product": float(avg_capital_per_product),
                    "products_above_10k": int(len(inventory_data_health[inventory_data_health['Capital_Investido'] > 10000]))
                }

            # 12. Performance Metrics
            test_report["performance_metrics"] = {
                "total_execution_time": sum([
                    result.get('metrics', {}).get('processing_time', 0) 
                    for result in test_report["results"].values() 
                    if isinstance(result, dict)
                ]),
                "memory_usage_mb": self._get_inventory_memory_usage(),
                "largest_dataset_processed": len(inventory_data_health) if not inventory_data_health.empty else 0
            }

            # 13. Análise Final
            test_report["metadata"]["status"] = "completed" if not test_report["errors"] else "completed_with_errors"
            print(f"\n✅✅✅ TESTE DE INVENTÁRIO COMPLETO - {len(test_report['errors'])} erros ✅✅✅")
            
            return json.dumps(test_report, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            test_report["metadata"]["status"] = "failed"
            self._log_inventory_test_error(test_report, e, "global")
            print(f"❌ TESTE DE INVENTÁRIO FALHOU: {str(e)}")
            return json.dumps(test_report, ensure_ascii=False, indent=2, default=str)

    def _log_inventory_test_error(self, report: dict, error: Exception, context: str) -> None:
        """Registra erros de teste de inventário de forma estruturada"""
        error_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc()
        }
        report["errors"].append(error_entry)

    def _perform_inventory_data_quality_check(self, df: pd.DataFrame) -> dict:
        """Executa verificações de qualidade específicas para dados de inventário"""
        checks = {
            "missing_product_codes": int(df['Codigo_Produto'].isnull().sum()) if 'Codigo_Produto' in df.columns else 0,
            "missing_dates": int(df['Data'].isnull().sum()) if 'Data' in df.columns else 0,
            "negative_quantities": int((df['Quantidade'] < 0).sum()) if 'Quantidade' in df.columns else 0,
            "zero_prices": int((df['Total_Liquido'] <= 0).sum()) if 'Total_Liquido' in df.columns else 0,
            "duplicate_transactions": int(df.duplicated().sum()),
            "products_without_sales": 0  # Será calculado durante agregação
        }
        return checks

    def _get_inventory_memory_usage(self) -> float:
        """Obtém uso de memória específico para análises de inventário"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Em MB
        except:
            return 0.0


# Exemplo de uso
if __name__ == "__main__":
    exporter = InventoryDataExporter()
    
    print("📦 Iniciando Teste Completo do Sistema de Inventário...")
    print("📁 Testando especificamente com: data/vendas.csv")
    
    # Executar teste usando especificamente data/vendas.csv
    report = exporter.run_full_inventory_test()
    
    # Salvar relatório
    os.makedirs("test_results", exist_ok=True)
    with open("test_results/inventory_test_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("✅ Relatório de inventário gerado em test_results/inventory_test_report.md")
    print(f"📁 Teste executado com arquivo: data/vendas.csv")
    print("\n" + "="*80)
    print(report[:1500])  # Exibir parte do relatório no console 