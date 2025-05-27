from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any, List
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings

# Importar módulos compartilhados
from .shared.data_preparation import DataPreparationMixin
from .shared.business_mixins import JewelryBusinessAnalysisMixin, JewelryRFMAnalysisMixin

warnings.filterwarnings('ignore')

class InventoryDataExporterInput(BaseModel):
    """Schema para exportação de dados de estoque."""
    
    data_csv: str = Field(
        default="data/vendas.csv", 
        description="Caminho para arquivo CSV de vendas"
    )
    
    output_path: str = Field(
        default="data/outputs/analise_estoque_dados_completos.csv",
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
             output_path: str = "data/outputs/analise_estoque_dados_completos.csv",
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
            'Preco_Unitario_mean': 'Preco_Medio_Venda',
            'Custo_Produto_mean': 'Custo_Unitario',
            'Margem_Real_sum': 'Margem_Total_Historica',
            'Margem_Percentual_mean': 'Margem_Percentual_Media'
        }
        
        inventory_aggregated.rename(columns=column_mapping, inplace=True)
        
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
        if 'Custo_Unitario' in inventory_aggregated.columns:
            inventory_aggregated['Capital_Investido'] = (
                inventory_aggregated['Estoque_Estimado'] * inventory_aggregated['Custo_Unitario']
            ).fillna(
                inventory_aggregated['Estoque_Estimado'] * inventory_aggregated['Preco_Medio_Venda'] * 0.4  # 40% como custo estimado
            )
        else:
            # Estimar capital usando preço de venda como proxy (40% como custo)
            inventory_aggregated['Capital_Investido'] = (
                inventory_aggregated['Estoque_Estimado'] * inventory_aggregated['Preco_Medio_Venda'] * 0.4
            ).fillna(0)
            
            # Criar coluna Custo_Unitario estimado
            inventory_aggregated['Custo_Unitario'] = (
                inventory_aggregated['Preco_Medio_Venda'] * 0.4
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
        
        return inventory_data
    
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