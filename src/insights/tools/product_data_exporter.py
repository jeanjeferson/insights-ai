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

class ProductDataExporterInput(BaseModel):
    """Schema para exportação de dados de produtos."""
    
    data_csv: str = Field(
        default="data/vendas.csv", 
        description="Caminho para arquivo CSV de vendas"
    )
    
    output_path: str = Field(
        default="data/outputs/analise_produtos_dados_completos.csv",
        description="Caminho de saída para o arquivo CSV exportado"
    )
    
    include_abc_classification: bool = Field(
        default=True,
        description="Incluir classificação ABC dos produtos"
    )
    
    include_bcg_matrix: bool = Field(
        default=True,
        description="Incluir classificação da matriz BCG"
    )
    
    include_lifecycle_analysis: bool = Field(
        default=True,
        description="Incluir análise de ciclo de vida dos produtos"
    )
    
    slow_mover_days: int = Field(
        default=120,
        description="Dias sem venda para considerar slow mover"
    )
    
    dead_stock_days: int = Field(
        default=180,
        description="Dias sem venda para considerar dead stock"
    )

class ProductDataExporter(BaseTool, DataPreparationMixin, JewelryBusinessAnalysisMixin, JewelryRFMAnalysisMixin):
    """
    Ferramenta especializada para exportar dados de produtos com classificações completas.
    
    Esta ferramenta gera um CSV abrangente com:
    - Classificação ABC automática
    - Matriz BCG (Stars, Cash Cows, Question Marks, Dogs)
    - Análise de ciclo de vida
    - Métricas de performance
    - Flags de alertas para ação
    """
    
    name: str = "Product Data Exporter"
    description: str = """
    Exporta dados completos de produtos em formato CSV para análise avançada.
    
    Inclui classificações ABC, BCG, métricas de performance, identificação de slow movers,
    dead stock e outras análises necessárias para tomada de decisão pelos analistas.
    
    Use esta ferramenta quando precisar de dados estruturados de produtos para:
    - Análises em planilhas (Excel, Google Sheets)
    - Dashboards externos (Power BI, Tableau)
    - Análises estatísticas customizadas
    - Filtros e segmentações avançadas
    """
    args_schema: Type[BaseModel] = ProductDataExporterInput

    def _run(self, data_csv: str = "data/vendas.csv", 
             output_path: str = "data/outputs/analise_produtos_dados_completos.csv",
             include_abc_classification: bool = True,
             include_bcg_matrix: bool = True, 
             include_lifecycle_analysis: bool = True,
             slow_mover_days: int = 120,
             dead_stock_days: int = 180) -> str:
        
        try:
            print("🚀 Iniciando exportação de dados de produtos...")
            
            # 1. Carregar e preparar dados
            print("📊 Carregando dados de vendas...")
            df = self._load_and_prepare_data(data_csv)
            
            if df.empty:
                return "❌ Erro: Dados de vendas não encontrados ou inválidos"
            
            print(f"✅ Dados carregados: {len(df):,} registros")
            
            # 2. Agregar dados por produto
            print("🔄 Agregando dados por produto...")
            product_data = self._aggregate_product_data(df)
            
            # 3. Aplicar classificações
            if include_abc_classification:
                print("🏆 Aplicando classificação ABC...")
                product_data = self._add_abc_classification(df, product_data)
            
            if include_bcg_matrix:
                print("📈 Aplicando matriz BCG...")
                product_data = self._add_bcg_classification(df, product_data)
            
            if include_lifecycle_analysis:
                print("🔄 Analisando ciclo de vida...")
                product_data = self._add_lifecycle_analysis(df, product_data, slow_mover_days, dead_stock_days)
            
            # 4. Adicionar métricas avançadas
            print("📊 Calculando métricas avançadas...")
            product_data = self._add_advanced_metrics(df, product_data)
            
            # 5. Adicionar flags de alertas
            print("🚨 Adicionando flags de alertas...")
            product_data = self._add_alert_flags(product_data, slow_mover_days, dead_stock_days)
            
            # 6. Exportar CSV
            print("💾 Exportando arquivo CSV...")
            success = self._export_to_csv(product_data, output_path)
            
            if success:
                return self._generate_export_summary(product_data, output_path)
            else:
                return "❌ Erro na exportação do arquivo CSV"
                
        except Exception as e:
            return f"❌ Erro na exportação de dados: {str(e)}"
    
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
            
            print(f"✅ Dados carregados: {len(df)} registros")
            
            # Preparar dados básicos para análise de produtos
            # Converter Data para datetime
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
    
    def _aggregate_product_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agregar dados básicos por produto."""
        
        # Definir agregações dinamicamente baseado nas colunas disponíveis
        agg_dict = {
            'Total_Liquido': ['sum', 'mean', 'count'],
            'Quantidade': 'sum',
            'Data': ['min', 'max'],
            'Descricao_Produto': 'first',
        }
        
        # Adicionar colunas opcionais se existirem
        optional_columns = {
            'Grupo_Produto': 'first',
            'Margem_Real': 'sum',
            'Margem_Percentual': 'mean',
            'Preco_Unitario': 'mean',
            'Custo_Produto': 'mean'
        }
        
        for col, agg in optional_columns.items():
            if col in df.columns:
                agg_dict[col] = agg
        
        # Agregar dados
        product_aggregated = df.groupby('Codigo_Produto').agg(agg_dict)
        
        # Flatten column names
        new_columns = []
        for col in product_aggregated.columns:
            if isinstance(col, tuple):
                if col[1] == 'first' or col[1] == '':
                    new_columns.append(col[0])
                else:
                    new_columns.append(f"{col[0]}_{col[1]}")
            else:
                new_columns.append(col)
        
        product_aggregated.columns = new_columns
        
        # Renomear colunas para formato mais limpo
        column_mapping = {
            'Total_Liquido_sum': 'Receita_Total',
            'Total_Liquido_mean': 'Ticket_Medio',
            'Total_Liquido_count': 'Num_Transacoes',
            'Quantidade_sum': 'Volume_Vendido',
            'Data_min': 'Primeira_Venda',
            'Data_max': 'Ultima_Venda',
            'Margem_Real_sum': 'Margem_Total',
            'Margem_Percentual_mean': 'Margem_Percentual',
            'Preco_Unitario_mean': 'Preco_Medio',
            'Custo_Produto_mean': 'Custo_Medio'
        }
        
        product_aggregated.rename(columns=column_mapping, inplace=True)
        
        # Calcular métricas básicas
        current_date = df['Data'].max()
        product_aggregated['Days_Since_Last_Sale'] = (
            current_date - pd.to_datetime(product_aggregated['Ultima_Venda'])
        ).dt.days
        
        # Calcular período de vida do produto
        product_aggregated['Lifecycle_Days'] = (
            pd.to_datetime(product_aggregated['Ultima_Venda']) - 
            pd.to_datetime(product_aggregated['Primeira_Venda'])
        ).dt.days + 1
        
        # Giro estimado (receita / dias de vida * 365)
        product_aggregated['Giro_Anual_Estimado'] = (
            product_aggregated['Num_Transacoes'] / product_aggregated['Lifecycle_Days'] * 365
        ).round(2)
        
        return product_aggregated.reset_index()
    
    def _add_abc_classification(self, df: pd.DataFrame, product_data: pd.DataFrame) -> pd.DataFrame:
        """Adicionar classificação ABC."""
        
        # Usar o mixin para análise ABC
        abc_analysis = self.perform_abc_analysis(df, dimension='product')
        
        if 'error' in abc_analysis:
            print(f"⚠️ Erro na classificação ABC: {abc_analysis['error']}")
            product_data['Classificacao_ABC'] = 'N/A'
            return product_data
        
        # Calcular classificação ABC manualmente para ter controle total
        product_data_sorted = product_data.sort_values('Receita_Total', ascending=False)
        
        total_revenue = product_data_sorted['Receita_Total'].sum()
        product_data_sorted['Revenue_Cumsum'] = product_data_sorted['Receita_Total'].cumsum()
        product_data_sorted['Revenue_Cumsum_Pct'] = (
            product_data_sorted['Revenue_Cumsum'] / total_revenue * 100
        )
        
        # Classificação ABC (70% - 90% - 100%)
        def classify_abc(row):
            pct = row['Revenue_Cumsum_Pct']
            if pct <= 70:
                return 'A'
            elif pct <= 90:
                return 'B'
            else:
                return 'C'
        
        product_data_sorted['Classificacao_ABC'] = product_data_sorted.apply(classify_abc, axis=1)
        
        # Calcular market share
        product_data_sorted['Market_Share_Pct'] = (
            product_data_sorted['Receita_Total'] / total_revenue * 100
        ).round(3)
        
        # Merge de volta com o índice original
        product_data = product_data.merge(
            product_data_sorted[['Codigo_Produto', 'Classificacao_ABC', 'Market_Share_Pct']],
            on='Codigo_Produto',
            how='left'
        )
        
        return product_data
    
    def _add_bcg_classification(self, df: pd.DataFrame, product_data: pd.DataFrame) -> pd.DataFrame:
        """Adicionar classificação da matriz BCG."""
        
        # Usar o mixin para análise BCG
        bcg_analysis = self.create_product_bcg_matrix(df)
        
        if 'error' in bcg_analysis:
            print(f"⚠️ Erro na classificação BCG: {bcg_analysis['error']}")
            product_data['Classificacao_BCG'] = 'N/A'
            return product_data
        
        # Implementar classificação BCG simplificada
        # Market Share = % da receita total
        # Growth Rate = receita diária (proxy para crescimento)
        
        total_revenue = product_data['Receita_Total'].sum()
        product_data['Market_Share_BCG'] = product_data['Receita_Total'] / total_revenue * 100
        product_data['Daily_Revenue'] = product_data['Receita_Total'] / product_data['Lifecycle_Days']
        
        # Medianas para classificação
        market_share_median = product_data['Market_Share_BCG'].median()
        daily_revenue_median = product_data['Daily_Revenue'].median()
        
        def classify_bcg(row):
            ms = row['Market_Share_BCG']
            dr = row['Daily_Revenue']
            
            if ms > market_share_median and dr > daily_revenue_median:
                return 'Stars'
            elif ms > market_share_median and dr <= daily_revenue_median:
                return 'Cash_Cows'
            elif ms <= market_share_median and dr > daily_revenue_median:
                return 'Question_Marks'
            else:
                return 'Dogs'
        
        product_data['Classificacao_BCG'] = product_data.apply(classify_bcg, axis=1)
        
        # Limpar colunas auxiliares
        product_data.drop(['Market_Share_BCG', 'Daily_Revenue'], axis=1, inplace=True)
        
        return product_data
    
    def _add_lifecycle_analysis(self, df: pd.DataFrame, product_data: pd.DataFrame, 
                               slow_mover_days: int, dead_stock_days: int) -> pd.DataFrame:
        """Adicionar análise de ciclo de vida."""
        
        def classify_lifecycle(row):
            days_since_last = row['Days_Since_Last_Sale']
            giro = row['Giro_Anual_Estimado']
            
            if days_since_last >= dead_stock_days:
                return 'Dead_Stock'
            elif days_since_last >= slow_mover_days:
                return 'Slow_Mover'
            elif giro >= 12:  # Mais de 1x por mês
                return 'High_Velocity'
            elif giro >= 6:   # Mais de 1x a cada 2 meses
                return 'Medium_Velocity'
            elif giro >= 2:   # Pelo menos 2x por ano
                return 'Low_Velocity'
            else:
                return 'Very_Low_Velocity'
        
        product_data['Status_Lifecycle'] = product_data.apply(classify_lifecycle, axis=1)
        
        return product_data
    
    def _add_advanced_metrics(self, df: pd.DataFrame, product_data: pd.DataFrame) -> pd.DataFrame:
        """Adicionar métricas avançadas."""
        
        # Revenue Score (normalizado 0-100)
        max_revenue = product_data['Receita_Total'].max()
        product_data['Revenue_Score'] = (
            product_data['Receita_Total'] / max_revenue * 100
        ).round(2)
        
        # Frequency Score (baseado em número de transações)
        max_transactions = product_data['Num_Transacoes'].max()
        product_data['Frequency_Score'] = (
            product_data['Num_Transacoes'] / max_transactions * 100
        ).round(2)
        
        # Recency Score (inverso dos dias desde última venda)
        max_days = product_data['Days_Since_Last_Sale'].max()
        product_data['Recency_Score'] = (
            (max_days - product_data['Days_Since_Last_Sale']) / max_days * 100
        ).round(2)
        
        # Score Geral (média ponderada)
        product_data['Score_Geral'] = (
            product_data['Revenue_Score'] * 0.4 +
            product_data['Frequency_Score'] * 0.3 +
            product_data['Recency_Score'] * 0.3
        ).round(2)
        
        return product_data
    
    def _add_alert_flags(self, product_data: pd.DataFrame, slow_mover_days: int, dead_stock_days: int) -> pd.DataFrame:
        """Adicionar flags de alertas para ação."""
        
        # Flag de Slow Mover
        product_data['Slow_Mover_Flag'] = (
            product_data['Days_Since_Last_Sale'] >= slow_mover_days
        ).astype(int)
        
        # Flag de Dead Stock
        product_data['Dead_Stock_Flag'] = (
            product_data['Days_Since_Last_Sale'] >= dead_stock_days
        ).astype(int)
        
        # Flag de Alto Valor (Classe A)
        product_data['High_Value_Flag'] = (
            product_data['Classificacao_ABC'] == 'A'
        ).astype(int)
        
        # Flag de Baixo Giro
        product_data['Low_Turnover_Flag'] = (
            product_data['Giro_Anual_Estimado'] < 2
        ).astype(int)
        
        # Flag de Ação Requerida (combinação de alertas)
        product_data['Action_Required_Flag'] = (
            (product_data['Slow_Mover_Flag'] == 1) |
            (product_data['Dead_Stock_Flag'] == 1) |
            (product_data['Low_Turnover_Flag'] == 1)
        ).astype(int)
        
        return product_data
    
    def _export_to_csv(self, product_data: pd.DataFrame, output_path: str) -> bool:
        """Exportar dados para CSV."""
        
        try:
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Ordenar por Score Geral (descendente)
            product_data_sorted = product_data.sort_values('Score_Geral', ascending=False)
            
            # Exportar CSV
            product_data_sorted.to_csv(output_path, index=False, sep=';', encoding='utf-8')
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao exportar CSV: {str(e)}")
            return False
    
    def _generate_export_summary(self, product_data: pd.DataFrame, output_path: str) -> str:
        """Gerar resumo da exportação."""
        
        total_products = len(product_data)
        
        # Estatísticas por classificação ABC
        abc_stats = product_data['Classificacao_ABC'].value_counts()
        
        # Estatísticas por BCG
        bcg_stats = product_data['Classificacao_BCG'].value_counts()
        
        # Estatísticas de alertas
        slow_movers = product_data['Slow_Mover_Flag'].sum()
        dead_stock = product_data['Dead_Stock_Flag'].sum()
        action_required = product_data['Action_Required_Flag'].sum()
        
        # Estatísticas de lifecycle
        lifecycle_stats = product_data['Status_Lifecycle'].value_counts()
        
        file_size = os.path.getsize(output_path) / 1024  # KB
        
        summary = f"""
✅ EXPORTAÇÃO DE DADOS DE PRODUTOS CONCLUÍDA!

📁 **ARQUIVO GERADO**: {output_path}
📊 **TAMANHO**: {file_size:.1f} KB
🔢 **TOTAL DE PRODUTOS**: {total_products:,}

### 📈 CLASSIFICAÇÃO ABC:
{chr(10).join([f"- **Classe {k}**: {v} produtos ({v/total_products*100:.1f}%)" for k, v in abc_stats.items()])}

### 🎯 MATRIZ BCG:
{chr(10).join([f"- **{k.replace('_', ' ')}**: {v} produtos ({v/total_products*100:.1f}%)" for k, v in bcg_stats.items()])}

### 🚨 ALERTAS DE AÇÃO:
- **Slow Movers**: {slow_movers} produtos ({slow_movers/total_products*100:.1f}%)
- **Dead Stock**: {dead_stock} produtos ({dead_stock/total_products*100:.1f}%)
- **Ação Requerida**: {action_required} produtos ({action_required/total_products*100:.1f}%)

### 🔄 STATUS DO CICLO DE VIDA:
{chr(10).join([f"- **{k.replace('_', ' ')}**: {v} produtos" for k, v in lifecycle_stats.head().items()])}

### 📋 COLUNAS INCLUÍDAS NO CSV:
{chr(10).join([f"- {col}" for col in product_data.columns[:15]])}
{f"... e mais {len(product_data.columns)-15} colunas" if len(product_data.columns) > 15 else ""}

### 💡 PRÓXIMOS PASSOS SUGERIDOS:
1. **Filtrar produtos Classe A** com Dead_Stock_Flag = 1 para ação imediata
2. **Analisar Slow_Movers** da Classe B para possível liquidação
3. **Revisar Question_Marks** da matriz BCG para investimento ou descontinuação
4. **Usar Score_Geral** para priorizar ações de gestão de produtos

🎯 **Dados prontos para análise em Excel, Power BI ou outras ferramentas!**
"""
        
        return summary.strip() 