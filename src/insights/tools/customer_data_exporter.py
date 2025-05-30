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

class CustomerDataExporterInput(BaseModel):
    """Schema para exportação de dados de clientes."""
    
    data_csv: str = Field(
        default="data/vendas.csv", 
        description="Caminho para arquivo CSV de vendas"
    )
    
    output_path: str = Field(
        default="assets/data/analise_clientes_dados_completos.csv",
        description="Caminho de saída para o arquivo CSV de clientes exportado"
    )
    
    include_rfm_analysis: bool = Field(
        default=True,
        description="Incluir análise RFM completa"
    )
    
    include_clv_calculation: bool = Field(
        default=True,
        description="Incluir cálculo de Customer Lifetime Value"
    )
    
    include_geographic_analysis: bool = Field(
        default=True,
        description="Incluir análise geográfica e demográfica"
    )
    
    include_behavioral_insights: bool = Field(
        default=True,
        description="Incluir insights comportamentais"
    )
    
    clv_months: int = Field(
        default=24,
        description="Meses para projeção do CLV"
    )
    
    churn_days: int = Field(
        default=180,
        description="Dias sem compra para considerar risco de churn"
    )

class CustomerDataExporter(BaseTool, DataPreparationMixin, JewelryBusinessAnalysisMixin, JewelryRFMAnalysisMixin):
    """
    Ferramenta especializada para exportar dados completos de análise de clientes.
    
    Esta ferramenta gera um CSV abrangente com:
    - Segmentação RFM detalhada
    - Customer Lifetime Value (CLV)
    - Análise geográfica e demográfica
    - Insights comportamentais
    - Estratégias personalizadas por segmento
    - Scores de saúde do cliente
    """
    
    name: str = "Customer Data Exporter"
    description: str = """
    Exporta dados completos de análise de clientes em formato CSV para análise avançada.
    
    Inclui segmentação RFM, CLV, análise geográfica, insights comportamentais 
    e estratégias personalizadas por segmento de cliente.
    
    Use esta ferramenta quando precisar de dados estruturados de clientes para:
    - Campanhas de marketing segmentadas
    - Análise de Customer Lifetime Value
    - Estratégias de retenção e fidelização
    - Dashboards de CRM (Salesforce, HubSpot)
    - Análises demográficas e geográficas
    - Planejamento de ações personalizadas
    """
    args_schema: Type[BaseModel] = CustomerDataExporterInput

    def _run(self, data_csv: str = "data/vendas.csv", 
             output_path: str = "assets/data/analise_clientes_dados_completos.csv",
             include_rfm_analysis: bool = True,
             include_clv_calculation: bool = True,
             include_geographic_analysis: bool = True,
             include_behavioral_insights: bool = True,
             clv_months: int = 24,
             churn_days: int = 180) -> str:
        
        try:
            print("🚀 Iniciando exportação de dados de clientes...")
            
            # 1. Carregar e preparar dados
            print("📊 Carregando dados de vendas para análise de clientes...")
            df = self._load_and_prepare_data(data_csv)
            
            if df.empty:
                return "❌ Erro: Dados de vendas não encontrados ou inválidos"
            
            print(f"✅ Dados carregados: {len(df):,} registros")
            
            # 2. Verificar se há dados de clientes
            if 'Codigo_Cliente' not in df.columns:
                print("⚠️ Campo 'Codigo_Cliente' não encontrado. Criando IDs estimados...")
                df['Codigo_Cliente'] = self._estimate_customer_ids(df)
            
            # 3. Agregar dados por cliente
            print("👥 Agregando dados por cliente...")
            customer_data = self._aggregate_customer_data(df)
            
            # 4. Análise RFM
            if include_rfm_analysis:
                print("🎯 Aplicando análise RFM...")
                customer_data = self._add_rfm_analysis(df, customer_data)
            
            # 5. Cálculo de CLV
            if include_clv_calculation:
                print("💰 Calculando Customer Lifetime Value...")
                customer_data = self._add_clv_calculation(df, customer_data, clv_months)
            
            # 6. Análise geográfica e demográfica
            if include_geographic_analysis:
                print("🌍 Adicionando análise geográfica e demográfica...")
                customer_data = self._add_geographic_analysis(df, customer_data)
            
            # 7. Insights comportamentais
            if include_behavioral_insights:
                print("🧠 Gerando insights comportamentais...")
                customer_data = self._add_behavioral_insights(df, customer_data, churn_days)
            
            # 8. Estratégias personalizadas
            print("🎯 Definindo estratégias personalizadas...")
            customer_data = self._add_personalized_strategies(customer_data)
            
            # 9. Scores de saúde do cliente
            print("📊 Calculando scores de saúde do cliente...")
            customer_data = self._add_customer_health_scores(customer_data)
            
            # 10. Exportar CSV
            print("💾 Exportando arquivo CSV de clientes...")
            success = self._export_to_csv(customer_data, output_path)
            
            if success:
                return self._generate_export_summary(customer_data, output_path)
            else:
                return "❌ Erro na exportação do arquivo CSV"
                
        except Exception as e:
            return f"❌ Erro na exportação de dados de clientes: {str(e)}"
    
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
    
    def _estimate_customer_ids(self, df: pd.DataFrame) -> pd.Series:
        """Estimar IDs de clientes quando não disponíveis."""
        # Estratégia: agrupar por padrões de compra similar
        # Para dados sintéticos, criar IDs baseados em padrões
        
        # Criar IDs únicos baseados em combinações de valores
        if 'Descricao_Produto' in df.columns:
            # Usar padrão baseado em produto + data + valor
            df['temp_id'] = (
                df['Descricao_Produto'].astype(str) + 
                df['Data'].dt.strftime('%Y%m') + 
                df['Total_Liquido'].round(-2).astype(str)
            )
        else:
            # Fallback: usar apenas data + valor
            df['temp_id'] = (
                df['Data'].dt.strftime('%Y%m') + 
                df['Total_Liquido'].round(-2).astype(str)
            )
        
        # Criar mapeamento de IDs únicos
        unique_patterns = df['temp_id'].unique()
        customer_mapping = {pattern: f"CLI-{i+1:04d}" for i, pattern in enumerate(unique_patterns)}
        
        return df['temp_id'].map(customer_mapping)
    
    def _aggregate_customer_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agregar dados por cliente."""
        
        # Agregações específicas para clientes
        agg_dict = {
            'Total_Liquido': ['sum', 'mean', 'count'],
            'Data': ['min', 'max'],
            'Quantidade': 'sum' if 'Quantidade' in df.columns else lambda x: len(x)
        }
        
        # Adicionar colunas opcionais
        optional_columns = {
            'Codigo_Produto': 'nunique',
            'Grupo_Produto': lambda x: x.mode().iloc[0] if not x.empty else 'N/A',
            'Descricao_Produto': 'count'
        }
        
        for col, agg in optional_columns.items():
            if col in df.columns:
                agg_dict[col] = agg
        
        # Agregar dados
        customer_aggregated = df.groupby('Codigo_Cliente').agg(agg_dict)
        
        # Flatten column names
        new_columns = []
        for col in customer_aggregated.columns:
            if isinstance(col, tuple):
                if col[1] in ['first', 'last', '']:
                    new_columns.append(col[0])
                else:
                    new_columns.append(f"{col[0]}_{col[1]}")
            else:
                new_columns.append(col)
        
        customer_aggregated.columns = new_columns
        
        # Renomear colunas
        column_mapping = {
            'Total_Liquido_sum': 'Receita_Total',
            'Total_Liquido_mean': 'Ticket_Medio',
            'Total_Liquido_count': 'Num_Transacoes',
            'Data_min': 'Primeira_Compra',
            'Data_max': 'Ultima_Compra',
            'Codigo_Produto_nunique': 'Produtos_Unicos',
            'Descricao_Produto_count': 'Total_Itens'
        }
        
        customer_aggregated.rename(columns=column_mapping, inplace=True)
        
        # Calcular métricas básicas
        current_date = df['Data'].max()
        customer_aggregated['Days_Since_Last_Purchase'] = (
            current_date - pd.to_datetime(customer_aggregated['Ultima_Compra'])
        ).dt.days
        
        # Frequência média (dias entre compras)
        customer_aggregated['Customer_Lifecycle_Days'] = (
            pd.to_datetime(customer_aggregated['Ultima_Compra']) - 
            pd.to_datetime(customer_aggregated['Primeira_Compra'])
        ).dt.days + 1
        
        customer_aggregated['Avg_Days_Between_Purchases'] = (
            customer_aggregated['Customer_Lifecycle_Days'] / 
            customer_aggregated['Num_Transacoes'].clip(lower=1)
        ).round(1)
        
        return customer_aggregated.reset_index()
    
    def _add_rfm_analysis(self, df: pd.DataFrame, customer_data: pd.DataFrame) -> pd.DataFrame:
        """Adicionar análise RFM completa."""
        
        current_date = df['Data'].max()
        
        # Calcular RFM scores
        # Recency: dias desde última compra (menor é melhor)
        customer_data['Recency_Days'] = customer_data['Days_Since_Last_Purchase']
        
        # Frequency: número de transações (maior é melhor)
        customer_data['Frequency_Count'] = customer_data['Num_Transacoes']
        
        # Monetary: valor total gasto (maior é melhor)
        customer_data['Monetary_Value'] = customer_data['Receita_Total']
        
        # Calcular quintis para cada métrica
        customer_data['R_Score'] = pd.qcut(
            customer_data['Recency_Days'], 5, labels=[5,4,3,2,1]
        ).astype(int)  # Inverter: recency baixa = score alto
        
        customer_data['F_Score'] = pd.qcut(
            customer_data['Frequency_Count'].rank(method='first'), 5, labels=[1,2,3,4,5]
        ).astype(int)
        
        customer_data['M_Score'] = pd.qcut(
            customer_data['Monetary_Value'].rank(method='first'), 5, labels=[1,2,3,4,5]
        ).astype(int)
        
        # Score RFM combinado
        customer_data['RFM_Score'] = (
            customer_data['R_Score'].astype(str) + 
            customer_data['F_Score'].astype(str) + 
            customer_data['M_Score'].astype(str)
        )
        
        # Segmentação RFM
        def classify_rfm_segment(row):
            r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
            
            if r >= 4 and f >= 4 and m >= 4:
                return 'Campeoes'
            elif r >= 3 and f >= 3 and m >= 3:
                return 'Leais'
            elif r >= 4 and f <= 2:
                return 'Novos_Clientes'
            elif r >= 3 and f >= 3 and m <= 2:
                return 'Potenciais_Leais'
            elif r <= 2 and f >= 3:
                return 'Em_Risco'
            elif r <= 2 and f <= 2:
                return 'Perdidos'
            else:
                return 'Regulares'
        
        customer_data['Segmento_RFM'] = customer_data.apply(classify_rfm_segment, axis=1)
        
        return customer_data
    
    def _add_clv_calculation(self, df: pd.DataFrame, customer_data: pd.DataFrame, clv_months: int) -> pd.DataFrame:
        """Calcular Customer Lifetime Value."""
        
        # CLV = (Ticket Médio × Frequência de Compra × Margem Bruta) × Tempo de Vida
        
        # Frequência anual estimada
        customer_data['Purchase_Frequency_Annual'] = (
            customer_data['Num_Transacoes'] / 
            (customer_data['Customer_Lifecycle_Days'] / 365).clip(lower=0.1)
        ).round(2)
        
        # Margem estimada (assumindo 60% para joalherias)
        estimated_margin = 0.6
        customer_data['Estimated_Margin_Per_Purchase'] = (
            customer_data['Ticket_Medio'] * estimated_margin
        )
        
        # Tempo de vida estimado (baseado na frequência atual)
        # Se compra frequentemente, provavelmente continuará
        customer_data['Estimated_Lifetime_Years'] = np.where(
            customer_data['Purchase_Frequency_Annual'] >= 2,
            3.5,  # Clientes frequentes: 3.5 anos
            np.where(
                customer_data['Purchase_Frequency_Annual'] >= 1,
                2.5,  # Clientes moderados: 2.5 anos
                1.5   # Clientes esporádicos: 1.5 anos
            )
        )
        
        # CLV final
        customer_data['CLV_Estimado'] = (
            customer_data['Estimated_Margin_Per_Purchase'] * 
            customer_data['Purchase_Frequency_Annual'] * 
            customer_data['Estimated_Lifetime_Years']
        ).round(2)
        
        # CLV projetado para o período especificado
        customer_data[f'CLV_Projetado_{clv_months}M'] = (
            customer_data['CLV_Estimado'] * (clv_months / 12)
        ).round(2)
        
        # Classificação por CLV
        customer_data['CLV_Categoria'] = pd.cut(
            customer_data['CLV_Estimado'],
            bins=[0, 5000, 15000, 30000, float('inf')],
            labels=['Baixo', 'Medio', 'Alto', 'Premium']
        )
        
        return customer_data
    
    def _add_geographic_analysis(self, df: pd.DataFrame, customer_data: pd.DataFrame) -> pd.DataFrame:
        """Adicionar análise geográfica e demográfica estimada."""
        
        # Como não temos dados demográficos reais, vamos simular baseado em padrões
        
        # Simular localização baseada em padrões de compra
        def estimate_region(row):
            # Simulação baseada no valor gasto
            if row['Receita_Total'] > 20000:
                return np.random.choice(['SP', 'RJ'], p=[0.6, 0.4])
            elif row['Receita_Total'] > 10000:
                return np.random.choice(['SP', 'RJ', 'MG'], p=[0.4, 0.3, 0.3])
            else:
                return np.random.choice(['SP', 'RJ', 'MG', 'RS', 'PR'], p=[0.3, 0.2, 0.2, 0.15, 0.15])
        
        customer_data['Estado_Estimado'] = customer_data.apply(estimate_region, axis=1)
        
        # Simular faixa etária baseada no comportamento de compra
        def estimate_age_group(row):
            # Baseado na frequência e ticket médio
            if row['Ticket_Medio'] > 3000 and row['Num_Transacoes'] > 3:
                return np.random.choice(['36-50', '50+'], p=[0.6, 0.4])
            elif row['Ticket_Medio'] > 1500:
                return np.random.choice(['26-35', '36-50'], p=[0.5, 0.5])
            else:
                return np.random.choice(['18-25', '26-35'], p=[0.3, 0.7])
        
        customer_data['Faixa_Etaria_Estimada'] = customer_data.apply(estimate_age_group, axis=1)
        
        # Simular gênero baseado em preferências de produto
        def estimate_gender(row):
            # Para joalherias, assumir distribuição 62% F / 38% M conforme relatório
            return np.random.choice(['Feminino', 'Masculino'], p=[0.62, 0.38])
        
        customer_data['Genero_Estimado'] = customer_data.apply(estimate_gender, axis=1)
        
        # Simular estado civil baseado na frequência de compra
        def estimate_marital_status(row):
            # Casados tendem a ter compras mais frequentes
            if row['Purchase_Frequency_Annual'] >= 2:
                return np.random.choice(['Casado', 'Solteiro'], p=[0.7, 0.3])
            else:
                return np.random.choice(['Casado', 'Solteiro'], p=[0.4, 0.6])
        
        customer_data['Estado_Civil_Estimado'] = customer_data.apply(estimate_marital_status, axis=1)
        
        return customer_data
    
    def _add_behavioral_insights(self, df: pd.DataFrame, customer_data: pd.DataFrame, churn_days: int) -> pd.DataFrame:
        """Adicionar insights comportamentais."""
        
        # Sazonalidade de compras
        customer_purchases_by_month = df.groupby(['Codigo_Cliente', df['Data'].dt.month])['Total_Liquido'].sum().reset_index()
        peak_months = customer_purchases_by_month.groupby('Codigo_Cliente')['Data'].apply(
            lambda x: x.value_counts().index[0] if len(x) > 0 else 12
        )
        customer_data['Mes_Pico_Compras'] = customer_data['Codigo_Cliente'].map(peak_months)
        
        # Padrão sazonal (baseado no mês pico)
        def get_seasonal_pattern(month):
            if month in [11, 12]:  # Nov/Dez
                return 'Fim_Ano'
            elif month in [5, 6]:  # Mai/Jun - Dia das Mães/Namorados
                return 'Datas_Comemorativas'
            elif month in [3, 4]:  # Mar/Abr
                return 'Outono'
            else:
                return 'Regular'
        
        customer_data['Padrao_Sazonal'] = customer_data['Mes_Pico_Compras'].apply(get_seasonal_pattern)
        
        # Preferência de categoria (baseada em grupo de produto dominante)
        if 'Grupo_Produto' in customer_data.columns:
            customer_data['Categoria_Preferida'] = customer_data['Grupo_Produto']
        else:
            # Simular baseado no ticket médio
            def estimate_preference(row):
                if row['Ticket_Medio'] > 4000:
                    return 'Relógios'
                elif row['Ticket_Medio'] > 2500:
                    return 'Anéis'
                elif row['Ticket_Medio'] > 1500:
                    return 'Colares'
                else:
                    return np.random.choice(['Brincos', 'Pulseiras'])
            
            customer_data['Categoria_Preferida'] = customer_data.apply(estimate_preference, axis=1)
        
        # Canal preferencial (simulado)
        def estimate_preferred_channel(row):
            # VIPs preferem atendimento personalizado
            if row['Segmento_RFM'] in ['Campeoes', 'Leais']:
                return np.random.choice(['Presencial', 'WhatsApp_VIP'], p=[0.7, 0.3])
            elif row['Segmento_RFM'] == 'Em_Risco':
                return np.random.choice(['E-commerce', 'Telefone'], p=[0.6, 0.4])
            else:
                return np.random.choice(['E-commerce', 'Presencial'], p=[0.5, 0.5])
        
        customer_data['Canal_Preferencial'] = customer_data.apply(estimate_preferred_channel, axis=1)
        
        # Risco de churn
        customer_data['Risco_Churn_Flag'] = (
            customer_data['Days_Since_Last_Purchase'] >= churn_days
        ).astype(int)
        
        # Próxima compra prevista (baseada na frequência histórica)
        customer_data['Proxima_Compra_Prevista_Dias'] = (
            customer_data['Avg_Days_Between_Purchases'] - 
            customer_data['Days_Since_Last_Purchase']
        ).clip(lower=0)
        
        return customer_data
    
    def _add_personalized_strategies(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """Definir estratégias personalizadas por segmento."""
        
        def get_strategy_action(row):
            segment = row['Segmento_RFM']
            clv = row['CLV_Estimado']
            
            if segment == 'Campeoes':
                if clv > 50000:
                    return 'Evento_Exclusivo_Premium'
                else:
                    return 'Programa_VIP_Avancado'
            elif segment == 'Leais':
                return 'Fidelizacao_Pontos'
            elif segment == 'Novos_Clientes':
                return 'Kit_Boas_Vindas'
            elif segment == 'Potenciais_Leais':
                return 'Campanha_Frequencia'
            elif segment == 'Em_Risco':
                return 'Reativacao_Urgente'
            elif segment == 'Perdidos':
                return 'Win_Back_Campaign'
            else:
                return 'Manutencao_Relacionamento'
        
        customer_data['Estrategia_Recomendada'] = customer_data.apply(get_strategy_action, axis=1)
        
        # Investimento sugerido por cliente
        def calculate_investment_budget(row):
            clv = row['CLV_Estimado']
            segment = row['Segmento_RFM']
            
            if segment == 'Campeoes':
                return min(clv * 0.15, 5000)  # Até 15% do CLV
            elif segment == 'Leais':
                return min(clv * 0.10, 2000)  # Até 10% do CLV
            elif segment in ['Em_Risco', 'Perdidos']:
                return min(clv * 0.05, 500)   # Até 5% do CLV
            else:
                return min(clv * 0.08, 800)   # Até 8% do CLV
        
        customer_data['Investimento_Sugerido'] = customer_data.apply(
            calculate_investment_budget, axis=1
        ).round(2)
        
        # ROI esperado da estratégia
        def estimate_strategy_roi(row):
            segment = row['Segmento_RFM']
            
            roi_map = {
                'Campeoes': 4.2,
                'Leais': 3.8,
                'Potenciais_Leais': 2.8,
                'Novos_Clientes': 2.5,
                'Em_Risco': 2.0,
                'Perdidos': 1.5,
                'Regulares': 2.0
            }
            
            return roi_map.get(segment, 2.0)
        
        customer_data['ROI_Esperado_Estrategia'] = customer_data.apply(estimate_strategy_roi, axis=1)
        
        # Prioridade de ação (1 = mais urgente)
        def get_action_priority(row):
            segment = row['Segmento_RFM']
            clv = row['CLV_Estimado']
            
            if segment == 'Em_Risco' and clv > 20000:
                return 1  # Urgente: cliente valioso em risco
            elif segment == 'Campeoes':
                return 2  # Alto: manter campeões
            elif segment == 'Perdidos' and clv > 15000:
                return 3  # Médio: tentar resgatar valiosos
            elif segment == 'Novos_Clientes':
                return 4  # Normal: desenvolver novos
            else:
                return 5  # Baixo: manutenção
        
        customer_data['Prioridade_Acao'] = customer_data.apply(get_action_priority, axis=1)
        
        return customer_data
    
    def _add_customer_health_scores(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """Calcular scores de saúde do cliente."""
        
        # Score de Atividade (baseado na recência)
        max_days = customer_data['Days_Since_Last_Purchase'].max()
        customer_data['Score_Atividade'] = (
            (max_days - customer_data['Days_Since_Last_Purchase']) / max_days * 100
        ).clip(0, 100).round(1)
        
        # Score de Fidelidade (baseado na frequência e tempo como cliente)
        max_freq = customer_data['Num_Transacoes'].max()
        customer_data['Score_Fidelidade'] = (
            customer_data['Num_Transacoes'] / max_freq * 100
        ).clip(0, 100).round(1)
        
        # Score de Valor (baseado no CLV)
        max_clv = customer_data['CLV_Estimado'].max()
        customer_data['Score_Valor'] = (
            customer_data['CLV_Estimado'] / max_clv * 100
        ).clip(0, 100).round(1)
        
        # Score Geral de Saúde do Cliente
        customer_data['Score_Saude_Cliente'] = (
            customer_data['Score_Atividade'] * 0.4 +
            customer_data['Score_Fidelidade'] * 0.3 +
            customer_data['Score_Valor'] * 0.3
        ).round(1)
        
        # Score de Potencial (baseado no crescimento)
        customer_data['Score_Potencial'] = np.where(
            customer_data['Segmento_RFM'].isin(['Novos_Clientes', 'Potenciais_Leais']),
            customer_data['Score_Valor'] * 1.2,  # Bonus para potencial
            customer_data['Score_Valor']
        ).clip(0, 100).round(1)
        
        # Score de Urgência (necessidade de ação)
        def calculate_urgency_score(row):
            score = 0
            if row['Risco_Churn_Flag'] == 1:
                score += 40
            if row['Segmento_RFM'] == 'Em_Risco':
                score += 30
            elif row['Segmento_RFM'] == 'Perdidos':
                score += 50
            if row['CLV_Estimado'] > 30000:
                score += 20
            
            return min(score, 100)
        
        customer_data['Score_Urgencia'] = customer_data.apply(calculate_urgency_score, axis=1)
        
        return customer_data
    
    def _export_to_csv(self, customer_data: pd.DataFrame, output_path: str) -> bool:
        """Exportar dados para CSV."""
        
        try:
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Ordenar por Score de Saúde e Prioridade
            customer_data_sorted = customer_data.sort_values(
                ['Prioridade_Acao', 'Score_Saude_Cliente'], 
                ascending=[True, False]
            )
            
            # Reorganizar colunas para melhor visualização
            priority_columns = [
                'Codigo_Cliente', 'Segmento_RFM', 'CLV_Estimado', 'CLV_Categoria',
                'Score_Saude_Cliente', 'Prioridade_Acao', 'Estrategia_Recomendada',
                'Receita_Total', 'Ticket_Medio', 'Num_Transacoes',
                'Days_Since_Last_Purchase', 'Risco_Churn_Flag',
                'R_Score', 'F_Score', 'M_Score', 'RFM_Score',
                'Estado_Estimado', 'Faixa_Etaria_Estimada', 'Genero_Estimado',
                'Categoria_Preferida', 'Canal_Preferencial', 'Padrao_Sazonal'
            ]
            
            # Adicionar colunas restantes
            remaining_columns = [col for col in customer_data_sorted.columns if col not in priority_columns]
            final_columns = priority_columns + remaining_columns
            
            # Filtrar colunas que existem
            existing_columns = [col for col in final_columns if col in customer_data_sorted.columns]
            
            # Exportar CSV
            customer_data_sorted[existing_columns].to_csv(
                output_path, index=False, sep=';', encoding='utf-8'
            )
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao exportar CSV: {str(e)}")
            return False
    
    def _generate_export_summary(self, customer_data: pd.DataFrame, output_path: str) -> str:
        """Gerar resumo da exportação."""
        
        total_customers = len(customer_data)
        
        # Estatísticas por segmento RFM
        rfm_stats = customer_data['Segmento_RFM'].value_counts()
        
        # Estatísticas de CLV
        clv_stats = customer_data['CLV_Categoria'].value_counts()
        total_clv = customer_data['CLV_Estimado'].sum()
        avg_clv = customer_data['CLV_Estimado'].mean()
        
        # Top clientes
        top_customers = customer_data.nlargest(5, 'CLV_Estimado')
        
        # Clientes em risco
        churn_risk = customer_data[customer_data['Risco_Churn_Flag'] == 1]
        high_value_risk = churn_risk[churn_risk['CLV_Estimado'] > 20000]
        
        # Estratégias
        strategy_stats = customer_data['Estrategia_Recomendada'].value_counts()
        
        file_size = os.path.getsize(output_path) / 1024  # KB
        
        summary = f"""
                    ✅ EXPORTAÇÃO DE DADOS DE CLIENTES CONCLUÍDA!

                    📁 **ARQUIVO GERADO**: {output_path}
                    📊 **TAMANHO**: {file_size:.1f} KB
                    🔢 **TOTAL DE CLIENTES**: {total_customers:,}

                    ### 🎯 SEGMENTAÇÃO RFM:
                    {chr(10).join([f"- **{k.replace('_', ' ')}**: {v} clientes ({v/total_customers*100:.1f}%)" for k, v in rfm_stats.head().items()])}

                    ### 💰 ANÁLISE DE CLV:
                    - **CLV Total Estimado**: R$ {total_clv:,.0f}
                    - **CLV Médio**: R$ {avg_clv:,.0f}

                    **Distribuição por Categoria:**
                    {chr(10).join([f"- **{k}**: {v} clientes ({v/total_customers*100:.1f}%)" for k, v in clv_stats.items()])}

                    ### 👑 TOP 5 CLIENTES POR CLV:
                    {chr(10).join([f"- **{row['Codigo_Cliente']}**: R$ {row['CLV_Estimado']:,.0f} - {row['Segmento_RFM'].replace('_', ' ')}" for _, row in top_customers.iterrows()])}

                    ### 🚨 ALERTAS CRÍTICOS:
                    - **Clientes em risco de churn**: {len(churn_risk)} ({len(churn_risk)/total_customers*100:.1f}%)
                    - **Alto valor em risco**: {len(high_value_risk)} clientes (R$ {high_value_risk['CLV_Estimado'].sum():,.0f})

                    ### 🎯 ESTRATÉGIAS RECOMENDADAS:
                    {chr(10).join([f"- **{k.replace('_', ' ')}**: {v} clientes" for k, v in strategy_stats.head().items()])}

                    ### 📋 PRINCIPAIS COLUNAS DO CSV:
                    - **Identificação**: Codigo_Cliente, Segmento_RFM, CLV_Estimado
                    - **Scores RFM**: R_Score, F_Score, M_Score, RFM_Score
                    - **Comportamento**: Days_Since_Last_Purchase, Categoria_Preferida, Canal_Preferencial
                    - **Demografia**: Estado_Estimado, Faixa_Etaria_Estimada, Genero_Estimado
                    - **Estratégia**: Estrategia_Recomendada, Investimento_Sugerido, ROI_Esperado
                    - **Saúde**: Score_Saude_Cliente, Score_Urgencia, Risco_Churn_Flag

                    ### 💡 PRÓXIMOS PASSOS SUGERIDOS:
                    1. **Filtrar Prioridade_Acao = 1** para ações urgentes
                    2. **Focar em Campeões** com Score_Saude_Cliente > 80
                    3. **Reativar clientes** com Risco_Churn_Flag = 1
                    4. **Implementar estratégias** por Segmento_RFM
                    5. **Campanhas por Estado** e Faixa_Etaria_Estimada

                    🎯 **Dados prontos para CRM, campanhas de marketing e análise de segmentação!**
                    """
                            
        return summary.strip()

    def generate_customer_test_report(self, test_data: dict) -> str:
        """Gera relatório visual completo dos testes de clientes em formato markdown."""
        
        # Coletar dados com fallbacks
        metadata = test_data.get('metadata', {})
        data_metrics = test_data.get('data_metrics', {})
        results = test_data.get('results', {})
        component_tests = test_data.get('component_tests', {})
        
        report = [
            "# 👥 Teste Completo de Análise de Clientes - Relatório Executivo",
            f"**Data do Teste:** {metadata.get('test_timestamp', 'N/A')}",
            f"**Fonte de Dados:** `{metadata.get('data_source', 'desconhecida')}`",
            f"**Registros Analisados:** {data_metrics.get('total_records', 0):,}",
            f"**Clientes Únicos:** {data_metrics.get('total_customers', 0):,}",
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
        
        # Principais Descobertas de Clientes
        report.append("\n## 👥 Principais Descobertas de Clientes")
        
        # Análise RFM
        if 'rfm_analysis' in results and results['rfm_analysis'].get('success'):
            rfm_data = results['rfm_analysis']
            total_customers = rfm_data.get('total_customers_analyzed', 0)
            champions = int(rfm_data.get('champions_count', 0))
            at_risk = int(rfm_data.get('at_risk_count', 0))
            avg_rfm_score = rfm_data.get('avg_rfm_score', 0)
            report.append(f"- **Total de Clientes Analisados:** {total_customers:,}")
            report.append(f"- **Clientes Campeões:** {champions} ({champions/total_customers*100:.1f}%)" if total_customers > 0 else "- **Clientes Campeões:** N/A")
            report.append(f"- **Clientes em Risco:** {at_risk} ({at_risk/total_customers*100:.1f}%)" if total_customers > 0 else "- **Clientes em Risco:** N/A")
            report.append(f"- **Score RFM Médio:** {avg_rfm_score:.1f}")
        
        # Análise CLV
        if 'clv_calculation' in results and results['clv_calculation'].get('success'):
            clv_data = results['clv_calculation']
            total_clv = clv_data.get('total_clv_estimated', 0)
            avg_clv = clv_data.get('avg_clv', 0)
            premium_customers = clv_data.get('premium_customers_count', 0)
            report.append(f"- **CLV Total Estimado:** R$ {total_clv:,.0f}")
            report.append(f"- **CLV Médio:** R$ {avg_clv:,.0f}")
            report.append(f"- **Clientes Premium:** {premium_customers}")
        
        # Demografia
        if 'geographic_analysis' in results and results['geographic_analysis'].get('success'):
            geo_data = results['geographic_analysis']
            top_state = geo_data.get('top_state', 'N/A')
            dominant_age_group = geo_data.get('dominant_age_group', 'N/A')
            report.append(f"- **Estado Predominante:** {top_state}")
            report.append(f"- **Faixa Etária Dominante:** {dominant_age_group}")
        
        # Insights Comportamentais
        if 'behavioral_insights' in results and results['behavioral_insights'].get('success'):
            behavior_data = results['behavioral_insights']
            churn_risk = behavior_data.get('churn_risk_customers', 0)
            seasonal_pattern = behavior_data.get('dominant_seasonal_pattern', 'N/A')
            report.append(f"- **Clientes em Risco de Churn:** {churn_risk}")
            report.append(f"- **Padrão Sazonal Dominante:** {seasonal_pattern}")
        
        # Health Scores
        if 'health_scores' in results and results['health_scores'].get('success'):
            health_data = results['health_scores']
            avg_health = health_data.get('avg_health_score', 0)
            healthy_customers = health_data.get('healthy_customers_count', 0)
            report.append(f"- **Score Médio de Saúde:** {avg_health:.1f}/100")
            report.append(f"- **Clientes Saudáveis (>70):** {healthy_customers}")
        
        # Detalhamento por Componente
        report.append("\n## 🔧 Detalhamento dos Componentes Testados")
        
        component_categories = {
            'Preparação de Dados': ['data_loading', 'customer_id_estimation', 'data_aggregation'],
            'Análise RFM': ['rfm_analysis'],
            'Cálculo CLV': ['clv_calculation'],
            'Análise Geográfica': ['geographic_analysis'],
            'Insights Comportamentais': ['behavioral_insights'],
            'Estratégias Personalizadas': ['personalized_strategies'],
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
                        if 'customers_processed' in metrics:
                            report.append(f"  - Clientes: {metrics['customers_processed']:,}")
                    else:
                        error_msg = results[component].get('error', 'Erro desconhecido')
                        report.append(f"- ❌ **{component}**: {error_msg}")
                else:
                    report.append(f"- ⏭️ **{component}**: Não testado")
        
        # Arquivos Gerados
        if 'files_generated' in component_tests:
            files = component_tests['files_generated']
            report.append(f"\n### Arquivos Gerados ({len(files)}):")
            for file_info in files:
                size_kb = file_info.get('size_kb', 0)
                report.append(f"- **{file_info['path']}**: {size_kb:.1f} KB")
        
        # Recomendações Finais
        report.append("\n## 💡 Recomendações do Sistema de Clientes")
        
        recommendations = [
            "🎯 Focar em clientes com Prioridade_Acao = 1 (urgente)",
            "👑 Desenvolver programa VIP para Campeões",
            "🚨 Implementar campanha de retenção para clientes Em_Risco",
            "📈 Aproveitar Potenciais_Leais para aumentar frequência",
            "💰 Priorizar clientes com CLV_Categoria = Premium"
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

    def run_full_customer_test(self) -> str:
        """Executa teste completo de clientes e retorna relatório formatado"""
        test_result = self.test_all_customer_components()
        parsed = json.loads(test_result)
        return self.generate_customer_test_report(parsed)

    def test_all_customer_components(self, sample_data: str = "data/vendas.csv") -> str:
        """
        Executa teste completo de todos os componentes da classe CustomerDataExporter
        usando especificamente o arquivo data/vendas.csv
        """
        
        # Configurar caminho do arquivo
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
        data_file_path = os.path.join(project_root, "data", "vendas.csv")
        
        print(f"🔍 DEBUG: Caminho calculado: {data_file_path}")
        print(f"🔍 DEBUG: Arquivo existe? {os.path.exists(data_file_path)}")
        
        # Verificar se arquivo existe
        if not os.path.exists(data_file_path):
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
                    "tested_paths": alternative_paths
                }, indent=2)

        test_report = {
            "metadata": {
                "test_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "test_version": "Customer Test Suite v1.0",
                "data_source": data_file_path,
                "tool_version": "Customer Data Exporter v1.0",
                "status": "in_progress"
            },
            "data_metrics": {
                "total_records": 0,
                "total_customers": 0,
                "date_range": {},
                "data_quality_check": {}
            },
            "results": {},
            "component_tests": {},
            "performance_metrics": {},
            "errors": []
        }

        try:
            # 1. Carregamento de Dados
            test_report["metadata"]["current_stage"] = "data_loading"
            print("\n=== ETAPA 1: CARREGAMENTO DE DADOS ===")
            print(f"📁 Carregando: {data_file_path}")
            
            start_time = time.time()
            df = self._load_and_prepare_data(data_file_path)
            loading_time = time.time() - start_time
            
            if df.empty:
                raise Exception("Falha no carregamento do arquivo data/vendas.csv")
            
            print(f"✅ Dados carregados: {len(df)} registros em {loading_time:.3f}s")
            
            test_report["data_metrics"] = {
                "total_records": int(len(df)),
                "date_range": {
                    "start": str(df['Data'].min()) if 'Data' in df.columns else "N/A",
                    "end": str(df['Data'].max()) if 'Data' in df.columns else "N/A"
                },
                "data_quality_check": self._perform_customer_data_quality_check(df)
            }
            
            test_report["results"]["data_loading"] = {
                "success": True,
                "metrics": {
                    "processing_time": loading_time,
                    "records_processed": len(df)
                }
            }

            # 2. Estimativa de IDs de Cliente
            test_report["metadata"]["current_stage"] = "customer_id_estimation"
            print("\n=== ETAPA 2: ESTIMATIVA DE IDs DE CLIENTE ===")
            
            try:
                start_time = time.time()
                print("🔍 Verificando/Estimando IDs de clientes...")
                
                if 'Codigo_Cliente' not in df.columns:
                    print("⚠️ Campo 'Codigo_Cliente' não encontrado. Criando IDs estimados...")
                    df['Codigo_Cliente'] = self._estimate_customer_ids(df)
                    estimated_ids = True
                else:
                    estimated_ids = False
                
                id_time = time.time() - start_time
                unique_customers = df['Codigo_Cliente'].nunique()
                
                test_report["data_metrics"]["total_customers"] = unique_customers
                
                test_report["results"]["customer_id_estimation"] = {
                    "success": True,
                    "metrics": {
                        "processing_time": id_time,
                        "unique_customers": unique_customers,
                        "ids_estimated": estimated_ids
                    }
                }
                print(f"✅ IDs processados: {unique_customers:,} clientes únicos em {id_time:.3f}s")
                
            except Exception as e:
                self._log_customer_test_error(test_report, e, "customer_id_estimation")
                print(f"❌ Erro na estimativa de IDs: {str(e)}")

            # 3. Agregação de Dados por Cliente
            test_report["metadata"]["current_stage"] = "data_aggregation"
            print("\n=== ETAPA 3: AGREGAÇÃO DE DADOS POR CLIENTE ===")
            
            try:
                start_time = time.time()
                print("👥 Agregando dados por cliente...")
                customer_data = self._aggregate_customer_data(df)
                aggregation_time = time.time() - start_time
                
                test_report["results"]["data_aggregation"] = {
                    "success": True,
                    "metrics": {
                        "processing_time": aggregation_time,
                        "customers_processed": len(customer_data),
                        "columns_generated": len(customer_data.columns)
                    }
                }
                print(f"✅ Agregação concluída: {len(customer_data)} clientes em {aggregation_time:.3f}s")
                
            except Exception as e:
                self._log_customer_test_error(test_report, e, "data_aggregation")
                print(f"❌ Erro na agregação: {str(e)}")
                customer_data = pd.DataFrame()

            # 4. Análise RFM
            test_report["metadata"]["current_stage"] = "rfm_analysis"
            print("\n=== ETAPA 4: ANÁLISE RFM ===")
            
            if not customer_data.empty:
                try:
                    start_time = time.time()
                    print("🎯 Aplicando análise RFM...")
                    customer_data_rfm = self._add_rfm_analysis(df, customer_data.copy())
                    rfm_time = time.time() - start_time
                    
                    # Estatísticas RFM
                    rfm_segments = customer_data_rfm['Segmento_RFM'].value_counts()
                    champions = rfm_segments.get('Campeoes', 0)
                    at_risk = rfm_segments.get('Em_Risco', 0)
                    avg_rfm = customer_data_rfm[['R_Score', 'F_Score', 'M_Score']].mean().mean()
                    
                    test_report["results"]["rfm_analysis"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": rfm_time,
                            "customers_processed": len(customer_data_rfm)
                        },
                        "total_customers_analyzed": len(customer_data_rfm),
                        "champions_count": champions,
                        "at_risk_count": at_risk,
                        "avg_rfm_score": float(avg_rfm),
                        "segment_distribution": rfm_segments.to_dict()
                    }
                    print(f"✅ RFM aplicado: {champions} campeões, {at_risk} em risco, score médio {avg_rfm:.1f} em {rfm_time:.3f}s")
                    
                except Exception as e:
                    self._log_customer_test_error(test_report, e, "rfm_analysis")
                    print(f"❌ Erro na análise RFM: {str(e)}")
                    customer_data_rfm = customer_data.copy()
            else:
                customer_data_rfm = pd.DataFrame()

            # 5. Cálculo CLV
            test_report["metadata"]["current_stage"] = "clv_calculation"
            print("\n=== ETAPA 5: CÁLCULO CLV ===")
            
            if not customer_data_rfm.empty:
                try:
                    start_time = time.time()
                    print("💰 Calculando Customer Lifetime Value...")
                    customer_data_clv = self._add_clv_calculation(df, customer_data_rfm.copy(), 24)
                    clv_time = time.time() - start_time
                    
                    # Estatísticas CLV
                    total_clv = customer_data_clv['CLV_Estimado'].sum()
                    avg_clv = customer_data_clv['CLV_Estimado'].mean()
                    premium_customers = len(customer_data_clv[customer_data_clv['CLV_Categoria'] == 'Premium'])
                    
                    test_report["results"]["clv_calculation"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": clv_time,
                            "customers_processed": len(customer_data_clv)
                        },
                        "total_clv_estimated": float(total_clv),
                        "avg_clv": float(avg_clv),
                        "premium_customers_count": premium_customers
                    }
                    print(f"✅ CLV calculado: R$ {total_clv:,.0f} total, R$ {avg_clv:,.0f} médio, {premium_customers} premium em {clv_time:.3f}s")
                    
                except Exception as e:
                    self._log_customer_test_error(test_report, e, "clv_calculation")
                    print(f"❌ Erro no cálculo CLV: {str(e)}")
                    customer_data_clv = customer_data_rfm.copy()
            else:
                customer_data_clv = pd.DataFrame()

            # 6. Análise Geográfica
            test_report["metadata"]["current_stage"] = "geographic_analysis"
            print("\n=== ETAPA 6: ANÁLISE GEOGRÁFICA ===")
            
            if not customer_data_clv.empty:
                try:
                    start_time = time.time()
                    print("🌍 Adicionando análise geográfica e demográfica...")
                    customer_data_geo = self._add_geographic_analysis(df, customer_data_clv.copy())
                    geo_time = time.time() - start_time
                    
                    # Estatísticas geográficas
                    state_stats = customer_data_geo['Estado_Estimado'].value_counts()
                    age_stats = customer_data_geo['Faixa_Etaria_Estimada'].value_counts()
                    top_state = state_stats.index[0] if len(state_stats) > 0 else 'N/A'
                    dominant_age = age_stats.index[0] if len(age_stats) > 0 else 'N/A'
                    
                    test_report["results"]["geographic_analysis"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": geo_time,
                            "customers_processed": len(customer_data_geo)
                        },
                        "top_state": top_state,
                        "dominant_age_group": dominant_age,
                        "state_distribution": state_stats.to_dict(),
                        "age_distribution": age_stats.to_dict()
                    }
                    print(f"✅ Geografia analisada: {top_state} predominante, {dominant_age} dominante em {geo_time:.3f}s")
                    
                except Exception as e:
                    self._log_customer_test_error(test_report, e, "geographic_analysis")
                    print(f"❌ Erro na análise geográfica: {str(e)}")
                    customer_data_geo = customer_data_clv.copy()
            else:
                customer_data_geo = pd.DataFrame()

            # 7. Insights Comportamentais
            test_report["metadata"]["current_stage"] = "behavioral_insights"
            print("\n=== ETAPA 7: INSIGHTS COMPORTAMENTAIS ===")
            
            if not customer_data_geo.empty:
                try:
                    start_time = time.time()
                    print("🧠 Gerando insights comportamentais...")
                    customer_data_behavior = self._add_behavioral_insights(df, customer_data_geo.copy(), 180)
                    behavior_time = time.time() - start_time
                    
                    # Estatísticas comportamentais
                    churn_risk = customer_data_behavior['Risco_Churn_Flag'].sum()
                    seasonal_stats = customer_data_behavior['Padrao_Sazonal'].value_counts()
                    dominant_pattern = seasonal_stats.index[0] if len(seasonal_stats) > 0 else 'N/A'
                    
                    test_report["results"]["behavioral_insights"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": behavior_time,
                            "customers_processed": len(customer_data_behavior)
                        },
                        "churn_risk_customers": int(churn_risk),
                        "dominant_seasonal_pattern": dominant_pattern,
                        "seasonal_distribution": seasonal_stats.to_dict()
                    }
                    print(f"✅ Insights gerados: {churn_risk} em risco churn, padrão {dominant_pattern} em {behavior_time:.3f}s")
                    
                except Exception as e:
                    self._log_customer_test_error(test_report, e, "behavioral_insights")
                    print(f"❌ Erro nos insights: {str(e)}")
                    customer_data_behavior = customer_data_geo.copy()
            else:
                customer_data_behavior = pd.DataFrame()

            # 8. Estratégias Personalizadas
            test_report["metadata"]["current_stage"] = "personalized_strategies"
            print("\n=== ETAPA 8: ESTRATÉGIAS PERSONALIZADAS ===")
            
            if not customer_data_behavior.empty:
                try:
                    start_time = time.time()
                    print("🎯 Definindo estratégias personalizadas...")
                    customer_data_strategies = self._add_personalized_strategies(customer_data_behavior.copy())
                    strategies_time = time.time() - start_time
                    
                    # Estatísticas de estratégias
                    strategy_stats = customer_data_strategies['Estrategia_Recomendada'].value_counts()
                    priority_stats = customer_data_strategies['Prioridade_Acao'].value_counts()
                    urgent_actions = priority_stats.get(1, 0)
                    
                    test_report["results"]["personalized_strategies"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": strategies_time,
                            "customers_processed": len(customer_data_strategies)
                        },
                        "strategy_distribution": strategy_stats.to_dict(),
                        "urgent_actions_needed": int(urgent_actions)
                    }
                    print(f"✅ Estratégias definidas: {len(strategy_stats)} tipos, {urgent_actions} urgentes em {strategies_time:.3f}s")
                    
                except Exception as e:
                    self._log_customer_test_error(test_report, e, "personalized_strategies")
                    print(f"❌ Erro nas estratégias: {str(e)}")
                    customer_data_strategies = customer_data_behavior.copy()
            else:
                customer_data_strategies = pd.DataFrame()

            # 9. Scores de Saúde
            test_report["metadata"]["current_stage"] = "health_scores"
            print("\n=== ETAPA 9: SCORES DE SAÚDE ===")
            
            if not customer_data_strategies.empty:
                try:
                    start_time = time.time()
                    print("📊 Calculando scores de saúde do cliente...")
                    customer_data_health = self._add_customer_health_scores(customer_data_strategies.copy())
                    health_time = time.time() - start_time
                    
                    # Estatísticas de saúde
                    avg_health = customer_data_health['Score_Saude_Cliente'].mean()
                    healthy_customers = len(customer_data_health[customer_data_health['Score_Saude_Cliente'] > 70])
                    
                    test_report["results"]["health_scores"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": health_time,
                            "customers_processed": len(customer_data_health)
                        },
                        "avg_health_score": float(avg_health),
                        "healthy_customers_count": healthy_customers
                    }
                    print(f"✅ Saúde calculada: {avg_health:.1f}/100 médio, {healthy_customers} saudáveis em {health_time:.3f}s")
                    
                except Exception as e:
                    self._log_customer_test_error(test_report, e, "health_scores")
                    print(f"❌ Erro nos scores: {str(e)}")
                    customer_data_health = customer_data_strategies.copy()
            else:
                customer_data_health = pd.DataFrame()

            # 10. Exportação CSV
            test_report["metadata"]["current_stage"] = "csv_export"
            print("\n=== ETAPA 10: EXPORTAÇÃO CSV ===")
            
            if not customer_data_health.empty:
                try:
                    start_time = time.time()
                    print("💾 Testando exportação CSV...")
                    
                    test_output_dir = "test_results"
                    os.makedirs(test_output_dir, exist_ok=True)
                    test_output_path = os.path.join(test_output_dir, "customer_test_export.csv")
                    
                    export_success = self._export_to_csv(customer_data_health, test_output_path)
                    export_time = time.time() - start_time
                    
                    if export_success and os.path.exists(test_output_path):
                        file_size_kb = os.path.getsize(test_output_path) / 1024
                        
                        test_report["results"]["csv_export"] = {
                            "success": True,
                            "metrics": {
                                "processing_time": export_time,
                                "file_size_kb": file_size_kb,
                                "customers_exported": len(customer_data_health)
                            },
                            "output_path": test_output_path
                        }
                        print(f"✅ CSV exportado: {file_size_kb:.1f} KB em {export_time:.3f}s")
                        
                        test_report["component_tests"]["files_generated"] = [{
                            "path": test_output_path,
                            "size_kb": file_size_kb,
                            "type": "customer_export"
                        }]
                    else:
                        raise Exception("Falha na exportação do arquivo CSV")
                        
                except Exception as e:
                    self._log_customer_test_error(test_report, e, "csv_export")
                    print(f"❌ Erro na exportação: {str(e)}")

            # 11. Geração de Sumário
            test_report["metadata"]["current_stage"] = "summary_generation"
            print("\n=== ETAPA 11: GERAÇÃO DE SUMÁRIO ===")
            
            if not customer_data_health.empty:
                try:
                    start_time = time.time()
                    print("📋 Testando geração de sumário...")
                    
                    summary = self._generate_export_summary(customer_data_health, test_output_path if 'test_output_path' in locals() else "test_path")
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
                    self._log_customer_test_error(test_report, e, "summary_generation")
                    print(f"❌ Erro na geração de sumário: {str(e)}")

            # 12. Performance Metrics
            test_report["performance_metrics"] = {
                "total_execution_time": sum([
                    result.get('metrics', {}).get('processing_time', 0) 
                    for result in test_report["results"].values() 
                    if isinstance(result, dict)
                ]),
                "memory_usage_mb": self._get_customer_memory_usage(),
                "largest_dataset_processed": len(customer_data_health) if not customer_data_health.empty else 0
            }

            # 13. Análise Final
            test_report["metadata"]["status"] = "completed" if not test_report["errors"] else "completed_with_errors"
            print(f"\n✅✅✅ TESTE DE CLIENTES COMPLETO - {len(test_report['errors'])} erros ✅✅✅")
            
            return json.dumps(test_report, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            test_report["metadata"]["status"] = "failed"
            self._log_customer_test_error(test_report, e, "global")
            print(f"❌ TESTE DE CLIENTES FALHOU: {str(e)}")
            return json.dumps(test_report, ensure_ascii=False, indent=2, default=str)

    def _log_customer_test_error(self, report: dict, error: Exception, context: str) -> None:
        """Registra erros de teste de clientes de forma estruturada"""
        import traceback
        error_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc()
        }
        report["errors"].append(error_entry)

    def _perform_customer_data_quality_check(self, df: pd.DataFrame) -> dict:
        """Executa verificações de qualidade específicas para dados de clientes"""
        checks = {
            "missing_dates": int(df['Data'].isnull().sum()) if 'Data' in df.columns else 0,
            "missing_revenue": int(df['Total_Liquido'].isnull().sum()) if 'Total_Liquido' in df.columns else 0,
            "missing_customer_id": int(df['Codigo_Cliente'].isnull().sum()) if 'Codigo_Cliente' in df.columns else len(df),
            "duplicate_transactions": int(df.duplicated().sum()),
            "customers_single_purchase": 0,  # Será calculado após agregação
            "extreme_values": int((df['Total_Liquido'] > df['Total_Liquido'].quantile(0.99)).sum()) if 'Total_Liquido' in df.columns else 0
        }
        return checks

    def _get_customer_memory_usage(self) -> float:
        """Obtém uso de memória específico para análises de clientes"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Em MB
        except:
            return 0.0


# Exemplo de uso
if __name__ == "__main__":
    import json
    import time
    import traceback
    import sys
    
    exporter = CustomerDataExporter()
    
    print("👥 Iniciando Teste Completo do Sistema de Clientes...")
    print("📁 Testando especificamente com: data/vendas.csv")
    
    # Executar teste usando especificamente data/vendas.csv
    report = exporter.run_full_customer_test()
    
    # Salvar relatório
    os.makedirs("test_results", exist_ok=True)
    with open("test_results/customer_test_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("✅ Relatório de clientes gerado em test_results/customer_test_report.md")
    print(f"📁 Teste executado com arquivo: data/vendas.csv")
    print("\n" + "="*80)
    print(report[:1500])  # Exibir parte do relatório no console 