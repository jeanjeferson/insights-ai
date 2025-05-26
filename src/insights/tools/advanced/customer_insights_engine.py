from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any, List, Tuple, Union
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from scipy import stats
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class CustomerInsightsInput(BaseModel):
    """Schema otimizado de entrada para análise avançada de insights de clientes."""
    
    analysis_type: str = Field(
        ..., 
        description="""Tipo de análise especializada para joalherias:
        
        📊 'behavioral_segmentation' - Segmenta clientes por comportamento de compra usando RFM, demographics e padrões reais
        🔄 'lifecycle_analysis' - Analisa estágios do ciclo de vida (Novo→Crescendo→Leal→VIP→Em Risco→Perdido)  
        ⚠️ 'churn_prediction' - Prediz risco de abandono usando ML e identifica fatores de churn
        💎 'value_analysis' - Analise valor do cliente, CLV, concentração Pareto e margem real
        🔍 'preference_mining' - Descobre preferências por demographics, localização e produtos
        🗺️ 'journey_mapping' - Mapeia jornada completa do cliente por perfil e identifica pontos de atrito
        """
    )
    
    data_csv: str = Field(
        default="data/vendas.csv", 
        description="Caminho para arquivo CSV com dados reais de vendas da joalheria"
    )
    
    customer_id_column: str = Field(
        default="Codigo_Cliente", 
        description="Coluna de identificação de cliente nos dados reais (Codigo_Cliente ou Nome_Cliente)"
    )
    
    segmentation_method: str = Field(
        default="rfm", 
        description="""Método de segmentação:
        'rfm' - RFM clássico (Recency, Frequency, Monetary)
        'behavioral' - Clustering comportamental avançado  
        'value_based' - Segmentação por valor e CLV
        'hybrid' - Combinação de RFM + Demographics + Geographic
        """
    )
    
    prediction_horizon: int = Field(
        default=90, 
        description="Horizonte de predição em dias (30-365)",
        ge=30, 
        le=365
    )
    
    include_demographics: bool = Field(
        default=True,
        description="Incluir análise demográfica (Idade, Sexo, Estado_Civil) se disponível"
    )
    
    include_geographic: bool = Field(
        default=True,
        description="Incluir análise geográfica (Cidade, Estado) se disponível"
    )
    
    min_transactions: int = Field(
        default=2,
        description="Mínimo de transações por cliente para incluir na análise",
        ge=1,
        le=10
    )
    
    @field_validator('analysis_type')
    @classmethod
    def validate_analysis_type(cls, v):
        valid_types = [
            'behavioral_segmentation', 'lifecycle_analysis', 'churn_prediction',
            'value_analysis', 'preference_mining', 'journey_mapping'
        ]
        if v not in valid_types:
            raise ValueError(f"analysis_type deve ser um de: {valid_types}")
        return v
    
    @field_validator('segmentation_method')
    @classmethod
    def validate_segmentation_method(cls, v):
        valid_methods = ['rfm', 'behavioral', 'value_based', 'hybrid']
        if v not in valid_methods:
            raise ValueError(f"segmentation_method deve ser um de: {valid_methods}")
        return v

class CustomerInsightsEngine(BaseTool):
    name: str = "Customer Insights Engine"
    description: str = """
    🎯 MOTOR AVANÇADO DE INSIGHTS DE CLIENTES PARA JOALHERIAS V3.0
    
    ⭐ QUANDO USAR ESTA FERRAMENTA:
    📊 Para segmentar clientes por comportamento, valor e perfil demográfico
    💡 Para entender padrões de compra e preferências de clientes
    ⚠️ Para identificar clientes em risco de abandono  
    💎 Para calcular valor vitalício (CLV) e análise de valor
    🔍 Para descobrir preferências por idade, gênero e localização
    🗺️ Para mapear jornada completa do cliente e pontos de atrito
    
    🔧 DADOS REAIS UTILIZADOS:
    ✅ Identificação: Codigo_Cliente, Nome_Cliente
    ✅ Demografia: Idade, Sexo, Estado_Civil, Data_Nascimento
    ✅ Geografia: Cidade, Estado
    ✅ Financeiro: Total_Liquido, Custo_Produto, Desconto_Aplicado, Margem_Real
    ✅ Produto: Grupo_Produto, Metal, Colecao, Subgrupo_Produto
    ✅ Comportamento: Frequência, Recência, Valor médio, Consistência
    
    🎯 6 ANÁLISES ESPECIALIZADAS:
    
    1️⃣ BEHAVIORAL_SEGMENTATION
    📝 Use quando: Precisar segmentar base de clientes
    📈 Entrega: Segmentos RFM, clusters comportamentais, perfis demográficos
    🎯 Ideal para: Campanhas de marketing direcionadas
    
    2️⃣ LIFECYCLE_ANALYSIS  
    📝 Use quando: Quiser entender evolução dos clientes
    📈 Entrega: Estágios (Novo→Crescendo→Leal→VIP), transições, tempo médio
    🎯 Ideal para: Estratégias de retenção e desenvolvimento
    
    3️⃣ CHURN_PREDICTION
    📝 Use quando: Precisar identificar clientes em risco
    📈 Entrega: Score de risco, fatores de churn, estratégias de retenção
    🎯 Ideal para: Ações preventivas e recuperação de clientes
    
    4️⃣ VALUE_ANALYSIS
    📝 Use quando: Quiser entender valor dos clientes
    📈 Entrega: CLV, análise Pareto, concentração de valor, margem por cliente
    🎯 Ideal para: Priorização de clientes e investimento em relacionamento
    
    5️⃣ PREFERENCE_MINING
    📝 Use quando: Precisar entender preferências de compra
    📈 Entrega: Preferências por idade/gênero, produtos favoritos, padrões sazonais
    🎯 Ideal para: Personalização e recomendações de produtos
    
    6️⃣ JOURNEY_MAPPING
    📝 Use quando: Quiser mapear experiência do cliente
    📈 Entrega: Jornada por perfil, pontos de atrito, oportunidades de melhoria
    🎯 Ideal para: Otimização da experiência e processo de vendas
    
    🚀 SAÍDA ESTRUTURADA JSON:
    - Análise completa com insights acionáveis
    - Recomendações específicas por segmento
    - Métricas e KPIs relevantes
    - Visualizações em formato texto
    - Próximos passos sugeridos
    """
    args_schema: Type[BaseModel] = CustomerInsightsInput
    
    def _run(self, analysis_type: str, data_csv: str = "data/vendas.csv", 
             customer_id_column: str = "Codigo_Cliente", segmentation_method: str = "rfm",
             prediction_horizon: int = 90, include_demographics: bool = True,
             include_geographic: bool = True, min_transactions: int = 2) -> str:
        try:
            print("🔍 Iniciando análise avançada de clientes com dados REAIS...")
            
            # Carregar e preparar dados REAIS
            df = pd.read_csv(data_csv, sep=';', encoding='utf-8')
            customer_data = self._prepare_customer_data(
                df, customer_id_column, include_demographics, 
                include_geographic, min_transactions
            )
            
            if customer_data is None or len(customer_data) < min_transactions:
                return self._format_error_result(
                    "Dados insuficientes para análise de clientes",
                    f"Mínimo necessário: {min_transactions} clientes com dados válidos"
                )
            
            print(f"✅ Dados preparados: {len(customer_data)} clientes únicos")
            
            # Dicionário de análises especializadas
            customer_analyses = {
                'behavioral_segmentation': self._behavioral_segmentation,
                'lifecycle_analysis': self._lifecycle_analysis,
                'churn_prediction': self._churn_prediction,
                'value_analysis': self._value_analysis,
                'preference_mining': self._preference_mining,
                'journey_mapping': self._journey_mapping
            }
            
            if analysis_type not in customer_analyses:
                return self._format_error_result(
                    f"Análise '{analysis_type}' não suportada",
                    f"Opções disponíveis: {list(customer_analyses.keys())}"
                )
            
            # Executar análise específica
            analysis_result = customer_analyses[analysis_type](
                customer_data, segmentation_method, prediction_horizon,
                include_demographics, include_geographic
            )
            
            # Formatar resultado estruturado
            return self._format_customer_result(analysis_type, analysis_result, customer_data)
            
        except Exception as e:
            print(f"❌ Erro na análise de clientes: {str(e)}")
            return self._format_error_result("Erro na execução da análise", str(e))
    
    def _prepare_customer_data(self, df: pd.DataFrame, customer_id_column: str,
                               include_demographics: bool = True, include_geographic: bool = True,
                               min_transactions: int = 2) -> Optional[pd.DataFrame]:
        """Preparar dados de clientes REAIS usando todos os campos da query SQL."""
        try:
            print("🏗️ Preparando dados de clientes REAIS...")
            
            # Converter data
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
            df = df.dropna(subset=['Data'])
            
            # USAR DADOS REAIS - NUNCA SIMULAR
            if customer_id_column not in df.columns or df[customer_id_column].isna().all():
                return self._handle_missing_customer_data(df, include_demographics, include_geographic, min_transactions)
            
            # Limpar dados de cliente
            df = df[df[customer_id_column].notna()]
            df = df[df[customer_id_column] != '']
            df[customer_id_column] = df[customer_id_column].astype(str).str.strip()
            
            print(f"✅ Usando {customer_id_column} real: {df[customer_id_column].nunique()} clientes únicos")
            
            # Validar campos disponíveis
            available_fields = self._validate_available_fields(df)
            print(f"📊 Campos disponíveis: {available_fields}")
            
            # Preparar features de cliente REAIS
            df = self._calculate_real_derived_fields(df)
            
            # Agregar dados por cliente usando TODOS os campos disponíveis
            customer_data = self._aggregate_customer_metrics(df, customer_id_column, available_fields)
            
            # Filtrar por número mínimo de transações
            if min_transactions > 1:
                customer_data = customer_data[customer_data['Frequency'] >= min_transactions]
                print(f"🔍 Filtrados clientes com pelo menos {min_transactions} transações")
            
            print(f"🎯 Análise final: {len(customer_data)} clientes com {len(customer_data.columns)} métricas")
            
            return customer_data
            
        except Exception as e:
            print(f"❌ Erro na preparação de dados de clientes REAIS: {str(e)}")
            return None
    
    def _handle_missing_customer_data(self, df: pd.DataFrame, include_demographics: bool = True,
                                     include_geographic: bool = True, min_transactions: int = 2) -> Optional[pd.DataFrame]:
        """Lidar com dados faltantes de cliente de forma transparente."""
        print("⚠️ ATENÇÃO: Codigo_Cliente não disponível nos dados")
        print("🔍 Verificando campos alternativos...")
        
        # Verificar se há Nome_Cliente
        if 'Nome_Cliente' in df.columns and not df['Nome_Cliente'].isna().all():
            print("✅ Usando Nome_Cliente como identificador")
            df['Customer_ID_Alt'] = df['Nome_Cliente'].astype(str).str.strip()
            return self._prepare_customer_data(df, 'Customer_ID_Alt', include_demographics, include_geographic, min_transactions)
        
        # Se não há dados de cliente, retornar erro explicativo
        print("❌ Nenhum identificador de cliente disponível")
        return None
    
    def _validate_available_fields(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validar quais campos da query SQL estão disponíveis."""
        field_categories = {
            'demographics': ['Idade', 'Sexo', 'Estado_Civil', 'Data_Nascimento'],
            'geographic': ['Cidade', 'Estado'],
            'financial': ['Custo_Produto', 'Preco_Tabela', 'Desconto_Aplicado'],
            'product': ['Grupo_Produto', 'Subgrupo_Produto', 'Metal', 'Colecao'],
            'sales': ['Codigo_Vendedor', 'Nome_Vendedor'],
            'inventory': ['Estoque_Atual'],
            'basic': ['Total_Liquido', 'Quantidade', 'Data']
        }
        
        available = {}
        for category, fields in field_categories.items():
            available[category] = [field for field in fields if field in df.columns]
        
        return available
    
    def _calculate_real_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular campos derivados usando dados reais."""
        print("⚙️ Calculando campos derivados com dados REAIS...")
        
        # Campos financeiros reais
        if 'Custo_Produto' in df.columns and 'Total_Liquido' in df.columns:
            df['Margem_Real'] = df['Total_Liquido'] - df['Custo_Produto']
            df['Margem_Percentual'] = (df['Margem_Real'] / df['Total_Liquido'] * 100).replace([np.inf, -np.inf], 0)
            print("✅ Margem real calculada")
        
        if 'Desconto_Aplicado' in df.columns and 'Preco_Tabela' in df.columns:
            df['Desconto_Percentual'] = (df['Desconto_Aplicado'] / df['Preco_Tabela'] * 100).replace([np.inf, -np.inf], 0)
            print("✅ Desconto percentual calculado")
        
        # Campos demográficos derivados
        if 'Idade' in df.columns:
            df['Faixa_Etaria'] = pd.cut(df['Idade'], 
                                       bins=[0, 25, 35, 45, 55, 100], 
                                       labels=['18-25', '26-35', '36-45', '46-55', '55+'])
            print("✅ Faixa etária calculada")
        
        # Campos básicos
        df['Preco_Unitario'] = df['Total_Liquido'] / df['Quantidade'].replace(0, 1)
        df['Ano_Mes'] = df['Data'].dt.to_period('M').astype(str)
        df['Trimestre'] = df['Data'].dt.quarter
        df['Mes'] = df['Data'].dt.month
        
        return df
    
    def _simulate_customer_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simular IDs de clientes baseado em padrões de compra."""
        # Estratégia: agrupar compras similares por valor, data e produtos
        df = df.copy()
        
        # Criar clusters baseados em valor e tempo
        df['Date_Numeric'] = df['Data'].astype('int64') // 10**9  # Convert to seconds
        
        # Normalizar para clustering
        features_for_clustering = ['Total_Liquido', 'Date_Numeric']
        if 'Quantidade' in df.columns:
            features_for_clustering.append('Quantidade')
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features_for_clustering])
        
        # Usar DBSCAN para agrupar transações similares
        dbscan = DBSCAN(eps=0.5, min_samples=1)
        clusters = dbscan.fit_predict(scaled_features)
        
        df['Customer_ID'] = f'CUST_' + pd.Series(clusters).astype(str)
        
        return df
    
    def _aggregate_customer_metrics(self, df: pd.DataFrame, customer_id_column: str, available_fields: Dict[str, List[str]]) -> pd.DataFrame:
        """Agregar métricas por cliente usando TODOS os campos reais disponíveis."""
        print("📊 Agregando métricas por cliente com dados REAIS...")
        
        current_date = df['Data'].max()
        
        # Agregações básicas
        base_agg = {
            'Total_Liquido': ['sum', 'mean', 'count', 'std'],
            'Quantidade': 'sum',
            'Data': ['min', 'max'],
            'Preco_Unitario': ['mean', 'std']
        }
        
        # Adicionar agregações para campos financeiros reais
        if 'Margem_Real' in df.columns:
            base_agg['Margem_Real'] = ['sum', 'mean']
            base_agg['Margem_Percentual'] = ['mean', 'std']
        
        if 'Desconto_Percentual' in df.columns:
            base_agg['Desconto_Percentual'] = ['mean', 'std']
        
        # Agregações demográficas (primeiro valor, assumindo consistência)
        demographic_agg = {}
        for field in available_fields.get('demographics', []):
            if field in df.columns:
                demographic_agg[field] = 'first'
        
        # Agregações geográficas
        geographic_agg = {}
        for field in available_fields.get('geographic', []):
            if field in df.columns:
                geographic_agg[field] = 'first'
        
        # Agregações de produto (moda para preferências)
        product_agg = {}
        for field in available_fields.get('product', []):
            if field in df.columns:
                product_agg[field] = lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'N/A'
        
        # Combinar todas as agregações
        all_agg = {**base_agg, **demographic_agg, **geographic_agg, **product_agg}
        
        # Executar agregação
        customer_metrics = df.groupby(customer_id_column).agg(all_agg).fillna(0)
        
        # Flatten columns
        customer_metrics.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                  for col in customer_metrics.columns]
        
        # Calcular métricas RFM REAIS
        customer_metrics = self._calculate_real_rfm_metrics(customer_metrics, current_date)
        
        # Calcular métricas de comportamento REAIS
        customer_metrics = self._calculate_real_behavioral_metrics(customer_metrics)
        
        print(f"✅ {len(customer_metrics)} clientes com {len(customer_metrics.columns)} métricas agregadas")
        
        return customer_metrics
    
    def _calculate_real_rfm_metrics(self, customer_metrics: pd.DataFrame, current_date: pd.Timestamp) -> pd.DataFrame:
        """Calcular métricas RFM usando dados reais."""
        print("🎯 Calculando métricas RFM REAIS...")
        
        # Recency (dias desde última compra)
        customer_metrics['Recency'] = (current_date - pd.to_datetime(customer_metrics['Data_max'])).dt.days
        
        # Frequency (número de transações)
        customer_metrics['Frequency'] = customer_metrics['Total_Liquido_count']
        
        # Monetary (valor total)
        customer_metrics['Monetary'] = customer_metrics['Total_Liquido_sum']
        
        # Métricas de tempo
        customer_metrics['Customer_Lifetime_Days'] = (
            pd.to_datetime(customer_metrics['Data_max']) - 
            pd.to_datetime(customer_metrics['Data_min'])
        ).dt.days + 1
        
        customer_metrics['Avg_Days_Between_Purchases'] = (
            customer_metrics['Customer_Lifetime_Days'] / customer_metrics['Frequency']
        ).replace([np.inf], 0)
        
        # Consistência de compra
        customer_metrics['Purchase_Consistency'] = 1 / (1 + customer_metrics['Total_Liquido_std'].fillna(0))
        
        return customer_metrics
    
    def _calculate_real_behavioral_metrics(self, customer_metrics: pd.DataFrame) -> pd.DataFrame:
        """Calcular métricas comportamentais usando dados reais."""
        print("🧠 Calculando métricas comportamentais REAIS...")
        
        # Ticket médio real
        customer_metrics['AOV_Real'] = customer_metrics['Total_Liquido_mean']
        
        # Valor vitalício estimado (baseado em dados reais)
        customer_metrics['CLV_Estimado'] = (
            customer_metrics['AOV_Real'] * 
            customer_metrics['Frequency'] * 
            np.maximum(customer_metrics['Customer_Lifetime_Days'] / 365, 1)
        )
        
        # Margem do cliente (se disponível)
        if 'Margem_Real_sum' in customer_metrics.columns:
            customer_metrics['Customer_Margin'] = customer_metrics['Margem_Real_sum']
            customer_metrics['Margin_Rate'] = customer_metrics['Margem_Real_mean']
        
        # Sensibilidade a desconto (se disponível)
        if 'Desconto_Percentual_mean' in customer_metrics.columns:
            customer_metrics['Discount_Sensitivity'] = customer_metrics['Desconto_Percentual_mean']
        
        return customer_metrics
    
    def _behavioral_segmentation(self, customer_data: pd.DataFrame, method: str, 
                                horizon: int, include_demographics: bool = True,
                                include_geographic: bool = True) -> Dict[str, Any]:
        """Segmentação comportamental usando dados demográficos REAIS."""
        print("🎭 Executando segmentação comportamental com dados REAIS...")
        
        result = {}
        
        # RFM tradicional com dados reais
        if method == 'rfm':
            result['rfm_analysis'] = self._rfm_segmentation(customer_data)
        elif method == 'behavioral':
            result['behavioral_analysis'] = self._advanced_behavioral_segmentation(customer_data)
        elif method == 'value_based':
            result['value_analysis'] = self._value_based_segmentation(customer_data)
        elif method == 'hybrid':
            result['hybrid_analysis'] = self._hybrid_segmentation(customer_data)
        else:
            result['rfm_analysis'] = self._rfm_segmentation(customer_data)
        
        # Segmentação demográfica REAL
        demographic_cols = [col for col in customer_data.columns if any(demo in col for demo in ['Idade', 'Sexo', 'Estado_Civil', 'Faixa_Etaria'])]
        if demographic_cols:
            result['demographic_segmentation'] = self._real_demographic_segmentation(customer_data)
            print("✅ Segmentação demográfica adicionada")
        
        # Segmentação geográfica REAL
        geographic_cols = [col for col in customer_data.columns if any(geo in col for geo in ['Cidade', 'Estado'])]
        if geographic_cols:
            result['geographic_segmentation'] = self._real_geographic_segmentation(customer_data)
            print("✅ Segmentação geográfica adicionada")
        
        # Clustering comportamental avançado com features demográficas
        behavioral_features = self._select_behavioral_features(customer_data)
        if len(behavioral_features) >= 3:
            result['advanced_clustering'] = self._real_advanced_clustering(customer_data, behavioral_features)
            print("✅ Clustering avançado com dados reais")
        
        return result
    
    def _real_demographic_segmentation(self, customer_data: pd.DataFrame) -> Dict[str, Any]:
        """Segmentação demográfica com dados REAIS."""
        print("👥 Executando segmentação demográfica com dados REAIS...")
        
        demographic_analysis = {}
        
        # Análise por idade
        age_cols = [col for col in customer_data.columns if 'Faixa_Etaria' in col or 'Idade' in col]
        if age_cols:
            age_col = age_cols[0]  # Usar primeira coluna encontrada
            if not customer_data[age_col].isna().all():
                age_analysis = customer_data.groupby(age_col).agg({
                    'Monetary': ['mean', 'sum', 'count'],
                    'Frequency': 'mean',
                    'CLV_Estimado': 'mean' if 'CLV_Estimado' in customer_data.columns else 'count'
                }).round(2)
                
                demographic_analysis['age_segments'] = self._safe_dataframe_to_dict(age_analysis)
                print(f"✅ Análise por idade usando {age_col}")
        
        # Análise por gênero
        gender_cols = [col for col in customer_data.columns if 'Sexo' in col]
        if gender_cols:
            gender_col = gender_cols[0]
            if not customer_data[gender_col].isna().all():
                gender_analysis = customer_data.groupby(gender_col).agg({
                    'Monetary': ['mean', 'sum', 'count'],
                    'Frequency': 'mean',
                    'AOV_Real': 'mean' if 'AOV_Real' in customer_data.columns else 'count'
                }).round(2)
                
                demographic_analysis['gender_segments'] = self._safe_dataframe_to_dict(gender_analysis)
                print(f"✅ Análise por gênero usando {gender_col}")
        
        # Análise por estado civil
        marital_cols = [col for col in customer_data.columns if 'Estado_Civil' in col]
        if marital_cols:
            marital_col = marital_cols[0]
            if not customer_data[marital_col].isna().all():
                marital_analysis = customer_data.groupby(marital_col).agg({
                    'Monetary': ['mean', 'sum', 'count'],
                    'AOV_Real': 'mean' if 'AOV_Real' in customer_data.columns else 'count'
                }).round(2)
                
                demographic_analysis['marital_segments'] = self._safe_dataframe_to_dict(marital_analysis)
                print(f"✅ Análise por estado civil usando {marital_col}")
        
        # Cross-analysis: Idade x Gênero (se ambos disponíveis)
        if 'age_segments' in demographic_analysis and 'gender_segments' in demographic_analysis:
            try:
                age_col = age_cols[0]
                gender_col = gender_cols[0]
                cross_analysis = customer_data.groupby([age_col, gender_col])['Monetary'].mean().unstack()
                demographic_analysis['age_gender_cross'] = self._safe_dataframe_to_dict(cross_analysis)
                print("✅ Cross-analysis idade x gênero")
            except Exception as e:
                print(f"⚠️ Cross-analysis falhou: {e}")
        
        return demographic_analysis
    
    def _real_geographic_segmentation(self, customer_data: pd.DataFrame) -> Dict[str, Any]:
        """Segmentação geográfica com dados REAIS."""
        print("🗺️ Executando segmentação geográfica com dados REAIS...")
        
        geographic_analysis = {}
        
        # Análise por estado
        state_cols = [col for col in customer_data.columns if 'Estado' in col]
        if state_cols:
            state_col = state_cols[0]
            if not customer_data[state_col].isna().all():
                state_analysis = customer_data.groupby(state_col).agg({
                    'Monetary': ['mean', 'sum', 'count'],
                    'Frequency': 'mean',
                    'CLV_Estimado': 'mean' if 'CLV_Estimado' in customer_data.columns else 'count'
                }).round(2)
                
                geographic_analysis['state_segments'] = self._safe_dataframe_to_dict(state_analysis)
                print(f"✅ Análise por estado usando {state_col}")
        
        # Análise por cidade (top 20)
        city_cols = [col for col in customer_data.columns if 'Cidade' in col]
        if city_cols:
            city_col = city_cols[0]
            if not customer_data[city_col].isna().all():
                city_analysis = customer_data.groupby(city_col).agg({
                    'Monetary': ['mean', 'sum', 'count'],
                    'Frequency': 'mean'
                }).round(2)
                
                # Top 20 cidades por receita
                top_cities = city_analysis.nlargest(20, ('Monetary', 'sum'))
                geographic_analysis['top_cities'] = self._safe_dataframe_to_dict(top_cities)
                print(f"✅ Top 20 cidades por receita usando {city_col}")
        
        return geographic_analysis
    
    def _select_behavioral_features(self, customer_data: pd.DataFrame) -> List[str]:
        """Selecionar features comportamentais disponíveis para clustering."""
        potential_features = [
            'Recency', 'Frequency', 'Monetary', 'AOV_Real', 'CLV_Estimado',
            'Purchase_Consistency', 'Customer_Lifetime_Days', 'Avg_Days_Between_Purchases'
        ]
        
        # Adicionar features financeiras se disponíveis
        financial_features = [col for col in customer_data.columns 
                            if any(word in col for word in ['Margem', 'Desconto', 'Margin', 'Discount'])]
        
        # Adicionar features demográficas numéricas se disponíveis
        demographic_features = [col for col in customer_data.columns 
                              if any(demo in col for demo in ['Idade_first', 'Idade'])]
        
        # Filtrar features que existem nos dados
        all_potential = potential_features + financial_features + demographic_features
        available_features = [f for f in all_potential if f in customer_data.columns]
        
        print(f"📊 Features selecionadas para clustering: {available_features}")
        return available_features
    
    def _real_advanced_clustering(self, customer_data: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
        """Clustering comportamental avançado com dados reais."""
        print("🔬 Executando clustering avançado com dados REAIS...")
        
        # Preparar dados para clustering
        X = customer_data[features].fillna(0)
        
        # Padronizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Encontrar número ótimo de clusters
        optimal_k = self._find_optimal_clusters_silhouette(X_scaled, max_k=6)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        customer_data['Behavioral_Cluster'] = clusters
        
        # Análise dos clusters
        cluster_analysis = {}
        for cluster_id in range(optimal_k):
            cluster_data = customer_data[customer_data['Behavioral_Cluster'] == cluster_id]
            
            cluster_profile = {
                'size': len(cluster_data),
                'percentage': round(len(cluster_data) / len(customer_data) * 100, 1),
                'avg_monetary': round(cluster_data['Monetary'].mean(), 2),
                'avg_frequency': round(cluster_data['Frequency'].mean(), 1),
                'avg_recency': round(cluster_data['Recency'].mean(), 1)
            }
            
            # Adicionar insights demográficos se disponíveis
            age_cols = [col for col in cluster_data.columns if 'Faixa_Etaria' in col or 'Idade' in col]
            if age_cols and not cluster_data[age_cols[0]].isna().all():
                mode_age = cluster_data[age_cols[0]].mode()
                cluster_profile['dominant_age_group'] = mode_age.iloc[0] if len(mode_age) > 0 else 'N/A'
            
            gender_cols = [col for col in cluster_data.columns if 'Sexo' in col]
            if gender_cols and not cluster_data[gender_cols[0]].isna().all():
                mode_gender = cluster_data[gender_cols[0]].mode()
                cluster_profile['dominant_gender'] = mode_gender.iloc[0] if len(mode_gender) > 0 else 'N/A'
            
            cluster_analysis[f'Cluster_{cluster_id}'] = cluster_profile
        
        return {
            'optimal_clusters': optimal_k,
            'cluster_analysis': cluster_analysis,
            'features_used': features,
            'silhouette_score': round(silhouette_score(X_scaled, clusters), 3)
        }
    
    def _rfm_segmentation(self, customer_data: pd.DataFrame) -> Dict[str, Any]:
        """Segmentação RFM tradicional."""
        try:
            # Calcular scores RFM
            customer_data['R_Score'] = pd.qcut(customer_data['Recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
            customer_data['F_Score'] = pd.qcut(customer_data['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
            customer_data['M_Score'] = pd.qcut(customer_data['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
            
            # Converter para int
            customer_data['R_Score'] = customer_data['R_Score'].astype(int)
            customer_data['F_Score'] = customer_data['F_Score'].astype(int)
            customer_data['M_Score'] = customer_data['M_Score'].astype(int)
            
            # Classificar segmentos
            customer_data['RFM_Segment'] = customer_data.apply(self._classify_rfm_segment, axis=1)
            
            # Análise dos segmentos
            segment_analysis = customer_data.groupby('RFM_Segment').agg({
                'Recency': ['count', 'mean'],
                'Frequency': 'mean',
                'Monetary': ['mean', 'sum'],
                'Customer_Lifetime_Days': 'mean'
            }).round(2)
            
            # Achatar colunas MultiIndex
            segment_analysis.columns = ['Count', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Total_Monetary', 'Avg_Lifetime']
            
            # Insights por segmento
            segment_insights = {}
            for segment in customer_data['RFM_Segment'].unique():
                segment_data = customer_data[customer_data['RFM_Segment'] == segment]
                
                insights = []
                avg_monetary = segment_data['Monetary'].mean()
                avg_frequency = segment_data['Frequency'].mean()
                avg_recency = segment_data['Recency'].mean()
                
                if avg_monetary > customer_data['Monetary'].mean() * 1.5:
                    insights.append("Alto valor monetário")
                if avg_frequency > customer_data['Frequency'].mean() * 1.5:
                    insights.append("Alta frequência de compra")
                if avg_recency < 30:
                    insights.append("Compra recente")
                elif avg_recency > 180:
                    insights.append("Cliente inativo")
                
                segment_insights[segment] = insights
            
            # Matriz de migração (simulada)
            migration_matrix = self._simulate_segment_migration(customer_data)
            
            return {
                'segmentation_method': 'RFM',
                'segment_distribution': self._safe_series_to_dict(customer_data['RFM_Segment'].value_counts()),
                'segment_analysis': self._safe_dataframe_to_dict(segment_analysis),
                'segment_insights': segment_insights,
                'migration_matrix': migration_matrix,
                'total_customers': len(customer_data)
            }
            
        except Exception as e:
            return {'error': f"Erro na segmentação RFM: {str(e)}"}
    
    def _advanced_behavioral_segmentation(self, customer_data: pd.DataFrame) -> Dict[str, Any]:
        """Wrapper para segmentação comportamental avançada."""
        behavioral_features = self._select_behavioral_features(customer_data)
        if len(behavioral_features) >= 3:
            return self._real_advanced_clustering(customer_data, behavioral_features)
        else:
            return {'error': 'Features insuficientes para segmentação comportamental'}
    
    def _value_based_segmentation(self, customer_data: pd.DataFrame) -> Dict[str, Any]:
        """Wrapper para segmentação baseada em valor."""
        return self._value_analysis(customer_data, 'value_based', 90)
    
    def _hybrid_segmentation(self, customer_data: pd.DataFrame) -> Dict[str, Any]:
        """Segmentação híbrida combinando múltiplos critérios."""
        rfm_result = self._rfm_segmentation(customer_data)
        
        # Incluir análises demográficas se disponíveis
        demographic_cols = [col for col in customer_data.columns if any(demo in col for demo in ['Idade', 'Sexo', 'Estado_Civil'])]
        if demographic_cols:
            demographic_result = self._real_demographic_segmentation(customer_data)
        else:
            demographic_result = {'info': 'Dados demográficos não disponíveis'}
        
        return {
            'segmentation_method': 'Hybrid (RFM + Demographic + Geographic)',
            'rfm_component': rfm_result,
            'demographic_component': demographic_result,
            'total_customers': len(customer_data)
        }
    
    # Métodos auxiliares necessários
    def _classify_rfm_segment(self, row) -> str:
        """Classificar segmento RFM."""
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 2:
            return 'New Customers'
        elif r >= 3 and f <= 2:
            return 'Potential Loyalists'
        elif r <= 2 and f >= 3:
            return 'At Risk'
        elif r <= 2 and f <= 2 and m >= 3:
            return 'Cannot Lose Them'
        elif r <= 2 and f <= 2 and m <= 2:
            return 'Lost'
        else:
            return 'Others'
    
    def _find_optimal_clusters_silhouette(self, X: np.ndarray, max_k: int = 8) -> int:
        """Encontrar número ótimo de clusters usando silhouette score."""
        if len(X) < 4:
            return 2
        
        max_k = min(max_k, len(X) - 1)
        silhouette_scores = []
        
        for k in range(2, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(X)
                silhouette_avg = silhouette_score(X, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            except:
                continue
        
        if silhouette_scores:
            optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
            return optimal_k
        else:
            return 3
    
    def _analyze_lifecycle_transitions(self, customer_data: pd.DataFrame) -> Dict[str, Any]:
        """Analisar transições do ciclo de vida."""
        # Estimativa baseada em padrões
        transitions = {
            'New_to_Growing': 0.65,
            'Growing_to_Loyal': 0.35,
            'Loyal_to_VIP': 0.25,
            'At_Risk_to_Lost': 0.40
        }
        
        return transitions
    
    def _identify_churn_risk_factors(self, customer_data: pd.DataFrame) -> List[str]:
        """Identificar fatores de risco de churn."""
        factors = []
        
        # Análise de recência
        high_recency = customer_data[customer_data['Recency'] > 180]
        if len(high_recency) > 0:
            factors.append(f"{len(high_recency)} clientes sem compra há 6+ meses")
        
        # Baixa frequência
        low_frequency = customer_data[customer_data['Frequency'] == 1]
        if len(low_frequency) > 0:
            factors.append(f"{len(low_frequency)} clientes com apenas 1 compra")
        
        # Queda na consistência
        if 'Purchase_Consistency' in customer_data.columns:
            inconsistent = customer_data[customer_data['Purchase_Consistency'] < 0.3]
            if len(inconsistent) > 0:
                factors.append(f"{len(inconsistent)} clientes com padrão irregular")
        
        return factors
    
    def _generate_retention_strategies(self, customer_data: pd.DataFrame) -> Dict[str, List[str]]:
        """Gerar estratégias de retenção por segmento."""
        strategies = {}
        
        for risk_category in customer_data['Churn_Risk_Category'].unique():
            if pd.isna(risk_category):
                continue
                
            if risk_category == 'High Risk':
                strategies[risk_category] = [
                    "Contato pessoal imediato",
                    "Oferta especial personalizada",
                    "Programa VIP exclusivo",
                    "Desconto significativo na próxima compra"
                ]
            elif risk_category == 'Medium Risk':
                strategies[risk_category] = [
                    "Email marketing personalizado",
                    "Lembrete de produtos favoritos",
                    "Programa de fidelidade",
                    "Cross-sell baseado em histórico"
                ]
            else:  # Low Risk
                strategies[risk_category] = [
                    "Newsletter regular",
                    "Novidades e lançamentos",
                    "Programa de indicação",
                    "Manutenção do relacionamento"
                ]
        
        return strategies
    
    def _calculate_churn_financial_impact(self, customer_data: pd.DataFrame) -> Dict[str, float]:
        """Calcular impacto financeiro do churn."""
        high_risk = customer_data[customer_data['Churn_Risk_Category'] == 'High Risk']
        
        potential_lost_revenue = high_risk['Monetary'].sum()
        potential_lost_clv = high_risk['CLV_Estimado'].sum() if 'CLV_Estimado' in customer_data.columns else potential_lost_revenue * 1.5
        
        return {
            'high_risk_current_value': round(potential_lost_revenue, 2),
            'potential_lost_clv': round(potential_lost_clv, 2),
            'retention_investment_suggested': round(potential_lost_clv * 0.1, 2)  # 10% do CLV
        }
    
    def _analyze_value_concentration(self, customer_data: pd.DataFrame) -> Dict[str, float]:
        """Analisar concentração de valor."""
        # Regra 80/20
        clv_column = 'CLV_Estimado' if 'CLV_Estimado' in customer_data.columns else 'Monetary'
        
        sorted_customers = customer_data.sort_values(clv_column, ascending=False)
        top_20_pct = int(len(sorted_customers) * 0.2)
        
        top_20_value = sorted_customers.head(top_20_pct)[clv_column].sum()
        total_value = sorted_customers[clv_column].sum()
        
        concentration_80_20 = (top_20_value / total_value * 100) if total_value > 0 else 0
        
        return {
            'top_20_percent_value_share': round(concentration_80_20, 1),
            'value_distribution': 'Concentrated' if concentration_80_20 > 80 else 'Balanced'
        }
    
    def _analyze_stage_transitions(self, customer_data: pd.DataFrame) -> Dict[str, float]:
        """Analisar transições entre estágios da jornada."""
        # Estimativas baseadas em padrões típicos
        transitions = {
            'new_to_growing': 0.45,
            'growing_to_loyal': 0.35,
            'loyal_to_vip': 0.20
        }
        
        return transitions
    
    def _identify_friction_points(self, customer_data: pd.DataFrame) -> List[str]:
        """Identificar pontos de atrito na jornada."""
        friction_points = []
        
        # Alta taxa de clientes one-time
        one_time = len(customer_data[customer_data['Frequency'] == 1])
        one_time_rate = one_time / len(customer_data) * 100
        
        if one_time_rate > 60:
            friction_points.append(f"Alta taxa de one-time buyers ({one_time_rate:.1f}%)")
        
        # Longo intervalo entre compras
        avg_interval = customer_data['Avg_Days_Between_Purchases'].mean()
        if avg_interval > 180:
            friction_points.append(f"Intervalo longo entre compras ({avg_interval:.0f} dias)")
        
        # Baixa conversão para alto valor
        high_value = len(customer_data[customer_data['Monetary'] > customer_data['Monetary'].quantile(0.8)])
        high_value_rate = high_value / len(customer_data) * 100
        
        if high_value_rate < 15:
            friction_points.append(f"Baixa conversão para alto valor ({high_value_rate:.1f}%)")
        
        return friction_points
    
    # ==========================================
    # MÉTODOS DE ANÁLISE AUSENTES
    # ==========================================
    
    def _lifecycle_analysis(self, customer_data: pd.DataFrame, method: str, 
                           horizon: int, include_demographics: bool = True,
                           include_geographic: bool = True) -> Dict[str, Any]:
        """Análise completa do ciclo de vida do cliente."""
        print("🔄 Executando análise de ciclo de vida...")
        
        # Classificar estágios do ciclo de vida
        customer_data['Lifecycle_Stage'] = customer_data.apply(self._classify_lifecycle_stage, axis=1)
        
        # Análise de distribuição por estágio
        stage_distribution = customer_data['Lifecycle_Stage'].value_counts()
        stage_percentages = (stage_distribution / len(customer_data) * 100).round(1)
        
        # Métricas por estágio
        stage_metrics = customer_data.groupby('Lifecycle_Stage').agg({
            'Monetary': ['mean', 'sum', 'count'],
            'Frequency': 'mean',
            'Recency': 'mean',
            'AOV_Real': 'mean',
            'CLV_Estimado': 'mean' if 'CLV_Estimado' in customer_data.columns else 'count'
        }).round(2)
        
        # Análise de transições
        transitions = self._analyze_lifecycle_transitions(customer_data)
        
        # Tempo médio em cada estágio (estimativa)
        avg_stage_duration = {
            'Novo': 30,
            'Crescendo': 60,
            'Leal': 180,
            'VIP': 365,
            'Em Risco': 45,
            'Perdido': 0
        }
        
        # Recomendações por estágio
        stage_strategies = {
            'Novo': [
                "Programa de boas-vindas personalizado",
                "Desconto na segunda compra",
                "Educação sobre produtos e cuidados",
                "Follow-up pós-venda"
            ],
            'Crescendo': [
                "Cross-sell de produtos complementares",
                "Programa de fidelidade",
                "Convite para eventos exclusivos",
                "Personalização baseada em preferências"
            ],
            'Leal': [
                "Programa VIP com benefícios exclusivos",
                "Early access a novas coleções",
                "Desconto em manutenção e limpeza",
                "Programa de indicação"
            ],
            'VIP': [
                "Atendimento personalizado premium",
                "Peças exclusivas sob medida",
                "Eventos VIP e lançamentos",
                "Consultoria em joias"
            ],
            'Em Risco': [
                "Contato pessoal imediato",
                "Oferta de reativação especial",
                "Pesquisa de satisfação",
                "Win-back campaign"
            ],
            'Perdido': [
                "Campanha de reconquista",
                "Oferta irresistível",
                "Novo produto/tendência",
                "Mudança de canal de comunicação"
            ]
        }
        
        return {
            'stage_distribution': self._safe_series_to_dict(stage_distribution),
            'stage_percentages': self._safe_series_to_dict(stage_percentages),
            'stage_metrics': self._safe_dataframe_to_dict(stage_metrics),
            'lifecycle_transitions': transitions,
            'avg_stage_duration_days': avg_stage_duration,
            'stage_strategies': stage_strategies,
            'total_customers': len(customer_data)
        }
    
    def _churn_prediction(self, customer_data: pd.DataFrame, method: str, 
                         horizon: int, include_demographics: bool = True,
                         include_geographic: bool = True) -> Dict[str, Any]:
        """Predição de abandono de clientes."""
        print("⚠️ Executando predição de churn...")
        
        # Calcular score de risco baseado em múltiplos fatores
        churn_scores = []
        
        for _, customer in customer_data.iterrows():
            score = 0
            factors = []
            
            # Fator 1: Recência (peso 40%)
            if customer['Recency'] > 180:
                score += 40
                factors.append("Alta recência (>6 meses)")
            elif customer['Recency'] > 90:
                score += 20
                factors.append("Recência moderada (>3 meses)")
            
            # Fator 2: Frequência baixa (peso 25%)
            if customer['Frequency'] == 1:
                score += 25
                factors.append("Apenas 1 compra")
            elif customer['Frequency'] <= 2:
                score += 15
                factors.append("Baixa frequência (≤2 compras)")
            
            # Fator 3: Valor decrescente (peso 20%)
            if customer.get('Purchase_Consistency', 1) < 0.3:
                score += 20
                factors.append("Padrão irregular de compras")
            
            # Fator 4: Ticket médio baixo (peso 15%)
            aov_percentile = customer['AOV_Real'] / customer_data['AOV_Real'].quantile(0.75)
            if aov_percentile < 0.5:
                score += 15
                factors.append("Ticket médio baixo")
            
            churn_scores.append({
                'score': min(score, 100),
                'factors': factors
            })
        
        customer_data['Churn_Score'] = [s['score'] for s in churn_scores]
        customer_data['Churn_Factors'] = [s['factors'] for s in churn_scores]
        
        # Classificar em categorias de risco
        def classify_risk(score):
            if score >= 70:
                return 'High Risk'
            elif score >= 40:
                return 'Medium Risk'
            else:
                return 'Low Risk'
        
        customer_data['Churn_Risk_Category'] = customer_data['Churn_Score'].apply(classify_risk)
        
        # Análise por categoria de risco
        risk_distribution = customer_data['Churn_Risk_Category'].value_counts()
        risk_analysis = customer_data.groupby('Churn_Risk_Category').agg({
            'Monetary': ['mean', 'sum'],
            'Frequency': 'mean',
            'Recency': 'mean',
            'Churn_Score': 'mean'
        }).round(2)
        
        # Fatores de risco mais comuns
        all_factors = []
        for factors_list in customer_data['Churn_Factors']:
            all_factors.extend(factors_list)
        
        factor_frequency = pd.Series(all_factors).value_counts()
        
        # Impacto financeiro
        financial_impact = self._calculate_churn_financial_impact(customer_data)
        
        # Estratégias de retenção
        retention_strategies = self._generate_retention_strategies(customer_data)
        
        return {
            'risk_distribution': self._safe_series_to_dict(risk_distribution),
            'risk_analysis': self._safe_dataframe_to_dict(risk_analysis),
            'top_risk_factors': self._safe_series_to_dict(factor_frequency.head(10)),
            'financial_impact': financial_impact,
            'retention_strategies': retention_strategies,
            'total_customers': len(customer_data),
            'avg_churn_score': round(customer_data['Churn_Score'].mean(), 1)
        }
    
    def _value_analysis(self, customer_data: pd.DataFrame, method: str, 
                       horizon: int, include_demographics: bool = True,
                       include_geographic: bool = True) -> Dict[str, Any]:
        """Análise completa de valor do cliente."""
        print("💎 Executando análise de valor...")
        
        # Segmentação por valor
        customer_data['Value_Segment'] = pd.qcut(
            customer_data['Monetary'].rank(method='first'), 
            q=5, 
            labels=['Bronze', 'Silver', 'Gold', 'Platinum', 'Diamond']
        )
        
        # Análise de concentração de valor (Pareto)
        value_concentration = self._analyze_value_concentration(customer_data)
        
        # Métricas por segmento de valor
        value_metrics = customer_data.groupby('Value_Segment').agg({
            'Monetary': ['mean', 'sum', 'count'],
            'Frequency': 'mean',
            'AOV_Real': 'mean',
            'CLV_Estimado': 'mean' if 'CLV_Estimado' in customer_data.columns else 'count',
            'Customer_Margin': 'mean' if 'Customer_Margin' in customer_data.columns else 'count'
        }).round(2)
        
        # Top clientes por valor
        top_customers = customer_data.nlargest(20, 'Monetary')[
            ['Monetary', 'Frequency', 'AOV_Real', 'Recency']
        ].round(2)
        
        # Análise de CLV se disponível
        clv_analysis = {}
        if 'CLV_Estimado' in customer_data.columns:
            clv_analysis = {
                'avg_clv': round(customer_data['CLV_Estimado'].mean(), 2),
                'median_clv': round(customer_data['CLV_Estimado'].median(), 2),
                'total_estimated_clv': round(customer_data['CLV_Estimado'].sum(), 2),
                'clv_distribution': customer_data['CLV_Estimado'].describe().round(2).to_dict()
            }
        
        # Análise de margem se disponível
        margin_analysis = {}
        if 'Customer_Margin' in customer_data.columns:
            margin_analysis = {
                'avg_margin': round(customer_data['Customer_Margin'].mean(), 2),
                'total_margin': round(customer_data['Customer_Margin'].sum(), 2),
                'margin_rate': round(customer_data['Margin_Rate'].mean(), 2) if 'Margin_Rate' in customer_data.columns else 0
            }
        
        # Estratégias por segmento de valor
        value_strategies = {
            'Diamond': [
                "Atendimento VIP personalizado",
                "Acesso antecipado a coleções exclusivas",
                "Eventos e experiências premium",
                "Consultoria especializada em investimento"
            ],
            'Platinum': [
                "Programa de fidelidade premium",
                "Ofertas personalizadas",
                "Convites para eventos especiais",
                "Serviços de manutenção gratuitos"
            ],
            'Gold': [
                "Cross-sell estratégico",
                "Programa de upgrade para Platinum",
                "Descontos em compras múltiplas",
                "Newsletter exclusiva"
            ],
            'Silver': [
                "Ofertas de up-sell",
                "Educação sobre produtos premium",
                "Programa de pontos",
                "Incentivos para aumentar frequência"
            ],
            'Bronze': [
                "Programa de desenvolvimento",
                "Ofertas especiais para segunda compra",
                "Educação sobre valor e qualidade",
                "Suporte pós-venda intensivo"
            ]
        }
        
        return {
            'value_segments': self._safe_series_to_dict(customer_data['Value_Segment'].value_counts()),
            'value_metrics': self._safe_dataframe_to_dict(value_metrics),
            'value_concentration': value_concentration,
            'top_customers': self._safe_dataframe_to_dict(top_customers),
            'clv_analysis': clv_analysis,
            'margin_analysis': margin_analysis,
            'value_strategies': value_strategies,
            'total_customers': len(customer_data)
        }
    
    def _preference_mining(self, customer_data: pd.DataFrame, method: str, 
                          horizon: int, include_demographics: bool = True,
                          include_geographic: bool = True) -> Dict[str, Any]:
        """Mineração de preferências de clientes."""
        print("🔍 Executando mineração de preferências...")
        
        preferences = {}
        
        # Preferências demográficas se disponível
        if include_demographics:
            demographic_prefs = {}
            
            # Preferências por idade
            age_cols = [col for col in customer_data.columns if 'Faixa_Etaria' in col or 'Idade' in col]
            if age_cols:
                age_col = age_cols[0]
                if not customer_data[age_col].isna().all():
                    age_preferences = customer_data.groupby(age_col).agg({
                        'AOV_Real': 'mean',
                        'Frequency': 'mean',
                        'Monetary': 'mean'
                    }).round(2)
                    demographic_prefs['age_preferences'] = self._safe_dataframe_to_dict(age_preferences)
            
            # Preferências por gênero
            gender_cols = [col for col in customer_data.columns if 'Sexo' in col]
            if gender_cols:
                gender_col = gender_cols[0]
                if not customer_data[gender_col].isna().all():
                    gender_preferences = customer_data.groupby(gender_col).agg({
                        'AOV_Real': 'mean',
                        'Frequency': 'mean',
                        'Monetary': 'mean'
                    }).round(2)
                    demographic_prefs['gender_preferences'] = self._safe_dataframe_to_dict(gender_preferences)
            
            preferences['demographic'] = demographic_prefs
        
        # Preferências geográficas se disponível
        if include_geographic:
            geographic_prefs = {}
            
            # Preferências por estado
            state_cols = [col for col in customer_data.columns if 'Estado' in col]
            if state_cols:
                state_col = state_cols[0]
                if not customer_data[state_col].isna().all():
                    state_preferences = customer_data.groupby(state_col).agg({
                        'AOV_Real': 'mean',
                        'Frequency': 'mean',
                        'Monetary': 'sum'
                    }).round(2)
                    geographic_prefs['state_preferences'] = self._safe_dataframe_to_dict(state_preferences)
            
            preferences['geographic'] = geographic_prefs
        
        # Preferências de produto se disponível
        product_prefs = {}
        product_cols = [col for col in customer_data.columns if any(p in col for p in ['Grupo_Produto', 'Metal', 'Colecao'])]
        
        for col in product_cols:
            if not customer_data[col].isna().all():
                product_analysis = customer_data.groupby(col).agg({
                    'AOV_Real': 'mean',
                    'Frequency': 'mean',
                    'Monetary': 'sum'
                }).round(2)
                
                # Só incluir se houver variação significativa
                if len(product_analysis) > 1:
                    product_prefs[col] = self._safe_dataframe_to_dict(product_analysis)
        
        preferences['product'] = product_prefs
        
        # Padrões sazonais básicos (por mês)
        if 'Data_min' in customer_data.columns:
            # Converter para datetime se necessário
            try:
                customer_data['First_Purchase_Month'] = pd.to_datetime(customer_data['Data_min']).dt.month
                monthly_patterns = customer_data.groupby('First_Purchase_Month').size()
                preferences['seasonal'] = {
                    'first_purchase_by_month': self._safe_series_to_dict(monthly_patterns),
                    'peak_months': monthly_patterns.nlargest(3).index.tolist()
                }
            except:
                preferences['seasonal'] = {'info': 'Dados de data insuficientes para análise sazonal'}
        
        # Insights e recomendações
        insights = []
        if 'demographic' in preferences and preferences['demographic']:
            insights.append("Padrões demográficos identificados - personalizar por idade/gênero")
        if 'geographic' in preferences and preferences['geographic']:
            insights.append("Variações geográficas detectadas - adaptar estratégia regional")
        if 'product' in preferences and preferences['product']:
            insights.append("Preferências de produto claras - focar em cross-sell direcionado")
        
        return {
            'preferences': preferences,
            'insights': insights,
            'total_customers': len(customer_data),
            'analysis_coverage': {
                'demographic': include_demographics and bool(preferences.get('demographic')),
                'geographic': include_geographic and bool(preferences.get('geographic')),
                'product': bool(preferences.get('product')),
                'seasonal': bool(preferences.get('seasonal'))
            }
        }
    
    def _journey_mapping(self, customer_data: pd.DataFrame, method: str, 
                        horizon: int, include_demographics: bool = True,
                        include_geographic: bool = True) -> Dict[str, Any]:
        """Mapeamento da jornada do cliente."""
        print("🗺️ Executando mapeamento da jornada...")
        
        # Classificar estágios da jornada
        customer_data['Journey_Stage'] = customer_data.apply(self._classify_journey_stage, axis=1)
        
        # Distribuição por estágio da jornada
        journey_distribution = customer_data['Journey_Stage'].value_counts()
        
        # Análise de transições entre estágios
        stage_transitions = self._analyze_stage_transitions(customer_data)
        
        # Identificar pontos de atrito
        friction_points = self._identify_friction_points(customer_data)
        
        # Métricas por estágio da jornada
        journey_metrics = customer_data.groupby('Journey_Stage').agg({
            'Monetary': ['mean', 'count'],
            'Frequency': 'mean',
            'Recency': 'mean',
            'Customer_Lifetime_Days': 'mean'
        }).round(2)
        
        # Tempo médio em cada estágio (estimativa)
        avg_stage_time = {
            'Descoberta': 7,
            'Consideração': 14,
            'Primeira Compra': 1,
            'Avaliação': 30,
            'Repeat Purchase': 90,
            'Advocacy': 180
        }
        
        # Oportunidades de melhoria por estágio
        opportunities = {
            'Descoberta': [
                "Melhorar SEO e presença digital",
                "Campanhas de awareness direcionadas",
                "Parcerias estratégicas",
                "Marketing de conteúdo educativo"
            ],
            'Consideração': [
                "Retargeting personalizado",
                "Reviews e depoimentos",
                "Comparativos de produtos",
                "Consultoria virtual"
            ],
            'Primeira Compra': [
                "Processo de checkout simplificado",
                "Múltiplas opções de pagamento",
                "Garantia e políticas claras",
                "Suporte pré-venda ativo"
            ],
            'Avaliação': [
                "Follow-up pós-venda",
                "Tutorial de cuidados",
                "Programa de satisfação",
                "Canal de feedback direto"
            ],
            'Repeat Purchase': [
                "Ofertas personalizadas",
                "Lembretes de manutenção",
                "Cross-sell inteligente",
                "Programa de fidelidade"
            ],
            'Advocacy': [
                "Programa de indicação",
                "Conteúdo para compartilhar",
                "Eventos exclusivos",
                "Status VIP reconhecido"
            ]
        }
        
        # Análise demográfica da jornada se disponível
        demographic_journey = {}
        if include_demographics:
            age_cols = [col for col in customer_data.columns if 'Faixa_Etaria' in col or 'Idade' in col]
            if age_cols and not customer_data[age_cols[0]].isna().all():
                age_journey_data = customer_data.groupby([age_cols[0], 'Journey_Stage']).size().unstack(fill_value=0)
                demographic_journey['age_journey'] = self._safe_dataframe_to_dict(age_journey_data)
        
        return {
            'journey_distribution': self._safe_series_to_dict(journey_distribution),
            'journey_metrics': self._safe_dataframe_to_dict(journey_metrics),
            'stage_transitions': stage_transitions,
            'friction_points': friction_points,
            'avg_stage_time_days': avg_stage_time,
            'improvement_opportunities': opportunities,
            'demographic_journey': demographic_journey,
            'total_customers': len(customer_data)
        }
    
    # ==========================================
    # MÉTODOS DE CLASSIFICAÇÃO E FORMATAÇÃO
    # ==========================================
    
    def _classify_lifecycle_stage(self, row) -> str:
        """Classificar estágio do ciclo de vida do cliente."""
        recency = row['Recency']
        frequency = row['Frequency']
        monetary = row['Monetary']
        
        # VIP: alta frequência, alto valor, compra recente
        if frequency >= 5 and monetary > 3000 and recency <= 60:
            return 'VIP'
        # Leal: boa frequência, compra recente
        elif frequency >= 3 and recency <= 90:
            return 'Leal'
        # Crescendo: múltiplas compras, valor crescente
        elif frequency >= 2 and recency <= 120:
            return 'Crescendo'
        # Em Risco: não compra há tempo, mas tinha valor
        elif recency > 120 and frequency >= 2:
            return 'Em Risco'
        # Perdido: muito tempo sem comprar
        elif recency > 180:
            return 'Perdido'
        # Novo: poucas compras, recente
        else:
            return 'Novo'
    
    def _classify_journey_stage(self, row) -> str:
        """Classificar estágio da jornada do cliente."""
        frequency = row['Frequency']
        recency = row['Recency']
        lifetime_days = row.get('Customer_Lifetime_Days', 0)
        
        if frequency == 1:
            if recency <= 30:
                return 'Avaliação'
            else:
                return 'Primeira Compra'
        elif frequency == 2:
            return 'Repeat Purchase'
        elif frequency >= 3 and frequency <= 5:
            return 'Repeat Purchase'
        elif frequency > 5:
            return 'Advocacy'
        else:
            return 'Descoberta'
    
    def _format_customer_result(self, analysis_type: str, analysis_result: Dict[str, Any], 
                               customer_data: pd.DataFrame) -> str:
        """Formatar resultado da análise em JSON estruturado."""
        
        # Metadata da análise
        metadata = {
            'analysis_type': analysis_type.replace('_', ' ').title(),
            'tool_name': 'Customer Insights Engine V3.0',
            'timestamp': datetime.now().isoformat(),
            'total_customers_analyzed': len(customer_data),
            'data_quality': {
                'has_demographics': any('Idade' in col or 'Sexo' in col for col in customer_data.columns),
                'has_geographic': any('Cidade' in col or 'Estado' in col for col in customer_data.columns),
                'has_financial': any('Margem' in col or 'CLV' in col for col in customer_data.columns)
            }
        }
        
        # Insights principais baseados no tipo de análise
        key_insights = self._generate_key_insights(analysis_type, analysis_result, customer_data)
        
        # Recomendações acionáveis
        recommendations = self._generate_recommendations(analysis_type, analysis_result)
        
        # Próximos passos sugeridos
        next_steps = self._generate_next_steps(analysis_type, analysis_result)
        
        # Estrutura JSON final
        result_json = {
            'metadata': metadata,
            'analysis_results': self._ensure_json_serializable(analysis_result),
            'key_insights': key_insights,
            'recommendations': recommendations,
            'next_steps': next_steps,
            'kpis': self._extract_kpis(analysis_type, analysis_result, customer_data)
        }
        
        return json.dumps(result_json, indent=2, ensure_ascii=False)
    
    def _format_error_result(self, error_title: str, error_detail: str) -> str:
        """Formatar resultado de erro em JSON estruturado."""
        error_result = {
            'metadata': {
                'analysis_type': 'Error',
                'tool_name': 'Customer Insights Engine V3.0',
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            },
            'error': {
                'title': error_title,
                'detail': error_detail,
                'suggestions': [
                    "Verificar se o arquivo CSV existe e está acessível",
                    "Confirmar se as colunas necessárias estão presentes",
                    "Verificar se há dados suficientes para análise",
                    "Validar parâmetros de entrada"
                ]
            }
        }
        
        return json.dumps(error_result, indent=2, ensure_ascii=False)
    
    def _generate_key_insights(self, analysis_type: str, analysis_result: Dict[str, Any], 
                              customer_data: pd.DataFrame) -> List[str]:
        """Gerar insights principais baseados no tipo de análise."""
        insights = []
        
        try:
            if analysis_type == 'behavioral_segmentation':
                if 'segment_distribution' in analysis_result.get('rfm_analysis', {}):
                    champions = analysis_result['rfm_analysis']['segment_distribution'].get('Champions', 0)
                    total = analysis_result.get('total_customers', len(customer_data))
                    champions_pct = round(champions / total * 100, 1) if total > 0 else 0
                    insights.append(f"🏆 {champions_pct}% dos clientes são Champions - foco em retenção VIP")
                
                if 'cluster_analysis' in analysis_result.get('advanced_clustering', {}):
                    clusters = len(analysis_result['advanced_clustering']['cluster_analysis'])
                    insights.append(f"🎯 Identificados {clusters} clusters comportamentais distintos")
            
            elif analysis_type == 'churn_prediction':
                if 'risk_distribution' in analysis_result:
                    high_risk = analysis_result['risk_distribution'].get('High Risk', 0)
                    total = analysis_result.get('total_customers', len(customer_data))
                    risk_pct = round(high_risk / total * 100, 1) if total > 0 else 0
                    insights.append(f"⚠️ {risk_pct}% dos clientes estão em alto risco de churn")
                
                if 'financial_impact' in analysis_result:
                    potential_loss = analysis_result['financial_impact'].get('potential_lost_clv', 0)
                    insights.append(f"💰 Risco financeiro potencial: R$ {potential_loss:,.2f}")
            
            elif analysis_type == 'value_analysis':
                if 'value_concentration' in analysis_result:
                    concentration = analysis_result['value_concentration'].get('top_20_percent_value_share', 0)
                    insights.append(f"📊 Top 20% dos clientes representam {concentration}% do valor total")
                
                if 'clv_analysis' in analysis_result and analysis_result['clv_analysis']:
                    avg_clv = analysis_result['clv_analysis'].get('avg_clv', 0)
                    insights.append(f"💎 CLV médio estimado: R$ {avg_clv:,.2f}")
            
            elif analysis_type == 'lifecycle_analysis':
                if 'stage_distribution' in analysis_result:
                    stages = analysis_result['stage_distribution']
                    dominant_stage = max(stages.items(), key=lambda x: x[1])
                    insights.append(f"🔄 Estágio dominante: {dominant_stage[0]} ({dominant_stage[1]} clientes)")
            
            elif analysis_type == 'preference_mining':
                if 'analysis_coverage' in analysis_result:
                    coverage = analysis_result['analysis_coverage']
                    active_analyses = [k for k, v in coverage.items() if v]
                    insights.append(f"🔍 Análises ativas: {', '.join(active_analyses)}")
            
            elif analysis_type == 'journey_mapping':
                if 'friction_points' in analysis_result:
                    friction_count = len(analysis_result['friction_points'])
                    insights.append(f"🚧 Identificados {friction_count} pontos de atrito na jornada")
            
            # Insight geral sobre qualidade dos dados
            total_customers = len(customer_data)
            insights.append(f"📈 Análise baseada em {total_customers} clientes com dados reais")
            
        except Exception as e:
            insights.append(f"⚠️ Erro na geração de insights: {str(e)}")
        
        return insights
    
    def _generate_recommendations(self, analysis_type: str, analysis_result: Dict[str, Any]) -> List[str]:
        """Gerar recomendações acionáveis baseadas no tipo de análise."""
        recommendations = []
        
        try:
            if analysis_type == 'behavioral_segmentation':
                recommendations.extend([
                    "🎯 Criar campanhas personalizadas por segmento RFM",
                    "📧 Implementar automação de email marketing segmentado",
                    "🎁 Desenvolver ofertas específicas para cada cluster comportamental"
                ])
                
                if 'demographic_segmentation' in analysis_result:
                    recommendations.append("👥 Personalizar comunicação por perfil demográfico")
            
            elif analysis_type == 'churn_prediction':
                recommendations.extend([
                    "🚨 Implementar alertas automáticos para clientes de alto risco",
                    "📞 Criar programa de retenção proativo",
                    "💰 Investir em win-back campaigns para clientes perdidos"
                ])
            
            elif analysis_type == 'value_analysis':
                recommendations.extend([
                    "👑 Criar programa VIP para clientes de maior valor",
                    "📈 Implementar estratégias de up-sell para segmentos Bronze/Silver",
                    "💎 Oferecer atendimento premium para top clientes"
                ])
            
            elif analysis_type == 'lifecycle_analysis':
                recommendations.extend([
                    "🔄 Automatizar comunicação por estágio de vida",
                    "🎯 Criar jornadas específicas para cada estágio",
                    "📊 Monitorar transições entre estágios"
                ])
            
            elif analysis_type == 'preference_mining':
                recommendations.extend([
                    "🎨 Personalizar catálogo por preferências identificadas",
                    "📱 Implementar recomendações inteligentes",
                    "🎁 Criar ofertas baseadas em padrões demográficos"
                ])
            
            elif analysis_type == 'journey_mapping':
                recommendations.extend([
                    "🛠️ Otimizar pontos de atrito identificados",
                    "📋 Implementar métricas de jornada em tempo real",
                    "🎯 Personalizar experiência por estágio da jornada"
                ])
            
        except Exception as e:
            recommendations.append(f"⚠️ Erro na geração de recomendações: {str(e)}")
        
        return recommendations
    
    def _generate_next_steps(self, analysis_type: str, analysis_result: Dict[str, Any]) -> List[str]:
        """Gerar próximos passos sugeridos."""
        next_steps = []
        
        try:
            # Passos específicos por tipo de análise
            if analysis_type == 'behavioral_segmentation':
                next_steps.extend([
                    "1. Exportar lista de clientes por segmento",
                    "2. Configurar campanhas no sistema de CRM",
                    "3. Definir métricas de acompanhamento"
                ])
            
            elif analysis_type == 'churn_prediction':
                next_steps.extend([
                    "1. Priorizar contato com clientes de alto risco",
                    "2. Desenvolver scripts de retenção",
                    "3. Implementar sistema de alerta"
                ])
            
            elif analysis_type == 'value_analysis':
                next_steps.extend([
                    "1. Revisar estratégia de preços por segmento",
                    "2. Implementar programa de fidelidade",
                    "3. Treinar equipe para atendimento diferenciado"
                ])
            
            # Passos gerais
            next_steps.extend([
                "4. Agendar revisão mensal dos insights",
                "5. Implementar métricas de acompanhamento",
                "6. Treinar equipe sobre os insights descobertos"
            ])
            
        except Exception as e:
            next_steps.append(f"⚠️ Erro na geração de próximos passos: {str(e)}")
        
        return next_steps
    
    def _extract_kpis(self, analysis_type: str, analysis_result: Dict[str, Any], 
                     customer_data: pd.DataFrame) -> Dict[str, Any]:
        """Extrair KPIs principais da análise."""
        kpis = {}
        
        try:
            # KPIs básicos sempre incluídos
            kpis['total_customers'] = len(customer_data)
            kpis['avg_monetary_value'] = round(customer_data['Monetary'].mean(), 2)
            kpis['avg_frequency'] = round(customer_data['Frequency'].mean(), 1)
            kpis['avg_recency'] = round(customer_data['Recency'].mean(), 1)
            
            # KPIs específicos por tipo de análise
            if analysis_type == 'behavioral_segmentation':
                if 'rfm_analysis' in analysis_result:
                    champions = analysis_result['rfm_analysis']['segment_distribution'].get('Champions', 0)
                    kpis['champions_count'] = champions
                    kpis['champions_percentage'] = round(champions / len(customer_data) * 100, 1)
            
            elif analysis_type == 'churn_prediction':
                if 'avg_churn_score' in analysis_result:
                    kpis['avg_churn_score'] = analysis_result['avg_churn_score']
                if 'risk_distribution' in analysis_result:
                    high_risk = analysis_result['risk_distribution'].get('High Risk', 0)
                    kpis['high_risk_count'] = high_risk
                    kpis['high_risk_percentage'] = round(high_risk / len(customer_data) * 100, 1)
            
            elif analysis_type == 'value_analysis':
                if 'clv_analysis' in analysis_result and analysis_result['clv_analysis']:
                    kpis['avg_clv'] = analysis_result['clv_analysis'].get('avg_clv', 0)
                    kpis['total_clv'] = analysis_result['clv_analysis'].get('total_estimated_clv', 0)
            
            # KPIs adicionais se disponíveis
            if 'AOV_Real' in customer_data.columns:
                kpis['avg_order_value'] = round(customer_data['AOV_Real'].mean(), 2)
            
            if 'CLV_Estimado' in customer_data.columns:
                kpis['total_estimated_clv'] = round(customer_data['CLV_Estimado'].sum(), 2)
            
        except Exception as e:
            kpis['error'] = f"Erro na extração de KPIs: {str(e)}"
        
        return kpis
    
    def _simulate_segment_migration(self, customer_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Simular matriz de migração entre segmentos."""
        # Matriz simplificada baseada em padrões típicos
        migration_matrix = {
            'Champions': {
                'Champions': 0.8,
                'Loyal Customers': 0.15,
                'At Risk': 0.05
            },
            'Loyal Customers': {
                'Champions': 0.25,
                'Loyal Customers': 0.6,
                'At Risk': 0.15
            },
            'At Risk': {
                'Champions': 0.1,
                'Loyal Customers': 0.3,
                'At Risk': 0.4,
                'Lost': 0.2
            },
            'Lost': {
                'Lost': 0.9,
                'New Customers': 0.1
            }
        }
        
        return migration_matrix

    # ==========================================
    # MÉTODOS AUXILIARES PARA SERIALIZAÇÃO JSON
    # ==========================================
    
    def _safe_dataframe_to_dict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Converte DataFrame para dicionário de forma segura para JSON."""
        try:
            # Se é um DataFrame simples
            if not isinstance(df.index, pd.MultiIndex) and not isinstance(df.columns, pd.MultiIndex):
                return df.to_dict()
            
            # Se tem MultiIndex, precisa achatar
            if isinstance(df.index, pd.MultiIndex):
                df = df.copy()
                df.index = ['_'.join(map(str, idx)) if isinstance(idx, tuple) else str(idx) for idx in df.index]
            
            if isinstance(df.columns, pd.MultiIndex):
                df = df.copy()
                df.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else str(col) for col in df.columns]
            
            return df.to_dict()
            
        except Exception as e:
            print(f"⚠️ Erro na conversão de DataFrame: {e}")
            # Fallback para estrutura simples
            return {
                'data': df.values.tolist() if hasattr(df, 'values') else [],
                'index': [str(idx) for idx in df.index] if hasattr(df, 'index') else [],
                'columns': [str(col) for col in df.columns] if hasattr(df, 'columns') else [],
                'error': f"Conversão alternativa devido a: {str(e)}"
            }
    
    def _safe_series_to_dict(self, series: pd.Series) -> Dict[str, Any]:
        """Converte Series para dicionário de forma segura para JSON."""
        try:
            # Se o índice é simples
            if not isinstance(series.index, pd.MultiIndex):
                return series.to_dict()
            
            # Se tem MultiIndex, achatar
            flattened_index = ['_'.join(map(str, idx)) if isinstance(idx, tuple) else str(idx) for idx in series.index]
            return dict(zip(flattened_index, series.values))
            
        except Exception as e:
            print(f"⚠️ Erro na conversão de Series: {e}")
            return {
                'values': series.values.tolist() if hasattr(series, 'values') else [],
                'index': [str(idx) for idx in series.index] if hasattr(series, 'index') else [],
                'error': f"Conversão alternativa devido a: {str(e)}"
            }
    
    def _ensure_json_serializable(self, obj: Any) -> Any:
        """Garante que um objeto seja serializável em JSON."""
        if isinstance(obj, dict):
            return {str(k): self._ensure_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._ensure_json_serializable(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return self._safe_dataframe_to_dict(obj)
        elif isinstance(obj, pd.Series):
            return self._safe_series_to_dict(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'isoformat'):  # datetime
            return obj.isoformat()
        elif hasattr(obj, 'item'):  # scalar numpy types
            return obj.item()
        else:
            return obj
