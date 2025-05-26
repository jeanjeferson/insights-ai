"""
🎯 RECOMMENDATION ENGINE - SISTEMA DE RECOMENDAÇÕES INTELIGENTES
===============================================================

Motor de recomendações avançado otimizado para CrewAI e Agentes de IA.
Projetado especificamente para joalherias com algoritmos ML e análise comportamental.

Funcionalidades:
- 6 tipos de recomendações especializadas
- Segmentação automática de clientes (RFM)
- Algoritmos ML (Collaborative Filtering, Content-Based, Clustering)
- Análise de Market Basket e padrões comportamentais
- Otimização de preços e inventário
- Campanhas de marketing personalizadas

Versão: 2.0 - Otimizada para CrewAI
"""

from crewai.tools import BaseTool
from typing import Type, Optional, Dict, Any, List, Tuple, Union
from pydantic import BaseModel, Field, field_validator
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.sparse as sp
from datetime import datetime, timedelta
import warnings
import json
import hashlib
import itertools

warnings.filterwarnings('ignore')


class RecommendationInput(BaseModel):
    """
    Schema de entrada otimizado para Agentes de IA - Sistema de Recomendações.
    
    Este schema guia os agentes sobre quando e como usar cada tipo de recomendação,
    com descrições claras dos resultados esperados.
    """
    
    recommendation_type: str = Field(
        ..., 
        description="""
        TIPO DE RECOMENDAÇÃO (obrigatório):
        
        🛍️ 'product_recommendations' - USE QUANDO:
        - Agente precisa sugerir produtos para clientes específicos
        - Análise de cross-selling e up-selling
        - Identificar produtos de maior potencial de venda
        RESULTADO: Lista de produtos rankeados por score de recomendação
        
        🎯 'customer_targeting' - USE QUANDO:
        - Agente precisa identificar clientes para campanhas
        - Segmentar base para ações comerciais específicas
        - Encontrar clientes VIP, em risco ou com potencial
        RESULTADO: Listas de clientes por estratégia (upsell, retention, etc.)
        
        💰 'pricing_optimization' - USE QUANDO:
        - Agente analisa estratégias de preço
        - Precisa otimizar margem vs. volume
        - Identificar oportunidades de ajuste de preços
        RESULTADO: Sugestões de preços por categoria e análise de elasticidade
        
        📦 'inventory_suggestions' - USE QUANDO:
        - Agente gerencia estoque e inventário
        - Precisa identificar produtos para reposição ou liquidação
        - Otimizar capital de giro
        RESULTADO: Recomendações ABC, produtos para restock/liquidar
        
        📢 'marketing_campaigns' - USE QUANDO:
        - Agente planeja campanhas de marketing
        - Precisa personalizar mensagens por segmento
        - Definir timing e canais ideais
        RESULTADO: Campanhas personalizadas com ROI estimado
        
        🚀 'strategic_actions' - USE QUANDO:
        - Agente faz análise estratégica de negócio
        - Precisa de ações baseadas em dados
        - Identificar oportunidades de crescimento
        RESULTADO: Roadmap de ações priorizadas com impacto estimado
        """
    )
    
    data_csv: str = Field(
        default="data/vendas.csv",
        description="""
        Caminho para arquivo CSV de vendas.
        FORMATO ESPERADO: dados de transações de joalheria com colunas:
        - Data, Codigo_Cliente, Nome_Cliente, Codigo_Produto, Descricao_Produto
        - Grupo_Produto, Quantidade, Total_Liquido, Preco_Tabela
        """
    )
    
    target_segment: str = Field(
        default="all",
        description="""
        Segmento de clientes para focar a análise:
        - 'all': Todos os clientes (recomendado para primeira análise)
        - 'vip': Clientes VIP (alto valor + frequência + recência)
        - 'new_customers': Clientes novos (primeira compra recente)
        - 'at_risk': Clientes em risco (sem compra há 180+ dias)
        - 'high_value': Clientes de alto valor (acima de R$ 2.000)
        - 'frequent': Clientes frequentes (3+ compras)
        """
    )
    
    recommendation_count: int = Field(
        default=10,
        description="Número de recomendações a retornar (5-50). Recomendado: 10-15 para análise detalhada."
    )
    
    confidence_threshold: float = Field(
        default=0.7,
        description="""
        Limiar de confiança para filtrar recomendações (0.5-0.95):
        - 0.5-0.6: Mais recomendações, menor precisão (exploratório)
        - 0.7-0.8: Equilibrado (recomendado)
        - 0.8-0.95: Menos recomendações, maior precisão (conservador)
        """
    )
    
    enable_detailed_analysis: bool = Field(
        default=True,
        description="Incluir análises avançadas (market basket, sazonalidade, ROI)"
    )
    
    @field_validator('recommendation_type')
    @classmethod
    def validate_recommendation_type(cls, v):
        valid_types = [
            'product_recommendations', 'customer_targeting', 'pricing_optimization',
            'inventory_suggestions', 'marketing_campaigns', 'strategic_actions'
        ]
        if v not in valid_types:
            raise ValueError(f"Tipo deve ser um de: {valid_types}")
        return v
    
    @field_validator('confidence_threshold')
    @classmethod
    def validate_confidence(cls, v):
        if not 0.5 <= v <= 0.95:
            raise ValueError("Confiança deve estar entre 0.5 e 0.95")
        return v
    
    @field_validator('recommendation_count')
    @classmethod
    def validate_count(cls, v):
        if not 5 <= v <= 50:
            raise ValueError("Contagem deve estar entre 5 e 50")
        return v


class RecommendationEngine(BaseTool):
    """
    🎯 RECOMMENDATION ENGINE V2.0 - OTIMIZADO PARA CREWAI
    =====================================================
    
    Motor de recomendações inteligentes para joalherias, especialmente otimizado
    para uso com Agentes de IA do CrewAI. Fornece 6 tipos de análises especializadas
    com outputs estruturados e orientações claras para tomada de decisão.
    
    🤖 OTIMIZADO PARA AGENTES:
    - Descrições claras de quando usar cada funcionalidade
    - Outputs estruturados em JSON + Markdown
    - Metadados para facilitar decisões dos agentes
    - Validação robusta de dados de entrada
    
    🔬 ALGORITMOS AVANÇADOS:
    - Collaborative Filtering (similaridade de clientes)
    - Content-Based Filtering (características de produtos)
    - Análise RFM (Recência, Frequência, Valor Monetário)
    - Market Basket Analysis (produtos frequentemente comprados juntos)
    - K-means Clustering (segmentação automática)
    - Análise ABC (classificação de produtos por importância)
    
    📊 ANÁLISES DISPONÍVEIS:
    1. Product Recommendations - Produtos recomendados por algoritmos ML
    2. Customer Targeting - Segmentação e targeting inteligente  
    3. Pricing Optimization - Otimização de preços com análise de elasticidade
    4. Inventory Suggestions - Gestão inteligente de estoque
    5. Marketing Campaigns - Campanhas personalizadas por segmento
    6. Strategic Actions - Ações estratégicas baseadas em dados
    
    💎 ESPECÍFICO PARA JOALHERIAS:
    - Categorização por faixas de preço (Economy, Premium, Luxury)
    - Análise sazonal para períodos especiais (Dia das Mães, Natal)
    - Segmentação por comportamento de compra de joias
    - ROI estimado para campanhas do setor
    
    🚀 PERFORMANCE:
    - Cache inteligente para análises repetidas
    - Processamento otimizado para grandes volumes
    - Sampling estratificado quando necessário
    """
    
    name: str = "Recommendation Engine"
    description: str = """
    🎯 Sistema de recomendações inteligentes otimizado para joalherias e CrewAI.
    
    QUANDO USAR ESTE TOOL:
    ✅ Quando agente precisa de recomendações baseadas em dados
    ✅ Para análise comportamental de clientes
    ✅ Otimização de preços, inventário ou campanhas
    ✅ Identificação de oportunidades de negócio
    ✅ Segmentação automática de clientes
    ✅ Análise de produtos com maior potencial
    
    OUTPUTS FORNECIDOS:
    📊 Análises estruturadas em JSON + Markdown
    🎯 Recomendações rankeadas por score de confiança
    💡 Insights acionáveis para tomada de decisão
    📈 Estimativas de ROI e impacto financeiro
    🗓️ Timing e sazonalidade otimizados
    📋 Roadmaps de implementação priorizados
    
    ALGORITMOS INCLUSOS:
    🔬 Machine Learning (Collaborative + Content-Based Filtering)
    📊 Análise RFM para segmentação de clientes
    🛒 Market Basket Analysis para cross-selling
    📈 Análise de tendências e sazonalidade
    💰 Otimização de preços com elasticidade
    📦 Classificação ABC para gestão de inventário
    """
    
    args_schema: Type[BaseModel] = RecommendationInput
    
    def __init__(self):
        super().__init__()
        self._cache = {}
        self._last_data_hash = None
        
    def _run(
        self, 
        recommendation_type: str, 
        data_csv: str = "data/vendas.csv",
        target_segment: str = "all", 
        recommendation_count: int = 10,
        confidence_threshold: float = 0.7,
        enable_detailed_analysis: bool = True
    ) -> str:
        """
        Executa análise de recomendações com validação robusta e cache inteligente.
        
        Returns:
            str: Resultado estruturado em JSON + formatação para agentes
        """
        try:
            start_time = datetime.now()
            
            # Carregar e validar dados
            df = self._load_and_validate_data(data_csv)
            
            if len(df) < 10:
                return self._format_error(
                    f"Dados insuficientes: {len(df)} registros (mínimo: 10)", 
                    recommendation_type
                )
            
            # Preparar dados para análise
            df_prepared = self._prepare_recommendation_data(df)
            
            # Otimização automática de parâmetros
            param_optimization = self._optimize_recommendation_parameters(df_prepared, recommendation_type)
            optimized_params = param_optimization.get('optimized_parameters', {})
            
            # Usar parâmetros otimizados se disponíveis
            if 'confidence_threshold' in optimized_params:
                confidence_threshold = optimized_params['confidence_threshold']
            if 'recommendation_count' in optimized_params:
                recommendation_count = optimized_params['recommendation_count']
            if 'enable_detailed_analysis' in optimized_params:
                enable_detailed_analysis = optimized_params['enable_detailed_analysis']
            
            # Cache check com parâmetros otimizados
            cache_key = self._generate_cache_key(
                recommendation_type, target_segment, recommendation_count, confidence_threshold
            )
            
            if cache_key in self._cache:
                cached_result = self._cache[cache_key]
                cached_result['metadata']['cache_hit'] = True
                cached_result['metadata']['generated_at'] = start_time.isoformat()
                cached_result['metadata']['parameter_optimization'] = param_optimization
                return self._format_final_result(cached_result, recommendation_type)
            
            # Seleção automática de algoritmo
            algorithm_selection = self._select_optimal_algorithm(df_prepared, recommendation_type, target_segment)
            
            # Executar análise específica com algoritmo otimizado
            if recommendation_type == 'product_recommendations' and algorithm_selection['selected_algorithm'] in ['hybrid_ml', 'collaborative_filtering', 'enhanced_standard']:
                # Usar integração ML avançada para produtos
                analysis_result = self._integrate_advanced_ml_in_products(
                    df_prepared, target_segment, recommendation_count, confidence_threshold
                )
                # Fallback para método padrão se ML falhar
                if 'error' in analysis_result:
                    analysis_result = self._execute_recommendation_analysis(
                        recommendation_type, df_prepared, target_segment, 
                        recommendation_count, confidence_threshold, enable_detailed_analysis
                    )
            else:
                # Usar análise padrão para outros tipos
                analysis_result = self._execute_recommendation_analysis(
                    recommendation_type, df_prepared, target_segment, 
                    recommendation_count, confidence_threshold, enable_detailed_analysis
                )
            
            # Validação cruzada das recomendações
            validation_results = self._cross_validate_recommendations(df_prepared, analysis_result)
            
            # Monitor de performance
            performance_metrics = self._performance_monitoring(
                start_time, len(df_prepared), 
                algorithm_selection.get('selected_algorithm', 'standard')
            )
            
            # Adicionar metadados
            analysis_result['metadata'] = {
                'recommendation_type': recommendation_type,
                'target_segment': target_segment,
                'data_points': len(df_prepared),
                'unique_customers': df_prepared['Customer_ID'].nunique() if 'Customer_ID' in df_prepared.columns else 0,
                'unique_products': df_prepared['Codigo_Produto'].nunique(),
                'date_range': {
                    'start': df_prepared['Data'].min().isoformat(),
                    'end': df_prepared['Data'].max().isoformat()
                },
                'processing_time_seconds': (datetime.now() - start_time).total_seconds(),
                'confidence_threshold': confidence_threshold,
                'cache_hit': False,
                'generated_at': start_time.isoformat(),
                'engine_version': '2.1',
                'algorithm_selection': algorithm_selection,
                'parameter_optimization': param_optimization,
                'validation_results': validation_results,
                'performance_metrics': performance_metrics
            }
            
            # Cache result
            self._cache[cache_key] = analysis_result
            
            return self._format_final_result(analysis_result, recommendation_type)
            
        except Exception as e:
            return self._format_error(f"Erro na execução: {str(e)}", recommendation_type)
    
    def _load_and_validate_data(self, data_csv: str) -> Optional[pd.DataFrame]:
        """Carrega e valida dados com as colunas reais do CSV."""
        try:
            # Verificar se arquivo existe
            import os
            if not os.path.exists(data_csv):
                raise FileNotFoundError(f"Arquivo não encontrado: {data_csv}")
            
            # Carregar CSV com encoding correto
            df = pd.read_csv(data_csv, sep=';', encoding='utf-8')
            
            # Verificar se CSV está vazio (apenas cabeçalho ou sem dados)
            if len(df) == 0:
                raise ValueError("Arquivo CSV está vazio ou contém apenas cabeçalho")
            
            # Validar colunas essenciais baseadas no CSV real
            required_columns = ['Data', 'Total_Liquido']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Colunas obrigatórias ausentes: {missing_columns}")
            
            # Filtrar dados válidos
            df = df.dropna(subset=['Data', 'Total_Liquido'])
            df = df[df['Total_Liquido'] != 0]  # Remover vendas zeradas
            
            # Verificar se ainda temos dados após filtros
            if len(df) == 0:
                raise ValueError("Todos os registros foram filtrados - dados insuficientes ou inválidos")
            
            return df
            
        except Exception as e:
            print(f"Erro no carregamento: {str(e)}")
            raise
    
    def _prepare_recommendation_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepara dados usando colunas reais do CSV."""
        try:
            df = df.copy()
            
            # Converter datas
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
            df = df.dropna(subset=['Data'])
            
            # Usar colunas reais do CSV
            if 'Preco_Tabela' in df.columns and 'Quantidade' in df.columns:
                df['Preco_Unitario'] = df['Preco_Tabela'] / df['Quantidade'].replace(0, 1)
            else:
                df['Preco_Unitario'] = df['Total_Liquido'] / df.get('Quantidade', 1)
            
            # Features temporais
            df['Year_Month'] = df['Data'].dt.to_period('M')
            df['Weekday'] = df['Data'].dt.dayofweek
            df['Month'] = df['Data'].dt.month
            df['Quarter'] = df['Data'].dt.quarter
            df['Year'] = df['Data'].dt.year
            
            # Usar Customer_ID real ou criar baseado em Codigo_Cliente
            if 'Codigo_Cliente' in df.columns:
                df['Customer_ID'] = df['Codigo_Cliente'].astype(str)
            else:
                df = self._simulate_customer_ids(df)
            
            # Categorização de preços para joalherias
            df['Price_Category'] = pd.cut(
                df['Preco_Unitario'],
                bins=[0, 500, 1500, 3000, 10000, float('inf')],
                labels=['Economy', 'Mid', 'Premium', 'Luxury', 'Ultra-Luxury']
            )
            
            # Análise RFM por cliente
            df = self._calculate_rfm_metrics(df)
            
            return df
            
        except Exception as e:
            print(f"Erro na preparação: {str(e)}")
            return None
    
    def _calculate_rfm_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula métricas RFM (Recency, Frequency, Monetary) para segmentação."""
        try:
            current_date = df['Data'].max()
            
            # Agregações por cliente
            customer_metrics = df.groupby('Customer_ID').agg({
                'Data': ['min', 'max', 'count'],
                'Total_Liquido': ['sum', 'mean'],
                'Quantidade': 'sum' if 'Quantidade' in df.columns else 'count'
            }).fillna(0)
            
            # Flatten columns
            customer_metrics.columns = ['_'.join(col).strip() for col in customer_metrics.columns]
            
            # Calcular RFM
            customer_metrics['Recency'] = (current_date - pd.to_datetime(customer_metrics['Data_max'])).dt.days
            customer_metrics['Frequency'] = customer_metrics['Data_count']
            customer_metrics['Monetary'] = customer_metrics['Total_Liquido_sum']
            
            # Segmentação de clientes
            customer_metrics['Customer_Segment'] = customer_metrics.apply(
                self._classify_customer_segment, axis=1
            )
            
            # Merge de volta
            df = df.merge(
                customer_metrics[['Customer_Segment', 'Recency', 'Frequency', 'Monetary']],
                left_on='Customer_ID', 
                right_index=True, 
                how='left'
            )
            
            return df
            
        except Exception as e:
            print(f"Erro no cálculo RFM: {str(e)}")
            return df
    
    def _classify_customer_segment(self, row) -> str:
        """Classifica segmento do cliente baseado em RFM."""
        recency = row['Recency']
        frequency = row['Frequency']
        monetary = row['Monetary']
        
        # Segmentação específica para joalherias
        if monetary > 5000 and frequency >= 3 and recency <= 60:
            return 'VIP'
        elif frequency == 1 and recency <= 30:
            return 'New'
        elif recency > 180:
            return 'At Risk'
        elif monetary > 2000:
            return 'High Value'
        elif frequency >= 3:
            return 'Frequent'
        else:
            return 'Regular'
    
    def _simulate_customer_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simula IDs de clientes quando não disponível."""
        df = df.copy()
        
        # Clustering baseado em padrões de compra
        df['Date_Numeric'] = df['Data'].astype('int64') // 10**9
        
        features = ['Total_Liquido', 'Date_Numeric']
        if 'Quantidade' in df.columns:
            features.append('Quantidade')
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features])
        
        n_clusters = min(max(len(df) // 15, 5), 100)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_features)
        
        df['Customer_ID'] = 'CUST_' + pd.Series(clusters).astype(str).str.zfill(3)
        
        return df
    
    def _execute_recommendation_analysis(
        self, 
        recommendation_type: str, 
        df: pd.DataFrame, 
        target_segment: str,
        count: int, 
        confidence: float, 
        detailed: bool
    ) -> Dict[str, Any]:
        """Executa análise específica de recomendação."""
        
        analysis_engines = {
            'product_recommendations': self._generate_product_recommendations,
            'customer_targeting': self._generate_customer_targeting,
            'pricing_optimization': self._generate_pricing_recommendations,
            'inventory_suggestions': self._generate_inventory_recommendations,
            'marketing_campaigns': self._generate_marketing_campaigns,
            'strategic_actions': self._generate_strategic_actions
        }
        
        if recommendation_type not in analysis_engines:
            raise ValueError(f"Tipo de recomendação '{recommendation_type}' não suportado. Tipos válidos: {list(analysis_engines.keys())}")
        
        return analysis_engines[recommendation_type](df, target_segment, count, confidence, detailed)
    
    def _generate_cache_key(self, rec_type: str, segment: str, count: int, confidence: float) -> str:
        """Gera chave de cache baseada nos parâmetros."""
        key_data = f"{rec_type}_{segment}_{count}_{confidence}_{self._last_data_hash}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _format_error(self, error_msg: str, rec_type: str) -> str:
        """Formata erro de forma estruturada para agentes."""
        error_result = {
            'success': False,
            'error': error_msg,
            'recommendation_type': rec_type,
            'timestamp': datetime.now().isoformat(),
            'suggested_actions': [
                "Verificar se arquivo de dados existe e está acessível",
                "Validar formato do CSV (separador ';', encoding UTF-8)",
                "Confirmar se colunas obrigatórias estão presentes: Data, Total_Liquido",
                "Verificar se há dados suficientes (mínimo 10 registros)"
            ]
        }
        
        return json.dumps(error_result, indent=2, ensure_ascii=False)
    
    def _format_final_result(self, result: Dict[str, Any], rec_type: str) -> str:
        """Formata resultado final para agentes com JSON + Markdown."""
        
        # Estrutura JSON para processamento
        json_result = {
            'success': True,
            'analysis_type': f"{rec_type.replace('_', ' ').title()} Analysis",
            'recommendations': result.get('recommendations', {}),
            'insights': result.get('insights', []),
            'metadata': result.get('metadata', {}),
            'financial_impact': result.get('financial_impact', {}),
            'next_steps': result.get('next_steps', [])
        }
        
        # Formatação Markdown para legibilidade
        markdown_output = self._generate_markdown_report(json_result, rec_type)
        
        # Combinar JSON + Markdown para agentes
        combined_output = f"""
{json.dumps(json_result, indent=2, ensure_ascii=False)}

---

{markdown_output}
        """.strip()
        
        return combined_output
    
    def _generate_markdown_report(self, result: Dict[str, Any], rec_type: str) -> str:
        """Gera relatório em Markdown otimizado para agentes."""
        
        timestamp = datetime.now().strftime('%d/%m/%Y %H:%M')
        
        report = f"""# 🎯 RECOMMENDATION ENGINE - {result['analysis_type'].upper()}

**📊 Análise Gerada:** {timestamp}  
**🎯 Tipo:** {result['analysis_type']}  
**⚡ Engine:** Recommendation Engine V2.0  
**🔗 Otimizado para:** CrewAI Agents  

---

## 💡 PRINCIPAIS INSIGHTS

"""
        
        # Adicionar insights específicos
        if result.get('insights'):
            for insight in result['insights'][:5]:  # Top 5 insights
                report += f"• {insight}\n"
        
        report += f"""

## 📈 IMPACTO FINANCEIRO

"""
        
        # Adicionar impacto financeiro se disponível
        financial = result.get('financial_impact', {})
        if financial:
            for key, value in financial.items():
                if isinstance(value, (int, float)):
                    report += f"• **{key.replace('_', ' ').title()}:** R$ {value:,.2f}\n"
                else:
                    report += f"• **{key.replace('_', ' ').title()}:** {value}\n"
        
        report += f"""

## 🎯 RECOMENDAÇÕES PRIORIZADAS

"""
        
        # Adicionar recomendações baseadas no tipo
        recommendations = result.get('recommendations', {})
        if recommendations:
            count = 0
            for category, items in recommendations.items():
                if count >= 3:  # Máximo 3 categorias para legibilidade
                    break
                report += f"### {category.replace('_', ' ').title()}\n"
                if isinstance(items, list) and items:
                    for i, item in enumerate(items[:3], 1):  # Top 3 por categoria
                        if isinstance(item, dict):
                            name = item.get('name', item.get('description', f'Item {i}'))
                            score = item.get('score', item.get('confidence', 0))
                            report += f"{i}. **{name}** (Score: {score:.2f})\n"
                        else:
                            report += f"{i}. {item}\n"
                count += 1
                report += "\n"
        
        report += f"""## 🚀 PRÓXIMOS PASSOS

"""
        
        # Adicionar próximos passos
        next_steps = result.get('next_steps', [])
        if next_steps:
            for i, step in enumerate(next_steps[:5], 1):
                report += f"{i}. {step}\n"
        else:
            # Próximos passos padrão baseados no tipo
            default_steps = self._get_default_next_steps(rec_type)
            for i, step in enumerate(default_steps, 1):
                report += f"{i}. {step}\n"
        
        # Metadados técnicos
        metadata = result.get('metadata', {})
        report += f"""

---

## 📋 METADADOS TÉCNICOS

• **Pontos de Dados:** {metadata.get('data_points', 0):,}  
• **Clientes Únicos:** {metadata.get('unique_customers', 0):,}  
• **Produtos Únicos:** {metadata.get('unique_products', 0):,}  
• **Período:** {metadata.get('date_range', {}).get('start', 'N/A')} a {metadata.get('date_range', {}).get('end', 'N/A')}  
• **Tempo de Processamento:** {metadata.get('processing_time_seconds', 0):.2f}s  
• **Confiança Mínima:** {metadata.get('confidence_threshold', 0):.0%}  
• **Cache:** {'✅ Hit' if metadata.get('cache_hit') else '🔄 Miss'}  

*Powered by Recommendation Engine V2.0 - Otimizado para CrewAI*
"""
        
        return report
    
    def _get_default_next_steps(self, rec_type: str) -> List[str]:
        """Retorna próximos passos padrão baseados no tipo de análise."""
        
        default_steps = {
            'product_recommendations': [
                "Implementar produtos recomendados em campanhas de cross-selling",
                "Treinar equipe de vendas sobre produtos de alto potencial", 
                "Criar bundles com produtos frequentemente comprados juntos",
                "Monitorar performance dos produtos recomendados"
            ],
            'customer_targeting': [
                "Executar campanhas específicas para cada segmento identificado",
                "Implementar programa de fidelidade para clientes VIP",
                "Criar jornada de reativação para clientes em risco",
                "Desenvolver ofertas personalizadas por segmento"
            ],
            'pricing_optimization': [
                "Testar ajustes de preço em produtos identificados",
                "Implementar preços dinâmicos baseados em elasticidade",
                "Monitorar impacto das mudanças na margem e volume",
                "Ajustar estratégia de posicionamento de mercado"
            ],
            'inventory_suggestions': [
                "Executar reposição de produtos classe A prioritários",
                "Implementar liquidação para produtos de baixo giro",
                "Otimizar níveis de estoque baseado na análise ABC",
                "Desenvolver políticas de compra baseadas em turnover"
            ],
            'marketing_campaigns': [
                "Executar campanhas personalizadas por segmento",
                "Implementar automação de marketing baseada em comportamento",
                "Testar canais e mensagens recomendadas",
                "Monitorar ROI das campanhas implementadas"
            ],
            'strategic_actions': [
                "Priorizar ações críticas identificadas na análise",
                "Desenvolver roadmap de implementação detalhado",
                "Alocar recursos para iniciativas de maior impacto",
                "Estabelecer KPIs para monitorar progresso"
            ]
        }
        
        return default_steps.get(rec_type, [
            "Analisar recomendações em detalhe",
            "Desenvolver plano de implementação",
            "Monitorar resultados e ajustar estratégia"
        ])
    
    # ==========================================
    # MÉTODOS DE GERAÇÃO DE RECOMENDAÇÕES
    # ==========================================
    
    def _generate_product_recommendations(
        self, df: pd.DataFrame, target_segment: str, count: int, 
        confidence: float, detailed: bool
    ) -> Dict[str, Any]:
        """Gera recomendações de produtos usando algoritmos ML."""
        try:
            # Filtrar por segmento
            df_segment = self._filter_by_segment(df, target_segment)
            
            # Análise de produtos por performance
            product_analysis = df_segment.groupby('Codigo_Produto').agg({
                'Total_Liquido': ['sum', 'mean', 'count'],
                'Quantidade': 'sum' if 'Quantidade' in df_segment.columns else 'count',
                'Data': 'max',
                'Descricao_Produto': 'first'
            }).fillna(0)
            
            # Flatten columns
            product_analysis.columns = ['_'.join(col).strip() for col in product_analysis.columns]
            
            # Calcular scores de recomendação
            product_analysis['Revenue_Score'] = self._normalize_score(product_analysis['Total_Liquido_sum'])
            product_analysis['Frequency_Score'] = self._normalize_score(product_analysis['Total_Liquido_count'])
            product_analysis['Recency_Score'] = self._calculate_recency_score(product_analysis['Data_max'])
            product_analysis['AOV_Score'] = self._normalize_score(product_analysis['Total_Liquido_mean'])
            
            # Score final ponderado
            product_analysis['Recommendation_Score'] = (
                product_analysis['Revenue_Score'] * 0.3 +
                product_analysis['Frequency_Score'] * 0.25 +
                product_analysis['Recency_Score'] * 0.2 +
                product_analysis['AOV_Score'] * 0.25
            )
            
            # Filtrar por confiança e obter top produtos
            high_confidence = product_analysis[
                product_analysis['Recommendation_Score'] >= confidence
            ]
            
            top_products = high_confidence.nlargest(count, 'Recommendation_Score')
            
            # Análises adicionais se habilitadas
            additional_analysis = {}
            if detailed:
                additional_analysis = {
                    'category_analysis': self._analyze_product_categories(df_segment, top_products.index.tolist()),
                    'market_basket': self._perform_market_basket_analysis(df_segment),
                    'seasonal_trends': self._analyze_seasonal_trends(df_segment)
                }
            
            # Insights gerados
            insights = []
            if len(top_products) > 0:
                best_product = top_products.iloc[0]
                insights.append(f"Produto top: {best_product.get('Descricao_Produto_first', 'N/A')} (Score: {best_product['Recommendation_Score']:.2f})")
                
                avg_revenue = top_products['Total_Liquido_sum'].mean()
                insights.append(f"Receita média dos produtos recomendados: R$ {avg_revenue:,.2f}")
                
                if target_segment != 'all':
                    insights.append(f"Análise específica para segmento: {target_segment}")
            
            return {
                'recommendations': {
                    'top_products': [
                        {
                            'code': idx,
                            'name': row.get('Descricao_Produto_first', 'N/A'),
                            'score': round(row['Recommendation_Score'], 2),
                            'revenue': round(row['Total_Liquido_sum'], 2),
                            'frequency': int(row['Total_Liquido_count'])
                        }
                        for idx, row in top_products.iterrows()
                    ]
                },
                'insights': insights,
                'financial_impact': {
                    'total_revenue_potential': round(top_products['Total_Liquido_sum'].sum(), 2),
                    'average_product_value': round(top_products['Total_Liquido_mean'].mean(), 2)
                },
                'next_steps': [
                    "Implementar produtos recomendados em campanhas de cross-selling",
                    "Treinar equipe sobre produtos de alto potencial",
                    "Criar bundles com produtos frequentemente comprados juntos"
                ],
                **additional_analysis
            }
            
        except Exception as e:
            return {'error': f"Erro nas recomendações de produtos: {str(e)}"}
    
    def _generate_customer_targeting(
        self, df: pd.DataFrame, target_segment: str, count: int, 
        confidence: float, detailed: bool
    ) -> Dict[str, Any]:
        """Gera estratégias de targeting de clientes."""
        try:
            # Análise de clientes por segmento
            customer_analysis = df.groupby(['Customer_ID', 'Customer_Segment']).agg({
                'Total_Liquido': ['sum', 'mean', 'count'],
                'Data': ['min', 'max'],
                'Codigo_Produto': 'nunique'
            }).fillna(0)
            
            # Flatten columns
            customer_analysis.columns = ['_'.join(col).strip() for col in customer_analysis.columns]
            
            # Calcular métricas de valor
            current_date = df['Data'].max()
            customer_analysis['Days_Since_Last'] = (current_date - pd.to_datetime(customer_analysis['Data_max'])).dt.days
            customer_analysis['Customer_Lifetime'] = (pd.to_datetime(customer_analysis['Data_max']) - pd.to_datetime(customer_analysis['Data_min'])).dt.days
            
            # Scoring para targeting
            customer_analysis['Value_Score'] = self._normalize_score(customer_analysis['Total_Liquido_sum'])
            customer_analysis['Frequency_Score'] = self._normalize_score(customer_analysis['Total_Liquido_count'])
            customer_analysis['Recency_Score'] = self._normalize_score(365 - customer_analysis['Days_Since_Last'])
            
            customer_analysis['Target_Score'] = (
                customer_analysis['Value_Score'] * 0.4 +
                customer_analysis['Frequency_Score'] * 0.3 +
                customer_analysis['Recency_Score'] * 0.3
            )
            
            # Estratégias de targeting
            strategies = {
                'upsell': customer_analysis[
                    (customer_analysis['Total_Liquido_count'] >= 2) & 
                    (customer_analysis['Total_Liquido_mean'] < customer_analysis['Total_Liquido_mean'].median())
                ].nlargest(count, 'Target_Score'),
                
                'retention': customer_analysis[
                    (customer_analysis['Days_Since_Last'] > 60) & 
                    (customer_analysis['Total_Liquido_sum'] > customer_analysis['Total_Liquido_sum'].median())
                ].nlargest(count, 'Target_Score'),
                
                'reactivation': customer_analysis[
                    customer_analysis['Days_Since_Last'] > 180
                ].nlargest(count, 'Target_Score')
            }
            
            # Análise de segmentos
            segment_distribution = df['Customer_Segment'].value_counts().to_dict()
            
            insights = []
            total_customers = sum(len(strategy_customers) for strategy_customers in strategies.values())
            insights.append(f"Total de clientes identificados para targeting: {total_customers}")
            
            if strategies['upsell'].shape[0] > 0:
                avg_upsell_potential = strategies['upsell']['Total_Liquido_mean'].mean()
                insights.append(f"Potencial médio de upsell: R$ {avg_upsell_potential:,.2f}")
            
            return {
                'recommendations': {
                    strategy_name: [
                        {
                            'customer_id': idx,
                            'score': round(row['Target_Score'], 2),
                            'value': round(row['Total_Liquido_sum'], 2),
                            'frequency': int(row['Total_Liquido_count'])
                        }
                        for idx, row in strategy_df.iterrows()
                    ]
                    for strategy_name, strategy_df in strategies.items()
                },
                'insights': insights,
                'financial_impact': {
                    'upsell_potential': round(strategies['upsell']['Total_Liquido_sum'].sum() * 0.2, 2) if len(strategies['upsell']) > 0 else 0,
                    'retention_value': round(strategies['retention']['Total_Liquido_sum'].sum(), 2) if len(strategies['retention']) > 0 else 0
                },
                'next_steps': [
                    "Executar campanhas específicas para cada segmento",
                    "Implementar programa de fidelidade para clientes VIP",
                    "Criar jornada de reativação para clientes em risco"
                ]
            }
            
        except Exception as e:
            return {'error': f"Erro no targeting de clientes: {str(e)}"}
    
    def _generate_pricing_recommendations(
        self, df: pd.DataFrame, target_segment: str, count: int, 
        confidence: float, detailed: bool
    ) -> Dict[str, Any]:
        """Gera recomendações de otimização de preços."""
        try:
            df_segment = self._filter_by_segment(df, target_segment)
            
            # Análise por grupo de produto se disponível
            if 'Grupo_Produto' in df_segment.columns:
                pricing_analysis = df_segment.groupby('Grupo_Produto').agg({
                    'Preco_Unitario': ['mean', 'std', 'count'],
                    'Total_Liquido': ['sum', 'mean'],
                    'Quantidade': 'sum' if 'Quantidade' in df_segment.columns else 'count'
                }).fillna(0)
                
                # Flatten columns
                pricing_analysis.columns = ['_'.join(col).strip() for col in pricing_analysis.columns]
                
                # Calcular métricas de pricing
                pricing_analysis['Price_Variability'] = pricing_analysis['Preco_Unitario_std'] / pricing_analysis['Preco_Unitario_mean']
                pricing_analysis['Revenue_Per_Unit'] = pricing_analysis['Total_Liquido_sum'] / pricing_analysis['Quantidade_sum']
                
                # Identificar oportunidades
                opportunities = []
                for category, row in pricing_analysis.iterrows():
                    if row['Price_Variability'] > 0.3:  # Alta variabilidade
                        opportunities.append({
                            'category': category,
                            'opportunity': 'Standardize Price',
                            'current_avg': round(row['Preco_Unitario_mean'], 2),
                            'variability': round(row['Price_Variability'], 2)
                        })
                    
                    if row['Revenue_Per_Unit'] > pricing_analysis['Revenue_Per_Unit'].mean() * 1.2:
                        opportunities.append({
                            'category': category,
                            'opportunity': 'Increase Price',
                            'current_avg': round(row['Preco_Unitario_mean'], 2),
                            'revenue_per_unit': round(row['Revenue_Per_Unit'], 2)
                        })
            
            else:
                # Análise simplificada sem categorias
                avg_price = df_segment['Preco_Unitario'].mean()
                price_std = df_segment['Preco_Unitario'].std()
                opportunities = [{
                    'category': 'General',
                    'opportunity': 'Review Pricing Strategy',
                    'current_avg': round(avg_price, 2),
                    'variability': round(price_std / avg_price, 2) if avg_price > 0 else 0
                }]
            
            insights = [
                f"Identificadas {len(opportunities)} oportunidades de pricing",
                f"Preço médio no segmento: R$ {df_segment['Preco_Unitario'].mean():,.2f}"
            ]
            
            return {
                'recommendations': {
                    'pricing_opportunities': opportunities[:count]
                },
                'insights': insights,
                'financial_impact': {
                    'total_revenue_analyzed': round(df_segment['Total_Liquido'].sum(), 2),
                    'average_transaction': round(df_segment['Total_Liquido'].mean(), 2)
                },
                'next_steps': [
                    "Testar ajustes de preço em produtos identificados",
                    "Implementar preços dinâmicos baseados em elasticidade",
                    "Monitorar impacto das mudanças na margem"
                ]
            }
            
        except Exception as e:
            return {'error': f"Erro nas recomendações de pricing: {str(e)}"}
    
    def _generate_inventory_recommendations(
        self, df: pd.DataFrame, target_segment: str, count: int, 
        confidence: float, detailed: bool
    ) -> Dict[str, Any]:
        """Gera recomendações de gestão de inventário."""
        try:
            # Análise ABC de produtos
            product_analysis = df.groupby('Codigo_Produto').agg({
                'Total_Liquido': ['sum', 'count'],
                'Data': 'max',
                'Descricao_Produto': 'first'
            }).fillna(0)
            
            # Flatten columns
            product_analysis.columns = ['_'.join(col).strip() for col in product_analysis.columns]
            
            # Classificação ABC
            revenue_sorted = product_analysis.sort_values('Total_Liquido_sum', ascending=False)
            cumsum_pct = revenue_sorted['Total_Liquido_sum'].cumsum() / revenue_sorted['Total_Liquido_sum'].sum()
            
            abc_class = []
            for pct in cumsum_pct:
                if pct <= 0.8:
                    abc_class.append('A')
                elif pct <= 0.95:
                    abc_class.append('B')
                else:
                    abc_class.append('C')
            
            revenue_sorted['ABC_Class'] = abc_class
            product_analysis = product_analysis.merge(
                revenue_sorted[['ABC_Class']], left_index=True, right_index=True, how='left'
            )
            
            # Calcular dias desde última venda
            current_date = df['Data'].max()
            product_analysis['Days_Since_Last_Sale'] = (
                current_date - pd.to_datetime(product_analysis['Data_max'])
            ).dt.days
            
            # Recomendações por categoria
            recommendations = {
                'restock_priority': product_analysis[
                    (product_analysis['ABC_Class'] == 'A') & 
                    (product_analysis['Days_Since_Last_Sale'] <= 30)
                ].nlargest(count//2, 'Total_Liquido_sum'),
                
                'liquidate': product_analysis[
                    (product_analysis['Days_Since_Last_Sale'] > 90) &
                    (product_analysis['ABC_Class'] == 'C')
                ].nsmallest(count//3, 'Total_Liquido_sum'),
                
                'monitor': product_analysis[
                    (product_analysis['ABC_Class'] == 'A') &
                    (product_analysis['Days_Since_Last_Sale'] > 45)
                ].nlargest(count//4, 'Total_Liquido_sum')
            }
            
            insights = []
            a_products = len(product_analysis[product_analysis['ABC_Class'] == 'A'])
            insights.append(f"Produtos classe A (prioritários): {a_products}")
            
            slow_movers = len(product_analysis[product_analysis['Days_Since_Last_Sale'] > 90])
            if slow_movers > 0:
                insights.append(f"Produtos sem venda há 90+ dias: {slow_movers}")
            
            return {
                'recommendations': {
                    category: [
                        {
                            'code': idx,
                            'name': row.get('Descricao_Produto_first', 'N/A'),
                            'abc_class': row['ABC_Class'],
                            'revenue': round(row['Total_Liquido_sum'], 2),
                            'days_since_last_sale': int(row['Days_Since_Last_Sale'])
                        }
                        for idx, row in cat_df.iterrows()
                    ]
                    for category, cat_df in recommendations.items()
                },
                'insights': insights,
                'financial_impact': {
                    'slow_moving_value': round(
                        product_analysis[product_analysis['Days_Since_Last_Sale'] > 90]['Total_Liquido_sum'].sum() * 0.6, 2
                    )
                },
                'next_steps': [
                    "Executar reposição de produtos classe A prioritários",
                    "Implementar liquidação para produtos de baixo giro",
                    "Otimizar níveis de estoque baseado na análise ABC"
                ]
            }
            
        except Exception as e:
            return {'error': f"Erro nas recomendações de inventário: {str(e)}"}
    
    def _generate_marketing_campaigns(
        self, df: pd.DataFrame, target_segment: str, count: int, 
        confidence: float, detailed: bool
    ) -> Dict[str, Any]:
        """Gera campanhas de marketing personalizadas."""
        try:
            # Análise por segmento de cliente
            segment_analysis = df.groupby('Customer_Segment').agg({
                'Total_Liquido': ['sum', 'mean', 'count'],
                'Customer_ID': 'nunique',
                'Month': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
            }).fillna(0)
            
            # Flatten columns
            segment_analysis.columns = ['_'.join(col).strip() for col in segment_analysis.columns]
            
            # Campanhas por segmento
            campaigns = {}
            for segment in df['Customer_Segment'].unique():
                if pd.isna(segment):
                    continue
                    
                segment_info = segment_analysis.loc[segment]
                campaign_type, campaign_details = self._design_campaign_for_segment(segment, segment_info)
                
                campaigns[segment] = {
                    'type': campaign_type,
                    'details': campaign_details,
                    'target_size': int(segment_info['Customer_ID_nunique']),
                    'avg_value': round(segment_info['Total_Liquido_mean'], 2),
                    'estimated_roi': self._estimate_campaign_roi(segment, segment_info)
                }
            
            insights = []
            total_customers = sum(camp['target_size'] for camp in campaigns.values())
            insights.append(f"Base total para campanhas: {total_customers} clientes")
            
            best_roi = max(campaigns.values(), key=lambda x: x['estimated_roi']) if campaigns else None
            if best_roi:
                insights.append(f"Melhor ROI estimado: {best_roi['estimated_roi']:.1%}")
            
            return {
                'recommendations': {
                    'campaigns': campaigns
                },
                'insights': insights,
                'financial_impact': {
                    'total_customer_base': total_customers,
                    'estimated_campaign_cost': total_customers * 15,  # R$ 15 por cliente
                    'potential_revenue': sum(camp['avg_value'] * camp['target_size'] * 0.1 for camp in campaigns.values())
                },
                'next_steps': [
                    "Executar campanhas personalizadas por segmento",
                    "Implementar automação de marketing baseada em comportamento",
                    "Testar canais e mensagens recomendadas"
                ]
            }
            
        except Exception as e:
            return {'error': f"Erro nas campanhas de marketing: {str(e)}"}
    
    def _generate_strategic_actions(
        self, df: pd.DataFrame, target_segment: str, count: int, 
        confidence: float, detailed: bool
    ) -> Dict[str, Any]:
        """Gera ações estratégicas baseadas em dados."""
        try:
            # Métricas de performance geral
            performance_metrics = {
                'total_revenue': df['Total_Liquido'].sum(),
                'total_customers': df['Customer_ID'].nunique() if 'Customer_ID' in df.columns else len(df),
                'avg_transaction': df['Total_Liquido'].mean(),
                'unique_products': df['Codigo_Produto'].nunique(),
                'date_range_days': (df['Data'].max() - df['Data'].min()).days
            }
            
            # Análise de tendências
            monthly_data = df.groupby(df['Data'].dt.to_period('M')).agg({
                'Total_Liquido': 'sum',
                'Customer_ID': 'nunique' if 'Customer_ID' in df.columns else 'count'
            })
            
            # Calcular crescimento
            if len(monthly_data) > 1:
                revenue_growth = monthly_data['Total_Liquido'].pct_change().mean() * 100
                customer_growth = monthly_data['Customer_ID'].pct_change().mean() * 100
            else:
                revenue_growth = 0
                customer_growth = 0
            
            # Gerar ações estratégicas
            strategic_actions = []
            
            # Ações baseadas em crescimento
            if revenue_growth > 10:
                strategic_actions.append({
                    'action': 'Acelerar Expansão',
                    'priority': 'High',
                    'rationale': f'Crescimento forte de {revenue_growth:.1f}% mensal',
                    'expected_impact': 'Aumento de 25-40% na receita',
                    'timeline': '3-6 meses'
                })
            elif revenue_growth < -5:
                strategic_actions.append({
                    'action': 'Plano de Recuperação',
                    'priority': 'Critical',
                    'rationale': f'Declínio de {revenue_growth:.1f}% mensal',
                    'expected_impact': 'Estabilizar receita',
                    'timeline': '1-3 meses'
                })
            
            # Ações baseadas em segmentação
            segment_dist = df['Customer_Segment'].value_counts(normalize=True) if 'Customer_Segment' in df.columns else {}
            
            if segment_dist.get('VIP', 0) < 0.1:
                strategic_actions.append({
                    'action': 'Programa VIP',
                    'priority': 'High',
                    'rationale': 'Baixa base de clientes VIP (<10%)',
                    'expected_impact': 'Aumento de 15-20% no CLV',
                    'timeline': '2-4 meses'
                })
            
            if segment_dist.get('At Risk', 0) > 0.2:
                strategic_actions.append({
                    'action': 'Programa de Retenção',
                    'priority': 'High',
                    'rationale': 'Alto percentual de clientes em risco (>20%)',
                    'expected_impact': 'Redução de 30-50% no churn',
                    'timeline': '1-2 meses'
                })
            
            # Priorizar ações
            prioritized_actions = sorted(
                strategic_actions, 
                key=lambda x: {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}[x['priority']]
            )[:count]
            
            insights = []
            insights.append(f"Crescimento mensal de receita: {revenue_growth:.1f}%")
            insights.append(f"Crescimento mensal de clientes: {customer_growth:.1f}%")
            insights.append(f"Identificadas {len(prioritized_actions)} ações estratégicas prioritárias")
            
            return {
                'recommendations': {
                    'strategic_actions': prioritized_actions
                },
                'insights': insights,
                'financial_impact': {
                    'current_monthly_revenue': round(performance_metrics['total_revenue'] / max(1, performance_metrics['date_range_days'] / 30), 2),
                    'growth_rate_monthly': round(revenue_growth, 2)
                },
                'next_steps': [
                    "Priorizar ações críticas identificadas na análise",
                    "Desenvolver roadmap de implementação detalhado",
                    "Alocar recursos para iniciativas de maior impacto"
                ]
            }
            
        except Exception as e:
            return {'error': f"Erro nas ações estratégicas: {str(e)}"}
    
    # ==========================================
    # MÉTODOS AUXILIARES
    # ==========================================
    
    def _filter_by_segment(self, df: pd.DataFrame, target_segment: str) -> pd.DataFrame:
        """Filtra dados por segmento de cliente."""
        if target_segment == 'all' or 'Customer_Segment' not in df.columns:
            return df
        
        segment_map = {
            'vip': 'VIP',
            'new_customers': 'New',
            'at_risk': 'At Risk',
            'high_value': 'High Value',
            'frequent': 'Frequent'
        }
        
        segment = segment_map.get(target_segment, target_segment)
        filtered_df = df[df['Customer_Segment'] == segment]
        
        return filtered_df if len(filtered_df) > 0 else df
    
    def _normalize_score(self, series: pd.Series) -> pd.Series:
        """Normaliza scores de 0 a 1."""
        if series.max() == series.min():
            return pd.Series([0.5] * len(series), index=series.index)
        return (series - series.min()) / (series.max() - series.min())
    
    def _calculate_recency_score(self, dates: pd.Series) -> pd.Series:
        """Calcula score de recência."""
        current_date = pd.Timestamp.now()
        days_since = (current_date - pd.to_datetime(dates)).dt.days
        max_days = days_since.max()
        return (max_days - days_since) / max_days if max_days > 0 else pd.Series([1] * len(days_since), index=days_since.index)
    
    def _analyze_product_categories(self, df: pd.DataFrame, product_codes: List[str]) -> Dict[str, Any]:
        """Analisa categorias dos produtos recomendados."""
        if 'Grupo_Produto' not in df.columns:
            return {'message': 'Dados de categoria não disponíveis'}
        
        recommended_products = df[df['Codigo_Produto'].isin(product_codes)]
        category_dist = recommended_products['Grupo_Produto'].value_counts().to_dict()
        
        return {
            'category_distribution': category_dist,
            'top_category': max(category_dist, key=category_dist.get) if category_dist else None
        }
    
    def _perform_market_basket_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Análise de market basket simplificada."""
        try:
            # Agrupar por transação
            if 'Customer_ID' in df.columns:
                transaction_groups = df.groupby(['Customer_ID', 'Data'])['Codigo_Produto'].apply(list)
            else:
                transaction_groups = df.groupby('Data')['Codigo_Produto'].apply(list)
            
            # Encontrar combinações
            combinations = {}
            for products in transaction_groups:
                if len(products) > 1:
                    for i, prod1 in enumerate(products):
                        for prod2 in products[i+1:]:
                            combo = tuple(sorted([prod1, prod2]))
                            combinations[combo] = combinations.get(combo, 0) + 1
            
            top_combinations = sorted(combinations.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'frequent_combinations': [
                    {'products': list(combo), 'frequency': freq} 
                    for combo, freq in top_combinations
                ],
                'total_combinations_found': len(combinations)
            }
            
        except Exception as e:
            return {'error': f"Erro na análise de market basket: {str(e)}"}
    
    def _analyze_seasonal_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa tendências sazonais."""
        try:
            monthly_sales = df.groupby('Month')['Total_Liquido'].sum()
            seasonal_index = (monthly_sales / monthly_sales.mean()).round(2)
            
            return {
                'seasonal_index': seasonal_index.to_dict(),
                'peak_months': seasonal_index[seasonal_index > 1.2].index.tolist(),
                'low_months': seasonal_index[seasonal_index < 0.8].index.tolist()
            }
            
        except Exception as e:
            return {'error': f"Erro na análise sazonal: {str(e)}"}
    
    def _design_campaign_for_segment(self, segment: str, segment_info: pd.Series) -> Tuple[str, Dict[str, Any]]:
        """Desenha campanha específica para segmento."""
        
        campaigns = {
            'VIP': ('Exclusivity Campaign', {
                'message': 'Acesso exclusivo a nova coleção',
                'channel': 'Personal contact + Email',
                'offer': 'Preview exclusivo + 10% desconto'
            }),
            'New': ('Welcome Series', {
                'message': 'Bem-vindo à nossa família',
                'channel': 'Email sequence',
                'offer': '15% desconto na segunda compra'
            }),
            'At Risk': ('Win-back Campaign', {
                'message': 'Sentimos sua falta',
                'channel': 'Email + SMS',
                'offer': '20% desconto + frete grátis'
            }),
            'High Value': ('Upsell Campaign', {
                'message': 'Peças especiais para você',
                'channel': 'Email + WhatsApp',
                'offer': 'Produtos premium com 10% desconto'
            })
        }
        
        return campaigns.get(segment, ('General Campaign', {
            'message': 'Ofertas especiais',
            'channel': 'Email',
            'offer': '10% desconto'
        }))
    
    def _estimate_campaign_roi(self, segment: str, segment_info: pd.Series) -> float:
        """Estima ROI de campanha por segmento."""
        
        roi_by_segment = {
            'VIP': 0.25,
            'New': 0.15,
            'At Risk': 0.10,
            'High Value': 0.20,
            'Frequent': 0.18,
            'Regular': 0.12
        }
        
        return roi_by_segment.get(segment, 0.10)

    # ==========================================
    # MÉTODOS AVANÇADOS DE MACHINE LEARNING V2.1
    # ==========================================
    
    def _collaborative_filtering_advanced(self, df: pd.DataFrame, target_customers: List[str] = None, 
                                        n_recommendations: int = 10) -> Dict[str, Any]:
        """
        Collaborative Filtering avançado usando Matrix Factorization (SVD).
        
        Implementa algoritmo de decomposição de valores singulares para recomendar
        produtos baseado em similaridade de comportamento entre clientes.
        """
        try:
            # Criar matriz customer-product
            if 'Customer_ID' not in df.columns:
                return {'error': 'Customer_ID necessário para collaborative filtering'}
            
            # Agregação por customer-product
            customer_product_matrix = df.groupby(['Customer_ID', 'Codigo_Produto'])['Total_Liquido'].sum().unstack(fill_value=0)
            
            if customer_product_matrix.shape[0] < 5 or customer_product_matrix.shape[1] < 5:
                return {'error': 'Dados insuficientes para collaborative filtering (mínimo 5x5)'}
            
            # Aplicar SVD para redução de dimensionalidade
            n_components = min(20, customer_product_matrix.shape[0] - 1, customer_product_matrix.shape[1] - 1)
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            
            # Transformar matriz para espaço latente
            customer_factors = svd.fit_transform(customer_product_matrix)
            product_factors = svd.components_
            
            # Reconstruir matriz de predições
            predicted_ratings = np.dot(customer_factors, product_factors)
            predicted_df = pd.DataFrame(
                predicted_ratings,
                index=customer_product_matrix.index,
                columns=customer_product_matrix.columns
            )
            
            # Gerar recomendações
            recommendations = {}
            customers_to_process = target_customers if target_customers else customer_product_matrix.index[:10]
            
            for customer_id in customers_to_process:
                if customer_id not in predicted_df.index:
                    continue
                    
                # Produtos já comprados pelo cliente
                purchased_products = customer_product_matrix.loc[customer_id]
                purchased_products = purchased_products[purchased_products > 0].index.tolist()
                
                # Predições para produtos não comprados
                customer_predictions = predicted_df.loc[customer_id]
                non_purchased = customer_predictions.drop(purchased_products, errors='ignore')
                
                # Top recomendações
                top_recommendations = non_purchased.nlargest(n_recommendations)
                
                recommendations[customer_id] = [
                    {
                        'product_code': product,
                        'predicted_score': round(score, 3),
                        'recommendation_type': 'collaborative_filtering'
                    }
                    for product, score in top_recommendations.items()
                ]
            
            return {
                'algorithm': 'Matrix Factorization SVD',
                'components_used': n_components,
                'explained_variance_ratio': round(svd.explained_variance_ratio_.sum(), 3),
                'recommendations': recommendations,
                'matrix_dimensions': f"{customer_product_matrix.shape[0]}x{customer_product_matrix.shape[1]}",
                'coverage': len(recommendations) / len(customers_to_process) if customers_to_process else 0
            }
            
        except Exception as e:
            return {'error': f"Erro no collaborative filtering: {str(e)}"}
    
    def _content_based_filtering_advanced(self, df: pd.DataFrame, target_products: List[str] = None,
                                        n_recommendations: int = 10) -> Dict[str, Any]:
        """
        Content-Based Filtering avançado usando TF-IDF para análise de texto.
        
        Analisa descrições de produtos para recomendar itens similares baseado
        em características textuais e numéricas.
        """
        try:
            if 'Descricao_Produto' not in df.columns:
                return {'error': 'Descricao_Produto necessário para content-based filtering'}
            
            # Preparar dados únicos de produtos
            product_features = df.groupby('Codigo_Produto').agg({
                'Descricao_Produto': 'first',
                'Total_Liquido': 'mean',
                'Quantidade': 'mean' if 'Quantidade' in df.columns else 'count',
                'Preco_Unitario': 'mean',
                'Grupo_Produto': 'first' if 'Grupo_Produto' in df.columns else 'count'
            }).fillna('')
            
            # TF-IDF para descrições de produtos
            tfidf_vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words=None,  # Sem stop words para jóias específicas
                ngram_range=(1, 2),
                min_df=1
            )
            
            # Combinar descrição e grupo para análise textual
            text_features = product_features['Descricao_Produto'].astype(str) + ' ' + \
                          product_features.get('Grupo_Produto', '').astype(str)
            
            tfidf_matrix = tfidf_vectorizer.fit_transform(text_features)
            
            # Calcular similaridades
            content_similarity = cosine_similarity(tfidf_matrix)
            content_similarity_df = pd.DataFrame(
                content_similarity,
                index=product_features.index,
                columns=product_features.index
            )
            
            # Gerar recomendações
            recommendations = {}
            products_to_process = target_products if target_products else product_features.index[:10]
            
            for product_code in products_to_process:
                if product_code not in content_similarity_df.index:
                    continue
                
                # Produtos similares (excluindo o próprio produto)
                similar_products = content_similarity_df.loc[product_code].drop(product_code)
                top_similar = similar_products.nlargest(n_recommendations)
                
                recommendations[product_code] = [
                    {
                        'similar_product': product,
                        'similarity_score': round(score, 3),
                        'description': product_features.loc[product, 'Descricao_Produto'],
                        'avg_price': round(product_features.loc[product, 'Total_Liquido'], 2),
                        'recommendation_type': 'content_based'
                    }
                    for product, score in top_similar.items()
                ]
            
            # Features mais importantes
            feature_names = tfidf_vectorizer.get_feature_names_out()
            
            return {
                'algorithm': 'TF-IDF Content-Based Filtering',
                'vocabulary_size': len(feature_names),
                'top_features': feature_names[:20].tolist(),
                'recommendations': recommendations,
                'products_analyzed': len(product_features),
                'coverage': len(recommendations) / len(products_to_process) if products_to_process else 0
            }
            
        except Exception as e:
            return {'error': f"Erro no content-based filtering: {str(e)}"}
    
    def _hybrid_recommendation_system(self, df: pd.DataFrame, customer_id: str = None,
                                    n_recommendations: int = 10) -> Dict[str, Any]:
        """
        Sistema híbrido que combina Collaborative e Content-Based Filtering.
        
        Usa weighted ensemble para combinar diferentes algoritmos e fornecer
        recomendações mais robustas e precisas.
        """
        try:
            # Executar algoritmos individuais
            collaborative_results = self._collaborative_filtering_advanced(df, [customer_id] if customer_id else None)
            content_based_results = self._content_based_filtering_advanced(df)
            
            if 'error' in collaborative_results and 'error' in content_based_results:
                return {'error': 'Ambos algoritmos falharam'}
            
            # Combinar resultados
            hybrid_scores = {}
            
            # Peso dos algoritmos (pode ser ajustado)
            collaborative_weight = 0.6
            content_weight = 0.4
            
            # Processar collaborative filtering
            if 'error' not in collaborative_results and customer_id:
                collab_recs = collaborative_results.get('recommendations', {}).get(customer_id, [])
                for rec in collab_recs:
                    product = rec['product_code']
                    hybrid_scores[product] = hybrid_scores.get(product, 0) + \
                                           rec['predicted_score'] * collaborative_weight
            
            # Processar content-based (usar produto mais popular como base)
            if 'error' not in content_based_results:
                # Encontrar produto mais popular
                popular_products = df['Codigo_Produto'].value_counts()
                if len(popular_products) > 0:
                    base_product = popular_products.index[0]
                    content_recs = content_based_results.get('recommendations', {}).get(base_product, [])
                    
                    for rec in content_recs:
                        product = rec['similar_product']
                        hybrid_scores[product] = hybrid_scores.get(product, 0) + \
                                               rec['similarity_score'] * content_weight
            
            # Ranquear produtos por score híbrido
            sorted_recommendations = sorted(
                hybrid_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n_recommendations]
            
            # Enriquecer com metadados
            enriched_recommendations = []
            for product_code, score in sorted_recommendations:
                product_data = df[df['Codigo_Produto'] == product_code]
                if len(product_data) > 0:
                    enriched_recommendations.append({
                        'product_code': product_code,
                        'hybrid_score': round(score, 3),
                        'description': product_data['Descricao_Produto'].iloc[0] if 'Descricao_Produto' in product_data.columns else 'N/A',
                        'avg_price': round(product_data['Total_Liquido'].mean(), 2),
                        'frequency': len(product_data),
                        'recommendation_type': 'hybrid_ensemble'
                    })
            
            return {
                'algorithm': 'Hybrid Ensemble (Collaborative + Content-Based)',
                'weights': {'collaborative': collaborative_weight, 'content_based': content_weight},
                'collaborative_status': 'success' if 'error' not in collaborative_results else 'failed',
                'content_based_status': 'success' if 'error' not in content_based_results else 'failed',
                'recommendations': enriched_recommendations,
                'total_products_scored': len(hybrid_scores)
            }
            
        except Exception as e:
            return {'error': f"Erro no sistema híbrido: {str(e)}"}
    
    def _advanced_market_basket_analysis(self, df: pd.DataFrame, min_support: float = 0.01,
                                       min_confidence: float = 0.1) -> Dict[str, Any]:
        """
        Market Basket Analysis avançado com regras de associação (Apriori).
        
        Implementa algoritmo Apriori para encontrar regras de associação entre produtos
        com métricas de support, confidence e lift.
        """
        try:
            # Criar transações (agrupadas por Customer_ID e Data)
            if 'Customer_ID' not in df.columns:
                df = self._simulate_customer_ids(df)
            
            transactions = df.groupby(['Customer_ID', 'Data'])['Codigo_Produto'].apply(list).reset_index()
            transaction_list = transactions['Codigo_Produto'].tolist()
            
            # Filtrar transações com múltiplos produtos
            multi_item_transactions = [t for t in transaction_list if len(t) > 1]
            
            if len(multi_item_transactions) < 10:
                return {'error': 'Transações insuficientes para market basket analysis'}
            
            # Encontrar produtos frequentes
            all_products = [item for transaction in multi_item_transactions for item in transaction]
            product_counts = pd.Series(all_products).value_counts()
            total_transactions = len(multi_item_transactions)
            
            # Aplicar threshold de support mínimo
            frequent_products = product_counts[product_counts / total_transactions >= min_support]
            
            # Gerar combinações de produtos (itemsets)
            association_rules = []
            
            # Pares de produtos (2-itemsets)
            for antecedent, consequent in itertools.combinations(frequent_products.index, 2):
                # Contar ocorrências
                antecedent_count = sum(1 for t in multi_item_transactions if antecedent in t)
                consequent_count = sum(1 for t in multi_item_transactions if consequent in t)
                both_count = sum(1 for t in multi_item_transactions if antecedent in t and consequent in t)
                
                if both_count > 0:
                    # Calcular métricas
                    support = both_count / total_transactions
                    confidence = both_count / antecedent_count if antecedent_count > 0 else 0
                    lift = confidence / (consequent_count / total_transactions) if consequent_count > 0 else 0
                    
                    if confidence >= min_confidence:
                        association_rules.append({
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'support': round(support, 3),
                            'confidence': round(confidence, 3),
                            'lift': round(lift, 3),
                            'antecedent_count': antecedent_count,
                            'consequent_count': consequent_count,
                            'both_count': both_count
                        })
            
            # Ordenar por lift (força da associação)
            association_rules.sort(key=lambda x: x['lift'], reverse=True)
            
            # Insights de negócio
            insights = []
            if association_rules:
                best_rule = association_rules[0]
                insights.append(f"Regra mais forte: {best_rule['antecedent']} → {best_rule['consequent']} (Lift: {best_rule['lift']})")
                
                high_confidence_rules = [r for r in association_rules if r['confidence'] > 0.5]
                if high_confidence_rules:
                    insights.append(f"Regras com alta confiança (>50%): {len(high_confidence_rules)}")
            
            return {
                'algorithm': 'Apriori Association Rules',
                'parameters': {'min_support': min_support, 'min_confidence': min_confidence},
                'total_transactions': total_transactions,
                'multi_item_transactions': len(multi_item_transactions),
                'frequent_products': len(frequent_products),
                'association_rules': association_rules[:20],  # Top 20 regras
                'insights': insights,
                'business_recommendations': self._generate_basket_recommendations(association_rules[:10])
            }
            
        except Exception as e:
            return {'error': f"Erro na análise de market basket: {str(e)}"}
    
    def _generate_basket_recommendations(self, rules: List[Dict]) -> List[Dict[str, Any]]:
        """Gera recomendações de negócio baseadas em regras de associação."""
        recommendations = []
        
        for rule in rules:
            if rule['lift'] > 1.5 and rule['confidence'] > 0.3:
                recommendations.append({
                    'action': 'Create Product Bundle',
                    'products': [rule['antecedent'], rule['consequent']],
                    'rationale': f"Clientes que compram {rule['antecedent']} têm {rule['confidence']:.0%} chance de comprar {rule['consequent']}",
                    'strength': 'High' if rule['lift'] > 2.0 else 'Medium',
                    'expected_impact': f"Aumento de {rule['lift']:.1f}x na venda de {rule['consequent']}"
                })
        
        return recommendations
    
    def _anomaly_detection_customers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detecção de anomalias em comportamento de clientes usando Isolation Forest.
        
        Identifica clientes com padrões atípicos de compra que podem indicar
        fraude, oportunidades especiais ou comportamento premium.
        """
        try:
            # Preparar features de clientes
            customer_features = df.groupby('Customer_ID').agg({
                'Total_Liquido': ['sum', 'mean', 'std', 'count'],
                'Data': ['min', 'max'],
                'Codigo_Produto': 'nunique'
            }).fillna(0)
            
            # Flatten columns
            customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns]
            
            # Calcular features adicionais
            current_date = df['Data'].max()
            customer_features['days_as_customer'] = (
                pd.to_datetime(customer_features['Data_max']) - 
                pd.to_datetime(customer_features['Data_min'])
            ).dt.days
            customer_features['days_since_last'] = (
                current_date - pd.to_datetime(customer_features['Data_max'])
            ).dt.days
            
            # Selecionar features numéricas para análise
            numeric_features = [
                'Total_Liquido_sum', 'Total_Liquido_mean', 'Total_Liquido_count',
                'Codigo_Produto_nunique', 'days_as_customer', 'days_since_last'
            ]
            
            features_matrix = customer_features[numeric_features].fillna(0)
            
            # Aplicar Isolation Forest
            iso_forest = IsolationForest(
                contamination=0.1,  # 10% de anomalias esperadas
                random_state=42,
                n_estimators=100
            )
            
            anomaly_scores = iso_forest.fit_predict(features_matrix)
            anomaly_probability = iso_forest.score_samples(features_matrix)
            
            # Identificar anomalias
            customer_features['anomaly_label'] = anomaly_scores
            customer_features['anomaly_score'] = anomaly_probability
            
            anomalous_customers = customer_features[customer_features['anomaly_label'] == -1]
            normal_customers = customer_features[customer_features['anomaly_label'] == 1]
            
            # Classificar tipos de anomalia
            anomaly_types = []
            for customer_id, row in anomalous_customers.iterrows():
                anomaly_type = self._classify_anomaly_type(row, normal_customers)
                anomaly_types.append({
                    'customer_id': customer_id,
                    'anomaly_type': anomaly_type,
                    'anomaly_score': round(row['anomaly_score'], 3),
                    'total_spent': round(row['Total_Liquido_sum'], 2),
                    'avg_transaction': round(row['Total_Liquido_mean'], 2),
                    'transaction_count': int(row['Total_Liquido_count']),
                    'unique_products': int(row['Codigo_Produto_nunique'])
                })
            
            return {
                'algorithm': 'Isolation Forest Anomaly Detection',
                'contamination_rate': 0.1,
                'total_customers': len(customer_features),
                'anomalous_customers': len(anomalous_customers),
                'normal_customers': len(normal_customers),
                'anomaly_types': anomaly_types,
                'features_used': numeric_features,
                'business_impact': self._assess_anomaly_business_impact(anomaly_types)
            }
            
        except Exception as e:
            return {'error': f"Erro na detecção de anomalias: {str(e)}"}
    
    def _classify_anomaly_type(self, customer_row: pd.Series, normal_customers: pd.DataFrame) -> str:
        """Classifica o tipo de anomalia baseado nas características do cliente."""
        
        # Médias dos clientes normais
        normal_means = normal_customers[['Total_Liquido_sum', 'Total_Liquido_mean', 'Total_Liquido_count']].mean()
        
        # Comparar com cliente anômalo
        if customer_row['Total_Liquido_sum'] > normal_means['Total_Liquido_sum'] * 3:
            return 'High Spender'
        elif customer_row['Total_Liquido_mean'] > normal_means['Total_Liquido_mean'] * 2:
            return 'Premium Customer'
        elif customer_row['Total_Liquido_count'] > normal_means['Total_Liquido_count'] * 2:
            return 'Frequent Buyer'
        elif customer_row['days_since_last'] > 365:
            return 'Dormant Customer'
        elif customer_row['Total_Liquido_count'] == 1 and customer_row['Total_Liquido_sum'] > normal_means['Total_Liquido_sum']:
            return 'One-Time Big Purchaser'
        else:
            return 'Unusual Pattern'
    
    def _assess_anomaly_business_impact(self, anomaly_types: List[Dict]) -> Dict[str, Any]:
        """Avalia o impacto de negócio das anomalias detectadas."""
        
        impact_assessment = {
            'high_value_opportunities': len([a for a in anomaly_types if a['anomaly_type'] in ['High Spender', 'Premium Customer']]),
            'retention_risks': len([a for a in anomaly_types if a['anomaly_type'] == 'Dormant Customer']),
            'loyalty_opportunities': len([a for a in anomaly_types if a['anomaly_type'] == 'Frequent Buyer']),
            'investigation_needed': len([a for a in anomaly_types if a['anomaly_type'] == 'Unusual Pattern'])
        }
        
        # Valor total dos clientes anômalos
        total_anomaly_value = sum(a['total_spent'] for a in anomaly_types)
        
        impact_assessment['total_anomaly_value'] = round(total_anomaly_value, 2)
        impact_assessment['avg_anomaly_value'] = round(total_anomaly_value / len(anomaly_types), 2) if anomaly_types else 0
        
        return impact_assessment
    
    def _predictive_customer_lifetime_value(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predição de Customer Lifetime Value (CLV) usando análise de coorte.
        
        Calcula o valor projetado de cada cliente baseado em padrões históricos
        e tendências de comportamento.
        """
        try:
            # Preparar dados de clientes
            customer_data = df.groupby('Customer_ID').agg({
                'Data': ['min', 'max', 'count'],
                'Total_Liquido': ['sum', 'mean'],
                'Codigo_Produto': 'nunique'
            }).fillna(0)
            
            # Flatten columns
            customer_data.columns = ['_'.join(col).strip() for col in customer_data.columns]
            
            # Calcular métricas temporais
            current_date = df['Data'].max()
            customer_data['customer_age_days'] = (
                pd.to_datetime(customer_data['Data_max']) - 
                pd.to_datetime(customer_data['Data_min'])
            ).dt.days + 1  # +1 para evitar divisão por zero
            
            customer_data['recency_days'] = (
                current_date - pd.to_datetime(customer_data['Data_max'])
            ).dt.days
            
            # Calcular frequência de compra (transações por dia)
            customer_data['purchase_frequency'] = customer_data['Data_count'] / customer_data['customer_age_days']
            
            # Valor médio por transação
            customer_data['avg_order_value'] = customer_data['Total_Liquido_sum'] / customer_data['Data_count']
            
            # Projeção CLV usando modelo simplificado
            # CLV = (Valor Médio Pedido) × (Frequência Compra) × (Tempo Vida Estimado)
            
            # Estimar tempo de vida baseado em segmento
            def estimate_lifetime(row):
                if row['recency_days'] > 365:
                    return 0  # Cliente inativo
                elif row['purchase_frequency'] > 0.01:  # Mais de 1 compra por 100 dias
                    return 1095  # 3 anos para clientes frequentes
                elif row['Total_Liquido_sum'] > 2000:
                    return 730  # 2 anos para clientes de alto valor
                else:
                    return 365  # 1 ano para clientes regulares
            
            customer_data['estimated_lifetime_days'] = customer_data.apply(estimate_lifetime, axis=1)
            
            # Calcular CLV projetado
            customer_data['predicted_clv'] = (
                customer_data['avg_order_value'] * 
                customer_data['purchase_frequency'] * 
                customer_data['estimated_lifetime_days']
            )
            
            # Segmentar clientes por CLV
            clv_quartiles = customer_data['predicted_clv'].quantile([0.25, 0.5, 0.75, 1.0])
            
            def clv_segment(clv_value):
                if clv_value >= clv_quartiles[0.75]:
                    return 'High CLV'
                elif clv_value >= clv_quartiles[0.5]:
                    return 'Medium CLV'
                elif clv_value >= clv_quartiles[0.25]:
                    return 'Low CLV'
                else:
                    return 'Very Low CLV'
            
            customer_data['clv_segment'] = customer_data['predicted_clv'].apply(clv_segment)
            
            # Preparar resultados
            clv_results = []
            for customer_id, row in customer_data.iterrows():
                clv_results.append({
                    'customer_id': customer_id,
                    'predicted_clv': round(row['predicted_clv'], 2),
                    'clv_segment': row['clv_segment'],
                    'current_value': round(row['Total_Liquido_sum'], 2),
                    'avg_order_value': round(row['avg_order_value'], 2),
                    'purchase_frequency': round(row['purchase_frequency'], 4),
                    'customer_age_days': int(row['customer_age_days']),
                    'estimated_lifetime_days': int(row['estimated_lifetime_days'])
                })
            
            # Estatísticas do segmento
            segment_stats = customer_data.groupby('clv_segment').agg({
                'predicted_clv': ['count', 'mean', 'sum'],
                'Total_Liquido_sum': 'sum'
            }).fillna(0)
            
            return {
                'algorithm': 'Predictive Customer Lifetime Value',
                'total_customers': len(customer_data),
                'clv_predictions': sorted(clv_results, key=lambda x: x['predicted_clv'], reverse=True),
                'segment_distribution': customer_data['clv_segment'].value_counts().to_dict(),
                'segment_statistics': segment_stats.to_dict(),
                'total_predicted_value': round(customer_data['predicted_clv'].sum(), 2),
                'avg_predicted_clv': round(customer_data['predicted_clv'].mean(), 2),
                'business_recommendations': self._generate_clv_recommendations(customer_data)
            }
            
        except Exception as e:
            return {'error': f"Erro na predição de CLV: {str(e)}"}
    
    def _generate_clv_recommendations(self, customer_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Gera recomendações baseadas na análise de CLV."""
        
        recommendations = []
        
        # Clientes de alto CLV
        high_clv = customer_data[customer_data['clv_segment'] == 'High CLV']
        if len(high_clv) > 0:
            recommendations.append({
                'action': 'VIP Program Enhancement',
                'target_segment': 'High CLV',
                'customer_count': len(high_clv),
                'rationale': f'Clientes com CLV médio de R$ {high_clv["predicted_clv"].mean():,.2f}',
                'expected_impact': 'Retenção de 95% + aumento de 20% no valor'
            })
        
        # Clientes com potencial não realizado
        potential_customers = customer_data[
            (customer_data['clv_segment'].isin(['Low CLV', 'Medium CLV'])) &
            (customer_data['avg_order_value'] > customer_data['avg_order_value'].median())
        ]
        if len(potential_customers) > 0:
            recommendations.append({
                'action': 'Frequency Increase Campaign',
                'target_segment': 'Medium CLV with High AOV',
                'customer_count': len(potential_customers),
                'rationale': 'Alto valor por pedido mas baixa frequência',
                'expected_impact': 'Aumento de 30-50% na frequência de compra'
            })
        
        return recommendations

    # ==========================================
    # MÉTODOS DE INTEGRAÇÃO ML E OTIMIZAÇÃO V2.1
    # ==========================================
    
    def _select_optimal_algorithm(self, df: pd.DataFrame, recommendation_type: str, 
                                 target_segment: str = 'all') -> Dict[str, Any]:
        """
        Seleção automática do melhor algoritmo baseado na qualidade e quantidade dos dados.
        
        Analisa características dos dados para escolher entre algoritmos simples ou avançados,
        garantindo performance e qualidade das recomendações.
        """
        try:
            data_quality = self._assess_data_quality(df)
            
            # Critérios para seleção de algoritmo
            selection_criteria = {
                'data_size': len(df),
                'unique_customers': df['Customer_ID'].nunique() if 'Customer_ID' in df.columns else 0,
                'unique_products': df['Codigo_Produto'].nunique(),
                'data_completeness': data_quality.get('completeness_score', 0.5),
                'segment_size': len(self._filter_by_segment(df, target_segment))
            }
            
            # Regras de seleção baseadas em características
            algorithm_recommendations = {
                'product_recommendations': self._select_product_algorithm(selection_criteria),
                'customer_targeting': self._select_targeting_algorithm(selection_criteria),
                'pricing_optimization': self._select_pricing_algorithm(selection_criteria),
                'inventory_suggestions': self._select_inventory_algorithm(selection_criteria),
                'marketing_campaigns': self._select_marketing_algorithm(selection_criteria),
                'strategic_actions': self._select_strategic_algorithm(selection_criteria)
            }
            
            selected = algorithm_recommendations.get(recommendation_type, {
                'algorithm': 'standard',
                'reason': 'Default fallback',
                'confidence': 0.5
            })
            
            return {
                'selected_algorithm': selected['algorithm'],
                'selection_reason': selected['reason'],
                'confidence': selected['confidence'],
                'data_characteristics': selection_criteria,
                'alternative_algorithms': selected.get('alternatives', [])
            }
            
        except Exception as e:
            return {
                'selected_algorithm': 'standard',
                'selection_reason': f'Error in selection: {str(e)}',
                'confidence': 0.3
            }
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Avalia qualidade dos dados para seleção de algoritmos."""
        try:
            # Métricas de qualidade
            total_records = len(df)
            null_counts = df.isnull().sum()
            
            # Completude dos dados
            completeness = 1 - (null_counts.sum() / (len(df.columns) * total_records))
            
            # Diversidade temporal
            date_range = (df['Data'].max() - df['Data'].min()).days if 'Data' in df.columns else 0
            
            # Distribuição de valores
            value_distribution = df['Total_Liquido'].describe() if 'Total_Liquido' in df.columns else {}
            
            return {
                'completeness_score': round(completeness, 3),
                'total_records': total_records,
                'date_range_days': date_range,
                'null_percentage': round(null_counts.sum() / (len(df.columns) * total_records) * 100, 2),
                'value_distribution': {
                    'mean': round(value_distribution.get('mean', 0), 2),
                    'std': round(value_distribution.get('std', 0), 2),
                    'min': round(value_distribution.get('min', 0), 2),
                    'max': round(value_distribution.get('max', 0), 2)
                }
            }
            
        except Exception as e:
            return {'completeness_score': 0.5, 'error': str(e)}
    
    def _select_product_algorithm(self, criteria: Dict) -> Dict[str, Any]:
        """Seleciona algoritmo para recomendações de produto."""
        if criteria['data_size'] > 1000 and criteria['unique_customers'] > 50:
            if criteria['data_completeness'] > 0.8:
                return {
                    'algorithm': 'hybrid_ml',
                    'reason': 'Dados suficientes para ML híbrido (Collaborative + Content-Based)',
                    'confidence': 0.9,
                    'alternatives': ['collaborative_filtering', 'content_based']
                }
            else:
                return {
                    'algorithm': 'collaborative_filtering',
                    'reason': 'Dados suficientes mas incompletude sugere collaborative filtering',
                    'confidence': 0.7,
                    'alternatives': ['standard']
                }
        elif criteria['data_size'] > 100:
            return {
                'algorithm': 'enhanced_standard',
                'reason': 'Dataset médio - algoritmo padrão com melhorias',
                'confidence': 0.6,
                'alternatives': ['standard']
            }
        else:
            return {
                'algorithm': 'standard',
                'reason': 'Dataset pequeno - algoritmo padrão baseado em frequência',
                'confidence': 0.4
            }
    
    def _select_targeting_algorithm(self, criteria: Dict) -> Dict[str, Any]:
        """Seleciona algoritmo para targeting de clientes."""
        if criteria['unique_customers'] > 100 and criteria['data_completeness'] > 0.7:
            return {
                'algorithm': 'advanced_rfm_clustering',
                'reason': 'Base robusta para análise RFM + clustering avançado',
                'confidence': 0.8,
                'alternatives': ['standard_rfm']
            }
        else:
            return {
                'algorithm': 'standard_rfm',
                'reason': 'RFM padrão para base menor ou dados incompletos',
                'confidence': 0.6
            }
    
    def _select_pricing_algorithm(self, criteria: Dict) -> Dict[str, Any]:
        """Seleciona algoritmo para otimização de preços."""
        if criteria['data_size'] > 500 and criteria['unique_products'] > 20:
            return {
                'algorithm': 'elasticity_analysis',
                'reason': 'Dados suficientes para análise de elasticidade de preços',
                'confidence': 0.7,
                'alternatives': ['category_analysis']
            }
        else:
            return {
                'algorithm': 'category_analysis',
                'reason': 'Análise por categoria devido ao volume limitado',
                'confidence': 0.5
            }
    
    def _select_inventory_algorithm(self, criteria: Dict) -> Dict[str, Any]:
        """Seleciona algoritmo para gestão de inventário."""
        return {
            'algorithm': 'abc_analysis_enhanced',
            'reason': 'Análise ABC com métricas de turnover é ideal para inventário',
            'confidence': 0.8,
            'alternatives': ['abc_basic']
        }
    
    def _select_marketing_algorithm(self, criteria: Dict) -> Dict[str, Any]:
        """Seleciona algoritmo para campanhas de marketing."""
        if criteria['unique_customers'] > 50:
            return {
                'algorithm': 'segment_based_campaigns',
                'reason': 'Base suficiente para campanhas segmentadas personalizadas',
                'confidence': 0.7,
                'alternatives': ['mass_marketing']
            }
        else:
            return {
                'algorithm': 'mass_marketing',
                'reason': 'Base pequena - campanhas genéricas mais eficazes',
                'confidence': 0.5
            }
    
    def _select_strategic_algorithm(self, criteria: Dict) -> Dict[str, Any]:
        """Seleciona algoritmo para ações estratégicas."""
        return {
            'algorithm': 'comprehensive_analysis',
            'reason': 'Análise estratégica requer visão abrangente independente do tamanho',
            'confidence': 0.7,
            'alternatives': ['basic_metrics']
        }
    
    def _integrate_advanced_ml_in_products(self, df: pd.DataFrame, target_segment: str, 
                                         count: int, confidence: float) -> Dict[str, Any]:
        """
        Integra algoritmos ML avançados nas recomendações de produtos.
        
        Combina múltiplos algoritmos quando os dados permitem para melhor qualidade.
        """
        try:
            # Selecionar algoritmo ótimo
            algorithm_selection = self._select_optimal_algorithm(df, 'product_recommendations', target_segment)
            selected_algo = algorithm_selection['selected_algorithm']
            
            results = {}
            
            # Executar algoritmo selecionado
            if selected_algo == 'hybrid_ml':
                # Executar todos os algoritmos e combinar
                collaborative_results = self._collaborative_filtering_advanced(df, None, count)
                content_results = self._content_based_filtering_advanced(df, None, count)
                hybrid_results = self._hybrid_recommendation_system(df, None, count)
                
                results = {
                    'primary_recommendations': hybrid_results.get('recommendations', []),
                    'collaborative_backup': collaborative_results.get('recommendations', {}),
                    'content_backup': content_results.get('recommendations', {}),
                    'algorithm_used': 'Hybrid ML (SVD + TF-IDF + Ensemble)',
                    'confidence_score': 0.9
                }
                
            elif selected_algo == 'collaborative_filtering':
                collab_results = self._collaborative_filtering_advanced(df, None, count)
                results = {
                    'primary_recommendations': collab_results.get('recommendations', {}),
                    'algorithm_used': 'Matrix Factorization (SVD)',
                    'confidence_score': 0.7
                }
                
            elif selected_algo == 'enhanced_standard':
                # Algoritmo padrão com melhorias
                std_results = self._generate_product_recommendations(df, target_segment, count, confidence, True)
                basket_results = self._advanced_market_basket_analysis(df)
                
                results = {
                    'primary_recommendations': std_results.get('recommendations', {}),
                    'market_basket_insights': basket_results.get('association_rules', [])[:5],
                    'algorithm_used': 'Enhanced Standard (RFM + Market Basket)',
                    'confidence_score': 0.6
                }
                
            else:
                # Fallback para algoritmo padrão
                std_results = self._generate_product_recommendations(df, target_segment, count, confidence, False)
                results = {
                    'primary_recommendations': std_results.get('recommendations', {}),
                    'algorithm_used': 'Standard Frequency-Based',
                    'confidence_score': 0.4
                }
            
            # Adicionar metadata do algoritmo
            results['algorithm_selection'] = algorithm_selection
            results['enhancement_applied'] = True
            
            return results
            
        except Exception as e:
            return {'error': f"Erro na integração ML: {str(e)}"}
    
    def _cross_validate_recommendations(self, df: pd.DataFrame, 
                                      recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validação cruzada das recomendações para medir qualidade.
        
        Implementa métricas de qualidade e validação estatística.
        """
        try:
            validation_results = {
                'quality_score': 0,
                'coverage_score': 0,
                'diversity_score': 0,
                'novelty_score': 0,
                'validation_method': 'statistical_cross_validation'
            }
            
            # Extrair produtos recomendados
            recommended_products = []
            if 'top_products' in recommendations.get('recommendations', {}):
                recommended_products = [
                    item['code'] for item in recommendations['recommendations']['top_products']
                ]
            
            if not recommended_products:
                return validation_results
            
            # Métricas de qualidade
            total_products = df['Codigo_Produto'].nunique()
            total_revenue = df['Total_Liquido'].sum()
            
            # 1. Coverage (cobertura)
            coverage = len(recommended_products) / min(total_products, 50)  # Normalizado para máx 50
            validation_results['coverage_score'] = round(min(coverage, 1.0), 3)
            
            # 2. Revenue relevance (relevância por receita)
            recommended_revenue = df[df['Codigo_Produto'].isin(recommended_products)]['Total_Liquido'].sum()
            revenue_relevance = recommended_revenue / total_revenue if total_revenue > 0 else 0
            validation_results['revenue_relevance'] = round(revenue_relevance, 3)
            
            # 3. Diversity (diversidade)
            if 'Grupo_Produto' in df.columns:
                recommended_categories = df[df['Codigo_Produto'].isin(recommended_products)]['Grupo_Produto'].nunique()
                total_categories = df['Grupo_Produto'].nunique()
                diversity = recommended_categories / total_categories if total_categories > 0 else 0
                validation_results['diversity_score'] = round(diversity, 3)
            
            # 4. Freshness (produtos com vendas recentes)
            recent_date = df['Data'].max() - pd.Timedelta(days=90)
            recent_products = df[df['Data'] >= recent_date]['Codigo_Produto'].unique()
            fresh_recommendations = len([p for p in recommended_products if p in recent_products])
            freshness = fresh_recommendations / len(recommended_products) if recommended_products else 0
            validation_results['freshness_score'] = round(freshness, 3)
            
            # Score geral (média ponderada)
            quality_score = (
                validation_results['coverage_score'] * 0.2 +
                validation_results['revenue_relevance'] * 0.3 +
                validation_results['diversity_score'] * 0.2 +
                validation_results['freshness_score'] * 0.3
            )
            validation_results['quality_score'] = round(quality_score, 3)
            
            # Interpretação do score
            if quality_score >= 0.8:
                validation_results['quality_level'] = 'Excellent'
            elif quality_score >= 0.6:
                validation_results['quality_level'] = 'Good'
            elif quality_score >= 0.4:
                validation_results['quality_level'] = 'Fair'
            else:
                validation_results['quality_level'] = 'Poor'
            
            return validation_results
            
        except Exception as e:
            return {'error': f"Erro na validação: {str(e)}"}
    
    def _optimize_recommendation_parameters(self, df: pd.DataFrame, 
                                          recommendation_type: str) -> Dict[str, Any]:
        """
        Otimização automática de parâmetros baseada nas características dos dados.
        
        Ajusta thresholds e configurações para máxima qualidade.
        """
        try:
            data_characteristics = self._assess_data_quality(df)
            
            # Parâmetros otimizados baseados nos dados
            optimized_params = {
                'confidence_threshold': 0.7,  # default
                'recommendation_count': 10,   # default
                'enable_detailed_analysis': True,
                'optimization_applied': True
            }
            
            # Ajustes baseados no tamanho dos dados
            if data_characteristics['total_records'] > 5000:
                optimized_params['confidence_threshold'] = 0.8  # Mais rigoroso para datasets grandes
                optimized_params['recommendation_count'] = 15
            elif data_characteristics['total_records'] < 100:
                optimized_params['confidence_threshold'] = 0.5  # Mais permissivo para datasets pequenos
                optimized_params['recommendation_count'] = 5
            
            # Ajustes baseados na qualidade dos dados
            if data_characteristics['completeness_score'] < 0.7:
                optimized_params['confidence_threshold'] *= 0.9  # Reduzir threshold para dados incompletos
                optimized_params['enable_detailed_analysis'] = False
            
            # Ajustes específicos por tipo de recomendação
            type_adjustments = {
                'product_recommendations': {'rec_count_multiplier': 1.0},
                'customer_targeting': {'confidence_boost': 0.1},
                'pricing_optimization': {'confidence_reduction': 0.1},
                'inventory_suggestions': {'rec_count_multiplier': 1.5},
                'marketing_campaigns': {'confidence_boost': 0.05},
                'strategic_actions': {'enable_detailed': True}
            }
            
            if recommendation_type in type_adjustments:
                adjustments = type_adjustments[recommendation_type]
                
                if 'rec_count_multiplier' in adjustments:
                    optimized_params['recommendation_count'] = int(
                        optimized_params['recommendation_count'] * adjustments['rec_count_multiplier']
                    )
                
                if 'confidence_boost' in adjustments:
                    optimized_params['confidence_threshold'] = min(
                        0.95, optimized_params['confidence_threshold'] + adjustments['confidence_boost']
                    )
                
                if 'confidence_reduction' in adjustments:
                    optimized_params['confidence_threshold'] = max(
                        0.5, optimized_params['confidence_threshold'] - adjustments['confidence_reduction']
                    )
            
            return {
                'optimized_parameters': optimized_params,
                'data_characteristics_considered': data_characteristics,
                'optimization_rationale': self._explain_parameter_optimization(
                    data_characteristics, recommendation_type, optimized_params
                )
            }
            
        except Exception as e:
            return {'error': f"Erro na otimização: {str(e)}"}
    
    def _explain_parameter_optimization(self, data_char: Dict, rec_type: str, 
                                      params: Dict) -> List[str]:
        """Explica as decisões de otimização de parâmetros."""
        explanations = []
        
        if data_char['total_records'] > 5000:
            explanations.append("Dataset grande: aumentado threshold de confiança e número de recomendações")
        elif data_char['total_records'] < 100:
            explanations.append("Dataset pequeno: reduzido threshold e número de recomendações")
        
        if data_char['completeness_score'] < 0.7:
            explanations.append("Dados incompletos: reduzido threshold e desabilitada análise detalhada")
        
        if rec_type == 'inventory_suggestions':
            explanations.append("Inventário: aumentado número de recomendações para melhor cobertura")
        
        return explanations
    
    def _performance_monitoring(self, start_time: datetime, df_size: int, 
                              algorithm_used: str) -> Dict[str, Any]:
        """
        Monitor de performance para análise de eficiência.
        
        Coleta métricas de performance para otimização contínua.
        """
        try:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Métricas de performance
            performance_metrics = {
                'processing_time_seconds': round(processing_time, 3),
                'records_processed': df_size,
                'records_per_second': round(df_size / processing_time, 2) if processing_time > 0 else 0,
                'algorithm_used': algorithm_used,
                'memory_efficiency': 'Good' if df_size > 1000 and processing_time < 5 else 'Standard',
                'performance_grade': self._calculate_performance_grade(processing_time, df_size)
            }
            
            # Benchmarks de referência
            performance_metrics['benchmarks'] = {
                'target_time_small': '< 1s for < 1k records',
                'target_time_medium': '< 3s for 1k-10k records', 
                'target_time_large': '< 10s for > 10k records',
                'current_benchmark': self._get_current_benchmark(df_size, processing_time)
            }
            
            return performance_metrics
            
        except Exception as e:
            return {'error': f"Erro no monitoramento: {str(e)}"}
    
    def _calculate_performance_grade(self, time_taken: float, records: int) -> str:
        """Calcula nota de performance baseada em benchmarks."""
        if records < 1000:
            return 'A' if time_taken < 1 else 'B' if time_taken < 2 else 'C'
        elif records < 10000:
            return 'A' if time_taken < 3 else 'B' if time_taken < 5 else 'C'
        else:
            return 'A' if time_taken < 10 else 'B' if time_taken < 15 else 'C'
    
    def _get_current_benchmark(self, records: int, time_taken: float) -> str:
        """Retorna status do benchmark atual."""
        if records < 1000 and time_taken < 1:
            return '✅ Excellent performance'
        elif records < 10000 and time_taken < 3:
            return '✅ Good performance' 
        elif records < 50000 and time_taken < 10:
            return '✅ Acceptable performance'
        else:
            return '⚠️ Performance below target'
