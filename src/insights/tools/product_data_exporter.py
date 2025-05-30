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
    try:
        from insights.tools.shared.data_preparation import DataPreparationMixin
        from insights.tools.shared.report_formatter import ReportFormatterMixin
        from insights.tools.shared.business_mixins import JewelryRFMAnalysisMixin, JewelryBusinessAnalysisMixin
    except ImportError:
        # Se não conseguir importar, usar versão local ou criar stubs
        print("⚠️ Importações locais não encontradas, usando implementação básica...")
        
        class DataPreparationMixin:
            """Stub básico para DataPreparationMixin"""
            pass
        
        class ReportFormatterMixin:
            """Stub básico para ReportFormatterMixin"""
            pass
        
        class JewelryRFMAnalysisMixin:
            """Stub básico para JewelryRFMAnalysisMixin"""
            def perform_abc_analysis(self, df, dimension='product'):
                return {"error": "Stub implementation"}
            
        class JewelryBusinessAnalysisMixin:
            """Stub básico para JewelryBusinessAnalysisMixin"""
            def create_product_bcg_matrix(self, df):
                return {"error": "Stub implementation"}

warnings.filterwarnings('ignore')

class ProductDataExporterInput(BaseModel):
    """Schema para exportação de dados de produtos."""
    
    data_csv: str = Field(
        default="data/vendas.csv", 
        description="Caminho para arquivo CSV de vendas"
    )
    
    output_path: str = Field(
        default="assets/data/analise_produtos_dados_completos.csv",
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
             output_path: str = "assets/data/analise_produtos_dados_completos.csv",
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
        
        # Implementar classificação ABC diretamente sem depender de colunas que podem não existir
        print("🔍 Implementando classificação ABC personalizada...")
        
        # Calcular classificação ABC manualmente usando apenas Receita_Total
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
        
        print(f"✅ Classificação ABC concluída sem erros")
        return product_data
    
    def _add_bcg_classification(self, df: pd.DataFrame, product_data: pd.DataFrame) -> pd.DataFrame:
        """Adicionar classificação da matriz BCG."""
        
        # Implementar classificação BCG usando apenas dados disponíveis
        print("🔍 Implementando classificação BCG personalizada...")
        
        # Market Share = % da receita total
        # Growth Rate = giro anual estimado (proxy para crescimento)
        
        total_revenue = product_data['Receita_Total'].sum()
        product_data['Market_Share_BCG'] = product_data['Receita_Total'] / total_revenue * 100
        
        # Usar giro anual como proxy para crescimento
        product_data['Growth_Rate_BCG'] = product_data['Giro_Anual_Estimado']
        
        # Medianas para classificação
        market_share_median = product_data['Market_Share_BCG'].median()
        growth_rate_median = product_data['Growth_Rate_BCG'].median()
        
        def classify_bcg(row):
            ms = row['Market_Share_BCG']
            gr = row['Growth_Rate_BCG']
            
            if ms > market_share_median and gr > growth_rate_median:
                return 'Stars'
            elif ms > market_share_median and gr <= growth_rate_median:
                return 'Cash_Cows'
            elif ms <= market_share_median and gr > growth_rate_median:
                return 'Question_Marks'
            else:
                return 'Dogs'
        
        product_data['Classificacao_BCG'] = product_data.apply(classify_bcg, axis=1)
        
        # Limpar colunas auxiliares
        product_data.drop(['Market_Share_BCG', 'Growth_Rate_BCG'], axis=1, inplace=True)
        
        print(f"✅ Classificação BCG concluída sem erros")
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
        min_revenue = product_data['Receita_Total'].min()
        product_data['Revenue_Score'] = (
            (product_data['Receita_Total'] - min_revenue) / (max_revenue - min_revenue) * 100
        ).round(2)
        
        # Frequency Score (baseado em número de transações - normalizado)
        max_transactions = product_data['Num_Transacoes'].max()
        min_transactions = product_data['Num_Transacoes'].min()
        if max_transactions > min_transactions:
            product_data['Frequency_Score'] = (
                (product_data['Num_Transacoes'] - min_transactions) / (max_transactions - min_transactions) * 100
            ).round(2)
        else:
            product_data['Frequency_Score'] = 50.0  # Score médio se todos iguais
        
        # Recency Score (inverso dos dias desde última venda - melhorado)
        max_days = product_data['Days_Since_Last_Sale'].max()
        
        # Usar escala logarítmica para penalizar menos produtos recentes
        import numpy as np
        
        # Inverter a escala: menos dias = score maior
        product_data['Recency_Score'] = np.where(
            product_data['Days_Since_Last_Sale'] <= 30,  # Últimos 30 dias = score alto
            100,
            np.where(
                product_data['Days_Since_Last_Sale'] <= 90,  # Últimos 90 dias = score médio-alto
                80 - (product_data['Days_Since_Last_Sale'] - 30) * 0.5,
                np.where(
                    product_data['Days_Since_Last_Sale'] <= 180,  # Últimos 180 dias = score médio
                    50 - (product_data['Days_Since_Last_Sale'] - 90) * 0.3,
                    np.maximum(10, 20 - (product_data['Days_Since_Last_Sale'] - 180) * 0.1)  # Mínimo 10
                )
            )
        ).round(2)
        
        # Turnover Score (baseado no giro anual)
        max_turnover = product_data['Giro_Anual_Estimado'].max()
        if max_turnover > 0:
            # Normalizar giro com caps
            turnover_capped = np.minimum(product_data['Giro_Anual_Estimado'], 50)  # Cap em 50 giros/ano
            product_data['Turnover_Score'] = (turnover_capped / 50 * 100).round(2)
        else:
            product_data['Turnover_Score'] = 0.0
        
        # Score Geral (média ponderada melhorada)
        product_data['Score_Geral'] = (
            product_data['Revenue_Score'] * 0.3 +      # Importância da receita
            product_data['Frequency_Score'] * 0.25 +   # Importância da frequência
            product_data['Recency_Score'] * 0.25 +     # Importância da recência
            product_data['Turnover_Score'] * 0.2       # Importância do giro
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

    def generate_product_test_report(self, test_data: dict) -> str:
        """Gera relatório visual completo dos testes de produtos em formato markdown."""
        
        # Coletar dados com fallbacks
        metadata = test_data.get('metadata', {})
        data_metrics = test_data.get('data_metrics', {})
        results = test_data.get('results', {})
        component_tests = test_data.get('component_tests', {})
        
        report = [
            "# 🏷️ Teste Completo de Produtos - Relatório Executivo",
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
        
        # Principais Descobertas dos Produtos
        report.append("\n## 📊 Principais Descobertas dos Produtos")
        
        # Estatísticas ABC
        if 'abc_classification' in results and results['abc_classification'].get('success'):
            abc_data = results['abc_classification']
            if 'abc_distribution' in abc_data:
                abc_dist = abc_data['abc_distribution']
                report.append(f"- **Classificação ABC:** Classe A: {abc_dist.get('A', 0)} produtos, Classe B: {abc_dist.get('B', 0)} produtos, Classe C: {abc_dist.get('C', 0)} produtos")
        
        # Matriz BCG
        if 'bcg_classification' in results and results['bcg_classification'].get('success'):
            bcg_data = results['bcg_classification']
            if 'bcg_distribution' in bcg_data:
                bcg_dist = bcg_data['bcg_distribution']
                report.append(f"- **Matriz BCG:** Stars: {bcg_dist.get('Stars', 0)}, Cash Cows: {bcg_dist.get('Cash_Cows', 0)}, Question Marks: {bcg_dist.get('Question_Marks', 0)}, Dogs: {bcg_dist.get('Dogs', 0)}")
        
        # Ciclo de Vida
        if 'lifecycle_analysis' in results and results['lifecycle_analysis'].get('success'):
            lifecycle_data = results['lifecycle_analysis']
            dead_stock = lifecycle_data.get('dead_stock_count', 0)
            slow_movers = lifecycle_data.get('slow_mover_count', 0)
            high_velocity = lifecycle_data.get('high_velocity_count', 0)
            report.append(f"- **Ciclo de Vida:** {high_velocity} alta velocidade, {slow_movers} slow movers, {dead_stock} dead stock")
        
        # Métricas Avançadas
        if 'advanced_metrics' in results and results['advanced_metrics'].get('success'):
            metrics_data = results['advanced_metrics']
            avg_score = metrics_data.get('avg_general_score', 0)
            report.append(f"- **Score Geral Médio:** {avg_score:.1f}/100")
        
        # Detalhamento por Componente
        report.append("\n## 🔧 Detalhamento dos Componentes Testados")
        
        component_categories = {
            'Preparação de Dados': ['data_loading', 'data_aggregation'],
            'Classificações': ['abc_classification', 'bcg_classification'],
            'Análises': ['lifecycle_analysis', 'advanced_metrics'],
            'Alertas e Flags': ['alert_flags'],
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
        report.append("\n## 💡 Recomendações do Sistema de Produtos")
        
        recommendations = [
            "📊 Focar na gestão de produtos Classe A para maximizar receita",
            "🚨 Implementar estratégias para produtos Dead Stock",
            "🌟 Investir no crescimento de produtos Stars da matriz BCG",
            "📈 Utilizar scores para priorizar ações de marketing",
            "🔄 Revisar ciclo de vida para otimizar portfólio"
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

    def run_full_product_test(self) -> str:
        """Executa teste completo e retorna relatório formatado"""
        test_result = self.test_all_product_components()
        parsed = json.loads(test_result)
        return self.generate_product_test_report(parsed)

    def test_all_product_components(self, sample_data: str = "data/vendas.csv") -> str:
        """
        Executa teste completo de todos os componentes da classe ProductDataExporter
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
                "test_version": "Product Test Suite v1.0",
                "data_source": data_file_path,
                "data_file_specified": "data/vendas.csv",
                "tool_version": "Product Data Exporter v1.0",
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
            print("\n=== ETAPA 1: CARREGAMENTO DE DADOS DE PRODUTOS ===")
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
                "data_quality_check": self._perform_product_data_quality_check(df)
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
                product_data = self._aggregate_product_data(df)
                aggregation_time = time.time() - start_time
                
                test_report["results"]["data_aggregation"] = {
                    "success": True,
                    "metrics": {
                        "processing_time": aggregation_time,
                        "products_aggregated": len(product_data),
                        "columns_generated": len(product_data.columns)
                    }
                }
                print(f"✅ Agregação concluída: {len(product_data)} produtos em {aggregation_time:.3f}s")
                
            except Exception as e:
                self._log_product_test_error(test_report, e, "data_aggregation")
                print(f"❌ Erro na agregação: {str(e)}")
                product_data = pd.DataFrame()  # Fallback vazio

            # 3. Teste de Classificação ABC
            test_report["metadata"]["current_stage"] = "abc_classification"
            print("\n=== ETAPA 3: TESTE DE CLASSIFICAÇÃO ABC ===")
            
            if not product_data.empty:
                try:
                    start_time = time.time()
                    print("🏆 Testando classificação ABC...")
                    product_data_abc = self._add_abc_classification(df, product_data.copy())
                    abc_time = time.time() - start_time
                    
                    # Analisar distribuição ABC
                    abc_distribution = product_data_abc['Classificacao_ABC'].value_counts().to_dict()
                    
                    test_report["results"]["abc_classification"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": abc_time,
                            "products_classified": len(product_data_abc)
                        },
                        "abc_distribution": abc_distribution
                    }
                    print(f"✅ Classificação ABC: {abc_distribution} em {abc_time:.3f}s")
                    
                except Exception as e:
                    self._log_product_test_error(test_report, e, "abc_classification")
                    print(f"❌ Erro na classificação ABC: {str(e)}")
                    product_data_abc = product_data.copy()
            else:
                product_data_abc = pd.DataFrame()

            # 4. Teste de Classificação BCG
            test_report["metadata"]["current_stage"] = "bcg_classification"
            print("\n=== ETAPA 4: TESTE DE MATRIZ BCG ===")
            
            if not product_data_abc.empty:
                try:
                    start_time = time.time()
                    print("📈 Testando matriz BCG...")
                    product_data_bcg = self._add_bcg_classification(df, product_data_abc.copy())
                    bcg_time = time.time() - start_time
                    
                    # Analisar distribuição BCG
                    bcg_distribution = product_data_bcg['Classificacao_BCG'].value_counts().to_dict()
                    
                    test_report["results"]["bcg_classification"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": bcg_time,
                            "products_classified": len(product_data_bcg)
                        },
                        "bcg_distribution": bcg_distribution
                    }
                    print(f"✅ Matriz BCG: {bcg_distribution} em {bcg_time:.3f}s")
                    
                except Exception as e:
                    self._log_product_test_error(test_report, e, "bcg_classification")
                    print(f"❌ Erro na matriz BCG: {str(e)}")
                    product_data_bcg = product_data_abc.copy()
            else:
                product_data_bcg = pd.DataFrame()

            # 5. Teste de Análise de Ciclo de Vida
            test_report["metadata"]["current_stage"] = "lifecycle_analysis"
            print("\n=== ETAPA 5: TESTE DE CICLO DE VIDA ===")
            
            if not product_data_bcg.empty:
                try:
                    start_time = time.time()
                    print("🔄 Testando análise de ciclo de vida...")
                    product_data_lifecycle = self._add_lifecycle_analysis(df, product_data_bcg.copy(), 120, 180)
                    lifecycle_time = time.time() - start_time
                    
                    # Contar por status de ciclo de vida
                    lifecycle_stats = product_data_lifecycle['Status_Lifecycle'].value_counts().to_dict()
                    dead_stock_count = lifecycle_stats.get('Dead_Stock', 0)
                    slow_mover_count = lifecycle_stats.get('Slow_Mover', 0)
                    high_velocity_count = lifecycle_stats.get('High_Velocity', 0)
                    
                    test_report["results"]["lifecycle_analysis"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": lifecycle_time,
                            "products_analyzed": len(product_data_lifecycle)
                        },
                        "dead_stock_count": int(dead_stock_count),
                        "slow_mover_count": int(slow_mover_count),
                        "high_velocity_count": int(high_velocity_count),
                        "lifecycle_distribution": lifecycle_stats
                    }
                    print(f"✅ Ciclo de vida: {high_velocity_count} alta velocidade, {slow_mover_count} slow movers em {lifecycle_time:.3f}s")
                    
                except Exception as e:
                    self._log_product_test_error(test_report, e, "lifecycle_analysis")
                    print(f"❌ Erro no ciclo de vida: {str(e)}")
                    product_data_lifecycle = product_data_bcg.copy()
            else:
                product_data_lifecycle = pd.DataFrame()

            # 6. Teste de Métricas Avançadas
            test_report["metadata"]["current_stage"] = "advanced_metrics"
            print("\n=== ETAPA 6: TESTE DE MÉTRICAS AVANÇADAS ===")
            
            if not product_data_lifecycle.empty:
                try:
                    start_time = time.time()
                    print("📊 Testando métricas avançadas...")
                    product_data_metrics = self._add_advanced_metrics(df, product_data_lifecycle.copy())
                    metrics_time = time.time() - start_time
                    
                    # Calcular estatísticas das métricas
                    avg_general_score = product_data_metrics['Score_Geral'].mean()
                    high_score_products = len(product_data_metrics[product_data_metrics['Score_Geral'] > 70])
                    
                    test_report["results"]["advanced_metrics"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": metrics_time,
                            "products_scored": len(product_data_metrics)
                        },
                        "avg_general_score": avg_general_score,
                        "high_score_products": high_score_products
                    }
                    print(f"✅ Métricas avançadas: score médio {avg_general_score:.1f}, {high_score_products} produtos com score alto em {metrics_time:.3f}s")
                    
                except Exception as e:
                    self._log_product_test_error(test_report, e, "advanced_metrics")
                    print(f"❌ Erro nas métricas avançadas: {str(e)}")
                    product_data_metrics = product_data_lifecycle.copy()
            else:
                product_data_metrics = pd.DataFrame()

            # 7. Teste de Flags de Alertas
            test_report["metadata"]["current_stage"] = "alert_flags"
            print("\n=== ETAPA 7: TESTE DE FLAGS DE ALERTAS ===")
            
            if not product_data_metrics.empty:
                try:
                    start_time = time.time()
                    print("🚨 Testando flags de alertas...")
                    product_data_alerts = self._add_alert_flags(product_data_metrics.copy(), 120, 180)
                    alerts_time = time.time() - start_time
                    
                    # Contar alertas
                    slow_mover_flag = product_data_alerts['Slow_Mover_Flag'].sum()
                    dead_stock_flag = product_data_alerts['Dead_Stock_Flag'].sum()
                    action_required_flag = product_data_alerts['Action_Required_Flag'].sum()
                    
                    test_report["results"]["alert_flags"] = {
                        "success": True,
                        "metrics": {
                            "processing_time": alerts_time,
                            "products_flagged": len(product_data_alerts)
                        },
                        "slow_mover_alerts": int(slow_mover_flag),
                        "dead_stock_alerts": int(dead_stock_flag),
                        "action_required_alerts": int(action_required_flag)
                    }
                    print(f"✅ Flags de alertas: {action_required_flag} produtos requerem ação em {alerts_time:.3f}s")
                    
                except Exception as e:
                    self._log_product_test_error(test_report, e, "alert_flags")
                    print(f"❌ Erro nas flags de alertas: {str(e)}")
                    product_data_alerts = product_data_metrics.copy()
            else:
                product_data_alerts = pd.DataFrame()

            # 8. Teste de Exportação CSV
            test_report["metadata"]["current_stage"] = "csv_export"
            print("\n=== ETAPA 8: TESTE DE EXPORTAÇÃO CSV ===")
            
            if not product_data_alerts.empty:
                try:
                    start_time = time.time()
                    print("💾 Testando exportação CSV...")
                    
                    # Criar pasta de teste
                    test_output_dir = "test_results"
                    os.makedirs(test_output_dir, exist_ok=True)
                    test_output_path = os.path.join(test_output_dir, "product_test_export.csv")
                    
                    export_success = self._export_to_csv(product_data_alerts, test_output_path)
                    export_time = time.time() - start_time
                    
                    if export_success and os.path.exists(test_output_path):
                        file_size_kb = os.path.getsize(test_output_path) / 1024
                        
                        test_report["results"]["csv_export"] = {
                            "success": True,
                            "metrics": {
                                "processing_time": export_time,
                                "file_size_kb": file_size_kb,
                                "records_exported": len(product_data_alerts)
                            },
                            "output_path": test_output_path
                        }
                        print(f"✅ CSV exportado: {file_size_kb:.1f} KB em {export_time:.3f}s")
                        
                        # Armazenar informação do arquivo gerado
                        test_report["component_tests"]["files_generated"] = [{
                            "path": test_output_path,
                            "size_kb": file_size_kb,
                            "type": "product_export"
                        }]
                    else:
                        raise Exception("Falha na exportação do arquivo CSV")
                        
                except Exception as e:
                    self._log_product_test_error(test_report, e, "csv_export")
                    print(f"❌ Erro na exportação: {str(e)}")

            # 9. Teste de Geração de Sumário
            test_report["metadata"]["current_stage"] = "summary_generation"
            print("\n=== ETAPA 9: TESTE DE GERAÇÃO DE SUMÁRIO ===")
            
            if not product_data_alerts.empty:
                try:
                    start_time = time.time()
                    print("📋 Testando geração de sumário...")
                    
                    summary = self._generate_export_summary(product_data_alerts, test_output_path if 'test_output_path' in locals() else "test_path")
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
                    self._log_product_test_error(test_report, e, "summary_generation")
                    print(f"❌ Erro na geração de sumário: {str(e)}")

            # 10. Teste de Configurações Diferentes
            test_report["metadata"]["current_stage"] = "configuration_testing"
            print("\n=== ETAPA 10: TESTE DE CONFIGURAÇÕES ===")
            
            config_tests = {}
            
            # Teste com configurações conservadoras (slow mover mais restritivo)
            try:
                print("🔧 Testando configuração conservadora...")
                start_time = time.time()
                conservative_result = self._run(
                    data_csv=data_file_path,
                    output_path="test_results/conservative_product_test.csv",
                    slow_mover_days=60,
                    dead_stock_days=90
                )
                config_tests["conservative"] = {
                    "success": "❌" not in conservative_result,
                    "description": "Configuração conservadora (60 dias slow mover, 90 dias dead stock)",
                    "execution_time": time.time() - start_time
                }
                print("✅ Configuração conservadora testada")
            except Exception as e:
                config_tests["conservative"] = {"success": False, "error": str(e)}
                print(f"❌ Erro na configuração conservadora: {str(e)}")
            
            # Teste com configurações agressivas (slow mover mais permissivo)
            try:
                print("🔧 Testando configuração agressiva...")
                start_time = time.time()
                aggressive_result = self._run(
                    data_csv=data_file_path,
                    output_path="test_results/aggressive_product_test.csv",
                    slow_mover_days=180,
                    dead_stock_days=365
                )
                config_tests["aggressive"] = {
                    "success": "❌" not in aggressive_result,
                    "description": "Configuração agressiva (180 dias slow mover, 365 dias dead stock)",
                    "execution_time": time.time() - start_time
                }
                print("✅ Configuração agressiva testada")
            except Exception as e:
                config_tests["aggressive"] = {"success": False, "error": str(e)}
                print(f"❌ Erro na configuração agressiva: {str(e)}")
            
            test_report["component_tests"]["configuration_tests"] = config_tests
            
            # 11. Análise de Receita
            if not product_data_alerts.empty and 'Receita_Total' in product_data_alerts.columns:
                total_revenue = product_data_alerts['Receita_Total'].sum()
                avg_revenue_per_product = product_data_alerts['Receita_Total'].mean()
                
                test_report["component_tests"]["revenue_analysis"] = {
                    "total_revenue": float(total_revenue),
                    "avg_revenue_per_product": float(avg_revenue_per_product),
                    "products_above_1k": int(len(product_data_alerts[product_data_alerts['Receita_Total'] > 1000]))
                }

            # 12. Performance Metrics
            test_report["performance_metrics"] = {
                "total_execution_time": sum([
                    result.get('metrics', {}).get('processing_time', 0) 
                    for result in test_report["results"].values() 
                    if isinstance(result, dict)
                ]),
                "memory_usage_mb": self._get_product_memory_usage(),
                "largest_dataset_processed": len(product_data_alerts) if not product_data_alerts.empty else 0
            }

            # 13. Análise Final
            test_report["metadata"]["status"] = "completed" if not test_report["errors"] else "completed_with_errors"
            print(f"\n✅✅✅ TESTE DE PRODUTOS COMPLETO - {len(test_report['errors'])} erros ✅✅✅")
            
            return json.dumps(test_report, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            test_report["metadata"]["status"] = "failed"
            self._log_product_test_error(test_report, e, "global")
            print(f"❌ TESTE DE PRODUTOS FALHOU: {str(e)}")
            return json.dumps(test_report, ensure_ascii=False, indent=2, default=str)

    def _log_product_test_error(self, report: dict, error: Exception, context: str) -> None:
        """Registra erros de teste de produtos de forma estruturada"""
        error_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc()
        }
        report["errors"].append(error_entry)

    def _perform_product_data_quality_check(self, df: pd.DataFrame) -> dict:
        """Executa verificações de qualidade específicas para dados de produtos"""
        checks = {
            "missing_product_codes": int(df['Codigo_Produto'].isnull().sum()) if 'Codigo_Produto' in df.columns else 0,
            "missing_dates": int(df['Data'].isnull().sum()) if 'Data' in df.columns else 0,
            "negative_quantities": int((df['Quantidade'] < 0).sum()) if 'Quantidade' in df.columns else 0,
            "zero_prices": int((df['Total_Liquido'] <= 0).sum()) if 'Total_Liquido' in df.columns else 0,
            "duplicate_transactions": int(df.duplicated().sum()),
            "missing_descriptions": int(df['Descricao_Produto'].isnull().sum()) if 'Descricao_Produto' in df.columns else 0
        }
        return checks

    def _get_product_memory_usage(self) -> float:
        """Obtém uso de memória específico para análises de produtos"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Em MB
        except:
            return 0.0


# Exemplo de uso
if __name__ == "__main__":
    exporter = ProductDataExporter()
    
    print("🏷️ Iniciando Teste Completo do Sistema de Produtos...")
    print("📁 Testando especificamente com: data/vendas.csv")
    
    # Executar teste usando especificamente data/vendas.csv
    report = exporter.run_full_product_test()
    
    # Salvar relatório
    os.makedirs("test_results", exist_ok=True)
    with open("test_results/product_test_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("✅ Relatório de produtos gerado em test_results/product_test_report.md")
    print(f"📁 Teste executado com arquivo: data/vendas.csv")
    print("\n" + "="*80)
    print(report[:1500])  # Exibir parte do relatório no console 