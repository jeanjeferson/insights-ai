"""
🔧 MÓDULO DE PREPARAÇÃO DE DADOS CONSOLIDADO
===================================================

Este módulo centraliza toda a lógica de preparação de dados para joalherias,
eliminando duplicação entre KPI Calculator Tool e Statistical Analysis Tool.

FUNCIONALIDADES:
✅ Validação e limpeza de dados
✅ Conversão de tipos com tratamento de erros
✅ Cálculo de campos derivados financeiros
✅ Cálculo de campos derivados demográficos
✅ Cálculo de campos derivados temporais
✅ Cálculo de campos derivados de estoque
✅ Validação de integridade dos dados
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataPreparationMixin:
    """Mixin class para preparação de dados de joalherias."""
    
    def prepare_jewelry_data(self, df: pd.DataFrame, validation_level: str = "standard") -> Optional[pd.DataFrame]:
        """
        Preparar dados de joalheria de forma completa e padronizada.
        
        Args:
            df: DataFrame com dados brutos
            validation_level: "basic", "standard", "strict"
            
        Returns:
            DataFrame preparado ou None se dados inválidos
        """
        try:
            print("🏗️ Iniciando preparação CONSOLIDADA de dados...")
            
            # 1. Validação inicial
            if not self._validate_data_structure(df, validation_level):
                return None
            
            # 2. Limpeza básica
            df = self._clean_basic_data(df)
            if df is None or len(df) == 0:
                return None
                
            # 3. Conversão de tipos
            df = self._convert_data_types(df)
            if df is None:
                return None
                
            # 4. Cálculo de campos derivados
            df = self._calculate_financial_derived_fields(df)
            df = self._calculate_demographic_derived_fields(df)
            df = self._calculate_temporal_derived_fields(df)
            df = self._calculate_inventory_derived_fields(df)
            df = self._calculate_product_derived_fields(df)
            
            # 5. Validação final
            df = self._final_data_validation(df)
            
            print(f"✅ Preparação concluída: {len(df)} registros, {len(df.columns)} campos")
            return df
            
        except Exception as e:
            print(f"❌ Erro na preparação de dados: {str(e)}")
            return None
    
    def _validate_data_structure(self, df: pd.DataFrame, validation_level: str) -> bool:
        """Validar estrutura básica dos dados."""
        print("🔍 Validando estrutura dos dados...")
        
        if df is None or len(df) == 0:
            print("❌ DataFrame vazio ou nulo")
            return False
        
        # Campos essenciais por nível
        essential_fields = {
            "basic": ['Data', 'Total_Liquido'],
            "standard": ['Data', 'Total_Liquido', 'Quantidade'],
            "strict": ['Data', 'Total_Liquido', 'Quantidade', 'Codigo_Produto']
        }
        
        # Verificar se há campos de cliente (opcional mas útil para logs)
        customer_fields = ['Codigo_Cliente', 'cliente_id', 'customer_id', 'id_cliente']
        has_customer_field = any(col in df.columns for col in customer_fields)
        if has_customer_field:
            customer_col = next(col for col in customer_fields if col in df.columns)
            print(f"✅ Campo de cliente encontrado: {customer_col}")
        else:
            print("⚠️ Nenhum campo de cliente identificado nos dados")
        
        required_cols = essential_fields.get(validation_level, essential_fields["standard"])
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Campos obrigatórios faltando ({validation_level}): {missing_cols}")
            return False
        
        print(f"✅ Estrutura validada - Nível: {validation_level}")
        return True
    
    def _clean_basic_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Limpeza básica de dados."""
        print("🧹 Executando limpeza básica...")
        
        initial_len = len(df)
        
        # Remover linhas completamente vazias
        df = df.dropna(how='all')
        
        # Remover duplicatas completas
        df = df.drop_duplicates()
        
        # Remover registros com valores negativos críticos
        if 'Total_Liquido' in df.columns:
            df = df[df['Total_Liquido'] > 0]
        
        if 'Quantidade' in df.columns:
            df = df[df['Quantidade'] > 0]
        
        final_len = len(df)
        removed = initial_len - final_len
        
        if removed > 0:
            print(f"🗑️ Removidos {removed} registros inválidos ({removed/initial_len*100:.1f}%)")
        
        if len(df) < 10:
            print("❌ Dados insuficientes após limpeza (< 10 registros)")
            return None
            
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Converter tipos de dados de forma robusta."""
        print("🔄 Convertendo tipos de dados...")
        
        try:
            # Data (essencial)
            df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
            df = df.dropna(subset=['Data'])
            
            # Campos numéricos essenciais
            numeric_fields = ['Total_Liquido', 'Quantidade']
            for field in numeric_fields:
                if field in df.columns:
                    df[field] = pd.to_numeric(df[field], errors='coerce')
            
            # Campos financeiros opcionais
            financial_fields = ['Custo_Produto', 'Preco_Tabela', 'Desconto_Aplicado', 'Estoque_Atual']
            for field in financial_fields:
                if field in df.columns:
                    df[field] = pd.to_numeric(df[field], errors='coerce')
                    print(f"✅ Campo financeiro convertido: {field}")
            
            # Campos demográficos
            if 'Idade' in df.columns:
                df['Idade'] = pd.to_numeric(df['Idade'], errors='coerce')
                print("✅ Campo demográfico convertido: Idade")
            
            # Remover registros com valores essenciais nulos após conversão
            essential_fields = ['Data', 'Total_Liquido']
            df = df.dropna(subset=essential_fields)
            
            return df
            
        except Exception as e:
            print(f"❌ Erro na conversão de tipos: {str(e)}")
            return None
    
    def _calculate_financial_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular campos financeiros derivados."""
        print("💰 Calculando campos financeiros derivados...")
        
        try:
            # Margem real e percentual
            if 'Custo_Produto' in df.columns and 'Total_Liquido' in df.columns:
                df['Margem_Real'] = df['Total_Liquido'] - df['Custo_Produto']
                df['Margem_Percentual'] = (
                    (df['Margem_Real'] / df['Total_Liquido'] * 100)
                    .replace([np.inf, -np.inf], 0)
                    .fillna(0)
                )
                print("✅ Margem real e percentual calculadas")
            
            # Desconto e preço final
            if 'Desconto_Aplicado' in df.columns and 'Preco_Tabela' in df.columns:
                df['Desconto_Percentual'] = (
                    (df['Desconto_Aplicado'] / df['Preco_Tabela'] * 100)
                    .replace([np.inf, -np.inf], 0)
                    .fillna(0)
                )
                df['Preco_Final'] = df['Preco_Tabela'] - df['Desconto_Aplicado']
                print("✅ Desconto percentual e preço final calculados")
            
            # Preço unitário
            if 'Quantidade' in df.columns:
                # Calcular preço unitário com tratamento seguro de divisão por zero
                zero_quantities = (df['Quantidade'] == 0).sum()
                if zero_quantities > 0:
                    print(f"🔧 Ajustando {zero_quantities} registros com quantidade zero")
                
                preco_unitario = df['Total_Liquido'] / df['Quantidade'].replace(0, 1)
                
                # Verificar e tratar valores infinitos
                infinite_values = np.isinf(preco_unitario).sum()
                if infinite_values > 0:
                    print(f"🔧 Tratando {infinite_values} valores infinitos no preço unitário")
                
                # Substituir valores infinitos pela média ou pelo próprio valor total
                preco_unitario = preco_unitario.replace([np.inf, -np.inf], np.nan)
                
                # Para valores NaN/infinitos, usar o próprio Total_Liquido (assumindo quantidade = 1)
                df['Preco_Unitario'] = preco_unitario.fillna(df['Total_Liquido'])
                print("✅ Preço unitário calculado com tratamento robusto")
            
            # ROI se houver custo
            if 'Margem_Real' in df.columns and 'Custo_Produto' in df.columns:
                df['ROI_Percentual'] = (
                    (df['Margem_Real'] / df['Custo_Produto'] * 100)
                    .replace([np.inf, -np.inf], 0)
                    .fillna(0)
                )
                print("✅ ROI percentual calculado")
                
        except Exception as e:
            print(f"⚠️ Erro no cálculo de campos financeiros: {str(e)}")
        
        return df
    
    def _calculate_demographic_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular campos demográficos derivados."""
        print("👥 Calculando campos demográficos derivados...")
        
        try:
            if 'Idade' in df.columns:
                # Faixa etária
                df['Faixa_Etaria'] = pd.cut(
                    df['Idade'], 
                    bins=[0, 25, 35, 45, 55, 100], 
                    labels=['18-25', '26-35', '36-45', '46-55', '55+'],
                    include_lowest=True
                )
                
                # Geração
                df['Geracao'] = pd.cut(
                    df['Idade'],
                    bins=[0, 27, 43, 58, 100],
                    labels=['Gen Z', 'Millennial', 'Gen X', 'Boomer'],
                    include_lowest=True
                )
                
                # Status de maturidade
                df['Maturidade_Compra'] = pd.cut(
                    df['Idade'],
                    bins=[0, 30, 50, 100],
                    labels=['Jovem', 'Adulto', 'Maduro'],
                    include_lowest=True
                )
                
                print("✅ Segmentação etária e geracional calculada")
            
            # Classificação por sexo e estado civil combinados
            if 'Sexo' in df.columns and 'Estado_Civil' in df.columns:
                df['Perfil_Demografico'] = df['Sexo'].astype(str) + '_' + df['Estado_Civil'].astype(str)
                df['Perfil_Demografico'] = df['Perfil_Demografico'].replace({'nan_nan': 'Não_Informado'})
                print("✅ Perfil demográfico combinado calculado")
                
        except Exception as e:
            print(f"⚠️ Erro no cálculo de campos demográficos: {str(e)}")
        
        return df
    
    def _calculate_temporal_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular campos temporais derivados."""
        print("📅 Calculando campos temporais derivados...")
        
        try:
            # Campos básicos
            df['Ano'] = df['Data'].dt.year
            df['Mes'] = df['Data'].dt.month
            df['Trimestre'] = df['Data'].dt.quarter
            df['Dia_Semana'] = df['Data'].dt.dayofweek
            df['Nome_Dia_Semana'] = df['Data'].dt.day_name()
            df['Semana_Ano'] = df['Data'].dt.isocalendar().week
            df['Ano_Mes'] = df['Data'].dt.to_period('M').astype(str)
            
            # Campos avançados
            df['Fim_Semana'] = df['Dia_Semana'].isin([5, 6])  # Sábado e domingo
            df['Dia_Mes'] = df['Data'].dt.day
            df['Quinzena'] = (df['Dia_Mes'] <= 15).map({True: 'Primeira', False: 'Segunda'})
            
            # Sazonalidade de joalherias
            seasonal_months = {
                12: 'Natal', 1: 'Pós-Natal', 2: 'Carnaval', 5: 'Dia_das_Mães',
                6: 'Dia_dos_Namorados', 8: 'Dia_dos_Pais', 10: 'Dia_das_Crianças'
            }
            df['Sazonalidade'] = df['Mes'].map(seasonal_months).fillna('Normal')
            
            print("✅ Campos temporais e sazonalidade calculados")
            
        except Exception as e:
            print(f"⚠️ Erro no cálculo de campos temporais: {str(e)}")
        
        return df
    
    def _calculate_inventory_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular campos de estoque derivados."""
        print("📦 Calculando campos de estoque derivados...")
        
        try:
            if 'Estoque_Atual' in df.columns and 'Quantidade' in df.columns:
                # Turnover de estoque
                df['Turnover_Estoque'] = (
                    df['Quantidade'] / df['Estoque_Atual'].replace(0, 1)
                ).replace([np.inf, -np.inf], 0)
                
                # Dias de estoque (estimativa)
                df['Dias_Estoque'] = (
                    (df['Estoque_Atual'] / df['Quantidade'].replace(0, 1)) * 30
                ).replace([np.inf, -np.inf], 999)
                
                # Classificação de estoque
                df['Status_Estoque'] = pd.cut(
                    df['Dias_Estoque'],
                    bins=[0, 30, 90, 180, 999],
                    labels=['Baixo', 'Normal', 'Alto', 'Excesso'],
                    include_lowest=True
                )
                
                print("✅ Métricas de turnover e status de estoque calculadas")
            
            # Velocity de vendas por produto (se houver código do produto)
            if 'Codigo_Produto' in df.columns:
                product_velocity = df.groupby('Codigo_Produto')['Quantidade'].sum()
                df['Velocity_Produto'] = df['Codigo_Produto'].map(product_velocity)
                print("✅ Velocity de vendas por produto calculada")
                
        except Exception as e:
            print(f"⚠️ Erro no cálculo de campos de estoque: {str(e)}")
        
        return df
    
    def _calculate_product_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular campos de produto derivados."""
        print("💎 Calculando campos de produto derivados...")
        
        try:
            # Categoria de preço
            if 'Preco_Unitario' in df.columns:
                df['Categoria_Preco'] = pd.qcut(
                    df['Preco_Unitario'],
                    q=4,
                    labels=['Econômico', 'Intermediário', 'Premium', 'Luxo'],
                    duplicates='drop'
                )
                print("✅ Categoria de preço calculada")
            
            # Mix de produto (Metal + Grupo)
            if 'Metal' in df.columns and 'Grupo_Produto' in df.columns:
                df['Mix_Produto'] = (
                    df['Metal'].astype(str) + '_' + df['Grupo_Produto'].astype(str)
                ).replace({'nan_nan': 'Não_Classificado'})
                print("✅ Mix de produto calculado")
            
            # Peso do produto no portfólio
            if 'Codigo_Produto' in df.columns:
                product_share = df.groupby('Codigo_Produto')['Total_Liquido'].sum()
                total_revenue = df['Total_Liquido'].sum()
                product_share_pct = (product_share / total_revenue * 100).round(2)
                df['Share_Produto'] = df['Codigo_Produto'].map(product_share_pct)
                print("✅ Share de produto no portfólio calculado")
                
        except Exception as e:
            print(f"⚠️ Erro no cálculo de campos de produto: {str(e)}")
        
        return df
    
    def _final_data_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validação final e ajustes de qualidade."""
        print("🔍 Executando validação final...")
        
        try:
            initial_len = len(df)
            
            # Remover registros com dados essenciais inválidos
            df = df.dropna(subset=['Data', 'Total_Liquido'])
            
            # Validar consistência de datas
            current_date = datetime.now()
            min_date = current_date - timedelta(days=365*5)  # 5 anos atrás
            df = df[(df['Data'] >= min_date) & (df['Data'] <= current_date)]
            
            # Validar consistência financeira básica
            if 'Margem_Real' in df.columns:
                # Remover registros com margens impossíveis (> 100% ou < -50%)
                df = df[(df['Margem_Percentual'] >= -50) & (df['Margem_Percentual'] <= 100)]
            
            final_len = len(df)
            
            if final_len < initial_len:
                removed = initial_len - final_len
                print(f"🔧 Ajustes finais: {removed} registros removidos por inconsistência")
            
            # Preencher campos opcionais que podem estar faltando
            optional_fields = {
                'Codigo_Produto': 'PROD_AUTO',
                'Descricao_Produto': 'Produto Genérico',
                'Grupo_Produto': 'Outros',
                'Metal': 'Não Especificado',
                'Colecao': 'Geral'
            }
            
            for field, default_value in optional_fields.items():
                if field not in df.columns:
                    df[field] = default_value
                else:
                    df[field] = df[field].fillna(default_value)
            
            print("✅ Validação final concluída com sucesso")
            
        except Exception as e:
            print(f"⚠️ Erro na validação final: {str(e)}")
        
        return df
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gerar relatório de qualidade dos dados preparados."""
        if df is None or len(df) == 0:
            return {'error': 'Dados não disponíveis'}
        
        try:
            quality_report = {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'date_range': {
                    'start': df['Data'].min().strftime('%Y-%m-%d'),
                    'end': df['Data'].max().strftime('%Y-%m-%d'),
                    'days': (df['Data'].max() - df['Data'].min()).days
                },
                'completeness': {
                    'financial_fields': sum([
                        'Margem_Real' in df.columns,
                        'Desconto_Percentual' in df.columns,
                        'Preco_Unitario' in df.columns
                    ]),
                    'demographic_fields': sum([
                        'Faixa_Etaria' in df.columns,
                        'Sexo' in df.columns,
                        'Estado_Civil' in df.columns
                    ]),
                    'geographic_fields': sum([
                        'Estado' in df.columns,
                        'Cidade' in df.columns
                    ]),
                    'inventory_fields': sum([
                        'Estoque_Atual' in df.columns,
                        'Turnover_Estoque' in df.columns
                    ])
                },
                'data_quality': {
                    'null_percentage': round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
                    'duplicate_records': df.duplicated().sum(),
                    'unique_products': df['Codigo_Produto'].nunique() if 'Codigo_Produto' in df.columns else 0,
                    'unique_customers': df['Codigo_Cliente'].nunique() if 'Codigo_Cliente' in df.columns else 0
                }
            }
            
            return quality_report
            
        except Exception as e:
            return {'error': f'Erro no relatório de qualidade: {str(e)}'} 