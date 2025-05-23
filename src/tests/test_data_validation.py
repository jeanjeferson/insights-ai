"""
🔍 TESTE: VALIDAÇÃO DE DADOS
============================

Valida a integridade e qualidade dos dados do projeto Insights-AI.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

def test_data_validation(verbose=False, quick=False):
    """
    Teste abrangente de validação de dados
    """
    result = {
        'success': False,
        'details': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        if verbose:
            print("🔍 Iniciando validação de dados...")
        
        # 1. Verificar existência do arquivo principal de dados
        data_path = Path("data/vendas.csv")
        if not data_path.exists():
            result['errors'].append("Arquivo data/vendas.csv não encontrado")
            return result
        
        # 2. Carregar dados
        try:
            df = pd.read_csv(data_path, sep=';', encoding='utf-8')
            if verbose:
                print(f"✅ Arquivo carregado com {len(df)} registros")
        except Exception as e:
            result['errors'].append(f"Erro ao carregar CSV: {str(e)}")
            return result
        
        # 3. Validar estrutura básica
        required_columns = [
            'Data', 'Total_Liquido', 'Quantidade', 'Codigo_Produto'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            result['errors'].append(f"Colunas obrigatórias ausentes: {missing_columns}")
            return result
        
        # 4. Validar tipos de dados
        validation_results = {}
        
        # Data
        try:
            df['Data'] = pd.to_datetime(df['Data'])
            validation_results['data_conversion'] = "OK"
        except:
            validation_results['data_conversion'] = "ERRO"
            result['warnings'].append("Problema na conversão de datas")
        
        # Valores numéricos
        numeric_columns = ['Total_Liquido', 'Quantidade']
        for col in numeric_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        result['warnings'].append(f"Coluna {col} tem {null_count} valores nulos")
                    validation_results[f'{col}_conversion'] = "OK"
                except:
                    validation_results[f'{col}_conversion'] = "ERRO"
                    result['errors'].append(f"Erro na conversão da coluna {col}")
        
        # 5. Análise de qualidade dos dados
        quality_metrics = {}
        
        # Período dos dados
        if 'Data' in df.columns:
            date_range = (df['Data'].min(), df['Data'].max())
            quality_metrics['period'] = f"{date_range[0]} até {date_range[1]}"
            
            # Verificar gaps temporais
            if len(df) > 0:
                date_diff = (date_range[1] - date_range[0]).days
                expected_records = date_diff * 10  # Estimativa de 10 transações por dia
                if len(df) < expected_records * 0.1:
                    result['warnings'].append("Volume de dados pode estar baixo")
        
        # Duplicatas
        duplicates = df.duplicated().sum()
        quality_metrics['duplicates'] = duplicates
        if duplicates > len(df) * 0.01:  # Mais de 1%
            result['warnings'].append(f"Alto número de duplicatas: {duplicates}")
        
        # Outliers em valores
        if 'Total_Liquido' in df.columns:
            q1 = df['Total_Liquido'].quantile(0.25)
            q3 = df['Total_Liquido'].quantile(0.75)
            iqr = q3 - q1
            outliers = len(df[(df['Total_Liquido'] < q1 - 1.5*iqr) | 
                             (df['Total_Liquido'] > q3 + 1.5*iqr)])
            quality_metrics['outliers_count'] = outliers
            quality_metrics['outliers_percentage'] = round(outliers/len(df)*100, 2)
        
        # 6. Validações específicas do negócio
        business_validations = {}
        
        # Valores negativos
        if 'Total_Liquido' in df.columns:
            negative_sales = (df['Total_Liquido'] < 0).sum()
            business_validations['negative_sales'] = negative_sales
            if negative_sales > 0:
                result['warnings'].append(f"Encontradas {negative_sales} vendas com valor negativo")
        
        # Quantidades inválidas
        if 'Quantidade' in df.columns:
            invalid_qty = (df['Quantidade'] <= 0).sum()
            business_validations['invalid_quantities'] = invalid_qty
            if invalid_qty > 0:
                result['warnings'].append(f"Encontradas {invalid_qty} transações com quantidade inválida")
        
        # 7. Análise de completude
        completeness = {}
        for col in df.columns:
            null_pct = df[col].isnull().sum() / len(df) * 100
            completeness[col] = round(100 - null_pct, 2)
        
        # Colunas com muitos nulos
        incomplete_columns = [col for col, pct in completeness.items() if pct < 90]
        if incomplete_columns:
            result['warnings'].append(f"Colunas com baixa completude (<90%): {incomplete_columns}")
        
        # 8. Teste rápido vs completo
        if quick:
            sample_size = min(1000, len(df))
            df_test = df.sample(sample_size) if len(df) > sample_size else df
            if verbose:
                print(f"🏃 Modo rápido: testando amostra de {len(df_test)} registros")
        else:
            df_test = df
            if verbose:
                print(f"🔍 Modo completo: testando todos os {len(df_test)} registros")
        
        # 9. Compilar resultados
        result['details'] = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'validation_results': validation_results,
            'quality_metrics': quality_metrics,
            'business_validations': business_validations,
            'completeness': completeness,
            'columns_list': list(df.columns)
        }
        
        # 10. Determinar sucesso
        if not result['errors']:
            result['success'] = True
            if verbose:
                print("✅ Validação de dados concluída com sucesso")
        else:
            if verbose:
                print(f"❌ Validação falhou: {len(result['errors'])} erros encontrados")
        
        return result
        
    except Exception as e:
        result['errors'].append(f"Erro inesperado na validação: {str(e)}")
        result['success'] = False
        return result

def validate_sample_data(data_df):
    """Validar um DataFrame de amostra"""
    validations = {
        'has_data': len(data_df) > 0,
        'has_required_columns': all(col in data_df.columns for col in ['Data', 'Total_Liquido']),
        'positive_values': (data_df['Total_Liquido'] > 0).all() if 'Total_Liquido' in data_df.columns else False,
        'valid_dates': data_df['Data'].dtype == 'datetime64[ns]' if 'Data' in data_df.columns else False
    }
    
    return all(validations.values()), validations

if __name__ == "__main__":
    # Teste standalone
    result = test_data_validation(verbose=True, quick=False)
    print("\n📊 RESULTADO DA VALIDAÇÃO:")
    print(f"✅ Sucesso: {result['success']}")
    print(f"⚠️  Warnings: {len(result['warnings'])}")
    print(f"❌ Erros: {len(result['errors'])}")
    
    if result['warnings']:
        print("\nWarnings:")
        for warning in result['warnings']:
            print(f"  - {warning}")
    
    if result['errors']:
        print("\nErros:")
        for error in result['errors']:
            print(f"  - {error}")
