"""
🔍 DATA QUALITY VALIDATOR - Validador de Qualidade de Dados
===========================================================

Sistema abrangente para validar qualidade de dados CSV antes dos testes,
incluindo análise estrutural, completude e consistência.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import os
from datetime import datetime


class DataQualityValidator:
    """
    Validador abrangente de qualidade de dados para arquivos CSV.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.quality_report = {}
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """
        Executar validação completa de qualidade dos dados.
        
        Returns:
            Dict contendo métricas detalhadas de qualidade
        """
        try:
            # 1. Validação básica do arquivo
            file_validation = self._validate_file_structure()
            if not file_validation['is_valid']:
                return {
                    'overall_score': 0,
                    'file_validation': file_validation,
                    'error': 'Arquivo inválido ou não encontrado'
                }
            
            # 2. Carregar dados
            self.df = pd.read_csv(self.file_path, sep=';', encoding='utf-8')
            
            # 3. Executar todas as validações
            validations = {
                'file_info': self._get_file_info(),
                'structure': self._validate_structure(),
                'completeness': self._validate_completeness(),
                'consistency': self._validate_consistency(),
                'business_rules': self._validate_business_rules(),
                'data_types': self._validate_data_types()
            }
            
            # 4. Calcular score geral
            overall_score = self._calculate_overall_score(validations)
            
            return {
                'overall_score': overall_score,
                'validation_timestamp': datetime.now().isoformat(),
                **validations
            }
            
        except Exception as e:
            return {
                'overall_score': 0,
                'error': f"Erro na validação: {str(e)}",
                'validation_timestamp': datetime.now().isoformat()
            }
    
    def _validate_file_structure(self) -> Dict[str, Any]:
        """Validar estrutura básica do arquivo."""
        try:
            if not os.path.exists(self.file_path):
                return {
                    'is_valid': False,
                    'error': 'Arquivo não encontrado',
                    'file_exists': False
                }
            
            file_size = os.path.getsize(self.file_path)
            if file_size == 0:
                return {
                    'is_valid': False,
                    'error': 'Arquivo vazio',
                    'file_exists': True,
                    'file_size_bytes': 0
                }
            
            # Tentar carregar primeira linha para validar estrutura
            try:
                sample_df = pd.read_csv(self.file_path, sep=';', encoding='utf-8', nrows=1)
                if len(sample_df.columns) < 5:
                    return {
                        'is_valid': False,
                        'error': 'Poucas colunas detectadas',
                        'columns_found': len(sample_df.columns)
                    }
            except Exception as e:
                return {
                    'is_valid': False,
                    'error': f'Erro ao ler arquivo: {str(e)}'
                }
            
            return {
                'is_valid': True,
                'file_exists': True,
                'file_size_bytes': file_size,
                'file_size_mb': round(file_size / 1024 / 1024, 2)
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'error': f'Erro na validação de estrutura: {str(e)}'
            }
    
    def _get_file_info(self) -> Dict[str, Any]:
        """Obter informações gerais do arquivo."""
        if self.df is None:
            return {'error': 'DataFrame não carregado'}
        
        return {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'file_size_mb': round(os.path.getsize(self.file_path) / 1024 / 1024, 2),
            'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            'columns': list(self.df.columns),
            'date_range': self._get_date_range() if 'Data' in self.df.columns else None
        }
    
    def _validate_structure(self) -> Dict[str, Any]:
        """Validar estrutura e colunas obrigatórias."""
        if self.df is None:
            return {'error': 'DataFrame não carregado'}
        
        # Colunas obrigatórias para análise de vendas
        required_columns = [
            'Data', 'Total_Liquido', 'Codigo_Produto', 'Quantidade'
        ]
        
        optional_important_columns = [
            'Codigo_Cliente', 'Grupo_Produto', 'Metal', 'Preco_Unitario',
            'Custo_Produto', 'Margem_Percentual'
        ]
        
        missing_required = [col for col in required_columns if col not in self.df.columns]
        present_optional = [col for col in optional_important_columns if col in self.df.columns]
        
        # Score baseado em colunas disponíveis
        required_score = (len(required_columns) - len(missing_required)) / len(required_columns) * 100
        optional_score = len(present_optional) / len(optional_important_columns) * 100
        structure_score = (required_score * 0.7) + (optional_score * 0.3)
        
        return {
            'has_required_columns': len(missing_required) == 0,
            'missing_required_columns': missing_required,
            'present_optional_columns': present_optional,
            'structure_score': round(structure_score, 1),
            'column_coverage': f"{len(self.df.columns)}/{len(required_columns + optional_important_columns)}"
        }
    
    def _validate_completeness(self) -> Dict[str, Any]:
        """Validar completude dos dados."""
        if self.df is None:
            return {'error': 'DataFrame não carregado'}
        
        total_cells = len(self.df) * len(self.df.columns)
        missing_cells = self.df.isnull().sum().sum()
        completeness_rate = (1 - missing_cells / total_cells) * 100
        
        # Análise por coluna
        column_completeness = {}
        critical_columns_missing = []
        
        for col in self.df.columns:
            missing_pct = (self.df[col].isnull().sum() / len(self.df)) * 100
            column_completeness[col] = round(100 - missing_pct, 1)
            
            # Verificar colunas críticas
            if col in ['Data', 'Total_Liquido', 'Quantidade'] and missing_pct > 5:
                critical_columns_missing.append(col)
        
        return {
            'overall_completeness_rate': round(completeness_rate, 2),
            'missing_cells_total': int(missing_cells),
            'column_completeness': column_completeness,
            'critical_columns_issues': critical_columns_missing,
            'completeness_score': min(completeness_rate, 100)
        }
    
    def _validate_consistency(self) -> Dict[str, Any]:
        """Validar consistência dos dados."""
        if self.df is None:
            return {'error': 'DataFrame não carregado'}
        
        consistency_issues = []
        consistency_score = 100
        
        # 1. Validar valores negativos em campos que não deveriam ter
        if 'Total_Liquido' in self.df.columns:
            negative_sales = (self.df['Total_Liquido'] < 0).sum()
            if negative_sales > 0:
                consistency_issues.append(f"{negative_sales} vendas com valor negativo")
                consistency_score -= min(10, negative_sales / len(self.df) * 100)
        
        # 2. Validar quantidades
        if 'Quantidade' in self.df.columns:
            zero_qty = (self.df['Quantidade'] <= 0).sum()
            if zero_qty > 0:
                consistency_issues.append(f"{zero_qty} transações com quantidade inválida")
                consistency_score -= min(5, zero_qty / len(self.df) * 100)
        
        # 3. Validar datas
        if 'Data' in self.df.columns:
            try:
                dates = pd.to_datetime(self.df['Data'], format='%Y-%m-%d', errors='coerce')
                invalid_dates = dates.isnull().sum()
                if invalid_dates > 0:
                    consistency_issues.append(f"{invalid_dates} datas inválidas")
                    consistency_score -= min(15, invalid_dates / len(self.df) * 100)
                
                # Verificar datas futuras
                future_dates = (dates > datetime.now()).sum()
                if future_dates > 0:
                    consistency_issues.append(f"{future_dates} datas futuras")
                    consistency_score -= min(5, future_dates / len(self.df) * 100)
                    
            except Exception:
                consistency_issues.append("Erro na validação de datas")
                consistency_score -= 20
        
        # 4. Validar duplicatas
        duplicate_rows = self.df.duplicated().sum()
        if duplicate_rows > 0:
            consistency_issues.append(f"{duplicate_rows} registros duplicados")
            consistency_score -= min(10, duplicate_rows / len(self.df) * 100)
        
        return {
            'consistency_score': max(0, round(consistency_score, 1)),
            'consistency_issues': consistency_issues,
            'issues_count': len(consistency_issues),
            'duplicate_rows': int(duplicate_rows) if 'duplicate_rows' in locals() else 0
        }
    
    def _validate_business_rules(self) -> Dict[str, Any]:
        """Validar regras de negócio específicas."""
        if self.df is None:
            return {'error': 'DataFrame não carregado'}
        
        business_issues = []
        business_score = 100
        
        # 1. Validar margem (se disponível)
        if all(col in self.df.columns for col in ['Total_Liquido', 'Custo_Produto']):
            try:
                calculated_margin = ((self.df['Total_Liquido'] - self.df['Custo_Produto']) / self.df['Total_Liquido'] * 100)
                negative_margin = (calculated_margin < 0).sum()
                extreme_margin = (calculated_margin > 200).sum()
                
                if negative_margin > 0:
                    business_issues.append(f"{negative_margin} produtos com margem negativa")
                    business_score -= min(15, negative_margin / len(self.df) * 100)
                
                if extreme_margin > 0:
                    business_issues.append(f"{extreme_margin} produtos com margem >200%")
                    business_score -= min(5, extreme_margin / len(self.df) * 100)
                    
            except Exception:
                business_issues.append("Erro no cálculo de margem")
                business_score -= 10
        
        # 2. Validar AOV (Average Order Value)
        if 'Total_Liquido' in self.df.columns:
            aov = self.df['Total_Liquido'].mean()
            very_low_sales = (self.df['Total_Liquido'] < 50).sum()
            very_high_sales = (self.df['Total_Liquido'] > 50000).sum()
            
            if very_low_sales > len(self.df) * 0.1:  # >10% de vendas muito baixas
                business_issues.append(f"Muitas vendas de baixo valor: {very_low_sales}")
                business_score -= 5
            
            if very_high_sales > len(self.df) * 0.01:  # >1% de vendas muito altas
                business_issues.append(f"Possíveis outliers de alto valor: {very_high_sales}")
        
        # 3. Validar diversidade de produtos
        if 'Codigo_Produto' in self.df.columns:
            unique_products = self.df['Codigo_Produto'].nunique()
            if unique_products < 10:
                business_issues.append(f"Baixa diversidade de produtos: {unique_products}")
                business_score -= 10
        
        return {
            'business_score': max(0, round(business_score, 1)),
            'business_issues': business_issues,
            'issues_count': len(business_issues)
        }
    
    def _validate_data_types(self) -> Dict[str, Any]:
        """Validar tipos de dados."""
        if self.df is None:
            return {'error': 'DataFrame não carregado'}
        
        type_issues = []
        type_score = 100
        
        expected_types = {
            'Total_Liquido': ['float64', 'int64'],
            'Quantidade': ['int64', 'float64'],
            'Data': ['object'],  # Will be converted to datetime
            'Codigo_Cliente': ['object'],
            'Codigo_Produto': ['object']
        }
        
        for col, expected in expected_types.items():
            if col in self.df.columns:
                actual_type = str(self.df[col].dtype)
                if actual_type not in expected:
                    type_issues.append(f"{col}: esperado {expected}, encontrado {actual_type}")
                    type_score -= 10
        
        return {
            'type_score': max(0, round(type_score, 1)),
            'type_issues': type_issues,
            'issues_count': len(type_issues),
            'column_types': {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        }
    
    def _get_date_range(self) -> Optional[Dict[str, str]]:
        """Obter range de datas dos dados."""
        try:
            if 'Data' in self.df.columns:
                dates = pd.to_datetime(self.df['Data'], format='%Y-%m-%d', errors='coerce')
                dates = dates.dropna()
                if len(dates) > 0:
                    return {
                        'start_date': dates.min().strftime('%Y-%m-%d'),
                        'end_date': dates.max().strftime('%Y-%m-%d'),
                        'total_days': (dates.max() - dates.min()).days + 1,
                        'valid_dates': len(dates)
                    }
        except Exception:
            pass
        return None
    
    def _calculate_overall_score(self, validations: Dict[str, Any]) -> float:
        """Calcular score geral de qualidade."""
        scores = []
        weights = []
        
        # Peso das diferentes validações
        score_mapping = [
            ('structure', 'structure_score', 0.25),
            ('completeness', 'completeness_score', 0.25),
            ('consistency', 'consistency_score', 0.25),
            ('business_rules', 'business_score', 0.15),
            ('data_types', 'type_score', 0.10)
        ]
        
        for validation_key, score_key, weight in score_mapping:
            if validation_key in validations and score_key in validations[validation_key]:
                score = validations[validation_key][score_key]
                if isinstance(score, (int, float)) and score >= 0:
                    scores.append(score)
                    weights.append(weight)
        
        if scores:
            weighted_score = sum(score * weight for score, weight in zip(scores, weights))
            total_weight = sum(weights)
            return round(weighted_score / total_weight, 1)
        
        return 0.0


def validate_test_data(file_path: str) -> Dict[str, Any]:
    """
    Função utilitária para validar dados de teste rapidamente.
    
    Args:
        file_path: Caminho para o arquivo CSV
        
    Returns:
        Dict com resumo da qualidade dos dados
    """
    validator = DataQualityValidator(file_path)
    quality_report = validator.validate_data_quality()
    
    # Retornar resumo simplificado
    return {
        'overall_score': quality_report.get('overall_score', 0),
        'is_valid_for_testing': quality_report.get('overall_score', 0) >= 70,
        'total_rows': quality_report.get('file_info', {}).get('total_rows', 0),
        'main_issues': _extract_main_issues(quality_report),
        'full_report': quality_report
    }


def _extract_main_issues(quality_report: Dict[str, Any]) -> List[str]:
    """Extrair principais problemas do relatório de qualidade."""
    issues = []
    
    for section in ['structure', 'completeness', 'consistency', 'business_rules']:
        if section in quality_report:
            section_data = quality_report[section]
            
            if 'missing_required_columns' in section_data:
                missing = section_data['missing_required_columns']
                if missing:
                    issues.extend([f"Coluna obrigatória ausente: {col}" for col in missing])
            
            if 'consistency_issues' in section_data:
                issues.extend(section_data['consistency_issues'])
            
            if 'business_issues' in section_data:
                issues.extend(section_data['business_issues'])
    
    return issues[:5]  # Top 5 issues 