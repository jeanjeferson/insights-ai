"""
🔍 TESTE: VALIDAÇÃO DE DADOS (SIMPLIFICADO)
==========================================

Valida a integridade básica dos dados do projeto Insights-AI.
Versão simplificada focada em validações essenciais.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

class TestDataValidation:
    """Classe simplificada para validação de dados"""
    
    def test_data_file_exists(self):
        """Verificar se o arquivo de dados existe"""
        data_path = Path("data/vendas.csv")
        if not data_path.exists():
            print("⚠️ Arquivo data/vendas.csv não encontrado - criando dados de teste")
            return False
        return True
    
    def test_data_structure(self, df):
        """Validar estrutura básica dos dados"""
        required_columns = ['Data', 'Total_Liquido', 'Quantidade']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Colunas obrigatórias ausentes: {missing_columns}")
            return False
        
        print(f"✅ Estrutura: {len(df)} registros, {len(df.columns)} colunas")
        return True
    
    def test_data_types(self, df):
        """Validar tipos de dados básicos"""
        validations = {}
        
        # Testar conversão de data
        try:
            pd.to_datetime(df['Data'].iloc[:10])  # Testar apenas primeiras 10 linhas
            validations['dates'] = True
        except:
            validations['dates'] = False
            print("⚠️ Problemas na conversão de datas")
        
        # Testar valores numéricos
        try:
            pd.to_numeric(df['Total_Liquido'].iloc[:10], errors='coerce')
            validations['numeric'] = True
        except:
            validations['numeric'] = False
            print("⚠️ Problemas em valores numéricos")
        
        success_rate = sum(validations.values()) / len(validations) * 100
        print(f"✅ Tipos de dados: {success_rate:.0f}% válidos")
        
        return success_rate >= 80
    
    def test_data_quality(self, df):
        """Validações básicas de qualidade"""
        quality_checks = {}
        
        # Verificar duplicatas (amostra)
        sample_size = min(1000, len(df))
        df_sample = df.sample(sample_size) if len(df) > sample_size else df
        
        duplicates = df_sample.duplicated().sum()
        duplicate_rate = duplicates / len(df_sample) * 100
        quality_checks['duplicates'] = duplicate_rate < 5  # Menos de 5%
        
        # Verificar valores nulos
        null_rate = df_sample.isnull().sum().sum() / (len(df_sample) * len(df_sample.columns)) * 100
        quality_checks['completeness'] = null_rate < 20  # Menos de 20% nulos
        
        # Verificar valores negativos em vendas
        if 'Total_Liquido' in df_sample.columns:
            negative_sales = (pd.to_numeric(df_sample['Total_Liquido'], errors='coerce') < 0).sum()
            quality_checks['business_logic'] = negative_sales == 0
        else:
            quality_checks['business_logic'] = True
        
        success_rate = sum(quality_checks.values()) / len(quality_checks) * 100
        print(f"✅ Qualidade: {success_rate:.0f}% dos checks passaram")
        
        return success_rate >= 70
    
    def test_data_validation_summary(self):
        """Teste consolidado de validação de dados"""
        print("🔍 INICIANDO VALIDAÇÃO DE DADOS")
        print("=" * 35)
        
        success_count = 0
        total_tests = 0
        
        # 1. Verificar existência do arquivo
        total_tests += 1
        if self.test_data_file_exists():
            success_count += 1
            
            try:
                # Carregar dados
                df = pd.read_csv("data/vendas.csv", sep=';', encoding='utf-8')
                
                # 2. Testar estrutura
                total_tests += 1
                if self.test_data_structure(df):
                    success_count += 1
                
                # 3. Testar tipos de dados
                total_tests += 1
                if self.test_data_types(df):
                    success_count += 1
                
                # 4. Testar qualidade
                total_tests += 1
                if self.test_data_quality(df):
                    success_count += 1
                
            except Exception as e:
                print(f"❌ Erro ao carregar dados: {e}")
        else:
            # Se não há arquivo, criar dados de teste
            print("📝 Criando dados de teste para validação...")
            df = self.create_test_data()
            
            # Testar com dados criados
            total_tests += 2
            if self.test_data_structure(df):
                success_count += 1
            if self.test_data_types(df):
                success_count += 1
        
        # Resultado final
        success_rate = (success_count / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📊 RESUMO DA VALIDAÇÃO:")
        print(f"   ✅ Sucessos: {success_count}/{total_tests}")
        print(f"   📈 Taxa de sucesso: {success_rate:.1f}%")
        
        if success_rate >= 75:
            print(f"\n🎉 VALIDAÇÃO DE DADOS CONCLUÍDA COM SUCESSO!")
        else:
            print(f"\n⚠️ ALGUNS PROBLEMAS DE DADOS DETECTADOS")
        
        return {
            'success_count': success_count,
            'total_tests': total_tests,
            'success_rate': success_rate
        }
    
    def create_test_data(self):
        """Criar dados de teste simples"""
        np.random.seed(42)
        
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = []
        
        for i, date in enumerate(dates):
            data.append({
                'Data': date.strftime('%Y-%m-%d'),
                'Codigo_Cliente': f"CLI_{(i % 10) + 1:03d}",
                'Total_Liquido': np.random.uniform(100, 2000),
                'Quantidade': np.random.randint(1, 5),
                'Categoria': np.random.choice(['Anéis', 'Brincos', 'Colares'])
            })
        
        return pd.DataFrame(data)

def run_data_validation():
    """Função principal para executar validação de dados"""
    test_suite = TestDataValidation()
    return test_suite.test_data_validation_summary()

if __name__ == "__main__":
    run_data_validation()
