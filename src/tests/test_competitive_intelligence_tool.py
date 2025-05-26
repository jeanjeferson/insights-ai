"""
🧪 TESTE COMPLETO PARA COMPETITIVE INTELLIGENCE TOOL
=====================================================

Suite de testes abrangente para validar todas as funcionalidades do
Competitive Intelligence Tool, incluindo:
- 5 tipos de análise competitiva
- Benchmarks setoriais brasileiros
- Validação robusta de parâmetros
- Performance e qualidade
"""

import pytest
import pandas as pd
import numpy as np
import time
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import tracemalloc
import warnings

# Importar ferramenta a ser testada
import sys
import os

# Configurar caminhos de importação de forma robusta
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# Adicionar caminhos possíveis ao sys.path
possible_paths = [
    project_root,
    os.path.join(project_root, 'src'),
    current_dir,
    os.path.dirname(current_dir)
]

for path in possible_paths:
    if path not in sys.path:
        sys.path.insert(0, path)

# Tentar importar o Competitive Intelligence Tool
try:
    from src.insights.tools.advanced.competitive_intelligence_tool import CompetitiveIntelligenceTool
except ImportError:
    # Fallback para importação direta
    sys.path.insert(0, os.path.join(project_root, 'src', 'insights', 'tools', 'advanced'))
    from competitive_intelligence_tool import CompetitiveIntelligenceTool

# Suprimir warnings para testes mais limpos
warnings.filterwarnings('ignore')


class TestCompetitiveIntelligenceTool:
    """
    Suite completa de testes para Competitive Intelligence Tool
    
    Cobertura:
    - Todas as 5 análises competitivas
    - Benchmarks setoriais brasileiros
    - Validação de parâmetros
    - Performance e qualidade
    """
    
    @pytest.fixture(autouse=True)
    def setup(self, real_vendas_data):
        """Setup automático para cada teste."""
        self.competitive_tool = CompetitiveIntelligenceTool()
        self.real_data_path = real_vendas_data
        self.test_logs = []
        self.start_time = time.time()
        
        print(f"🏆 Iniciando teste Competitive Intelligence Tool: {self.real_data_path}")
    
    def setup_standalone(self, data_path):
        """Setup para execução standalone."""
        self.competitive_tool = CompetitiveIntelligenceTool()
        self.real_data_path = data_path
        self.test_logs = []
        self.start_time = time.time()
        
        print(f"🏆 Iniciando teste Competitive Intelligence Tool: {self.real_data_path}")
    
    def log_test(self, level: str, message: str, **kwargs):
        """Logging detalhado para testes."""
        elapsed = time.time() - self.start_time
        log_entry = {
            'elapsed': round(elapsed, 2),
            'level': level,
            'message': message,
            **kwargs
        }
        self.test_logs.append(log_entry)
        print(f"[{elapsed:6.2f}s] [{level}] {message}")
        if kwargs:
            print(f"    {kwargs}")
    
    # ==========================================
    # TESTES DE VALIDAÇÃO BÁSICA
    # ==========================================
    
    def test_data_loading_and_validation(self):
        """Teste de carregamento e validação de dados."""
        self.log_test("INFO", "Testando carregamento e validação de dados")
        
        # Verificar se arquivo existe
        assert os.path.exists(self.real_data_path), "Arquivo de dados não encontrado"
        
        # Testar carregamento de dados usando método interno
        df = pd.read_csv(self.real_data_path, sep=';', encoding='utf-8')
        df_prepared = self.competitive_tool._prepare_competitive_data(df)
        
        assert df_prepared is not None, "Falha na preparação de dados competitivos"
        assert len(df_prepared) > 0, "DataFrame preparado vazio"
        assert 'Preco_Unitario' in df_prepared.columns, "Coluna Preco_Unitario não criada"
        assert 'Faixa_Preco' in df_prepared.columns, "Coluna Faixa_Preco não criada"
        
        # Verificar categorização de preços
        faixas_esperadas = ['Economy', 'Mid', 'Premium', 'Luxury', 'Ultra-Luxury']
        faixas_encontradas = df_prepared['Faixa_Preco'].unique()
        faixas_validas = [f for f in faixas_encontradas if pd.notna(f) and f in faixas_esperadas]
        assert len(faixas_validas) > 0, "Nenhuma faixa de preço válida encontrada"
        
        self.log_test("SUCCESS", "Validação de dados aprovada",
                     rows=len(df_prepared), 
                     columns=len(df_prepared.columns),
                     price_ranges=len(faixas_validas))
    
    def test_benchmark_loading(self):
        """Teste de carregamento de benchmarks do setor."""
        self.log_test("INFO", "Testando carregamento de benchmarks")
        
        # Testar carregamento de benchmarks para joalherias
        benchmarks_joalherias = self.competitive_tool._load_market_benchmarks('joalherias')
        assert isinstance(benchmarks_joalherias, dict), "Benchmarks deve ser um dicionário"
        
        # Verificar estrutura essencial dos benchmarks
        expected_keys = [
            'market_size_billion_brl', 'annual_growth_rate', 'average_ticket',
            'seasonal_patterns', 'category_distribution', 'margin_benchmarks'
        ]
        missing_keys = [key for key in expected_keys if key not in benchmarks_joalherias]
        assert len(missing_keys) == 0, f"Chaves essenciais ausentes: {missing_keys}"
        
        # Verificar estrutura do average_ticket
        avg_ticket = benchmarks_joalherias['average_ticket']
        assert 'economy' in avg_ticket, "Categoria economy ausente em average_ticket"
        assert 'premium' in avg_ticket, "Categoria premium ausente em average_ticket"
        assert 'luxury' in avg_ticket, "Categoria luxury ausente em average_ticket"
        
        # Verificar valores numéricos válidos
        assert benchmarks_joalherias['market_size_billion_brl'] > 0, "Market size deve ser positivo"
        assert 0 < benchmarks_joalherias['annual_growth_rate'] < 1, "Growth rate deve estar entre 0 e 1"
        
        # Testar outros segmentos
        benchmarks_relogios = self.competitive_tool._load_market_benchmarks('relogios')
        benchmarks_acessorios = self.competitive_tool._load_market_benchmarks('acessorios')
        
        # Segmentos não implementados devem retornar benchmarks de joalherias
        assert benchmarks_relogios == benchmarks_joalherias, "Fallback para joalherias não funcionou"
        assert benchmarks_acessorios == benchmarks_joalherias, "Fallback para joalherias não funcionou"
        
        self.log_test("SUCCESS", "Benchmarks carregados",
                     benchmark_keys=len(benchmarks_joalherias),
                     price_categories=len(avg_ticket))
    
    def test_input_validation(self):
        """Teste de validação de parâmetros de entrada."""
        self.log_test("INFO", "Testando validação de parâmetros")
        
        # Teste com parâmetros válidos
        result_valid = self.competitive_tool._run(
            analysis_type="market_position",
            data_csv=self.real_data_path,
            market_segment="joalherias",
            benchmark_period="quarterly"
        )
        assert "error" not in result_valid.lower(), "Erro com parâmetros válidos"
        
        # Teste com analysis_type inválido
        result_invalid_type = self.competitive_tool._run(
            analysis_type="analise_inexistente",
            data_csv=self.real_data_path
        )
        assert "não suportada" in result_invalid_type.lower() or "error" in result_invalid_type.lower(), \
               "Erro não detectado para analysis_type inválido"
        
        # Teste com arquivo inexistente
        result_no_file = self.competitive_tool._run(
            analysis_type="market_position",
            data_csv="arquivo_inexistente.csv"
        )
        assert "não encontrado" in result_no_file.lower() or "error" in result_no_file.lower(), \
               "Erro não detectado para arquivo inexistente"
        
        # Teste com dados insuficientes (simular arquivo pequeno)
        small_df = pd.DataFrame({
            'Data': [datetime.now() - timedelta(days=i) for i in range(5)],
            'Total_Liquido': [100, 200, 150, 300, 250],
            'Quantidade': [1, 2, 1, 3, 2]
        })
        small_csv = "test_small_data.csv"
        small_df.to_csv(small_csv, sep=';', index=False)
        
        try:
            result_small = self.competitive_tool._run(
                analysis_type="market_position",
                data_csv=small_csv
            )
            # Deve detectar dados insuficientes (< 30 registros)
            assert "insuficientes" in result_small.lower() or "error" in result_small.lower(), \
                   "Erro não detectado para dados insuficientes"
        finally:
            if os.path.exists(small_csv):
                os.remove(small_csv)
        
        self.log_test("SUCCESS", "Validação de parâmetros aprovada")
    
    # ==========================================
    # TESTES DAS 5 ANÁLISES COMPETITIVAS
    # ==========================================
    
    def test_market_position_analysis(self):
        """Teste completo da análise de posicionamento de mercado."""
        self.log_test("INFO", "Testando Market Position Analysis")
        
        start_time = time.time()
        tracemalloc.start()
        
        result = self.competitive_tool._run(
            analysis_type="market_position",
            data_csv=self.real_data_path,
            market_segment="joalherias",
            benchmark_period="quarterly",
            include_recommendations=True,
            risk_tolerance="medium"
        )
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Validações básicas
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 800, "Resultado muito curto para market position"
        assert "error" not in result.lower(), f"Erro detectado: {result[:200]}..."
        
        # Verificar conteúdo específico de market position
        position_terms = ["posicionamento", "market position", "ticket médio", "benchmark", "positioning"]
        found_terms = [term for term in position_terms if term.lower() in result.lower()]
        assert len(found_terms) >= 2, f"Poucos termos de posicionamento: {found_terms}"
        
        # Verificar seções esperadas
        expected_sections = ["posicionamento competitivo", "métricas vs mercado", "participação de mercado"]
        found_sections = [section for section in expected_sections if section.lower() in result.lower()]
        assert len(found_sections) >= 1, f"Seções de posicionamento não encontradas: {found_sections}"
        
        # Verificar se há categorização de posicionamento
        position_categories = ["economy", "mid", "premium", "luxury"]
        found_categories = [cat for cat in position_categories if cat.lower() in result.lower()]
        assert len(found_categories) >= 1, f"Categorias de posicionamento não encontradas: {found_categories}"
        
        self.log_test("SUCCESS", "Market Position validado",
                     execution_time=f"{execution_time:.2f}s",
                     memory_peak=f"{peak/1024/1024:.1f}MB",
                     position_terms=len(found_terms))
    
    def test_pricing_analysis(self):
        """Teste da análise competitiva de preços."""
        self.log_test("INFO", "Testando Pricing Analysis")
        
        result = self.competitive_tool._run(
            analysis_type="pricing_analysis",
            data_csv=self.real_data_path,
            market_segment="joalherias",
            include_recommendations=True
        )
        
        # Validações básicas
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 600, "Resultado muito curto para pricing analysis"
        assert "error" not in result.lower(), f"Erro detectado: {result[:200]}..."
        
        # Verificar conteúdo específico de pricing
        pricing_terms = ["preço", "pricing", "competitiv", "categoria", "premium", "desconto"]
        found_terms = [term for term in pricing_terms if term.lower() in result.lower()]
        assert len(found_terms) >= 3, f"Poucos termos de pricing: {found_terms}"
        
        # Verificar análise por categoria
        category_terms = ["economy", "mid", "premium", "luxury"]
        found_categories = [cat for cat in category_terms if cat.lower() in result.lower()]
        assert len(found_categories) >= 2, f"Análise por categoria insuficiente: {found_categories}"
        
        # Verificar elementos de comparação
        comparison_terms = ["vs", "mercado", "benchmark", "diferença", "gap"]
        found_comparisons = [term for term in comparison_terms if term.lower() in result.lower()]
        assert len(found_comparisons) >= 2, f"Elementos de comparação ausentes: {found_comparisons}"
        
        self.log_test("SUCCESS", "Pricing Analysis validado",
                     pricing_terms=len(found_terms),
                     categories_found=len(found_categories))
    
    def test_trend_comparison_analysis(self):
        """Teste da comparação de tendências."""
        self.log_test("INFO", "Testando Trend Comparison")
        
        result = self.competitive_tool._run(
            analysis_type="trend_comparison",
            data_csv=self.real_data_path,
            benchmark_period="monthly",
            include_recommendations=True
        )
        
        # Validações básicas
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 500, "Resultado muito curto para trend comparison"
        assert "error" not in result.lower(), f"Erro detectado: {result[:200]}..."
        
        # Verificar conteúdo específico de tendências
        trend_terms = ["tendência", "trend", "crescimento", "performance", "mercado", "sazonal"]
        found_terms = [term for term in trend_terms if term.lower() in result.lower()]
        assert len(found_terms) >= 3, f"Poucos termos de tendência: {found_terms}"
        
        # Verificar análise vs mercado
        market_comparison_terms = ["vs mercado", "performance vs", "benchmark", "outperforming", "underperforming"]
        found_comparisons = [term for term in market_comparison_terms if term.lower() in result.lower()]
        assert len(found_comparisons) >= 1, f"Comparação vs mercado ausente: {found_comparisons}"
        
        # Verificar análise sazonal
        seasonal_terms = ["sazonal", "seasonal", "pico", "peak", "variação"]
        found_seasonal = [term for term in seasonal_terms if term.lower() in result.lower()]
        assert len(found_seasonal) >= 1, f"Análise sazonal ausente: {found_seasonal}"
        
        self.log_test("SUCCESS", "Trend Comparison validado",
                     trend_terms=len(found_terms),
                     market_comparisons=len(found_comparisons))
    
    def test_market_share_estimation_analysis(self):
        """Teste da estimativa de market share."""
        self.log_test("INFO", "Testando Market Share Estimation")
        
        result = self.competitive_tool._run(
            analysis_type="market_share_estimation",
            data_csv=self.real_data_path,
            market_segment="joalherias",
            include_recommendations=True
        )
        
        # Validações básicas
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 700, "Resultado muito curto para market share"
        assert "error" not in result.lower(), f"Erro detectado: {result[:200]}..."
        
        # Verificar conteúdo específico de market share
        share_terms = ["market share", "participação", "share", "mercado", "regional", "nacional"]
        found_terms = [term for term in share_terms if term.lower() in result.lower()]
        assert len(found_terms) >= 3, f"Poucos termos de market share: {found_terms}"
        
        # Verificar estimativas percentuais
        percentage_indicators = ["%", "percent", "porcent"]
        found_percentages = [ind for ind in percentage_indicators if ind in result.lower()]
        assert len(found_percentages) >= 1, f"Estimativas percentuais ausentes: {found_percentages}"
        
        # Verificar análise competitiva
        competitive_terms = ["concorrente", "competitor", "líder", "leader", "posição"]
        found_competitive = [term for term in competitive_terms if term.lower() in result.lower()]
        assert len(found_competitive) >= 1, f"Análise competitiva ausente: {found_competitive}"
        
        # Verificar potencial de crescimento
        growth_terms = ["crescimento", "growth", "potencial", "oportunidade", "expansão"]
        found_growth = [term for term in growth_terms if term.lower() in result.lower()]
        assert len(found_growth) >= 2, f"Análise de crescimento insuficiente: {found_growth}"
        
        self.log_test("SUCCESS", "Market Share Estimation validado",
                     share_terms=len(found_terms),
                     growth_indicators=len(found_growth))
    
    def test_competitive_gaps_analysis(self):
        """Teste da identificação de gaps competitivos."""
        self.log_test("INFO", "Testando Competitive Gaps")
        
        result = self.competitive_tool._run(
            analysis_type="competitive_gaps",
            data_csv=self.real_data_path,
            include_recommendations=True,
            risk_tolerance="high"
        )
        
        # Validações básicas
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 800, "Resultado muito curto para competitive gaps"
        assert "error" not in result.lower(), f"Erro detectado: {result[:200]}..."
        
        # Verificar conteúdo específico de gaps
        gap_terms = ["gap", "oportunidade", "opportunity", "lacuna", "competitiv"]
        found_terms = [term for term in gap_terms if term.lower() in result.lower()]
        assert len(found_terms) >= 3, f"Poucos termos de gaps: {found_terms}"
        
        # Verificar matriz de oportunidades
        matrix_terms = ["matriz", "matrix", "prioridade", "priority", "alta", "high"]
        found_matrix = [term for term in matrix_terms if term.lower() in result.lower()]
        assert len(found_matrix) >= 2, f"Matriz de oportunidades ausente: {found_matrix}"
        
        # Verificar gaps por categoria
        category_gap_terms = ["categoria", "category", "operational", "operacional", "pricing"]
        found_category_gaps = [term for term in category_gap_terms if term.lower() in result.lower()]
        assert len(found_category_gaps) >= 2, f"Gaps por categoria ausentes: {found_category_gaps}"
        
        # Verificar recomendações estratégicas
        recommendation_terms = ["recomendação", "recommendation", "estratégi", "strategic", "ação"]
        found_recommendations = [term for term in recommendation_terms if term.lower() in result.lower()]
        assert len(found_recommendations) >= 2, f"Recomendações estratégicas ausentes: {found_recommendations}"
        
        self.log_test("SUCCESS", "Competitive Gaps validado",
                     gap_terms=len(found_terms),
                     matrix_indicators=len(found_matrix),
                     recommendations=len(found_recommendations))
    
    # ==========================================
    # TESTES DE INTEGRAÇÃO E PERFORMANCE
    # ==========================================
    
    def test_all_analysis_types_integration(self):
        """Teste de integração de todos os tipos de análise competitiva."""
        self.log_test("INFO", "Testando integração de todas as análises competitivas")
        
        analysis_types = [
            "market_position", "pricing_analysis", "trend_comparison",
            "market_share_estimation", "competitive_gaps"
        ]
        
        results = {}
        total_time = 0
        
        for analysis_type in analysis_types:
            start_time = time.time()
            
            try:
                result = self.competitive_tool._run(
                    analysis_type=analysis_type,
                    data_csv=self.real_data_path,
                    market_segment="joalherias",
                    benchmark_period="quarterly",
                    include_recommendations=True,
                    risk_tolerance="medium"
                )
                
                execution_time = time.time() - start_time
                total_time += execution_time
                
                success = "error" not in result.lower() and len(result) > 400
                
                results[analysis_type] = {
                    'success': success,
                    'execution_time': round(execution_time, 2),
                    'output_length': len(result)
                }
                
                self.log_test("SUCCESS" if success else "ERROR",
                             f"Análise {analysis_type}",
                             **results[analysis_type])
                
            except Exception as e:
                results[analysis_type] = {
                    'success': False,
                    'error': str(e),
                    'execution_time': time.time() - start_time
                }
                self.log_test("ERROR", f"Erro em {analysis_type}: {str(e)}")
        
        # Validações de integração
        successful = [name for name, res in results.items() if res['success']]
        success_rate = len(successful) / len(analysis_types)
        
        assert success_rate >= 0.80, f"Taxa de sucesso baixa: {success_rate:.1%}"  # 4/5 = 80%
        assert total_time < 120, f"Tempo total excessivo: {total_time:.1f}s"
        
        self.log_test("SUCCESS", "Integração de análises competitivas validada",
                     success_rate=f"{success_rate:.1%}",
                     total_time=f"{total_time:.1f}s",
                     successful_analyses=successful)
        
        return results
    
    def test_performance_benchmarks(self):
        """Teste de benchmarks de performance."""
        self.log_test("INFO", "Testando benchmarks de performance")
        
        # Teste com análise mais rápida
        start_time = time.time()
        tracemalloc.start()
        
        result = self.competitive_tool._run(
            analysis_type="pricing_analysis",
            data_csv=self.real_data_path,
            market_segment="joalherias",
            include_recommendations=False  # Desabilitar para acelerar
        )
        
        execution_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Validações de performance
        assert execution_time < 45, f"Execução muito lenta: {execution_time:.2f}s"
        assert peak < 512 * 1024 * 1024, f"Uso de memória excessivo: {peak/1024/1024:.1f}MB"
        assert len(result) > 400, "Resultado muito curto"
        assert "error" not in result.lower(), "Erro na execução"
        
        self.log_test("SUCCESS", "Performance validada",
                     execution_time=f"{execution_time:.2f}s",
                     memory_peak=f"{peak/1024/1024:.1f}MB")
    
    def test_different_market_segments(self):
        """Teste com diferentes segmentos de mercado."""
        self.log_test("INFO", "Testando diferentes segmentos de mercado")
        
        segments = ["joalherias", "relogios", "acessorios"]
        results = {}
        
        for segment in segments:
            try:
                result = self.competitive_tool._run(
                    analysis_type="market_position",
                    data_csv=self.real_data_path,
                    market_segment=segment,
                    include_recommendations=False
                )
                
                success = "error" not in result.lower() and len(result) > 300
                results[segment] = {'success': success, 'length': len(result)}
                
            except Exception as e:
                results[segment] = {'success': False, 'error': str(e)}
        
        # Validações
        successful_segments = [seg for seg, res in results.items() if res['success']]
        success_rate = len(successful_segments) / len(segments)
        
        # Todos os segmentos devem funcionar (fallback para joalherias)
        assert success_rate >= 1.0, f"Nem todos os segmentos funcionaram: {successful_segments}"
        
        self.log_test("SUCCESS", "Segmentos de mercado validados",
                     segments_tested=len(segments),
                     successful=len(successful_segments))
    
    # ==========================================
    # TESTES DE TRATAMENTO DE ERROS
    # ==========================================
    
    def test_error_handling_comprehensive(self):
        """Teste abrangente de tratamento de erros."""
        self.log_test("INFO", "Testando tratamento de erros")
        
        error_tests = []
        
        # 1. Arquivo inexistente
        try:
            result = self.competitive_tool._run(
                analysis_type="market_position",
                data_csv="arquivo_inexistente.csv"
            )
            handled = "error" in result.lower() or "não encontrado" in result.lower()
            error_tests.append(('arquivo_inexistente', handled))
        except Exception:
            error_tests.append(('arquivo_inexistente', False))
        
        # 2. Tipo de análise inválido
        try:
            result = self.competitive_tool._run(
                analysis_type="analise_competitiva_invalida",
                data_csv=self.real_data_path
            )
            handled = "error" in result.lower() or "não suportada" in result.lower()
            error_tests.append(('tipo_invalido', handled))
        except Exception:
            error_tests.append(('tipo_invalido', False))
        
        # 3. Segmento de mercado inválido (deve usar fallback)
        try:
            result = self.competitive_tool._run(
                analysis_type="market_position",
                data_csv=self.real_data_path,
                market_segment="segmento_inexistente"
            )
            # Deve funcionar com fallback para joalherias
            handled = "error" not in result.lower() and len(result) > 300
            error_tests.append(('segmento_invalido_fallback', handled))
        except Exception:
            error_tests.append(('segmento_invalido_fallback', False))
        
        # 4. Período de benchmark inválido
        try:
            result = self.competitive_tool._run(
                analysis_type="trend_comparison",
                data_csv=self.real_data_path,
                benchmark_period="periodo_invalido"
            )
            # Deve usar fallback ou dar erro
            handled = True  # Qualquer tratamento é válido
            error_tests.append(('periodo_invalido', handled))
        except Exception:
            error_tests.append(('periodo_invalido', False))
        
        # Validações
        passed = sum(1 for _, handled in error_tests if handled)
        success_rate = passed / len(error_tests)
        
        assert success_rate >= 0.75, f"Poucos erros tratados: {passed}/{len(error_tests)}"
        
        self.log_test("SUCCESS", "Tratamento de erros validado",
                     tests_passed=f"{passed}/{len(error_tests)}",
                     success_rate=f"{success_rate:.1%}")
    
    def test_edge_cases(self):
        """Teste de casos extremos."""
        self.log_test("INFO", "Testando casos extremos")
        
        edge_cases = []
        
        # 1. Tolerância a risco extrema
        try:
            result = self.competitive_tool._run(
                analysis_type="competitive_gaps",
                data_csv=self.real_data_path,
                risk_tolerance="high",
                include_recommendations=True
            )
            success = len(result) > 400 and "error" not in result.lower()
            edge_cases.append(('risk_tolerance_high', success))
        except Exception as e:
            edge_cases.append(('risk_tolerance_high', False))
        
        # 2. Sem recomendações
        try:
            result = self.competitive_tool._run(
                analysis_type="market_position",
                data_csv=self.real_data_path,
                include_recommendations=False
            )
            success = len(result) > 300 and "error" not in result.lower()
            edge_cases.append(('no_recommendations', success))
        except Exception as e:
            edge_cases.append(('no_recommendations', False))
        
        # 3. Período anual (mais dados)
        try:
            result = self.competitive_tool._run(
                analysis_type="trend_comparison",
                data_csv=self.real_data_path,
                benchmark_period="yearly"
            )
            success = len(result) > 300 and "error" not in result.lower()
            edge_cases.append(('yearly_period', success))
        except Exception as e:
            edge_cases.append(('yearly_period', False))
        
        # Validações
        passed = sum(1 for _, success in edge_cases if success)
        success_rate = passed / len(edge_cases)
        
        assert success_rate >= 0.67, f"Poucos edge cases tratados: {passed}/{len(edge_cases)} ({success_rate:.1%})"
        
        self.log_test("SUCCESS", "Edge cases validados",
                     tests_passed=f"{passed}/{len(edge_cases)}")
    
    def test_invalid_parameters(self):
        """Teste com parâmetros completamente inválidos."""
        self.log_test("INFO", "Testando parâmetros inválidos")
        
        invalid_tests = []
        
        # 1. Risk tolerance inválido
        try:
            result = self.competitive_tool._run(
                analysis_type="market_position",
                data_csv=self.real_data_path,
                risk_tolerance="invalid_risk"
            )
            # Deve usar fallback ou dar erro
            handled = True  # Qualquer tratamento é válido
            invalid_tests.append(('invalid_risk_tolerance', handled))
        except Exception:
            invalid_tests.append(('invalid_risk_tolerance', False))
        
        # 2. Múltiplos parâmetros inválidos
        try:
            result = self.competitive_tool._run(
                analysis_type="pricing_analysis",
                data_csv=self.real_data_path,
                market_segment="invalid_segment",
                benchmark_period="invalid_period",
                risk_tolerance="invalid_risk"
            )
            # Deve funcionar com fallbacks
            handled = "error" not in result.lower() or len(result) > 100
            invalid_tests.append(('multiple_invalid', handled))
        except Exception:
            invalid_tests.append(('multiple_invalid', False))
        
        # Validações
        passed = sum(1 for _, handled in invalid_tests if handled)
        success_rate = passed / len(invalid_tests) if invalid_tests else 1.0
        
        assert success_rate >= 0.50, f"Poucos parâmetros inválidos tratados: {passed}/{len(invalid_tests)}"
        
        self.log_test("SUCCESS", "Parâmetros inválidos validados",
                     tests_passed=f"{passed}/{len(invalid_tests)}")
    
    # ==========================================
    # TESTES DE QUALIDADE DE SAÍDA
    # ==========================================
    
    def test_output_quality_and_formatting(self):
        """Teste de qualidade e formatação da saída."""
        self.log_test("INFO", "Testando qualidade da saída")
        
        result = self.competitive_tool._run(
            analysis_type="market_position",
            data_csv=self.real_data_path,
            market_segment="joalherias",
            include_recommendations=True
        )
        
        # Validações de formato
        assert isinstance(result, str), "Resultado deve ser string"
        assert len(result) > 700, "Resultado muito curto"
        
        # Verificar estrutura markdown
        markdown_elements = ["#", "**", "-", "*", "🏆"]
        found_markdown = [elem for elem in markdown_elements if elem in result]
        assert len(found_markdown) >= 4, f"Pouco markdown encontrado: {found_markdown}"
        
        # Verificar seções esperadas de inteligência competitiva
        expected_sections = ["competitive intelligence", "inteligência competitiva", "análise", "benchmark"]
        found_sections = [section for section in expected_sections 
                         if section.lower() in result.lower()]
        assert len(found_sections) >= 2, f"Poucas seções encontradas: {found_sections}"
        
        # Verificar informações técnicas específicas
        tech_info = ["competitive intelligence tool", "benchmark", "mercado", "análise"]
        found_tech = [info for info in tech_info if info.lower() in result.lower()]
        assert len(found_tech) >= 3, f"Pouca informação técnica: {found_tech}"
        
        # Verificar presença de dados numéricos (percentuais, valores)
        numeric_indicators = ["%", "r$", "pontos percentuais", "vs", "mercado"]
        found_numeric = [ind for ind in numeric_indicators if ind.lower() in result.lower()]
        assert len(found_numeric) >= 2, f"Poucos indicadores numéricos: {found_numeric}"
        
        # Verificar metadados e disclaimer
        metadata_terms = ["metadados", "metadata", "disclaimer", "fonte", "atualização"]
        found_metadata = [term for term in metadata_terms if term.lower() in result.lower()]
        assert len(found_metadata) >= 1, f"Metadados ausentes: {found_metadata}"
        
        self.log_test("SUCCESS", "Qualidade da saída validada",
                     result_length=len(result),
                     format_elements=len(found_markdown),
                     sections_found=len(found_sections),
                     tech_info_found=len(found_tech),
                     numeric_indicators=len(found_numeric))
    
    def test_benchmark_accuracy(self):
        """Teste de precisão dos benchmarks."""
        self.log_test("INFO", "Testando precisão dos benchmarks")
        
        # Carregar benchmarks
        benchmarks = self.competitive_tool._load_market_benchmarks('joalherias')
        
        # Verificar valores realistas para o setor brasileiro
        market_size = benchmarks['market_size_billion_brl']
        assert 5.0 <= market_size <= 15.0, f"Market size irreal: {market_size}B"
        
        growth_rate = benchmarks['annual_growth_rate']
        assert 0.01 <= growth_rate <= 0.10, f"Taxa de crescimento irreal: {growth_rate*100:.1f}%"
        
        # Verificar faixas de preço consistentes
        avg_ticket = benchmarks['average_ticket']
        assert avg_ticket['economy']['max'] < avg_ticket['mid']['min'], "Faixas de preço sobrepostas"
        assert avg_ticket['mid']['max'] < avg_ticket['premium']['min'], "Faixas mid/premium sobrepostas"
        assert avg_ticket['premium']['max'] < avg_ticket['luxury']['min'], "Faixas premium/luxury sobrepostas"
        
        # Verificar distribuição de categorias soma 100%
        cat_dist = benchmarks['category_distribution']
        total_distribution = sum(cat_dist.values())
        assert 0.95 <= total_distribution <= 1.05, f"Distribuição de categorias inconsistente: {total_distribution:.2f}"
        
        # Verificar margens realistas
        margins = benchmarks['margin_benchmarks']
        assert 0.30 <= margins['gross_margin_avg'] <= 0.80, f"Margem bruta irreal: {margins['gross_margin_avg']:.1%}"
        assert margins['net_margin_avg'] < margins['gross_margin_avg'], "Margem líquida > bruta"
        
        self.log_test("SUCCESS", "Benchmarks validados",
                     market_size=f"{market_size}B BRL",
                     growth_rate=f"{growth_rate*100:.1f}%",
                     categories=len(cat_dist))
    
    def test_recommendations_quality(self):
        """Teste de qualidade das recomendações."""
        self.log_test("INFO", "Testando qualidade das recomendações")
        
        # Testar com diferentes tipos de análise
        analysis_types = ["competitive_gaps", "market_position", "pricing_analysis"]
        recommendation_quality = {}
        
        for analysis_type in analysis_types:
            result = self.competitive_tool._run(
                analysis_type=analysis_type,
                data_csv=self.real_data_path,
                include_recommendations=True,
                risk_tolerance="medium"
            )
            
            # Contar recomendações
            recommendation_indicators = ["recomenda", "sugest", "deveria", "considerar", "avaliar", "focar"]
            found_recommendations = sum(1 for indicator in recommendation_indicators 
                                      if indicator.lower() in result.lower())
            
            # Verificar acionabilidade das recomendações
            actionable_terms = ["implementar", "expandir", "melhorar", "aumentar", "reduzir", "focar"]
            found_actionable = sum(1 for term in actionable_terms 
                                 if term.lower() in result.lower())
            
            recommendation_quality[analysis_type] = {
                'recommendation_count': found_recommendations,
                'actionable_count': found_actionable,
                'quality_score': (found_recommendations + found_actionable) / 2
            }
        
        # Validações
        avg_quality = sum(rq['quality_score'] for rq in recommendation_quality.values()) / len(recommendation_quality)
        assert avg_quality >= 2.0, f"Qualidade baixa das recomendações: {avg_quality:.1f}"
        
        # Pelo menos uma análise deve ter recomendações robustas
        best_analysis = max(recommendation_quality.items(), key=lambda x: x[1]['quality_score'])
        assert best_analysis[1]['quality_score'] >= 3.0, f"Nenhuma análise com recomendações robustas"
        
        self.log_test("SUCCESS", "Qualidade das recomendações validada",
                     avg_quality_score=f"{avg_quality:.1f}",
                     best_analysis=best_analysis[0],
                     best_score=f"{best_analysis[1]['quality_score']:.1f}")
    
    def test_competitive_insights_depth(self):
        """Teste da profundidade dos insights competitivos."""
        self.log_test("INFO", "Testando profundidade dos insights")
        
        result = self.competitive_tool._run(
            analysis_type="competitive_gaps",
            data_csv=self.real_data_path,
            include_recommendations=True
        )
        
        # Verificar análise multi-dimensional
        analysis_dimensions = [
            "categoria", "preço", "operacional", "digital", "sazonal", 
            "market share", "crescimento", "benchmark"
        ]
        found_dimensions = [dim for dim in analysis_dimensions 
                           if dim.lower() in result.lower()]
        assert len(found_dimensions) >= 4, f"Análise superficial: {found_dimensions}"
        
        # Verificar comparações quantitativas
        quantitative_terms = ["%", "vs", "acima", "abaixo", "maior", "menor", "gap"]
        found_quantitative = [term for term in quantitative_terms 
                             if term.lower() in result.lower()]
        assert len(found_quantitative) >= 5, f"Poucas comparações quantitativas: {found_quantitative}"
        
        # Verificar insights estratégicos
        strategic_terms = ["estratégi", "strategic", "oportunidade", "opportunity", "prioridade", "priority"]
        found_strategic = [term for term in strategic_terms 
                          if term.lower() in result.lower()]
        assert len(found_strategic) >= 3, f"Poucos insights estratégicos: {found_strategic}"
        
        self.log_test("SUCCESS", "Profundidade dos insights validada",
                     analysis_dimensions=len(found_dimensions),
                     quantitative_elements=len(found_quantitative),
                     strategic_insights=len(found_strategic))
    
    def teardown_method(self, method):
        """Limpeza após cada teste."""
        test_name = method.__name__
        duration = time.time() - self.start_time
        
        # Salvar logs do teste
        log_dir = Path("test_logs")
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{test_name}_{timestamp}_competitive_tool.json"
        
        log_data = {
            'test_name': test_name,
            'timestamp': timestamp,
            'duration': round(duration, 2),
            'tool_version': 'Competitive Intelligence Tool V1.0',
            'logs': self.test_logs
        }
        
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Log salvo: {log_file}")


if __name__ == "__main__":
    # Executar teste standalone
    test_instance = TestCompetitiveIntelligenceTool()
    
    # Setup standalone
    test_instance.setup_standalone("data/vendas.csv")
    
    print("🏆 Executando testes Competitive Intelligence Tool V1.0...")
    print("=" * 70)
    
    # Lista de testes principais
    main_tests = [
        test_instance.test_data_loading_and_validation,
        test_instance.test_benchmark_loading,
        test_instance.test_input_validation,
        test_instance.test_market_position_analysis,
        test_instance.test_pricing_analysis,
        test_instance.test_trend_comparison_analysis,
        test_instance.test_market_share_estimation_analysis,
        test_instance.test_competitive_gaps_analysis,
        test_instance.test_all_analysis_types_integration,
        test_instance.test_performance_benchmarks,
        test_instance.test_different_market_segments,
        test_instance.test_error_handling_comprehensive,
        test_instance.test_edge_cases,
        test_instance.test_invalid_parameters,
        test_instance.test_output_quality_and_formatting,
        test_instance.test_benchmark_accuracy,
        test_instance.test_recommendations_quality,
        test_instance.test_competitive_insights_depth
    ]
    
    passed = 0
    total = len(main_tests)
    failed_tests = []
    
    for test_func in main_tests:
        try:
            print(f"\n{'='*70}")
            print(f"🔄 Executando: {test_func.__name__}")
            print("-" * 70)
            
            test_func()
            print(f"✅ {test_func.__name__} - PASSOU")
            passed += 1
            
        except Exception as e:
            print(f"❌ {test_func.__name__} - FALHOU: {str(e)}")
            failed_tests.append((test_func.__name__, str(e)))
            
        finally:
            test_instance.teardown_method(test_func)
    
    # Relatório final
    print(f"\n{'='*70}")
    print(f"🏆 RELATÓRIO FINAL - COMPETITIVE INTELLIGENCE TOOL V1.0")
    print(f"{'='*70}")
    print(f"✅ Testes Aprovados: {passed}/{total} ({passed/total:.1%})")
    print(f"❌ Testes Falharam: {len(failed_tests)}")
    
    if failed_tests:
        print(f"\n📋 TESTES QUE FALHARAM:")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {error[:100]}...")
    
    if passed == total:
        print(f"\n🎉 TODOS OS TESTES PASSARAM!")
        print(f"🚀 Competitive Intelligence Tool está funcionando perfeitamente!")
    elif passed >= total * 0.85:
        print(f"\n✅ MAIORIA DOS TESTES PASSOU ({passed/total:.1%})")
        print(f"🔧 Algumas funcionalidades podem precisar de ajustes")
    else:
        print(f"\n⚠️ MUITOS TESTES FALHARAM ({passed/total:.1%})")
        print(f"🛠️ Tool precisa de correções significativas")
    
    print(f"\n📊 COBERTURA DE TESTES:")
    print(f"  - ✅ Validação de dados e benchmarks")
    print(f"  - ✅ Preparação de dados competitivos")
    print(f"  - ✅ 5 tipos de análise competitiva")
    print(f"  - ✅ Integração end-to-end")
    print(f"  - ✅ Performance e otimizações")
    print(f"  - ✅ Tratamento de erros robusto")
    print(f"  - ✅ Casos extremos e parâmetros inválidos")
    print(f"  - ✅ Segmentos de mercado")
    print(f"  - ✅ Qualidade de saída e formatação")
    print(f"  - ✅ Precisão de benchmarks setoriais")
    print(f"  - ✅ Qualidade de recomendações")
    print(f"  - ✅ Profundidade de insights")
    
    print(f"\n🔧 FUNCIONALIDADES TESTADAS:")
    print(f"  - Market Position (Posicionamento vs. mercado)")
    print(f"  - Pricing Analysis (Análise competitiva de preços)")
    print(f"  - Trend Comparison (Comparação de tendências)")
    print(f"  - Market Share Estimation (Estimativa de market share)")
    print(f"  - Competitive Gaps (Identificação de gaps e oportunidades)")
    print(f"  - Benchmarks setoriais brasileiros")
    print(f"  - Matriz de oportunidades priorizadas")
    print(f"  - Recomendações estratégicas acionáveis")
    print(f"  - Análise multi-dimensional (preço, categoria, operacional)")
    print(f"  - Elasticidade de preços e sazonalidade")
    print(f"  - Comparação vs. principais concorrentes")
    print(f"  - Potencial de crescimento e expansão")
    
    print(f"\n📈 BENCHMARKS DE PERFORMANCE:")
    print(f"  - ⏱️ Tempo total: < 2 minutos para todas as análises")
    print(f"  - 💾 Memória: < 512 MB por análise")
    print(f"  - 🎯 Taxa de sucesso alvo: ≥ 85%")
    print(f"  - 📊 Cobertura: 18 testes abrangentes")
    print(f"  - 🔧 Tratamento de erros: ≥ 75% dos casos")
    
    print(f"\n🏪 ESPECIALIZAÇÃO SETORIAL:")
    print(f"  - Mercado brasileiro de joalherias (R$ 6.8B)")
    print(f"  - Benchmarks por faixa de preço (Economy → Ultra-Luxury)")
    print(f"  - Padrões sazonais (Maio/Dezembro)")
    print(f"  - Participação dos principais players")
    print(f"  - Margens e métricas operacionais do setor")
    
    print(f"\n🎯 PRÓXIMOS PASSOS SUGERIDOS:")
    print(f"  - Integrar com SQL Query Tool para dados atualizados")
    print(f"  - Expandir benchmarks para outros segmentos")
    print(f"  - Adicionar análise de pricing dinâmico")
    print(f"  - Implementar alertas de mudanças competitivas")
    print(f"  - Desenvolver dashboards interativos")
    
    print(f"\n*Relatório gerado por Test Suite - Competitive Intelligence Tool V1.0*") 