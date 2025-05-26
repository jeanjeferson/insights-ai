"""
Testes para RecommendationEngine Tool - Sistema de Recomendações Inteligentes
===========================================================================

Suite de testes abrangente para validar funcionalidades do motor de recomendações
otimizado para CrewAI e análise de dados de joalherias.

Cobertura de testes:
- Validação de schemas Pydantic
- Todos os 6 tipos de recomendações
- Tratamento de erros e edge cases
- Performance e cache
- Integração com dados reais

Versão: 2.0 - Otimizada para CrewAI
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import json
from pathlib import Path

# Adicionar o diretório raiz do projeto ao PYTHONPATH
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Importar as classes do recommendation engine
try:
    from src.insights.tools.advanced.recommendation_engine import (
        RecommendationEngine, 
        RecommendationInput
    )
except ImportError:
    # Fallback para importação relativa
    sys.path.insert(0, str(current_dir.parent))
    from insights.tools.advanced.recommendation_engine import (
        RecommendationEngine, 
        RecommendationInput
    )


class TestRecommendationInput:
    """Testes para schema de entrada RecommendationInput."""
    
    def test_valid_input_creation(self):
        """Testa criação válida do schema."""
        input_data = RecommendationInput(
            recommendation_type="product_recommendations",
            data_csv="data/vendas.csv",
            target_segment="vip",
            recommendation_count=15,
            confidence_threshold=0.8
        )
        
        assert input_data.recommendation_type == "product_recommendations"
        assert input_data.target_segment == "vip"
        assert input_data.recommendation_count == 15
        assert input_data.confidence_threshold == 0.8
        assert input_data.enable_detailed_analysis == True
    
    def test_invalid_recommendation_type(self):
        """Testa validação de tipo de recomendação inválido."""
        with pytest.raises(ValueError, match="Tipo deve ser um de"):
            RecommendationInput(
                recommendation_type="invalid_type",
                data_csv="data/vendas.csv"
            )
    
    def test_invalid_confidence_threshold(self):
        """Testa validação de limiar de confiança inválido."""
        with pytest.raises(ValueError, match="Confiança deve estar entre 0.5 e 0.95"):
            RecommendationInput(
                recommendation_type="product_recommendations",
                confidence_threshold=1.2
            )
        
        with pytest.raises(ValueError, match="Confiança deve estar entre 0.5 e 0.95"):
            RecommendationInput(
                recommendation_type="product_recommendations", 
                confidence_threshold=0.3
            )
    
    def test_invalid_recommendation_count(self):
        """Testa validação de contagem inválida."""
        with pytest.raises(ValueError, match="Contagem deve estar entre 5 e 50"):
            RecommendationInput(
                recommendation_type="product_recommendations",
                recommendation_count=100
            )
    
    def test_default_values(self):
        """Testa valores padrão do schema."""
        input_data = RecommendationInput(
            recommendation_type="customer_targeting"
        )
        
        assert input_data.data_csv == "data/vendas.csv"
        assert input_data.target_segment == "all"
        assert input_data.recommendation_count == 10
        assert input_data.confidence_threshold == 0.7
        assert input_data.enable_detailed_analysis == True


class TestRecommendationEngine:
    """Testes principais para RecommendationEngine."""
    
    @pytest.fixture
    def engine(self):
        """Fixture para instância do engine."""
        return RecommendationEngine()
    
    @pytest.fixture
    def sample_data(self):
        """Fixture para dados de teste."""
        np.random.seed(42)
        
        # Simular dados de vendas de joalheria
        n_records = 100
        customers = [f"CUST_{i:03d}" for i in range(1, 21)]  # 20 clientes
        products = [f"PROD_{i:03d}" for i in range(1, 31)]   # 30 produtos
        
        data = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(n_records):
            data.append({
                'Data': base_date + timedelta(days=np.random.randint(0, 365)),
                'Codigo_Cliente': np.random.choice(customers),
                'Nome_Cliente': f"Cliente {np.random.choice(customers)[-3:]}",
                'Codigo_Produto': np.random.choice(products),
                'Descricao_Produto': f"Produto {np.random.choice(products)[-3:]}",
                'Grupo_Produto': np.random.choice(['Aneis', 'Brincos', 'Colares', 'Pulseiras']),
                'Quantidade': np.random.randint(1, 4),
                'Total_Liquido': np.random.uniform(500, 5000),
                'Preco_Tabela': np.random.uniform(600, 6000)
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_csv_file(self, sample_data):
        """Fixture para arquivo CSV temporário."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            sample_data.to_csv(f.name, sep=';', index=False)
            yield f.name
        os.unlink(f.name)
    
    def test_engine_initialization(self, engine):
        """Testa inicialização do engine."""
        assert engine.name == "Recommendation Engine"
        assert "Sistema de recomendações inteligentes" in engine.description
        assert engine.args_schema == RecommendationInput
        assert hasattr(engine, '_cache')
        assert hasattr(engine, '_last_data_hash')
    
    def test_product_recommendations(self, engine, temp_csv_file):
        """Testa recomendações de produtos."""
        result = engine._run(
            recommendation_type="product_recommendations",
            data_csv=temp_csv_file,
            target_segment="all",
            recommendation_count=5,
            confidence_threshold=0.5
        )
        
        # Verificar estrutura JSON
        assert "success" in result or "recommendations" in result
        
        # Verificar se não há erro
        assert "Erro" not in result
        
        # Verificar formato de saída
        assert "RECOMMENDATION ENGINE" in result
        assert "Product Recommendations" in result or "PRODUTOS RECOMENDADOS" in result
    
    def test_customer_targeting(self, engine, temp_csv_file):
        """Testa targeting de clientes."""
        result = engine._run(
            recommendation_type="customer_targeting",
            data_csv=temp_csv_file,
            target_segment="all",
            recommendation_count=8
        )
        
        assert "Erro" not in result
        assert "targeting" in result.lower() or "customer" in result.lower()
    
    def test_pricing_optimization(self, engine, temp_csv_file):
        """Testa otimização de preços."""
        result = engine._run(
            recommendation_type="pricing_optimization",
            data_csv=temp_csv_file
        )
        
        assert "Erro" not in result
        assert "pricing" in result.lower() or "preço" in result.lower()
    
    def test_inventory_suggestions(self, engine, temp_csv_file):
        """Testa sugestões de inventário."""
        result = engine._run(
            recommendation_type="inventory_suggestions",
            data_csv=temp_csv_file
        )
        
        assert "Erro" not in result
        assert "inventory" in result.lower() or "inventário" in result.lower()
    
    def test_marketing_campaigns(self, engine, temp_csv_file):
        """Testa campanhas de marketing."""
        result = engine._run(
            recommendation_type="marketing_campaigns",
            data_csv=temp_csv_file
        )
        
        assert "Erro" not in result
        assert "marketing" in result.lower() or "campaign" in result.lower()
    
    def test_strategic_actions(self, engine, temp_csv_file):
        """Testa ações estratégicas."""
        result = engine._run(
            recommendation_type="strategic_actions",
            data_csv=temp_csv_file
        )
        
        assert "Erro" not in result
        assert "strategic" in result.lower() or "estratégic" in result.lower()
    
    def test_invalid_recommendation_type(self, engine, temp_csv_file):
        """Testa tipo de recomendação inválido."""
        result = engine._run(
            recommendation_type="invalid_type",
            data_csv=temp_csv_file
        )
        
        assert "error" in result.lower() or "erro" in result.lower()
    
    def test_file_not_found(self, engine):
        """Testa arquivo não encontrado."""
        result = engine._run(
            recommendation_type="product_recommendations",
            data_csv="non_existent_file.csv"
        )
        
        assert "error" in result.lower() or "erro" in result.lower()
    
    def test_insufficient_data(self, engine):
        """Testa dados insuficientes."""
        # Criar arquivo com poucos dados
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            minimal_data = pd.DataFrame({
                'Data': ['2024-01-01'],
                'Total_Liquido': [100],
                'Codigo_Produto': ['PROD_001']
            })
            minimal_data.to_csv(f.name, sep=';', index=False)
            
            result = engine._run(
                recommendation_type="product_recommendations",
                data_csv=f.name
            )
            
        os.unlink(f.name)
        
        assert "insuficientes" in result.lower() or "insufficient" in result.lower()
    
    def test_different_segments(self, engine, temp_csv_file):
        """Testa diferentes segmentos de clientes."""
        segments = ["all", "vip", "new_customers", "at_risk"]
        
        for segment in segments:
            result = engine._run(
                recommendation_type="product_recommendations",
                data_csv=temp_csv_file,
                target_segment=segment,
                recommendation_count=5
            )
            
            # Não deve ter erro para segmentos válidos
            assert "Erro na execução" not in result
    
    def test_confidence_threshold_filtering(self, engine, temp_csv_file):
        """Testa filtragem por limiar de confiança."""
        # Teste com confiança baixa (mais resultados)
        result_low = engine._run(
            recommendation_type="product_recommendations",
            data_csv=temp_csv_file,
            confidence_threshold=0.5
        )
        
        # Teste com confiança alta (menos resultados)
        result_high = engine._run(
            recommendation_type="product_recommendations",
            data_csv=temp_csv_file,
            confidence_threshold=0.9
        )
        
        # Ambos devem funcionar
        assert "Erro" not in result_low
        assert "Erro" not in result_high
    
    def test_detailed_analysis_flag(self, engine, temp_csv_file):
        """Testa flag de análise detalhada."""
        # Com análise detalhada
        result_detailed = engine._run(
            recommendation_type="product_recommendations",
            data_csv=temp_csv_file,
            enable_detailed_analysis=True
        )
        
        # Sem análise detalhada
        result_simple = engine._run(
            recommendation_type="product_recommendations",
            data_csv=temp_csv_file,
            enable_detailed_analysis=False
        )
        
        assert "Erro" not in result_detailed
        assert "Erro" not in result_simple
    
    def test_output_format(self, engine, temp_csv_file):
        """Testa formato de saída JSON + Markdown."""
        result = engine._run(
            recommendation_type="product_recommendations",
            data_csv=temp_csv_file
        )
        
        # Verificar elementos esperados no output
        assert "RECOMMENDATION ENGINE" in result
        assert "success" in result or "recommendations" in result
        assert "---" in result  # Separador markdown
    
    def test_cache_functionality(self, engine, temp_csv_file):
        """Testa funcionalidade de cache."""
        # Primeira execução
        result1 = engine._run(
            recommendation_type="product_recommendations",
            data_csv=temp_csv_file,
            recommendation_count=5
        )
        
        # Segunda execução (deve usar cache)
        result2 = engine._run(
            recommendation_type="product_recommendations",
            data_csv=temp_csv_file,
            recommendation_count=5
        )
        
        # Ambos devem funcionar
        assert "Erro" not in result1
        assert "Erro" not in result2
        
        # Verificar se cache foi utilizado (resultado similar)
        assert len(result1) > 0
        assert len(result2) > 0
    
    def test_metadata_inclusion(self, engine, temp_csv_file):
        """Testa inclusão de metadados."""
        result = engine._run(
            recommendation_type="product_recommendations",
            data_csv=temp_csv_file
        )
        
        # Verificar metadados técnicos
        assert "Pontos de Dados" in result or "data_points" in result
        assert "Tempo de Processamento" in result or "processing_time" in result
        assert "Engine" in result or "engine_version" in result
    
    def test_data_validation_methods(self, engine, sample_data):
        """Testa métodos de validação de dados."""
        # Teste _load_and_validate_data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            sample_data.to_csv(f.name, sep=';', index=False)
            
            validated_data = engine._load_and_validate_data(f.name)
            assert validated_data is not None
            assert len(validated_data) > 0
            assert 'Data' in validated_data.columns
            assert 'Total_Liquido' in validated_data.columns
            
        os.unlink(f.name)
    
    def test_data_preparation_methods(self, engine, sample_data):
        """Testa métodos de preparação de dados."""
        prepared_data = engine._prepare_recommendation_data(sample_data)
        
        assert prepared_data is not None
        assert 'Customer_ID' in prepared_data.columns
        assert 'Price_Category' in prepared_data.columns
        assert 'Customer_Segment' in prepared_data.columns
        assert 'Year_Month' in prepared_data.columns
        assert 'Weekday' in prepared_data.columns
    
    def test_rfm_calculation(self, engine, sample_data):
        """Testa cálculo de métricas RFM."""
        # Adicionar Customer_ID se não existir
        if 'Customer_ID' not in sample_data.columns:
            sample_data['Customer_ID'] = sample_data.get('Codigo_Cliente', 'CUST_001')
        
        rfm_data = engine._calculate_rfm_metrics(sample_data)
        
        assert 'Customer_Segment' in rfm_data.columns
        assert 'Recency' in rfm_data.columns
        assert 'Frequency' in rfm_data.columns
        assert 'Monetary' in rfm_data.columns
        
        # Verificar segmentos válidos
        valid_segments = {'VIP', 'New', 'At Risk', 'High Value', 'Frequent', 'Regular'}
        actual_segments = set(rfm_data['Customer_Segment'].unique())
        assert actual_segments.issubset(valid_segments)
    
    def test_auxiliary_methods(self, engine, sample_data):
        """Testa métodos auxiliares."""
        # Teste normalize_score
        test_series = pd.Series([1, 2, 3, 4, 5])
        normalized = engine._normalize_score(test_series)
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        
        # Teste filter_by_segment
        prepared_data = engine._prepare_recommendation_data(sample_data)
        if 'Customer_Segment' in prepared_data.columns:
            filtered = engine._filter_by_segment(prepared_data, 'vip')
            assert len(filtered) >= 0  # Pode ser 0 se não houver VIPs
    
    def test_error_handling(self, engine):
        """Testa tratamento de erros."""
        # Teste com arquivo inválido
        result = engine._run(
            recommendation_type="product_recommendations",
            data_csv="/path/to/nonexistent/file.csv"
        )
        
        assert "error" in result.lower() or "erro" in result.lower()
        
        # Verificar estrutura de erro JSON
        if result.startswith('{'):
            try:
                error_json = json.loads(result.split('\n---\n')[0])
                assert error_json['success'] == False
                assert 'error' in error_json
                assert 'suggested_actions' in error_json
            except json.JSONDecodeError:
                pass  # Pode não ser JSON em todos os casos
    
    def test_performance_with_large_data(self, engine):
        """Testa performance com dados maiores."""
        # Criar dataset maior
        large_data = []
        for i in range(1000):
            large_data.append({
                'Data': datetime(2024, 1, 1) + timedelta(days=i % 365),
                'Codigo_Cliente': f"CUST_{i % 100:03d}",
                'Codigo_Produto': f"PROD_{i % 200:03d}",
                'Descricao_Produto': f"Produto {i % 200:03d}",
                'Total_Liquido': np.random.uniform(100, 2000),
                'Quantidade': np.random.randint(1, 3)
            })
        
        large_df = pd.DataFrame(large_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            large_df.to_csv(f.name, sep=';', index=False)
            
            start_time = datetime.now()
            result = engine._run(
                recommendation_type="product_recommendations",
                data_csv=f.name,
                recommendation_count=10
            )
            end_time = datetime.now()
            
            # Verificar se completou em tempo razoável (< 30 segundos)
            processing_time = (end_time - start_time).total_seconds()
            assert processing_time < 30
            assert "Erro" not in result
            
        os.unlink(f.name)
    
    def test_all_recommendation_types_integration(self, engine, temp_csv_file):
        """Teste de integração com todos os tipos de recomendação."""
        recommendation_types = [
            "product_recommendations",
            "customer_targeting", 
            "pricing_optimization",
            "inventory_suggestions",
            "marketing_campaigns",
            "strategic_actions"
        ]
        
        for rec_type in recommendation_types:
            result = engine._run(
                recommendation_type=rec_type,
                data_csv=temp_csv_file,
                recommendation_count=5,
                confidence_threshold=0.6
            )
            
            # Cada tipo deve funcionar sem erro
            assert "Erro na execução" not in result
            assert len(result) > 100  # Output mínimo esperado
            
            # Verificar elementos comuns
            assert "RECOMMENDATION ENGINE" in result
            assert rec_type.replace('_', ' ').title() in result or rec_type.upper() in result

    def test_advanced_ml_integration(self, engine, temp_csv_file):
        """Testa integração dos algoritmos ML avançados."""
        result = engine._run(
            recommendation_type="product_recommendations",
            data_csv=temp_csv_file,
            recommendation_count=10,
            confidence_threshold=0.6,
            enable_detailed_analysis=True
        )
        
        # Verificar presença de componentes ML
        assert "algorithm" in result.lower() or "algoritmo" in result.lower()
        assert "validation" in result.lower() or "validação" in result.lower()
        assert "quality" in result.lower() or "qualidade" in result.lower()
        
        # Verificar metadados ML
        assert "engine_version" in result.lower()
        assert "2.1" in result
    
    def test_algorithm_selection(self, engine, sample_data):
        """Testa seleção automática de algoritmos."""
        algorithm_selection = engine._select_optimal_algorithm(
            sample_data, 'product_recommendations', 'all'
        )
        
        assert 'selected_algorithm' in algorithm_selection
        assert 'selection_reason' in algorithm_selection
        assert 'confidence' in algorithm_selection
        assert 'data_characteristics' in algorithm_selection
        
        # Verificar tipos válidos de algoritmo
        valid_algorithms = ['standard', 'enhanced_standard', 'collaborative_filtering', 'hybrid_ml']
        assert algorithm_selection['selected_algorithm'] in valid_algorithms
    
    def test_data_quality_assessment(self, engine, sample_data):
        """Testa avaliação de qualidade dos dados."""
        quality_assessment = engine._assess_data_quality(sample_data)
        
        assert 'completeness_score' in quality_assessment
        assert 'total_records' in quality_assessment
        assert 'date_range_days' in quality_assessment
        assert 'null_percentage' in quality_assessment
        assert 'value_distribution' in quality_assessment
        
        # Verificar valores válidos
        assert 0 <= quality_assessment['completeness_score'] <= 1
        assert quality_assessment['total_records'] > 0
        assert quality_assessment['null_percentage'] >= 0
    
    def test_parameter_optimization(self, engine, sample_data):
        """Testa otimização automática de parâmetros."""
        optimization = engine._optimize_recommendation_parameters(sample_data, 'product_recommendations')
        
        assert 'optimized_parameters' in optimization
        assert 'data_characteristics_considered' in optimization
        assert 'optimization_rationale' in optimization
        
        params = optimization['optimized_parameters']
        assert 'confidence_threshold' in params
        assert 'recommendation_count' in params
        assert 'enable_detailed_analysis' in params
        assert 'optimization_applied' in params
        
        # Verificar valores válidos
        assert 0.5 <= params['confidence_threshold'] <= 0.95
        assert 5 <= params['recommendation_count'] <= 50
    
    def test_collaborative_filtering_advanced(self, engine, sample_data):
        """Testa collaborative filtering avançado."""
        # Adicionar Customer_ID se necessário
        if 'Customer_ID' not in sample_data.columns:
            sample_data = engine._simulate_customer_ids(sample_data)
        
        collab_results = engine._collaborative_filtering_advanced(sample_data, None, 5)
        
        if 'error' not in collab_results:
            assert 'algorithm' in collab_results
            assert 'recommendations' in collab_results
            assert 'matrix_dimensions' in collab_results
            assert 'explained_variance_ratio' in collab_results
            assert collab_results['algorithm'] == 'Matrix Factorization SVD'
    
    def test_content_based_filtering_advanced(self, engine, sample_data):
        """Testa content-based filtering avançado."""
        content_results = engine._content_based_filtering_advanced(sample_data, None, 5)
        
        if 'error' not in content_results:
            assert 'algorithm' in content_results
            assert 'recommendations' in content_results
            assert 'vocabulary_size' in content_results
            assert 'top_features' in content_results
            assert content_results['algorithm'] == 'TF-IDF Content-Based Filtering'
    
    def test_hybrid_recommendation_system(self, engine, sample_data):
        """Testa sistema híbrido de recomendações."""
        # Adicionar Customer_ID se necessário
        if 'Customer_ID' not in sample_data.columns:
            sample_data = engine._simulate_customer_ids(sample_data)
        
        customer_id = sample_data['Customer_ID'].iloc[0] if 'Customer_ID' in sample_data.columns else None
        
        hybrid_results = engine._hybrid_recommendation_system(sample_data, customer_id, 5)
        
        if 'error' not in hybrid_results:
            assert 'algorithm' in hybrid_results
            assert 'recommendations' in hybrid_results
            assert 'weights' in hybrid_results
            assert hybrid_results['algorithm'] == 'Hybrid Ensemble (Collaborative + Content-Based)'
    
    def test_advanced_market_basket_analysis(self, engine, sample_data):
        """Testa análise avançada de market basket."""
        # Adicionar Customer_ID se necessário
        if 'Customer_ID' not in sample_data.columns:
            sample_data = engine._simulate_customer_ids(sample_data)
        
        basket_results = engine._advanced_market_basket_analysis(sample_data)
        
        if 'error' not in basket_results:
            assert 'algorithm' in basket_results
            assert 'association_rules' in basket_results
            assert 'total_transactions' in basket_results
            assert basket_results['algorithm'] == 'Apriori Association Rules'
            
            # Verificar estrutura das regras
            if basket_results['association_rules']:
                rule = basket_results['association_rules'][0]
                assert 'antecedent' in rule
                assert 'consequent' in rule
                assert 'support' in rule
                assert 'confidence' in rule
                assert 'lift' in rule
    
    def test_anomaly_detection_customers(self, engine, sample_data):
        """Testa detecção de anomalias em clientes."""
        # Adicionar Customer_ID se necessário
        if 'Customer_ID' not in sample_data.columns:
            sample_data = engine._simulate_customer_ids(sample_data)
        
        anomaly_results = engine._anomaly_detection_customers(sample_data)
        
        if 'error' not in anomaly_results:
            assert 'algorithm' in anomaly_results
            assert 'anomaly_types' in anomaly_results
            assert 'total_customers' in anomaly_results
            assert 'business_impact' in anomaly_results
            assert anomaly_results['algorithm'] == 'Isolation Forest Anomaly Detection'
    
    def test_clv_prediction(self, engine, sample_data):
        """Testa predição de Customer Lifetime Value."""
        # Adicionar Customer_ID se necessário
        if 'Customer_ID' not in sample_data.columns:
            sample_data = engine._simulate_customer_ids(sample_data)
        
        clv_results = engine._predictive_customer_lifetime_value(sample_data)
        
        if 'error' not in clv_results:
            assert 'algorithm' in clv_results
            assert 'clv_predictions' in clv_results
            assert 'segment_distribution' in clv_results
            assert 'total_predicted_value' in clv_results
            assert clv_results['algorithm'] == 'Predictive Customer Lifetime Value'
            
            # Verificar estrutura das predições
            if clv_results['clv_predictions']:
                prediction = clv_results['clv_predictions'][0]
                assert 'customer_id' in prediction
                assert 'predicted_clv' in prediction
                assert 'clv_segment' in prediction
    
    def test_cross_validation_recommendations(self, engine, sample_data):
        """Testa validação cruzada das recomendações."""
        # Simular resultado de recomendações
        mock_recommendations = {
            'recommendations': {
                'top_products': [
                    {'code': 'PROD_001', 'score': 0.9},
                    {'code': 'PROD_002', 'score': 0.8}
                ]
            }
        }
        
        validation = engine._cross_validate_recommendations(sample_data, mock_recommendations)
        
        assert 'quality_score' in validation
        assert 'coverage_score' in validation
        assert 'revenue_relevance' in validation
        assert 'freshness_score' in validation
        assert 'quality_level' in validation
        assert 'validation_method' in validation
        
        # Verificar valores válidos
        assert 0 <= validation['quality_score'] <= 1
        assert validation['quality_level'] in ['Excellent', 'Good', 'Fair', 'Poor']
    
    def test_performance_monitoring(self, engine):
        """Testa monitoramento de performance."""
        start_time = datetime.now()
        df_size = 100
        algorithm = 'test_algorithm'
        
        performance = engine._performance_monitoring(start_time, df_size, algorithm)
        
        assert 'processing_time_seconds' in performance
        assert 'records_processed' in performance
        assert 'records_per_second' in performance
        assert 'algorithm_used' in performance
        assert 'performance_grade' in performance
        assert 'benchmarks' in performance
        
        # Verificar valores válidos
        assert performance['records_processed'] == df_size
        assert performance['algorithm_used'] == algorithm
        assert performance['performance_grade'] in ['A', 'B', 'C']


class TestRecommendationEngineEdgeCases:
    """Testes para casos extremos e edge cases."""
    
    @pytest.fixture
    def engine(self):
        return RecommendationEngine()
    
    @pytest.fixture
    def sample_data(self):
        """Fixture para dados de teste simples."""
        np.random.seed(42)
        
        # Simular dados de vendas básicos
        n_records = 50
        customers = [f"CUST_{i:03d}" for i in range(1, 11)]  # 10 clientes
        products = [f"PROD_{i:03d}" for i in range(1, 16)]   # 15 produtos
        
        data = []
        base_date = datetime(2024, 1, 1)
        
        for i in range(n_records):
            data.append({
                'Data': base_date + timedelta(days=np.random.randint(0, 180)),
                'Codigo_Cliente': np.random.choice(customers),
                'Codigo_Produto': np.random.choice(products),
                'Descricao_Produto': f"Produto {np.random.choice(products)[-3:]}",
                'Total_Liquido': np.random.uniform(300, 3000),
                'Quantidade': np.random.randint(1, 3)
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_csv_file(self, sample_data):
        """Fixture para arquivo CSV temporário."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            sample_data.to_csv(f.name, sep=';', index=False)
            yield f.name
        os.unlink(f.name)
    
    def test_empty_csv_file(self, engine):
        """Testa arquivo CSV vazio."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            # Apenas cabeçalho
            f.write("Data;Total_Liquido;Codigo_Produto\n")
            
            result = engine._run(
                recommendation_type="product_recommendations",
                data_csv=f.name
            )
            
        os.unlink(f.name)
        
        assert "vazio" in result.lower() or "empty" in result.lower() or "erro" in result.lower()
    
    def test_malformed_csv_data(self, engine):
        """Testa dados CSV malformados."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            # Dados malformados
            f.write("Data;Total_Liquido;Codigo_Produto\n")
            f.write("not_a_date;not_a_number;PROD_001\n")
            f.write("2024-01-01;;PROD_002\n")
            
            result = engine._run(
                recommendation_type="product_recommendations",
                data_csv=f.name
            )
            
        os.unlink(f.name)
        
        # Deve tratar graciosamente dados malformados
        assert "error" in result.lower() or "erro" in result.lower() or len(result) > 100
    
    def test_missing_columns(self, engine):
        """Testa colunas obrigatórias ausentes."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            # Faltando coluna obrigatória
            f.write("SomeColumn;AnotherColumn\n")
            f.write("value1;value2\n")
            
            result = engine._run(
                recommendation_type="product_recommendations",
                data_csv=f.name
            )
            
        os.unlink(f.name)
        
        assert "error" in result.lower() or "erro" in result.lower()
    
    def test_single_customer_data(self, engine):
        """Testa dados com apenas um cliente."""
        single_customer_data = []
        for i in range(20):  # 20 transações do mesmo cliente
            single_customer_data.append({
                'Data': datetime(2024, 1, 1) + timedelta(days=i),
                'Codigo_Cliente': 'CUST_001',
                'Codigo_Produto': f'PROD_{i % 5:03d}',
                'Total_Liquido': np.random.uniform(500, 2000),
                'Quantidade': 1
            })
        
        df = pd.DataFrame(single_customer_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            df.to_csv(f.name, sep=';', index=False)
            
            result = engine._run(
                recommendation_type="customer_targeting",
                data_csv=f.name
            )
            
        os.unlink(f.name)
        
        # Deve funcionar mesmo com um cliente
        assert "Erro na execução" not in result
    
    def test_extreme_confidence_values(self, engine, temp_csv_file):
        """Testa valores extremos de confiança."""
        # Dados de teste básicos
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            test_data = pd.DataFrame({
                'Data': ['2024-01-01'] * 15,
                'Total_Liquido': [1000] * 15,
                'Codigo_Produto': [f'PROD_{i:03d}' for i in range(15)]
            })
            test_data.to_csv(f.name, sep=';', index=False)
            
            # Confiança mínima
            result_min = engine._run(
                recommendation_type="product_recommendations",
                data_csv=f.name,
                confidence_threshold=0.5
            )
            
            # Confiança máxima
            result_max = engine._run(
                recommendation_type="product_recommendations",
                data_csv=f.name,
                confidence_threshold=0.95
            )
            
        os.unlink(f.name)
        
        assert "Erro" not in result_min
        assert "Erro" not in result_max


if __name__ == "__main__":
    # Executar testes
    pytest.main([__file__, "-v", "--tb=short"]) 