# 📊 DOCUMENTAÇÃO COMPLETA DAS TOOLS - INSIGHTS AI

## 📋 Índice

### 🔬 [Tools de Análise Avançada](#tools-de-análise-avançada)
1. [Advanced Analytics Engine Tool](#1-advanced-analytics-engine-tool)
2. [Statistical Analysis Tool](#2-statistical-analysis-tool)
3. [Customer Insights Engine](#3-customer-insights-engine)

### 🎯 [Tools de Inteligência de Negócio](#tools-de-inteligência-de-negócio)
4. [Business Intelligence Tool](#4-business-intelligence-tool)
5. [KPI Calculator Tool](#5-kpi-calculator-tool)
6. [Competitive Intelligence Tool](#6-competitive-intelligence-tool)

### 🤖 [Tools de Recomendação e Predição](#tools-de-recomendação-e-predição)
7. [Recommendation Engine](#7-recommendation-engine)
8. [Prophet Forecast Tool](#8-prophet-forecast-tool)
9. [Risk Assessment Tool](#9-risk-assessment-tool)

### 🗄️ [Tools de Dados e Consultas](#tools-de-dados-e-consultas)
10. [SQL Query Tool](#10-sql-query-tool)
11. [SQL Query Tool Improved](#11-sql-query-tool-improved)
12. [DuckDuckGo Search Tool](#12-duckduckgo-search-tool)

### 📁 [Tools de Exportação e Geração de Arquivos](#tools-de-exportação-e-geração-de-arquivos)
13. [Customer Data Exporter](#13-customer-data-exporter)
14. [Financial Data Exporter](#14-financial-data-exporter)
15. [Inventory Data Exporter](#15-inventory-data-exporter)
16. [Product Data Exporter](#16-product-data-exporter)
17. [File Generation Tool](#17-file-generation-tool)

### 🔧 [Módulos Compartilhados](#módulos-compartilhados)
18. [Data Preparation Mixin](#18-data-preparation-mixin)
19. [Report Formatter Mixin](#19-report-formatter-mixin)
20. [Business Mixins](#20-business-mixins)
21. [Performance Optimizations](#21-performance-optimizations)

---

## 🔬 Tools de Análise Avançada

### 1. Advanced Analytics Engine Tool

**Descrição Breve:** Motor de análises avançadas com Machine Learning para insights profundos de joalherias.

**Objetivo Principal:** Descobrir padrões ocultos complexos usando algoritmos ML (Random Forest, XGBoost), realizar previsões avançadas, detectar anomalias e otimizar processos baseado em evidências estatísticas.

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `_run()` | Função principal que executa análises ML baseadas no tipo selecionado |
| `_load_and_prepare_ml_data()` | Carrega e prepara dados especificamente para análises de Machine Learning |
| `_add_ml_features()` | Adiciona features temporais, agregações e encoding para ML |
| `_ml_insights_analysis()` | Análise de insights ML com Random Forest e XGBoost |
| `_anomaly_detection_analysis()` | Detecção de anomalias usando Isolation Forest |
| `_demand_forecasting_analysis()` | Previsão de demanda adaptativa com ensemble de modelos |
| `_customer_behavior_analysis()` | Análise comportamental de clientes com clustering ML |
| `_product_lifecycle_analysis()` | Análise de ciclo de vida de produtos (placeholder) |
| `_price_optimization_analysis()` | Otimização de preços com elasticidade ML (placeholder) |
| `_inventory_optimization_analysis()` | Otimização de inventário com análise ABC ML (placeholder) |
| `_select_ml_features()` | Seleciona features otimizadas para algoritmos ML |
| `_configure_ml_models()` | Configura modelos ML baseado na complexidade desejada |
| `_generate_ml_insights()` | Gera insights de negócio baseados em resultados ML |
| `_generate_ml_recommendations()` | Cria recomendações estratégicas baseadas em ML |
| `_generate_adaptive_predictions()` | Gera previsões adaptativas usando modelos treinados |
| `_calculate_prediction_confidence()` | Calcula intervalos de confiança para previsões |
| `_generate_forecasting_insights()` | Gera insights adaptativos para forecasting |
| `_generate_forecasting_recommendations()` | Cria recomendações específicas para forecasting |

---

### 2. Statistical Analysis Tool

**Descrição Breve:** Motor de análises estatísticas avançadas para descobrir padrões ocultos em dados de joalherias.

**Objetivo Principal:** Realizar análises estatísticas rigorosas com testes de significância, clustering, correlações e segmentações para insights profundos sobre comportamento de clientes e performance de produtos.

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `_run()` | Executa análise estatística baseada no tipo selecionado |
| `_load_and_prepare_statistical_data()` | Carrega e prepara dados para análises estatísticas |
| `_add_statistical_features()` | Adiciona features estatísticas como Z-scores e percentis |
| `_advanced_correlation_analysis()` | Análise de correlação multi-dimensional com testes de significância |
| `_multidimensional_clustering_analysis()` | Clustering avançado (K-means, Hierárquico, DBSCAN) |
| `_comprehensive_outlier_analysis()` | Detecção de outliers usando múltiplos métodos estatísticos |
| `_advanced_distribution_analysis()` | Análise de distribuições e testes de normalidade |
| `_temporal_trend_analysis()` | Testes de tendência temporal e sazonalidade |
| `_demographic_patterns_analysis()` | Padrões demográficos avançados por idade/sexo/estado civil |
| `_generational_analysis()` | Análise geracional (Gen Z, Millennial, Gen X, Boomer) |
| `_behavioral_customer_segmentation()` | Segmentação comportamental avançada |
| `_geographic_performance_analysis()` | Performance estatística por estado e cidade |
| `_regional_patterns_analysis()` | Padrões sazonais e comportamentais regionais |
| `_price_elasticity_analysis()` | Análise de elasticidade de preços e sensibilidade |
| `_profitability_pattern_analysis()` | Padrões de rentabilidade com análise estatística |
| `_select_clustering_features()` | Seleciona features ótimas para clustering |
| `_select_optimal_clustering_method()` | Escolhe método de clustering baseado nos dados |
| `_perform_kmeans_clustering()` | Executa clustering K-means otimizado |
| `_perform_hierarchical_clustering()` | Executa clustering hierárquico |
| `_perform_dbscan_clustering()` | Executa clustering DBSCAN para outliers |
| `_interpret_correlation_strength()` | Interpreta força de correlações estatísticas |
| `_generate_correlation_insights()` | Gera insights baseados em correlações |
| `_generate_clustering_insights()` | Cria insights baseados em clusters identificados |

---

### 3. Customer Insights Engine

**Descrição Breve:** Motor avançado de insights de clientes para segmentação, análise comportamental e predição de churn.

**Objetivo Principal:** Segmentar clientes por comportamento, valor e perfil demográfico, entender padrões de compra, identificar riscos de abandono e calcular valor vitalício (CLV).

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `_run()` | Executa análise de insights de clientes baseada no tipo |
| `_prepare_customer_data()` | Prepara dados de clientes para análise |
| `_handle_missing_customer_data()` | Trata dados faltantes de clientes |
| `_validate_available_fields()` | Valida campos disponíveis nos dados |
| `_calculate_real_derived_fields()` | Calcula campos derivados reais |
| `_simulate_customer_ids()` | Simula IDs de clientes quando necessário |
| `_aggregate_customer_metrics()` | Agrega métricas por cliente |
| `_calculate_real_rfm_metrics()` | Calcula métricas RFM (Recency, Frequency, Monetary) |
| `_calculate_real_behavioral_metrics()` | Calcula métricas comportamentais avançadas |
| `_behavioral_segmentation()` | Segmentação comportamental de clientes |
| `_lifecycle_analysis()` | Análise de estágios do ciclo de vida do cliente |
| `_churn_prediction()` | Predição de risco de abandono de clientes |
| `_value_analysis()` | Análise de valor do cliente e CLV |
| `_preference_mining()` | Descoberta de preferências por demographics e produtos |
| `_journey_mapping()` | Mapeamento da jornada completa do cliente |
| `_real_demographic_segmentation()` | Segmentação demográfica real |
| `_real_geographic_segmentation()` | Segmentação geográfica real |
| `_real_advanced_clustering()` | Clustering avançado de clientes |
| `_rfm_segmentation()` | Segmentação RFM clássica |
| `_classify_rfm_segment()` | Classifica segmentos RFM |
| `_classify_lifecycle_stage()` | Classifica estágio do ciclo de vida |
| `_classify_journey_stage()` | Classifica estágio da jornada |
| `_analyze_lifecycle_transitions()` | Analisa transições entre estágios |
| `_identify_churn_risk_factors()` | Identifica fatores de risco de churn |
| `_generate_retention_strategies()` | Gera estratégias de retenção |
| `_calculate_churn_financial_impact()` | Calcula impacto financeiro do churn |
| `_analyze_value_concentration()` | Analisa concentração de valor nos clientes |

---

## 🎯 Tools de Inteligência de Negócio

### 4. Business Intelligence Tool

**Descrição Breve:** Plataforma unificada de Business Intelligence para relatórios executivos e dashboards interativos.

**Objetivo Principal:** Criar análises visuais profissionais, forecasting, segmentação de clientes e benchmarks do setor para tomada de decisão estratégica e monitoramento de performance.

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `_run()` | Executa análise de BI baseada no tipo selecionado |
| `_prepare_data_unified()` | Prepara dados unificados para análises de BI |
| `_filter_by_period()` | Filtra dados por período temporal |
| `_add_derived_metrics()` | Adiciona métricas derivadas para BI |
| `_simulate_missing_data()` | Simula dados faltantes quando necessário |
| `_calculate_kpis_unified()` | Calcula KPIs unificados para BI |
| `_calculate_customer_segments_unified()` | Calcula segmentação unificada de clientes |
| `_calculate_abc_analysis_unified()` | Executa análise ABC unificada |
| `_create_executive_summary()` | Cria resumo executivo C-level |
| `_create_executive_dashboard()` | Cria dashboard executivo interativo |
| `_create_financial_analysis()` | Cria análise financeira com forecasting |
| `_create_profitability_analysis()` | Análise de rentabilidade com custos reais |
| `_create_customer_intelligence()` | Inteligência de clientes com RFM |
| `_create_product_performance()` | Performance de produtos com ABC |
| `_create_demographic_analysis()` | Análise demográfica detalhada |
| `_create_geographic_analysis()` | Análise geográfica com mapas |
| `_create_sales_team_analysis()` | Análise de performance de vendedores |
| `_create_comprehensive_report()` | Relatório executivo integrado completo |
| `_calculate_health_score()` | Calcula score de saúde do negócio |
| `_format_output_unified()` | Formata saída unificada |
| `_format_interactive_result()` | Formata resultado interativo JSON |
| `_format_text_result()` | Formata resultado em texto |
| `_export_plotly_figure()` | Exporta figuras Plotly |
| `_export_html_report()` | Exporta relatório HTML completo |
| `_format_html_content()` | Formata conteúdo HTML estruturado |

---

### 5. KPI Calculator Tool

**Descrição Breve:** Calculadora avançada de KPIs para joalherias com alertas automáticos e benchmarks.

**Objetivo Principal:** Calcular métricas essenciais de negócio, comparar com padrões do setor, gerar alertas automáticos e insights acionáveis para monitoramento contínuo de performance.

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `_run()` | Executa cálculo de KPIs baseado na categoria |
| `_load_and_prepare_data()` | Carrega e prepara dados para cálculo de KPIs |
| `_calculate_business_health_score()` | Calcula score geral de saúde do negócio |
| `_calculate_financial_health_score()` | Score de saúde financeira |
| `_calculate_operational_health_score()` | Score de saúde operacional |
| `_calculate_inventory_health_score()` | Score de saúde de estoque |
| `_calculate_customer_health_score()` | Score de saúde de clientes |
| `_calculate_product_health_score()` | Score de saúde de produtos |
| `_generate_health_recommendations()` | Gera recomendações baseadas em saúde |
| `_calculate_financial_kpis_v3()` | KPIs financeiros versão 3 |
| `_calculate_operational_kpis_v3()` | KPIs operacionais versão 3 |
| `_calculate_inventory_kpis_v3()` | KPIs de estoque versão 3 |
| `_calculate_customer_kpis_v3()` | KPIs de clientes versão 3 |
| `_calculate_product_kpis_v3()` | KPIs de produtos versão 3 |
| `_calculate_benchmark_comparison_v3()` | Comparação com benchmarks v3 |
| `_generate_intelligent_alerts()` | Gera alertas inteligentes automáticos |
| `_generate_business_insights_v3()` | Gera insights de negócio v3 |
| `_integrate_statistical_insights()` | Integra insights estatísticos |
| `_calculate_growth_acceleration_v3()` | Calcula aceleração de crescimento |
| `_calculate_temporal_performance_v3()` | Performance temporal v3 |
| `_calculate_inventory_turnover_v3()` | Giro de estoque v3 |
| `_estimate_customer_metrics()` | Estima métricas de clientes |
| `_calculate_gini_coefficient()` | Calcula coeficiente de Gini |

---

### 6. Competitive Intelligence Tool

**Descrição Breve:** Ferramenta especializada em inteligência competitiva para análise de posicionamento de mercado.

**Objetivo Principal:** Analisar posicionamento competitivo, estratégias de preço, tendências de mercado, estimativa de market share e identificação de gaps/oportunidades competitivas.

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `_run()` | Executa análise de inteligência competitiva |
| `_prepare_competitive_data()` | Prepara dados para análise competitiva |
| `_load_market_benchmarks()` | Carrega benchmarks de mercado |
| `_analyze_market_position()` | Analisa posicionamento no mercado |
| `_analyze_competitive_pricing()` | Análise de precificação competitiva |
| `_compare_market_trends()` | Compara tendências com o mercado |
| `_estimate_market_share()` | Estima participação de mercado |
| `_identify_competitive_gaps()` | Identifica gaps competitivos |
| `_analyze_price_positioning()` | Analisa posicionamento de preços |
| `_calculate_growth_metrics()` | Calcula métricas de crescimento |
| `_estimate_local_market_share()` | Estima market share local |
| `_analyze_category_strength()` | Analisa força por categoria |
| `_calculate_price_elasticity_competitive()` | Elasticidade de preços competitiva |
| `_identify_price_mix_opportunities()` | Identifica oportunidades de mix de preços |
| `_analyze_company_seasonality()` | Analisa sazonalidade da empresa |
| `_get_segment_market_percentage()` | Obtém percentual de mercado por segmento |
| `_analyze_competitive_landscape()` | Analisa cenário competitivo |
| `_generate_expansion_recommendations()` | Gera recomendações de expansão |
| `_generate_trend_recommendations()` | Gera recomendações de tendências |
| `_generate_pricing_recommendations()` | Gera recomendações de preços |
| `_generate_market_share_recommendations()` | Recomendações de market share |
| `_identify_pricing_gaps()` | Identifica gaps de precificação |
| `_analyze_digital_gaps()` | Analisa gaps digitais |
| `_create_opportunity_matrix()` | Cria matriz de oportunidades |
| `_generate_gaps_recommendations()` | Gera recomendações para gaps |
| `_format_competitive_result()` | Formata resultado competitivo |

---

## 🤖 Tools de Recomendação e Predição

### 7. Recommendation Engine

**Descrição Breve:** Sistema de recomendações inteligentes otimizado para joalherias e CrewAI.

**Objetivo Principal:** Fornecer recomendações baseadas em ML (Collaborative + Content-Based Filtering), análise RFM, Market Basket Analysis e otimização de preços/inventário/campanhas.

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `_run()` | Executa sistema de recomendações baseado no tipo |
| `_load_and_validate_data()` | Carrega e valida dados para recomendações |
| `_prepare_recommendation_data()` | Prepara dados para algoritmos de recomendação |
| `_calculate_rfm_metrics()` | Calcula métricas RFM para recomendações |
| `_classify_customer_segment()` | Classifica segmento do cliente |
| `_simulate_customer_ids()` | Simula IDs de clientes para recomendações |
| `_execute_recommendation_analysis()` | Executa análise de recomendação |
| `_generate_product_recommendations()` | Gera recomendações de produtos |
| `_generate_customer_targeting()` | Gera targeting de clientes |
| `_generate_pricing_recommendations()` | Gera recomendações de preços |
| `_generate_inventory_recommendations()` | Recomendações de inventário |
| `_generate_marketing_campaigns()` | Gera campanhas de marketing |
| `_generate_strategic_actions()` | Gera ações estratégicas |
| `_filter_by_segment()` | Filtra dados por segmento |
| `_normalize_score()` | Normaliza scores de recomendação |
| `_calculate_recency_score()` | Calcula score de recência |
| `_analyze_product_categories()` | Analisa categorias de produtos |
| `_perform_market_basket_analysis()` | Executa análise de cesta de mercado |
| `_analyze_seasonal_trends()` | Analisa tendências sazonais |
| `_design_campaign_for_segment()` | Projeta campanha por segmento |
| `_estimate_campaign_roi()` | Estima ROI de campanhas |
| `_collaborative_filtering_advanced()` | Filtragem colaborativa avançada |
| `_content_based_filtering_advanced()` | Filtragem baseada em conteúdo |
| `_hybrid_recommendation_system()` | Sistema híbrido de recomendações |
| `_advanced_market_basket_analysis()` | Análise avançada de cesta |
| `_anomaly_detection_customers()` | Detecção de anomalias em clientes |
| `_predictive_customer_lifetime_value()` | CLV preditivo |
| `_select_optimal_algorithm()` | Seleciona algoritmo ótimo |

---

### 8. Prophet Forecast Tool

**Descrição Breve:** Ferramenta de previsão profissional usando Prophet para análise de séries temporais.

**Objetivo Principal:** Criar projeções precisas de vendas considerando tendências, sazonalidade e feriados para planejamento estratégico e gestão de estoque.

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `_run()` | Executa previsão Prophet com parâmetros otimizados |
| `_prepare_data_for_prophet()` | Prepara dados no formato Prophet |
| `_configure_prophet_model()` | Configura modelo Prophet |
| `_create_brazilian_holidays()` | Cria calendário de feriados brasileiros |
| `_calculate_model_metrics()` | Calcula métricas de precisão do modelo |
| `_extract_business_insights()` | Extrai insights de negócio das previsões |
| `_format_predictions()` | Formata previsões para saída |
| `_extract_model_components()` | Extrai componentes do modelo (tendência, sazonalidade) |
| `_generate_business_recommendations()` | Gera recomendações baseadas em previsões |

---

### 9. Risk Assessment Tool

**Descrição Breve:** Ferramenta de avaliação de riscos para identificação e mitigação de riscos empresariais.

**Objetivo Principal:** Avaliar riscos empresariais, financeiros, operacionais, de mercado e de clientes, fornecendo estratégias de mitigação e planos de contingência.

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `_run()` | Executa avaliação de riscos baseada no tipo |
| `_prepare_risk_data()` | Prepara dados para avaliação de riscos |
| `_simulate_customer_ids()` | Simula IDs para análise de risco |
| `_assess_business_risk()` | Avalia riscos gerais do negócio |
| `_assess_financial_risk()` | Avalia riscos financeiros e de liquidez |
| `_assess_operational_risk()` | Avalia riscos operacionais e de processo |
| `_assess_market_risk()` | Avalia riscos de mercado e competição |
| `_assess_customer_risk()` | Avalia riscos relacionados a clientes |
| `_assess_comprehensive_risk()` | Avaliação completa de todos os riscos |
| `_classify_risk_level()` | Classifica nível de risco |
| `_assess_risk_tolerance_fit()` | Avalia adequação à tolerância de risco |
| `_estimate_liquidity_risk()` | Estima risco de liquidez |
| `_calculate_financial_health_indicators()` | Indicadores de saúde financeira |
| `_calculate_revenue_concentration_risk()` | Risco de concentração de receita |
| `_calculate_financial_stability_score()` | Score de estabilidade financeira |
| `_analyze_seasonal_dependency()` | Analisa dependência sazonal |
| `_calculate_operational_efficiency()` | Calcula eficiência operacional |
| `_generate_operational_recommendations()` | Recomendações operacionais |
| `_analyze_category_trends()` | Analisa tendências por categoria |
| `_analyze_competitive_position()` | Analisa posição competitiva |
| `_identify_market_opportunities()` | Identifica oportunidades de mercado |
| `_analyze_customer_lifetime_value()` | Analisa valor vitalício do cliente |
| `_create_risk_matrix()` | Cria matriz de riscos |
| `_create_integrated_mitigation_plan()` | Plano integrado de mitigação |
| `_create_monitoring_plan()` | Plano de monitoramento de riscos |

---

## 🗄️ Tools de Dados e Consultas

### 10. SQL Query Tool

**Descrição Breve:** Ferramenta especializada para extrair dados de vendas do SQL Server com filtros dinâmicos.

**Objetivo Principal:** Executar consultas otimizadas no banco de dados, aplicar filtros de data específicos e retornar dados estruturados prontos para análises.

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `_run()` | Executa consulta SQL com filtros de data |
| `_format_summary()` | Formata resumo dos dados extraídos |
| `_save_to_csv()` | Salva resultados em formato CSV |
| `_execute_query_and_save_to_csv()` | Executa query e salva em CSV |

---

### 11. SQL Query Tool Improved

**Descrição Breve:** Versão melhorada da ferramenta SQL com timeouts, logs detalhados e tratamento robusto de erros.

**Objetivo Principal:** Extrair dados do SQL Server com maior confiabilidade, incluindo timeouts configuráveis, logs de progresso e fallbacks automáticos.

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `test_connection()` | Testa conectividade com o banco antes da query |
| `_build_connection_string()` | Constrói string de conexão segura |
| `_run()` | Executa consulta com timeouts e monitoramento |
| `_progress_monitor()` | Monitor de progresso em tempo real |
| `_execute_query_with_timeout()` | Executa query com timeout configurável |
| `_format_output()` | Formata saída com múltiplos formatos |
| `_format_summary()` | Formata resumo detalhado dos dados |
| `_try_fallback()` | Tenta fallback para dados existentes |

---

### 12. DuckDuckGo Search Tool

**Descrição Breve:** Pesquisa web inteligente para contexto de mercado, tendências e análise competitiva.

**Objetivo Principal:** Buscar informações externas que complementem análises internas, validar hipóteses, entender contexto econômico e identificar tendências de mercado.

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `_run()` | Executa pesquisa web com rate limiting |
| `_extract_key_insights()` | Extrai insights chave dos resultados |
| `_assess_business_relevance()` | Avalia relevância para o negócio |
| `_generate_recommendations()` | Gera recomendações baseadas na pesquisa |

---

## 📁 Tools de Exportação e Geração de Arquivos

**QUANDO USAR ESTAS FERRAMENTAS:**
- 📊 **Customer Data Exporter**: Quando precisar de dados de clientes para CRM, campanhas segmentadas ou análises externas
- 💰 **Financial Data Exporter**: Para relatórios board, dashboards de BI ou análises financeiras em planilhas
- 📦 **Inventory Data Exporter**: Para gestão de compras, identificação de produtos críticos ou otimização de estoque
- 🛍️ **Product Data Exporter**: Para análises de portfólio, classificações ABC/BCG ou planejamento de mix de produtos  
- 📁 **File Generation Tool**: Para criar dashboards específicos, visualizações personalizadas ou arquivos mencionados em relatórios

**INTEGRAÇÃO COM OUTRAS FERRAMENTAS:**
- Use **SQL Query Tool** ANTES para extrair dados atualizados
- Use **KPI Calculator** ou **Business Intelligence** ANTES para identificar que dados exportar
- Use as ferramentas de exportação APÓS análises para disponibilizar dados estruturados
- Use **File Generation Tool** para criar visualizações dos dados exportados

### 13. Customer Data Exporter

**Descrição Breve:** Ferramenta especializada para exportar dados completos de análise de clientes com segmentação RFM, CLV e insights comportamentais.

**Objetivo Principal:** Gerar arquivo CSV abrangente com análise completa de clientes incluindo segmentação RFM detalhada, Customer Lifetime Value (CLV), análise geográfica e demográfica, insights comportamentais, estratégias personalizadas por segmento e scores de saúde do cliente.

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `_run()` | Função principal que executa toda a exportação de dados de clientes |
| `_load_and_prepare_data()` | Carrega e prepara dados básicos de vendas |
| `_estimate_customer_ids()` | Estima IDs de clientes quando não disponíveis nos dados |
| `_aggregate_customer_data()` | Agrega dados por cliente com métricas essenciais |
| `_add_rfm_analysis()` | Adiciona análise RFM completa (Recency, Frequency, Monetary) |
| `_add_clv_calculation()` | Calcula Customer Lifetime Value com projeções configuráveis |
| `_add_geographic_analysis()` | Adiciona análise geográfica e demográfica estimada |
| `_add_behavioral_insights()` | Gera insights comportamentais e padrões de compra |
| `_add_personalized_strategies()` | Define estratégias personalizadas por segmento de cliente |
| `_add_customer_health_scores()` | Calcula scores de saúde e urgência de ação |
| `_export_to_csv()` | Exporta dados estruturados para arquivo CSV |
| `_generate_export_summary()` | Gera resumo completo da exportação |

---

### 14. Financial Data Exporter

**Descrição Breve:** Ferramenta especializada para exportar dados completos de análise financeira com KPIs, margens, tendências e projeções.

**Objetivo Principal:** Criar arquivo CSV estruturado com KPIs financeiros críticos, métricas de margens e rentabilidade, análise de tendências e sazonalidade, projeções financeiras e insights estratégicos por período para relatórios executivos e dashboards de BI.

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `_run()` | Função principal que executa toda a exportação de dados financeiros |
| `_load_and_prepare_data()` | Carrega e prepara dados de vendas para análise financeira |
| `_aggregate_financial_data()` | Agrega dados financeiros por período (diário, semanal, mensal, trimestral) |
| `_add_kpi_analysis()` | Calcula KPIs financeiros críticos com confiança estatística |
| `_add_margin_analysis()` | Analisa margens e rentabilidade por período e categoria |
| `_add_trend_analysis()` | Analisa tendências, sazonalidade e padrões temporais |
| `_add_financial_projections()` | Gera projeções financeiras (30/60/90 dias) |
| `_add_strategic_insights()` | Identifica insights e oportunidades estratégicas |
| `_add_financial_health_scores()` | Calcula scores de saúde financeira por período |
| `_export_to_csv()` | Exporta dados financeiros estruturados para CSV |
| `_generate_export_summary()` | Gera resumo executivo da análise financeira |

---

### 15. Inventory Data Exporter

**Descrição Breve:** Ferramenta especializada para exportar dados completos de gestão de estoque com classificação ABC, análise de riscos e recomendações ML.

**Objetivo Principal:** Gerar arquivo CSV abrangente com classificação ABC baseada em capital investido, análise de giro e turnover de estoque, identificação de riscos (ruptura/obsolescência), recomendações ML para restock e liquidação, e scores de saúde de estoque.

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `_run()` | Função principal que executa toda a exportação de dados de estoque |
| `_load_and_prepare_data()` | Carrega e prepara dados focados em análise de estoque |
| `_aggregate_inventory_data()` | Agrega dados por produto com métricas de estoque |
| `_add_abc_capital_classification()` | Aplica classificação ABC baseada em capital investido |
| `_add_turnover_metrics()` | Calcula métricas de giro, turnover e cobertura de estoque |
| `_add_risk_analysis()` | Analisa riscos de ruptura, obsolescência e baixo giro |
| `_add_ml_recommendations()` | Gera recomendações ML para restock, liquidação e promoções |
| `_add_health_scores()` | Calcula scores de saúde, cobertura e urgência de ação |
| `_export_to_csv()` | Exporta dados de estoque estruturados para CSV |
| `_generate_export_summary()` | Gera resumo executivo da gestão de estoque |

---

### 16. Product Data Exporter

**Descrição Breve:** Ferramenta especializada para exportar dados completos de produtos com classificações ABC, BCG Matrix e análise de ciclo de vida.

**Objetivo Principal:** Criar arquivo CSV estruturado com classificação ABC automática, Matriz BCG (Stars, Cash Cows, Question Marks, Dogs), análise de ciclo de vida, métricas de performance e flags de alertas para tomada de decisão.

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `_run()` | Função principal que executa toda a exportação de dados de produtos |
| `_load_and_prepare_data()` | Carrega e prepara dados básicos para análise de produtos |
| `_aggregate_product_data()` | Agrega dados por produto com métricas de performance |
| `_add_abc_classification()` | Aplica classificação ABC baseada em receita e volume |
| `_add_bcg_classification()` | Aplica Matriz BCG (crescimento vs participação de mercado) |
| `_add_lifecycle_analysis()` | Analisa ciclo de vida identificando slow movers e dead stock |
| `_add_advanced_metrics()` | Calcula métricas avançadas de performance e competitividade |
| `_add_alert_flags()` | Adiciona flags de alertas para ação (restock, liquidação, promoção) |
| `_export_to_csv()` | Exporta dados de produtos estruturados para CSV |
| `_generate_export_summary()` | Gera resumo executivo da análise de produtos |

---

### 17. File Generation Tool

**Descrição Breve:** Ferramenta especializada para gerar arquivos específicos mencionados nos relatórios, incluindo dashboards HTML interativos e planilhas CSV processadas.

**Objetivo Principal:** Criar arquivos específicos sob demanda como dashboards HTML interativos, planilhas CSV com dados processados, mapas geográficos, visualizações e matrizes de análise ML para complementar relatórios e análises.

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `_run()` | Função principal que roteia para métodos específicos de geração |
| `_load_data()` | Carrega e prepara dados básicos para geração de arquivos |
| `_create_customer_rfm_dashboard()` | Cria dashboard RFM interativo em HTML com Plotly |
| `_create_customer_clusters_csv()` | Gera matriz de clusters ML em formato CSV |
| `_create_geographic_heatmap()` | Cria mapa interativo de distribuição geográfica |
| `_create_product_abc_dashboard()` | Gera dashboard ABC de produtos em HTML |
| `_create_market_basket_matrix()` | Cria matriz de market basket analysis em HTML |
| `_create_financial_dashboard()` | Gera dashboard financeiro executivo interativo |
| `_create_sales_team_dashboard()` | Cria dashboard de performance de equipe de vendas |
| `_create_inventory_recommendations_csv()` | Gera planilha de recomendações ML de estoque |

---

## 🔧 Módulos Compartilhados

### 18. Data Preparation Mixin

**Descrição Breve:** Módulo compartilhado para preparação e limpeza de dados de joalherias.

**Objetivo Principal:** Padronizar preparação de dados, validação de estrutura, limpeza, conversão de tipos e cálculo de campos derivados.

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `prepare_jewelry_data()` | Prepara dados de joalheria com validação |
| `_validate_data_structure()` | Valida estrutura dos dados |
| `_clean_basic_data()` | Limpeza básica de dados |
| `_convert_data_types()` | Converte tipos de dados |
| `_calculate_financial_derived_fields()` | Calcula campos financeiros derivados |
| `_calculate_demographic_derived_fields()` | Calcula campos demográficos derivados |
| `_calculate_temporal_derived_fields()` | Calcula campos temporais derivados |
| `_calculate_inventory_derived_fields()` | Calcula campos de estoque derivados |
| `_calculate_product_derived_fields()` | Calcula campos de produto derivados |
| `_final_data_validation()` | Validação final dos dados |
| `get_data_quality_report()` | Relatório de qualidade dos dados |

---

### 19. Report Formatter Mixin

**Descrição Breve:** Módulo compartilhado para formatação padronizada de relatórios e saídas.

**Objetivo Principal:** Padronizar formatação de relatórios, KPIs, tabelas e insights em formato consistente e profissional.

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `format_business_kpi_report()` | Formata relatório de KPIs de negócio |
| `format_statistical_analysis_report()` | Formata relatório de análise estatística |
| `format_data_table()` | Formata tabelas de dados |
| `format_insights_section()` | Formata seção de insights |
| `format_alerts_section()` | Formata seção de alertas |
| `format_numeric_value()` | Formata valores numéricos |
| `_get_kpi_section_title()` | Obtém título de seção KPI |
| `_format_kpi_section_data()` | Formata dados de seção KPI |
| `_format_kpi_value()` | Formata valor individual de KPI |
| `_format_business_specific_analysis()` | Formatação específica de negócio |
| `_format_nested_dict()` | Formata dicionários aninhados |
| `_format_dict_as_table()` | Formata dicionário como tabela |

---

### 20. Business Mixins

**Descrição Breve:** Conjunto de mixins especializados em análises de negócio para joalherias.

**Objetivo Principal:** Fornecer análises especializadas RFM, BCG Matrix, ABC Analysis e benchmarks específicos do setor joalheiro.

**Componentes:**

#### JewelryRFMAnalysisMixin
| Função | Descrição |
|--------|-----------|
| `analyze_product_rfm()` | Análise RFM de produtos |
| `analyze_customer_rfm()` | Análise RFM de clientes |
| `_categorize_jewelry_product_rfm()` | Categoriza produtos por RFM |
| `_categorize_jewelry_customer_rfm()` | Categoriza clientes por RFM |
| `_generate_jewelry_rfm_recommendations()` | Recomendações RFM para joalherias |
| `_create_customer_segment_profiles()` | Cria perfis de segmentos |

#### JewelryBusinessAnalysisMixin
| Função | Descrição |
|--------|-----------|
| `create_product_bcg_matrix()` | Cria matriz BCG de produtos |
| `perform_abc_analysis()` | Executa análise ABC |
| `_generate_bcg_recommendations()` | Recomendações BCG |
| `_generate_abc_recommendations()` | Recomendações ABC |

#### JewelryBenchmarkMixin
| Função | Descrição |
|--------|-----------|
| `get_jewelry_industry_benchmarks()` | Benchmarks da indústria joalheira |
| `compare_with_benchmarks()` | Compara com benchmarks |
| `_assess_overall_benchmark_performance()` | Avalia performance vs benchmarks |
| `_generate_benchmark_recommendations()` | Recomendações baseadas em benchmarks |

---

### 21. Performance Optimizations

**Descrição Breve:** Módulo de otimizações de performance para análises pesadas e processamento eficiente.

**Objetivo Principal:** Fornecer cache inteligente, processamento paralelo, amostragem estratificada e detecção de drift para otimizar performance das análises.

**Componentes:**

#### CacheManager
| Função | Descrição |
|--------|-----------|
| `get_cached_result()` | Obtém resultado do cache |
| `save_result_to_cache()` | Salva resultado no cache |
| `clear()` | Limpa cache |
| `get_cache_stats()` | Estatísticas do cache |
| `_cleanup_cache_if_needed()` | Limpeza automática do cache |

#### ParallelProcessor
| Função | Descrição |
|--------|-----------|
| `parallel_model_training()` | Treinamento paralelo de modelos |
| `parallel_category_analysis()` | Análise paralela por categoria |

#### StratifiedSampler
| Função | Descrição |
|--------|-----------|
| `should_sample()` | Verifica se deve fazer amostragem |
| `create_stratified_sample()` | Cria amostra estratificada |
| `_temporal_stratified_sample()` | Amostra estratificada temporal |
| `_value_stratified_sample()` | Amostra estratificada por valor |
| `_product_stratified_sample()` | Amostra estratificada por produto |
| `_validate_sample_representativeness()` | Valida representatividade da amostra |

#### DataDriftDetector
| Função | Descrição |
|--------|-----------|
| `detect_drift()` | Detecta drift nos dados |
| `_detect_temporal_drift()` | Detecta drift temporal |
| `_detect_distribution_drift()` | Detecta drift de distribuição |
| `_detect_business_pattern_drift()` | Detecta drift de padrões de negócio |
| `_generate_drift_recommendations()` | Recomendações para drift |

---

## 📈 Resumo de Capacidades

### 🎯 Análises Principais
- **17 Tools Especializadas** para diferentes aspectos do negócio
- **Machine Learning Avançado** com Random Forest, XGBoost, clustering
- **Análises Estatísticas Rigorosas** com testes de significância
- **Inteligência Competitiva** com benchmarks de mercado
- **Previsões Temporais** com Prophet e algoritmos adaptativos
- **Exportação Completa de Dados** com 4 exportadores especializados
- **Geração Automática de Arquivos** com dashboards e visualizações

### 🔧 Infraestrutura Robusta
- **Módulos Compartilhados** para reutilização e consistência
- **Cache Inteligente** para otimização de performance
- **Processamento Paralelo** para análises pesadas
- **Validação Automática** de qualidade dos dados
- **Formatação Padronizada** de relatórios

### 📊 Outputs Profissionais
- **Relatórios Executivos** em HTML e JSON
- **Dashboards Interativos** com Plotly
- **Insights Acionáveis** baseados em evidências
- **Recomendações Estratégicas** automatizadas
- **Alertas Inteligentes** para problemas críticos
- **Arquivos CSV Estruturados** para análises externas
- **Dashboards HTML Personalizados** com visualizações avançadas
- **Matrizes de Análise ML** exportáveis
- **Mapas Geográficos Interativos** para distribuição regional

---

*Documentação gerada automaticamente para o sistema Insights AI v4.1 - Atualizada com Tools de Exportação e Geração de Arquivos* 