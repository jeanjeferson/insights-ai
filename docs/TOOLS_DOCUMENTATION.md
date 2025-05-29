# 🚀 GUIA COMPLETO DE TOOLS - INSIGHTS AI
*Documentação para implementação em novos projetos CrewAI*

## 📋 ÍNDICE ESTRUTURADO

### 🎯 [GUIA RÁPIDO DE IMPLEMENTAÇÃO](#guia-rápido-de-implementação)
### 🔗 [MAPEAMENTO AGENTE → TOOLS](#mapeamento-agente--tools)  
### 🛠️ [CATEGORIAS DE TOOLS](#categorias-de-tools)
### 📝 [WORKFLOWS E INTEGRAÇÕES](#workflows-e-integrações)
### ⚡ [TROUBLESHOOTING](#troubleshooting)

---

## 🎯 GUIA RÁPIDO DE IMPLEMENTAÇÃO

### **Para Novos Projetos CrewAI:**

1. **📊 Analytics Básico** → Usar: Statistical Analysis + KPI Calculator + Business Intelligence
2. **🤖 ML Avançado** → Usar: Advanced Analytics Engine + Customer Insights + Recommendation Engine
3. **💼 Business Intelligence** → Usar: Business Intelligence + Competitive Intelligence + Risk Assessment
4. **📈 Forecasting** → Usar: Prophet + Advanced Analytics + Statistical Analysis
5. **🗄️ Gestão de Dados** → Usar: SQL Tools + Exporters + File Generation

### **Templates de Agentes Recomendados:**

```yaml
# AGENTE ANALISTA BÁSICO
analista_basico:
  tools: [StatisticalAnalysisTool, KPICalculatorTool, BusinessIntelligenceTool]
  complexity: Básico
  use_cases: [KPIs, relatórios executivos, análises estatísticas simples]

# AGENTE ML ESPECIALISTA  
ml_especialista:
  tools: [AdvancedAnalyticsEngineTool, CustomerInsightsEngine, RecommendationEngine]
  complexity: Avançado
  use_cases: [ML insights, segmentação avançada, recomendações]

# AGENTE ESTRATÉGICO
estrategista:
  tools: [CompetitiveIntelligenceTool, RiskAssessmentTool, ProphetForecastTool]
  complexity: Intermediário
  use_cases: [análise competitiva, gestão de riscos, forecasting]
```

---

## 🔗 MAPEAMENTO AGENTE → TOOLS

### **🔧 ENGENHEIRO DE DADOS**
**Tools Principais:** SQL Query Tool, Statistical Analysis Tool, Advanced Analytics Engine
**Complexidade:** Intermediário
**Casos de Uso:** ETL, qualidade de dados, validações, transformações

### **📈 ANALISTA DE TENDÊNCIAS** 
**Tools Principais:** Statistical Analysis Tool, DuckDuckGo Search Tool, Prophet Forecast Tool
**Complexidade:** Intermediário  
**Casos de Uso:** Análise temporal, correlações, tendências de mercado

### **🌊 ESPECIALISTA EM SAZONALIDADE**
**Tools Principais:** Statistical Analysis Tool, Advanced Analytics Engine, Prophet Forecast Tool
**Complexidade:** Avançado
**Casos de Uso:** Decomposição STL, modelagem sazonal, eventos especiais

### **🎯 ANALISTA DE CLIENTES**
**Tools Principais:** Customer Insights Engine, Recommendation Engine, Statistical Analysis Tool  
**Complexidade:** Avançado
**Casos de Uso:** Segmentação RFM, churn prediction, CLV, jornada do cliente

### **💰 ANALISTA FINANCEIRO**
**Tools Principais:** Business Intelligence Tool, KPI Calculator Tool, Risk Assessment Tool
**Complexidade:** Intermediário
**Casos de Uso:** KPIs financeiros, rentabilidade, análise de riscos

### **📦 ESPECIALISTA EM PRODUTOS**
**Tools Principais:** Advanced Analytics Engine, Statistical Analysis Tool, Recommendation Engine
**Complexidade:** Intermediário  
**Casos de Uso:** Análise ABC, BCG matrix, performance de produtos

### **🏆 ANALISTA COMPETITIVO**
**Tools Principais:** Competitive Intelligence Tool, DuckDuckGo Search Tool, Statistical Analysis Tool
**Complexidade:** Avançado
**Casos de Uso:** Benchmarking, market share, análise competitiva

### **📊 BUSINESS INTELLIGENCE**
**Tools Principais:** Business Intelligence Tool, KPI Calculator Tool, File Generation Tool
**Complexidade:** Básico
**Casos de Uso:** Dashboards, relatórios executivos, exportações

---

## 🛠️ CATEGORIAS DE TOOLS

### 🔬 **ANALYTICS & MACHINE LEARNING**
*Complexidade: Avançada | Tempo: 30-120s*

#### **Advanced Analytics Engine Tool**
- **Função:** Motor ML com Random Forest, XGBoost, clustering avançado
- **Casos de Uso:** Insights ocultos, anomaly detection, demand forecasting, customer behavior
- **Inputs Principais:** `analysis_type`, `data_csv`, `target_column`, `model_complexity`
- **Outputs:** Insights ML, recomendações baseadas em evidências, métricas de performance

#### **Statistical Analysis Tool** 
- **Função:** Análises estatísticas rigorosas com testes de significância
- **Casos de Uso:** Correlações, clustering, outliers, distribuições, tendências temporais
- **Inputs Principais:** `analysis_type`, `data_csv`, `statistical_tests`, `confidence_level`
- **Outputs:** Testes estatísticos, clustering, insights baseados em significância

#### **Customer Insights Engine**
- **Função:** Segmentação avançada, RFM, lifecycle, churn prediction
- **Casos de Uso:** Behavioral segmentation, lifecycle analysis, churn prediction, value analysis
- **Inputs Principais:** `analysis_type`, `segmentation_method`, `customer_id_column`
- **Outputs:** Segmentos de clientes, scores de risco, estratégias de retenção

---

### 🎯 **BUSINESS INTELLIGENCE**
*Complexidade: Básica a Intermediária | Tempo: 15-60s*

#### **Business Intelligence Tool**
- **Função:** Plataforma unificada de BI com dashboards interativos
- **Casos de Uso:** Executive summary, dashboards, análises financeiras, customer intelligence
- **Inputs Principais:** `analysis_type`, `time_period`, `output_format`, `include_forecasts`
- **Outputs:** Relatórios HTML, dashboards Plotly, análises executivas

#### **KPI Calculator Tool**
- **Função:** Calculadora de KPIs com alertas automáticos e benchmarks
- **Casos de Uso:** Monitoramento KPIs, benchmarking, alertas de problemas críticos
- **Inputs Principais:** `categoria`, `periodo`, `benchmark_mode`, `alert_threshold`
- **Outputs:** KPIs categorizados, scores de saúde, alertas inteligentes

#### **Competitive Intelligence Tool**
- **Função:** Inteligência competitiva e análise de market share
- **Casos de Uso:** Market positioning, pricing analysis, trend comparison, competitive gaps
- **Inputs Principais:** `analysis_type`, `market_segment`, `benchmark_period`
- **Outputs:** Posicionamento competitivo, market share estimado, recomendações estratégicas

---

### 🤖 **FORECASTING & PREDICTION**
*Complexidade: Intermediária a Avançada | Tempo: 20-90s*

#### **Prophet Forecast Tool**
- **Função:** Previsões profissionais com Prophet considerando sazonalidade
- **Casos de Uso:** Projeções de vendas, demand forecasting, planejamento estratégico
- **Inputs Principais:** `target_column`, `periods`, `seasonality_mode`, `include_holidays`
- **Outputs:** Previsões com intervalos de confiança, decomposição sazonal

#### **Risk Assessment Tool**
- **Função:** Avaliação de riscos empresariais com estratégias de mitigação
- **Casos de Uso:** Business risk, financial risk, operational risk, market risk
- **Inputs Principais:** `assessment_type`, `risk_tolerance`, `time_horizon`
- **Outputs:** Matriz de riscos, scores de risco, planos de mitigação

#### **Recommendation Engine**
- **Função:** Sistema ML de recomendações (Collaborative + Content-Based)
- **Casos de Uso:** Product recommendations, customer targeting, pricing optimization
- **Inputs Principais:** `recommendation_type`, `target_segment`, `confidence_threshold`
- **Outputs:** Recomendações rankeadas, ROI estimado, campanhas personalizadas

---

### 🗄️ **DATA MANAGEMENT**
*Complexidade: Básica | Tempo: 5-30s*

#### **SQL Query Tool / SQL Query Tool Improved**
- **Função:** Extração de dados do SQL Server com filtros temporais
- **Casos de Uso:** ETL, extração de dados filtrados, alimentação de outras tools
- **Inputs Principais:** `date_start`, `date_end`, `output_format`
- **Outputs:** Dados estruturados CSV/JSON, validações de integridade

#### **DuckDuckGo Search Tool**
- **Função:** Pesquisa web para contexto externo e tendências
- **Casos de Uso:** Trends de mercado, análise competitiva, contexto econômico
- **Inputs Principais:** `query`, `domain`, `max_results`
- **Outputs:** Insights contextualizados, links para fontes, recomendações

---

### 📊 **EXPORT & REPORTING**
*Complexidade: Básica | Tempo: 10-30s*

#### **Customer Data Exporter**
- **Função:** Exporta dados completos de análise de clientes
- **Outputs:** CSV com RFM, CLV, segmentação, análise geográfica/demográfica

#### **Financial Data Exporter**  
- **Função:** Exporta dados financeiros com KPIs e projeções
- **Outputs:** CSV com KPIs financeiros, análise de margens, projeções

#### **Inventory Data Exporter**
- **Função:** Exporta dados de gestão de estoque
- **Outputs:** CSV com classificação ABC, análise de giro, recomendações ML

#### **Product Data Exporter**
- **Função:** Exporta dados de produtos com classificações
- **Outputs:** CSV com ABC, BCG matrix, lifecycle analysis, métricas de performance

#### **File Generation Tool**
- **Função:** Gera arquivos específicos (dashboards HTML, matrizes CSV)
- **Casos de Uso:** Customer RFM dashboard, geographic heatmap, product ABC dashboard

---

### 🔧 **SHARED INFRASTRUCTURE**
*Módulos compartilhados para otimização*

#### **Data Preparation Mixin**
- **Função:** Limpeza e validação de dados padronizada
- **Uso:** Inherited por todas as tools que processam dados

#### **Report Formatter Mixin**
- **Função:** Formatação unificada de relatórios
- **Uso:** Formatting consistente entre todas as tools

#### **Business Mixins**
- **Função:** Análises de negócio padronizadas (RFM, ABC, BCG)
- **Uso:** Análises consistency entre diferentes tools

#### **Performance Optimizations**
- **Função:** Cache, parallel processing, sampling estratificado
- **Uso:** Otimização automática para datasets grandes

---

## 📝 WORKFLOWS E INTEGRAÇÕES

### **🏆 WORKFLOW BÁSICO DE ANÁLISE**
```
1. SQL Query Tool → Extrair dados filtrados
2. Statistical Analysis Tool → Análise exploratória 
3. KPI Calculator Tool → KPIs essenciais
4. Business Intelligence Tool → Relatório executivo
5. Customer Data Exporter → Dados estruturados para BI externo
```

### **🤖 WORKFLOW AVANÇADO DE ML**
```
1. SQL Query Tool → Dados filtrados
2. Advanced Analytics Engine → ML insights + anomaly detection  
3. Customer Insights Engine → Segmentação avançada
4. Recommendation Engine → Recomendações personalizadas
5. Prophet Forecast Tool → Previsões
6. Financial Data Exporter → Dados consolidados
```

### **🏆 WORKFLOW COMPETITIVO**
```
1. SQL Query Tool → Dados internos
2. DuckDuckGo Search Tool → Contexto de mercado
3. Competitive Intelligence Tool → Análise competitiva
4. Statistical Analysis Tool → Validação estatística
5. Risk Assessment Tool → Avaliação de riscos estratégicos
```

### **📊 INTEGRAÇÃO ENTRE TOOLS**

**Sequência Recomendada:**
- **Dados → Análise → Insights → Ação**
- SQL Tools primeiro (dados)
- Statistical/Analytics depois (análise)
- BI/KPI para insights
- Exporters para ação

**Dependências Críticas:**
- Todas as tools de análise dependem de dados limpos (SQL Tools)
- Tools avançadas podem usar outputs de tools básicas
- Exporters são sempre o último passo no workflow

---

## ⚡ TROUBLESHOOTING

### **🚨 PROBLEMAS COMUNS**

#### **"Dados Insuficientes"**
- **Causa:** Filtro temporal muito restritivo
- **Solução:** Ampliar range de datas no SQL Query Tool
- **Prevenção:** Verificar volume de dados antes de análises avançadas

#### **"Timeout na Query SQL"** 
- **Causa:** Query muito pesada ou conexão instável
- **Solução:** Usar SQL Query Tool Improved com timeouts configuráveis
- **Prevenção:** Filtrar por períodos menores

#### **"Erro de Validação de Schema"**
- **Causa:** Parâmetros obrigatórios ausentes ou formato inválido
- **Solução:** Verificar documentation específica da tool
- **Prevenção:** Usar templates de inputs recomendados

#### **"Performance Lenta"**
- **Causa:** Dataset muito grande sem sampling
- **Solução:** Habilitar cache_results=True e sampling automático
- **Prevenção:** Monitorar tamanho dos datasets

### **✅ BEST PRACTICES**

1. **Sempre começar com SQL Query Tool** para dados filtrados
2. **Usar cache_results=True** em análises repetitivas  
3. **Validar inputs** antes de executar tools complexas
4. **Combinar tools complementares** (ex: Statistical + KPI)
5. **Exportar resultados** para análises externas quando necessário

### **📊 MONITORAMENTO DE PERFORMANCE**

**Tempos Esperados:**
- SQL Tools: 5-30s
- Statistical/KPI: 15-60s  
- Advanced Analytics: 30-120s
- Exporters: 10-30s

**Indicadores de Problema:**
- Tempo > 3x do esperado
- Errors de memory/timeout
- Resultados vazios ou inconsistentes

---

## 📚 REFERÊNCIAS PARA IMPLEMENTAÇÃO

### **Templates de YAML para Agentes:**
```yaml
analista_exemplo:
  role: "Analista de Dados Especializado"
  goal: "Realizar análises estatísticas e gerar insights acionáveis"
  backstory: "Especialista em análise de dados com foco em insights de negócio"
  tools: [StatisticalAnalysisTool, KPICalculatorTool, BusinessIntelligenceTool]
  verbose: true
  memory: true
```

### **Templates de Tasks:**
```yaml
analise_exemplo_task:
  description: "Realizar análise completa usando {data_inicio} e {data_fim}"
  expected_output: "Relatório estruturado com insights e recomendações"
  agent: analista_exemplo
```

### **Checklist de Implementação:**
- [ ] Definir agentes e suas tools especializadas
- [ ] Configurar workflows de integração entre tools
- [ ] Testar com dados de amostra
- [ ] Validar outputs esperados
- [ ] Configurar error handling
- [ ] Documentar casos de uso específicos

**Esta documentação serve como guia completo para implementação das Tools do Insights AI em novos projetos CrewAI, garantindo uso eficiente e resultados consistentes.** 

---

## 🎯 CASOS DE USO POR SETOR

### **📈 E-COMMERCE / VAREJO**
- **Tools Principais:** Customer Insights, Recommendation Engine, Business Intelligence
- **Foco:** Segmentação de clientes, recomendações de produtos, análise de conversão

### **💎 JOALHERIAS** 
- **Tools Principais:** Competitive Intelligence, Statistical Analysis, Prophet Forecast
- **Foco:** Sazonalidade, análise de luxo vs. premium, eventos especiais

### **🏪 RETAIL FÍSICO**
- **Tools Principais:** Geographic Analysis, Inventory Management, Risk Assessment  
- **Foco:** Análise regional, gestão de estoque, riscos operacionais

### **💰 SERVIÇOS FINANCEIROS**
- **Tools Principais:** Risk Assessment, Advanced Analytics, KPI Calculator
- **Foco:** Gestão de riscos, compliance, métricas financeiras

---

## 🔧 REFERÊNCIA TÉCNICA AVANÇADA - TODAS AS FUNÇÕES

### 🔬 **ADVANCED ANALYTICS ENGINE TOOL**

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

### 📊 **STATISTICAL ANALYSIS TOOL**

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

### 🎯 **CUSTOMER INSIGHTS ENGINE**

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

### 📈 **BUSINESS INTELLIGENCE TOOL**

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

### ⚡ **KPI CALCULATOR TOOL**

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

### 🏆 **COMPETITIVE INTELLIGENCE TOOL**

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

### 🤖 **RECOMMENDATION ENGINE**

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

### 🔮 **PROPHET FORECAST TOOL**

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

### ⚠️ **RISK ASSESSMENT TOOL**

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

### 🗄️ **SQL QUERY TOOL**

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `_run()` | Executa consulta SQL com filtros de data |
| `_format_summary()` | Formata resumo dos dados extraídos |
| `_save_to_csv()` | Salva resultados em formato CSV |
| `_execute_query_and_save_to_csv()` | Executa query e salva em CSV |

---

### 🗄️ **SQL QUERY TOOL IMPROVED**

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

### 🌐 **DUCKDUCKGO SEARCH TOOL**

**Funções Principais:**

| Função | Descrição |
|--------|-----------|
| `_run()` | Executa pesquisa web com rate limiting |
| `_extract_key_insights()` | Extrai insights chave dos resultados |
| `_assess_business_relevance()` | Avalia relevância para o negócio |
| `_generate_recommendations()` | Gera recomendações baseadas na pesquisa |

---

### 📊 **CUSTOMER DATA EXPORTER**

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

### 💰 **FINANCIAL DATA EXPORTER**

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

### 📦 **INVENTORY DATA EXPORTER**

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

### 🛍️ **PRODUCT DATA EXPORTER**

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

### 📁 **FILE GENERATION TOOL**

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

## 🔧 **MÓDULOS COMPARTILHADOS - FUNÇÕES DETALHADAS**

### **Data Preparation Mixin**

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

### **Report Formatter Mixin**

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

### **Business Mixins - JewelryRFMAnalysisMixin**

| Função | Descrição |
|--------|-----------|
| `analyze_product_rfm()` | Análise RFM de produtos |
| `analyze_customer_rfm()` | Análise RFM de clientes |
| `_categorize_jewelry_product_rfm()` | Categoriza produtos por RFM |
| `_categorize_jewelry_customer_rfm()` | Categoriza clientes por RFM |
| `_generate_jewelry_rfm_recommendations()` | Recomendações RFM para joalherias |
| `_create_customer_segment_profiles()` | Cria perfis de segmentos |

### **Business Mixins - JewelryBusinessAnalysisMixin**

| Função | Descrição |
|--------|-----------|
| `create_product_bcg_matrix()` | Cria matriz BCG de produtos |
| `perform_abc_analysis()` | Executa análise ABC |
| `_generate_bcg_recommendations()` | Recomendações BCG |
| `_generate_abc_recommendations()` | Recomendações ABC |

### **Business Mixins - JewelryBenchmarkMixin**

| Função | Descrição |
|--------|-----------|
| `get_jewelry_industry_benchmarks()` | Benchmarks da indústria joalheira |
| `compare_with_benchmarks()` | Compara com benchmarks |
| `_assess_overall_benchmark_performance()` | Avalia performance vs benchmarks |
| `_generate_benchmark_recommendations()` | Recomendações baseadas em benchmarks |

### **Performance Optimizations - CacheManager**

| Função | Descrição |
|--------|-----------|
| `get_cached_result()` | Obtém resultado do cache |
| `save_result_to_cache()` | Salva resultado no cache |
| `clear()` | Limpa cache |
| `get_cache_stats()` | Estatísticas do cache |
| `_cleanup_cache_if_needed()` | Limpeza automática do cache |

### **Performance Optimizations - ParallelProcessor**

| Função | Descrição |
|--------|-----------|
| `parallel_model_training()` | Treinamento paralelo de modelos |
| `parallel_category_analysis()` | Análise paralela por categoria |

### **Performance Optimizations - StratifiedSampler**

| Função | Descrição |
|--------|-----------|
| `should_sample()` | Verifica se deve fazer amostragem |
| `create_stratified_sample()` | Cria amostra estratificada |
| `_temporal_stratified_sample()` | Amostra estratificada temporal |
| `_value_stratified_sample()` | Amostra estratificada por valor |
| `_product_stratified_sample()` | Amostra estratificada por produto |
| `_validate_sample_representativeness()` | Valida representatividade da amostra |

### **Performance Optimizations - DataDriftDetector**

| Função | Descrição |
|--------|-----------|
| `detect_drift()` | Detecta drift nos dados |
| `_detect_temporal_drift()` | Detecta drift temporal |
| `_detect_distribution_drift()` | Detecta drift de distribuição |
| `_detect_business_pattern_drift()` | Detecta drift de padrões de negócio |
| `_generate_drift_recommendations()` | Recomendações para drift |

---

**Esta documentação completa fornece todas as informações necessárias para implementar as Tools do Insights AI em qualquer projeto CrewAI, com exemplos práticos, workflows de integração e troubleshooting.** 