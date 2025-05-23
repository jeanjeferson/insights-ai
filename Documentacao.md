# 📚 DOCUMENTAÇÃO COMPLETA - FERRAMENTAS INSIGHTS AI

## 🎯 Visão Geral

Este documento apresenta a documentação detalhada de todas as ferramentas disponíveis no sistema **Insights AI** para análise de dados de joalherias. As ferramentas estão organizadas em categorias funcionais e oferecem análises especializadas para diferentes aspectos do negócio.

---

## 📋 Índice

### 🔧 FERRAMENTAS PRINCIPAIS
1. [SQL Query Tool](#sql-query-tool)
2. [Prophet Forecast Tool](#prophet-forecast-tool)
3. [KPI Calculator Tool](#kpi-calculator-tool)
4. [Statistical Analysis Tool](#statistical-analysis-tool)
5. [Advanced Visualization Tool](#advanced-visualization-tool)
6. [DuckDuckGo Search Tool](#duckduckgo-search-tool)
7. [Custom Tool](#custom-tool)

### 🚀 FERRAMENTAS AVANÇADAS
8. [Customer Insights Engine](#customer-insights-engine)
9. [Risk Assessment Tool](#risk-assessment-tool)
10. [Recommendation Engine](#recommendation-engine)
11. [Advanced Analytics Engine](#advanced-analytics-engine)
12. [Business Intelligence Dashboard](#business-intelligence-dashboard)
13. [Competitive Intelligence Tool](#competitive-intelligence-tool)

---

## 🔧 FERRAMENTAS PRINCIPAIS

### SQL Query Tool

**Descrição**: Ferramenta para execução de consultas SQL personalizadas no banco de dados de vendas.

#### 📥 Parâmetros de Entrada
```python
{
    "data_inicio": "2024-01-01",        # Data inicial (YYYY-MM-DD)
    "data_fim": "2024-12-31",           # Data final (YYYY-MM-DD)
    "filtro_grupo": "",                 # Filtro por grupo de produto (opcional)
    "filtro_cliente": "",               # Filtro por cliente (opcional)
    "incluir_vendedor": True            # Incluir dados do vendedor
}
```

#### 🎯 Funcionalidades
- Execução de consultas SQL otimizadas
- Filtros temporais flexíveis
- Filtros por produto, cliente e vendedor
- Agregação automática de dados
- Validação de parâmetros

#### 📤 Outputs
- **Formato**: DataFrame/CSV com colunas:
  - Data, Ano, Mes
  - Codigo_Cliente, Nome_Cliente, dados demográficos
  - Codigo_Vendedor, Nome_Vendedor
  - Codigo_Produto, Descricao_Produto, hierarquia de produtos
  - Estoque_Atual, Custo_Produto, Preco_Tabela
  - Desconto_Aplicado, Total_Liquido, Quantidade

#### 💡 Casos de Uso
- Extração de dados para análises específicas
- Preparação de datasets personalizados
- Análises ad-hoc de vendas
- Relatórios customizados

---

### Prophet Forecast Tool

**Descrição**: Ferramenta de previsão de séries temporais usando o modelo Prophet do Facebook.

#### 📥 Parâmetros de Entrada
```python
{
    "data": "string_json_dataframe",     # JSON string do DataFrame
    "data_column": "Data",               # Nome da coluna de datas
    "target_column": "Total_Liquido",    # Coluna a ser prevista
    "periods": 15,                       # Períodos futuros a prever
    "include_history": True,             # Incluir dados históricos
    "seasonality_mode": "multiplicative", # Modo de sazonalidade
    "daily_seasonality": True,           # Sazonalidade diária
    "weekly_seasonality": True,          # Sazonalidade semanal
    "yearly_seasonality": True           # Sazonalidade anual
}
```

#### 🎯 Funcionalidades
- Previsões automáticas com detecção de sazonalidade
- Múltiplos cenários (base, conservador, otimista)
- Análise de componentes (tendência, sazonalidade)
- Intervalos de confiança
- Visualizações interativas

#### 📤 Outputs
```json
{
    "forecast_data": "JSON com previsões",
    "plot": "base64_image_forecast",
    "components_plot": "base64_image_components",
    "model_params": "parâmetros_utilizados",
    "business_impact": "impacto_no_negócio"
}
```

#### 💡 Casos de Uso
- Planejamento de vendas futuras
- Previsão de demanda
- Análise de tendências
- Planejamento de estoque
- Orçamento anual

---

### KPI Calculator Tool

**Descrição**: Calcula 30+ KPIs críticos especializados para joalherias.

#### 📥 Parâmetros de Entrada
```python
{
    "data_csv": "data/vendas.csv",    # Caminho do arquivo CSV
    "categoria": "all",               # Categoria de KPIs
    "periodo": "monthly",             # Período de análise
    "benchmark_mode": True            # Incluir benchmarks do setor
}
```

**Categorias Disponíveis**:
- `all`: Todos os KPIs
- `revenue`: Financeiros
- `operational`: Operacionais
- `inventory`: Inventário
- `customer`: Clientes

#### 🎯 Funcionalidades
- **KPIs Financeiros**: Revenue Growth, AOV, Margem por categoria, ROI
- **KPIs Operacionais**: Giro de estoque, Velocidade de vendas, Concentração
- **KPIs de Clientes**: CLV estimado, Repeat purchase rate, Segmentação
- **KPIs de Inventário**: Days sales inventory, ABC analysis, Performance por produto
- **Benchmarks**: Comparação com padrões do setor de joalherias

#### 📤 Outputs
```json
{
    "financeiros": {
        "total_revenue": 150000.50,
        "mom_growth_rate": 5.2,
        "aov": 850.30,
        "revenue_by_category": {...},
        "top_products_revenue": {...}
    },
    "operacionais": {
        "produtos_ativos": 450,
        "sales_velocity_daily": 12.5,
        "concentration_80_20_pct": 75.3
    },
    "insights": ["Lista de insights automáticos"],
    "alertas": ["Alertas importantes"]
}
```

#### 💡 Casos de Uso
- Dashboards executivos
- Relatórios de performance
- Benchmarking setorial
- Monitoramento de saúde do negócio
- Identificação de oportunidades

---

### Statistical Analysis Tool

**Descrição**: Análises estatísticas avançadas com interpretações específicas para joalherias.

#### 📥 Parâmetros de Entrada
```python
{
    "analysis_type": "correlation",      # Tipo de análise
    "data_csv": "data/vendas.csv",      # Arquivo de dados
    "target_column": "Total_Liquido",   # Coluna alvo
    "group_column": "Grupo_Produto"     # Coluna de agrupamento
}
```

**Tipos de Análise**:
- `correlation`: Análise de correlação entre variáveis
- `clustering`: Segmentação de produtos/clientes
- `outliers`: Detecção de anomalias e valores atípicos
- `rfm_products`: Análise RFM adaptada para produtos
- `trend_test`: Testes de significância de tendências
- `distribution`: Análise de distribuição e normalidade

#### 🎯 Funcionalidades
- Análises de correlação com testes de significância
- Clustering automático com interpretação de negócio
- Detecção de outliers com múltiplos métodos
- RFM analysis para produtos
- Testes estatísticos de tendências
- Análise de distribuições

#### 📤 Outputs
```json
{
    "correlation_matrix": "matriz_correlacao",
    "significant_correlations": "correlacoes_significativas",
    "category_specific": "analise_por_categoria",
    "insights": ["insights_automaticos"],
    "cluster_analysis": "analise_clusters",
    "statistical_tests": "resultados_testes"
}
```

#### 💡 Casos de Uso
- Análise exploratória de dados
- Identificação de padrões ocultos
- Validação de hipóteses de negócio
- Segmentação estatística
- Detecção de anomalias

---

### Advanced Visualization Tool

**Descrição**: Cria visualizações interativas sofisticadas especializadas para joalherias.

#### 📥 Parâmetros de Entrada
```python
{
    "chart_type": "executive_dashboard",  # Tipo de visualização
    "data_csv": "data/vendas.csv",       # Arquivo de dados
    "title": "Análise de Vendas",        # Título do gráfico
    "export_format": "html",             # Formato de exportação
    "save_output": True                  # Salvar arquivo
}
```

**Tipos de Visualização**:
- `executive_dashboard`: Dashboard executivo completo
- `sales_trends`: Análise detalhada de tendências
- `product_analysis`: Performance de produtos e categorias
- `seasonal_heatmap`: Mapa de calor de sazonalidade
- `category_performance`: Performance comparativa por categoria
- `inventory_matrix`: Matriz de inventário (ABC + giro)
- `customer_segments`: Análise de segmentação de clientes
- `financial_overview`: Visão geral financeira com KPIs

#### 🎯 Funcionalidades
- Visualizações interativas com Plotly
- Design profissional otimizado para setor de luxo
- Múltiplos formatos de exportação (HTML, PNG, JSON)
- Responsividade para diferentes dispositivos
- Métricas especializadas do setor

#### 📤 Outputs
- **HTML**: Visualização interativa completa
- **PNG**: Imagem estática de alta qualidade
- **JSON**: Dados estruturados da visualização

#### 💡 Casos de Uso
- Apresentações executivas
- Relatórios visuais
- Dashboards interativos
- Análise exploratória visual
- Comunicação de insights

---

### DuckDuckGo Search Tool

**Descrição**: Ferramenta de busca na web para informações complementares.

#### 📥 Parâmetros de Entrada
```python
{
    "query": "termo_de_busca",    # Termo para buscar
    "domain": "br"               # Domínio de busca (padrão: br)
}
```

#### 🎯 Funcionalidades
- Busca web através do DuckDuckGo
- Rate limiting automático
- Filtros por domínio
- Resultados estruturados

#### 📤 Outputs
- Texto com resultados da busca formatados

#### 💡 Casos de Uso
- Pesquisa de benchmarks de mercado
- Informações sobre concorrentes
- Tendências do setor
- Pesquisas complementares

---

### Custom Tool

**Descrição**: Template para criação de ferramentas personalizadas.

#### 📥 Parâmetros de Entrada
```python
{
    "argument": "string"    # Argumento genérico
}
```

#### 🎯 Funcionalidades
- Template base para desenvolvimento
- Estrutura padrão CrewAI
- Exemplo de implementação

#### 💡 Casos de Uso
- Desenvolvimento de novas ferramentas
- Prototipagem rápida
- Extensões personalizadas

---

## 🚀 FERRAMENTAS AVANÇADAS

### Customer Insights Engine

**Descrição**: Motor avançado de insights de clientes com análises comportamentais profundas.

#### 📥 Parâmetros de Entrada
```python
{
    "analysis_type": "behavioral_segmentation",  # Tipo de análise
    "data_csv": "data/vendas.csv",              # Arquivo de dados
    "customer_id_column": "Codigo_Cliente",     # Coluna de ID do cliente
    "segmentation_method": "rfm",               # Método de segmentação
    "prediction_horizon": 90                    # Horizonte de predição (dias)
}
```

**Tipos de Análise**:
- `behavioral_segmentation`: Segmentação comportamental avançada
- `lifecycle_analysis`: Análise do ciclo de vida do cliente
- `churn_prediction`: Predição de abandono de clientes
- `value_analysis`: Análise de valor do cliente (CLV, CAC)
- `preference_mining`: Mineração de preferências e padrões
- `journey_mapping`: Mapeamento da jornada do cliente

**Métodos de Segmentação**:
- `rfm`: Recency, Frequency, Monetary
- `behavioral`: Baseado em comportamento de compra
- `value_based`: Segmentação por valor
- `hybrid`: Combinação de múltiplos critérios

#### 🎯 Funcionalidades
- Segmentação automática com RFM e ML
- Predição de churn com scores de risco
- Cálculo de CLV (Customer Lifetime Value)
- Análise de jornada do cliente
- Mineração de preferências de produtos
- Estratégias de retenção personalizadas
- Análise de ciclo de vida

#### 📤 Outputs
```json
{
    "segmentation_results": {
        "segments": "distribuicao_segmentos",
        "segment_analysis": "analise_detalhada",
        "migration_matrix": "matriz_transicao"
    },
    "churn_prediction": {
        "risk_scores": "scores_risco",
        "high_risk_customers": "clientes_alto_risco",
        "retention_strategies": "estrategias_retencao",
        "financial_impact": "impacto_financeiro"
    },
    "value_analysis": {
        "clv_distribution": "distribuicao_clv",
        "top_customers": "top_clientes",
        "growth_potential": "potencial_crescimento"
    },
    "insights": ["insights_automaticos"],
    "recommendations": ["recomendacoes_acao"]
}
```

#### 💡 Casos de Uso
- Programas de fidelidade
- Campanhas de retenção
- Personalização de ofertas
- Análise de valor do cliente
- Estratégias de marketing
- Predição de comportamento

---

### Risk Assessment Tool

**Descrição**: Ferramenta de avaliação de riscos empresariais com matrizes de impacto.

#### 📥 Parâmetros de Entrada
```python
{
    "assessment_type": "comprehensive_risk",  # Tipo de avaliação
    "data_csv": "data/vendas.csv",           # Arquivo de dados
    "risk_tolerance": "medium",              # Tolerância ao risco
    "time_horizon": "6_months",              # Horizonte temporal
    "include_mitigation": True               # Incluir estratégias de mitigação
}
```

**Tipos de Avaliação**:
- `business_risk`: Riscos gerais do negócio
- `financial_risk`: Riscos financeiros e de liquidez
- `operational_risk`: Riscos operacionais e de processo
- `market_risk`: Riscos de mercado e competição
- `customer_risk`: Riscos relacionados à base de clientes
- `comprehensive_risk`: Avaliação completa de todos os riscos

**Tolerância ao Risco**:
- `low`: Conservadora
- `medium`: Moderada
- `high`: Agressiva

#### 🎯 Funcionalidades
- Identificação automática de riscos críticos
- Matriz de probabilidade vs impacto
- Scores de risco por categoria
- Estratégias de mitigação personalizadas
- Planos de monitoramento
- Planos de contingência
- Alertas automáticos

#### 📤 Outputs
```json
{
    "risk_assessment": {
        "overall_risk_score": 6.5,
        "risk_matrix": "matriz_riscos",
        "top_risks": "principais_riscos",
        "risk_by_category": "riscos_por_categoria"
    },
    "mitigation_strategies": {
        "immediate_actions": "acoes_imediatas",
        "medium_term_plans": "planos_medio_prazo",
        "monitoring_plan": "plano_monitoramento"
    },
    "financial_impact": "impacto_financeiro_estimado",
    "recommendations": ["recomendacoes_gestao_risco"]
}
```

#### 💡 Casos de Uso
- Gestão de riscos empresariais
- Compliance e auditoria
- Planejamento estratégico
- Seguros empresariais
- Análise de investimentos
- Planos de contingência

---

### Recommendation Engine

**Descrição**: Motor de recomendações inteligentes com algoritmos de ML para diferentes aspectos do negócio.

#### 📥 Parâmetros de Entrada
```python
{
    "recommendation_type": "product_recommendations",  # Tipo de recomendação
    "data_csv": "data/vendas.csv",                    # Arquivo de dados
    "target_segment": "all",                          # Segmento alvo
    "recommendation_count": 10,                       # Número de recomendações
    "confidence_threshold": 0.7                       # Limiar de confiança
}
```

**Tipos de Recomendação**:
- `product_recommendations`: Recomendações de produtos baseadas em padrões
- `customer_targeting`: Targeting inteligente de clientes
- `pricing_optimization`: Sugestões de otimização de preços
- `inventory_suggestions`: Recomendações de gestão de estoque
- `marketing_campaigns`: Campanhas personalizadas por segmento
- `strategic_actions`: Ações estratégicas baseadas em dados

**Segmentos Alvo**:
- `all`: Todos os clientes
- `vip`: Clientes VIP
- `new_customers`: Novos clientes
- `at_risk`: Clientes em risco

#### 🎯 Funcionalidades
- Collaborative Filtering para recomendações de produtos
- Content-Based Filtering baseado em características
- Market Basket Analysis para cross-selling
- Otimização de preços com elasticidade
- Campanhas personalizadas por segmento
- Sugestões de ações estratégicas
- ROI estimado das recomendações

#### 📤 Outputs
```json
{
    "product_recommendations": {
        "recommended_products": "produtos_recomendados",
        "cross_sell_opportunities": "oportunidades_cross_sell",
        "market_basket_analysis": "analise_cesta_mercado"
    },
    "pricing_optimization": {
        "price_adjustments": "ajustes_preco",
        "elasticity_analysis": "analise_elasticidade",
        "revenue_impact": "impacto_receita"
    },
    "marketing_campaigns": {
        "personalized_campaigns": "campanhas_personalizadas",
        "optimal_timing": "timing_otimo",
        "expected_roi": "roi_esperado"
    },
    "strategic_actions": {
        "prioritized_actions": "acoes_priorizadas",
        "implementation_roadmap": "roadmap_implementacao",
        "success_metrics": "metricas_sucesso"
    }
}
```

#### 💡 Casos de Uso
- Sistemas de recomendação de produtos
- Otimização de preços
- Campanhas de marketing direcionadas
- Gestão de inventário inteligente
- Cross-selling e up-selling
- Estratégias de crescimento

---

### Advanced Analytics Engine

**Descrição**: Motor de análises avançadas com machine learning e estatística aplicada.

#### 📥 Parâmetros de Entrada
```python
{
    "analysis_type": "predictive_analytics",    # Tipo de análise
    "data_csv": "data/vendas.csv",             # Arquivo de dados
    "model_type": "auto",                      # Tipo de modelo
    "target_metric": "revenue",                # Métrica alvo
    "forecast_horizon": 90                     # Horizonte de previsão
}
```

**Tipos de Análise**:
- `predictive_analytics`: Análises preditivas
- `pattern_recognition`: Reconhecimento de padrões
- `anomaly_detection`: Detecção de anomalias
- `market_analysis`: Análise de mercado
- `performance_optimization`: Otimização de performance

#### 🎯 Funcionalidades
- Modelos preditivos automatizados
- Detecção de padrões complexos
- Análise de anomalias em tempo real
- Otimização automática de hiperparâmetros
- Validação cruzada e métricas de performance
- Interpretabilidade de modelos

#### 📤 Outputs
- Previsões com intervalos de confiança
- Importância de features
- Métricas de performance do modelo
- Visualizações de resultados
- Recomendações baseadas em análises

#### 💡 Casos de Uso
- Previsão de demanda avançada
- Otimização de operações
- Detecção de fraudes
- Análise de sentimento
- Modelagem preditiva

---

### Business Intelligence Dashboard

**Descrição**: Dashboard executivo completo com métricas em tempo real.

#### 📥 Parâmetros de Entrada
```python
{
    "dashboard_type": "executive",         # Tipo de dashboard
    "data_source": "data/vendas.csv",     # Fonte de dados
    "refresh_interval": "daily",          # Intervalo de atualização
    "kpi_focus": "all"                    # Foco em KPIs específicos
}
```

#### 🎯 Funcionalidades
- Dashboards interativos em tempo real
- KPIs executivos automatizados
- Drill-down em dados
- Alertas automáticos
- Exportação em múltiplos formatos
- Responsividade mobile

#### 📤 Outputs
- Dashboard HTML interativo
- Relatórios PDF automatizados
- Alertas por email
- APIs de dados

#### 💡 Casos de Uso
- Monitoramento executivo
- Relatórios automáticos
- Tomada de decisão em tempo real
- Apresentações para stakeholders

---

### Competitive Intelligence Tool

**Descrição**: Ferramenta de inteligência competitiva e análise de mercado.

#### 📥 Parâmetros de Entrada
```python
{
    "analysis_type": "market_positioning",    # Tipo de análise
    "competitor_data": "dados_concorrentes",  # Dados dos concorrentes
    "market_segment": "luxury_jewelry",       # Segmento de mercado
    "geographic_scope": "national"            # Escopo geográfico
}
```

#### 🎯 Funcionalidades
- Análise de posicionamento competitivo
- Benchmarking de preços
- Análise de gap de mercado
- Monitoramento de tendências
- Inteligência de produto
- Análise de share of voice

#### 📤 Outputs
- Mapas de posicionamento
- Análises de gap competitivo
- Relatórios de benchmarking
- Oportunidades de mercado
- Recomendações estratégicas

#### 💡 Casos de Uso
- Estratégia competitiva
- Desenvolvimento de produtos
- Precificação estratégica
- Identificação de oportunidades
- Planejamento de marketing

---

## 🔗 Integração Entre Ferramentas

### Fluxos Recomendados

#### 📊 Análise Completa de Negócio
1. **SQL Query Tool** → Extração de dados
2. **KPI Calculator** → Métricas básicas
3. **Statistical Analysis** → Padrões estatísticos
4. **Advanced Visualization** → Visualização de insights
5. **Risk Assessment** → Identificação de riscos
6. **Recommendation Engine** → Ações recomendadas

#### 👥 Análise de Clientes
1. **SQL Query Tool** → Dados de clientes
2. **Customer Insights Engine** → Segmentação e análise comportamental
3. **Advanced Analytics** → Modelos preditivos
4. **Recommendation Engine** → Estratégias personalizadas

#### 📈 Planejamento Estratégico
1. **Business Intelligence Dashboard** → Situação atual
2. **Competitive Intelligence** → Análise competitiva
3. **Prophet Forecast** → Previsões
4. **Risk Assessment** → Análise de riscos
5. **Recommendation Engine** → Plano de ação

---

## 📋 Requisitos Técnicos

### Dependências Principais
- Python 3.8+
- pandas, numpy
- scikit-learn
- plotly
- prophet
- crewai

### Estrutura de Dados
- Formato CSV com separador ';'
- Encoding UTF-8
- Colunas obrigatórias: Data, Total_Liquido, Quantidade
- Colunas recomendadas: Codigo_Cliente, Codigo_Produto, Grupo_Produto

### Performance
- Mínimo 10 registros para análises básicas
- Recomendado 1000+ registros para análises avançadas
- Processamento otimizado para datasets até 1M registros

---

## 🚀 Próximos Passos

### Expansões Planejadas
- Integração com APIs externas
- Modelos de ML mais sofisticados
- Análises em tempo real
- Automação de relatórios
- Integração com ERPs

### Customizações
- Ferramentas específicas por setor
- Modelos personalizados
- Integrações customizadas
- Dashboards personalizados

---

*Documentação atualizada em: Dezembro 2024*
*Versão: 2.0*
*Sistema: Insights AI - Análise Inteligente para Joalherias* 