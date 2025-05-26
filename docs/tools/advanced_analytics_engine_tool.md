# 🤖 ADVANCED ANALYTICS ENGINE TOOL - DOCUMENTAÇÃO COMPLETA

## 📋 **VISÃO GERAL**

O `AdvancedAnalyticsEngineTool` é um motor de análises avançadas com Machine Learning especialmente otimizado para joalherias. Combina algoritmos como Random Forest, XGBoost e técnicas de clustering para descobrir padrões ocultos, prever demanda e otimizar processos de negócio.

### **Características Principais:**
- 🤖 **Machine Learning Avançado**: Random Forest, XGBoost, Isolation Forest
- 📊 **Análises Especializadas**: 7 tipos de análises para diferentes necessidades
- 🎯 **Adaptativo**: Ajusta-se automaticamente à quantidade de dados disponíveis
- 📈 **Previsões Robustas**: Intervalos de confiança e validação estatística
- 💾 **Cache Inteligente**: Otimização de performance para datasets grandes
- 📋 **Schema Pydantic**: Validações rigorosas de entrada

---

## 🏗️ **ARQUITETURA DA CLASSE**

### **Herança e Mixins:**
```python
class AdvancedAnalyticsEngineTool(BaseTool,
                                  DataPreparationMixin,
                                  ReportFormatterMixin,
                                  JewelryRFMAnalysisMixin,
                                  JewelryBusinessAnalysisMixin)
```

### **Schema de Entrada (Pydantic):**
- `analysis_type`: Tipo de análise (7 opções disponíveis)
- `data_csv`: Caminho do arquivo CSV (padrão: "data/vendas.csv")
- `target_column`: Coluna alvo (padrão: "Total_Liquido")
- `prediction_horizon`: Horizonte de predição em dias (7-365)
- `confidence_level`: Nível de confiança (0.80-0.99)
- `model_complexity`: Complexidade do modelo ("simple", "balanced", "complex")
- `enable_ensemble`: Usar ensemble de modelos (recomendado: True)
- `sample_size`: Tamanho da amostra (5000-100000, None = todos)
- `cache_results`: Usar cache (recomendado: True)

---

## 🎯 **FUNÇÕES PRINCIPAIS**

### **1. `_run()` - Função Principal de Execução**

**Objetivo:** Coordena todo o processo de análise ML, desde validação até geração de resultados.

**Funcionamento:**
1. Valida disponibilidade de bibliotecas ML
2. Carrega e prepara dados usando módulos consolidados
3. Valida qualidade dos dados para ML
4. Executa análise específica selecionada
5. Adiciona metadados e salva no cache
6. Retorna resultado JSON estruturado

**Interpretação dos Resultados:**
- **Sucesso**: JSON com análise completa e metadados
- **Erro**: JSON com erro detalhado e troubleshooting
- **Metadados**: Informações sobre o processo executado

---

### **2. `_load_and_prepare_ml_data()` - Carregamento e Preparação**

**Objetivo:** Carregar dados do CSV e prepará-los especificamente para análises ML.

**Funcionamento:**
1. Verifica cache existente
2. Carrega dados do CSV com encoding correto
3. Aplica amostragem se necessário
4. Executa preparação strict usando mixin consolidado
5. Adiciona features específicas para ML
6. Salva no cache para reutilização

**Parâmetros:**
- `data_csv`: Caminho do arquivo
- `use_cache`: Usar cache (True/False)
- `sample_size`: Tamanho da amostra (opcional)

**Interpretação:**
- **Sucesso**: DataFrame preparado com features ML
- **Falha**: None (dados insuficientes ou erro de carregamento)

---

### **3. `_add_ml_features()` - Engenharia de Features**

**Objetivo:** Criar features específicas para Machine Learning a partir dos dados brutos.

**Features Criadas:**

#### **Features Temporais:**
- `Days_Since_Start`: Dias desde o início do período
- `Month_Sin/Cos`: Sazonalidade circular mensal
- `Day_Of_Week`: Dia da semana (0-6)

#### **Features de Agregação:**
- `Customer_Frequency`: Frequência de compras do cliente
- `Customer_AOV`: Ticket médio do cliente
- `Customer_Total`: Total gasto pelo cliente
- `Product_Frequency`: Frequência de venda do produto
- `Product_AOV`: Preço médio do produto

#### **Encoding Categórico:**
- **Label Encoding**: Para variáveis com >10 categorias
- **One-Hot Encoding**: Para variáveis com ≤10 categorias

#### **Normalização:**
- Campos numéricos são normalizados com StandardScaler
- Criadas versões `_scaled` dos campos originais

**Interpretação:**
- Features temporais capturam sazonalidade
- Features de agregação capturam comportamento
- Encoding permite uso de variáveis categóricas
- Normalização melhora performance dos algoritmos

---

## 🔬 **ANÁLISES ESPECIALIZADAS**

### **4. `_ml_insights_analysis()` - Insights com ML**

**Objetivo:** Descobrir padrões ocultos usando Random Forest e XGBoost.

**Algoritmos Utilizados:**
- **Random Forest**: Ensemble de árvores de decisão
- **XGBoost**: Gradient boosting otimizado (se disponível)

**Configuração por Complexidade:**
- **Simple**: Random Forest com 50 estimadores
- **Balanced**: Random Forest (100) + XGBoost (100)
- **Complex**: Random Forest (200) + XGBoost (200)

**Estrutura do Resultado:**
```json
{
  "analysis_type": "ML Insights Analysis",
  "target_column": "Total_Liquido",
  "model_performance": {
    "Random Forest": {
      "mae": 123.45,
      "mse": 15432.10,
      "rmse": 124.23,
      "r2_score": 0.847
    }
  },
  "feature_importance": {
    "top_features": [
      {"feature": "Customer_AOV", "importance": 0.234},
      {"feature": "Product_Frequency", "importance": 0.187}
    ],
    "best_model": "Random Forest"
  },
  "business_insights": [...],
  "recommendations": [...]
}
```

**Interpretação das Métricas:**
- **MAE** (Mean Absolute Error): Erro médio absoluto em unidades originais
- **MSE** (Mean Squared Error): Erro quadrático médio
- **RMSE** (Root MSE): Raiz do erro quadrático (mesma unidade da variável)
- **R² Score**: Coeficiente de determinação (0-1, quanto maior melhor)
  - > 0.8: Modelo excelente
  - 0.6-0.8: Modelo bom
  - < 0.6: Modelo limitado

**Feature Importance:**
- Indica quais variáveis mais influenciam o resultado
- Use para identificar fatores-chave do negócio
- Priorize otimizações nas features mais importantes

---

### **5. `_anomaly_detection_analysis()` - Detecção de Anomalias**

**Objetivo:** Identificar transações ou padrões anômalos usando Isolation Forest.

**Algoritmo:** Isolation Forest
- Identifica pontos que são facilmente "isolados" do restante dos dados
- Contamination = 0.1 (10% das observações consideradas anômalas)

**Estrutura do Resultado:**
```json
{
  "analysis_type": "Anomaly Detection Analysis",
  "target_column": "Total_Liquido",
  "total_anomalies": 142,
  "anomaly_percentage": 2.3,
  "anomaly_summary": {
    "avg_value": 1245.67,
    "max_value": 15432.10,
    "min_value": 23.45
  },
  "business_insights": [...],
  "recommendations": [...]
}
```

**Interpretação:**
- **total_anomalies**: Número absoluto de anomalias detectadas
- **anomaly_percentage**: Percentual do total de dados
- **anomaly_summary**: Estatísticas das transações anômalas

**Casos de Uso:**
- Identificar vendas excepcionalmente altas (oportunidades)
- Detectar vendas suspeitas (possíveis erros)
- Encontrar padrões de compra únicos
- Investigar transações para insights de negócio

---

### **6. `_demand_forecasting_analysis()` - Previsão de Demanda Adaptativa**

**Objetivo:** Prever demanda futura adaptando-se ao volume de dados disponíveis.

**Estratégia Adaptativa:**

#### **Dados Limitados (<14 dias):**
- **Método**: Médias móveis simples
- **Features**: Médias dos últimos 7/14 dias
- **Confiabilidade**: Baixa - experimental

#### **Dados Moderados (14-30 dias):**
- **Método**: Random Forest básico (50 estimadores)
- **Features**: day_of_week, month, lag_3, lag_7, rolling_mean_3
- **Confiabilidade**: Média - adequado para orientação

#### **Dados Abundantes (>30 dias):**
- **Método**: Random Forest completo (50-200 estimadores)
- **Features**: Lags múltiplos, médias móveis, tendências, sazonalidade
- **Confiabilidade**: Alta - adequado para planejamento

**Features Utilizadas:**
- **Lags**: Valores históricos (lag_3, lag_7, lag_14, lag_30)
- **Rolling**: Médias e desvios móveis
- **Temporais**: Dia da semana, mês, trimestre
- **Tendência**: Tendência linear e sazonalidade circular

**Estrutura do Resultado:**
```json
{
  "analysis_type": "Demand Forecasting Analysis",
  "target_column": "Total_Liquido",
  "prediction_horizon": 30,
  "data_summary": {
    "total_period_days": 730,
    "actual_data_days": 485,
    "data_coverage": 66.4,
    "model_type": "Random Forest (Balanced)",
    "features_count": 12
  },
  "forecast_summary": {
    "avg_predicted": 1234.56,
    "total_predicted": 37036.80,
    "min_predicted": 856.23,
    "max_predicted": 1876.45,
    "confidence_lower": 987.65,
    "confidence_upper": 1481.47
  },
  "historical_baseline": {
    "avg_daily": 1198.34,
    "recent_avg": 1289.67,
    "trend": "crescente"
  },
  "daily_predictions": [
    {
      "date": "2024-01-01",
      "predicted_value": 1245.67,
      "confidence_lower": 996.54,
      "confidence_upper": 1494.80
    }
  ],
  "business_insights": [...],
  "recommendations": [...]
}
```

**Interpretação dos Resultados:**

#### **Data Summary:**
- **total_period_days**: Período total dos dados
- **actual_data_days**: Dias com dados reais
- **data_coverage**: Percentual de cobertura
- **model_type**: Tipo de modelo utilizado
- **features_count**: Número de features

#### **Forecast Summary:**
- **avg_predicted**: Demanda média prevista
- **total_predicted**: Demanda total no período
- **confidence_lower/upper**: Intervalos de confiança

#### **Historical Baseline:**
- **avg_daily**: Média histórica diária
- **recent_avg**: Média dos últimos 7 dias
- **trend**: Direção da tendência

#### **Daily Predictions:**
- Previsões diárias detalhadas
- Intervalos de confiança por dia
- Primeiros 15 dias do horizonte

**Como Usar:**
1. **Planejamento de Estoque**: Use total_predicted para compras
2. **Gestão de Caixa**: Use avg_predicted para fluxo diário
3. **Análise de Risco**: Use intervalos de confiança
4. **Monitoramento**: Compare realizados vs. previstos

---

### **7. `_customer_behavior_analysis()` - Segmentação Comportamental**

**Objetivo:** Segmentar clientes baseado em padrões comportamentais usando clustering ML.

**Algoritmo:** K-Means Clustering
- Utiliza StandardScaler para normalização
- K ótimo calculado heuristicamente (min(5, clientes/10))

**Features para Clustering:**
- **Frequência**: Número de transações
- **Valor Monetário**: Soma e média de gastos
- **Recência**: Dias desde última compra
- **Volume**: Quantidade total comprada

**Estrutura do Resultado:**
```json
{
  "analysis_type": "Customer Behavior Analysis",
  "target_column": "Total_Liquido",
  "total_customers": 1247,
  "clusters_identified": 4,
  "cluster_profiles": {
    "Cluster_0": {
      "size": 312,
      "size_percentage": 25.0,
      "avg_revenue": 2345.67,
      "avg_frequency": 8.5,
      "avg_recency": 15.2,
      "total_revenue": 731849.04,
      "classification": "VIP"
    }
  },
  "business_insights": [...],
  "recommendations": [...]
}
```

**Classificação Automática dos Clusters:**
- **VIP**: Revenue > percentil 80
- **Frequente**: Frequency > percentil 70
- **Ativo**: Recency < 30 dias
- **Regular**: Demais casos

**Interpretação dos Profiles:**
- **size**: Número de clientes no cluster
- **size_percentage**: Percentual do total
- **avg_revenue**: Receita média por cliente
- **avg_frequency**: Frequência média de compras
- **avg_recency**: Dias médios desde última compra
- **total_revenue**: Receita total do cluster

**Estratégias por Segmento:**
- **VIP**: Programas de fidelidade premium, atendimento personalizado
- **Frequente**: Recompensas por frequência, produtos exclusivos
- **Ativo**: Manter engajamento, cross-sell
- **Regular**: Campanhas de ativação, ofertas especiais

---

## 🛠️ **FUNÇÕES AUXILIARES**

### **8. `_select_ml_features()` - Seleção de Features**

**Objetivo:** Selecionar automaticamente as melhores features para ML.

**Critérios de Seleção:**
1. Apenas campos numéricos
2. Excluir variáveis não-preditivas (IDs, datas)
3. Remover campos com todos valores nulos
4. Limitar a 20 features (evitar overfitting)

**Features Priorizadas:**
- Features engineered (_scaled, _encoded)
- Campos de valor e quantidade
- Features de agregação (Customer_*, Product_*)
- Features temporais (Month_Sin, Day_Of_Week)

---

### **9. `_configure_ml_models()` - Configuração de Modelos**

**Objetivo:** Configurar modelos ML baseado na complexidade desejada.

**Configurações por Complexidade:**

#### **Simple:**
- Random Forest: 50 estimadores
- Execução rápida, menor precisão

#### **Balanced (Padrão):**
- Random Forest: 100 estimadores
- XGBoost: 100 estimadores (se disponível)
- Equilíbrio entre velocidade e precisão

#### **Complex:**
- Random Forest: 200 estimadores, max_depth=10
- XGBoost: 200 estimadores, max_depth=6
- Maior precisão, execução mais lenta

---

### **10. `_generate_adaptive_predictions()` - Previsões Adaptativas**

**Objetivo:** Gerar previsões futuras usando modelo treinado e simulação temporal.

**Processo:**
1. Para cada dia futuro, constrói features baseadas em:
   - Data futura (day_of_week, month, etc.)
   - Valores históricos (lags)
   - Médias móveis calculadas
2. Aplica modelo treinado
3. Valida resultado (não negativo, limite superior)
4. Atualiza histórico para próxima previsão

**Sanidade Checks:**
- Valores não podem ser negativos
- Limite superior: 3x a média recente
- Continuidade temporal nas previsões

---

### **11. `_calculate_prediction_confidence()` - Intervalos de Confiança**

**Objetivo:** Calcular intervalos de confiança adaptativos para previsões.

**Metodologia:**
1. Calcula coeficiente de variação histórico
2. Define fator de confiança entre 10% e 30%
3. Aplica fatores às previsões

**Interpretação:**
- **confidence_level**: Nível de confiança calculado
- **lower/upper_factor**: Multiplicadores para intervalos
- Maior variabilidade histórica = intervalos mais amplos

---

## 📊 **INTERPRETAÇÃO AVANÇADA DE RESULTADOS**

### **Métricas de Performance dos Modelos:**

#### **R² Score (Coeficiente de Determinação):**
- **0.9-1.0**: Excelente - Modelo explica >90% da variância
- **0.7-0.9**: Muito Bom - Adequado para decisões estratégicas
- **0.5-0.7**: Bom - Útil para análises táticas
- **0.3-0.5**: Regular - Insights limitados
- **<0.3**: Pobre - Necessita mais dados ou features

#### **RMSE vs. Média:**
- **RMSE < 10% da média**: Excelente precisão
- **RMSE 10-20% da média**: Boa precisão
- **RMSE 20-30% da média**: Precisão limitada
- **RMSE > 30% da média**: Baixa precisão

### **Feature Importance:**
- **>0.2**: Feature crítica - foco estratégico
- **0.1-0.2**: Feature importante - monitorar
- **0.05-0.1**: Feature moderada - considerar
- **<0.05**: Feature irrelevante - pode remover

### **Confiança das Previsões:**
- **Confidence Level >0.8**: Previsões muito confiáveis
- **Confidence Level 0.6-0.8**: Previsões confiáveis
- **Confidence Level <0.6**: Previsões indicativas

---

## 🎯 **CASOS DE USO PRÁTICOS**

### **1. Planejamento de Compras:**
```python
# Usar demand_forecasting com horizon=90 para trimestre
result = tool._run(
    analysis_type="demand_forecasting",
    prediction_horizon=90,
    model_complexity="balanced"
)
# Interpretar: total_predicted para volume de compras
```

### **2. Identificação de Oportunidades:**
```python
# Usar anomaly_detection para encontrar padrões únicos
result = tool._run(
    analysis_type="anomaly_detection",
    target_column="Total_Liquido"
)
# Investigar anomalias com valores altos
```

### **3. Segmentação para Marketing:**
```python
# Usar customer_behavior para campanhas direcionadas
result = tool._run(
    analysis_type="customer_behavior"
)
# Desenvolver estratégias por cluster_profiles
```

### **4. Otimização de Fatores-Chave:**
```python
# Usar ml_insights para identificar drivers de vendas
result = tool._run(
    analysis_type="ml_insights",
    model_complexity="complex",
    enable_ensemble=True
)
# Focar nas top_features para maximizar impacto
```

---

## ⚠️ **TROUBLESHOOTING COMUM**

### **Problemas de Dados:**

#### **"Dados insuficientes para ML":**
- **Causa**: Menos de 100 registros
- **Solução**: Aumentar período ou usar Statistical Analysis Tool

#### **"Coluna alvo não encontrada":**
- **Causa**: target_column incorreto
- **Solução**: Verificar nomes exatos das colunas (use 'Total_Liquido')

#### **"Features insuficientes":**
- **Causa**: Dados muito simples ou muitos NaN
- **Solução**: Melhorar qualidade dos dados ou usar KPI Calculator

### **Problemas de Performance:**

#### **Execução muito lenta:**
- Use `sample_size=10000` para datasets grandes
- Configure `model_complexity="simple"`
- Ative `cache_results=True`

#### **Baixa precisão do modelo:**
- Aumente `model_complexity="complex"`
- Colete mais dados históricos
- Verifique qualidade dos dados

### **Problemas de Bibliotecas:**

#### **"Scikit-learn não disponível":**
```bash
pip install scikit-learn
```

#### **"XGBoost não disponível":**
```bash
pip install xgboost
```

---

## 📈 **RECOMENDAÇÕES DE USO**

### **Para Datasets Pequenos (<1000 registros):**
- Use `model_complexity="simple"`
- Configure `sample_size=None`
- Prefira Statistical Analysis Tool para análises básicas

### **Para Datasets Médios (1000-10000 registros):**
- Use `model_complexity="balanced"` (padrão)
- Configure `enable_ensemble=True`
- Ative `cache_results=True`

### **Para Datasets Grandes (>10000 registros):**
- Use `model_complexity="complex"` para máxima precisão
- Configure `sample_size=50000` se necessário
- Use `cache_results=True` obrigatoriamente

### **Para Análises em Produção:**
- Sempre valide resultados com equipe de negócio
- Monitore desvios entre previsto vs. realizado
- Atualize modelos mensalmente com novos dados
- Mantenha histórico de performance dos modelos

---

## 🔄 **PRÓXIMAS VERSÕES**

### **Placeholders Ativos (v4.1):**
- `_product_lifecycle_analysis()`: Análise de ciclo de vida completa
- `_price_optimization_analysis()`: Otimização de preços com elasticidade
- `_inventory_optimization_analysis()`: Gestão otimizada de estoque

### **Melhorias Planejadas:**
- Deep Learning com redes neurais
- Análise de séries temporais com ARIMA/Prophet
- A/B Testing automatizado
- Integração com dados externos (economia, concorrência)

---

*Documentação gerada automaticamente - Advanced Analytics Engine Tool v4.0* 