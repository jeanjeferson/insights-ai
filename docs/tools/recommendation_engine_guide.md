# RecommendationEngine V2.0 - Guia Completo

## 📋 Índice
1. [Visão Geral](#visão-geral)
2. [Instalação e Configuração](#instalação-e-configuração)
3. [Tipos de Recomendações](#tipos-de-recomendações)
4. [Integração com CrewAI](#integração-com-crewai)
5. [Exemplos Práticos](#exemplos-práticos)
6. [API Reference](#api-reference)
7. [Performance e Monitoramento](#performance-e-monitoramento)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Visão Geral

O **RecommendationEngine V2.0** é um sistema inteligente de recomendações otimizado para joalherias e integração com CrewAI. Utiliza algoritmos avançados de Machine Learning para fornecer insights acionáveis em 6 áreas principais:

### ✅ Funcionalidades Implementadas
- **🛍️ Recomendações de Produtos** - ML Híbrido (Collaborative + Content-based filtering)
- **🎯 Segmentação de Clientes** - RFM Analysis + K-means Clustering
- **💰 Otimização de Preços** - Análise de elasticidade e variabilidade
- **📦 Gestão de Estoque** - Análise ABC + Rotatividade
- **📢 Campanhas de Marketing** - Segmentação avançada + ROI prediction
- **🎯 Ações Estratégicas** - Business Intelligence + Trend analysis

### 🚀 Algoritmos ML Avançados
- **Collaborative Filtering** usando Matrix Factorization (SVD)
- **Content-based Filtering** com TF-IDF vectorization
- **Market Basket Analysis** com algoritmo Apriori
- **Anomaly Detection** usando Isolation Forest
- **Customer Lifetime Value** prediction
- **Seleção automática de algoritmos** baseada nos dados

---

## 🛠 Instalação e Configuração

### Pré-requisitos
```bash
Python 3.8+
pandas, numpy, scikit-learn, mlxtend
crewai (para integração)
```

### Instalação
```bash
# Clonar repositório
git clone <repo-url>
cd insights-ai

# Instalar dependências
pip install -r requirements.txt

# Verificar instalação
python -c "from src.insights.tools.advanced.recommendation_engine import RecommendationEngine; print('✅ Instalação OK')"
```

### Estrutura de Dados Requerida
O sistema espera um arquivo CSV com as seguintes colunas:
```
Data, Codigo_Cliente, Nome_Cliente, Codigo_Produto, 
Descricao_Produto, Grupo_Produto, Quantidade, 
Total_Liquido, Preco_Tabela
```

---

## 🔍 Tipos de Recomendações

### 1. 🛍️ Product Recommendations
**Objetivo**: Recomendar produtos baseado em padrões de compra e similaridade

**Algoritmos utilizados**:
- Collaborative filtering (Matrix Factorization)
- Content-based filtering (TF-IDF)
- Hybrid recommendation system

**Exemplo de uso**:
```python
from src.insights.tools.advanced.recommendation_engine import RecommendationEngine

engine = RecommendationEngine()
result = engine._run(
    recommendation_type="product_recommendations",
    data_csv="data/vendas.csv",
    target_segment="vip",
    recommendation_count=10,
    confidence_threshold=0.8
)
```

**Output típico**:
```json
{
  "success": true,
  "analysis_type": "Product Recommendations Analysis",
  "recommendations": {
    "top_products": [
      {
        "product": "ALIANCA 750",
        "score": 0.95,
        "reason": "High affinity with VIP segment"
      }
    ]
  }
}
```

### 2. 🎯 Customer Targeting
**Objetivo**: Segmentar clientes para campanhas direcionadas

**Algoritmos utilizados**:
- RFM Analysis (Recency, Frequency, Monetary)
- K-means clustering
- Customer scoring

**Segmentos automáticos**:
- **VIP**: Alto valor + alta frequência
- **High Value**: Alto valor + baixa frequência  
- **Potential**: Médio valor + crescimento
- **At Risk**: Baixa atividade recente

### 3. 💰 Pricing Optimization
**Objetivo**: Otimizar preços baseado em elasticidade e competição

**Análises realizadas**:
- Price elasticity analysis
- Competitive pricing
- Margin optimization
- Seasonal adjustments

### 4. 📦 Inventory Suggestions
**Objetivo**: Otimizar gestão de estoque

**Análises realizadas**:
- ABC Analysis (produtos por revenue)
- Turnover rate analysis
- Restock priority
- Slow-moving items identification

### 5. 📢 Marketing Campaigns
**Objetivo**: Criar campanhas personalizadas por segmento

**Funcionalidades**:
- Campaign type selection
- Channel optimization
- ROI estimation
- Message personalization

### 6. 🎯 Strategic Actions
**Objetivo**: Identificar ações estratégicas prioritárias

**Análises realizadas**:
- Growth trend analysis
- Market opportunity identification
- Business intelligence insights
- Priority matrix

---

## 🤝 Integração com CrewAI

### Configuração Básica
```python
from crewai import Agent, Task, Crew
from src.insights.tools.advanced.recommendation_engine import RecommendationEngine

# Criar agente com RecommendationEngine
sales_agent = Agent(
    role='Consultor de Vendas Especializado',
    goal='Maximizar vendas através de recomendações personalizadas',
    backstory='''Especialista em joalheria com 10 anos de experiência.
    Utiliza análise de dados avançada para identificar oportunidades de venda.''',
    tools=[RecommendationEngine()],
    verbose=True
)
```

### Exemplos de Agentes

#### 🎯 Sales Intelligence Agent
```python
sales_agent = Agent(
    role='Sales Intelligence Specialist',
    goal='Identify high-value sales opportunities through data analysis',
    backstory='Expert sales analyst with access to advanced ML recommendations',
    tools=[RecommendationEngine()],
    verbose=True
)

sales_task = Task(
    description='''
    Analyze customer purchase patterns and provide product recommendations
    for our VIP customers. Focus on cross-selling opportunities and 
    seasonal trends.
    ''',
    agent=sales_agent,
    expected_output='Detailed sales recommendations with confidence scores'
)
```

#### 📊 Marketing Strategy Agent
```python
marketing_agent = Agent(
    role='Marketing Strategy Analyst',
    goal='Design data-driven marketing campaigns',
    backstory='Marketing specialist with expertise in customer segmentation',
    tools=[RecommendationEngine()],
    verbose=True
)

campaign_task = Task(
    description='''
    Segment our customer base and design personalized marketing campaigns.
    Include ROI projections and recommended channels for each segment.
    ''',
    agent=marketing_agent,
    expected_output='Marketing campaign strategy with segment-specific recommendations'
)
```

#### 💰 Pricing Strategy Agent
```python
pricing_agent = Agent(
    role='Pricing Strategy Specialist', 
    goal='Optimize pricing for maximum profitability',
    backstory='Pricing analyst with expertise in jewelry market dynamics',
    tools=[RecommendationEngine()],
    verbose=True
)

pricing_task = Task(
    description='''
    Analyze our current pricing strategy and identify optimization opportunities.
    Consider market elasticity and competitive positioning.
    ''',
    agent=pricing_agent,
    expected_output='Pricing optimization recommendations with impact projections'
)
```

### Crew Assembly
```python
# Criar crew com múltiplos agentes
business_crew = Crew(
    agents=[sales_agent, marketing_agent, pricing_agent],
    tasks=[sales_task, campaign_task, pricing_task],
    verbose=2
)

# Executar análise completa
result = business_crew.kickoff()
```

---

## 💡 Exemplos Práticos

### Exemplo 1: Análise Rápida de Vendas
```python
from src.insights.tools.advanced.recommendation_engine import RecommendationEngine

# Inicializar engine
engine = RecommendationEngine()

# Análise rápida de produtos
result = engine._run(
    recommendation_type="product_recommendations", 
    data_csv="data/vendas.csv",
    target_segment="all",
    recommendation_count=5,
    confidence_threshold=0.7
)

print(f"📊 Resultado: {len(result)} caracteres de análise")
```

### Exemplo 2: Segmentação de Clientes VIP
```python
# Focar em clientes VIP para campanha especial
vip_analysis = engine._run(
    recommendation_type="customer_targeting",
    data_csv="data/vendas.csv", 
    target_segment="vip",
    recommendation_count=15,
    confidence_threshold=0.8,
    enable_detailed_analysis=True
)

# Resultado inclui scoring detalhado e ações recomendadas
```

### Exemplo 3: Otimização de Preços Sazonal
```python
# Análise de otimização para período específico
pricing_optimization = engine._run(
    recommendation_type="pricing_optimization",
    data_csv="data/vendas.csv",
    target_segment="all",
    confidence_threshold=0.75
)

# Identifica oportunidades de ajuste de preço
```

### Exemplo 4: Pipeline Completo de Análise
```python
def run_complete_analysis(data_file):
    """Executa análise completa do negócio."""
    engine = RecommendationEngine()
    
    analysis_types = [
        "product_recommendations",
        "customer_targeting", 
        "pricing_optimization",
        "inventory_suggestions",
        "marketing_campaigns",
        "strategic_actions"
    ]
    
    results = {}
    for analysis in analysis_types:
        results[analysis] = engine._run(
            recommendation_type=analysis,
            data_csv=data_file,
            target_segment="all",
            recommendation_count=10,
            confidence_threshold=0.7
        )
    
    return results

# Usar
complete_analysis = run_complete_analysis("data/vendas.csv")
```

---

## 📚 API Reference

### RecommendationInput Schema
```python
class RecommendationInput(BaseModel):
    recommendation_type: str  # Tipo de análise
    data_csv: str = "data/vendas.csv"  # Caminho do arquivo
    target_segment: str = "all"  # Segmento alvo
    recommendation_count: int = 10  # Número de recomendações
    confidence_threshold: float = 0.7  # Limiar de confiança
    enable_detailed_analysis: bool = True  # Análise detalhada
```

### Tipos de Recomendação Válidos
```python
VALID_RECOMMENDATION_TYPES = [
    "product_recommendations",
    "customer_targeting", 
    "pricing_optimization",
    "inventory_suggestions",
    "marketing_campaigns",
    "strategic_actions"
]
```

### Segmentos Suportados
```python
VALID_SEGMENTS = ["all", "vip", "high_value", "potential", "at_risk"]
```

### Método Principal
```python
def _run(
    self,
    recommendation_type: str,
    data_csv: str = "data/vendas.csv",
    target_segment: str = "all", 
    recommendation_count: int = 10,
    confidence_threshold: float = 0.7,
    enable_detailed_analysis: bool = True
) -> str:
    """
    Executa análise de recomendação.
    
    Returns:
        str: JSON string com resultados da análise
    """
```

---

## ⚡ Performance e Monitoramento

### Métricas de Performance Típicas
```
📊 Benchmark Atual (dados reais - 4,461 registros):
- product_recommendations: ~0.37s
- customer_targeting: ~0.10s  
- pricing_optimization: ~0.09s
- inventory_suggestions: ~0.11s
- marketing_campaigns: ~0.09s
- strategic_actions: ~0.09s

Tempo médio por execução: ~0.14s
```

### Cache Sistema
O sistema implementa cache inteligente:
- Hash dos dados para detectar mudanças
- Cache por tipo de análise
- Invalidação automática quando dados mudam

### Monitoramento de Qualidade dos Dados
```python
# Verificar qualidade dos dados
quality_assessment = engine._assess_data_quality(df)
print(f"Score de Completude: {quality_assessment['completeness_score']:.1%}")
```

### Logs e Debugging
```python
import logging
logging.basicConfig(level=logging.INFO)

# Executar com logs detalhados
result = engine._run("product_recommendations", enable_detailed_analysis=True)
```

---

## 🔧 Troubleshooting

### Problemas Comuns

#### ❌ ImportError: Módulo não encontrado
```bash
# Solução: Adicionar ao PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/insights-ai"

# Ou no código Python:
import sys
sys.path.append('/path/to/insights-ai')
```

#### ❌ FileNotFoundError: Arquivo de dados não encontrado
```python
# Verificar se arquivo existe
import os
if not os.path.exists("data/vendas.csv"):
    print("❌ Arquivo não encontrado")
    
# Usar caminho absoluto se necessário
result = engine._run(
    recommendation_type="product_recommendations",
    data_csv="/caminho/completo/para/vendas.csv"
)
```

#### ❌ ValueError: Dados insuficientes
```python
# Verificar tamanho dos dados
df = pd.read_csv("data/vendas.csv", sep=';')
print(f"Registros: {len(df)}")
print(f"Clientes únicos: {df['Codigo_Cliente'].nunique()}")

# Mínimo recomendado: 50+ registros, 10+ clientes
```

#### ❌ MemoryError: Dados muito grandes
```python
# Para datasets grandes (>100k registros):
# 1. Filtrar por período
df_filtered = df[df['Data'] >= '2024-01-01']

# 2. Usar amostragem
df_sample = df.sample(n=10000, random_state=42)
```

### Performance Issues

#### 🐌 Execução lenta
```python
# Verificar tamanho dos dados
# Considerar sampling para datasets grandes
# Verificar qualidade dos dados

# Usar cache quando possível
# Executar análises em lotes durante horários de baixo uso
```

#### 💾 Alto uso de memória
```python
# Processar dados em chunks
# Limpar cache periodicamente: engine._cache.clear()
# Usar tipos de dados otimizados (categorical, etc.)
```

---

## 📈 Roadmap e Melhorias Futuras

### ✅ Implementado (V2.0)
- Sistema completo de recomendações
- 6 tipos de análises
- ML avançado (SVD, TF-IDF, Isolation Forest)
- Integração CrewAI
- Cache inteligente
- Validação de dados
- Sistema de testes completo

### 🔄 Em Desenvolvimento (V2.1)
- API REST para integração externa
- Dashboard web interativo  
- Alertas automáticos
- Integração com bancos de dados
- Modelos de deep learning

### 🚀 Planejado (V3.0)
- Real-time recommendations
- A/B testing framework
- Advanced anomaly detection
- Multi-tenant support
- Cloud deployment

---

## 📞 Suporte

### Documentação Adicional
- `examples/recommendation_engine_demo.py` - Demonstração completa
- `src/tests/test_recommendation_engine_tool.py` - Suite de testes
- `docs/` - Documentação técnica

### Executar Testes
```bash
# Executar todos os testes
pytest src/tests/test_recommendation_engine_tool.py -v

# Executar teste específico
pytest src/tests/test_recommendation_engine_tool.py::TestRecommendationEngine::test_product_recommendations -v
```

### Demonstração Completa
```bash
# Executar demo interativa
python examples/recommendation_engine_demo.py
```

---

**RecommendationEngine V2.0** - Sistema Inteligente de Recomendações  
Desenvolvido para joalherias | Otimizado para CrewAI | ML Avançado  
© 2025 Insights AI Team 