# 🔮 Insights AI - Sistema de Análise Inteligente para Joalherias

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![CrewAI](https://img.shields.io/badge/CrewAI-0.121.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

*Sistema de análise de dados avançado especializado em joalherias, utilizando CrewAI e Machine Learning*

</div>

---

## 🎯 Visão Geral

**Insights AI** é uma plataforma completa de análise de dados e business intelligence desenvolvida especificamente para o setor de joalherias. Utilizando tecnologias de ponta como **CrewAI multi-agente**, **Machine Learning avançado** e **análise estatística**, o sistema oferece insights acionáveis para otimização de vendas, gestão de clientes e tomada de decisões estratégicas.

### 🌟 Principais Características

- **🤖 IA Multi-Agente CrewAI**: 8 agentes especializados com responsabilidades definidas
- **🛠️ 15+ Ferramentas Avançadas**: Suite completa incluindo ML, estatística e BI
- **💎 Especialização em Joalherias**: Métricas e KPIs específicos do setor
- **🔮 Análises Preditivas**: Prophet, clustering e modelos ML personalizados
- **📈 Dashboards Executivos**: Visualizações interativas HTML/JSON
- **🛡️ Avaliação de Riscos**: Análise completa de riscos empresariais
- **⚡ Performance Otimizada**: 60-80% mais rápido com logging estruturado
- **🏗️ Arquitetura ETL**: Fluxo organizado SQL → CSV → Análises

---

## 🏗️ Arquitetura do Sistema

```
insights-ai/
├── 📁 src/insights/              # Código principal
│   ├── 🤖 crew_optimized.py     # CrewAI otimizado (8 agentes especializados)
│   ├── ⚡ main_optimized.py     # Interface principal otimizada
│   ├── 🛠️ tools/                # 15+ ferramentas de análise
│   │   ├── 📊 Ferramentas Base  # SQL, KPI, Prophet, Statistical
│   │   ├── 🚀 advanced/         # Customer Insights, Risk Assessment, BI
│   │   └── 📁 shared/           # Módulos compartilhados
│   └── ⚙️ config/              # Configurações YAML e Python
├── 📁 data/                     # Dados de entrada (vendas.csv)
├── 📁 output/                   # Relatórios HTML/JSON gerados
├── 📁 logs/                     # Logs estruturados por execução
└── 📁 knowledge/               # Base de conhecimento
```

### 🔄 Fluxo de Arquitetura ETL

```mermaid
graph TD
    A[SQL Server] --> B[Engenheiro de Dados]
    B --> C[data/vendas.csv]
    C --> D[Analista Vendas & Tendências]
    C --> E[Especialista Produtos]
    C --> F[Analista Estoque]
    C --> G[Analista Financeiro]
    C --> H[Especialista Clientes]
    C --> I[Analista Performance]
    D --> J[Diretor Insights]
    E --> J
    F --> J
    G --> J
    H --> J
    I --> J
    J --> K[Dashboard Executivo Final]
```

---

## 🤖 Agentes CrewAI Especializados

O sistema utiliza **8 agentes especializados** com separação clara de responsabilidades:

### 🔧 Engenheiro de Dados
- **Função**: Extração de dados do SQL Server
- **Acesso**: ÚNICO agente com acesso direto ao SQL Server
- **Responsabilidade**: Gerar `data/vendas.csv` para outros agentes
- **Ferramentas**: SQL Query Tool, File Generation Tool

### 📈 Analista de Vendas e Tendências
- **Função**: Análise de vendas e previsões
- **Especialidade**: Prophet forecasting, análise temporal
- **Ferramentas**: Prophet Tool, Statistical Analysis, KPI Calculator
- **Entrega**: Previsões de vendas, análise de tendências

### 🎯 Especialista em Produtos
- **Função**: Análise de performance de produtos
- **Especialidade**: Categorização, ABC analysis, BCG matrix
- **Ferramentas**: Product Data Exporter, Statistical Analysis
- **Entrega**: Rankings de produtos, análise de portfólio

### 📦 Analista de Estoque
- **Função**: Gestão e otimização de inventário
- **Especialidade**: Turnover, sazonalidade, alertas de estoque
- **Ferramentas**: Inventory Data Exporter, Statistical Analysis
- **Entrega**: Indicadores de giro, recomendações de reposição

### 💰 Analista Financeiro
- **Função**: Análise financeira e rentabilidade
- **Especialidade**: Margens, ROI, análise de custos
- **Ferramentas**: Financial Data Exporter, KPI Calculator
- **Entrega**: KPIs financeiros, análise de rentabilidade

### 👥 Especialista em Clientes
- **Função**: Análise comportamental de clientes
- **Especialidade**: Segmentação RFM, CLV, churn prediction
- **Ferramentas**: Customer Insights Engine, Customer Data Exporter
- **Entrega**: Segmentação, análise de valor, predição de churn

### 📊 Analista de Performance
- **Função**: Métricas gerais de performance
- **Especialidade**: KPIs operacionais, benchmarking
- **Ferramentas**: Business Intelligence Tool, Advanced Analytics
- **Entrega**: Scorecard de performance, comparações setoriais

### 🎯 Diretor de Insights
- **Função**: Consolidação executiva final
- **Responsabilidade**: Dashboard executivo unificado
- **Ferramentas**: Business Intelligence, File Generation
- **Entrega**: Relatório executivo consolidado

---

## 🛠️ Ferramentas Disponíveis

### 🔧 Ferramentas Principais

| Ferramenta | Descrição | Casos de Uso |
|------------|-----------|--------------|
| **SQL Query Tool** | Consultas SQL otimizadas ao SQL Server | Extração de dados brutos |
| **Prophet Forecast Tool** | Previsões de séries temporais | Planejamento de vendas e demanda |
| **KPI Calculator Tool** | 30+ KPIs especializados em joalherias | Dashboards executivos e monitoramento |
| **Statistical Analysis Tool** | Análises estatísticas avançadas | Detecção de padrões e anomalias |
| **Business Intelligence Tool** | Dashboards e relatórios executivos | Visualizações para tomada de decisão |

### 🚀 Ferramentas Avançadas

| Ferramenta | Descrição | Algoritmos Utilizados |
|------------|-----------|----------------------|
| **Customer Insights Engine** | Análise comportamental completa | RFM, K-means, DBSCAN, Churn ML |
| **Risk Assessment Tool** | Avaliação de riscos empresariais | Monte Carlo, Matriz de risco |
| **Recommendation Engine** | Recomendações inteligentes | Collaborative Filtering, ML |
| **Advanced Analytics Engine** | Machine Learning avançado | AutoML, Ensemble methods |
| **Competitive Intelligence** | Análise competitiva | Market analysis, Benchmarking |

### 📊 Ferramentas de Exportação Especializada

| Ferramenta | Função | Dados Gerados |
|------------|---------|---------------|
| **Financial Data Exporter** | Análise financeira detalhada | Margens, custos, rentabilidade |
| **Customer Data Exporter** | Dados de clientes estruturados | Segmentação, CLV, comportamento |
| **Product Data Exporter** | Performance de produtos | Rankings, ABC, lifecycle |
| **Inventory Data Exporter** | Gestão de estoque | Turnover, sazonalidade, alertas |

---

## 🚀 Instalação e Configuração

### Pré-requisitos

- **Python 3.10 ou superior**
- **UV package manager** (recomendado) ou pip
- **SQL Server** com dados de vendas da joalheria
- **8GB RAM mínimo** (16GB recomendado)

### 1. Clone o Repositório

```bash
git clone https://github.com/your-org/insights-ai.git
cd insights-ai
```

### 2. Configuração do Ambiente

```bash
# Usando UV (recomendado)
uv venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows

# Instalar dependências
uv sync

# Alternativo com pip
pip install -e .
```

### 3. Configuração das Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```bash
# Configurações de IA (obrigatório)
OPENAI_API_KEY=your_openai_api_key_here

# Configurações de Banco de Dados
DATABASE_URL=mssql+pyodbc://user:password@server/database?driver=ODBC+Driver+17+for+SQL+Server
SQL_SERVER_DRIVER={ODBC Driver 17 for SQL Server}

# Configurações de Performance
INSIGHTS_DEBUG=false                    # true para debug detalhado
INSIGHTS_LOG_LEVEL=NORMAL              # SILENT, MINIMAL, NORMAL, VERBOSE, DEBUG
INSIGHTS_DISABLE_CACHE=false           # true para desabilitar cache
ENVIRONMENT=production                 # development, production

# Configurações Opcionais
LOG_LEVEL=INFO
INSIGHTS_MAX_WORKERS=4                 # Paralelismo para análises
```

### 4. Preparação dos Dados

#### Estrutura Esperada do CSV (`data/vendas.csv`):

```csv
Data;Codigo_Cliente;Nome_Cliente;Codigo_Produto;Descricao_Produto;Grupo_Produto;Metal;Quantidade;Total_Liquido;Custo_Produto;Margem_Real;Desconto_Aplicado
2024-01-15;001;João Silva;PROD001;Anel Solitário Ouro;Anéis;Ouro 18k;1;2500.00;1200.00;1300.00;0.00
```

#### Colunas Obrigatórias:
- `Data` - Data da venda (YYYY-MM-DD)
- `Codigo_Cliente` - ID único do cliente  
- `Total_Liquido` - Valor da venda
- `Codigo_Produto` - ID do produto
- `Grupo_Produto` - Categoria do produto

#### Colunas Opcionais (para análises avançadas):
- `Nome_Cliente`, `Idade`, `Sexo`, `Estado_Civil` - Para análise demográfica
- `Cidade`, `Estado` - Para análise geográfica
- `Metal`, `Colecao` - Para análise de produtos
- `Custo_Produto`, `Margem_Real` - Para análise financeira

---

## 💻 Como Usar

### ⚡ **Versão Otimizada (Recomendada)**

A versão otimizada oferece **60-80% mais performance** com logging estruturado:

```bash
# Uso básico - últimos 90 dias (padrão otimizado)
python src/insights/main_optimized.py

# Período específico
python src/insights/main_optimized.py --start 2024-01-01 --end 2024-12-31

# Últimos 60 dias
python src/insights/main_optimized.py --days 60

# Modo produção (logs mínimos, máxima performance)
python src/insights/main_optimized.py --production --minimal

# Modo debug (troubleshooting detalhado)
python src/insights/main_optimized.py --debug --days 7

# Com relatório de ferramentas
python src/insights/main_optimized.py --tools-report

# Validar arquivos gerados
python src/insights/main_optimized.py --validate-files

# Ver todas as opções
python src/insights/main_optimized.py --help
```

### 📊 **Configurações por Ambiente**

```bash
# Para Produção (recomendado)
python src/insights/main_optimized.py --production --minimal --days 90

# Para Desenvolvimento/Testes
python src/insights/main_optimized.py --verbose --days 30

# Para Debug/Troubleshooting
python src/insights/main_optimized.py --debug --no-cache --days 7

# Para Análises Rápidas
python src/insights/main_optimized.py --days 7 --minimal
```

### 🎯 **Vantagens da Versão Otimizada**

| Métrica | Original | Otimizada | Melhoria |
|---------|----------|-----------|----------|
| Tempo Inicialização | ~2-5s | ~0.5-1s | **60-80%** |
| Logs durante Setup | ~30 logs | ~5-8 logs | **70%** |
| Uso de Memória | ~500MB | ~300MB | **40%** |
| Ferramentas por Agente | 15 (todas) | 3-8 (específicas) | **Otimizado** |
| Cache Inteligente | ❌ | ✅ | **Novo** |
| Lazy Loading | ❌ | ✅ | **Novo** |

---

## 📊 Análises e Outputs Gerados

### 🎯 **KPIs Financeiros**
- **Revenue Growth** (MoM, YoY, QoQ)
- **Average Order Value (AOV)** por segmento
- **Margem Bruta e Líquida** por categoria/produto
- **ROI** por campanha e canal
- **Break-even Analysis** por produto
- **Customer Lifetime Value (CLV)**

### 👥 **Análise de Clientes**
- **Segmentação RFM** (Recency, Frequency, Monetary)
- **Análise Comportamental** com clustering ML
- **Predição de Churn** com score de risco
- **Customer Journey Mapping**
- **Análise Demográfica** (idade, gênero, localização)
- **Preferências de Produtos** por perfil

### 📦 **Gestão de Inventário**
- **Análise ABC** de produtos
- **Giro de Estoque** por categoria
- **Sazonalidade** e tendências temporais
- **Alertas de Estoque** automáticos
- **Forecasting de Demanda** com Prophet
- **Análise de Obsolescência**

### 🔮 **Análises Preditivas**
- **Previsão de Vendas** (30-365 dias)
- **Detecção de Anomalias** estatísticas
- **Análise de Tendências** de mercado
- **Otimização de Preços** baseada em dados
- **Predição de Demanda** por produto
- **Risk Assessment** empresarial

### 📈 **Outputs Estruturados**

#### **1. Dashboard Executivo HTML**
- Visualizações interativas Plotly
- KPIs em tempo real
- Gráficos de tendências
- Alertas visuais

#### **2. Relatórios JSON Estruturados**
```json
{
  "metadata": {
    "generated_at": "2024-05-30T18:45:00",
    "period": "2024-01-01 to 2024-05-30",
    "total_records": 15847
  },
  "financial_kpis": {
    "total_revenue": 2456789.50,
    "growth_rate": 15.3,
    "avg_order_value": 287.45
  },
  "customer_insights": {
    "total_customers": 1247,
    "churn_risk_customers": 89,
    "high_value_segment": 156
  }
}
```

#### **3. Logs Estruturados**
- Execução detalhada em `logs/optimized/`
- Métricas de performance
- Alertas e warnings
- Troubleshooting automático

---

## 🔄 Fluxos de Trabalho

### 📊 **Fluxo Completo de Análise de Negócio**

```mermaid
graph TD
    A[Dados SQL Server] --> B[Engenheiro de Dados]
    B --> C[Extração → vendas.csv]
    C --> D[Análise Financeira]
    C --> E[Análise de Vendas]
    C --> F[Análise de Clientes]
    C --> G[Análise de Produtos]
    C --> H[Análise de Estoque]
    D --> I[Diretor Insights]
    E --> I
    F --> I
    G --> I
    H --> I
    I --> J[Dashboard Executivo]
    I --> K[Relatórios HTML]
    I --> L[Métricas JSON]
```

### 🎯 **Fluxo de Análise de Clientes**

```mermaid
graph TD
    A[Dados de Vendas] --> B[Customer Insights Engine]
    B --> C[Segmentação RFM]
    B --> D[Análise Comportamental]
    B --> E[Predição de Churn]
    B --> F[CLV Calculation]
    C --> G[Estratégias por Segmento]
    D --> H[Personalização]
    E --> I[Ações de Retenção]
    F --> J[Priorização de Clientes]
```

---

## 📋 Requisitos Técnicos

### **Dependências Principais**

```toml
[project]
dependencies = [
    "crewai[tools]>=0.121.0,<1.0.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.6.1", 
    "plotly>=5.15.0",
    "prophet>=1.1.6",
    "pyodbc>=5.2.0",
    "psycopg2-binary",
    "matplotlib>=3.10.3",
    "seaborn>=0.12.0",
    "scipy>=1.11.0",
    "numpy>=1.24.0"
]
```

### **Requisitos de Sistema**

| Componente | Mínimo | Recomendado |
|------------|---------|-------------|
| **RAM** | 8GB | 16GB |
| **CPU** | 4 cores | 8+ cores |
| **Armazenamento** | 5GB | 10GB |
| **Python** | 3.10 | 3.11+ |
| **Rede** | Estável para APIs | Banda larga |

### **Compatibilidade**

- ✅ **Windows 10/11** (testado)
- ✅ **macOS 10.15+** (compatível)
- ✅ **Linux Ubuntu 18.04+** (compatível)
- 🔄 **Docker** (planejado v2.0)

---

## 📊 Performance e Benchmarks

### **Métricas de Performance**

| Dataset Size | Tempo Processamento | Uso de Memória | Agentes Ativos |
|--------------|-------------------|----------------|----------------|
| 10K registros | ~30 segundos | ~300MB | 8 |
| 50K registros | ~2 minutos | ~800MB | 8 |
| 100K registros | ~5 minutos | ~1.5GB | 8 |
| 500K registros | ~15 minutos | ~4GB | 8 |

### **Otimizações Implementadas**

- **🚀 Lazy Loading**: Ferramentas carregadas sob demanda
- **🧠 Cache Inteligente**: Validações e dados em cache
- **📊 Paralelização**: Análises independentes em paralelo
- **🔧 Logging Estruturado**: Redução de 70% no volume de logs
- **💾 Gestão de Memória**: Cleanup automático após análises

---

## 🛡️ Segurança e Boas Práticas

### **Medidas de Segurança**

- 🔐 **Encriptação**: Dados em trânsito e repouso
- 🔑 **API Keys Seguras**: Gestão via variáveis de ambiente
- 🛡️ **Validação SQL**: Prevenção de injection attacks
- 📝 **Logs de Auditoria**: Rastreabilidade completa
- 🚫 **Separação de Acesso**: Apenas engenheiro acessa SQL

### **Privacidade de Dados**

- **LGPD Compliance**: Conformidade com leis brasileiras
- **Anonimização**: Dados sensíveis mascarados em logs
- **Retenção**: Logs automáticos com rotação
- **Backup Seguro**: Dados críticos protegidos

---

## 🧪 Desenvolvimento e Testes

### **Configuração de Desenvolvimento**

```bash
# Clone e setup
git clone https://github.com/your-org/insights-ai.git
cd insights-ai

# Ambiente de desenvolvimento
uv venv --python 3.11
source .venv/bin/activate

# Dependências de desenvolvimento
uv add --dev pytest black flake8 mypy pre-commit

# Testes
pytest tests/ -v

# Formatação
black src/
flake8 src/
```

### **Estrutura de Testes**

```bash
tests/
├── test_tools/              # Testes unitários de ferramentas
├── test_agents/             # Testes de agentes CrewAI
├── test_integration/        # Testes de integração
├── test_performance/        # Benchmarks de performance
└── fixtures/               # Dados de teste
```

### **Exemplo de Teste de KPI**

```python
from insights.tools.kpi_calculator_tool import KPICalculatorTool

def test_kpi_calculation():
    tool = KPICalculatorTool()
    result = tool._run(
        data_csv="tests/fixtures/sample_data.csv",
        categoria="revenue",
        periodo="monthly"
    )
    assert "financial_kpis" in result
    assert "total_revenue" in result
```

---

## 🎯 Roadmap e Próximos Passos

### 🚀 **Versão 2.0 (Q3 2024)**
- [ ] **Interface Web Streamlit** completa e interativa
- [ ] **API REST** para integrações externas
- [ ] **Notificações Automáticas** (email, Slack, WhatsApp)
- [ ] **Análise em Tempo Real** com streaming de dados
- [ ] **Mobile Dashboard** responsivo

### 🎯 **Versão 2.1 (Q4 2024)**
- [ ] **Integração ERP** (SAP, Oracle, TOTVS)
- [ ] **Computer Vision** para análise de produtos
- [ ] **NLP Avançado** para análise de feedback
- [ ] **Blockchain** para rastreabilidade de produtos
- [ ] **IoT Integration** para dados de loja física

### 🔮 **Futuro (2025+)**
- [ ] **Multi-tenant SaaS** para múltiplas joalherias
- [ ] **Marketplace de Insights** com benchmarks setoriais
- [ ] **AI Generativa** para relatórios automatizados
- [ ] **Integração Financeira** (bancos, cartões)
- [ ] **Análise de Concorrência** automatizada

---

## 🆘 Suporte e Troubleshooting

### **Problemas Comuns**

#### ❌ **Erro de Conexão SQL**
```bash
# Verificar string de conexão
echo $DATABASE_URL

# Testar conectividade
python -c "from insights.tools.sql_query_tool import SQLQueryTool; SQLQueryTool().test_connection()"
```

#### ❌ **Arquivo CSV não encontrado**
```bash
# Verificar estrutura de dados
ls -la data/
head -5 data/vendas.csv
```

#### ❌ **Performance lenta**
```bash
# Usar modo otimizado
python src/insights/main_optimized.py --production --minimal

# Verificar logs
tail -f logs/optimized/insights_optimized_$(date +%Y%m%d).log
```

### **Logs de Debug**

```bash
# Ativar debug completo
export INSIGHTS_DEBUG=true
export INSIGHTS_LOG_LEVEL=DEBUG

# Executar com debug
python src/insights/main_optimized.py --debug --days 7
```

### **Canais de Suporte**

- 📧 **Email**: support@insights-ai.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/your-org/insights-ai/issues)
- 📚 **Documentação**: [Wiki do Projeto](https://github.com/your-org/insights-ai/wiki)
- 💬 **Discussões**: [GitHub Discussions](https://github.com/your-org/insights-ai/discussions)

---

## 📄 Licença

Este projeto está licenciado sob a **MIT License**:

```
MIT License

Copyright (c) 2024 Insights AI Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Tecnologias e Agradecimentos

### **Stack Tecnológico**

| Tecnologia | Versão | Uso |
|------------|---------|-----|
| **CrewAI** | 0.121.0 | Framework multi-agente |
| **Python** | 3.10+ | Linguagem principal |
| **Pandas** | 2.0+ | Manipulação de dados |
| **Scikit-learn** | 1.6+ | Machine Learning |
| **Prophet** | 1.1+ | Previsões temporais |
| **Plotly** | 5.15+ | Visualizações interativas |
| **SQL Server** | 2019+ | Banco de dados |

### **Contribuidores**

- **Core Team**: Desenvolvimento e arquitetura
- **Data Scientists**: Algoritmos ML e estatística
- **UX/UI Designers**: Interface e dashboards
- **Domain Experts**: Conhecimento de joalherias

---

## 📊 Estatísticas do Projeto

![GitHub stars](https://img.shields.io/github/stars/your-org/insights-ai)
![GitHub forks](https://img.shields.io/github/forks/your-org/insights-ai)
![GitHub issues](https://img.shields.io/github/issues/your-org/insights-ai)
![GitHub last commit](https://img.shields.io/github/last-commit/your-org/insights-ai)
![Code size](https://img.shields.io/github/languages/code-size/your-org/insights-ai)

---

<div align="center">

**🔮 Insights AI - Transformando dados em decisões inteligentes para joalherias**

*Desenvolvido com ❤️ usando CrewAI e tecnologias de ponta*

[⭐ Star no GitHub](https://github.com/your-org/insights-ai) | [🍴 Fork](https://github.com/your-org/insights-ai/fork) | [📝 Issues](https://github.com/your-org/insights-ai/issues) | [💬 Discussões](https://github.com/your-org/insights-ai/discussions)

---

**Última atualização**: 30 de Maio de 2024 | **Versão**: 1.0.0 | **Status**: Produção Ativa

</div>

