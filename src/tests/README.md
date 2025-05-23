# 🧪 Insights-AI Testing Suite

Suite completa de testes para validar todas as ferramentas e funcionalidades do projeto Insights-AI.

## 📋 Visão Geral

A suite de testes é composta por 10 módulos especializados que validam diferentes aspectos do sistema:

### 🔧 Testes de Ferramentas Básicas
- **`test_sql_query_tool.py`** - Testa conexão SQL Server e queries
- **`test_kpi_calculator_tool.py`** - Valida cálculos de KPIs críticos
- **`test_prophet_tool.py`** - Testa forecasting com Facebook Prophet
- **`test_statistical_analysis_tool.py`** - Valida análises estatísticas
- **`test_visualization_tool.py`** - Testa geração de gráficos e dashboards

### 🔬 Testes Avançados
- **`test_advanced_tools.py`** - Testa ferramentas de IA avançadas
- **`test_integration.py`** - Valida integração entre ferramentas
- **`test_performance.py`** - Testes de performance e stress
- **`test_data_validation.py`** - Valida qualidade e integridade dos dados

### 🎯 Coordenação
- **`test_main.py`** - Coordenador principal e interface de linha de comando

## 🚀 Como Usar

### Executar Todos os Testes
```bash
# Execução completa
python src/tests/test_main.py

# Modo verboso (recomendado)
python src/tests/test_main.py --verbose

# Testes rápidos (para desenvolvimento)
python src/tests/test_main.py --quick --verbose
```

### Executar Testes Específicos
```bash
# Testar apenas SQL Query Tool
python src/tests/test_main.py --tool sql --verbose

# Testar apenas KPI Calculator
python src/tests/test_main.py --tool kpi --verbose

# Testar forecasting Prophet
python src/tests/test_main.py --tool prophet --verbose

# Testar análises estatísticas
python src/tests/test_main.py --tool stats --verbose

# Testar visualizações
python src/tests/test_main.py --tool viz --verbose

# Testar ferramentas avançadas
python src/tests/test_main.py --tool advanced --verbose

# Testar validação de dados
python src/tests/test_main.py --tool data --verbose

# Testar integração entre ferramentas
python src/tests/test_main.py --tool integration --verbose

# Testar performance e stress
python src/tests/test_main.py --tool performance --verbose
```

### Executar Teste Individual
```bash
# Executar um arquivo de teste específico
python src/tests/test_kpi_calculator_tool.py
python src/tests/test_prophet_tool.py
python src/tests/test_integration.py
```

## 📊 Tipos de Testes

### 🧪 Testes Unitários
- Validam funcionalidades individuais de cada ferramenta
- Verificam inputs/outputs corretos
- Testam tratamento de erros

### 🔗 Testes de Integração
- Validam fluxo de dados entre ferramentas
- Testam compatibilidade de formatos
- Verificam pipeline completo

### ⚡ Testes de Performance
- Escalabilidade com diferentes volumes de dados
- Uso de memória e detecção de vazamentos
- Tempo de execução e throughput

### 🔍 Testes de Validação
- Qualidade e integridade dos dados
- Completude e consistência
- Formatos e encodings

## 📈 Interpretando Resultados

### ✅ Status de Sucesso
- **SUCCESS**: Teste passou completamente
- **FAILED**: Teste falhou mas não houve exceção
- **ERROR**: Exceção durante execução
- **SLOW**: Funciona mas demorou muito (>60s)

### 📊 Relatório Final
```
📊 RELATÓRIO FINAL DOS TESTES
================================================================================
📈 Total de Testes: 8
✅ Sucessos: 7
❌ Falhas: 1
⏱️  Tempo Total: 45.23s
📊 Taxa de Sucesso: 87.5%

📋 DETALHES POR TESTE:
------------------------------------------------------------
✅ Data Validation               PASS      (2.15s)
✅ SQL Query Tool               PASS      (1.23s)
✅ KPI Calculator Tool          PASS      (8.45s)
✅ Statistical Analysis Tool    PASS      (5.67s)
❌ Prophet Forecast Tool        FAIL      (0.05s)
    🔍 Erro: Prophet library não está instalada
✅ Visualization Tool           PASS      (12.34s)
✅ Advanced Tools Suite         PASS      (15.67s)
✅ Integration Tests            PASS      (3.89s)
```

## 🔧 Configuração e Dependências

### Dependências Básicas
```bash
pip install pytest pandas numpy scipy scikit-learn matplotlib seaborn
```

### Dependências Opcionais (para testes completos)
```bash
pip install prophet plotly psutil
```

### Estrutura de Dados
Os testes esperam um arquivo `data/vendas.csv` com as colunas:
- `Data` - Data da transação
- `Total_Liquido` - Valor da venda
- `Quantidade` - Quantidade vendida
- `Codigo_Produto` - Código do produto
- `Codigo_Cliente` - Código do cliente (opcional)

## 🐛 Debugging e Troubleshooting

### Problemas Comuns

#### 1. **Arquivo de dados não encontrado**
```
❌ Erro: Arquivo data/vendas.csv não encontrado
```
**Solução**: Certifique-se que o arquivo `data/vendas.csv` existe no diretório raiz do projeto.

#### 2. **Dependências ausentes**
```
❌ Erro: Prophet library não está instalada
```
**Solução**: 
```bash
pip install prophet
```

#### 3. **Erro de encoding**
```
❌ Erro na conversão da coluna Total_Liquido
```
**Solução**: Verifique se o CSV está em UTF-8 e usa `;` como separador.

#### 4. **Timeout em testes**
```
⚠️ Warning: Teste demorou mais de 60 segundos
```
**Solução**: Use `--quick` para testes mais rápidos durante desenvolvimento.

### Modo Debug
Para mais detalhes sobre falhas:
```bash
python src/tests/test_main.py --verbose --tool [ferramenta_com_problema]
```

## 📝 Adicionando Novos Testes

### Estrutura de um Teste
```python
def test_minha_ferramenta(verbose=False, quick=False):
    \"\"\"Teste da Minha Ferramenta\"\"\"
    result = {
        'success': False,
        'details': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        if verbose:
            print("🔧 Testando Minha Ferramenta...")
        
        # Seus testes aqui
        
        result['success'] = True
        return result
        
    except Exception as e:
        result['errors'].append(f"Erro: {str(e)}")
        return result
```

### Adicionando ao test_main.py
1. Importe sua função de teste
2. Adicione à `test_suite`
3. Adicione ao `tool_map`

## 🎯 Melhores Práticas

### Para Desenvolvimento
- Use `--quick --verbose` durante desenvolvimento
- Teste ferramentas individuais primeiro
- Corrija um erro por vez

### Para CI/CD
- Execute suite completa sem flags
- Configure timeout apropriado (10-15 minutos)
- Monitore taxa de sucesso >90%

### Para Produção
- Execute testes de integração regularmente
- Monitore testes de performance
- Configure alertas para falhas críticas

## 📞 Suporte

Em caso de problemas:
1. Execute com `--verbose` para mais detalhes
2. Verifique dependências e dados
3. Teste ferramentas individualmente
4. Consulte logs de erro específicos

---

*Documentação atualizada: Dezembro 2024*
