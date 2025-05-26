#!/usr/bin/env python3
"""
🧪 TESTE INDIVIDUAL - ENGENHEIRO DE DADOS
Script para testar isoladamente o agente engenheiro_dados e sua task.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Adicionar o diretório src ao path para importações
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Imports do CrewAI
from crewai import Agent, Crew, Process, Task, LLM

# Imports das ferramentas
from insights.tools.sql_query_tool import SQLServerQueryTool

# Load environment
load_dotenv()

# =============== CONFIGURAÇÃO DE LOGGING ===============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_engenheiro_individual.log')
    ]
)

logger = logging.getLogger(__name__)

# =============== CONFIGURAÇÃO DO LLM ===============
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    logger.error("❌ OPENROUTER_API_KEY não encontrada no ambiente!")
    sys.exit(1)

llm = LLM(
    model="openrouter/deepseek/deepseek-r1",
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# =============== CONFIGURAÇÃO DOS INPUTS DE TESTE ===============
def get_test_inputs():
    """Gerar inputs de teste padrão para o agente"""
    # Usar últimos 30 dias como padrão
    data_fim = datetime.now().strftime('%Y-%m-%d')
    data_inicio = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    return {
        'data_inicio': data_inicio,
        'data_fim': data_fim
    }

# =============== CONFIGURAÇÃO SIMPLIFICADA DOS AGENTES ===============
def create_engenheiro_dados_agent():
    """Criar o agente engenheiro_dados de forma simplificada"""
    logger.info("🔧 Criando agente Engenheiro de Dados...")
    
    # Instanciar ferramenta SQL
    sql_tool = SQLServerQueryTool()
    
    return Agent(
        role="Engenheiro de Dados Senior",
        goal="Extrair dados precisos do sistema ERP de joalherias e preparar datasets otimizados para análise",
        backstory="""Você é um especialista em engenharia de dados com 12 anos de experiência em sistemas ERP do varejo de luxo. 
        Com mestrado em Ciência da Computação e certificações em SQL Server, você desenvolveu pipelines de ETL 
        que reduziram o tempo de processamento em 75%. Sua especialidade é garantir a integridade dos dados 
        e otimizar consultas SQL para performance máxima, mesmo em sistemas com milhões de registros diários.""",
        verbose=True,
        llm=llm,
        tools=[sql_tool],
        allow_delegation=False,
        max_iter=5
    )

# =============== CONFIGURAÇÃO SIMPLIFICADA DAS TASKS ===============
def create_engenheiro_dados_task(agent, inputs):
    """Criar a task do engenheiro_dados de forma simplificada"""
    logger.info("📋 Criando task do Engenheiro de Dados...")
    
    data_inicio = inputs['data_inicio']
    data_fim = inputs['data_fim']
    
    description = f"""🔧 ENGENHARIA DE DADOS AVANÇADA COM FILTROS TEMPORAIS:

**IMPORTANTE: Use os inputs data_inicio={data_inicio} e data_fim={data_fim} fornecidos para filtrar os dados!**

1. **Extração Multi-Source com Filtro Temporal**:
   - **OBRIGATÓRIO**: Usar SQL Server Query Tool com os parâmetros:
     * date_start: {data_inicio} 
     * date_end: {data_fim}
     * output_format: "csv"
   - Exemplo de chamada: SQL Server Query Tool com date_start="{data_inicio}", date_end="{data_fim}", output_format="csv"
   - Validar integridade referencial entre fontes

2. **Análise de Qualidade dos Dados**:
   - Detectar outliers nos dados filtrados
   - Identificar gaps temporais entre {data_inicio} e {data_fim}
   - Aplicar validações de consistência no período especificado

3. **Transformações Básicas**:
   - Criar features derivadas (Preco_Unitario, Margem_Estimada, etc.)
   - Calcular métricas temporais para o período analisado
   - Aplicar limpeza básica nos dados

4. **Validação Final**:
   - Gerar relatório de qualidade de dados para o período {data_inicio} a {data_fim}
   - Documentar transformações aplicadas
   - Confirmar que os dados estão dentro do range temporal solicitado"""
    
    expected_output = f"""📋 DATASET EMPRESARIAL VALIDADO PARA O PERÍODO {data_inicio} A {data_fim}:

1. **Dados Limpos e Validados**:
   - DataFrame principal com dados filtrados entre {data_inicio} e {data_fim}
   - Outliers identificados e tratados no período especificado
   - Campos derivados calculados
   - Metadados de qualidade de dados temporais

2. **Relatório de Qualidade Temporal**:
   - Estatísticas descritivas por dimensão no período
   - Gaps identificados entre {data_inicio} e {data_fim}
   - Confirmação do range temporal dos dados extraídos
   - Recomendações para coleta futura

3. **Features Engineered Temporais**:
   - Campos calculados otimizados para análise temporal
   - Segmentações automáticas (preço, frequência, etc.) no período
   - Métricas de evolução temporal quando aplicável

**VALIDAÇÃO CRÍTICA**: Confirmar que todos os dados estão dentro do período {data_inicio} a {data_fim}"""
    
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        output_file=f'output/teste_engenheiro_dados_{data_inicio}_{data_fim}.md'
    )

# =============== FUNÇÃO PRINCIPAL DE TESTE ===============
def test_engenheiro_dados():
    """Função principal para testar o agente engenheiro_dados"""
    logger.info("🚀 INICIANDO TESTE INDIVIDUAL - ENGENHEIRO DE DADOS")
    logger.info("=" * 60)
    
    try:
        # 1. Preparar inputs de teste
        logger.info("📋 Preparando inputs de teste...")
        inputs = get_test_inputs()
        logger.info(f"📅 Período de teste: {inputs['data_inicio']} até {inputs['data_fim']}")
        
        # 2. Criar agente
        logger.info("🔧 Criando agente...")
        agent = create_engenheiro_dados_agent()
        logger.info("✅ Agente criado com sucesso")
        
        # 3. Criar task
        logger.info("📋 Criando task...")
        task = create_engenheiro_dados_task(agent, inputs)
        logger.info("✅ Task criada com sucesso")
        
        # 4. Criar crew mínimo
        logger.info("🚀 Criando crew mínimo...")
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
            memory=False,
            max_rpm=10  # Limite conservador para testes
        )
        logger.info("✅ Crew criado com sucesso")
        
        # 5. Executar teste
        logger.info("🎯 Executando teste...")
        logger.info("⏰ Iniciando execução...")
        start_time = datetime.now()
        
        result = crew.kickoff(inputs=inputs)
        
        end_time = datetime.now()
        execution_time = end_time - start_time
        
        # 6. Exibir resultados
        logger.info("✅ TESTE CONCLUÍDO COM SUCESSO!")
        logger.info("=" * 60)
        logger.info(f"⏱️ Tempo de execução: {execution_time}")
        logger.info(f"📊 Tipo do resultado: {type(result)}")
        logger.info(f"📝 Tamanho do output: {len(str(result))} caracteres")
        
        # Salvar resultado completo
        result_file = f'output/resultado_teste_engenheiro_{inputs["data_inicio"]}_{inputs["data_fim"]}.txt'
        os.makedirs('output', exist_ok=True)
        
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"TESTE INDIVIDUAL - ENGENHEIRO DE DADOS\n")
            f.write(f"Data/Hora: {datetime.now()}\n")
            f.write(f"Período: {inputs['data_inicio']} até {inputs['data_fim']}\n")
            f.write(f"Tempo de execução: {execution_time}\n")
            f.write("\n" + "="*60 + "\n")
            f.write("RESULTADO:\n")
            f.write(str(result))
        
        logger.info(f"💾 Resultado salvo em: {result_file}")
        
        # Exibir preview do resultado
        logger.info("\n📋 PREVIEW DO RESULTADO:")
        logger.info("-" * 40)
        preview = str(result)[:500] + "..." if len(str(result)) > 500 else str(result)
        logger.info(preview)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ERRO NO TESTE: {e}")
        logger.exception("Detalhes do erro:")
        return False

# =============== FUNÇÃO DE VALIDAÇÃO DE FERRAMENTAS ===============
def validate_tools():
    """Validar se as ferramentas necessárias estão funcionando"""
    logger.info("🔧 Validando ferramentas...")
    
    try:
        # Testar SQL Tool
        sql_tool = SQLServerQueryTool()
        logger.info("✅ SQLServerQueryTool instanciada com sucesso")
        
        # Verificar se tem método _run
        if hasattr(sql_tool, '_run'):
            logger.info("✅ Método _run encontrado")
        else:
            logger.warning("⚠️ Método _run não encontrado")
            
        # Listar métodos disponíveis
        methods = [method for method in dir(sql_tool) 
                  if not method.startswith('_') and callable(getattr(sql_tool, method))]
        logger.info(f"🔧 Métodos disponíveis: {methods[:5]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro na validação de ferramentas: {e}")
        return False

# =============== MENU INTERATIVO ===============
def interactive_menu():
    """Menu interativo para diferentes opções de teste"""
    print("\n🧪 TESTE INDIVIDUAL - ENGENHEIRO DE DADOS")
    print("=" * 50)
    print("1. Executar teste com período padrão (últimos 30 dias)")
    print("2. Executar teste com período customizado")
    print("3. Validar apenas as ferramentas")
    print("4. Sair")
    
    choice = input("\nEscolha uma opção (1-4): ").strip()
    
    if choice == "1":
        return test_engenheiro_dados()
        
    elif choice == "2":
        print("\n📅 Digite o período para análise:")
        data_inicio = input("Data início (YYYY-MM-DD): ").strip()
        data_fim = input("Data fim (YYYY-MM-DD): ").strip()
        
        # Validar formato
        try:
            datetime.strptime(data_inicio, '%Y-%m-%d')
            datetime.strptime(data_fim, '%Y-%m-%d')
            
            # Atualizar inputs globais para o teste
            inputs = {'data_inicio': data_inicio, 'data_fim': data_fim}
            
            # Executar teste customizado
            logger.info(f"📅 Período customizado: {data_inicio} até {data_fim}")
            agent = create_engenheiro_dados_agent()
            task = create_engenheiro_dados_task(agent, inputs)
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
                memory=False,
                max_rpm=10
            )
            
            result = crew.kickoff(inputs=inputs)
            logger.info("✅ Teste customizado concluído!")
            return True
            
        except ValueError:
            logger.error("❌ Formato de data inválido! Use YYYY-MM-DD")
            return False
            
    elif choice == "3":
        return validate_tools()
        
    elif choice == "4":
        logger.info("👋 Saindo...")
        return True
        
    else:
        logger.error("❌ Opção inválida!")
        return False

# =============== ENTRY POINT ===============
if __name__ == "__main__":
    # Verificar se o diretório existe
    if not os.path.exists('src/insights'):
        logger.error("❌ Diretório src/insights não encontrado!")
        logger.error("Execute este script a partir do diretório raiz do projeto")
        sys.exit(1)
    
    # Criar diretório de output se não existir
    os.makedirs('output', exist_ok=True)
    
    try:
        # Executar validação inicial
        if not validate_tools():
            logger.error("❌ Falha na validação de ferramentas!")
            sys.exit(1)
            
        # Menu interativo ou execução direta
        if len(sys.argv) > 1 and sys.argv[1] == "--auto":
            # Execução automática para CI/CD
            success = test_engenheiro_dados()
            sys.exit(0 if success else 1)
        else:
            # Menu interativo
            interactive_menu()
            
    except KeyboardInterrupt:
        logger.info("\n👋 Teste interrompido pelo usuário")
    except Exception as e:
        logger.error(f"❌ Erro inesperado: {e}")
        sys.exit(1) 