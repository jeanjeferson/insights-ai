#!/usr/bin/env python
"""
🔗 INTEGRAÇÃO FLOW <-> CREW TRADICIONAL

Este módulo garante compatibilidade entre:
- O novo sistema Flow (flow_main.py)
- O sistema tradicional Crew (crew.py) 
- A interface existente (main.py)

Permite migração gradual sem quebrar funcionalidades existentes.
"""

import sys
import warnings
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union

# Suprimir warnings específicos
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# Importar ambos os sistemas
from insights.crew import Insights as CrewTradicional
from insights.flow_main import InsightsFlow, criar_flow_com_parametros, executar_flow_completo

# =============== CONFIGURAÇÃO DE LOGGING ===============

logger = logging.getLogger(__name__)

def setup_integration_logging():
    """Configurar logging para o módulo de integração"""
    integration_logger = logging.getLogger('insights_integration')
    integration_logger.setLevel(logging.INFO)
    
    if not integration_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] INTEGRATION: %(message)s'
        )
        handler.setFormatter(formatter)
        integration_logger.addHandler(handler)
    
    return integration_logger

integration_logger = setup_integration_logging()

# =============== CLASSE DE INTEGRAÇÃO ===============

class InsightsRunner:
    """
    🔄 EXECUTOR HÍBRIDO INSIGHTS-AI
    
    Permite escolher entre:
    - Sistema tradicional (Crew)
    - Sistema otimizado (Flow)
    - Modo automático (decide baseado em critérios)
    """
    
    def __init__(self):
        self.modo_preferido = "auto"  # auto, crew, flow
        self.fallback_ativo = True
        self.monitoramento_ativo = True
        
    def executar_com_modo(self, 
                         data_inicio: str, 
                         data_fim: str, 
                         modo: str = "auto",
                         **kwargs) -> Dict[str, Any]:
        """
        Executar análise com modo específico
        
        Args:
            data_inicio: Data inicial (YYYY-MM-DD)
            data_fim: Data final (YYYY-MM-DD)
            modo: 'auto', 'crew', 'flow'
            **kwargs: Parâmetros adicionais
        
        Returns:
            Dict com resultado da execução
        """
        integration_logger.info(f"🚀 INICIANDO EXECUÇÃO - Modo: {modo}")
        integration_logger.info(f"📅 Período: {data_inicio} a {data_fim}")
        
        start_time = time.time()
        
        try:
            if modo == "auto":
                return self._executar_modo_automatico(data_inicio, data_fim, **kwargs)
            elif modo == "flow":
                return self._executar_com_flow(data_inicio, data_fim, **kwargs)
            elif modo == "crew":
                return self._executar_com_crew(data_inicio, data_fim, **kwargs)
            else:
                raise ValueError(f"Modo inválido: {modo}. Use 'auto', 'crew' ou 'flow'")
                
        except Exception as e:
            execution_time = time.time() - start_time
            integration_logger.error(f"❌ Erro na execução ({execution_time:.2f}s): {e}")
            
            if self.fallback_ativo and modo != "crew":
                integration_logger.info("🔄 Tentando fallback para modo tradicional...")
                return self._executar_com_crew(data_inicio, data_fim, **kwargs)
            else:
                raise
    
    def _executar_modo_automatico(self, data_inicio: str, data_fim: str, **kwargs) -> Dict[str, Any]:
        """Decide automaticamente entre Flow e Crew baseado em critérios"""
        integration_logger.info("🤖 MODO AUTOMÁTICO - Analisando melhor opção...")
        
        # Critérios para decidir o modo
        criterios = self._analisar_criterios_execucao(data_inicio, data_fim, **kwargs)
        
        if criterios['usar_flow']:
            integration_logger.info("✅ Critérios favorecem FLOW - Executando com sistema otimizado")
            try:
                return self._executar_com_flow(data_inicio, data_fim, **kwargs)
            except Exception as e:
                integration_logger.warning(f"⚠️ Flow falhou: {e}")
                if self.fallback_ativo:
                    integration_logger.info("🔄 Fallback automático para CREW")
                    return self._executar_com_crew(data_inicio, data_fim, **kwargs)
                else:
                    raise
        else:
            integration_logger.info("✅ Critérios favorecem CREW - Executando com sistema tradicional")
            return self._executar_com_crew(data_inicio, data_fim, **kwargs)
    
    def _analisar_criterios_execucao(self, data_inicio: str, data_fim: str, **kwargs) -> Dict[str, Any]:
        """Analisar critérios para decidir entre Flow e Crew"""
        criterios = {
            'usar_flow': True,  # Padrão: preferir Flow
            'razoes': []
        }
        
        try:
            # Critério 1: Verificar se Flow está disponível
            try:
                import crewai.flow.flow
                criterios['razoes'].append("✅ CrewAI Flow disponível")
            except ImportError:
                criterios['usar_flow'] = False
                criterios['razoes'].append("❌ CrewAI Flow não disponível")
                return criterios
            
            # Critério 2: Analisar período de dados
            data_inicio_dt = datetime.strptime(data_inicio, '%Y-%m-%d')
            data_fim_dt = datetime.strptime(data_fim, '%Y-%m-%d')
            dias_analise = (data_fim_dt - data_inicio_dt).days
            
            if dias_analise > 365:
                criterios['razoes'].append(f"✅ Período longo ({dias_analise} dias) - Flow otimizado")
            else:
                criterios['razoes'].append(f"ℹ️ Período moderado ({dias_analise} dias)")
            
            # Critério 3: Verificar kwargs especiais
            if kwargs.get('modo_rapido', False):
                criterios['razoes'].append("✅ Modo rápido solicitado - Flow preferível")
            
            if kwargs.get('monitoramento_detalhado', True):
                criterios['razoes'].append("✅ Monitoramento detalhado - Flow superior")
            
            # Critério 4: Verificar disponibilidade de recursos
            try:
                import psutil
                memoria_disponivel = psutil.virtual_memory().available / (1024**3)  # GB
                if memoria_disponivel < 2:
                    criterios['usar_flow'] = False
                    criterios['razoes'].append(f"⚠️ Pouca memória ({memoria_disponivel:.1f}GB) - Crew mais leve")
                else:
                    criterios['razoes'].append(f"✅ Memória adequada ({memoria_disponivel:.1f}GB)")
            except ImportError:
                criterios['razoes'].append("ℹ️ Não foi possível verificar recursos")
            
        except Exception as e:
            integration_logger.warning(f"⚠️ Erro ao analisar critérios: {e}")
            criterios['usar_flow'] = False
            criterios['razoes'].append(f"❌ Erro na análise: {e}")
        
        # Log das decisões
        integration_logger.info(f"📊 Decisão: {'FLOW' if criterios['usar_flow'] else 'CREW'}")
        for razao in criterios['razoes']:
            integration_logger.info(f"   {razao}")
        
        return criterios
    
    def _executar_com_flow(self, data_inicio: str, data_fim: str, **kwargs) -> Dict[str, Any]:
        """Executar com sistema Flow otimizado"""
        integration_logger.info("🚀 EXECUTANDO COM FLOW OTIMIZADO")
        integration_logger.info("-" * 50)
        
        start_time = time.time()
        
        try:
            # Configurar Flow
            modo_execucao = kwargs.get('modo_execucao', 'completo')
            flow = criar_flow_com_parametros(data_inicio, data_fim, modo_execucao)
            
            # Configurar monitoramento se solicitado
            if kwargs.get('monitoramento_detalhado', True):
                monitor_thread = self._iniciar_monitoramento_flow(flow)
            else:
                monitor_thread = None
            
            # Executar Flow
            resultado_flow = flow.kickoff()
            execution_time = time.time() - start_time
            
            # Parar monitoramento
            if monitor_thread:
                self._parar_monitoramento(monitor_thread)
            
            # Preparar resultado estruturado
            status_final = flow.get_status_detalhado()
            
            resultado = {
                'status': 'sucesso',
                'modo_execucao': 'flow',
                'tempo_execucao': execution_time,
                'resultado_principal': resultado_flow,
                'flow_status': status_final,
                'analises_concluidas': len(flow.state.analises_concluidas),
                'qualidade_dados': flow.state.qualidade_dados,
                'arquivos_gerados': flow.state.arquivos_gerados,
                'warnings': flow.state.warnings
            }
            
            integration_logger.info(f"✅ FLOW CONCLUÍDO em {execution_time:.2f}s")
            integration_logger.info(f"📊 Análises concluídas: {len(flow.state.analises_concluidas)}")
            integration_logger.info(f"📁 Arquivos gerados: {len(flow.state.arquivos_gerados)}")
            
            return resultado
            
        except Exception as e:
            execution_time = time.time() - start_time
            integration_logger.error(f"❌ Erro no Flow ({execution_time:.2f}s): {e}")
            raise Exception(f"Falha na execução do Flow: {e}")
    
    def _executar_com_crew(self, data_inicio: str, data_fim: str, **kwargs) -> Dict[str, Any]:
        """Executar com sistema Crew tradicional"""
        integration_logger.info("🚀 EXECUTANDO COM CREW TRADICIONAL") 
        integration_logger.info("-" * 50)
        
        start_time = time.time()
        
        try:
            # Preparar inputs para Crew tradicional
            inputs = {
                'data_inicio': data_inicio,
                'data_fim': data_fim
            }
            
            # Configurar monitoramento se solicitado
            if kwargs.get('monitoramento_detalhado', True):
                monitor_thread = self._iniciar_monitoramento_crew()
            else:
                monitor_thread = None
            
            # Executar Crew tradicional
            crew_instance = CrewTradicional()
            resultado_crew = crew_instance.crew().kickoff(inputs=inputs)
            execution_time = time.time() - start_time
            
            # Parar monitoramento
            if monitor_thread:
                self._parar_monitoramento(monitor_thread)
            
            # Preparar resultado estruturado
            resultado = {
                'status': 'sucesso',
                'modo_execucao': 'crew',
                'tempo_execucao': execution_time,
                'resultado_principal': resultado_crew,
                'crew_info': {
                    'agents_count': len(crew_instance.agents),
                    'tasks_count': len(crew_instance.tasks)
                }
            }
            
            integration_logger.info(f"✅ CREW CONCLUÍDO em {execution_time:.2f}s")
            
            return resultado
            
        except Exception as e:
            execution_time = time.time() - start_time
            integration_logger.error(f"❌ Erro no Crew ({execution_time:.2f}s): {e}")
            raise Exception(f"Falha na execução do Crew: {e}")
    
    def _iniciar_monitoramento_flow(self, flow: InsightsFlow) -> threading.Thread:
        """Iniciar monitoramento em tempo real do Flow"""
        def monitor():
            while getattr(threading.current_thread(), "do_run", True):
                try:
                    status = flow.get_status_detalhado()
                    integration_logger.info(
                        f"📊 FLOW STATUS: {status['fase_atual']} | "
                        f"Progresso: {status['progresso_percent']:.1f}% | "
                        f"Tempo: {status['tempo_decorrido']:.1f}s"
                    )
                    time.sleep(30)  # Log a cada 30 segundos
                except Exception as e:
                    integration_logger.warning(f"⚠️ Erro no monitoramento: {e}")
                    break
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        integration_logger.info("📡 Monitoramento do Flow iniciado")
        return thread
    
    def _iniciar_monitoramento_crew(self) -> threading.Thread:
        """Iniciar monitoramento simples do Crew"""
        def monitor():
            start_time = time.time()
            while getattr(threading.current_thread(), "do_run", True):
                elapsed = time.time() - start_time
                integration_logger.info(f"📊 CREW executando há {elapsed:.1f}s...")
                time.sleep(60)  # Log a cada 60 segundos
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        integration_logger.info("📡 Monitoramento do Crew iniciado")
        return thread
    
    def _parar_monitoramento(self, thread: threading.Thread):
        """Parar thread de monitoramento"""
        if thread and thread.is_alive():
            thread.do_run = False
            integration_logger.info("📡 Monitoramento finalizado")

# =============== INTERFACE COMPATÍVEL ===============

def run_insights(data_inicio: str = None, 
                data_fim: str = None, 
                modo: str = "auto",
                **kwargs) -> Dict[str, Any]:
    """
    🚀 INTERFACE PRINCIPAL PARA EXECUÇÃO DO INSIGHTS-AI
    
    Função compatível com main.py existente, mas com opções avançadas.
    
    Args:
        data_inicio: Data inicial (YYYY-MM-DD). Se None, usa últimos 2 anos
        data_fim: Data final (YYYY-MM-DD). Se None, usa hoje
        modo: 'auto', 'crew', 'flow'
        **kwargs: Parâmetros adicionais
    
    Returns:
        Dict com resultado da execução
    """
    # Configurar datas padrão se não fornecidas
    if data_fim is None:
        data_fim = datetime.now().strftime('%Y-%m-%d')
    
    if data_inicio is None:
        data_inicio = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    # Executar com runner híbrido
    runner = InsightsRunner()
    return runner.executar_com_modo(data_inicio, data_fim, modo, **kwargs)

# =============== COMPATIBILIDADE COM MAIN.PY ===============

def run():
    """
    Função compatível com main.py existente
    Mantém mesma interface, mas usa sistema híbrido
    """
    # Usar configuração padrão do main.py original
    data_fim = datetime.now().strftime('%Y-%m-%d')
    data_inicio = (datetime.now() - timedelta(days=1460)).strftime('%Y-%m-%d')
    
    try:
        resultado = run_insights(
            data_inicio=data_inicio,
            data_fim=data_fim,
            modo="auto",
            monitoramento_detalhado=True
        )
        
        logger.info("✅ EXECUÇÃO CONCLUÍDA COM SUCESSO")
        logger.info(f"📊 Modo utilizado: {resultado['modo_execucao']}")
        logger.info(f"⏱️ Tempo total: {resultado['tempo_execucao']:.2f}s")
        
        return resultado['resultado_principal']
        
    except KeyboardInterrupt:
        logger.warning("⚠️ EXECUÇÃO INTERROMPIDA pelo usuário")
        raise
    except Exception as e:
        logger.error(f"❌ ERRO na execução: {str(e)}")
        raise Exception(f"An error occurred while running the crew: {e}")

if __name__ == "__main__":
    # Execução direta compatível
    resultado = run()
    print(f"✅ Execução finalizada: {type(resultado)}") 