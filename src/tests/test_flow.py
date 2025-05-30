#!/usr/bin/env python
"""
🧪 TESTES DE VALIDAÇÃO DO INSIGHTS-AI FLOW

Este arquivo testa:
- Funcionalidade básica do Flow
- Integração com sistema existente 
- Compatibilidade entre sistemas
- Performance e monitoramento
"""

import unittest
import sys
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Importar módulos para teste
from insights.flow_main import InsightsFlow, criar_flow_com_parametros
from insights.flow_integration import InsightsRunner, run_insights

class TestInsightsFlow(unittest.TestCase):
    """Testes para a classe InsightsFlow"""
    
    def setUp(self):
        """Configurar ambiente de teste"""
        self.data_inicio = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        self.data_fim = datetime.now().strftime('%Y-%m-%d')
    
    def test_flow_creation(self):
        """Testar criação básica do Flow"""
        try:
            flow = criar_flow_com_parametros(
                self.data_inicio, 
                self.data_fim, 
                "completo"
            )
            
            # Verificar se foi criado corretamente
            self.assertIsInstance(flow, InsightsFlow)
            self.assertEqual(flow.state.data_inicio, self.data_inicio)
            self.assertEqual(flow.state.data_fim, self.data_fim)
            self.assertEqual(flow.state.modo_execucao, "completo")
            
            print("✅ Test Flow Creation: PASSOU")
            
        except Exception as e:
            print(f"❌ Test Flow Creation: FALHOU - {e}")
            self.fail(f"Falha na criação do Flow: {e}")
    
    def test_flow_state_structure(self):
        """Testar estrutura do estado do Flow"""
        try:
            flow = criar_flow_com_parametros(self.data_inicio, self.data_fim)
            
            # Verificar campos obrigatórios do estado
            required_fields = [
                'data_inicio', 'data_fim', 'modo_execucao', 'flow_id',
                'fase_atual', 'progresso_percent', 'dados_extraidos',
                'qualidade_dados', 'analises_concluidas', 'erros_detectados'
            ]
            
            for field in required_fields:
                self.assertTrue(hasattr(flow.state, field), f"Campo {field} não encontrado no estado")
            
            # Verificar valores iniciais
            self.assertEqual(flow.state.fase_atual, "inicializando")
            self.assertEqual(flow.state.progresso_percent, 0.0)
            self.assertFalse(flow.state.dados_extraidos)
            self.assertEqual(len(flow.state.analises_concluidas), 0)
            
            print("✅ Test Flow State Structure: PASSOU")
            
        except Exception as e:
            print(f"❌ Test Flow State Structure: FALHOU - {e}")
            self.fail(f"Falha na estrutura do estado: {e}")
    
    def test_flow_validation(self):
        """Testar validação de inputs do Flow"""
        try:
            # Teste com dados válidos
            flow = criar_flow_com_parametros(self.data_inicio, self.data_fim)
            resultado = flow.inicializar_flow()
            self.assertEqual(resultado, "inputs_validados")
            
            # Teste com dados inválidos
            flow_invalid = InsightsFlow()
            flow_invalid.state.data_inicio = ""
            flow_invalid.state.data_fim = ""
            resultado_invalid = flow_invalid.inicializar_flow()
            self.assertEqual(resultado_invalid, "erro_inputs")
            
            print("✅ Test Flow Validation: PASSOU")
            
        except Exception as e:
            print(f"❌ Test Flow Validation: FALHOU - {e}")
            self.fail(f"Falha na validação: {e}")
    
    def test_status_monitoring(self):
        """Testar sistema de monitoramento de status"""
        try:
            flow = criar_flow_com_parametros(self.data_inicio, self.data_fim)
            
            # Testar método de status
            status = flow.get_status_detalhado()
            
            required_status_fields = [
                'flow_id', 'fase_atual', 'progresso_percent', 'tempo_decorrido',
                'dados_extraidos', 'analises_concluidas', 'erros_count'
            ]
            
            for field in required_status_fields:
                self.assertIn(field, status, f"Campo {field} não encontrado no status")
            
            print("✅ Test Status Monitoring: PASSOU")
            
        except Exception as e:
            print(f"❌ Test Status Monitoring: FALHOU - {e}")
            self.fail(f"Falha no monitoramento: {e}")

class TestInsightsIntegration(unittest.TestCase):
    """Testes para a integração híbrida"""
    
    def setUp(self):
        """Configurar ambiente de teste"""
        self.data_inicio = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        self.data_fim = datetime.now().strftime('%Y-%m-%d')
    
    def test_runner_creation(self):
        """Testar criação do runner híbrido"""
        try:
            runner = InsightsRunner()
            
            # Verificar atributos
            self.assertEqual(runner.modo_preferido, "auto")
            self.assertTrue(runner.fallback_ativo)
            self.assertTrue(runner.monitoramento_ativo)
            
            print("✅ Test Runner Creation: PASSOU")
            
        except Exception as e:
            print(f"❌ Test Runner Creation: FALHOU - {e}")
            self.fail(f"Falha na criação do runner: {e}")
    
    def test_criteria_analysis(self):
        """Testar análise de critérios para escolha do modo"""
        try:
            runner = InsightsRunner()
            
            # Testar análise de critérios
            criterios = runner._analisar_criterios_execucao(
                self.data_inicio, 
                self.data_fim,
                modo_rapido=True
            )
            
            # Verificar estrutura do resultado
            self.assertIn('usar_flow', criterios)
            self.assertIn('razoes', criterios)
            self.assertIsInstance(criterios['usar_flow'], bool)
            self.assertIsInstance(criterios['razoes'], list)
            
            print("✅ Test Criteria Analysis: PASSOU")
            
        except Exception as e:
            print(f"❌ Test Criteria Analysis: FALHOU - {e}")
            self.fail(f"Falha na análise de critérios: {e}")
    
    @patch('insights.flow_integration.CrewTradicional')
    def test_crew_fallback(self, mock_crew):
        """Testar fallback para crew tradicional"""
        try:
            # Configurar mock
            mock_instance = MagicMock()
            mock_crew.return_value = mock_instance
            mock_instance.crew.return_value.kickoff.return_value = "resultado_mock"
            mock_instance.agents = ["agent1", "agent2"]
            mock_instance.tasks = ["task1", "task2"]
            
            runner = InsightsRunner()
            
            # Forçar uso do crew
            resultado = runner._executar_com_crew(
                self.data_inicio, 
                self.data_fim,
                monitoramento_detalhado=False
            )
            
            # Verificar resultado
            self.assertEqual(resultado['status'], 'sucesso')
            self.assertEqual(resultado['modo_execucao'], 'crew')
            self.assertIn('resultado_principal', resultado)
            
            print("✅ Test Crew Fallback: PASSOU")
            
        except Exception as e:
            print(f"❌ Test Crew Fallback: FALHOU - {e}")
            self.fail(f"Falha no fallback: {e}")

class TestPerformanceAndLogging(unittest.TestCase):
    """Testes de performance e logging"""
    
    def test_logging_setup(self):
        """Testar configuração de logging"""
        try:
            from insights.flow_main import setup_flow_logging
            
            # Testar setup de logging
            logger, log_file = setup_flow_logging()
            
            # Verificar se logger foi criado
            self.assertIsNotNone(logger)
            self.assertIsNotNone(log_file)
            
            # Testar escrita de log
            logger.info("Teste de logging")
            
            print("✅ Test Logging Setup: PASSOU")
            
        except Exception as e:
            print(f"❌ Test Logging Setup: FALHOU - {e}")
            self.fail(f"Falha no logging: {e}")
    
    def test_performance_monitoring(self):
        """Testar monitoramento de performance"""
        try:
            flow = criar_flow_com_parametros(
                (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d')
            )
            
            # Simular execução rápida
            start_time = time.time()
            flow.inicializar_flow()
            execution_time = time.time() - start_time
            
            # Verificar que foi executado rapidamente
            self.assertLess(execution_time, 5.0, "Inicialização muito lenta")
            
            # Verificar métricas de performance no estado
            self.assertIn('tempo_por_analise', flow.state.__dict__)
            
            print("✅ Test Performance Monitoring: PASSOU")
            
        except Exception as e:
            print(f"❌ Test Performance Monitoring: FALHOU - {e}")
            self.fail(f"Falha no monitoramento de performance: {e}")

def run_basic_flow_test():
    """Teste básico de funcionamento do Flow (sem unittest)"""
    print("🧪 EXECUTANDO TESTE BÁSICO DO FLOW")
    print("=" * 50)
    
    try:
        # Configurar período de teste pequeno
        data_fim = datetime.now().strftime('%Y-%m-%d')
        data_inicio = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        print(f"📅 Período de teste: {data_inicio} a {data_fim}")
        
        # Testar criação do Flow
        print("🔧 Testando criação do Flow...")
        flow = criar_flow_com_parametros(data_inicio, data_fim, "completo")
        print(f"✅ Flow criado: ID {flow.state.flow_id}")
        
                            # Testar inicialização
        print("🚀 Testando inicialização...")
        try:
            resultado_init = flow.inicializar_flow()
            print(f"✅ Inicialização: {resultado_init}")
        except Exception as e:
            print(f"❌ Erro na inicialização: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Testar status
        print("📊 Testando monitoramento...")
        status = flow.get_status_detalhado()
        print(f"✅ Status obtido: {status['fase_atual']}")
        
        # Testar sistema híbrido
        print("🔄 Testando sistema híbrido...")
        runner = InsightsRunner()
        criterios = runner._analisar_criterios_execucao(data_inicio, data_fim)
        print(f"✅ Critérios analisados: {len(criterios['razoes'])} razões")
        
        print("=" * 50)
        print("🎉 TESTE BÁSICO CONCLUÍDO COM SUCESSO!")
        return True
        
    except Exception as e:
        print(f"❌ TESTE BÁSICO FALHOU: {e}")
        return False

def run_compatibility_test():
    """Teste de compatibilidade com sistema existente"""
    print("🔗 EXECUTANDO TESTE DE COMPATIBILIDADE")
    print("=" * 50)
    
    try:
        # Testar importações
        print("📦 Testando importações...")
        
        # Sistema Flow
        from insights.flow_main import InsightsFlow
        print("✅ Flow importado")
        
        # Sistema tradicional
        from old.crew import Insights
        print("✅ Crew tradicional importado")
        
        # Sistema híbrido
        from insights.flow_integration import run_insights
        print("✅ Sistema híbrido importado")
        
        # Testar interface compatível
        print("🔄 Testando interface compatível...")
        from old.main import run_with_crew, run_with_flow
        print("✅ Interfaces compatíveis disponíveis")
        
        print("=" * 50)
        print("🎉 TESTE DE COMPATIBILIDADE PASSOU!")
        return True
        
    except Exception as e:
        print(f"❌ TESTE DE COMPATIBILIDADE FALHOU: {e}")
        return False

if __name__ == "__main__":
    print("🧪 INICIANDO TESTES DO INSIGHTS-AI FLOW")
    print("=" * 60)
    
    # Executar testes básicos primeiro
    basic_test_ok = run_basic_flow_test()
    print()
    
    compatibility_test_ok = run_compatibility_test()
    print()
    
    if basic_test_ok and compatibility_test_ok:
        print("🚀 EXECUTANDO TESTES UNITÁRIOS DETALHADOS")
        print("=" * 60)
        
        # Executar testes unitários
        unittest.main(verbosity=2, exit=False)
        
        print("=" * 60)
        print("✅ TODOS OS TESTES CONCLUÍDOS!")
        print()
        print("🎯 PRÓXIMOS PASSOS:")
        print("   1. Execute: python main.py --mode auto")
        print("   2. Teste: python main.py --mode flow --quick")
        print("   3. Compare: python main.py --mode crew")
        print("=" * 60)
    else:
        print("❌ TESTES BÁSICOS FALHARAM - Verificar configuração")
        sys.exit(1) 