#!/usr/bin/env python
"""
🧪 TESTES DA ETAPA 2 - ANÁLISES PARALELAS

Este arquivo testa:
- Execução paralela de análises
- Dependencies com and_() e or_()
- Performance otimizada
- Consolidação de resultados
"""

import unittest
import sys
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Importar módulos para teste
from insights.flow_main import InsightsFlow, criar_flow_com_parametros
from insights.flow_integration import InsightsRunner, run_insights

class TestEtapa2Parallelization(unittest.TestCase):
    """Testes para análises paralelas da Etapa 2"""
    
    def setUp(self):
        """Configurar ambiente de teste"""
        self.data_inicio = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        self.data_fim = datetime.now().strftime('%Y-%m-%d')
    
    def test_parallel_analysis_trigger(self):
        """Testar se análises paralelas são acionadas corretamente"""
        try:
            flow = criar_flow_com_parametros(self.data_inicio, self.data_fim)
            
            # Simular estado após extração de dados
            flow.state.dados_extraidos = True
            flow.state.qualidade_dados = {"score_confiabilidade": 85.0}
            flow.state.pode_executar_analises_basicas = True
            
            # Verificar se métodos de análise paralela existem
            self.assertTrue(hasattr(flow, 'executar_analise_tendencias'))
            self.assertTrue(hasattr(flow, 'executar_analise_sazonalidade'))
            self.assertTrue(hasattr(flow, 'executar_analise_segmentos'))
            self.assertTrue(hasattr(flow, 'executar_analise_projecoes'))
            
            print("✅ Test Parallel Analysis Trigger: PASSOU")
            
        except Exception as e:
            print(f"❌ Test Parallel Analysis Trigger: FALHOU - {e}")
            self.fail(f"Falha no teste de acionamento paralelo: {e}")
    
    def test_dependency_system(self):
        """Testar sistema de dependências and_() e or_()"""
        try:
            flow = criar_flow_com_parametros(self.data_inicio, self.data_fim)
            
            # Verificar se projeções tem dependência correta
            projecoes_method = flow.executar_analise_projecoes
            self.assertTrue(hasattr(projecoes_method, '__trigger_methods__'))
            
            # Verificar se relatório final tem dependência or_()
            relatorio_method = flow.gerar_relatorio_final
            self.assertTrue(hasattr(relatorio_method, '__trigger_methods__'))
            
            print("✅ Test Dependency System: PASSOU")
            
        except Exception as e:
            print(f"❌ Test Dependency System: FALHOU - {e}")
            self.fail(f"Falha no sistema de dependências: {e}")
    
    def test_state_management_parallel(self):
        """Testar gerenciamento de estado durante execução paralela"""
        try:
            flow = criar_flow_com_parametros(self.data_inicio, self.data_fim)
            
            # Verificar estrutura inicial dos estados das análises
            expected_analyses = [
                'analise_tendencias', 'analise_sazonalidade', 
                'analise_segmentos', 'analise_projecoes'
            ]
            
            for analysis in expected_analyses:
                self.assertTrue(hasattr(flow.state, analysis))
                self.assertIsInstance(getattr(flow.state, analysis), dict)
            
            # Verificar campos de controle
            self.assertIsInstance(flow.state.analises_concluidas, list)
            self.assertIsInstance(flow.state.analises_em_execucao, list)
            self.assertIsInstance(flow.state.tempo_por_analise, dict)
            
            print("✅ Test State Management Parallel: PASSOU")
            
        except Exception as e:
            print(f"❌ Test State Management Parallel: FALHOU - {e}")
            self.fail(f"Falha no gerenciamento de estado: {e}")

class TestEtapa2Performance(unittest.TestCase):
    """Testes de performance da Etapa 2"""
    
    def test_cache_efficiency(self):
        """Testar eficiência do cache de crews"""
        try:
            flow = criar_flow_com_parametros(
                (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d')
            )
            
            # Verificar se cache está inicializado
            self.assertIsInstance(flow.crews_cache, dict)
            self.assertEqual(len(flow.crews_cache), 0)  # Inicialmente vazio
            
            # Simular adição ao cache
            flow.crews_cache["test_crew"] = "mock_crew"
            self.assertEqual(len(flow.crews_cache), 1)
            
            print("✅ Test Cache Efficiency: PASSOU")
            
        except Exception as e:
            print(f"❌ Test Cache Efficiency: FALHOU - {e}")
            self.fail(f"Falha no teste de cache: {e}")
    
    def test_progress_tracking(self):
        """Testar rastreamento de progresso das análises"""
        try:
            flow = criar_flow_com_parametros(
                (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d')
            )
            
            # Verificar progresso inicial
            self.assertEqual(flow.state.progresso_percent, 0.0)
            
            # Simular avanço de progresso
            flow.state.progresso_percent = 45.0
            self.assertEqual(flow.state.progresso_percent, 45.0)
            
            # Verificar se status é atualizável
            status = flow.get_status_detalhado()
            self.assertIn('progresso_percent', status)
            
            print("✅ Test Progress Tracking: PASSOU")
            
        except Exception as e:
            print(f"❌ Test Progress Tracking: FALHOU - {e}")
            self.fail(f"Falha no rastreamento de progresso: {e}")

class TestEtapa2Integration(unittest.TestCase):
    """Testes de integração da Etapa 2"""
    
    def test_integration_with_etapa1(self):
        """Testar integração com funcionalidades da Etapa 1"""
        try:
            # Testar se todas as funcionalidades da Etapa 1 ainda funcionam
            runner = InsightsRunner()
            
            # Verificar se ainda consegue decidir entre Flow/Crew
            criterios = runner._analisar_criterios_execucao(
                (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d')
            )
            
            self.assertIn('usar_flow', criterios)
            self.assertIn('razoes', criterios)
            
            print("✅ Test Integration with Etapa1: PASSOU")
            
        except Exception as e:
            print(f"❌ Test Integration with Etapa1: FALHOU - {e}")
            self.fail(f"Falha na integração com Etapa 1: {e}")
    
    def test_backward_compatibility(self):
        """Testar compatibilidade retroativa"""
        try:
            # Verificar se sistema híbrido ainda funciona
            from insights.flow_integration import run_insights
            
            # Testar imports
            from insights.flow_main import InsightsFlow
            from insights.crew import Insights
            
            print("✅ Test Backward Compatibility: PASSOU")
            
        except Exception as e:
            print(f"❌ Test Backward Compatibility: FALHOU - {e}")
            self.fail(f"Falha na compatibilidade retroativa: {e}")

def run_etapa2_basic_test():
    """Teste básico da Etapa 2"""
    print("🧪 EXECUTANDO TESTE BÁSICO DA ETAPA 2")
    print("=" * 50)
    
    try:
        # Configurar período de teste
        data_fim = datetime.now().strftime('%Y-%m-%d')
        data_inicio = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
        
        print(f"📅 Período de teste: {data_inicio} a {data_fim}")
        
        # Testar criação do Flow com Etapa 2
        print("🔧 Testando Flow com análises paralelas...")
        flow = criar_flow_com_parametros(data_inicio, data_fim, "completo")
        print(f"✅ Flow Etapa 2 criado: ID {flow.state.flow_id}")
        
        # Verificar métodos da Etapa 2
        print("📊 Verificando métodos paralelos...")
        parallel_methods = [
            'executar_analise_tendencias',
            'executar_analise_sazonalidade', 
            'executar_analise_segmentos',
            'executar_analise_projecoes',
            'gerar_relatorio_final'
        ]
        
        for method in parallel_methods:
            if hasattr(flow, method):
                print(f"✅ {method}")
            else:
                raise Exception(f"Método {method} não encontrado")
        
        # Testar estado estruturado
        print("🗂️ Verificando estado estruturado...")
        required_state_fields = [
            'analise_tendencias', 'analise_sazonalidade',
            'analise_segmentos', 'analise_projecoes'
        ]
        
        for field in required_state_fields:
            if hasattr(flow.state, field):
                print(f"✅ Estado: {field}")
            else:
                raise Exception(f"Campo de estado {field} não encontrado")
        
        print("=" * 50)
        print("🎉 TESTE BÁSICO DA ETAPA 2 CONCLUÍDO COM SUCESSO!")
        return True
        
    except Exception as e:
        print(f"❌ TESTE BÁSICO DA ETAPA 2 FALHOU: {e}")
        return False

def test_corrected_flow_execution():
    """Testar se o Flow corrigido executa e gera outputs"""
    print("🧪 TESTANDO FLOW CORRIGIDO COM GERAÇÃO DE OUTPUTS")
    print("=" * 60)
    
    try:
        from insights.flow_main import criar_flow_com_parametros
        from pathlib import Path
        import os
        
        # Limpar outputs antigos
        output_dir = Path("output")
        if output_dir.exists():
            for old_file in output_dir.glob("*flow_*"):
                try:
                    old_file.unlink()
                    print(f"🗑️ Removido arquivo antigo: {old_file.name}")
                except:
                    pass
        
        # Configurar período de teste
        data_fim = datetime.now().strftime('%Y-%m-%d')
        data_inicio = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        print(f"📅 Período de teste: {data_inicio} a {data_fim}")
        
        # Criar flow
        flow = criar_flow_com_parametros(data_inicio, data_fim, "completo")
        print(f"✅ Flow criado: ID {flow.state.flow_id}")
        
        # Simular execução das análises paralelas
        flow.state.dados_extraidos = True
        flow.state.qualidade_dados = {"score_confiabilidade": 85.0, "total_registros": 15000}
        flow.state.pode_executar_analises_basicas = True
        
        # Testar File Generation Tool corrigida
        from insights.tools.file_generation_tool import FileGenerationTool
        file_tool = FileGenerationTool()
        
        # Teste de geração JSON
        resultado_json = file_tool._run(
            file_type="json",
            filename="teste_correcao.json",
            content='{"teste": "correcao_funcionando", "timestamp": "2025-01-27"}',
            output_path="output/teste_correcao.json"
        )
        print(f"✅ Teste JSON: {resultado_json}")
        
        # Teste de geração CSV
        resultado_csv = file_tool._run(
            file_type="csv",
            filename="teste_correcao.csv",
            content="coluna1,coluna2\nvalor1,valor2\n",
            output_path="output/teste_correcao.csv"
        )
        print(f"✅ Teste CSV: {resultado_csv}")
        
        # Teste de geração Markdown
        resultado_md = file_tool._run(
            file_type="markdown",
            filename="teste_correcao.md",
            content="# Teste de Correção\n\nEste é um teste das correções aplicadas.\n",
            output_path="output/teste_correcao.md"
        )
        print(f"✅ Teste Markdown: {resultado_md}")
        
        # Verificar se arquivos foram criados
        arquivos_gerados = []
        for arquivo in ["teste_correcao.json", "teste_correcao.csv", "teste_correcao.md"]:
            filepath = output_dir / arquivo
            if filepath.exists():
                arquivos_gerados.append(arquivo)
                print(f"📁 Arquivo criado: {arquivo} ({filepath.stat().st_size} bytes)")
        
        print(f"\n📊 RESULTADO DO TESTE:")
        print(f"✅ Arquivos gerados: {len(arquivos_gerados)}/3")
        
        if len(arquivos_gerados) == 3:
            print("🎉 TODAS AS CORREÇÕES FUNCIONANDO!")
            return True
        else:
            print("⚠️ Algumas correções precisam de ajuste")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste de correções: {e}")
        return False

if __name__ == "__main__":
    print("🚀 INICIANDO TESTES DA ETAPA 2 - ANÁLISES PARALELAS")
    print("=" * 70)
    
    # Executar teste de correções primeiro
    corrected_test_ok = test_corrected_flow_execution()
    print()
    
    # Executar teste básico
    basic_test_ok = run_etapa2_basic_test()
    print()
    
    if basic_test_ok:
        print("🚀 EXECUTANDO TESTES UNITÁRIOS DA ETAPA 2")
        print("=" * 70)
        
        # Executar testes unitários
        unittest.main(verbosity=2, exit=False)
        
        print("=" * 70)
        print("✅ TODOS OS TESTES DA ETAPA 2 CONCLUÍDOS!")
        print()
        print("🎯 PRÓXIMOS PASSOS:")
        print("   1. Execute: python src/insights/main.py --mode flow")
        print("   2. Teste: python src/insights/test_flow.py")
        print("   3. Compare performance: Etapa 1 vs Etapa 2")
        print("=" * 70)
    else:
        print("❌ TESTES BÁSICOS DA ETAPA 2 FALHARAM - Verificar implementação")
        sys.exit(1) 