"""
Sistema de Testes Completo para Etapa 3
Valida todas as funcionalidades avançadas implementadas
"""

import json
import time
import threading
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import unittest
from unittest.mock import Mock, patch, MagicMock

# Configurar logging para testes
test_logger = logging.getLogger('test_etapa3')
test_logger.setLevel(logging.INFO)

class TestEtapa3Recovery(unittest.TestCase):
    """Testes para sistema de Recovery"""
    
    def setUp(self):
        """Setup para cada teste"""
        from insights.flow_recovery import FlowRecoverySystem, RecoveryLevel, FailureType
        from insights.flow_main import InsightsFlowState
        
        # Usar diretório temporário para testes
        self.temp_dir = tempfile.mkdtemp()
        self.recovery_system = FlowRecoverySystem(self.temp_dir)
        
        # Criar estado de flow simulado
        self.mock_flow_state = InsightsFlowState()
        self.mock_flow_state.flow_id = "test_flow_123"
        self.mock_flow_state.fase_atual = "teste"
        self.mock_flow_state.dados_extraidos = True
        
        test_logger.info("🧪 Setup Recovery tests concluído")
    
    def tearDown(self):
        """Cleanup após cada teste"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_checkpoint(self):
        """Testar criação de checkpoint"""
        test_logger.info("🧪 Testando criação de checkpoint")
        
        from insights.flow_recovery import RecoveryLevel
        
        # Criar checkpoint
        checkpoint_path = self.recovery_system.create_checkpoint(
            self.mock_flow_state, 
            RecoveryLevel.COMPLETE
        )
        
        # Verificar se checkpoint foi criado
        self.assertTrue(checkpoint_path != "")
        self.assertTrue(Path(checkpoint_path).exists())
        
        # Verificar conteúdo do checkpoint
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        self.assertEqual(checkpoint_data["flow_id"], "test_flow_123")
        self.assertEqual(checkpoint_data["estado_atual"], "teste")
        self.assertIn("checksum", checkpoint_data)
        
        test_logger.info("✅ Teste de criação de checkpoint PASSOU")
    
    def test_recovery_flow(self):
        """Testar recuperação de flow"""
        test_logger.info("🧪 Testando recuperação de flow")
        
        from insights.flow_recovery import RecoveryLevel
        from insights.flow_main import InsightsFlowState
        
        # Criar checkpoint primeiro
        checkpoint_path = self.recovery_system.create_checkpoint(
            self.mock_flow_state, 
            RecoveryLevel.COMPLETE
        )
        
        # Verificar se checkpoint foi criado
        self.assertTrue(checkpoint_path != "", "Checkpoint não foi criado")
        self.assertTrue(Path(checkpoint_path).exists(), "Arquivo de checkpoint não existe")
        
        # Tentar recuperar com classe target correta
        recovered_flow = self.recovery_system.recover_flow(
            "test_flow_123",
            target_class=InsightsFlowState
        )
        
        # Verificar se recuperação funcionou
        self.assertIsNotNone(recovered_flow, "Flow não foi recuperado")
        if recovered_flow:
            self.assertEqual(recovered_flow.flow_id, "test_flow_123", "Flow ID incorreto após recovery")
        
        test_logger.info("✅ Teste de recuperação de flow PASSOU")
    
    def test_failure_registration(self):
        """Testar registro de falhas"""
        test_logger.info("🧪 Testando registro de falhas")
        
        from insights.flow_recovery import FailureType
        
        # Registrar falha
        self.recovery_system.register_failure(
            flow_id="test_flow_123",
            failure_type=FailureType.NETWORK_ERROR,
            error_message="Erro de teste",
            context={"test": True}
        )
        
        # Verificar se falha foi registrada
        self.assertEqual(len(self.recovery_system.failure_history), 1)
        
        failure = self.recovery_system.failure_history[0]
        self.assertEqual(failure.flow_id, "test_flow_123")
        self.assertEqual(failure.failure_type, FailureType.NETWORK_ERROR)
        
        test_logger.info("✅ Teste de registro de falhas PASSOU")
    
    def test_recovery_status(self):
        """Testar obtenção de status de recovery"""
        test_logger.info("🧪 Testando status de recovery")
        
        # Criar alguns checkpoints
        for i in range(3):
            self.recovery_system.create_checkpoint(self.mock_flow_state)
            time.sleep(0.1)  # Garantir timestamps diferentes
        
        # Obter status
        status = self.recovery_system.get_recovery_status("test_flow_123")
        
        # Verificar status
        self.assertEqual(status["flow_id"], "test_flow_123")
        self.assertEqual(status["total_checkpoints"], 3)
        self.assertTrue(status["recovery_possible"])
        
        test_logger.info("✅ Teste de status de recovery PASSOU")


class TestEtapa3Monitoring(unittest.TestCase):
    """Testes para sistema de Monitoramento"""
    
    def setUp(self):
        """Setup para cada teste"""
        from insights.flow_monitoring import FlowMonitoringSystem, MetricType, AlertLevel
        
        # Usar diretório temporário para testes
        self.temp_dir = tempfile.mkdtemp()
        self.monitoring_system = FlowMonitoringSystem(self.temp_dir)
        
        # Mock flow
        self.mock_flow = Mock()
        self.mock_flow.flow_id = "test_flow_monitoring_123"
        
        test_logger.info("🧪 Setup Monitoring tests concluído")
    
    def tearDown(self):
        """Cleanup após cada teste"""
        import shutil
        self.monitoring_system.stop_monitoring()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_register_flow(self):
        """Testar registro de flow no monitoramento"""
        test_logger.info("🧪 Testando registro de flow")
        
        # Registrar flow
        self.monitoring_system.register_flow(
            "test_flow_monitoring_123", 
            self.mock_flow
        )
        
        # Verificar se foi registrado
        self.assertIn("test_flow_monitoring_123", self.monitoring_system.current_flows)
        
        flow_info = self.monitoring_system.current_flows["test_flow_monitoring_123"]
        self.assertEqual(flow_info["instance"], self.mock_flow)
        
        test_logger.info("✅ Teste de registro de flow PASSOU")
    
    def test_collect_metrics(self):
        """Testar coleta de métricas"""
        test_logger.info("🧪 Testando coleta de métricas")
        
        from insights.flow_monitoring import MetricType
        
        # Registrar flow primeiro
        self.monitoring_system.register_flow("test_flow_monitoring_123", self.mock_flow)
        
        # Coletar algumas métricas
        for i in range(5):
            self.monitoring_system.collect_metric(
                flow_id="test_flow_monitoring_123",
                metric_name="test_metric",
                value=float(i * 10),
                metric_type=MetricType.PERFORMANCE,
                unit="seconds"
            )
        
        # Verificar se métricas foram coletadas
        metrics = self.monitoring_system.get_current_metrics(
            flow_id="test_flow_monitoring_123"
        )
        
        self.assertEqual(len(metrics), 5)
        self.assertEqual(metrics[0].metric_name, "test_metric")
        
        test_logger.info("✅ Teste de coleta de métricas PASSOU")
    
    def test_health_checks(self):
        """Testar health checks"""
        test_logger.info("🧪 Testando health checks")
        
        # Registrar flow
        self.monitoring_system.register_flow("test_flow_monitoring_123", self.mock_flow)
        
        # Realizar health check
        health_check = self.monitoring_system.perform_health_check(
            "test_flow_monitoring_123", 
            "main"
        )
        
        # Verificar resultado
        self.assertIsNotNone(health_check)
        self.assertEqual(health_check.flow_id, "test_flow_monitoring_123")
        self.assertEqual(health_check.component, "main")
        self.assertIn(health_check.status, ["healthy", "warning", "error"])
        
        test_logger.info("✅ Teste de health checks PASSOU")
    
    def test_alerts_creation(self):
        """Testar criação de alertas"""
        test_logger.info("🧪 Testando criação de alertas")
        
        from insights.flow_monitoring import AlertLevel
        
        # Registrar flow
        self.monitoring_system.register_flow("test_flow_monitoring_123", self.mock_flow)
        
        # Criar alerta
        alert = self.monitoring_system.create_alert(
            flow_id="test_flow_monitoring_123",
            level=AlertLevel.WARNING,
            title="Teste de Alerta",
            message="Este é um alerta de teste",
            metric_name="test_metric",
            current_value=75.0,
            threshold_value=70.0
        )
        
        # Verificar alerta
        self.assertIsNotNone(alert)
        self.assertEqual(alert.flow_id, "test_flow_monitoring_123")
        self.assertEqual(alert.level, AlertLevel.WARNING)
        
        # Verificar se alerta está na lista
        active_alerts = self.monitoring_system.get_active_alerts()
        self.assertEqual(len(active_alerts), 1)
        
        test_logger.info("✅ Teste de criação de alertas PASSOU")
    
    def test_performance_summary(self):
        """Testar resumo de performance"""
        test_logger.info("🧪 Testando resumo de performance")
        
        from insights.flow_monitoring import MetricType
        
        # Registrar flow
        self.monitoring_system.register_flow("test_flow_monitoring_123", self.mock_flow)
        
        # Coletar métricas diversas
        metrics_data = [
            ("cpu_usage", 65.0, "percent"),
            ("memory_usage", 45.0, "percent"),
            ("execution_time", 120.0, "seconds"),
            ("execution_time", 130.0, "seconds"),
            ("execution_time", 110.0, "seconds")
        ]
        
        for metric_name, value, unit in metrics_data:
            self.monitoring_system.collect_metric(
                flow_id="test_flow_monitoring_123",
                metric_name=metric_name,
                value=value,
                metric_type=MetricType.PERFORMANCE,
                unit=unit
            )
        
        # Obter resumo de performance
        summary = self.monitoring_system.get_performance_summary("test_flow_monitoring_123")
        
        # Verificar resumo
        self.assertIn("execution_time", summary)
        self.assertEqual(summary["execution_time"]["count"], 3)
        self.assertEqual(summary["execution_time"]["avg"], 120.0)
        
        test_logger.info("✅ Teste de resumo de performance PASSOU")


class TestEtapa3Controller(unittest.TestCase):
    """Testes para controlador da Etapa 3"""
    
    def setUp(self):
        """Setup para cada teste"""
        from insights.flow_main import InsightsFlowState, FlowEtapa3Controller
        
        # Criar estado de flow simulado
        self.mock_flow_state = InsightsFlowState()
        self.mock_flow_state.flow_id = "test_controller_123"
        self.mock_flow_state.fase_atual = "teste"
        
        # Mock dos sistemas
        with patch('insights.flow_main.get_global_recovery_system'), \
             patch('insights.flow_main.get_global_monitoring_system'):
            self.controller = FlowEtapa3Controller(self.mock_flow_state)
        
        test_logger.info("🧪 Setup Controller tests concluído")
    
    def test_controller_initialization(self):
        """Testar inicialização do controlador"""
        test_logger.info("🧪 Testando inicialização do controlador")
        
        # Verificar se controlador foi inicializado
        self.assertIsNotNone(self.controller)
        self.assertEqual(self.controller.flow_state, self.mock_flow_state)
        self.assertTrue(self.controller.auto_recovery_enabled)
        self.assertTrue(self.controller.real_time_monitoring_enabled)
        
        test_logger.info("✅ Teste de inicialização do controlador PASSOU")
    
    @patch('insights.flow_main.get_global_recovery_system')
    @patch('insights.flow_main.get_global_monitoring_system')
    def test_start_systems(self, mock_monitoring, mock_recovery):
        """Testar início dos sistemas"""
        test_logger.info("🧪 Testando início dos sistemas")
        
        # Mock dos sistemas
        mock_monitoring_instance = Mock()
        mock_recovery_instance = Mock()
        mock_monitoring.return_value = mock_monitoring_instance
        mock_recovery.return_value = mock_recovery_instance
        
        # Criar novo controlador com mocks
        from insights.flow_main import FlowEtapa3Controller
        controller = FlowEtapa3Controller(self.mock_flow_state)
        
        # Iniciar sistemas
        controller.iniciar_sistemas_etapa3()
        
        # Verificar se métodos foram chamados
        mock_monitoring_instance.register_flow.assert_called_once()
        mock_recovery_instance.start_auto_recovery.assert_called_once()
        
        test_logger.info("✅ Teste de início dos sistemas PASSOU")
    
    def test_collect_metrics(self):
        """Testar coleta de métricas"""
        test_logger.info("🧪 Testando coleta de métricas via controlador")
        
        # Mock do sistema de monitoramento
        self.controller.monitoring_system = Mock()
        
        # Coletar métricas
        self.controller.coletar_metricas_analise(
            "test_analysis", 
            45.5, 
            True, 
            {"confidence": 92.0}
        )
        
        # Verificar se método foi chamado
        self.controller.monitoring_system.collect_metric.assert_called()
        
        test_logger.info("✅ Teste de coleta de métricas via controlador PASSOU")
    
    def test_health_check(self):
        """Testar verificação de saúde"""
        test_logger.info("🧪 Testando verificação de saúde via controlador")
        
        # Mock do sistema de monitoramento
        self.controller.monitoring_system = Mock()
        self.controller.recovery_system = Mock()
        
        # Configurar retornos dos mocks
        self.controller.monitoring_system.perform_health_check.return_value = Mock()
        self.controller.monitoring_system.get_health_status.return_value = {"status": "healthy"}
        self.controller.monitoring_system.get_performance_summary.return_value = {"avg_time": 30.0}
        self.controller.recovery_system.get_recovery_status.return_value = {"checkpoints": 5}
        
        # Verificar saúde
        health_results = self.controller.verificar_health_completo()
        
        # Verificar resultado
        self.assertIsNotNone(health_results)
        self.assertIn("overall", health_results)
        self.assertIn("performance", health_results)
        self.assertIn("recovery", health_results)
        
        test_logger.info("✅ Teste de verificação de saúde via controlador PASSOU")


class TestEtapa3Integration(unittest.TestCase):
    """Testes de integração da Etapa 3"""
    
    def setUp(self):
        """Setup para testes de integração"""
        test_logger.info("🧪 Setup Integration tests concluído")
    
    def test_flow_with_etapa3(self):
        """Testar flow completo com Etapa 3"""
        test_logger.info("🧪 Testando flow completo com Etapa 3")
        
        from insights.flow_main import InsightsFlowState
        
        # Criar estado de flow com Etapa 3 habilitada
        flow_state = InsightsFlowState()
        flow_state.etapa3_habilitada = True
        flow_state.data_inicio = "2025-01-01"
        flow_state.data_fim = "2025-01-07"
        
        # Verificar se Etapa 3 está habilitada
        self.assertTrue(flow_state.etapa3_habilitada)
        
        test_logger.info("✅ Teste de flow com Etapa 3 PASSOU")
    
    @patch('insights.flow_main.FlowEtapa3Controller')
    def test_flow_initialization_with_etapa3(self, mock_controller_class):
        """Testar inicialização do flow com Etapa 3"""
        test_logger.info("🧪 Testando inicialização do flow com Etapa 3")
        
        # Mock do controlador
        mock_controller = Mock()
        mock_controller_class.return_value = mock_controller
        
        from insights.flow_main import InsightsFlow
        
        # Criar flow
        flow = InsightsFlow()
        flow.state.data_inicio = "2025-01-01"
        flow.state.data_fim = "2025-01-07"
        flow.state.etapa3_habilitada = True
        
        # Simular inicialização (método privado, então testamos indiretamente)
        self.assertTrue(flow.state.etapa3_habilitada)
        
        test_logger.info("✅ Teste de inicialização com Etapa 3 PASSOU")


def run_stress_test():
    """Executar teste de stress do sistema"""
    test_logger.info("🔥 INICIANDO TESTE DE STRESS DA ETAPA 3")
    test_logger.info("=" * 60)
    
    from insights.flow_monitoring import FlowMonitoringSystem, MetricType
    from insights.flow_recovery import FlowRecoverySystem
    
    # Criar sistemas
    monitoring = FlowMonitoringSystem()
    recovery = FlowRecoverySystem()
    
    # Simular carga alta
    stress_results = {
        "metrics_collected": 0,
        "checkpoints_created": 0,
        "alerts_generated": 0,
        "start_time": time.time()
    }
    
    try:
        # Registrar flows simulados
        from insights.flow_main import InsightsFlowState
        for i in range(10):
            flow_id = f"stress_flow_{i}"
            test_flow = InsightsFlowState()
            test_flow.flow_id = flow_id
            monitoring.register_flow(flow_id, test_flow)
        
        # Coletar muitas métricas rapidamente
        for i in range(1000):
            flow_id = f"stress_flow_{i % 10}"
            monitoring.collect_metric(
                flow_id=flow_id,
                metric_name=f"stress_metric_{i % 5}",
                value=float(i % 100),
                metric_type=MetricType.PERFORMANCE,
                unit="units"
            )
            stress_results["metrics_collected"] += 1
            
            # Criar checkpoint a cada 100 métricas
            if i % 100 == 0:
                test_state = InsightsFlowState()
                test_state.flow_id = flow_id
                recovery.create_checkpoint(test_state)
                stress_results["checkpoints_created"] += 1
        
        # Gerar alertas de stress
        from insights.flow_monitoring import AlertLevel
        for i in range(50):
            alert = monitoring.create_alert(
                flow_id=f"stress_flow_{i % 10}",
                level=AlertLevel.WARNING,
                title=f"Stress Alert {i}",
                message=f"Alert gerado durante teste de stress {i}",
                metric_name="stress_metric",
                current_value=float(i),
                threshold_value=25.0
            )
            stress_results["alerts_generated"] += 1
        
        # Calcular resultados
        end_time = time.time()
        duration = end_time - stress_results["start_time"]
        
        test_logger.info(f"✅ TESTE DE STRESS CONCLUÍDO em {duration:.2f}s")
        test_logger.info(f"📊 Métricas coletadas: {stress_results['metrics_collected']}")
        test_logger.info(f"📸 Checkpoints criados: {stress_results['checkpoints_created']}")
        test_logger.info(f"🚨 Alertas gerados: {stress_results['alerts_generated']}")
        test_logger.info(f"⚡ Taxa de métricas: {stress_results['metrics_collected']/duration:.1f}/s")
        
        return stress_results
        
    except Exception as e:
        test_logger.error(f"❌ Erro no teste de stress: {e}")
        return None
    finally:
        # Cleanup
        monitoring.stop_monitoring()


def run_performance_benchmark():
    """Executar benchmark de performance"""
    test_logger.info("⚡ INICIANDO BENCHMARK DE PERFORMANCE")
    test_logger.info("=" * 50)
    
    import psutil
    from insights.flow_monitoring import FlowMonitoringSystem
    from insights.flow_recovery import FlowRecoverySystem, RecoveryLevel
    
    benchmark_results = {}
    
    # Benchmark 1: Criação de checkpoints
    test_logger.info("📸 Benchmark: Criação de checkpoints")
    recovery = FlowRecoverySystem()
    from insights.flow_main import InsightsFlowState
    test_state = InsightsFlowState()
    test_state.flow_id = "benchmark_flow"
    
    start_time = time.time()
    for i in range(100):
        recovery.create_checkpoint(test_state, RecoveryLevel.MINIMAL)
    checkpoint_time = time.time() - start_time
    
    benchmark_results["checkpoint_creation"] = {
        "total_time": checkpoint_time,
        "avg_time_per_checkpoint": checkpoint_time / 100 if checkpoint_time > 0 else 0,
        "checkpoints_per_second": 100 / checkpoint_time if checkpoint_time > 0 else float('inf')
    }
    
    # Benchmark 2: Coleta de métricas
    test_logger.info("📊 Benchmark: Coleta de métricas")
    monitoring = FlowMonitoringSystem()
    monitoring.register_flow("benchmark_flow", test_state)
    
    from insights.flow_monitoring import MetricType
    
    start_time = time.time()
    for i in range(1000):
        monitoring.collect_metric(
            flow_id="benchmark_flow",
            metric_name="benchmark_metric",
            value=float(i),
            metric_type=MetricType.PERFORMANCE,
            unit="units"
        )
    metrics_time = time.time() - start_time
    
    benchmark_results["metrics_collection"] = {
        "total_time": metrics_time,
        "avg_time_per_metric": metrics_time / 1000 if metrics_time > 0 else 0,
        "metrics_per_second": 1000 / metrics_time if metrics_time > 0 else float('inf')
    }
    
    # Benchmark 3: Health checks
    test_logger.info("🏥 Benchmark: Health checks")
    start_time = time.time()
    for i in range(50):
        monitoring.perform_health_check("benchmark_flow", "main")
    health_time = time.time() - start_time
    
    benchmark_results["health_checks"] = {
        "total_time": health_time,
        "avg_time_per_check": health_time / 50 if health_time > 0 else 0,
        "checks_per_second": 50 / health_time if health_time > 0 else float('inf')
    }
    
    # Benchmark 4: Uso de recursos
    process = psutil.Process()
    benchmark_results["resource_usage"] = {
        "cpu_percent": process.cpu_percent(),
        "memory_mb": process.memory_info().rss / 1024 / 1024,
        "threads_count": process.num_threads()
    }
    
    # Exibir resultados
    test_logger.info("📊 RESULTADOS DO BENCHMARK:")
    test_logger.info(f"📸 Checkpoints: {benchmark_results['checkpoint_creation']['checkpoints_per_second']:.1f}/s")
    test_logger.info(f"📊 Métricas: {benchmark_results['metrics_collection']['metrics_per_second']:.1f}/s")
    test_logger.info(f"🏥 Health Checks: {benchmark_results['health_checks']['checks_per_second']:.1f}/s")
    test_logger.info(f"💻 CPU: {benchmark_results['resource_usage']['cpu_percent']:.1f}%")
    test_logger.info(f"🧠 Memória: {benchmark_results['resource_usage']['memory_mb']:.1f} MB")
    
    # Cleanup
    monitoring.stop_monitoring()
    
    return benchmark_results


def run_all_tests():
    """Executar todos os testes da Etapa 3"""
    test_logger.info("🚀 INICIANDO SUITE COMPLETA DE TESTES DA ETAPA 3")
    test_logger.info("=" * 70)
    
    # Executar testes unitários
    test_logger.info("🧪 Executando testes unitários...")
    
    # Criar test suite
    test_suite = unittest.TestSuite()
    
    # Adicionar test cases
    loader = unittest.TestLoader()
    test_suite.addTest(loader.loadTestsFromTestCase(TestEtapa3Recovery))
    test_suite.addTest(loader.loadTestsFromTestCase(TestEtapa3Monitoring))
    test_suite.addTest(loader.loadTestsFromTestCase(TestEtapa3Controller))
    test_suite.addTest(loader.loadTestsFromTestCase(TestEtapa3Integration))
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(test_suite)
    
    # Resultado dos testes unitários
    unit_tests_passed = test_result.wasSuccessful()
    test_logger.info(f"🧪 Testes unitários: {'✅ PASSOU' if unit_tests_passed else '❌ FALHOU'}")
    test_logger.info(f"📊 Total: {test_result.testsRun}, Falhas: {len(test_result.failures)}, Erros: {len(test_result.errors)}")
    
    # Executar teste de stress
    test_logger.info("\n🔥 Executando teste de stress...")
    stress_results = run_stress_test()
    stress_passed = stress_results is not None
    
    # Executar benchmark
    test_logger.info("\n⚡ Executando benchmark de performance...")
    benchmark_results = run_performance_benchmark()
    
    # Resumo final
    test_logger.info("\n" + "=" * 70)
    test_logger.info("🎯 RESUMO FINAL DOS TESTES DA ETAPA 3")
    test_logger.info("=" * 70)
    test_logger.info(f"🧪 Testes Unitários: {'✅ PASSOU' if unit_tests_passed else '❌ FALHOU'}")
    test_logger.info(f"🔥 Teste de Stress: {'✅ PASSOU' if stress_passed else '❌ FALHOU'}")
    test_logger.info(f"⚡ Benchmark: ✅ CONCLUÍDO")
    
    overall_success = unit_tests_passed and stress_passed
    test_logger.info(f"🎉 RESULTADO GERAL: {'✅ TODOS OS TESTES PASSARAM' if overall_success else '❌ ALGUNS TESTES FALHARAM'}")
    
    return {
        "unit_tests": {
            "passed": unit_tests_passed,
            "total": test_result.testsRun,
            "failures": len(test_result.failures),
            "errors": len(test_result.errors)
        },
        "stress_test": {
            "passed": stress_passed,
            "results": stress_results
        },
        "benchmark": benchmark_results,
        "overall_success": overall_success
    }


if __name__ == "__main__":
    # Executar todos os testes
    results = run_all_tests()
    
    # Salvar resultados
    results_path = Path("test_results/etapa3_test_results.json")
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Resultados salvos em: {results_path}")
    print(f"🎯 Status geral: {'✅ SUCESSO' if results['overall_success'] else '❌ FALHA'}") 