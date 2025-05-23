#!/usr/bin/env python3
"""
🚀 AUTOMAÇÃO CI/CD - INSIGHTS-AI
===============================

Script para automação de testes em pipelines de CI/CD.
Suporta diferentes ambientes e configurações.
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

class CITestRunner:
    """Runner de testes para CI/CD"""
    
    def __init__(self, environment='development', coverage=False, output_format='console'):
        self.environment = environment
        self.coverage = coverage
        self.output_format = output_format
        self.start_time = datetime.now()
        self.results = {}
        
    def log(self, message, level="INFO"):
        """Log com timestamp para CI/CD"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = {
            "INFO": "ℹ️",
            "SUCCESS": "✅", 
            "WARNING": "⚠️",
            "ERROR": "❌",
            "DEBUG": "🐛"
        }.get(level, "ℹ️")
        
        print(f"[{timestamp}] {prefix} {message}")
        
        # Para CI/CD, também log para stderr se for erro
        if level == "ERROR":
            print(f"::error::{message}", file=sys.stderr)
        elif level == "WARNING":
            print(f"::warning::{message}", file=sys.stderr)
    
    def setup_environment(self):
        """Configurar ambiente para testes"""
        self.log("Configurando ambiente de testes...")
        
        # Verificar Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.log(f"Python version: {python_version}")
        
        # Verificar estrutura de projeto
        required_paths = [
            "src/insights",
            "src/tests", 
            "data",
            "pyproject.toml"
        ]
        
        missing_paths = []
        for path in required_paths:
            if not Path(path).exists():
                missing_paths.append(path)
        
        if missing_paths:
            self.log(f"Estrutura de projeto incompleta: {missing_paths}", "ERROR")
            return False
        
        # Instalar dependências se necessário
        if self.environment in ['ci', 'production']:
            self.log("Verificando dependências...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], 
                             check=True, capture_output=True)
                self.log("Dependências instaladas com sucesso")
            except subprocess.CalledProcessError as e:
                self.log(f"Erro ao instalar dependências: {e}", "ERROR")
                return False
        
        return True
    
    def run_tests_with_pytest(self):
        """Executar testes usando pytest"""
        self.log("Executando testes com pytest...")
        
        # Construir comando pytest
        cmd = [sys.executable, "-m", "pytest", "src/tests/"]
        
        # Adicionar flags baseadas no ambiente
        if self.environment == 'ci':
            cmd.extend([
                "--tb=short",
                "--disable-warnings",
                "-v",
                "--junitxml=test-results.xml"
            ])
        elif self.environment == 'development':
            cmd.extend([
                "--tb=long", 
                "-v",
                "-x"  # Parar no primeiro erro
            ])
        elif self.environment == 'production':
            cmd.extend([
                "--tb=no",
                "--disable-warnings",
                "-q"
            ])
        
        # Adicionar coverage se solicitado
        if self.coverage:
            cmd.extend([
                "--cov=src/insights",
                "--cov-report=xml",
                "--cov-report=html",
                "--cov-report=term-missing"
            ])
        
        # Executar pytest
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Log output
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            
            return result.returncode == 0
            
        except Exception as e:
            self.log(f"Erro ao executar pytest: {e}", "ERROR")
            return False
    
    def run_custom_tests(self):
        """Executar testes customizados usando test_main.py"""
        self.log("Executando testes customizados...")
        
        # Adicionar src ao path
        sys.path.insert(0, 'src')
        
        try:
            from tests.test_main import InsightsTestRunner
            
            # Configurar runner baseado no ambiente
            if self.environment == 'ci':
                verbose = False
                quick = True
            elif self.environment == 'development':
                verbose = True
                quick = True
            else:  # production
                verbose = False
                quick = False
            
            # Executar testes
            runner = InsightsTestRunner(verbose=verbose, quick=quick)
            runner.run_all_tests()
            
            # Coletar resultados
            self.results = runner.results
            
            # Determinar sucesso
            total_tests = len(self.results)
            passed_tests = len([r for r in self.results.values() if r['status'] == 'PASS'])
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            
            self.log(f"Testes executados: {total_tests}")
            self.log(f"Testes aprovados: {passed_tests}")
            self.log(f"Taxa de sucesso: {success_rate:.1f}%")
            
            # Considerar sucesso se >80% dos testes passaram
            return success_rate >= 80
            
        except Exception as e:
            self.log(f"Erro ao executar testes customizados: {e}", "ERROR")
            return False
    
    def generate_reports(self):
        """Gerar relatórios de teste"""
        self.log("Gerando relatórios...")
        
        if not self.results:
            self.log("Nenhum resultado para relatório", "WARNING")
            return
        
        # Relatório JSON para CI/CD
        if self.output_format in ['json', 'all']:
            report_data = {
                'timestamp': self.start_time.isoformat(),
                'environment': self.environment,
                'duration': (datetime.now() - self.start_time).total_seconds(),
                'summary': {
                    'total_tests': len(self.results),
                    'passed': len([r for r in self.results.values() if r['status'] == 'PASS']),
                    'failed': len([r for r in self.results.values() if r['status'] in ['FAIL', 'EXCEPTION']]),
                    'success_rate': len([r for r in self.results.values() if r['status'] == 'PASS']) / len(self.results) * 100 if self.results else 0
                },
                'results': self.results
            }
            
            with open('test-report.json', 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.log("Relatório JSON gerado: test-report.json")
        
        # Relatório HTML para CI/CD
        if self.output_format in ['html', 'all']:
            self.generate_html_report()
    
    def generate_html_report(self):
        """Gerar relatório HTML"""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results.values() if r['status'] == 'PASS'])
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Insights-AI Test Report</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; }}
        .summary {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; }}
        .metric h3 {{ margin: 0 0 10px 0; color: #333; }}
        .metric .value {{ font-size: 24px; font-weight: bold; }}
        .pass {{ color: #28a745; }}
        .fail {{ color: #dc3545; }}
        .results {{ margin-top: 30px; }}
        .test-item {{ background: white; margin: 10px 0; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .test-name {{ font-weight: bold; margin-bottom: 5px; }}
        .test-status {{ padding: 4px 8px; border-radius: 4px; color: white; font-size: 12px; }}
        .status-pass {{ background: #28a745; }}
        .status-fail {{ background: #dc3545; }}
        .test-duration {{ color: #666; font-size: 14px; }}
        .progress-bar {{ width: 100%; height: 20px; background: #e9ecef; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: linear-gradient(90deg, #28a745, #20c997); transition: width 0.3s ease; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Insights-AI Test Report</h1>
        <p><strong>Environment:</strong> {self.environment}</p>
        <p><strong>Timestamp:</strong> {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Duration:</strong> {(datetime.now() - self.start_time).total_seconds():.2f}s</p>
    </div>
    
    <div class="summary">
        <div class="metric">
            <h3>Total Tests</h3>
            <div class="value">{total_tests}</div>
        </div>
        <div class="metric">
            <h3>Passed</h3>
            <div class="value pass">{passed_tests}</div>
        </div>
        <div class="metric">
            <h3>Failed</h3>
            <div class="value fail">{failed_tests}</div>
        </div>
        <div class="metric">
            <h3>Success Rate</h3>
            <div class="value">{success_rate:.1f}%</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {success_rate}%"></div>
            </div>
        </div>
    </div>
    
    <div class="results">
        <h2>Test Results</h2>
        """
        
        for test_name, result in self.results.items():
            status = result.get('status', 'UNKNOWN')
            duration = result.get('duration', 0)
            error = result.get('error', '')
            
            status_class = 'status-pass' if status == 'PASS' else 'status-fail'
            
            html_content += f"""
        <div class="test-item">
            <div class="test-name">{test_name}</div>
            <span class="test-status {status_class}">{status}</span>
            <div class="test-duration">Duration: {duration:.2f}s</div>
            """
            
            if error:
                html_content += f'<div style="color: #dc3545; margin-top: 10px; font-size: 14px;"><strong>Error:</strong> {error}</div>'
            
            html_content += "</div>"
        
        html_content += """
    </div>
</body>
</html>
        """
        
        with open('test-report.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.log("Relatório HTML gerado: test-report.html")
    
    def check_quality_gates(self):
        """Verificar quality gates para CI/CD"""
        if not self.results:
            self.log("Nenhum resultado para verificar quality gates", "WARNING")
            return False
        
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results.values() if r['status'] == 'PASS'])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Quality gates baseados no ambiente
        quality_gates = {
            'development': {'min_success_rate': 70, 'max_duration': 300},
            'ci': {'min_success_rate': 80, 'max_duration': 600},
            'production': {'min_success_rate': 95, 'max_duration': 900}
        }
        
        gates = quality_gates.get(self.environment, quality_gates['development'])
        
        # Verificar taxa de sucesso
        if success_rate < gates['min_success_rate']:
            self.log(f"Quality gate FALHOU: Taxa de sucesso {success_rate:.1f}% < {gates['min_success_rate']}%", "ERROR")
            return False
        
        # Verificar duração
        total_duration = (datetime.now() - self.start_time).total_seconds()
        if total_duration > gates['max_duration']:
            self.log(f"Quality gate FALHOU: Duração {total_duration:.1f}s > {gates['max_duration']}s", "ERROR")
            return False
        
        self.log(f"Quality gates APROVADOS: {success_rate:.1f}% sucesso em {total_duration:.1f}s", "SUCCESS")
        return True
    
    def cleanup(self):
        """Limpeza pós-execução"""
        self.log("Executando limpeza...")
        
        # Remover arquivos temporários
        temp_files = [
            'temp_*.csv',
            '.coverage',
            '__pycache__'
        ]
        
        for pattern in temp_files:
            for file_path in Path('.').glob(pattern):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        import shutil
                        shutil.rmtree(file_path)
                except Exception as e:
                    self.log(f"Erro ao remover {file_path}: {e}", "WARNING")
    
    def run(self, test_runner='custom'):
        """Executar pipeline completo de testes"""
        self.log(f"Iniciando pipeline de testes - Ambiente: {self.environment}")
        
        try:
            # 1. Setup do ambiente
            if not self.setup_environment():
                return False
            
            # 2. Executar testes
            if test_runner == 'pytest':
                success = self.run_tests_with_pytest()
            else:
                success = self.run_custom_tests()
            
            # 3. Gerar relatórios
            self.generate_reports()
            
            # 4. Verificar quality gates
            if not self.check_quality_gates():
                success = False
            
            # 5. Limpeza
            self.cleanup()
            
            # Resultado final
            duration = (datetime.now() - self.start_time).total_seconds()
            if success:
                self.log(f"Pipeline CONCLUÍDO COM SUCESSO em {duration:.1f}s", "SUCCESS")
            else:
                self.log(f"Pipeline FALHOU após {duration:.1f}s", "ERROR")
            
            return success
            
        except Exception as e:
            self.log(f"Erro inesperado no pipeline: {e}", "ERROR")
            return False

def main():
    """Função principal para CI/CD"""
    parser = argparse.ArgumentParser(description='CI/CD Test Runner for Insights-AI')
    parser.add_argument('--environment', choices=['development', 'ci', 'production'], 
                       default='development', help='Environment to run tests in')
    parser.add_argument('--coverage', action='store_true', help='Enable code coverage')
    parser.add_argument('--output-format', choices=['console', 'json', 'html', 'all'], 
                       default='console', help='Output format for reports')
    parser.add_argument('--test-runner', choices=['custom', 'pytest'], 
                       default='custom', help='Test runner to use')
    
    args = parser.parse_args()
    
    # Executar pipeline
    runner = CITestRunner(
        environment=args.environment,
        coverage=args.coverage,
        output_format=args.output_format
    )
    
    success = runner.run(test_runner=args.test_runner)
    
    # Exit code para CI/CD
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
