#!/usr/bin/env python
"""
Monitor em tempo real para o Insights-AI Crew
Mostra logs, progresso e uso de recursos
"""

import time
import threading
import os
import sys
from datetime import datetime
import logging

def monitor_log_file(log_file="insights_execution.log", crew_log="crew_execution.log"):
    """Monitor logs em tempo real"""
    print("🔍 MONITOR INSIGHTS-AI CREW")
    print("=" * 50)
    print(f"📋 Monitorando: {log_file}")
    print(f"📋 Crew log: {crew_log}")
    print("🔥 Pressione Ctrl+C para parar\n")
    
    # Posições dos arquivos para ler apenas novas linhas
    positions = {log_file: 0, crew_log: 0}
    
    while True:
        try:
            # Monitorar ambos os arquivos de log
            for log_path in [log_file, crew_log]:
                if os.path.exists(log_path):
                    with open(log_path, 'r', encoding='utf-8') as f:
                        f.seek(positions[log_path])
                        new_lines = f.readlines()
                        positions[log_path] = f.tell()
                        
                        for line in new_lines:
                            # Colorir logs baseado no tipo
                            line = line.strip()
                            if line:
                                timestamp = datetime.now().strftime('%H:%M:%S')
                                
                                if "ERROR" in line or "❌" in line:
                                    print(f"🔴 {timestamp} | {line}")
                                elif "WARNING" in line or "⚠️" in line:
                                    print(f"🟡 {timestamp} | {line}")
                                elif "✅" in line or "SUCCESS" in line:
                                    print(f"🟢 {timestamp} | {line}")
                                elif "🚀" in line or "INICIANDO" in line:
                                    print(f"🚀 {timestamp} | {line}")
                                elif "⏱️" in line or "EXECUTANDO" in line:
                                    print(f"⏱️ {timestamp} | {line}")
                                else:
                                    print(f"📋 {timestamp} | {line}")
            
            time.sleep(2)  # Verificar a cada 2 segundos
            
        except KeyboardInterrupt:
            print("\n🛑 Monitor interrompido pelo usuário")
            break
        except Exception as e:
            print(f"❌ Erro no monitor: {e}")
            time.sleep(5)

def show_resource_usage():
    """Mostrar uso de recursos do sistema"""
    try:
        import psutil
        
        while True:
            # CPU e Memória geral
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Procurar processos do Python/CrewAI
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                try:
                    if 'python' in proc.info['name'].lower():
                        python_processes.append(proc.info)
                except:
                    pass
            
            print(f"\n📊 RECURSOS DO SISTEMA ({datetime.now().strftime('%H:%M:%S')})")
            print(f"🖥️ CPU: {cpu_percent}% | 💾 RAM: {memory.percent}% ({memory.available/1024/1024/1024:.1f}GB livre)")
            
            if python_processes:
                print(f"🐍 Processos Python ativos: {len(python_processes)}")
            
            time.sleep(30)  # Atualizar a cada 30 segundos
            
    except ImportError:
        print("⚠️ psutil não disponível - monitoramento de recursos desabilitado")
    except KeyboardInterrupt:
        pass

def main():
    """Função principal do monitor"""
    print("🚀 INSIGHTS-AI CREW MONITOR")
    print("=" * 40)
    
    # Verificar se os arquivos de log existem
    log_files = ["insights_execution.log", "crew_execution.log"]
    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"✅ {log_file} encontrado")
        else:
            print(f"⚠️ {log_file} não encontrado (será criado quando o crew executar)")
    
    print("\n🔍 Iniciando monitoramento...")
    
    # Executar monitor de logs em thread principal
    try:
        # Thread para recursos (se disponível)
        resource_thread = threading.Thread(target=show_resource_usage, daemon=True)
        resource_thread.start()
        
        # Monitor principal de logs
        monitor_log_file()
        
    except KeyboardInterrupt:
        print("\n👋 Monitor finalizado")

if __name__ == "__main__":
    main() 