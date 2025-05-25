#!/usr/bin/env python3
"""
🔍 VERIFICADOR DE DEPENDÊNCIAS - ADVANCED ANALYTICS ENGINE V2.0
===============================================================

Script para verificar se todas as dependências necessárias
para executar os testes estão instaladas corretamente.
"""

import sys
import importlib
from typing import Dict, List, Tuple

def check_dependency(module_name: str, package_name: str = None) -> Tuple[bool, str]:
    """
    Verificar se uma dependência está disponível.
    
    Args:
        module_name: Nome do módulo para importar
        package_name: Nome do pacote para instalação (se diferente do módulo)
    
    Returns:
        Tuple[bool, str]: (disponível, versão ou erro)
    """
    if package_name is None:
        package_name = module_name
    
    try:
        module = importlib.import_module(module_name)
        
        # Tentar obter versão
        version = "desconhecida"
        if hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'version'):
            version = module.version
        elif hasattr(module, 'VERSION'):
            version = module.VERSION
        
        return True, version
        
    except ImportError as e:
        return False, f"pip install {package_name}"

def main():
    print("🔍 VERIFICADOR DE DEPENDÊNCIAS")
    print("=" * 50)
    print("Verificando dependências para Advanced Analytics Engine V2.0...")
    print()
    
    # Dependências obrigatórias
    required_deps = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "scikit-learn"),
        ("pytest", "pytest")
    ]
    
    # Dependências opcionais
    optional_deps = [
        ("xgboost", "xgboost"),
        ("scipy", "scipy"),
        ("tracemalloc", None)  # Built-in no Python 3.4+
    ]
    
    # Verificar dependências obrigatórias
    print("📋 DEPENDÊNCIAS OBRIGATÓRIAS:")
    print("-" * 30)
    
    missing_required = []
    
    for module_name, package_name in required_deps:
        available, info = check_dependency(module_name, package_name)
        
        if available:
            print(f"✅ {module_name:<15} v{info}")
        else:
            print(f"❌ {module_name:<15} AUSENTE - {info}")
            missing_required.append((module_name, info))
    
    print()
    
    # Verificar dependências opcionais
    print("🔧 DEPENDÊNCIAS OPCIONAIS:")
    print("-" * 30)
    
    missing_optional = []
    
    for module_name, package_name in optional_deps:
        if module_name == "tracemalloc":
            # tracemalloc é built-in
            try:
                import tracemalloc
                print(f"✅ {module_name:<15} built-in")
            except ImportError:
                print(f"⚠️ {module_name:<15} não disponível (Python < 3.4)")
        else:
            available, info = check_dependency(module_name, package_name)
            
            if available:
                print(f"✅ {module_name:<15} v{info}")
            else:
                print(f"⚠️ {module_name:<15} ausente - {info}")
                missing_optional.append((module_name, info))
    
    print()
    
    # Verificar versão do Python
    print("🐍 VERSÃO DO PYTHON:")
    print("-" * 20)
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    if sys.version_info >= (3, 7):
        print(f"✅ Python {python_version} (compatível)")
    elif sys.version_info >= (3, 6):
        print(f"⚠️ Python {python_version} (funciona, mas recomendado 3.7+)")
    else:
        print(f"❌ Python {python_version} (incompatível - necessário 3.6+)")
    
    print()
    
    # Resumo e instruções
    print("📊 RESUMO:")
    print("-" * 10)
    
    if not missing_required:
        print("✅ Todas as dependências obrigatórias estão instaladas!")
        
        if not missing_optional:
            print("✅ Todas as dependências opcionais estão instaladas!")
            print("🎉 Sistema totalmente configurado para testes!")
        else:
            print("⚠️ Algumas dependências opcionais estão ausentes.")
            print("   Os testes funcionarão, mas com funcionalidades limitadas.")
    else:
        print("❌ Dependências obrigatórias ausentes!")
        print("   Os testes NÃO funcionarão até que sejam instaladas.")
    
    print()
    
    # Instruções de instalação
    if missing_required or missing_optional:
        print("🛠️ INSTRUÇÕES DE INSTALAÇÃO:")
        print("-" * 30)
        
        if missing_required:
            print("Dependências obrigatórias:")
            for module_name, install_cmd in missing_required:
                print(f"  {install_cmd}")
        
        if missing_optional:
            print("\nDependências opcionais (recomendadas):")
            for module_name, install_cmd in missing_optional:
                print(f"  {install_cmd}")
        
        print("\nOu instalar todas de uma vez:")
        print("  pip install pandas numpy scikit-learn pytest xgboost scipy")
    
    print()
    
    # Verificação específica do Advanced Analytics Engine
    print("🔧 VERIFICAÇÃO DO ENGINE:")
    print("-" * 25)
    
    try:
        # Tentar diferentes caminhos de importação
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        
        # Adicionar caminhos possíveis
        possible_paths = [
            project_root,
            os.path.join(project_root, 'src'),
            current_dir,
            os.path.dirname(current_dir)
        ]
        
        for path in possible_paths:
            if path not in sys.path:
                sys.path.insert(0, path)
        
        # Tentar importar o engine
        try:
            from src.insights.tools.advanced.advanced_analytics_engine_tool import AdvancedAnalyticsEngineTool
        except ImportError:
            # Fallback para importação direta
            sys.path.insert(0, os.path.join(project_root, 'src', 'insights', 'tools', 'advanced'))
            from advanced_analytics_engine_tool import AdvancedAnalyticsEngineTool
        
        print("✅ Advanced Analytics Engine V2.0 importado com sucesso!")
        
        # Verificar se pode ser instanciado
        engine = AdvancedAnalyticsEngineTool()
        print("✅ Engine instanciado com sucesso!")
        
        # Verificar métodos principais
        main_methods = ['_run', '_load_data', '_validate_inputs', '_prepare_features']
        available_methods = [method for method in main_methods if hasattr(engine, method)]
        print(f"✅ Métodos principais disponíveis: {len(available_methods)}/{len(main_methods)}")
        
    except ImportError as e:
        print(f"❌ Erro ao importar engine: {e}")
        print("   Verificando localização do arquivo...")
        
        # Verificar se arquivo existe
        engine_path = os.path.join(project_root, 'src', 'insights', 'tools', 'advanced', 'advanced_analytics_engine_tool.py')
        if os.path.exists(engine_path):
            print(f"   ✅ Arquivo encontrado: {engine_path}")
            print("   ❌ Problema de importação de módulo")
        else:
            print(f"   ❌ Arquivo não encontrado: {engine_path}")
            print("   Verifique se o arquivo está no local correto")
            
    except Exception as e:
        print(f"⚠️ Engine importado mas erro na instanciação: {e}")
    
    print()
    
    # Status final
    if not missing_required:
        print("🎯 STATUS: PRONTO PARA EXECUTAR TESTES!")
        print("   Execute: python test_advanced_analytics_engine_tool.py")
        return 0
    else:
        print("🚫 STATUS: DEPENDÊNCIAS AUSENTES")
        print("   Instale as dependências obrigatórias primeiro.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 