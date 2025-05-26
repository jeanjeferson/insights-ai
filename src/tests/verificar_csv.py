import pandas as pd

print("🔍 VERIFICANDO ARQUIVO CSV")
print("=" * 30)

try:
    # Ler primeiras linhas
    df = pd.read_csv('data/vendas.csv', sep=';', encoding='utf-8', nrows=3)
    
    print(f"📊 Colunas encontradas: {len(df.columns)}")
    print(f"📋 Nomes das colunas:")
    for i, col in enumerate(df.columns):
        print(f"   {i+1:2d}. {col}")
    
    print(f"\n📈 Primeiras 3 linhas:")
    print(df.to_string(index=False))
    
    # Verificar total de linhas
    total_lines = sum(1 for line in open('data/vendas.csv', 'r', encoding='utf-8'))
    print(f"\n📊 Total de linhas no arquivo: {total_lines:,}")
    
    # Tamanho do arquivo
    import os
    file_size = os.path.getsize('data/vendas.csv')
    print(f"📁 Tamanho do arquivo: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
except Exception as e:
    print(f"❌ Erro: {e}") 