"""Script de teste rápido para verificar o carregamento do dataset."""
import sys
import os

# Adicionar src ao path
sys.path.insert(0, 'src')

from rag_engine import PayrollRAGEngine

try:
    print("🔄 Carregando RAG Engine...")
    engine = PayrollRAGEngine('data/payroll.csv')
    print(f"✅ Dataset carregado com sucesso!")
    print(f"📊 Total de registros: {len(engine.df)}")
    print(f"📦 Total de chunks: {len(engine.chunks)}")
    print(f"\n🔍 Primeiras linhas:")
    print(engine.df.head())
    print(f"\n✅ Tudo funcionando! Você pode rodar: streamlit run app.py")
except Exception as e:
    print(f"❌ Erro: {e}")
    import traceback
    traceback.print_exc() 