"""
Testes para o motor RAG.
"""
import os
import sys
import pytest
import pandas as pd

# Adicionar diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_engine import PayrollRAGEngine


@pytest.fixture
def rag_engine():
    """Fixture para criar instância do RAG Engine."""
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'payroll.csv')
    engine = PayrollRAGEngine(csv_path)
    return engine


def test_rag_engine_initialization(rag_engine):
    """Testa inicialização do RAG Engine."""
    assert rag_engine is not None
    assert len(rag_engine.df) > 0
    assert len(rag_engine.chunks) > 0
    assert rag_engine.embeddings is not None


def test_search_by_employee_name(rag_engine):
    """Testa busca por nome de funcionário."""
    results = rag_engine.search("Ana Souza", top_k=3)
    
    assert len(results) > 0
    assert results[0]['score'] > 0.5  # Alta similaridade
    
    # Verificar se Ana aparece nos resultados
    found_ana = any('Ana' in r['metadata']['name'] for r in results)
    assert found_ana


def test_search_by_competency(rag_engine):
    """Testa busca por competência."""
    results = rag_engine.search("março de 2025", top_k=3)
    
    assert len(results) > 0
    # Verificar se algum resultado é de março
    found_march = any('2025-03' in str(r['metadata']['competency']) for r in results)
    assert found_march


def test_search_net_pay(rag_engine):
    """Testa busca por pagamento líquido."""
    results = rag_engine.search("pagamento líquido da Ana em março", top_k=3)
    
    assert len(results) > 0
    
    # Verificar se o primeiro resultado tem os dados corretos
    top_result = results[0]
    assert 'Ana' in top_result['metadata']['name']


def test_filter_by_criteria(rag_engine):
    """Testa filtros estruturados."""
    # Filtrar por nome
    filtered = rag_engine.filter_by_criteria(name="Ana")
    assert len(filtered) > 0
    assert all('Ana' in name for name in filtered['name'].values)
    
    # Filtrar por competência
    filtered = rag_engine.filter_by_criteria(competency="2025-03")
    assert len(filtered) > 0
    assert all(comp == "2025-03" for comp in filtered['competency'].values)
    
    # Filtrar por employee_id
    filtered = rag_engine.filter_by_criteria(employee_id="E001")
    assert len(filtered) > 0
    assert all(eid == "E001" for eid in filtered['employee_id'].values)


def test_get_statistics(rag_engine):
    """Testa cálculo de estatísticas."""
    stats = rag_engine.get_statistics()
    
    assert 'total_records' in stats
    assert 'unique_employees' in stats
    assert 'avg_net_pay' in stats
    assert 'max_net_pay' in stats
    assert 'min_net_pay' in stats
    
    assert stats['total_records'] > 0
    assert stats['unique_employees'] > 0
    assert stats['avg_net_pay'] > 0


def test_sanitize_query(rag_engine):
    """Testa sanitização de queries maliciosas."""
    # Query normal deve passar
    result = rag_engine._sanitize_query("Ana Souza salário")
    assert result == "Ana Souza salário"
    
    # Query com SQL injection deve ser bloqueada
    result = rag_engine._sanitize_query("DROP TABLE users")
    assert result == ""
    
    result = rag_engine._sanitize_query("DELETE FROM payroll")
    assert result == ""


def test_monetary_formatting(rag_engine):
    """Testa formatação de valores monetários nas respostas."""
    results = rag_engine.search("salário da Ana em janeiro", top_k=1)
    
    assert len(results) > 0
    text = results[0]['text']
    
    # Verificar se contém formatação brasileira de moeda
    assert 'R$' in text
    # Deve usar vírgula para decimais
    assert ',' in text 