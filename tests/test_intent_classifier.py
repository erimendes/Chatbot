"""
Testes para o classificador de intenções.
"""
import os
import sys
import pytest

# Adicionar diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from intent_classifier import IntentClassifier


@pytest.fixture
def classifier():
    """Fixture para criar instância do classificador."""
    return IntentClassifier()


def test_payroll_intent(classifier):
    """Testa detecção de consultas de folha de pagamento."""
    queries = [
        "Qual o salário da Ana?",
        "Mostre o pagamento do Bruno em março",
        "Quanto foi o bônus em maio?",
        "Quais os descontos de INSS?"
    ]
    
    for query in queries:
        intent, confidence = classifier.classify(query)
        assert intent == IntentClassifier.INTENT_PAYROLL
        assert confidence > 0


def test_stats_intent(classifier):
    """Testa detecção de consultas de estatísticas."""
    queries = [
        "Qual a média de pagamentos?",
        "Quantos funcionários temos?",
        "Total pago no período",
        "Estatísticas gerais"
    ]
    
    for query in queries:
        intent, confidence = classifier.classify(query)
        assert intent == IntentClassifier.INTENT_STATS
        assert confidence > 0


def test_help_intent(classifier):
    """Testa detecção de pedidos de ajuda."""
    queries = [
        "Ajuda",
        "O que você pode fazer?",
        "Como usar este chatbot?",
        "Help"
    ]
    
    for query in queries:
        intent, confidence = classifier.classify(query)
        assert intent == IntentClassifier.INTENT_HELP
        assert confidence > 0


def test_general_intent(classifier):
    """Testa detecção de conversa geral."""
    queries = [
        "Olá",
        "Bom dia",
        "Como vai?",
        "Obrigado"
    ]
    
    for query in queries:
        intent, confidence = classifier.classify(query)
        assert intent == IntentClassifier.INTENT_GENERAL


def test_extract_filters(classifier):
    """Testa extração de filtros das queries."""
    # Extrair nome
    filters = classifier.extract_filters("Qual o salário da Ana?")
    assert 'name' in filters
    assert filters['name'] == 'Ana'
    
    # Extrair competência por código
    filters = classifier.extract_filters("Pagamento em 2025-03")
    assert 'competency' in filters
    assert filters['competency'] == '2025-03'
    
    # Extrair competência por nome do mês
    filters = classifier.extract_filters("Salário de março")
    assert 'competency' in filters
    assert filters['competency'] == '2025-03'
    
    # Extrair employee_id
    filters = classifier.extract_filters("Dados do funcionário E001")
    assert 'employee_id' in filters
    assert filters['employee_id'] == 'E001'


def test_multiple_intents(classifier):
    """Testa queries com múltiplos indicadores."""
    # Query que mistura payroll e stats
    intent, confidence = classifier.classify("Qual o total de bônus da Ana?")
    # Deve priorizar payroll pois tem nome específico
    assert intent == IntentClassifier.INTENT_PAYROLL 