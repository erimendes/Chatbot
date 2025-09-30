"""
Testes para o gerenciador de conversação.
"""
import os
import sys
import pytest

# Adicionar diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from conversation_manager import ConversationManager


@pytest.fixture
def conversation():
    """Fixture para criar instância do gerenciador."""
    return ConversationManager(max_history=5)


def test_initialization(conversation):
    """Testa inicialização do gerenciador."""
    assert conversation is not None
    assert conversation.max_history == 5
    assert len(conversation.get_messages()) == 0


def test_add_message(conversation):
    """Testa adição de mensagens."""
    conversation.add_message('user', 'Olá!')
    conversation.add_message('assistant', 'Olá! Como posso ajudar?')
    
    messages = conversation.get_messages()
    assert len(messages) == 2
    assert messages[0]['role'] == 'user'
    assert messages[0]['content'] == 'Olá!'
    assert messages[1]['role'] == 'assistant'


def test_message_limit(conversation):
    """Testa limite de mensagens no histórico."""
    # Adicionar mais mensagens que o limite
    for i in range(10):
        conversation.add_message('user', f'Mensagem {i}')
    
    messages = conversation.get_messages()
    # Deve manter apenas as últimas 5
    assert len(messages) == 5
    # A primeira mensagem deve ser "Mensagem 5"
    assert messages[0]['content'] == 'Mensagem 5'


def test_metadata(conversation):
    """Testa armazenamento de metadados."""
    metadata = {
        'intent': 'payroll_query',
        'confidence': 0.85,
        'sources': [1, 2, 3]
    }
    
    conversation.add_message('user', 'Qual o salário da Ana?', metadata)
    
    last_metadata = conversation.get_last_metadata()
    assert last_metadata == metadata


def test_clear_history(conversation):
    """Testa limpeza do histórico."""
    conversation.add_message('user', 'Olá')
    conversation.add_message('assistant', 'Oi')
    
    assert len(conversation.get_messages()) == 2
    
    conversation.clear()
    
    assert len(conversation.get_messages()) == 0


def test_export_conversation(conversation):
    """Testa exportação da conversa."""
    conversation.add_message('user', 'Olá')
    conversation.add_message('assistant', 'Oi', {'test': True})
    
    export = conversation.export_conversation()
    
    assert 'messages' in export
    assert 'metadata' in export
    assert 'exported_at' in export
    assert 'total_messages' in export
    assert export['total_messages'] == 2


def test_timestamp(conversation):
    """Testa adição de timestamp nas mensagens."""
    conversation.add_message('user', 'Teste')
    
    messages = conversation.get_messages()
    assert 'timestamp' in messages[0]
    # Verificar formato ISO
    assert 'T' in messages[0]['timestamp'] 