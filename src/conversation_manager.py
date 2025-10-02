# src/conversation_manager.py
"""
Gerenciador de conversação com memória de contexto.
"""
import os
import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationManager:
    """Gerencia o histórico e contexto da conversa."""
    
    def __init__(self, max_history: int = None):
        """
        Inicializa o gerenciador de conversação.
        
        Args:
            max_history: Número máximo de mensagens no histórico
        """
        self.max_history = max_history or int(os.getenv('MAX_CONVERSATION_HISTORY', 10))
        self.messages: List[Dict[str, str]] = []
        self.metadata: List[Dict[str, Any]] = []
        logger.info(f"ConversationManager inicializado (max_history={self.max_history})")
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """
        Adiciona uma mensagem ao histórico.
        
        Args:
            role: Papel (user ou assistant)
            content: Conteúdo da mensagem
            metadata: Metadados adicionais (contexto RAG, scores, etc)
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        
        self.messages.append(message)
        self.metadata.append(metadata or {})
        
        # Manter apenas as últimas N mensagens
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
            self.metadata = self.metadata[-self.max_history:]
        
        logger.debug(f"Mensagem adicionada: {role} - {len(content)} chars")
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Retorna o histórico de mensagens."""
        return self.messages.copy()
    
    def get_last_metadata(self) -> Dict[str, Any]:
        """Retorna os metadados da última mensagem."""
        if self.metadata:
            return self.metadata[-1]
        return {}
    
    def clear(self):
        """Limpa o histórico."""
        self.messages = []
        self.metadata = []
        logger.info("Histórico limpo")
    
    def export_conversation(self) -> Dict[str, Any]:
        """
        Exporta a conversa completa com metadados.
        Útil para análise e debugging.
        """
        return {
            'messages': self.messages,
            'metadata': self.metadata,
            'exported_at': datetime.now().isoformat(),
            'total_messages': len(self.messages)
        } 