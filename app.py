"""
Chatbot de Folha de Pagamento com RAG
Aplicação Streamlit com conversação geral e consultas especializadas.
"""
import os
import sys
import logging
import streamlit as st
from dotenv import load_dotenv
import json
from datetime import datetime

# Adicionar diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rag_engine import PayrollRAGEngine
from llm_interface import LLMInterface
from conversation_manager import ConversationManager
from intent_classifier import IntentClassifier

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/chatbot.log', mode='a', encoding='utf-8')
    ] if os.path.exists('logs') or os.makedirs('logs', exist_ok=True) else [logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="Chatbot de Folha de Pagamento",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS customizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1f77b4;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .stats-card {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_engine():
    """Inicializa o motor RAG (cached para performance)."""
    try:
        csv_path = os.path.join('data', 'payroll.csv')
        logger.info(f"Inicializando RAG Engine com {csv_path}")
        engine = PayrollRAGEngine(csv_path)
        return engine
    except Exception as e:
        logger.error(f"Erro ao inicializar RAG Engine: {e}")
        st.error(f"Erro ao carregar dataset: {e}")
        return None


@st.cache_resource
def initialize_llm():
    """Inicializa a interface LLM (cached)."""
    try:
        llm = LLMInterface()
        return llm
    except Exception as e:
        logger.error(f"Erro ao inicializar LLM: {e}")
        st.error(f"Erro ao inicializar LLM: {e}")
        return None


def initialize_session_state():
    """Inicializa o estado da sessão."""
    if 'conversation' not in st.session_state:
        st.session_state.conversation = ConversationManager()
    
    if 'intent_classifier' not in st.session_state:
        st.session_state.intent_classifier = IntentClassifier()
    
    if 'message_count' not in st.session_state:
        st.session_state.message_count = 0


def display_chat_history():
    """Exibe o histórico de conversação."""
    messages = st.session_state.conversation.get_messages()
    
    for msg in messages:
        role = msg['role']
        content = msg['content']
        
        if role == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 Você:</strong><br/>
                {content}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistente:</strong><br/>
                {content}
            </div>
            """, unsafe_allow_html=True)


def process_query(query: str, rag_engine: PayrollRAGEngine, llm: LLMInterface):
    """Processa a query do usuário."""
    try:
        # Classificar intenção
        intent, confidence = st.session_state.intent_classifier.classify(query)
        logger.info(f"Query: '{query}' | Intent: {intent} | Confidence: {confidence:.2f}")
        
        # Adicionar mensagem do usuário
        st.session_state.conversation.add_message('user', query)
        
        # Processar baseado na intenção
        context = []
        
        if intent == IntentClassifier.INTENT_PAYROLL:
            # Buscar no RAG
            context = rag_engine.search(query, top_k=3)
            logger.info(f"RAG encontrou {len(context)} resultados")
        
        elif intent == IntentClassifier.INTENT_STATS:
            # Retornar estatísticas
            stats = rag_engine.get_statistics()
            response = format_statistics(stats)
            st.session_state.conversation.add_message('assistant', response, {'stats': stats})
            return
        
        # Gerar resposta do LLM
        messages = st.session_state.conversation.get_messages()
        response = llm.generate_response(messages, context)
        
        # Adicionar resposta ao histórico
        metadata = {
            'intent': intent,
            'confidence': confidence,
            'context_size': len(context)
        }
        if context:
            metadata['sources'] = [c['row_index'] for c in context]
        
        st.session_state.conversation.add_message('assistant', response, metadata)
        st.session_state.message_count += 1
        
    except Exception as e:
        logger.error(f"Erro ao processar query: {e}", exc_info=True)
        error_msg = f"Desculpe, ocorreu um erro ao processar sua mensagem: {str(e)}"
        st.session_state.conversation.add_message('assistant', error_msg)


def format_statistics(stats: dict) -> str:
    """Formata estatísticas do dataset."""
    def fmt_money(val):
        return f"R$ {float(val):,.2f}".replace(',', '_').replace('.', ',').replace('_', '.')
    
    response = "📊 **Estatísticas do Dataset de Folha de Pagamento:**\n\n"
    response += f"• Total de Registros: {stats['total_records']}\n"
    response += f"• Funcionários Únicos: {stats['unique_employees']}\n"
    response += f"• Competências: {', '.join(stats['competencies'])}\n"
    response += f"• Pagamento Médio: {fmt_money(stats['avg_net_pay'])}\n"
    response += f"• Maior Pagamento: {fmt_money(stats['max_net_pay'])}\n"
    response += f"• Menor Pagamento: {fmt_money(stats['min_net_pay'])}\n"
    response += f"• Total Pago (período): {fmt_money(stats['total_paid'])}\n"
    
    return response


def main():
    """Função principal da aplicação."""
    # Header
    st.markdown('<div class="main-header">💼 Chatbot de Folha de Pagamento</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Inicializar componentes
    initialize_session_state()
    rag_engine = initialize_rag_engine()
    llm = initialize_llm()
    
    if not rag_engine or not llm:
        st.error("Erro ao inicializar o sistema. Verifique os logs.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurações")
        
        # Informações do sistema
        st.subheader("📊 Status do Sistema")
        st.info(f"""
        **LLM Provider:** {llm.provider}  
        **Mensagens:** {st.session_state.message_count}  
        **Histórico:** {len(st.session_state.conversation.get_messages())}
        """)
        
        # Estatísticas do dataset
        if st.button("📈 Ver Estatísticas"):
            stats = rag_engine.get_statistics()
            st.markdown('<div class="stats-card">', unsafe_allow_html=True)
            st.markdown(format_statistics(stats))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Exportar conversa
        if st.button("💾 Exportar Conversa"):
            conversation_data = st.session_state.conversation.export_conversation()
            json_str = json.dumps(conversation_data, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"conversa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Limpar histórico
        if st.button("🗑️ Limpar Histórico"):
            st.session_state.conversation.clear()
            st.session_state.message_count = 0
            st.rerun()
        
        # Ajuda
        st.markdown("---")
        st.subheader("💡 Exemplos de Perguntas")
        st.markdown("""
        - Qual o salário líquido da Ana em março?
        - Mostre os pagamentos do Bruno Lima
        - Quanto foi o bônus em maio de 2025?
        - Quais os descontos de INSS da Ana?
        - Quando foi o pagamento de abril?
        """)
    
    # Área principal
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Conversa")
        
        # Container para o chat
        chat_container = st.container()
        with chat_container:
            display_chat_history()
        
        # Input do usuário
        with st.form(key='message_form', clear_on_submit=True):
            user_input = st.text_input(
                "Digite sua mensagem:",
                placeholder="Ex: Qual o salário da Ana em março de 2025?",
                key='user_input'
            )
            submit_button = st.form_submit_button("Enviar 📤")
            
            if submit_button and user_input:
                with st.spinner("Processando..."):
                    process_query(user_input, rag_engine, llm)
                st.rerun()
    
    with col2:
        st.subheader("📋 Metadados")
        
        # Mostrar metadados da última interação
        metadata = st.session_state.conversation.get_last_metadata()
        if metadata:
            st.json(metadata)
        else:
            st.info("Nenhuma mensagem ainda")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <small>Chatbot de Folha de Pagamento v1.0 | Powered by RAG + Streamlit</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main() 