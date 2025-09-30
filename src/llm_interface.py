"""
Interface para LLMs com suporte a mock e providers reais.
"""
import os
import logging
from typing import List, Dict, Any
import random

logger = logging.getLogger(__name__)


class LLMInterface:
    """Interface unificada para diferentes providers de LLM."""
    
    def __init__(self, provider: str = None):
        """
        Inicializa a interface LLM.
        
        Args:
            provider: Provider a usar (mock, openai, anthropic)
        """
        self.provider = provider or os.getenv('LLM_PROVIDER', 'mock')
        logger.info(f"LLM Interface inicializada com provider: {self.provider}")
        
        if self.provider == 'openai':
            self._init_openai()
        elif self.provider == 'anthropic':
            self._init_anthropic()
        elif self.provider == 'mock':
            logger.info("Usando LLM Mock para demonstração")
        else:
            raise ValueError(f"Provider desconhecido: {self.provider}")
    
    def _init_openai(self):
        """Inicializa cliente OpenAI."""
        try:
            import openai
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY não configurada")
            openai.api_key = api_key
            self.client = openai
            logger.info("Cliente OpenAI inicializado")
        except ImportError:
            raise ImportError("Instale: pip install openai")
    
    def _init_anthropic(self):
        """Inicializa cliente Anthropic."""
        try:
            import anthropic
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY não configurada")
            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info("Cliente Anthropic inicializado")
        except ImportError:
            raise ImportError("Instale: pip install anthropic")
    
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        context: List[Dict[str, Any]] = None,
        temperature: float = None
    ) -> str:
        """
        Gera resposta do LLM.
        
        Args:
            messages: Histórico de mensagens
            context: Contexto RAG (chunks relevantes)
            temperature: Temperatura para geração
            
        Returns:
            Resposta do modelo
        """
        temperature = temperature or float(os.getenv('TEMPERATURE', 0.7))
        
        if self.provider == 'mock':
            return self._generate_mock_response(messages, context)
        elif self.provider == 'openai':
            return self._generate_openai_response(messages, context, temperature)
        elif self.provider == 'anthropic':
            return self._generate_anthropic_response(messages, context, temperature)
    
    def _generate_mock_response(
        self,
        messages: List[Dict[str, str]],
        context: List[Dict[str, Any]] = None
    ) -> str:
        """
        Gera resposta mock inteligente baseada no contexto.
        """
        last_message = messages[-1]['content'].lower()
        
        # Se tem contexto RAG, criar resposta baseada nos dados
        if context and len(context) > 0:
            return self._create_rag_response(context, last_message)
        
        # Respostas gerais baseadas em padrões
        if any(word in last_message for word in ['olá', 'oi', 'bom dia', 'boa tarde', 'boa noite']):
            return "Olá! Sou seu assistente de folha de pagamento. Posso ajudar você com informações sobre salários, bônus, descontos e datas de pagamento. Como posso ajudar?"
        
        if any(word in last_message for word in ['ajuda', 'help', 'o que você faz']):
            return """Posso ajudar você com:
• Consultas sobre folha de pagamento de funcionários
• Informações sobre salários, bônus e descontos
• Dados de competência específica (ex: janeiro/2025)
• Comparações entre períodos
• Estatísticas gerais

Tente perguntar algo como: "Qual o salário líquido da Ana em março de 2025?" ou "Mostre os pagamentos do Bruno Lima"."""
        
        if any(word in last_message for word in ['obrigado', 'obrigada', 'valeu']):
            return "Por nada! Fico feliz em ajudar. Se precisar de mais alguma informação, é só perguntar! 😊"
        
        # Resposta padrão
        return "Entendo sua pergunta. Para consultas sobre folha de pagamento, posso buscar informações específicas por nome, competência, ou período. Poderia reformular sua pergunta incluindo um nome de funcionário ou período?"
    
    def _create_rag_response(self, context: List[Dict[str, Any]], query: str) -> str:
        """Cria resposta baseada no contexto RAG."""
        
        # Extrair informações dos chunks
        responses = []
        
        for i, chunk in enumerate(context[:3], 1):
            metadata = chunk['metadata']
            score = chunk.get('score', 0)
            
            # Formatar dinheiro
            def fmt_money(val):
                return f"R$ {float(val):,.2f}".replace(',', '_').replace('.', ',').replace('_', '.')
            
            # Criar resposta estruturada
            response_parts = []
            
            # Identificação
            response_parts.append(
                f"**{metadata['name']}** (ID: {metadata['employee_id']}) - "
                f"Competência: {metadata['competency']}"
            )
            
            # Informações financeiras baseadas na pergunta
            if any(word in query for word in ['líquido', 'liquido', 'net', 'total']):
                response_parts.append(f"• Pagamento Líquido: {fmt_money(metadata['net_pay'])}")
            
            if any(word in query for word in ['salário', 'salario', 'base']):
                response_parts.append(f"• Salário Base: {fmt_money(metadata['base_salary'])}")
            
            if any(word in query for word in ['bônus', 'bonus']):
                response_parts.append(f"• Bônus: {fmt_money(metadata['bonus'])}")
            
            if any(word in query for word in ['inss']):
                response_parts.append(f"• Desconto INSS: {fmt_money(metadata['deductions_inss'])}")
            
            if any(word in query for word in ['irrf', 'imposto']):
                response_parts.append(f"• Desconto IRRF: {fmt_money(metadata['deductions_irrf'])}")
            
            if any(word in query for word in ['benefício', 'beneficio', 'vt', 'vr']):
                response_parts.append(f"• Benefícios (VT/VR): {fmt_money(metadata['benefits_vt_vr'])}")
            
            if any(word in query for word in ['data', 'quando', 'pagamento']):
                response_parts.append(f"• Data de Pagamento: {metadata['payment_date'][:10]}")
            
            # Se não encontrou nenhum filtro específico, mostrar resumo
            if len(response_parts) == 1:
                response_parts.append(f"• Salário Base: {fmt_money(metadata['base_salary'])}")
                response_parts.append(f"• Bônus: {fmt_money(metadata['bonus'])}")
                response_parts.append(f"• Pagamento Líquido: {fmt_money(metadata['net_pay'])}")
            
            responses.append('\n'.join(response_parts))
        
        # Montar resposta final
        if len(responses) == 1:
            final_response = f"Encontrei a seguinte informação:\n\n{responses[0]}"
        else:
            numbered_responses = [f"{i}. {resp}" for i, resp in enumerate(responses, 1)]
            final_response = f"Encontrei {len(responses)} resultados:\n\n" + '\n\n'.join(numbered_responses)
        
        # Adicionar nota sobre fonte
        final_response += f"\n\n_Fonte: linhas {', '.join([str(c['row_index']+2) for c in context[:3]])} do dataset de folha de pagamento._"
        
        return final_response
    
    def _generate_openai_response(
        self,
        messages: List[Dict[str, str]],
        context: List[Dict[str, Any]],
        temperature: float
    ) -> str:
        """Gera resposta usando OpenAI."""
        # Preparar mensagens com contexto
        system_message = self._build_system_prompt(context)
        
        api_messages = [{"role": "system", "content": system_message}]
        api_messages.extend(messages)
        
        response = self.client.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=api_messages,
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    def _generate_anthropic_response(
        self,
        messages: List[Dict[str, str]],
        context: List[Dict[str, Any]],
        temperature: float
    ) -> str:
        """Gera resposta usando Anthropic."""
        system_message = self._build_system_prompt(context)
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            system=system_message,
            messages=messages,
            temperature=temperature
        )
        
        return response.content[0].text
    
    def _build_system_prompt(self, context: List[Dict[str, Any]]) -> str:
        """Constrói prompt de sistema com contexto RAG."""
        base_prompt = """Você é um assistente especializado em folha de pagamento.
Você deve responder perguntas sobre dados de folha de pagamento de forma clara e precisa.
Sempre cite a fonte dos dados (employee_id, competência) quando disponível.
Formate valores monetários em reais brasileiros (R$)."""
        
        if context and len(context) > 0:
            context_text = "\n\nContexto relevante:\n"
            for chunk in context:
                context_text += f"\n{chunk['text']}\n"
            base_prompt += context_text
        
        return base_prompt 