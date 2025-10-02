# src/llm_interface.py
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
        logger.info(
            f"LLM Interface inicializada com provider: {self.provider}")

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

        # Respostas conversacionais mais naturais
        if any(word in last_message for word in ['olá', 'oi', 'bom dia', 'boa tarde', 'boa noite']):
            greetings = [
                "Oi! Tudo bem? 😊 Como posso te ajudar hoje?",
                "Olá! Espero que esteja tudo bem! Em que posso ser útil?",
                "Oi! Que bom te ver por aqui! Como posso ajudar?",
                "Olá! Tudo certo? O que você gostaria de saber?"
            ]
            import random
            return random.choice(greetings)

        if any(word in last_message for word in ['tudo bem', 'como você está', 'como vai']):
            return "Estou ótimo, obrigado por perguntar! 😊 E você, como está? Posso ajudar com alguma coisa?"

        if any(word in last_message for word in ['obrigado', 'obrigada', 'valeu', 'obrigadão']):
            thanks_responses = [
                "Por nada! Fico feliz em ajudar! 😊",
                "De nada! Sempre às ordens!",
                "Imagina! Foi um prazer ajudar!",
                "Por nada! Se precisar de mais alguma coisa, é só falar!"
            ]
            import random
            return random.choice(thanks_responses)

        if any(word in last_message for word in ['ajuda', 'help', 'o que você faz', 'o que voce faz']):
            return """Olha, eu sou um assistente bem versátil! 😊 

Posso conversar sobre diversos assuntos, mas tenho uma especialidade: **folha de pagamento**! 

Se você quiser saber sobre:
• Salários, bônus e descontos de funcionários
• Dados de competência específica (ex: março/2025)
• Comparações entre períodos
• Estatísticas gerais

É só perguntar! Por exemplo: "Qual o salário da Ana em maio?" ou "Mostre os pagamentos do Bruno Lima".

Mas também posso bater um papo sobre outras coisas! O que você gostaria de saber?"""

        if any(word in last_message for word in ['quem é você', 'quem voce é', 'seu nome']):
            return "Oi! Eu sou um assistente virtual bem simpático! 😊 Posso conversar sobre várias coisas, mas sou especialista em folha de pagamento. E você, como se chama?"

        if any(word in last_message for word in ['tchau', 'até logo', 'até mais', 'bye']):
            goodbye_responses = [
                "Tchau! Foi um prazer conversar! 😊",
                "Até logo! Volte sempre!",
                "Tchauzinho! Qualquer coisa é só chamar!",
                "Até mais! Foi ótimo te ajudar!"
            ]
            import random
            return random.choice(goodbye_responses)

        # Respostas para perguntas sobre folha de pagamento sem contexto específico
        if any(word in last_message for word in ['salário', 'salario', 'pagamento', 'folha', 'funcionário', 'funcionario']):
            return "Ah, você quer saber sobre folha de pagamento! 😊 Posso te ajudar com informações sobre salários, bônus, descontos e datas de pagamento. Tem algum funcionário específico em mente? Por exemplo, Ana ou Bruno?"

        # Respostas mais conversacionais para outros assuntos
        if any(word in last_message for word in ['tempo', 'clima', 'chuva', 'sol']):
            return "Opa, falando de tempo! 😄 Infelizmente não tenho acesso a dados meteorológicos em tempo real, mas posso te ajudar com outras informações! Tem alguma pergunta sobre folha de pagamento ou outro assunto?"

        if any(word in last_message for word in ['futebol', 'futebol', 'jogo', 'time']):
            return "Eita, falando de futebol! ⚽ Que legal! Infelizmente não tenho dados sobre esportes, mas adoraria bater um papo sobre isso! E se quiser saber sobre folha de pagamento, estou aqui também! 😊"

        if any(word in last_message for word in ['comida', 'restaurante', 'receita', 'culinária']):
            return "Nossa, falando de comida! 🤤 Que delícia! Infelizmente não tenho receitas, mas posso te ajudar com outras coisas! E se precisar de informações sobre folha de pagamento, é só falar!"

        # Resposta padrão mais conversacional
        return "Hmm, interessante pergunta! 🤔 Não tenho informações específicas sobre isso, mas posso conversar sobre outras coisas! Se quiser saber sobre folha de pagamento, posso te ajudar com dados de salários, bônus e descontos. O que você acha?"

    def _create_rag_response(self, context: List[Dict[str, Any]], query: str) -> str:
        """Cria resposta baseada no contexto RAG."""

        # Extrair mês específico da pergunta
        target_month = self._extract_target_month(query)

        # Extrair nome do funcionário da pergunta
        target_employee = self._extract_target_employee(query)

        # Filtrar contextos pelo mês e funcionário específicos
        filtered_context = context

        # Primeiro filtrar por funcionário se especificado
        if target_employee:
            filtered_context = [
                c for c in filtered_context
                if target_employee.lower() in c['metadata']['name'].lower()
            ]
            # Se não encontrou resultados para o funcionário específico, usar todos os resultados
            if not filtered_context:
                filtered_context = context
        # Se não especificou funcionário, incluir ambos (Ana e Bruno)
        else:
            # Manter todos os resultados para mostrar ambos funcionários
            pass

        # Depois filtrar por mês se especificado
        if target_month:
            filtered_context = [
                c for c in filtered_context if target_month in c['metadata']['competency']]
            # Se não encontrou resultados para o mês específico, usar todos os resultados
            if not filtered_context:
                filtered_context = context

        # Detectar se é uma pergunta sobre campo específico
        is_specific_field_query = self._is_specific_field_query(query)

        # Se é pergunta específica sobre campo E especificou funcionário, pegar apenas o melhor resultado
        if is_specific_field_query and target_employee and filtered_context:
            # Pegar apenas o resultado com maior score
            filtered_context = [filtered_context[0]]
        # Se não especificou funcionário, organizar por funcionário para mostrar ambos
        elif not target_employee and filtered_context:
            # Agrupar por funcionário e pegar o melhor resultado de cada um
            employee_groups = {}
            for chunk in filtered_context:
                employee_name = chunk['metadata']['name']
                if employee_name not in employee_groups:
                    employee_groups[employee_name] = []
                employee_groups[employee_name].append(chunk)

            # Pegar o melhor resultado de cada funcionário
            filtered_context = []
            for employee_name, chunks in employee_groups.items():
                # Ordenar por score e pegar o melhor
                best_chunk = max(chunks, key=lambda x: x.get('score', 0))
                filtered_context.append(best_chunk)

        # Extrair informações dos chunks filtrados
        responses = []

        for i, chunk in enumerate(filtered_context[:3], 1):
            metadata = chunk['metadata']
            score = chunk.get('score', 0)

            # Formatar dinheiro
            def fmt_money(val):
                return f"R$ {float(val):,.2f}".replace(',', '_').replace('.', ',').replace('_', '.')

            # Formatar data
            def fmt_date(date_str):
                try:
                    from datetime import datetime
                    date_obj = datetime.strptime(
                        str(date_str)[:10], '%Y-%m-%d')
                    return date_obj.strftime('%d/%m/%Y')
                except:
                    return str(date_str)[:10]

            # Criar resposta estruturada e mais conversacional
            response_parts = []

            # Identificação mais amigável
            employee_name = metadata['name']
            competency = metadata['competency']
            year, month = competency.split('-')
            month_names = {
                '01': 'janeiro', '02': 'fevereiro', '03': 'março', '04': 'abril',
                '05': 'maio', '06': 'junho', '07': 'julho', '08': 'agosto',
                '09': 'setembro', '10': 'outubro', '11': 'novembro', '12': 'dezembro'
            }
            month_name = month_names.get(month, month)

            response_parts.append(
                f"**{employee_name}** (ID: {metadata['employee_id']}) - {month_name.title()} de {year}"
            )

            # Informações financeiras baseadas na pergunta com contexto
            if any(word in query for word in ['líquido', 'liquido', 'net', 'total', 'recebeu', 'recebe']):
                net_pay = fmt_money(metadata['net_pay'])
                response_parts.append(f"💰 **Pagamento Líquido:** {net_pay}")

            if any(word in query for word in ['salário', 'salario', 'base', 'bruto']):
                base_salary = fmt_money(metadata['base_salary'])
                response_parts.append(f"💵 **Salário Base:** {base_salary}")

            if any(word in query for word in ['bônus', 'bonus', 'adicional']):
                bonus = fmt_money(metadata['bonus'])
                if float(metadata['bonus']) > 0:
                    response_parts.append(f"🎁 **Bônus:** {bonus}")
                else:
                    response_parts.append(
                        f"🎁 **Bônus:** R$ 0,00 (não houve bônus neste mês)")

            if any(word in query for word in ['inss', 'previdência', 'previdencia']):
                inss = fmt_money(metadata['deductions_inss'])
                response_parts.append(f"🏛️ **Desconto INSS:** {inss}")

            if any(word in query for word in ['irrf', 'imposto', 'renda']):
                irrf = fmt_money(metadata['deductions_irrf'])
                response_parts.append(f"📊 **Desconto IRRF:** {irrf}")

            if any(word in query for word in ['benefício', 'beneficio', 'vt', 'vr', 'vale']):
                benefits = fmt_money(metadata['benefits_vt_vr'])
                response_parts.append(f"🎫 **Benefícios (VT/VR):** {benefits}")

            if any(word in query for word in ['data', 'quando', 'pagamento', 'pago']):
                payment_date = fmt_date(metadata['payment_date'])
                response_parts.append(
                    f"📅 **Data de Pagamento:** {payment_date}")

            # Se não encontrou nenhum filtro específico, mostrar resumo completo
            if len(response_parts) == 1:
                response_parts.extend([
                    f"💵 **Salário Base:** {fmt_money(metadata['base_salary'])}",
                    f"🎁 **Bônus:** {fmt_money(metadata['bonus'])}",
                    f"🏛️ **Desconto INSS:** {fmt_money(metadata['deductions_inss'])}",
                    f"📊 **Desconto IRRF:** {fmt_money(metadata['deductions_irrf'])}",
                    f"💰 **Pagamento Líquido:** {fmt_money(metadata['net_pay'])}",
                    f"📅 **Data de Pagamento:** {fmt_date(metadata['payment_date'])}"
                ])

            responses.append('\n'.join(response_parts))

        # Montar resposta final mais conversacional
        if len(responses) == 1:
            final_response = f"Perfeito! Encontrei a informação que você procurava: 😊\n\n{responses[0]}"
        else:
            # Se não especificou funcionário, mostrar informações de ambos
            if not target_employee:
                final_response = f"Aqui estão as informações de ambos os funcionários: 😊\n\n" + \
                    '\n\n'.join(responses)
            else:
                numbered_responses = [f"{i}. {resp}" for i,
                                      resp in enumerate(responses, 1)]
                final_response = f"Encontrei {len(responses)} resultados para você: 😊\n\n" + \
                    '\n\n'.join(numbered_responses)

        # Adicionar nota sobre fonte de forma mais amigável
        final_response += f"\n\n_📋 Fonte: dados extraídos das linhas {', '.join([str(c['row_index']+2) for c in filtered_context[:3]])} do sistema de folha de pagamento._"

        return final_response

    def _extract_target_month(self, query: str) -> str:
        """Extrai o mês específico mencionado na pergunta."""
        import re

        query_lower = query.lower()

        # Mapeamento de nomes de meses para códigos (incluindo abreviações)
        month_mapping = {
            'janeiro': '01', 'jan': '01',
            'fevereiro': '02', 'fev': '02',
            'março': '03', 'marco': '03', 'mar': '03',
            'abril': '04', 'abr': '04',
            'maio': '05', 'mai': '05',
            'junho': '06', 'jun': '06',
            'julho': '07', 'jul': '07',
            'agosto': '08', 'ago': '08',
            'setembro': '09', 'set': '09',
            'outubro': '10', 'out': '10',
            'novembro': '11', 'nov': '11',
            'dezembro': '12', 'dez': '12'
        }

        # Procurar por padrões de mês
        # Padrão 1: "maio", "março", "jun", etc.
        for month_name, month_code in month_mapping.items():
            if month_name in query_lower:
                return f"-{month_code}"

        # Padrão 2: "05", "03", etc. (código numérico)
        month_pattern = r'\b(0[1-9]|1[0-2])\b'
        month_match = re.search(month_pattern, query)
        if month_match:
            return f"-{month_match.group(1)}"

        # Padrão 3: "2025-05", "2025-03", etc. (formato completo)
        full_pattern = r'\b(202[0-9])-(0[1-9]|1[0-2])\b'
        full_match = re.search(full_pattern, query)
        if full_match:
            return f"-{full_match.group(2)}"

        return None

    def _extract_target_employee(self, query: str) -> str:
        """Extrai o nome do funcionário mencionado na pergunta."""
        import re

        query_lower = query.lower()

        # Mapeamento de nomes e variações
        employee_mapping = {
            'ana': 'Ana',
            'ana souza': 'Ana Souza',
            'souza': 'Ana Souza',
            'bruno': 'Bruno',
            'bruno lima': 'Bruno Lima',
            'lima': 'Bruno Lima'
        }

        # Procurar por nomes específicos
        for name_variation, full_name in employee_mapping.items():
            if name_variation in query_lower:
                return full_name

        # Procurar por padrões mais específicos
        name_patterns = [
            r'\b(ana\s+souza)\b',
            r'\b(bruno\s+lima)\b',
            r'\b(ana)\b',
            r'\b(bruno)\b'
        ]

        for pattern in name_patterns:
            match = re.search(pattern, query_lower)
            if match:
                found_name = match.group(1)
                if 'ana' in found_name:
                    return 'Ana Souza' if 'souza' in found_name else 'Ana'
                elif 'bruno' in found_name:
                    return 'Bruno Lima' if 'lima' in found_name else 'Bruno'

        return None

    def _is_specific_field_query(self, query: str) -> bool:
        """Verifica se a pergunta é sobre um campo específico."""
        query_lower = query.lower()

        # Palavras que indicam pergunta sobre campo específico
        specific_field_indicators = [
            'qual foi o',
            'quanto foi o',
            'quanto é o',
            'qual o',
            'mostre o',
            'mostre os',
            'desconto de',
            'desconto do',
            'bônus de',
            'bônus do',
            'salário de',
            'salário do',
            'pagamento de',
            'pagamento do',
            'benefício de',
            'benefício do',
            'inss de',
            'inss do',
            'irrf de',
            'irrf do'
        ]

        # Verificar se contém indicadores de campo específico
        for indicator in specific_field_indicators:
            if indicator in query_lower:
                return True

        # Verificar se pergunta é sobre um campo específico
        specific_fields = [
            'inss', 'previdência', 'previdencia',
            'irrf', 'imposto', 'renda',
            'bônus', 'bonus', 'adicional',
            'benefício', 'beneficio', 'vt', 'vr', 'vale',
            'salário', 'salario', 'base', 'bruto',
            'líquido', 'liquido', 'net', 'total'
        ]

        # Se pergunta contém apenas um campo específico, é pergunta específica
        field_count = sum(
            1 for field in specific_fields if field in query_lower)
        if field_count == 1:
            return True

        return False

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
