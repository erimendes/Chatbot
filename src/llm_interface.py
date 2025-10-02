# src/llm_interface.py

import requests
import logging

class LLMInterface:
    def __init__(self, model_name="mistral"):
        self.logger = logging.getLogger(__name__)
        self.api_url = "http://localhost:11434/api/generate"
        self.model_name = model_name
        self.provider = f"Ollama ({model_name})"

    def generate_response(self, messages, context=None):
        try:
            prompt = self._build_prompt(messages, context)

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }

            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()

            result = response.json()
            return result["response"]

        except Exception as e:
            self.logger.error(f"Erro ao chamar Ollama: {e}", exc_info=True)
            return "Desculpe, houve um erro ao gerar a resposta."

    def _build_prompt(self, messages, context):
        """
        Monta o prompt a partir do histórico de mensagens + contexto RAG.
        """
        chat_log = ""
        if context:
            chat_log += "Informações relevantes encontradas:\n"
            for i, c in enumerate(context, 1):
                if 'text' in c:
                    chat_log += f"{i}. {c['text']}\n"
                else:
                    chat_log += f"{i}. [sem texto no chunk]\n"

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                chat_log += f"Usuário: {content}\n"
            else:
                chat_log += f"Assistente: {content}\n"

        chat_log += "Assistente:"
        return chat_log
