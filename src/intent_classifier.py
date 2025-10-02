# src/intent_classifier.py
"""
Classificador de intenções para roteamento de queries.
"""
import logging
import re
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class IntentClassifier:
    """Classifica a intenção do usuário para roteamento adequado."""

    INTENT_PAYROLL = "payroll_query"
    INTENT_GENERAL = "general_chat"
    INTENT_STATS = "statistics"
    INTENT_HELP = "help"

    def __init__(self):
        """Inicializa o classificador."""
        # Padrões para identificar consultas de folha de pagamento (mais específicos)
        self.payroll_patterns = [
            # Nomes específicos dos funcionários
            r'\b(ana\s+souza|bruno\s+lima)\b',
            # Termos financeiros específicos
            r'\b(salário|salario|pagamento|líquido|liquido|bruto)\b',
            r'\b(bônus|bonus|adicional)\b',
            r'\b(inss|irrf|desconto|imposto|renda)\b',
            r'\b(benefício|beneficio|vt|vr|vale)\b',
            # Períodos específicos
            r'\b(janeiro|fevereiro|março|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\b',
            r'\b(2025-\d{2})\b',  # competência
            r'\b(competência|competencia)\b',
            # IDs de funcionários
            r'\b(E00\d)\b',
            # Perguntas específicas sobre folha
            r'\b(quanto\s+(recebeu|ganhou|foi\s+pago))\b',
            r'\b(qual\s+(o\s+)?(salário|pagamento))\b',
        ]

        # Padrões para estatísticas
        self.stats_patterns = [
            r'\b(estatística|estatistica|média|media|total|soma)\b',
            r'\b(quantos|quantas|quanto)\s+(funcionários|funcionarios|registros)\b',
            r'\b(todos|todas|geral)\s+(os\s+)?(funcionários|funcionarios|pagamentos)\b',
            r'\b(resumo|sumário|sumario)\b',
        ]

        # Padrões para ajuda
        self.help_patterns = [
            r'\b(ajuda|help|como\s+(usar|funciona))\b',
            r'\b(o\s+que\s+(você|voce)\s+faz)\b',
            r'\b(pode\s+fazer|consegue\s+fazer)\b',
            r'\b(quem\s+(é|eh)\s+(você|voce))\b',
        ]

        # Padrões para conversa geral (não relacionada a folha)
        self.general_chat_patterns = [
            r'\b(olá|oi|bom\s+dia|boa\s+tarde|boa\s+noite)\b',
            r'\b(tudo\s+bem|como\s+(você|voce)\s+está|como\s+vai)\b',
            r'\b(obrigado|obrigada|valeu|tchau|até\s+logo)\b',
            r'\b(tempo|clima|chuva|sol|frio|calor)\b',
            r'\b(futebol|jogo|time|esporte)\b',
            r'\b(comida|restaurante|receita|culinária|culinaria)\b',
            r'\b(filme|música|musica|livro|série|serie)\b',
            r'\b(trabalho|empresa|carreira|profissão|profissao)\b',
        ]

    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classifica a intenção da query.

        Args:
            query: Pergunta do usuário

        Returns:
            Tupla (intent, confidence) onde intent é uma das constantes INTENT_*
            e confidence é um score de 0 a 1
        """
        query_lower = query.lower()

        # Verificar padrões de conversa geral primeiro
        general_matches = sum(
            1 for pattern in self.general_chat_patterns
            if re.search(pattern, query_lower)
        )

        # Verificar padrões de ajuda
        help_matches = sum(
            1 for pattern in self.help_patterns
            if re.search(pattern, query_lower)
        )

        # Verificar padrões de estatísticas
        stats_matches = sum(
            1 for pattern in self.stats_patterns
            if re.search(pattern, query_lower)
        )

        # Verificar padrões de folha de pagamento
        payroll_matches = sum(
            1 for pattern in self.payroll_patterns
            if re.search(pattern, query_lower)
        )

        # Decidir intenção baseado em matches e prioridades
        if help_matches > 0:
            confidence = min(help_matches / len(self.help_patterns), 1.0)
            logger.debug(f"Intent: HELP (confidence={confidence:.2f})")
            return self.INTENT_HELP, confidence

        # Se tem muitos matches de conversa geral e poucos de folha, é conversa geral
        if general_matches > 0 and payroll_matches <= 1:
            confidence = min(general_matches /
                             len(self.general_chat_patterns), 1.0)
            logger.debug(f"Intent: GENERAL (confidence={confidence:.2f})")
            return self.INTENT_GENERAL, confidence

        # Se tem matches de folha de pagamento, priorizar isso
        if payroll_matches > 0:
            confidence = min(payroll_matches / len(self.payroll_patterns), 1.0)
            logger.debug(f"Intent: PAYROLL (confidence={confidence:.2f})")
            return self.INTENT_PAYROLL, confidence

        # Se tem matches de estatísticas
        if stats_matches > 0:
            confidence = min(stats_matches / len(self.stats_patterns), 1.0)
            logger.debug(f"Intent: STATS (confidence={confidence:.2f})")
            return self.INTENT_STATS, confidence

        # Default: chat geral
        logger.debug("Intent: GENERAL (default)")
        return self.INTENT_GENERAL, 0.5

    def extract_filters(self, query: str) -> Dict[str, Any]:
        """
        Extrai filtros estruturados da query.
        Útil para consultas diretas ao dataset.
        """
        filters = {}

        # Extrair nomes
        name_pattern = r'\b(ana|bruno)\b'
        name_match = re.search(name_pattern, query.lower())
        if name_match:
            filters['name'] = name_match.group(1).title()

        # Extrair competência
        comp_pattern = r'(2025-\d{2})'
        comp_match = re.search(comp_pattern, query)
        if comp_match:
            filters['competency'] = comp_match.group(1)

        # Extrair meses por nome
        month_mapping = {
            'janeiro': '2025-01', 'fevereiro': '2025-02', 'março': '2025-03',
            'marco': '2025-03', 'abril': '2025-04', 'maio': '2025-05',
            'junho': '2025-06'
        }
        for month_name, comp in month_mapping.items():
            if month_name in query.lower():
                filters['competency'] = comp
                break

        # Extrair employee_id
        emp_pattern = r'\b(E00\d)\b'
        emp_match = re.search(emp_pattern, query)
        if emp_match:
            filters['employee_id'] = emp_match.group(1)

        if filters:
            logger.debug(f"Filtros extraídos: {filters}")

        return filters
