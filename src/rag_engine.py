# src/rag_engine.py
"""
Motor RAG para consultas de folha de pagamento.
Implementa embeddings, chunking, ranking e recuperação de informações.
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime
import hashlib
import pickle
import pathlib

logger = logging.getLogger(__name__)


class PayrollRAGEngine:
    """Motor RAG para consultas de folha de pagamento."""

    def _get_csv_hash(self) -> str:
        """Gera um hash SHA256 do conteúdo do CSV para controle de cache."""
        with open(self.csv_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        return file_hash

    def __init__(self, csv_path: str, model_name: str = None):
        """
        Inicializa o motor RAG.

        Args:
            csv_path: Caminho para o arquivo CSV de folha de pagamento
            model_name: Nome do modelo de embeddings a usar
        """
        self.csv_path = csv_path
        self.model_name = model_name or os.getenv(
            'EMBEDDING_MODEL',
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        )

        logger.info(f"Inicializando RAG Engine com modelo: {self.model_name}")

        # Carregar modelo de embeddings
        self.model = SentenceTransformer(self.model_name)

        # Carregar e processar dataset
        self.df = self._load_and_validate_data()

        # Criar chunks e embeddings
        # self.chunks = self._create_chunks()
        # self.embeddings = self._create_embeddings()
        self.cache_dir = pathlib.Path(".rag_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.csv_hash = self._get_csv_hash()

        if self._load_cache():
            logger.info("Cache carregado com sucesso.")
        else:
            logger.info("Cache não encontrado ou desatualizado. Gerando novamente.")
            self.chunks = self._create_chunks()
            self.embeddings = self._create_embeddings()
            self._save_cache()


        logger.info(f"RAG Engine inicializado com {len(self.chunks)} chunks")

    def _load_and_validate_data(self) -> pd.DataFrame:
        """Carrega e valida o dataset de folha de pagamento."""
        try:
            df = pd.read_csv(self.csv_path)

            # Validar colunas esperadas
            expected_cols = [
                'employee_id', 'name', 'competency', 'base_salary', 'bonus',
                'benefits_vt_vr', 'other_earnings', 'deductions_inss',
                'deductions_irrf', 'other_deductions', 'net_pay', 'payment_date'
            ]

            missing_cols = set(expected_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Colunas faltando no CSV: {missing_cols}")

            # Limpar espaços em branco das colunas de texto
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].str.strip(
                ) if df[col].dtype == 'object' else df[col]

            # Converter datas com formato ISO8601
            df['payment_date'] = pd.to_datetime(
                df['payment_date'], format='ISO8601', errors='coerce')

            # Se falhar, tentar formato específico
            if df['payment_date'].isna().any():
                df['payment_date'] = pd.to_datetime(
                    df['payment_date'], errors='coerce')

            # Verificar se há datas inválidas
            if df['payment_date'].isna().any():
                logger.warning("Algumas datas não puderam ser convertidas")

            logger.info(f"Dataset carregado: {len(df)} registros")
            return df

        except Exception as e:
            logger.error(f"Erro ao carregar dataset: {e}")
            raise

    def _create_chunks(self) -> List[Dict[str, Any]]:
        """
        Cria chunks de texto a partir do dataset.
        Cada linha vira um chunk com informações estruturadas.
        """
        chunks = []

        for idx, row in self.df.iterrows():
            # Criar texto descritivo para cada linha
            text = self._row_to_text(row)

            chunk = {
                'id': f"chunk_{idx}",
                'text': text,
                'metadata': row.to_dict(),
                'row_index': idx
            }
            chunks.append(chunk)

        return chunks

    def _row_to_text(self, row: pd.Series) -> str:
        """Converte uma linha do dataset em texto descritivo."""
        # Formatar dinheiro
        def fmt_money(val):
            return f"R$ {float(val):,.2f}".replace(',', '_').replace('.', ',').replace('_', '.')

        # Extrair ano e mês
        comp_parts = row['competency'].split('-')
        year, month = comp_parts[0], comp_parts[1]
        month_names = {
            '01': 'janeiro', '02': 'fevereiro', '03': 'março', '04': 'abril',
            '05': 'maio', '06': 'junho', '07': 'julho', '08': 'agosto',
            '09': 'setembro', '10': 'outubro', '11': 'novembro', '12': 'dezembro'
        }
        month_name = month_names.get(month, month)

        text = f"""
        Funcionário: {row['name']} (ID: {row['employee_id']})
        Competência: {month_name} de {year} ({row['competency']})
        Salário Base: {fmt_money(row['base_salary'])}
        Bônus: {fmt_money(row['bonus'])}
        Benefícios (VT/VR): {fmt_money(row['benefits_vt_vr'])}
        Outros Proventos: {fmt_money(row['other_earnings'])}
        Desconto INSS: {fmt_money(row['deductions_inss'])}
        Desconto IRRF: {fmt_money(row['deductions_irrf'])}
        Outros Descontos: {fmt_money(row['other_deductions'])}
        Pagamento Líquido: {fmt_money(row['net_pay'])}
        Data de Pagamento: {row['payment_date'].strftime('%d/%m/%Y')}
        """.strip()

        return text

    def _create_embeddings(self) -> np.ndarray:
        """Cria embeddings para todos os chunks."""
        texts = [chunk['text'] for chunk in self.chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings

    def _sanitize_query(self, query: str) -> str:
        """
        Sanitiza a query do usuário para prevenir ataques básicos.
        """
        # Remover comandos SQL básicos
        dangerous_patterns = [
            r'drop\s+table',
            r'delete\s+from',
            r'insert\s+into',
            r'update\s+set',
            r'<script>',
            r'javascript:',
        ]

        query_lower = query.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, query_lower):
                logger.warning(f"Potencial ataque detectado na query: {query}")
                return ""

        return query.strip()

    def _extract_target_month(self, query: str) -> str:
        """Extrai o mês específico mencionado na pergunta."""
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

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Busca chunks relevantes para a query usando similaridade coseno.

        Args:
            query: Pergunta do usuário
            top_k: Número de resultados a retornar

        Returns:
            Lista de chunks ordenados por relevância
        """
        if not query or len(query.strip()) == 0:
            logger.warning("Query vazia recebida")
            return []

        # Sanitizar query
        query = self._sanitize_query(query)
        if not query:
            return []

        top_k = top_k or int(os.getenv('TOP_K_RESULTS', 3))

        try:
            # Extrair mês específico da query
            target_month = self._extract_target_month(query)

            # Extrair funcionário específico da query
            target_employee = self._extract_target_employee(query)

            # Criar embedding da query
            query_embedding = self.model.encode([query])

            # Calcular similaridades
            similarities = cosine_similarity(
                query_embedding, self.embeddings)[0]

            # Aplicar boosts para resultados específicos
            for idx, chunk in enumerate(self.chunks):
                boost = 0.0

                # Boost para mês específico
                if target_month and target_month in chunk['metadata']['competency']:
                    boost += 0.1

                # Boost para funcionário específico
                if target_employee and target_employee.lower() in chunk['metadata']['name'].lower():
                    boost += 0.15  # Boost maior para funcionário específico

                similarities[idx] += boost

            # Pegar top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            # Montar resultados
            results = []
            for idx in top_indices:
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(similarities[idx])
                results.append(chunk)

            logger.info(
                f"Query: '{query}' -> {len(results)} resultados (scores: {[r['score'] for r in results]})")

            return results

        except Exception as e:
            logger.error(f"Erro na busca: {e}")
            return []

    def filter_by_criteria(
        self,
        name: str = None,
        competency: str = None,
        employee_id: str = None,
        min_net_pay: float = None,
        max_net_pay: float = None
    ) -> pd.DataFrame:
        """
        Filtra o dataset por critérios específicos.
        Útil para consultas estruturadas.
        """
        filtered_df = self.df.copy()

        if name:
            filtered_df = filtered_df[
                filtered_df['name'].str.contains(name, case=False, na=False)
            ]

        if competency:
            filtered_df = filtered_df[filtered_df['competency'] == competency]

        if employee_id:
            filtered_df = filtered_df[filtered_df['employee_id']
                                      == employee_id]

        if min_net_pay is not None:
            filtered_df = filtered_df[filtered_df['net_pay'] >= min_net_pay]

        if max_net_pay is not None:
            filtered_df = filtered_df[filtered_df['net_pay'] <= max_net_pay]

        return filtered_df

    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do dataset."""
        return {
            'total_records': len(self.df),
            'unique_employees': self.df['employee_id'].nunique(),
            'competencies': sorted(self.df['competency'].unique().tolist()),
            'avg_net_pay': float(self.df['net_pay'].mean()),
            'max_net_pay': float(self.df['net_pay'].max()),
            'min_net_pay': float(self.df['net_pay'].min()),
            'total_paid': float(self.df['net_pay'].sum()),
        }

    # 👇 Adicione aqui dentro da classe
    def gerar_relatorio_trimestral(self, nome: str, ano: str, trimestre: int) -> str:
        df_filtrado = self._filtrar_por_trimestre(nome, ano, trimestre)

        nomes_meses = {
            '01': 'janeiro', '02': 'fevereiro', '03': 'março',
            '04': 'abril', '05': 'maio', '06': 'junho',
            '07': 'julho', '08': 'agosto', '09': 'setembro',
            '10': 'outubro', '11': 'novembro', '12': 'dezembro'
        }

        detalhes = ""
        total = 0.0

        for _, row in df_filtrado.iterrows():
            comp = row['competency']
            _, mes = comp.split('-')
            net = row['net_pay']
            total += net
            mes_nome = nomes_meses.get(mes, mes)
            detalhes += f"\n- {mes_nome}: R$ {net:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")

        total_fmt = f"R$ {total:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")

        prompt = (
            f"Com base nos dados da folha de pagamento, os valores líquidos recebidos por {nome} no {trimestre}º trimestre de {ano} foram:"
            f"{detalhes}\n\n"
            f"O total líquido no período foi de {total_fmt}. "
            "Explique isso de forma clara e natural para o usuário."
        )

        return prompt

    def _filtrar_por_trimestre(self, nome: str, ano: str, trimestre: int) -> pd.DataFrame:
        meses = range((trimestre - 1) * 3 + 1, trimestre * 3 + 1)
        meses_str = [f"{m:02d}" for m in meses]
        competencias = [f"{ano}-{m}" for m in meses_str]

        return self.df[
            (self.df['name'].str.contains(nome, case=False, na=False)) &
            (self.df['competency'].isin(competencias))
        ]

    def _cache_paths(self) -> Tuple[pathlib.Path, pathlib.Path]:
        """Retorna os caminhos dos arquivos de cache."""
        chunks_path = self.cache_dir / f"chunks_{self.csv_hash}.pkl"
        embeddings_path = self.cache_dir / f"embeddings_{self.csv_hash}.npy"
        return chunks_path, embeddings_path

    def _save_cache(self):
        """Salva os chunks e embeddings em disco."""
        chunks_path, embeddings_path = self._cache_paths()
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        np.save(embeddings_path, self.embeddings)
        logger.info("Cache salvo com sucesso.")

    def _load_cache(self) -> bool:
        """Tenta carregar chunks e embeddings do cache."""
        chunks_path, embeddings_path = self._cache_paths()
        if chunks_path.exists() and embeddings_path.exists():
            try:
                with open(chunks_path, "rb") as f:
                    self.chunks = pickle.load(f)
                self.embeddings = np.load(embeddings_path)
                return True
            except Exception as e:
                logger.warning(f"Erro ao carregar cache: {e}")
                return False
        return False
