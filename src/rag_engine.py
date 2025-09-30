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

logger = logging.getLogger(__name__)


class PayrollRAGEngine:
    """Motor RAG para consultas de folha de pagamento."""
    
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
        self.chunks = self._create_chunks()
        self.embeddings = self._create_embeddings()
        
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
                df[col] = df[col].str.strip() if df[col].dtype == 'object' else df[col]
            
            # Converter datas com formato ISO8601
            df['payment_date'] = pd.to_datetime(df['payment_date'], format='ISO8601', errors='coerce')
            
            # Se falhar, tentar formato específico
            if df['payment_date'].isna().any():
                df['payment_date'] = pd.to_datetime(df['payment_date'], errors='coerce')
            
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
            # Criar embedding da query
            query_embedding = self.model.encode([query])
            
            # Calcular similaridades
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Pegar top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Montar resultados
            results = []
            for idx in top_indices:
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(similarities[idx])
                results.append(chunk)
            
            logger.info(f"Query: '{query}' -> {len(results)} resultados (scores: {[r['score'] for r in results]})")
            
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
            filtered_df = filtered_df[filtered_df['employee_id'] == employee_id]
        
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