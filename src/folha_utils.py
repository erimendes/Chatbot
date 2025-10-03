import pandas as pd
import re

def interpretar_comando(texto: str):
    """
    Interpreta comando do tipo "reajuste os salários de 2027 em 8%".
    
    Retorna:
        ano (int): o ano do reajuste
        indice_reajuste (float): índice de reajuste (ex: 1.08 para 8%)
    
    Lança ValueError se não encontrar ano ou percentual no texto.
    """
    # Procurar o ano (4 dígitos a partir de 2000)
    ano_match = re.search(r'\b(20\d{2})\b', texto)
    if not ano_match:
        raise ValueError("Ano não encontrado no comando.")
    ano = int(ano_match.group(1))
    
    # Procurar o percentual (número seguido de %)
    perc_match = re.search(r'(\d+(\.\d+)?)\s*%', texto)
    if not perc_match:
        raise ValueError("Percentual não encontrado no comando.")
    percentual = float(perc_match.group(1))
    
    # Converte percentual para índice (ex: 8% -> 1.08)
    indice = 1 + percentual / 100
    
    return ano, indice


def adicionar_ano_com_reajuste(
    df: pd.DataFrame,
    ano_destino: int,
    indice_reajuste: float,
    ano_base: int = None
) -> pd.DataFrame:
    """
    Adiciona à tabela de salários os valores reajustados para um novo ano.

    Args:
        df (pd.DataFrame): DataFrame original com colunas da folha.
        ano_destino (int): Ano para o qual os dados devem ser gerados.
        indice_reajuste (float): Reajuste percentual (ex: 1.05 = 5% de aumento).
        ano_base (int, opcional): Ano de referência para os dados. Se None, usa ano_destino - 1.

    Returns:
        pd.DataFrame: DataFrame com os dados do novo ano adicionados.
    """
    ano_base = ano_base or (ano_destino - 1)
    
    # Filtra apenas os dados do ano base (ex: "2025-01", "2025-02", etc)
    df_base = df[df['competency'].str.startswith(str(ano_base))].copy()
    
    if df_base.empty:
        raise ValueError(f"Não há dados para o ano base {ano_base}.")
    
    novos_registros = []

    for _, row in df_base.iterrows():
        ano, mes = row['competency'].split('-')
        nova_competencia = f"{ano_destino}-{mes}"
        nova_data_pagamento = f"{ano_destino}-{mes}-28"  # exemplo fixo para dia 28

        # Ajustar valores com índice de reajuste
        base = round(row['base_salary'] * indice_reajuste, 2)
        bonus = round(row['bonus'] * indice_reajuste, 2)
        vt_vr = round(row['benefits_vt_vr'] * indice_reajuste, 2)
        outros_proventos = row['other_earnings']  # sem reajuste, pode ajustar se quiser
        inss = round(row['deductions_inss'] * indice_reajuste, 2)
        irrf = round(row['deductions_irrf'] * indice_reajuste, 2)
        outros_desc = row['other_deductions']  # sem reajuste, pode ajustar

        net = base + bonus + vt_vr + outros_proventos - inss - irrf - outros_desc

        novos_registros.append({
            'employee_id': row['employee_id'],
            'name': row['name'],
            'competency': nova_competencia,
            'base_salary': base,
            'bonus': bonus,
            'benefits_vt_vr': vt_vr,
            'other_earnings': outros_proventos,
            'deductions_inss': inss,
            'deductions_irrf': irrf,
            'other_deductions': outros_desc,
            'net_pay': round(net, 2),
            'payment_date': nova_data_pagamento
        })

    df_novo_ano = pd.DataFrame(novos_registros)

    # Evitar duplicação: verificar se já existe dados para o ano_destino
    if not df[df['competency'].str.startswith(str(ano_destino))].empty:
        raise ValueError(f"Já existem dados para o ano {ano_destino}. Evitando duplicação.")

    # Concatenar original + novos dados e retornar
    df_final = pd.concat([df, df_novo_ano], ignore_index=True)
    return df_final
