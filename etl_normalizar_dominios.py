# etl_normalizar_dominios.py

import pandas as pd
import numpy as np

COLS_DOMINIOS = [
    "dom_cognitivo",
    "dom_psicologico",
    "dom_sensorial",
    "dom_locomotor",
    "dom_vitalidade",
]

def run_etl_normalizar_dominios(
    conn,
    tabela_origem: str = "tbl_dominios_individuo",
    tabela_destino: str = "tbl_dominios_individuo_norm_v2",
) -> int:
    """
    Normaliza (0–1) os domínios da própria tabela_origem usando MIN/MAX
    calculados na própria tabela_origem.
    Salva/atualiza em tabela_destino.
    Retorna quantidade de registros normalizados.
    """
    # 1) Lê origem
    df = conn.query(
        f"SELECT id, {','.join(COLS_DOMINIOS)} FROM public.{tabela_origem}",
        ttl=0
    )

    if df.empty:
        return 0

    # 2) Calcula min/max da própria origem
    mins = df[COLS_DOMINIOS].min(skipna=True)
    maxs = df[COLS_DOMINIOS].max(skipna=True)

    # 3) Normaliza
    df_out = df.copy()

    for col in COLS_DOMINIOS:
        denom = maxs[col] - mins[col]

        if pd.isna(denom) or denom == 0:
            df_out[col] = np.nan
        else:
            df_out[col] = (df[col] - mins[col]) / denom

    # opcional
    for col in COLS_DOMINIOS:
        df_out[col] = df_out[col].clip(0, 1)

    # 4) Grava destino no Postgres / Neon
    with conn.session as session:
        df_out.to_sql(
            tabela_destino,
            con=session.bind,
            schema="public",
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=1000,
        )
        session.commit()

    return len(df_out)
