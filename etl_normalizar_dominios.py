# etl_normalizar_dominios.py
import sqlite3
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
    con = sqlite3.connect(db_path)
    try:
        # 1) Lê origem
        df = pd.read_sql_query(
            f"SELECT id, {','.join(COLS_DOMINIOS)} FROM {tabela_origem}",
            con
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

        # 4) Cria destino + UPSERT
        con.execute(f"""
        CREATE TABLE IF NOT EXISTS {tabela_destino} (
            id INTEGER PRIMARY KEY,
            dom_cognitivo REAL,
            dom_psicologico REAL,
            dom_sensorial REAL,
            dom_locomotor REAL,
            dom_vitalidade REAL
        )
        """)

        cols = list(df_out.columns)
        placeholders = ",".join(["?"] * len(cols))
        colnames = ",".join(cols)
        update_set = ",".join([f"{c}=excluded.{c}" for c in cols if c != "id"])

        sql_upsert = f"""
        INSERT INTO {tabela_destino} ({colnames})
        VALUES ({placeholders})
        ON CONFLICT(id) DO UPDATE SET
        {update_set}
        """

        rows = df_out.where(pd.notna(df_out), None).values.tolist()
        con.executemany(sql_upsert, rows)
        con.commit()

        return len(df_out)

    finally:
        con.close()
