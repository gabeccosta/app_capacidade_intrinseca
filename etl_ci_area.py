# etl_ci_area.py
import warnings
warnings.filterwarnings('ignore')

import sqlite3
import pandas as pd
import numpy as np


def run_etl_ci_area(
    db_path: str,
    tabela_origem: str = "tbl_dominios_individuo_norm_v2",   # <- AGORA É A NORMALIZADA
    tabela_destino: str = "tbl_ci_area_individuo",
    cols_dominios: list[str] | None = None,
) -> int:
    """
    Lê domínios normalizados (0–1) em `tabela_origem`, calcula ci_area_pentagono
    e faz UPSERT em `tabela_destino`.
    Retorna quantidade de registros processados.
    """
    if cols_dominios is None:
        cols_dominios = [
            "dom_cognitivo",
            "dom_psicologico",
            "dom_sensorial",
            "dom_locomotor",
            "dom_vitalidade",
        ]

    # =========================
    # FUNÇÃO – ÁREA DO PENTÁGONO (Shoelace)
    # =========================
    def calcular_area_pentagono(valores):
        valores = np.asarray(valores, dtype=float)

        # Se algum domínio for NaN, não calcula (igual sua regra do notebook)
        if np.isnan(valores).any():
            return np.nan

        num_vars = len(valores)

        # Ângulos igualmente espaçados
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

        # Coordenadas cartesianas
        x = valores * np.cos(angles)
        y = valores * np.sin(angles)

        # Fechar o polígono
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        # Área (Shoelace)
        area = 0.5 * np.abs(
            np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
        )

        return float(area)

    # =========================
    # (1) LER ORIGEM (NORMALIZADA)
    # =========================
    con = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            f"SELECT id, {','.join(cols_dominios)} FROM {tabela_origem}",
            con
        )

        # origem vazia => não quebra o app
        if df.empty:
            return 0

        # Garantir colunas (por segurança)
        for c in cols_dominios:
            if c not in df.columns:
                df[c] = np.nan

        # =========================
        # (2) CALCULAR ÁREA
        # =========================
        df["ci_area_pentagono"] = df.apply(
            lambda row: calcular_area_pentagono([row[c] for c in cols_dominios]),
            axis=1
        )

        # =========================
        # (3) DF FINAL
        # =========================
        df_out = df[["id", "ci_area_pentagono"]].copy()

        # =========================
        # (4) CRIAR TABELA + UPSERT
        # =========================
        con.execute(f"""
        CREATE TABLE IF NOT EXISTS {tabela_destino} (
            id INTEGER PRIMARY KEY,
            ci_area_pentagono REAL
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