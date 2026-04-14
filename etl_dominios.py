# etl_dominios.py
import warnings
warnings.filterwarnings('ignore')

import sqlite3
import pandas as pd
import numpy as np


def run_etl_dominios(
    conn,
    tabela_origem: str = "tbl_indicadores_individuo",
    tabela_destino: str = "tbl_dominios_individuo",
    normalizar_continuas: bool = False,
    cols_continuas: list[str] | None = None,
) -> int:
    """
    Lê `tabela_origem` (indicadores), calcula domínios ponderados (CFA) e faz UPSERT em `tabela_destino`.
    Retorna quantidade de registros processados (len(df_out)).
    """

    if cols_continuas is None:
        cols_continuas = ["gait_speed", "imc"]

    # =========================
    # PESOS (CFA)
    # =========================
    pesos_cfa = {
        "dom_cognitivo": {
            "temporal_orientation": 0.85,
            "memory_recall": 0.86,
            "semantic_memory": 0.85,
            "verbal_fluency_category": 0.80,
        },
        "dom_psicologico": {
            "depression_invertida": 0.73,
            "sleep_quality_invertida": 0.76,
        },
        "dom_sensorial": {
            "hearing_deficit_invertida": 0.75,
            "distance_vision_invertida": 0.82,
            "near_vision_invertida": 0.83,
        },
        "dom_locomotor": {
            "gait_speed": 0.73,
            "balance_test": 0.77,
        },
        "dom_vitalidade": {
            "grip_strength_category": 0.76,
            "weight_loss_invertida": 0.85,
            "self_report_exhaustion_invertida": 0.91,
            "poor_endurance_invertida": 0.90,
        },
    }

    # =========================
    # FUNÇÕES AUXILIARES
    # =========================
    def zscore_serie(s: pd.Series) -> pd.Series:
        """Z-score ignorando NaN (mantém NaN)."""
        mu = s.mean(skipna=True)
        sd = s.std(skipna=True)
        if pd.isna(sd) or sd == 0:
            return s * np.nan
        return (s - mu) / sd

    # =========================
    # (1) LER ORIGEM
    # =========================
    con = sqlite3.connect(db_path, timeout=30)
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA busy_timeout = 30000;")
        df = pd.read_sql_query(f"SELECT * FROM {tabela_origem}", con)

        # Se não tiver nada, retorna 0 sem quebrar o app
        if df.empty:
            return 0

        # =========================
        # (2) NORMALIZAÇÃO OPCIONAL
        # =========================
        if normalizar_continuas:
            for c in cols_continuas:
                if c in df.columns:
                    df[c] = zscore_serie(df[c].astype(float))

        # =========================
        # (3) APLICAR PESOS
        # =========================
        for dominio, variaveis in pesos_cfa.items():
            for variavel, peso in variaveis.items():
                if variavel in df.columns:
                    df[f"{variavel}_ponderada"] = df[variavel] * peso
                else:
                    df[f"{variavel}_ponderada"] = np.nan

        # =========================
        # (4) CALCULAR DOMÍNIOS
        # =========================
        df["dom_cognitivo"] = df[
            [
                "verbal_fluency_category_ponderada",
                "memory_recall_ponderada",
                "semantic_memory_ponderada",
                "temporal_orientation_ponderada",
            ]
        ].mean(axis=1)

        df["dom_psicologico"] = df[
            [
                "depression_invertida_ponderada",
                "sleep_quality_invertida_ponderada",
            ]
        ].mean(axis=1)

        df["dom_sensorial"] = df[
            [
                "hearing_deficit_invertida_ponderada",
                "distance_vision_invertida_ponderada",
                "near_vision_invertida_ponderada",
            ]
        ].mean(axis=1)

        df["dom_locomotor"] = df[
            [
                "gait_speed_ponderada",
                "balance_test_ponderada",
            ]
        ].mean(axis=1)

        df["dom_vitalidade"] = df[
            [
                "grip_strength_category_ponderada",
                "weight_loss_invertida_ponderada",
                "self_report_exhaustion_invertida_ponderada",
                "poor_endurance_invertida_ponderada",
            ]
        ].mean(axis=1)

        df["ci_geral"] = df[
            ["dom_cognitivo", "dom_psicologico", "dom_sensorial", "dom_locomotor", "dom_vitalidade"]
        ].mean(axis=1)

        # =========================
        # (5) DF FINAL
        # =========================
        df_out = df[[
            "id",
            "dom_cognitivo",
            "dom_psicologico",
            "dom_sensorial",
            "dom_locomotor",
            "dom_vitalidade",
            "ci_geral",
        ]].copy()

        # =========================
        # (6) CRIAR DESTINO + UPSERT
        # =========================
        con.execute(f"""
        CREATE TABLE IF NOT EXISTS {tabela_destino} (
            id INTEGER PRIMARY KEY,
            dom_cognitivo REAL,
            dom_psicologico REAL,
            dom_sensorial REAL,
            dom_locomotor REAL,
            dom_vitalidade REAL,
            ci_geral REAL
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
