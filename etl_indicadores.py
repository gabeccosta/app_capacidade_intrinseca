# etl_indicadores.py
import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import numpy as np
from sqlalchemy import text


def run_etl(
    conn,
    tabela_origem: str = "tbl_dados_individuo",
    tabela_destino: str = "tbl_indicadores_individuo",
) -> int:
    """
    Lê dados de `tabela_origem`, calcula indicadores e faz UPSERT em `tabela_destino`.
    Retorna a quantidade de registros processados (len(df_out)).
    """

    # =========================
    # FUNÇÕES
    # =========================
    def soma_nan(row, lista_colunas):
        if row[lista_colunas].isna().all():
            return np.nan
        else:
            return row[lista_colunas].sum()

    def convert_to_timedelta(minutos, segundos, centesimos):
        total_seconds = minutos * 60 + segundos + centesimos / 100
        return pd.to_timedelta(total_seconds, unit='s')

    def balance_test_score(value, test):
        if value == 9888:
            return np.nan
        elif value in [9666, 9999]:
            return 0
        else:
            if test in ['mf30', 'mf31']:
                return 1 if value >= 10 else 0
            if test == 'mf32':
                if value >= 10:
                    return 2
                elif value >= 3:
                    return 1
                else:
                    return 0

    def maior_valor_nan(row, lista_colunas):
        valores_validos = row[lista_colunas].dropna()
        if valores_validos.empty:
            return np.nan
        return valores_validos.max()

    # Ranges de fluência verbal por faixa etária (Tombaugh)
    fluency_ranges = {
        '50-59': {'mean': 20.1, 'sd': 4.9, 'low': 20.1 - (2 * 4.9), 'high': 20.1 + (2 * 4.9)},
        '60-69': {'mean': 17.6, 'sd': 4.7, 'low': 17.6 - (2 * 4.7), 'high': 17.6 + (2 * 4.7)},
        '70-79': {'mean': 16.1, 'sd': 4.0, 'low': 16.1 - (2 * 4.0), 'high': 16.1 + (2 * 4.0)},
        '80-89': {'mean': 14.3, 'sd': 3.9, 'low': 14.3 - (2 * 3.9), 'high': 14.3 + (2 * 3.9)},
        '> 90': {'mean': 13.0, 'sd': 3.8, 'low': 13.0 - (2 * 3.8), 'high': 13.0 + (2 * 3.8)},
    }

    def categorize_fluency(row):
        if pd.isna(row.get('verbal_fluency')):
            return np.nan

        age_group = None
        for key in fluency_ranges.keys():
            if key == '> 90':
                if row['idade'] > 90:
                    age_group = key
                    break
            else:
                age_min, age_max = map(int, key.split('-'))
                if age_min <= row['idade'] <= age_max:
                    age_group = key
                    break

        if not age_group:
            return np.nan

        low = fluency_ranges[age_group]['low']
        high = fluency_ranges[age_group]['high']
        vf = row['verbal_fluency']

        if vf < low:
            return 0
        elif low <= vf <= high:
            return 1
        else:
            return 2

    # Ranges de Grip Strength (Amaral 2023) – já com low/high
    grip_strength_ranges = {
        'Masculino': {
            '50-59': {'mean': 41.2, 'sd': 8.65, 'low': 41.2 - (2 * 8.65), 'high': 41.2 + (2 * 8.65)},
            '60-69': {'mean': 36.2, 'sd': 8.15, 'low': 36.2 - (2 * 8.15), 'high': 36.2 + (2 * 8.15)},
            '70-79': {'mean': 31.3, 'sd': 6.97, 'low': 31.3 - (2 * 6.97), 'high': 31.3 + (2 * 6.97)},
            '80 e acima': {'mean': 25.7, 'sd': 5.81, 'low': 25.7 - (2 * 5.81), 'high': 25.7 + (2 * 5.81)},
        },
        'Feminino': {
            '50-59': {'mean': 24.2, 'sd': 6.06, 'low': 24.2 - (2 * 6.06), 'high': 24.2 + (2 * 6.06)},
            '60-69': {'mean': 23.0, 'sd': 5.55, 'low': 23.0 - (2 * 5.55), 'high': 23.0 + (2 * 5.55)},
            '70-79': {'mean': 20.3, 'sd': 5.05, 'low': 20.3 - (2 * 5.05), 'high': 20.3 + (2 * 5.05)},
            '80 e acima': {'mean': 17.1, 'sd': 4.98, 'low': 17.1 - (2 * 4.98), 'high': 17.1 + (2 * 4.98)},
        }
    }

    def categorize_grip_strength(row):
        if pd.isna(row.get('grip_strength')):
            return np.nan

        sexo = 'Masculino' if row['sexo'] == 0 else 'Feminino'
        idade = row['idade']

        if idade >= 80:
            faixa = '80 e acima'
        elif 70 <= idade <= 79:
            faixa = '70-79'
        elif 60 <= idade <= 69:
            faixa = '60-69'
        elif 50 <= idade <= 59:
            faixa = '50-59'
        else:
            return np.nan

        low = grip_strength_ranges[sexo][faixa]['low']
        high = grip_strength_ranges[sexo][faixa]['high']

        if row['grip_strength'] < low:
            return 0
        elif row['grip_strength'] <= high:
            return 1
        else:
            return 2

    # =========================
    # (1) LER ORIGEM DO SQLITE
    # =========================
    df = conn.query(f"SELECT * FROM public.{tabela_origem}", ttl=0)
    
        # =========================
        # (2) RENOMEAR COLUNAS
        # =========================
        renomear_colunas = {
            'e2': 'nacionalidade',
            'e7': 'situacao_conjugal',
            'e9': 'cor',
            'e22': 'escolaridade',
            'rendadom': 'renda_mensal_familiar',
            'rendadompc': 'renda_mensal_familiar_por_pessoa',
        }
        df.rename(columns=renomear_colunas, inplace=True)

        # =========================
        # (3) RIQUEZA
        # =========================
        mapeamento = {0: 0, '0': 0, 88: np.nan, 99: np.nan}

        cat_b4 = {'1': 5000,'2': 30000,'3': 75000,'4': 150000,'5': 200000,
                  1: 5000,2: 30000,3: 75000,4: 150000,5: 200000,'8': 0,8: 0,'9': 0,9: 0}
        cat_b6 = {'1': 25000,'2': 75000,'3': 150000,'4': 250000,'5': 350000,'6': 450000,'7': 55000,'8': 800000,'9': 1250000,'10': 1750000,'11': 2000000,
                  1: 25000,2: 75000,3: 150000,4: 250000,5: 350000,6: 450000,7: 55000,8: 800000,9: 1250000,10: 1750000,11: 2000000,'88': 0,88: 0,'99': 0,99: 0}
        cat_b8 = cat_b6.copy()
        cat_b37 = {
            '1': 5000,'2': 12500,'3': 17500,'4': 22500,'5': 27500,'6': 32500,'7': 37500,'8': 42500,'9': 47500,'10': 52500,'11': 57500,'12': 62500,'13': 67500,'14': 72500,'15': 77500,'16': 82500,'17': 87500,'18': 92500,'19': 97500,'20': 100000,
            1: 5000,2: 12500,3: 17500,4: 22500,5: 27500,6: 32500,7: 37500,8: 42500,9: 47500,10: 52500,11: 57500,12: 62500,13: 67500,14: 72500,15: 77500,16: 82500,17: 87500,18: 92500,19: 97500,20: 100000,
            '88': 0,88: 0,'99': 0,99: 0
        }

        for col in ['b4', 'b6', 'b8', 'b37']:
            if col not in df.columns:
                df[col] = np.nan

        df.replace({'b6': mapeamento, 'b4': mapeamento, 'b8': mapeamento, 'b37': mapeamento}, inplace=True)

        df['b4_monetario'] = df['b4'].replace(cat_b4)
        df['b6_monetario'] = df['b6'].replace(cat_b6)
        df['b8_monetario'] = df['b8'].replace(cat_b8)
        df['b37_monetario'] = df['b37'].replace(cat_b37)

        df[['b4_monetario','b6_monetario','b8_monetario','b37_monetario']] = df[['b4_monetario','b6_monetario','b8_monetario','b37_monetario']].fillna(0)
        df['riqueza'] = df['b6_monetario'] - df['b4_monetario'] + df['b8_monetario'] + df['b37_monetario']

        # =========================
        # (4) COGNITIVO
        # =========================
        for col in ['q7','q8','q9','q10']:
            if col not in df.columns:
                df[col] = np.nan

        df[['q7_tratado','q8_tratado','q9_tratado','q10_tratado']] = df[['q7','q8','q9','q10']].replace({8: np.nan, 9: 0, 10: 0})
        df['temporal_orientation'] = df[['q7_tratado','q8_tratado','q9_tratado','q10_tratado']].sum(axis=1, skipna=False)

        if 'q13' not in df.columns:
            df['q13'] = np.nan
        df['memory_recall'] = df['q13'].replace({88: np.nan})

        for col in ['q18','q19','q20','q21']:
            if col not in df.columns:
                df[col] = np.nan

        df['q18_tratado'] = df['q18'].replace({2: 0, 8: np.nan, 9: 0, 10: np.nan})
        df['q19_tratado'] = df['q19'].replace({2: 0, 8: np.nan, 9: 0, 10: np.nan})
        df['q20_tratado'] = df['q20'].replace({2: 0, 8: np.nan, 9: 0, 10: np.nan})
        df['q21_tratado'] = df['q21'].replace({2: 0, 8: np.nan, 9: 0, 10: np.nan})

        df['semantic_memory'] = df[['q18_tratado','q19_tratado','q20_tratado','q21_tratado']].sum(axis=1, skipna=False)

        if 'q14' not in df.columns:
            df['q14'] = np.nan
        df['verbal_fluency'] = df['q14'].replace({888: np.nan, 999: np.nan})
        df['verbal_fluency_category'] = df.apply(categorize_fluency, axis=1)

        # =========================
        # (5) PSICOLÓGICO
        # =========================
        colunas_depression = ['r2','r3','r4','r5','r6','r7','r8','r9']
        for col in colunas_depression:
            if col not in df.columns:
                df[col] = np.nan

        df[colunas_depression] = df[colunas_depression].replace({8: np.nan, 9: np.nan})
        df['depression'] = df.apply(soma_nan, axis=1, lista_colunas=colunas_depression)
        df['depression_invertida'] = df['depression'].max() - df['depression']

        # =========================
        # (6) SLEEP QUALITY
        # =========================
        for col in ['n74','n75']:
            if col not in df.columns:
                df[col] = np.nan

        df[['n74_tratado','n75_tratado']] = df[['n74','n75']].replace({9: np.nan})
        df['sleep_quality'] = df.apply(soma_nan, axis=1, lista_colunas=['n74_tratado','n75_tratado'])
        df['sleep_quality_invertida'] = (df['sleep_quality'].max() - df['sleep_quality']) + 1

        # =========================
        # (7) SENSORY
        # =========================
        for col in ['n16','n6','n7']:
            if col not in df.columns:
                df[col] = np.nan

        df['hearing_deficit'] = df['n16'].replace({9: np.nan})
        df['hearing_deficit_invertida'] = (df['hearing_deficit'].max() - df['hearing_deficit']) + 1

        df['distance_vision'] = df['n6'].replace({9: np.nan})
        df['distance_vision_invertida'] = (df['distance_vision'].max() - df['distance_vision']) + 1

        df['near_vision'] = df['n7'].replace({9: np.nan})
        df['near_vision_invertida'] = (df['near_vision'].max() - df['near_vision']) + 1

        # =========================
        # (8) LOCOMOTOR
        # =========================
        for col in ['mf33','mf34','mf35','mf36','mf37','mf38']:
            if col not in df.columns:
                df[col] = np.nan

        df[['mf33','mf34','mf35','mf36','mf37','mf38']] = df[['mf33','mf34','mf35','mf36','mf37','mf38']].replace(
            {9888: np.nan, 9666: np.nan, 8888: np.nan}
        )

        df['speed1'] = df.apply(lambda r: convert_to_timedelta(r['mf33'], r['mf34'], r['mf35']), axis=1)
        df['speed2'] = df.apply(lambda r: convert_to_timedelta(r['mf36'], r['mf37'], r['mf38']), axis=1)

        df['speed'] = (df['speed1'] + df['speed2']) / 2
        df['speed'] = df['speed'].dt.total_seconds()
        df['gait_speed'] = round(3 / df['speed'], 2)

        for col in ['mf30','mf31','mf32']:
            if col not in df.columns:
                df[col] = np.nan

        df['mf30_tratado'] = df['mf30'].apply(balance_test_score, test='mf30')
        df['mf31_tratado'] = df['mf31'].apply(balance_test_score, test='mf31')
        df['mf32_tratado'] = df['mf32'].apply(balance_test_score, test='mf32')
        df['balance_test'] = df[['mf30_tratado','mf31_tratado','mf32_tratado']].sum(axis=1)

        # =========================
        # (9) VITALITY
        # =========================
        for col in ['mf27','mf28','mf29']:
            if col not in df.columns:
                df[col] = np.nan

        df[['mf27','mf28','mf29']] = df[['mf27','mf28','mf29']].replace(
            {9555: 0, 9666: 0, 9777: np.nan, 9888: np.nan, 8888: np.nan, 888: np.nan}
        )

        df['grip_strength'] = df.apply(lambda r: maior_valor_nan(r, ['mf27','mf28','mf29']), axis=1)
        df['grip_strength_category'] = df.apply(categorize_grip_strength, axis=1)

        # =========================
        # (10) FRAILTY
        # =========================
        for col in ['n69','n72','n73']:
            if col not in df.columns:
                df[col] = np.nan

        df['weight_loss'] = df['n69'].replace({9: np.nan})
        df['weight_loss_invertida'] = (df['weight_loss'].max() - df['weight_loss'])

        df['self_report_exhaustion'] = df['n72'].replace({9: np.nan})
        df['self_report_exhaustion_invertida'] = (df['self_report_exhaustion'].max() - df['self_report_exhaustion']) + 1

        df['poor_endurance'] = df['n73'].replace({9: np.nan})
        df['poor_endurance_invertida'] = (df['poor_endurance'].max() - df['poor_endurance']) + 1

        # =========================
        # (11) IMC
        # =========================
        for col in ['mf13','mf22']:
            if col not in df.columns:
                df[col] = np.nan

        df[['mf13','mf22']] = df[['mf13','mf22']].replace({99999: np.nan})
        df['imc'] = df['mf22'] / (df['mf13'] ** 2)

        # =========================
        # (12) DF FINAL (DESTINO)
        # =========================
        df_out = df[[
            'id',
            'regiao', 'zona', 'sexo', 'idade',
            'nacionalidade', 'situacao_conjugal', 'cor', 'escolaridade',
            'renda_mensal_familiar', 'renda_mensal_familiar_por_pessoa',
            'riqueza',
            'temporal_orientation', 'memory_recall', 'semantic_memory',
            'verbal_fluency_category',
            'depression_invertida',
            'sleep_quality_invertida',
            'hearing_deficit_invertida',
            'distance_vision_invertida',
            'near_vision_invertida',
            'gait_speed',
            'balance_test',
            'grip_strength_category',
            'weight_loss_invertida',
            'self_report_exhaustion_invertida',
            'poor_endurance_invertida',
            'imc'
        ]].copy()

        # =========================
        # (13) GARANTIR TABELA DESTINO + UPSERT
        # =========================
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

