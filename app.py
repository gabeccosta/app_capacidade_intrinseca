import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from mapas_variaveis import *

from etl_orchestrator import run_all_etl

DB_PATH = r"C:\Users\55219\Projetos\TCC\backend\banco.db"



# -------------------------
# Helpers DB
# -------------------------
def get_con():
    con = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    con.execute("PRAGMA busy_timeout = 30000;")
    return con

def init_db():
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.commit()
    finally:
        con.close()

init_db()

def insert_individuo(data: dict) -> int:
    cols = list(data.keys())
    placeholders = ",".join(["?"] * len(cols))
    colnames = ",".join(cols)

    sql = f"INSERT INTO tbl_dados_individuo ({colnames}) VALUES ({placeholders})"

    with sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False) as con:
        con.execute("PRAGMA busy_timeout = 30000;")
        cur = con.cursor()
        cur.execute(sql, [data[c] for c in cols])
        con.commit()
        return int(cur.lastrowid)

def load_view():
    con = get_con()
    try:
        return pd.read_sql_query("SELECT * FROM vw_individuo_api", con)
    finally:
        con.close()


def load_base_comparacao():
    con = get_con()
    try:
        df = pd.read_sql_query("SELECT * FROM tbl_base_comparacao", con)

        df = df.rename(columns={
            "dom_cognitivo": "dom_cognitivo_norm",
            "dom_psicologico": "dom_psicologico_norm",
            "dom_sensorial": "dom_sensorial_norm",
            "dom_locomotor": "dom_locomotor_norm",
            "dom_vitalidade": "dom_vitalidade_norm"
        })

        return df
    finally:
        con.close()

# -------------------------
# Radar
# -------------------------
DOM_COLS = [
    "dom_cognitivo_norm",
    "dom_psicologico_norm",
    "dom_sensorial_norm",
    "dom_locomotor_norm",
    "dom_vitalidade_norm",
]

def radar_plot(values, labels):
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    values = list(values)
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))
    ax.set_ylim(0, 1)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    return fig

def adicionar_valores_vertices_comparacao(ax, angulos, valores, deslocamento=0.05, fontsize=8, color="black"):
    for ang, val in zip(angulos[:-1], valores[:-1]):
        ax.text(
            ang,
            min(val + deslocamento, 1.08),
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            fontweight="bold"
        )

def radar_plot_compare(vals1, vals2, labels, label1="Indivíduo", label2="Grupo"):
    v1 = list(vals1)
    v2 = list(vals2)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    v1 += v1[:1]
    v2 += v2[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))

    ax.plot(angles, v1, linewidth=1.2, label=label1)
    ax.fill(angles, v1, alpha=0.20)

    ax.plot(angles, v2, linewidth=1.0, linestyle="--", label=label2)
    ax.fill(angles, v2, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=6)

    ax.set_ylim(0, 1.10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([], fontsize=6)
    ax.set_rlabel_position(20)

    cor_individuo = "tab:blue"
    cor_grupo = "tab:orange"

    # valores nos vértices
    adicionar_valores_vertices_comparacao(ax, angles, v1, deslocamento=0.04, fontsize=7, color=cor_individuo)
    adicionar_valores_vertices_comparacao(ax, angles, v2, deslocamento=0.10, fontsize=7, color=cor_grupo)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        fontsize=7,
        frameon=True
    )

    plt.tight_layout()
    return fig

def adicionar_valores_vertices(ax, angulos, valores, fontsize=7, color="tab:blue"):
    for ang, val in zip(angulos[:-1], valores[:-1]):
        deslocamento = 0.06
        ax.text(
            ang,
            min(val + deslocamento, 1.05),
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            fontweight="bold"
        )

def radar_plot_single(vals, labels):
    valores = list(vals)
    angulos = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    valores += valores[:1]
    angulos += angulos[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))

    ax.plot(angulos, valores, linewidth=1.2, label="Indivíduo")
    ax.fill(angulos, valores, alpha=0.20)

    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(labels, fontsize=6)

    ax.set_ylim(0, 1.08)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([], fontsize=6)
    ax.set_rlabel_position(20)

    adicionar_valores_vertices(ax, angulos, valores, fontsize=7)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(0.5, 1.12),
        fontsize=7,
        frameon=True
    )

    plt.tight_layout()
    return fig

def calcular_area_radar(valores):
    """
    Calcula a área do polígono do radar a partir dos valores dos domínios.
    Funciona para N eixos igualmente espaçados.
    """
    vals = np.array(valores, dtype=float)

    if np.any(pd.isna(vals)):
        return np.nan

    n = len(vals)
    angulos = np.linspace(0, 2 * np.pi, n, endpoint=False)

    x = vals * np.cos(angulos)
    y = vals * np.sin(angulos)

    # fórmula do polígono (shoelace)
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)

def calcular_percentil(valor, serie):
    serie = pd.Series(serie).dropna().astype(float)
    if serie.empty:
        return None
    return float((serie <= valor).mean() * 100)

LABELS = {
    "regiao": "Regiao",
    "zona": "Zona",
    "sexo": "Sexo",
    "idade": "Idade",
    "e2": "Nacionalidade",
    "e7": "Situação Conjugal",
    "e9": "Cor",
    "e22": "Escolaridade",
    "rendadom": "Renda Familiar Mensal",
    "rendadompc": "Renda Familiar Mensal Per Capita",
    "b4": "Quanto falta para quitar o imovel?",
    "b6": "Qual o valor do imovel",
    "b8": "Valor de outras propriedades",
    "b37": "Qual o valor do(s) veiculo(s)",
    "q7": "Dia esta correto?",
    "q8": "Mês esta correto?",
    "q9": "Ano esta correto?",
    "q10": "Dia da semana esta correto?",
    "q13": "mi_teste_palavras",
    "q18": "ms_pergunta_planta",
    "q19": "ms_pergunta_cortar_papel",
    "q20": "ms_pergunta_atual_presidente",
    "q21": "ms_pergunta_atual_vice_presidente",
    "q14": "fv_teste_animais",
    "mf33": "vm_1_minutos",
    "mf34": "vm_1_segundos",
    "mf35": "vm_1_centesimo",
    "mf36": "vm_2_minutos",
    "mf37": "vm_2_segundos",
    "mf38": "vm_2_centesimo",
    "mf30": "te_pes_lado",
    "mf31": "te_pes_pouco_frente",
    "mf32": "te_pes_atras",
    "r2": "sd_deprimido",
    "r3": "sd_dificuldade",
    "r4": "sd_sono",
    "r5": "sd_feliz",
    "r6": "sd_solitario",
    "r7": "sd_vida",
    "r8": "sd_triste",
    "r9": "sd_continuar",
    "n74": "qs_qualidade_sono",
    "n75": "qs_remedio_para_dormir",
    "n6": "dv_visao_longe",
    "n7": "dv_visao_perto",
    "n16": "da_deficiencia_auditiva",
    "mf27": "fo_pressao_manual_01",
    "mf28": "fo_pressao_manual_02",
    "mf29": "fo_pressao_manual_03",
    "n69": "ppi_perda_peso",
    "n72": "vi_exaustao",
    "n73": "vi_resistencia",
    "mf22": "vi_imc_peso",
    "mf13": "vi_imc_altura",
}

# -------------------------
# UI
# -------------------------

st.markdown("""
<style>
div[data-testid="stMetricValue"] {
    font-size: 16px;
}
div[data-testid="stMetricLabel"] {
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="TCC - IRCI", layout="wide")
st.title("Índice Relativo de Capacidade Intrínseca")

tab1, tab2 = st.tabs(["1) Cadastro", "2) IRCI"])

# =========================================================
# TAB 1 — CADASTRO
# =========================================================
with tab1:
    st.subheader("Cadastrar novo indivíduo")

    with st.form("form_cadastro", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            regiao_nome = st.selectbox(
                "Região",
                options=list(REGIAO.keys())
            )
            regiao = REGIAO[regiao_nome]
            zona_nome = st.selectbox(
                "Zona",
                options=list(ZONA.keys())
            )
            zona = ZONA[zona_nome]
        with c2:
            sexo_nome = st.selectbox(
                "Sexo",
                options=list(SEXO.keys())
            )
            sexo = SEXO[sexo_nome]
            idade = st.slider("Idade", 0,115,50)
        with c3:
            nacionalidade_nome = st.selectbox(
                "Nasceu no Brasil",
                options=list(NACIONALIDADE.keys())
            )
            e2 = NACIONALIDADE[nacionalidade_nome]
            conjugal_nome = st.selectbox(
                "Situação conjugal",
                options=list(SITUACAO_CONJUGAL.keys())
            )
            e7 = SITUACAO_CONJUGAL[conjugal_nome]
        with c4:
            cor_nome = st.selectbox(
                "Cor",
                options=list(COR.keys())
            )
            e9 = COR[cor_nome]
            esc_nome = st.selectbox(
                "Escolaridade",
                options=list(ESCOLARIDADE.keys())
            )
            e22 = ESCOLARIDADE[esc_nome]

        st.markdown("---")
        st.subheader("Patrimônio")
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            b4_nome = st.selectbox(
                "Quanto ainda falta para terminar de pagar esta casa/propriedade (ou seja, qual o valor da dívida com a hipoteca ou financiamento do imóvel)?",
                options=list(B4_VALOR_QUITACAO.keys())
            )
            b4 = B4_VALOR_QUITACAO[b4_nome]
        with c6:
            b6_nome = st.selectbox(
                "Quanto valeria esta casa se vendida agora?",
                options=list(B6_VALOR_PROPRIEDADE.keys())
            )
            b6 = B6_VALOR_PROPRIEDADE[b6_nome]
        with c7:
            b8_nome = st.selectbox(
                "Excluindo essa casa, quanto valeriam todas as outras propriedades se vendidas agora?",
                options=list(B6_VALOR_PROPRIEDADE.keys())
            )
            b8 = B6_VALOR_PROPRIEDADE[b8_nome]
        with c8:
            b37_nome = st.selectbox(
                "Se todos os veículos fossem vendidos hoje, qual seria o valor total?",
                options=list(B37_VALOR_VEICULOS.keys())
            )
            b37 = B37_VALOR_VEICULOS[b37_nome]

        st.markdown("---")
        st.subheader("Renda Mensal")
        c55, c65 = st.columns(2)
        with c55:
            rendadom = st.number_input(" Renda familiar mensal", min_value=0, value=0)
        with c65:
            rendadompc = st.number_input("Renda familiar mensal per capita", min_value=0, value=0)

        st.markdown("---")
        
        st.subheader("Orientação Temporal")
        st.caption("Perguntar a data de hoje e o dia da semana")
        c9, c10, c11, c12 = st.columns(4)
        with c9:
            q7_nome = st.selectbox(
                "O dia esta correto correto?",
                options=list(SIM_NAO.keys())
            )
            q7 = SIM_NAO[q7_nome]
        with c10:
            q8_nome = st.selectbox(
                "O mês esta correto?",
                options=list(SIM_NAO.keys())
            )
            q8 = SIM_NAO[q8_nome]
        with c11:
            q9_nome = st.selectbox(
                "O ano esta correto?",
                options=list(SIM_NAO.keys())
            )
            q9 = SIM_NAO[q9_nome]
        with c12:
            q10_nome = st.selectbox(
                "O dia da semana esta correto?",
                options=list(SIM_NAO.keys())
            )
            q10 = SIM_NAO[q10_nome]

        st.markdown("---")
        st.subheader("Memória Imediata")
        st.caption("Informar uma lista de 10 palavras. O indivíduo terá 2 minutos para dizer as palavras que se lembra")
        c13, c14, c15, c16 = st.columns(4)
        with c13:
                q13 = st.slider("Quantidade de palavras lembradas", 0,10,5)

        st.markdown("---")
        st.subheader("Memória Semântica")
        c17, c18, c19, c20 = st.columns(4)
        with c17:
            q18_nome = st.selectbox(
                "Qual a planta de folha longa e verde que dá um fruto amarelo e comprido(quando maduro), e que a gente descasca para comer?",
                options=list(MS_FRUTA.keys())
            )
            q18 = MS_FRUTA[q18_nome]
        with c18:
            q19_nome = st.selectbox(
                "O que geralmente as pessoas usam para cortar o papel?",
                options=list(MS_TESOURA.keys())
            )
            q19 = MS_TESOURA[q19_nome]
        with c19:
            q20_nome = st.selectbox(
                "Quem é o(a) atual presidente do Brasil?",
                options=list(MS_PRESIDENTE.keys())
            )
            q20 = MS_PRESIDENTE[q20_nome]
        with c20:
            q21_nome = st.selectbox(
                "Quem é o(a) vice-presidente do Brasil?",
                options=list(MS_VICE_PRESIDENTE.keys())
            )
            q21 = MS_VICE_PRESIDENTE[q21_nome]

        st.markdown("---")
        st.subheader("Fluência Verbal")
        st.caption("Solicite o nome de diferentes animais dentro de 1 minuto")
        c21, c22, c23, c24 = st.columns(4)
        with c21:
            q14 = st.number_input("Quantidade de animais lembrados", min_value=0, value=0)
        
        st.markdown("---")
        st.subheader("Velocidade da Marcha")
        st.caption("Solicitar que o indivíduo caminhe pelo percurso de 3 metros em velocidade normal de caminhada.")
        c25, c26 = st.columns(2)
        with c25:
            st.caption("Primeira tentativa:")
            c27, c28, c29 = st.columns(3)
            with c27:
                mf33 = st.number_input("minutos", min_value=0, value=0)
            with c28:
                mf34 = st.number_input("segundos", min_value=0, value=0)
            with c29:
                 mf35 = st.number_input("centésimos", min_value=0, value=0)
        with c26:
            st.caption("Segunda tentativa:")
            c30, c31, c32 = st.columns(3)
            with c30:
                mf36 = st.number_input("minutos_2", min_value=0, value=0)
            with c31:
                mf37 = st.number_input("segundos_2", min_value=0, value=0)
            with c32:
                 mf38 = st.number_input("centésimos_2", min_value=0, value=0)

        st.markdown("---")
        st.subheader("Teste de Equilíbrio")
        c33, c34, c35 = st.columns(3)
        with c33:
            mf30_nome = st.selectbox(
                "Quantos segundos o indivíduo conseguiu se manter com os pés lado a lado?",
                options=list(TESTE_EQUILIBRIO.keys())
            )
            mf30 = TESTE_EQUILIBRIO[mf30_nome]
        with c34:
            mf31_nome = st.selectbox(
                "Quantos segundos o indivíduo conseguiu se manter com um pé um pouco à frente do outro?",
                options=list(TESTE_EQUILIBRIO.keys())
            )
            mf31 = TESTE_EQUILIBRIO[mf31_nome]
        with c35:
            mf32_nome = st.selectbox(
                "Quantos segundos o indivíduo conseguiu se manter com um pé atrás do outro?",
                options=list(TESTE_EQUILIBRIO.keys())
            )
            mf32 = TESTE_EQUILIBRIO[mf32_nome]

        st.markdown("---")
        st.subheader("Sintomas Depressivos")
        c36, c37, c38, c39 = st.columns(4)
        with c36:
            r1_nome = st.selectbox(
                "A entrevista está sendo respondida com a ajuda de outra pessoa?",
                options=list(SD_SIM_NAO.keys())
            )
            r1 = SD_SIM_NAO[r1_nome]
            r2_nome = st.selectbox(
                "Durante a ÚLTIMA SEMANA, na maior parte do tempo, o(a) Sr(a) se sentiu deprimido(a)",
                options=list(SD_SIM_NAO.keys())
            )
            r2 = SD_SIM_NAO[r2_nome]
            r9_nome = st.selectbox(
                "Durante a ÚLTIMA SEMANA, na maior parte do tempo, o(a) Sr(a) sentiu que não conseguiria levar adiante as suas coisas",
                options=list(SD_SIM_NAO.keys())
            )
            r9 = SD_SIM_NAO[r9_nome]
        with c37:
            r3_nome = st.selectbox(
                "Durante a ÚLTIMA SEMANA, na maior parte do tempo, o(a) Sr(a) sentiu que as coisas estavam mais difíceis do que costumavam ser antes?",
                options=list(SD_SIM_NAO.keys())
            )
            r3 = SD_SIM_NAO[r3_nome]
            r4_nome = st.selectbox(
                "Durante a ÚLTIMA SEMANA, na maior parte do tempo, o(a) Sr(a) sentiu que o seu sono não era reparador, ou seja, o(a) Sr(a) acordava sem se sentir descansado(a)?",
                options=list(SD_SIM_NAO.keys())
            )
            r4 = SD_SIM_NAO[r4_nome]
        with c38:
            r5_nome = st.selectbox(
                "Durante a ÚLTIMA SEMANA, na maior parte do tempo, o(a) Sr(a) se sentiu feliz?",
                options=list(SD_SIM_NAO.keys())
            )
            r5 = SD_SIM_NAO[r5_nome]
            r6_nome = st.selectbox(
                "Durante a ÚLTIMA SEMANA, na maior parte do tempo, o(a) Sr(a) se sentiu solitário(a)?",
                options=list(SD_SIM_NAO.keys())
            )
            r6 = SD_SIM_NAO[r6_nome]
        with c39:
            r7_nome = st.selectbox(
                "Durante a ÚLTIMA SEMANA, na maior parte do tempo, o(a) Sr(a) aproveitou ou sentiu prazer pela vida?",
                options=list(SD_SIM_NAO.keys())
            )
            r7 = SD_SIM_NAO[r7_nome]
            r8_nome = st.selectbox(
                "Durante a ÚLTIMA SEMANA, na maior parte do tempo, o(a) Sr(a) se sentiu triste?",
                options=list(SD_SIM_NAO.keys())
            )
            r8 = SD_SIM_NAO[r8_nome]

        st.markdown("---")
        st.subheader("Qualidade do sono")
        c40, c41 = st.columns(2)
        with c40:
            n74_nome = st.selectbox(
                "Como o(a) Sr(a) avalia a qualidade do seu sono?",
                options=list(QS_QUALIDADE.keys())
            )
            n74 = QS_QUALIDADE[n74_nome]
        with c41:
            n75_nome = st.selectbox(
                "Durante o último mês o(a) Sr(a) tomou remédio para dormir?",
                options=list(QS_REMEDIO_DORMIR.keys())
            )
            n75 = QS_REMEDIO_DORMIR[n75_nome]

        st.markdown("---")
        st.subheader("Autoavaliação de visão")
        c42, c43 = st.columns(2)
        with c42:
            n6_nome = st.selectbox(
                "Como o(a) Sr(a) avalia a sua visão para enxergar de longe (MESMO USANDO ÓCULOS OU LENTES DE CONTATO). Ou seja, reconhecer uma pessoa conhecida do outro lado da rua a uma distância de mais ou menos 20 metros?",
                options=list(QS_QUALIDADE.keys())
            )
            n6 = QS_QUALIDADE[n6_nome]
        with c43:
            n7_nome = st.selectbox(
                "Como o(a) Sr(a) avalia a sua visão para enxergar de perto (MESMO USANDO ÓCULOS OU LENTES DE CONTATO). Ou seja, reconhecer um objeto que esteja ao alcance das mãos ou ler um jornal?",
                options=list(QS_QUALIDADE.keys())
            )
            n7 = QS_QUALIDADE[n7_nome]

        st.markdown("---")
        st.subheader("Deficiência Auditiva")
        c44, c45 = st.columns(2)
        with c44:
            n16_nome = st.selectbox(
                "Como o(a) Sr(a) avalia a sua audição (mesmo usando aparelho auditivo)?",
                options=list(QS_QUALIDADE.keys())
            )
            n16 = QS_QUALIDADE[n16_nome]
        
        st.markdown("---")
        st.subheader("Força de Preensão Manual")
        c46, c47, c48 = st.columns(3)
        with c46:
            mf27 = st.number_input("Primeira Medida", min_value=0, step=1, value=0)
        with c47:
            mf28 = st.number_input("Segunda Medida", min_value=0, step=1, value=0)
        with c48:
            mf29 = st.number_input("Terceira Medida", min_value=0, step=1, value=0)

        st.markdown("---")
        st.subheader("Perda de Peso Involuntária")
        c49, c50, c51 = st.columns(3)
        with c49:
            n69_nome = st.selectbox(
                "Nos ÚLTIMOS 3 MESES, o(a) Sr(a) perdeu peso sem fazer nenhuma dieta?",
                options=list(NACIONALIDADE.keys())
            )
            n69 = NACIONALIDADE[n69_nome]

        st.markdown("---")
        st.subheader("Exaustão Auto Relatada")
        c52, c53, c54 = st.columns(3)
        with c52:
            n72_nome = st.selectbox(
                "Na ÚLTIMA SEMANA, com que frequência o(a) Sr(a) sentiu que não conseguiria levar adiante suas coisas (iniciava alguma coisa, mas não conseguia terminar)?",
                options=list(EXAUSTAO.keys())
            )
            n72 = EXAUSTAO[n72_nome]

        st.markdown("---")
        st.subheader("Resistência")
        c55, c56, c57 = st.columns(3)
        with c55:
            n73_nome = st.selectbox(
                "Na ÚLTIMA SEMANA, com que frequência a realização de suas atividades rotineiras exigiram do(a) Sr(a) um grande esforço para serem realizadas?",
                options=list(EXAUSTAO.keys())
            )
            n73 = EXAUSTAO[n73_nome]

        st.markdown("---")
        st.subheader("Índice de Massa Corpórea – IMC")
        c58, c59 = st.columns(2)
        with c58:
            mf22 = st.number_input("Média do peso (em kg)", min_value=0.0, value=0.0)
        with c59:
            mf13 = st.number_input("Média da altura (em cm)", min_value=0, step=1, value=0)

        submitted = st.form_submit_button("Salvar paciente")

    if submitted:
        # Monta dict com TODOS os campos da tabela (exceto id)
        data = dict(
            regiao=int(regiao),
            zona=int(zona),
            sexo=int(sexo),
            idade=int(idade),

            e2=int(e2),
            e7=int(e7),
            e9=int(e9),
            e22=int(e22),

            rendadom=int(rendadom),
            rendadompc=float(rendadompc),

            b4=int(b4),
            b6=int(b6),
            b8=int(b8),
            b37=int(b37),

            q7=int(q7), q8=int(q8), q9=int(q9), q10=int(q10),
            q13=int(q13),
            q18=int(q18), q19=int(q19), q20=int(q20), q21=int(q21),
            q14=int(q14),

            mf33=int(mf33), mf34=int(mf34), mf35=int(mf35),
            mf36=int(mf36), mf37=int(mf37), mf38=int(mf38),

            mf30=int(mf30), mf31=int(mf31), mf32=int(mf32),

            r2=int(r2), r3=int(r3), r4=int(r4), r5=int(r5),
            r6=int(r6), r7=int(r7), r8=int(r8), r9=int(r9),

            n74=int(n74), n75=int(n75),
            n6=int(n6), n7=int(n7), n16=int(n16),

            mf27=int(mf27), mf28=int(mf28), mf29=int(mf29),

            n69=int(n69), n72=int(n72), n73=int(n73),

            mf22=float(mf22),
            mf13=float(mf13),
        )

        if submitted:
            try:
                new_id = insert_individuo(data)
                st.success(f"Paciente salvo com ID = {new_id}")

                resumo = run_all_etl(DB_PATH)
                st.success(f"ETL OK: {resumo}")

            except Exception as e:
                st.error("Erro ao salvar ou processar ETL.")
                st.code(str(e))
    if st.button("Processar ETL", use_container_width=True):
            try:
                with st.spinner("Processando ETL..."):
                    resumo = run_all_etl(DB_PATH)
                st.success(f"ETL executado com sucesso: {resumo}")
                st.rerun()
            except Exception as e:
                st.error("Erro ao executar o ETL.")
                st.code(str(e))
# =========================================================
# TAB 2 — COMPARAÇÃO
# =========================================================

with tab2:

    try:
        df = load_view()                  # paciente(s) cadastrados
        df_base = load_base_comparacao()  # base completa de referência
    except Exception as e:
        st.error("Não consegui carregar os dados da comparação.")
        st.code(str(e))
        st.stop()

    if df.empty:
        st.warning("A view vw_individuo_api está vazia.")
        st.stop()

    if df_base.empty:
        st.warning("A tabela tbl_base_comparacao está vazia.")
        st.stop()

    required_cols_view = [
        "id",
        "sexo_nome",
        "cor_nome",
        "faixa_etaria",
        "faixa_riqueza",
        "regiao_nome",
        "zona_nome",
        "escolaridade_nome",
        *DOM_COLS
    ]
    missing_view = [c for c in required_cols_view if c not in df.columns]
    if missing_view:
        st.error(f"A view vw_individuo_api está sem estas colunas: {missing_view}")
        st.write("Colunas disponíveis na view:", df.columns.tolist())
        st.stop()

    required_cols_base = [
        "sexo_nome",
        "cor_nome",
        "faixa_etaria",
        "faixa_riqueza",
        "regiao_nome",
        "zona_nome",
        "escolaridade_nome",
        *DOM_COLS
    ]
    missing_base = [c for c in required_cols_base if c not in df_base.columns]
    if missing_base:
        st.error(f"A tabela tbl_base_comparacao está sem estas colunas: {missing_base}")
        st.write("Colunas disponíveis na base:", df_base.columns.tolist())
        st.stop()

    labels = ["Cognitivo", "Psicológico", "Sensorial", "Locomotor", "Vitalidade"]

    st.header("Consulta do indivíduo")

    col1, col2 = st.columns([1, 2])

    with col1:
        ids = sorted(df["id"].dropna().astype(int).unique().tolist())
        id_selecionado = st.selectbox("Selecione o ID do indivíduo", ids)

        pac_df = df[df["id"] == id_selecionado]

    if pac_df.empty:
        st.warning("Indivíduo não encontrado.")
        st.stop()

    pac = pac_df.iloc[0]

    with col2:
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Sexo", pac.get("sexo_nome", "Sem informação"))
            st.metric("Região", pac.get("regiao_nome", "Sem informação"))

        with c2:
            st.metric("Cor/Raça", pac.get("cor_nome", "Sem informação"))
            st.metric("Zona", pac.get("zona_nome", "Sem informação"))

        with c3:
            st.metric("Faixa etária", pac.get("faixa_etaria", "Sem informação"))
            st.metric("Escolaridade", pac.get("escolaridade_nome", "Sem informação"))

        with c4:
            st.metric("Faixa de riqueza", pac.get("faixa_riqueza", "Sem informação"))

    try:
        vals_pac = [float(pac[c]) for c in DOM_COLS]
    except Exception:
        st.warning("Paciente ainda sem domínios disponíveis para comparação.")
        st.stop()

    if any(pd.isna(v) for v in vals_pac):
        st.warning("Paciente ainda sem domínios normalizados.")
        st.stop()

    # ========= GRÁFICO INICIAL: SÓ INDIVÍDUO =========
    st.markdown("### Gráfico de Radar")

    if "grupo_aplicado_tab2" not in st.session_state:
        st.session_state["grupo_aplicado_tab2"] = False

    if "vals_grupo_tab2" not in st.session_state:
        st.session_state["vals_grupo_tab2"] = None

    if "modo_grupo_tab2" not in st.session_state:
        st.session_state["modo_grupo_tab2"] = "Média"

    if "n_grupo_tab2" not in st.session_state:
        st.session_state["n_grupo_tab2"] = None

    if "area_grupo_tab2" not in st.session_state:
        st.session_state["area_grupo_tab2"] = None

    # Se ainda não aplicou filtros, mostra só o indivíduo
    col_esq, col_centro, col_dir = st.columns([1, 3, 1])
    with col_centro:
        if not st.session_state["grupo_aplicado_tab2"] or st.session_state["vals_grupo_tab2"] is None:
            fig = radar_plot_single(vals_pac, labels)
            fig.set_size_inches(4.0, 4.0)
            st.pyplot(fig, use_container_width=False)
        else:
            vals_grupo = st.session_state["vals_grupo_tab2"]
            fig = radar_plot_compare(vals_pac, vals_grupo, labels)
            fig.set_size_inches(4.0, 4.0)
            st.pyplot(fig, use_container_width=False)

    # ========= ÁREAS =========
    area_individuo = calcular_area_radar(vals_pac)
    area_grupo = st.session_state["area_grupo_tab2"]

    irci = None
    if area_grupo is not None and area_grupo > 0:
        irci = (area_individuo / area_grupo) * 100
        irci = min(irci, 100.0)

    st.markdown("### Indicadores de Área")
    a1, a2, a3 = st.columns(3)

    with a1:
        st.metric("Área do indivíduo", f"{area_individuo:.4f}")

    with a2:
        st.metric(
            "Área do grupo",
            f"{area_grupo:.4f}" if area_grupo is not None else "-"
        )

    with a3:
        st.metric(
            "IRCI",
            f"{irci:.2f}%" if irci is not None else "-"
        )

    # ========= FILTROS ABAIXO DO GRÁFICO =========
    st.markdown("### Recorte Sociodemográfico")

    with st.form("form_recorte_grupo"):
        c1, c2, c3, c4 = st.columns(4)
        c5, c6, c7, c8 = st.columns(4)
        c9, c10, c11 = st.columns(3)

        with c1:
            sexo_opts = ["Todos"] + sorted(df_base["sexo_nome"].dropna().astype(str).unique().tolist())
            sexo_opt = st.selectbox("Sexo", options=sexo_opts)

        with c2:
            raca_opts = ["Todas"] + sorted(df_base["cor_nome"].dropna().astype(str).unique().tolist())
            raca_opt = st.selectbox("Raça", options=raca_opts)

        with c3:
            faixa_etaria_opts = ["Todas"] + sorted(df_base["faixa_etaria"].dropna().astype(str).unique().tolist())
            faixa_etaria_opt = st.selectbox("Faixa Etária", options=faixa_etaria_opts)

        with c4:
            faixa_riqueza_opts = ["Todas"] + sorted(df_base["faixa_riqueza"].dropna().astype(str).unique().tolist())
            faixa_riqueza_opt = st.selectbox("Faixa de Riqueza", options=faixa_riqueza_opts)

        with c5:
            regiao_opts = ["Todas"] + sorted(df_base["regiao_nome"].dropna().astype(str).unique().tolist())
            regiao_opt = st.selectbox("Região", options=regiao_opts)

        with c6:
            zona_opts = ["Todas"] + sorted(df_base["zona_nome"].dropna().astype(str).unique().tolist())
            zona_opt = st.selectbox("Zona", options=zona_opts)

        with c7:
            escolaridade_opts = ["Todas"] + sorted(df_base["escolaridade_nome"].dropna().astype(str).unique().tolist())
            escolaridade_opt = st.selectbox("Escolaridade", options=escolaridade_opts)

        with c9:
            modo_grupo = st.radio(
                "Escolha a estatística:",
                options=["Média", "Mediana", "Máximo"],
                horizontal=True  
            )

        aplicar_filtros = st.form_submit_button("Aplicar recorte ao gráfico")

    # ========= APLICAÇÃO DOS FILTROS SOMENTE AO CLICAR =========
    if aplicar_filtros:
        grupo = df_base.copy()

        if sexo_opt != "Todos":
            grupo = grupo[grupo["sexo_nome"] == sexo_opt]

        if raca_opt != "Todas":
            grupo = grupo[grupo["cor_nome"] == raca_opt]

        if faixa_etaria_opt != "Todas":
            grupo = grupo[grupo["faixa_etaria"] == faixa_etaria_opt]

        if faixa_riqueza_opt != "Todas":
            grupo = grupo[grupo["faixa_riqueza"] == faixa_riqueza_opt]

        if regiao_opt != "Todas":
            grupo = grupo[grupo["regiao_nome"] == regiao_opt]

        if zona_opt != "Todas":
            grupo = grupo[grupo["zona_nome"] == zona_opt]

        if escolaridade_opt != "Todas":
            grupo = grupo[grupo["escolaridade_nome"] == escolaridade_opt]

        grupo_ok = grupo.dropna(subset=DOM_COLS)

        if grupo_ok.empty:
            st.warning("Nenhum registro encontrado na base de referência com os filtros selecionados.")
            st.session_state["grupo_aplicado_tab2"] = False
            st.session_state["vals_grupo_tab2"] = None
            st.session_state["modo_grupo_tab2"] = modo_grupo
            st.session_state["n_grupo_tab2"] = 0
            st.session_state["area_grupo_tab2"] = None
        else:
            if modo_grupo == "Média":
                vals_grupo = [float(grupo_ok[c].mean()) for c in DOM_COLS]
            elif modo_grupo == "Mediana":
                vals_grupo = [float(grupo_ok[c].median()) for c in DOM_COLS]
            else:  # Máximo
                vals_grupo = [float(grupo_ok[c].max()) for c in DOM_COLS]

            area_grupo_calc = calcular_area_radar(vals_grupo)

            st.session_state["grupo_aplicado_tab2"] = True
            st.session_state["vals_grupo_tab2"] = vals_grupo
            st.session_state["modo_grupo_tab2"] = modo_grupo
            st.session_state["n_grupo_tab2"] = len(grupo_ok)
            st.session_state["area_grupo_tab2"] = area_grupo_calc

            st.rerun()

    # ========= TABELA DE COMPARAÇÃO =========
    if st.session_state["grupo_aplicado_tab2"] and st.session_state["vals_grupo_tab2"] is not None:
        vals_grupo = st.session_state["vals_grupo_tab2"]
        nome_modo = st.session_state["modo_grupo_tab2"]

        st.markdown(f"### Comparação por domínio: Indivíduo vs {nome_modo.lower()} do grupo")

        comparacao_df = pd.DataFrame({
            "Domínio": labels,
            "Indivíduo": vals_pac,
            f"{nome_modo} do grupo": vals_grupo
        })
        comparacao_df["Diferença"] = comparacao_df["Indivíduo"] - comparacao_df[f"{nome_modo} do grupo"]

        st.dataframe(
            comparacao_df.style.format({
                "Indivíduo": "{:.4f}",
                f"{nome_modo} do grupo": "{:.4f}",
                "Diferença": "{:.4f}"
            }),
            use_container_width=True
        )

        st.write(f"Grupo filtrado na base de referência: **{st.session_state['n_grupo_tab2']}** indivíduos")
