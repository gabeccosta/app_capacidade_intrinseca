# etl_orchestrator.py
from etl_indicadores import run_etl
from etl_dominios import run_etl_dominios
from etl_normalizar_dominios import run_etl_normalizar_dominios
from etl_ci_area import run_etl_ci_area

def run_all_etl(conn) -> dict:
    """
    Roda toda a pipeline na ordem correta.
    Retorna um resumo com quantidades processadas.
    """
    resumo = {}

    n1 = run_etl(conn)
    resumo["indicadores"] = n1

    n2 = run_etl_dominios(conn)
    resumo["dominios"] = n2

    n3 = run_etl_normalizar_dominios(conn)
    resumo["dominios_norm"] = n3

    n4 = run_etl_ci_area(conn)
    resumo["ci_area"] = n4

    resumo["status"] = "ok"
    return resumo
