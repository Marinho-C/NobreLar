import importlib.util
from pathlib import Path
import pandas as pd


VEICULOS = {
    "HR / Bongo": {"peso_kg": 1700.0, "volume_m3": 2.1793},
    "Acello 815": {"peso_kg": 4800.0, "volume_m3": 2.4543},
}


def _carregar_app():
    base = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("app", base / "app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


def processar_todos_os_eixos():
    app = _carregar_app()
    orders, audit, ranking = app.load_data()
    routes = app.DEFAULT_ROUTES

    relatorio = []
    for eixo_nome, cidades in routes.items():
        eligible = app.filter_axis(orders, cidades)
        if eligible.empty:
            continue

        for veiculo_nome, vd in VEICULOS.items():
            chosen_idx, msg = app.solve_knapsack(
                eligible, vd["peso_kg"], vd["volume_m3"], objective="max_lucro"
            )
            if not chosen_idx:
                continue
            chosen = eligible.iloc[chosen_idx]
            p_tot = chosen["peso_kg"].sum()
            v_tot = chosen["volume_m3"].sum()
            p_perc = p_tot / vd["peso_kg"] * 100
            v_perc = v_tot / vd["volume_m3"] * 100

            if p_perc >= v_perc:
                gargalo, perfil = "PESO (kg)", "Carga densa (pisos/argamassa)"
            else:
                gargalo, perfil = "VOLUME (m3)", "Carga volumosa (leve)"

            relatorio.append({
                "Eixo": eixo_nome,
                "Veiculo": veiculo_nome,
                "Pedidos": len(chosen),
                "Peso (kg)": round(p_tot, 1),
                "Ocup. peso (%)": round(p_perc, 1),
                "Volume (m3)": round(v_tot, 4),
                "Ocup. volume (%)": round(v_perc, 1),
                "Gargalo": gargalo,
                "Perfil": perfil,
                "Solver": msg,
            })

    return pd.DataFrame(relatorio)


if __name__ == "__main__":
    df = processar_todos_os_eixos()
    if df is not None and not df.empty:
        print(df.to_string(index=False))
    else:
        print("Nada a processar.")