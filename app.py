import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ============================================================
# CONFIGURACAO
# ============================================================

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"


def locate_file(pattern):
    search_roots = [DATA, BASE]
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.glob(pattern):
            key = str(path.resolve()).lower()
            if key not in seen and path.is_file():
                seen.add(key)
                return path
        for path in root.rglob(pattern):
            key = str(path.resolve()).lower()
            if key not in seen and path.is_file():
                seen.add(key)
                return path
    return None


def read_csv(path):
    path = Path(path)
    errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        for sep in (";", ",", "\t", "|"):
            try:
                df = pd.read_csv(
                    path,
                    sep=sep,
                    encoding=encoding,
                    dtype=object,
                    keep_default_na=True,
                    engine="python",
                )
                if len(df.columns) > 1:
                    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
                    return df
                errors.append(f"{encoding}/{repr(sep)}: 1 coluna")
            except Exception as exc:
                errors.append(f"{encoding}/{repr(sep)}: {exc}")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=object, engine="python", sep=None)
        df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
        return df
    except Exception as exc:
        errors.append(str(exc))
    raise ValueError(
        f"Nao foi possivel ler o arquivo {path.name}. Tentativas: "
        + " | ".join(errors[-8:])
    )


DEFAULT_VEHICLES = {
    "HR / Bongo": {"peso_kg": 1700.0, "volume_m3": 2.1793},
    "Acello 815": {"peso_kg": 4800.0, "volume_m3": 2.4543},
}

DEFAULT_ROUTES = {
    "Eixo 1": ["Buriti dos Montes", "Barro Vermelho", "Tucuns", "Queimadas", "Filomena"],
    "Eixo 2": ["Sucesso", "Tamboril", "Nova Russas", "Fazenda"],
    "Eixo 3": ["Independencia"],
    "Eixo 4": ["Ipaporanga", "Poranga", "Ararenda", "Vaca Morta", "Curral Velho", "Curral do Meio"],
    "Eixo 5": ["Novo Oriente", "Realejo", "Santana", "Monte Nebo", "Santo Andre", "Quiterianopolis", "Barra dos Simioes"],
}


# ============================================================
# UTILITARIOS
# ============================================================

def norm(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[_\-/]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.upper().strip()


def num(x, default=0.0):
    if pd.isna(x):
        return default
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    if not s:
        return default
    s = s.replace("R$", "").replace("%", "").replace(" ", "")
    s = re.sub(r"[^0-9,.-]", "", s)
    if not s or s in {"-", ".", ","}:
        return default
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return default


def code_norm(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", "", s)
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    return s


def code_variants(x):
    c = code_norm(x)
    if not c:
        return set()
    variants = {c}
    if c.isdigit():
        variants.add(c.lstrip("0") or "0")
    return variants


def money(x):
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def find_col(df, candidates, exclude=None):
    exclude = set(exclude or [])
    normalized = {}
    for col in df.columns:
        if col not in exclude:
            normalized.setdefault(norm(col), col)
    for candidate in candidates:
        c = norm(candidate)
        if c in normalized:
            return normalized[c]
    scored = []
    for col in df.columns:
        if col in exclude:
            continue
        n = norm(col)
        if not n:
            continue
        for candidate in candidates:
            c = norm(candidate)
            if c and c in n:
                scored.append((100 + len(c), col))
            elif c and n in c and len(n) >= 4:
                scored.append((50 + len(n), col))
    if scored:
        scored.sort(key=lambda x: (-x[0], str(x[1])))
        return scored[0][1]
    return None


def parse_items(text):
    if pd.isna(text):
        return []
    raw = str(text).strip()
    if not raw:
        return []

    chunks = re.split(r";|\n|\|", raw)
    expanded = []
    for chunk in chunks:
        parts = re.split(
            r"(?<=\))\s*,\s*(?=[A-Za-z0-9._/-]+\s*-)",
            chunk.strip()
        )
        expanded.extend([p.strip() for p in parts if p.strip()])

    items = []
    pattern = re.compile(
        r"^\s*(?P<codigo>[A-Za-z0-9._/-]+)"
        r"(?:\s*-\s*(?P<desc>.*?))?"
        r"\s*\(\s*(?P<qtd>[-\d.,]+)"
        r"\s*(?P<un>[A-Za-zÀ-ÿ²³0-9]+)?\s*\)\s*$",
        re.I,
    )

    for chunk in expanded:
        m = pattern.match(chunk)
        if m:
            items.append({
                "codigo": m.group("codigo").strip(),
                "descricao": (m.group("desc") or "").strip(),
                "quantidade": num(m.group("qtd")),
                "unidade": (m.group("un") or "UN").strip(),
            })
            continue
        m2 = re.search(
            r"(?P<codigo>[A-Za-z0-9._/-]+).*?"
            r"\(\s*(?P<qtd>[-\d.,]+)\s*"
            r"(?P<un>[A-Za-zÀ-ÿ²³0-9]+)?\s*\)",
            chunk,
            re.I,
        )
        if m2:
            items.append({
                "codigo": m2.group("codigo").strip(),
                "descricao": chunk,
                "quantidade": num(m2.group("qtd")),
                "unidade": (m2.group("un") or "UN").strip(),
            })
    return items


def m2_por_caixa(row):
    candidates = [
        "m2 por caixa",
        "m2/caixa",
        "metros por caixa",
        "metragem caixa",
        "rendimento caixa",
        "area caixa",
    ]
    for c in candidates:
        col = find_col(pd.DataFrame([row]), [c])
        if col:
            value = num(row[col], 0)
            if value > 0:
                return value
    for col, value in row.items():
        text = f"{col} {value}"
        patterns = [
            r"(?:=|:|com)\s*(\d+(?:[.,]\d+)?)\s*(?:m2|m²)\b",
            r"(\d+(?:[.,]\d+)?)\s*(?:m2|m²)\s*(?:/|por)\s*caixa",
        ]
        for pattern in patterns:
            m = re.search(pattern, str(text), re.I)
            if m:
                value_num = num(m.group(1), 0)
                if value_num > 0:
                    return value_num
    return 0.0


# ============================================================
# ROTAS / VEICULOS
# ============================================================

def load_route_file():
    candidates = []
    for pattern in ["*Rota*.csv", "*rota*.csv", "*Eixo*.csv", "*eixo*.csv",
                    "*Veiculo*.csv", "*veiculo*.csv"]:
        candidates.extend(DATA.glob(pattern))

    if not candidates:
        return DEFAULT_ROUTES.copy(), DEFAULT_VEHICLES.copy(), None

    path = sorted(set(candidates))[0]
    try:
        df = read_csv(path)
    except Exception:
        return DEFAULT_ROUTES.copy(), DEFAULT_VEHICLES.copy(), path

    routes = {k: list(v) for k, v in DEFAULT_ROUTES.items()}
    vehicles = {k: dict(v) for k, v in DEFAULT_VEHICLES.items()}

    eixo_col = find_col(df, ["eixo", "rota", "eixo de entrega"])
    cidade_col = find_col(df, ["cidade", "destino", "localidade", "municipio"])
    veiculo_col = find_col(df, ["veiculo", "tipo veiculo"])
    peso_col = find_col(df, ["peso maximo", "capacidade kg", "capacidade peso", "kg"])
    volume_col = find_col(df, ["volume maximo", "capacidade m3", "m3"])

    if eixo_col and cidade_col:
        grouped = {}
        for _, row in df.iterrows():
            eixo_raw = str(row[eixo_col]).strip()
            cidade = str(row[cidade_col]).strip()
            if not eixo_raw or not cidade or norm(cidade) == "NAN":
                continue
            m = re.search(r"(\d+)", eixo_raw)
            eixo = f"Eixo {m.group(1)}" if m else eixo_raw
            grouped.setdefault(eixo, [])
            if norm(cidade) not in [norm(x) for x in grouped[eixo]]:
                grouped[eixo].append(cidade)
        for eixo, cidades in grouped.items():
            if cidades:
                routes[eixo] = cidades

    if veiculo_col:
        for _, row in df.iterrows():
            nome = str(row[veiculo_col]).strip()
            if not nome or nome.lower() == "nan":
                continue
            peso = num(row[peso_col], 0) if peso_col else 0
            volume = num(row[volume_col], 0) if volume_col else 0
            if peso > 0 or volume > 0:
                vehicles[nome] = {
                    "peso_kg": peso if peso > 0 else 0,
                    "volume_m3": volume if volume > 0 else 0,
                }

    for name, vals in DEFAULT_VEHICLES.items():
        if name not in vehicles:
            vehicles[name] = dict(vals)
        else:
            if vehicles[name]["peso_kg"] <= 0:
                vehicles[name]["peso_kg"] = vals["peso_kg"]
            if vehicles[name]["volume_m3"] <= 0:
                vehicles[name]["volume_m3"] = vals["volume_m3"]

    return routes, vehicles, path


# ============================================================
# LEITURA E TRATAMENTO DOS DADOS
# ============================================================

def is_cancelled(value):
    s = norm(value)
    return any(x in s for x in ["CANCEL", "CANCELADO", "CANCELADA"])


def is_counter_pickup(value):
    s = norm(value)
    return any(x in s for x in ["BALCAO", "RETIRADA", "RETIRA", "COUNTER"])


def is_crateus(value):
    return norm(value) == "CRATEUS"


def load_data():
    semana = getattr(load_data, "_semana", "Todas")
    weekly_files = []
    if semana == "Todas":
        for i in range(1, 5):
            p = locate_file(f"Pedidos_Filtrados_Semana_{i}_Anonimizado*.csv")
            if p:
                weekly_files.append(p)
    else:
        m = re.search(r"(\d+)", str(semana))
        if m:
            p = locate_file(f"Pedidos_Filtrados_Semana_{m.group(1)}_Anonimizado*.csv")
            if p:
                weekly_files.append(p)

    if not weekly_files:
        candidatos = sorted(DATA.glob("*Pedido*Semana*.csv"))
        weekly_files = candidatos

    ranking_file = locate_file("Ranking_Top85_Materiais*.csv")
    if ranking_file is None:
        ranking_file = locate_file("*Ranking*Top85*.csv")

    if not weekly_files:
        raise FileNotFoundError(f"Nenhum arquivo semanal encontrado em: {DATA}")
    if ranking_file is None:
        raise FileNotFoundError(f"Arquivo Ranking nao encontrado em: {DATA}")

    ranking = read_csv(ranking_file)

    code_rank_col = find_col(ranking, ["codigo", "cod produto", "codigo produto", "material", "cod material"])
    peso_col = find_col(ranking, ["peso kg", "peso unitario", "peso por unidade", "peso"])
    volume_col = find_col(ranking, ["volume m3", "cubagem m3", "cubagem", "volume unitario", "volume"])
    valor_unit_col = find_col(ranking, ["valor unitario", "preco unitario", "preco", "valor unitario venda", "valor"])

    if not code_rank_col:
        raise ValueError("Nao encontrei a coluna de codigo do produto no Ranking.")

    ranking_map = {}
    for _, row in ranking.iterrows():
        code = code_norm(row[code_rank_col])
        if not code:
            continue
        product = {
            "peso_unit_kg": num(row[peso_col], 0) if peso_col else 0,
            "volume_unit_m3": num(row[volume_col], 0) if volume_col else 0,
            "valor_unit": num(row[valor_unit_col], 0) if valor_unit_col else 0,
            "m2_caixa": m2_por_caixa(row),
            "descricao": str(row.get(find_col(ranking, ["descricao", "produto"]), "")),
        }
        for variant in code_variants(code):
            ranking_map[variant] = product

    all_rows = []
    audit = {
        "arquivos": [str(x.name) for x in weekly_files],
        "ranking": ranking_file.name,
        "removidos": [],
        "conversoes": [],
        "avisos": [],
        "itens_lidos": 0,
        "itens_com_match": 0,
        "itens_sem_match": 0,
        "colunas": [],
        "amostra_itens": [],
        "amostra_ranking_peso": [],
    }

    for _, row in ranking.head(5).iterrows():
        audit["amostra_ranking_peso"].append({
            "codigo": str(row[code_rank_col]),
            "peso_col": peso_col or "(nao achou)",
            "peso_raw": str(row[peso_col]) if peso_col else "",
            "volume_col": volume_col or "(nao achou)",
            "volume_raw": str(row[volume_col]) if volume_col else "",
        })

    for file in weekly_files:
        df = read_csv(file)

        order_col = find_col(df, ["pedido", "numero pedido", "ordem", "id pedido"])
        used = [order_col] if order_col else []

        city_col = find_col(df, ["cidade", "municipio", "destino", "cidade destino"], exclude=used)
        used += [city_col] if city_col else []

        value_col = find_col(df, ["valor do pedido", "valor pedido", "valor total",
                                  "total pedido", "vl pedido", "preco total", "valor"],
                             exclude=used)
        used += [value_col] if value_col else []

        items_col = find_col(df, ["itens resumo", "itens_resumo", "itens do pedido",
                                  "produtos", "resumo itens", "itens"],
                             exclude=used)
        used += [items_col] if items_col else []

        status_col = find_col(df, ["status", "situacao", "status pedido"], exclude=used)
        used += [status_col] if status_col else []

        pickup_col = find_col(df, ["tipo entrega", "forma entrega", "modalidade", "origem pedido"],
                              exclude=used)
        used += [pickup_col] if pickup_col else []

        date_col = find_col(df, ["data", "data pedido", "data entrega"], exclude=used)

        roles = {"pedido": order_col, "cidade": city_col, "valor": value_col, "itens": items_col}
        nonempty_roles = [v for v in roles.values() if v]
        if len(nonempty_roles) != len(set(nonempty_roles)):
            raise ValueError(f"Mapeamento de colunas ambiguo em {file.name}: {roles}.")

        audit["colunas"].append({
            "arquivo": file.name,
            "pedido": order_col or "-",
            "cidade": city_col or "-",
            "valor": value_col or "-",
            "itens": items_col or "-",
        })

        if not order_col:
            raise ValueError(f"Nao encontrei a coluna de pedido em {file.name}.")
        if not city_col:
            raise ValueError(f"Nao encontrei a coluna de cidade em {file.name}.")
        if not items_col:
            raise ValueError(f"Nao encontrei a coluna de itens em {file.name}.")

        sample_count = 0
        for _, row in df.iterrows():
            if sample_count >= 5:
                break
            raw = row[items_col]
            if pd.notna(raw) and str(raw).strip():
                audit["amostra_itens"].append({
                    "arquivo": file.name,
                    "raw": str(raw)[:300],
                    "parseado": str(parse_items(raw))[:300],
                })
                sample_count += 1

        for _, row in df.iterrows():
            order = str(row[order_col]).strip()
            city = str(row[city_col]).strip()
            status = str(row[status_col]).strip() if status_col else ""
            pickup = str(row[pickup_col]).strip() if pickup_col else ""

            reason = None
            if not order or order.lower() == "nan":
                reason = "Pedido sem numero."
            elif is_cancelled(status):
                reason = "Pedido cancelado."
            elif is_counter_pickup(pickup) or is_counter_pickup(city):
                reason = "Retirada no balcao."
            elif is_crateus(city):
                reason = "Entrega urbana fora do escopo."

            if reason:
                audit["removidos"].append({"pedido": order, "motivo": reason, "arquivo": file.name})
                continue

            items = parse_items(row[items_col])

            total_weight = 0.0
            total_volume = 0.0
            conversion_notes = []
            missing_codes = []

            audit["itens_lidos"] += len(items)
            for item in items:
                code = code_norm(item["codigo"])
                qty = float(item["quantidade"])
                unit = norm(item["unidade"])

                prod = None
                for variant in code_variants(code):
                    if variant in ranking_map:
                        prod = ranking_map[variant]
                        break

                if not prod:
                    missing_codes.append(code)
                    audit["itens_sem_match"] += 1
                    continue

                audit["itens_com_match"] += 1
                effective_qty = qty

                if unit in {"M2", "MT", "MTS"}:
                    factor = prod["m2_caixa"]
                    if factor > 0:
                        effective_qty = qty / factor
                        conversion_notes.append(
                            f"{code}: {qty:g} m2 / {factor:g} = {effective_qty:.2f} caixas"
                        )
                        audit["conversoes"].append({
                            "pedido": order,
                            "codigo": code,
                            "origem": f"{qty:g} m2",
                            "fator": factor,
                            "destino": f"{effective_qty:.2f} caixas",
                        })
                    else:
                        audit["avisos"].append(
                            f"Pedido {order}, produto {code}: unidade em m2 sem fator m2/caixa."
                        )

                total_weight += effective_qty * prod["peso_unit_kg"]
                total_volume += effective_qty * prod["volume_unit_m3"]

            if missing_codes:
                audit["avisos"].append(
                    f"Pedido {order}: codigo(s) sem match: " + ", ".join(sorted(set(missing_codes)))
                )

            order_value = num(row[value_col], 0) if value_col else 0
            if order_value <= 0 and items:
                derived = 0.0
                for item in items:
                    code = code_norm(item["codigo"])
                    qty = float(item["quantidade"])
                    unit = norm(item["unidade"])
                    prod = None
                    for variant in code_variants(code):
                        if variant in ranking_map:
                            prod = ranking_map[variant]
                            break
                    if prod:
                        eff = qty
                        if unit in {"M2", "MT", "MTS"} and prod["m2_caixa"] > 0:
                            eff = qty / prod["m2_caixa"]
                        derived += eff * prod["valor_unit"]
                if derived > 0:
                    order_value = derived
                    audit["avisos"].append(
                        f"Pedido {order}: valor calculado pelo ranking."
                    )

            all_rows.append({
                "pedido": order,
                "cidade": city,
                "valor": order_value,
                "peso_kg": total_weight,
                "volume_m3": total_volume,
                "data": str(row[date_col]).strip() if date_col else "",
                "arquivo": file.name,
                "conversoes": " | ".join(conversion_notes),
                "codigos_faltantes": ", ".join(sorted(set(missing_codes))),
            })

    orders = pd.DataFrame(all_rows)
    if orders.empty:
        raise ValueError("Apos os filtros, nenhum pedido ficou disponivel.")

    agg = {
        "cidade": "first",
        "valor": "sum",
        "peso_kg": "sum",
        "volume_m3": "sum",
        "data": "first",
        "arquivo": lambda x: " | ".join(sorted(set(map(str, x)))),
        "conversoes": lambda x: " | ".join([str(v) for v in x if str(v).strip()]),
        "codigos_faltantes": lambda x: " | ".join([str(v) for v in x if str(v).strip()]),
    }
    orders = orders.groupby("pedido", as_index=False).agg(agg)

    return orders, audit, ranking


# ============================================================
# ROTEAMENTO
# ============================================================

def city_matches(city, route_city):
    a = norm(city)
    b = norm(route_city)
    if a == b:
        return True
    return a in b or b in a


def filter_axis(orders, route):
    mask = orders["cidade"].apply(lambda city: any(city_matches(city, rc) for rc in route))
    return orders[mask].copy()


def route_position(city, route):
    for i, rc in enumerate(route):
        if city_matches(city, rc):
            return i
    return 9999


# ============================================================
# OTIMIZACAO 0/1
# ============================================================

def solve_knapsack(orders, max_weight, max_volume, objective="max_lucro"):
    if orders.empty:
        return [], "Nenhum pedido elegivel."

    weights = orders["peso_kg"].astype(float).to_numpy()
    volumes = orders["volume_m3"].astype(float).to_numpy()
    values = orders["valor"].astype(float).to_numpy()

    individually_impossible = (weights > max_weight + 1e-9) | (volumes > max_volume + 1e-9)
    idx = np.where(~individually_impossible)[0]

    if len(idx) == 0:
        return [], "Todos os pedidos excedem individualmente peso ou volume."

    w = weights[idx]
    v = volumes[idx]
    val = values[idx]

    if SCIPY_OK:
        if objective == "max_pedidos":
            M = float(np.sum(np.abs(val)) + 1.0)
            objective_value = M + val
            description = "Opcao priorizando maior quantidade de pedidos; empate por valor."
        else:
            objective_value = val
            description = "Opcao que maximiza o valor da carga."

        c = -objective_value
        A = np.vstack([w, v])
        constraints = LinearConstraint(
            A,
            lb=np.array([-np.inf, -np.inf]),
            ub=np.array([max_weight, max_volume]),
        )
        result = milp(
            c=c,
            integrality=np.ones(len(idx)),
            bounds=Bounds(0, 1),
            constraints=constraints,
            options={"time_limit": 120},
        )
        if result.success and result.x is not None:
            chosen = idx[np.where(result.x > 0.5)[0]].tolist()
            return chosen, "MILP: " + description

    if objective == "max_pedidos":
        score = 1.0 / (0.5 * np.maximum(w / max_weight, 1e-9) + 0.5 * np.maximum(v / max_volume, 1e-9))
        description = "Fallback guloso: mais pedidos primeiro."
    else:
        score = val / (0.5 * np.maximum(w / max_weight, 1e-9) + 0.5 * np.maximum(v / max_volume, 1e-9))
        description = "Fallback guloso: valor por recurso."

    order_idx = idx[np.argsort(-score)]
    chosen = []
    used_w = 0.0
    used_v = 0.0
    for i in order_idx:
        if used_w + weights[i] <= max_weight + 1e-9 and used_v + volumes[i] <= max_volume + 1e-9:
            chosen.append(int(i))
            used_w += weights[i]
            used_v += volumes[i]
    return chosen, description


def build_solution(eligible, chosen_idx, axis, vehicle, margin, routes, vehicle_data, solver_msg):
    chosen = eligible.iloc[chosen_idx].copy()
    if not chosen.empty:
        chosen["lucro_estimado"] = chosen["valor"] * margin / 100.0
        chosen["ordem_descarga"] = chosen["cidade"].apply(lambda c: route_position(c, routes[axis]))
        chosen = chosen.sort_values(["ordem_descarga", "cidade", "pedido"])

    excluded = eligible.drop(index=chosen.index, errors="ignore").copy()
    if not excluded.empty:
        excluded["motivo"] = excluded.apply(
            lambda row: explain_exclusion(row, chosen, vehicle_data["peso_kg"], vehicle_data["volume_m3"]),
            axis=1,
        )

    return {
        "chosen": chosen,
        "excluded": excluded,
        "solver": solver_msg,
        "axis": axis,
        "vehicle": vehicle,
        "margin": margin,
        "route": routes[axis],
        "vehicle_data": vehicle_data,
    }


def explain_exclusion(row, chosen, max_weight, max_volume):
    if row["peso_kg"] > max_weight + 1e-9:
        return "Excede sozinho o limite de peso."
    if row["volume_m3"] > max_volume + 1e-9:
        return "Excede sozinho o limite de volume."
    cw = chosen["peso_kg"].sum() if not chosen.empty else 0
    cv = chosen["volume_m3"].sum() if not chosen.empty else 0
    pw = cw + row["peso_kg"]
    pv = cv + row["volume_m3"]
    if pw > max_weight + 1e-9 and pv > max_volume + 1e-9:
        return "Nao cabe: ultrapassaria peso e volume."
    if pw > max_weight + 1e-9:
        return "Nao cabe: limite de peso."
    if pv > max_volume + 1e-9:
        return "Nao cabe: limite de volume."
    return "Ficou fora da combinacao otima."


# ============================================================
# INTERFACE
# ============================================================

def main():
    st.set_page_config(page_title="Montagem de Carga", layout="wide")
    st.title("Montagem automatica de carga por eixo")
    st.caption("Otimizacao 0/1 com duas restricoes simultaneas: peso (kg) e volume (m3).")

    if not SCIPY_OK:
        st.warning("scipy nao instalado: usando fallback guloso. Instale com pip install scipy.")

    routes, vehicles, route_file = load_route_file()

    st.sidebar.header("Configuracao")
    axis = st.sidebar.selectbox("Eixo de entrega", list(routes.keys()))
    vehicle = st.sidebar.selectbox("Veiculo", list(vehicles.keys()))
    vehicle_data = vehicles[vehicle]
    margin = st.sidebar.number_input("Margem estimada (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)

    st.sidebar.divider()
    st.sidebar.write("**Capacidade do veiculo**")
    st.sidebar.metric("Peso maximo", f"{vehicle_data['peso_kg']:,.0f} kg")
    st.sidebar.metric("Volume maximo", f"{vehicle_data['volume_m3']:.4f} m3")

    st.sidebar.write("**Sequencia do eixo**")
    for i, city in enumerate(routes[axis], start=1):
        st.sidebar.write(f"{i}. {city}")

    if route_file:
        st.sidebar.caption(f"Arquivo de rota: {route_file.name}")

    semanas_disponiveis = []
    for i in range(1, 5):
        if locate_file(f"Pedidos_Filtrados_Semana_{i}_Anonimizado*.csv"):
            semanas_disponiveis.append(f"Semana {i}")
    if not semanas_disponiveis:
        semanas_disponiveis = ["Todas"]

    semana = st.sidebar.selectbox("Semana para o resumo", semanas_disponiveis, index=0)
    load_data._semana = semana

    try:
        orders, audit, ranking = load_data()
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        st.info(f"Confira a pasta: {DATA}")
        st.stop()

    eligible = filter_axis(orders, routes[axis])
    if eligible.empty:
        st.warning("Nenhum pedido encontrado para este eixo.")
        st.stop()

    st.info(f"{len(eligible)} pedidos encontrados para {axis} em {semana}, apos os filtros.")

    total_items = int(audit.get("itens_lidos", 0))
    matched_items = int(audit.get("itens_com_match", 0))

    suspect_zero = (
        total_items == 0
        or matched_items == 0
        or eligible["peso_kg"].sum() == 0
        or eligible["volume_m3"].sum() == 0
    )

    with st.expander("Diagnostico dos dados", expanded=suspect_zero):
        st.write("Arquivos utilizados:")
        st.write(audit.get("arquivos", []))
        st.write("Mapeamento das colunas:")
        st.dataframe(pd.DataFrame(audit["colunas"]), use_container_width=True)
        st.write(
            f"Itens interpretados: {total_items} | "
            f"Com match no ranking: {matched_items} | "
            f"Sem match: {audit.get('itens_sem_match', 0)}"
        )

        st.write("**Amostra do ranking (peso/volume crus):**")
        if audit.get("amostra_ranking_peso"):
            st.dataframe(pd.DataFrame(audit["amostra_ranking_peso"]), use_container_width=True)

        st.write("**Amostra dos itens crus do CSV de pedidos:**")
        if audit.get("amostra_itens"):
            for s in audit["amostra_itens"]:
                st.code(f"ARQUIVO: {s['arquivo']}\nRAW: {s['raw']}\nPARSEADO: {s['parseado']}")

        st.write("**Colunas do ranking:**", list(ranking.columns))
        st.dataframe(ranking.head(3), use_container_width=True)

    if total_items == 0 or matched_items == 0:
        st.error("Os itens nao foram cruzados com o ranking. Veja a amostra acima.")

    if st.button("MONTAR CARGAS AUTOMATICAMENTE", type="primary"):
        idx_qtd, msg_qtd = solve_knapsack(eligible, vehicle_data["peso_kg"], vehicle_data["volume_m3"], objective="max_pedidos")
        idx_lucro, msg_lucro = solve_knapsack(eligible, vehicle_data["peso_kg"], vehicle_data["volume_m3"], objective="max_lucro")

        opcoes = {
            "Mais pedidos": build_solution(eligible, idx_qtd, axis, vehicle, margin, routes, vehicle_data, msg_qtd),
            "Maior lucro": build_solution(eligible, idx_lucro, axis, vehicle, margin, routes, vehicle_data, msg_lucro),
        }
        st.session_state["opcoes"] = opcoes
        st.session_state["opcao_selecionada"] = "Mais pedidos"

    opcoes = st.session_state.get("opcoes")
    if not opcoes:
        st.markdown(
            """
            ### Como funciona
            1. Le os pedidos semanais em `data/`.
            2. Remove cancelados, retiradas no balcao e entregas urbanas fora do escopo.
            3. Usa o codigo do produto para cruzar com o ranking.
            4. Calcula peso e cubagem de cada item e soma por pedido.
            5. Converte m2 para caixas quando aplicavel.
            6. Resolve o problema 0/1 respeitando peso e volume.
            7. Gera o romaneio na sequencia de descarga.
            """
        )
        return

    opcao = st.radio(
        "Estrategia da carga:",
        list(opcoes.keys()),
        index=list(opcoes.keys()).index(st.session_state.get("opcao_selecionada", "Mais pedidos")),
        horizontal=True,
        key="seletor_opcao",
    )
    st.session_state["opcao_selecionada"] = opcao

    result = opcoes[opcao]
    chosen = result["chosen"]
    excluded = result["excluded"]
    vd = result["vehicle_data"]

    st.success(f"{opcao}: {result['solver']}")

    if chosen.empty:
        st.error("Nenhum pedido coube na capacidade selecionada.")
        return

    total_value = chosen["valor"].sum()
    total_weight = chosen["peso_kg"].sum()
    total_volume = chosen["volume_m3"].sum()
    total_profit = chosen["lucro_estimado"].sum()

    weight_pct = total_weight / vd["peso_kg"] * 100
    volume_pct = total_volume / vd["volume_m3"] * 100

    st.divider()
    st.subheader("Resultado da carga")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pedidos carregados", len(chosen))
    c2.metric("Valor da carga", money(total_value))
    c3.metric("Lucro estimado", money(total_profit))
    c4.metric("Peso", f"{total_weight:,.1f} / {vd['peso_kg']:,.0f} kg")

    c5, c6 = st.columns(2)
    with c5:
        st.write(f"**Ocupacao de peso: {weight_pct:.1f}%**")
        st.progress(min(max(weight_pct / 100, 0), 1))
    with c6:
        st.write(f"**Ocupacao de volume: {volume_pct:.1f}%**")
        st.progress(min(max(volume_pct / 100, 0), 1))

    st.subheader("Romaneio")
    display = chosen[["ordem_descarga", "pedido", "cidade", "valor", "peso_kg", "volume_m3", "lucro_estimado"]].copy()
    display.columns = ["Seq.", "Pedido", "Cidade", "Valor", "Peso (kg)", "Volume (m3)", "Lucro estimado"]
    display["Valor"] = display["Valor"].map(money)
    display["Lucro estimado"] = display["Lucro estimado"].map(money)
    display["Peso (kg)"] = display["Peso (kg)"].map(lambda x: f"{x:,.1f}")
    display["Volume (m3)"] = display["Volume (m3)"].map(lambda x: f"{x:.4f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Sequencia de descarga")
    for i, city in enumerate(result["route"], start=1):
        loaded_here = [str(p) for _, row in chosen.iterrows() if city_matches(row["cidade"], city) for p in [row["pedido"]]]
        if loaded_here:
            st.write(f"**{i}. {city}** - pedidos: " + ", ".join(loaded_here))
        else:
            st.write(f"**{i}. {city}** - sem pedido nesta carga")

    st.subheader("Pedidos que ficaram fora")
    if excluded.empty:
        st.success("Todos os pedidos elegiveis foram carregados.")
    else:
        ex_display = excluded[["pedido", "cidade", "valor", "peso_kg", "volume_m3", "motivo"]].copy()
        ex_display["valor"] = ex_display["valor"].map(money)
        ex_display["peso_kg"] = ex_display["peso_kg"].map(lambda x: f"{x:,.1f}")
        ex_display["volume_m3"] = ex_display["volume_m3"].map(lambda x: f"{x:.4f}")
        st.dataframe(ex_display, use_container_width=True, hide_index=True)

    st.subheader("Comparacao das opcoes")
    comparacao = []
    for nome, sol in opcoes.items():
        ch = sol["chosen"]
        comparacao.append({
            "Opcao": nome,
            "Pedidos": len(ch),
            "Valor": ch["valor"].sum() if not ch.empty else 0,
            "Lucro estimado": ch["lucro_estimado"].sum() if not ch.empty else 0,
            "Peso (kg)": ch["peso_kg"].sum() if not ch.empty else 0,
            "Volume (m3)": ch["volume_m3"].sum() if not ch.empty else 0,
            "Ocupacao peso (%)": ch["peso_kg"].sum() / vd["peso_kg"] * 100 if not ch.empty else 0,
            "Ocupacao volume (%)": ch["volume_m3"].sum() / vd["volume_m3"] * 100 if not ch.empty else 0,
        })
    comp_df = pd.DataFrame(comparacao)
    comp_df["Valor"] = comp_df["Valor"].map(money)
    comp_df["Lucro estimado"] = comp_df["Lucro estimado"].map(money)
    comp_df["Peso (kg)"] = comp_df["Peso (kg)"].map(lambda x: f"{x:,.1f}")
    comp_df["Volume (m3)"] = comp_df["Volume (m3)"].map(lambda x: f"{x:.4f}")
    comp_df["Ocupacao peso (%)"] = comp_df["Ocupacao peso (%)"].map(lambda x: f"{x:.1f}%")
    comp_df["Ocupacao volume (%)"] = comp_df["Ocupacao volume (%)"].map(lambda x: f"{x:.1f}%")
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.subheader("Auditoria dos dados")
    a1, a2, a3 = st.columns(3)
    a1.metric("Pedidos removidos", len(audit["removidos"]))
    a2.metric("Conversoes m2 -> caixas", len(audit["conversoes"]))
    a3.metric("Avisos", len(audit["avisos"]))

    with st.expander("Ver correcoes e filtros aplicados"):
        if audit["removidos"]:
            st.write("**Pedidos removidos:**")
            st.dataframe(pd.DataFrame(audit["removidos"]), use_container_width=True, hide_index=True)
        if audit["conversoes"]:
            st.write("**Conversoes:**")
            st.dataframe(pd.DataFrame(audit["conversoes"]), use_container_width=True, hide_index=True)
        if audit["avisos"]:
            st.write("**Avisos:**")
            for warning in audit["avisos"]:
                st.warning(warning)
        st.write("**Arquivos utilizados:**", ", ".join(audit["arquivos"]))
        st.write(f"**Ranking:** {audit['ranking']}")

    with st.expander("Auditoria matematica"):
        st.write(f"**Peso:** {total_weight:.2f} kg <= {vd['peso_kg']:.2f} kg")
        st.write(f"**Volume:** {total_volume:.4f} m3 <= {vd['volume_m3']:.4f} m3")
        st.write(f"**Valor maximizado:** {money(total_value)}")
        st.write(f"**Lucro estimado:** {money(total_profit)} (margem {result['margin']:.1f}%)")

    csv = chosen[["ordem_descarga", "pedido", "cidade", "valor", "peso_kg", "volume_m3", "lucro_estimado"]].to_csv(
        index=False, sep=";", encoding="utf-8-sig"
    )
    st.download_button("Baixar romaneio CSV", data=csv, file_name="romaneio.csv", mime="text/csv")


if __name__ == "__main__":
    main()