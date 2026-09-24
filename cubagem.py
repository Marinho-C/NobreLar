import re
import os
import numpy as np
import pandas as pd


def parse_numero_ptbr(valor):
    if valor is None:
        return np.nan
    if isinstance(valor, (int, float)):
        return float(valor) if not pd.isna(valor) else np.nan

    texto = str(valor).strip()
    if texto == "" or texto.lower() in ("nan", "none", "-"):
        return np.nan

    texto = texto.replace("≈", "").strip()
    texto = re.sub(r"\(.*?\)", "", texto)
    texto = re.sub(r"/\s*\w+", "", texto)
    texto = re.sub(r"[^\d.,\-]", "", texto).strip()

    if texto == "":
        return np.nan

    if "," in texto:
        texto = texto.replace(".", "").replace(",", ".")

    try:
        return float(texto)
    except ValueError:
        return np.nan


def extrair_m2_por_caixa(texto_unidade):
    if not isinstance(texto_unidade, str):
        return None
    m = re.search(r"=\s*([\d\.,]+)\s*m", texto_unidade, flags=re.IGNORECASE)
    if m:
        return parse_numero_ptbr(m.group(1))
    return None


def _detectar_coluna(colunas, candidatas):
    cols_lower = {c.lower().strip(): c for c in colunas}
    for cand in candidatas:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def limpar_ranking_materiais(path=None):
    if path is None:
        path = os.path.join("data", "Ranking_Top85_Materiais.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ranking nao encontrado: {path}")

    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    col_codigo = _detectar_coluna(df.columns, ["codigo", "código", "cod_produto", "cod"])
    col_desc = _detectar_coluna(df.columns, ["descricao", "descrição", "produto"])
    col_unidade = _detectar_coluna(df.columns, ["unidade_venda", "unidade de venda", "unidade"])
    col_peso = _detectar_coluna(df.columns, ["peso_kg", "peso (kg)", "peso"])
    col_volume = _detectar_coluna(df.columns, ["volume_m3", "volume (m3)", "volume (m³)", "volume"])

    faltando = [n for n, c in [
        ("codigo", col_codigo), ("descricao", col_desc), ("unidade_venda", col_unidade),
        ("peso_kg", col_peso), ("volume_m3", col_volume),
    ] if c is None]
    if faltando:
        raise ValueError(
            f"Colunas ausentes no Ranking: {faltando}. Disponiveis: {list(df.columns)}"
        )

    registros = []
    for _, row in df.iterrows():
        descricao = str(row[col_desc])
        unidade_venda = str(row[col_unidade])
        eh_piso = ("piso" in descricao.lower()) or (
            "caixa" in unidade_venda.lower() and "m²" in unidade_venda.lower()
        )
        registros.append({
            "codigo": str(row[col_codigo]).strip(),
            "descricao": descricao,
            "unidade_venda": unidade_venda,
            "peso_kg_unidade": parse_numero_ptbr(row[col_peso]),
            "volume_m3_unidade": parse_numero_ptbr(row[col_volume]),
            "eh_piso": eh_piso,
            "m2_por_caixa": extrair_m2_por_caixa(unidade_venda) if eh_piso else np.nan,
        })

    return pd.DataFrame(registros)


if __name__ == "__main__":
    df = limpar_ranking_materiais()
    print(f"Ranking limpo: {len(df)} produtos")
    print(df[df["eh_piso"]].head())