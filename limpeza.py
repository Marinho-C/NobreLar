import os
import pandas as pd


def limpar_pedidos():
    caminho = os.path.join("data", "DADOS DE ENTREGAS  - Vendas_Faturamento_Entregas.csv")
    if not os.path.exists(caminho):
        print(f"Arquivo nao encontrado: {caminho}")
        return None

    df = pd.read_csv(caminho, sep=None, engine="python", encoding="utf-8-sig")
    print(f"Registros originais: {len(df)}")

    df.columns = [str(c).strip().lower() for c in df.columns]

    if "situacao" in df.columns:
        termos = "cancelado|retirada|balcao|balcão"
        df = df[~df["situacao"].astype(str).str.lower().str.contains(termos, na=False)]

    if "cidade" in df.columns:
        df = df[~df["cidade"].astype(str).str.strip().str.lower().isin(["crateus", "crateús"])]

    if "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"], errors="coerce")
        df = df.dropna(subset=["data"])

    col_valor = None
    for c in ["valor do pedido", "valor pedido", "valor total", "valor"]:
        if c in df.columns:
            col_valor = c
            break

    if col_valor:
        df["valor_numerico"] = (
            df[col_valor]
            .astype(str)
            .str.replace("R$", "", regex=False)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.strip()
        )
        df["valor_numerico"] = pd.to_numeric(df["valor_numerico"], errors="coerce").fillna(0.0)

    print(f"Registros apos limpeza: {len(df)}")

    # PADRAO: salvar SEMPRE com ';' e utf-8-sig
    saida = os.path.join("data", "Pedidos_Limpos.csv")
    df.to_csv(saida, index=False, encoding="utf-8-sig", sep=";")
    print(f"Salvo: {saida}")
    return df


if __name__ == "__main__":
    limpar_pedidos()