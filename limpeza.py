import pandas as pd
import os

def limpar_pedidos():
    caminho_arquivo = os.path.join('data', 'DADOS DE ENTREGAS  - Vendas_Faturamento_Entregas.csv')
    
    if not os.path.exists(caminho_arquivo):
        print(f"Erro: O arquivo {caminho_arquivo} não foi encontrado.")
        return None

    # 1. Carregar os dados
    df = pd.read_csv(caminho_arquivo)
    print(f"Total de registros originais: {len(df)}")

    # Padronizar nomes das colunas (tudo minúsculo e sem espaços)
    df.columns = [col.strip().lower() for col in df.columns]

    # 2. FILTRAGEM CORRETA (Mantém Crateús, remove apenas cancelados e retiradas)
    if 'situacao' in df.columns:
        # Remove APENAS se a situação contiver 'cancelado', 'retirada' ou 'balcão'
        termos_excluir = 'cancelado|retirada|balcão|balcao'
        df = df[~df['situacao'].astype(str).str.lower().str.contains(termos_excluir, na=False)]
    if 'cidade' in df.columns:
        cidades_ignoradas = ['crateus', 'crateús']
        df = df[~df['cidade'].astype(str).str.strip().str.lower().isin(cidades_ignoradas)]
        
    # 3. CORREÇÃO DE DATAS
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
        df = df.dropna(subset=['data'])

    # 4. CONVERSÃO DE VALORES PARA NÚMERO
    if 'valor do pedido' in df.columns:
        df['valor_numerico'] = (
            df['valor do pedido']
            .astype(str)
            .str.replace('R$', '', regex=False)
            .str.replace('.', '', regex=False)
            .str.replace(',', '.', regex=False)
            .str.strip()
        )
        df['valor_numerico'] = pd.to_numeric(df['valor_numerico'], errors='coerce').fillna(0.0)

    print(f"Total de registros após a limpeza: {len(df)}")
    
    # Salvar a base limpa
    caminho_saida = os.path.join('data', 'Pedidos_Limpos.csv')
    df.to_csv(caminho_saida, index=False, encoding='utf-8')
    print(f"Arquivo limpo salvo com sucesso em: {caminho_saida}")
    
    return df

if __name__ == '__main__':
    limpar_pedidos()