import pandas as pd
import os

def processar_etapa_2(eixo_escolhido):
    caminho_pedidos = os.path.join('data', 'Pedidos_Limpos.csv')
    caminho_ranking = os.path.join('data', 'Ranking_Top85_Materiais.csv')

    # Lê os arquivos forçando o separador correto ';'
    df_pedidos = pd.read_csv(caminho_pedidos, sep=';', encoding='utf-8', low_memory=False)
    
    # O ranking está agrupado por colunas com ';'
    df_ranking = pd.read_csv(caminho_ranking, sep=';', encoding='utf-8')

    # Padroniza os nomes das colunas para minúsculas para evitar erro de maiúscula/minúscula
    df_pedidos.columns = [c.strip().lower() for c in df_pedidos.columns]
    df_ranking.columns = [c.strip().lower() for c in df_ranking.columns]

    # Simulação de cruzamento e cálculo de peso/volume total para o Eixo selecionado
    # Filtra pedidos de acordo com a cidade/logística correspondente ao eixo
    if 'cidade' in df_pedidos.columns:
        # Exemplo de filtro básico baseado no eixo
        df_filtrado = df_pedidos.head(10).copy() # Pegando uma amostra para teste inicial
    else:
        df_filtrado = df_pedidos.head(10).copy()

    # Atribuindo colunas fictícias de peso e volume para o teste da interface web
    df_filtrado['peso_total'] = 150.0
    df_filtrado['volume_total'] = 2.5

    return df_filtrado