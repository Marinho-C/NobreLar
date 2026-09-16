import pandas as pd
import numpy as np
import os

def otimizar_mochila_carga(df_pedidos, peso_max=4000.0, volume_max=22.0):
    if df_pedidos.empty:
        return df_pedidos, 0.0, 0.0

    df_ordenado = df_pedidos.sort_values(by='peso_total', ascending=False).copy()
    
    pedidos_alocados = []
    peso_acumulado = 0.0
    volume_acumulado = 0.0

    for _, row in df_ordenado.iterrows():
        p = float(row['peso_total'])
        v = float(row['volume_total'])

        if (peso_acumulado + p <= peso_max) and (volume_acumulado + v <= volume_max):
            pedidos_alocados.append(row)
            peso_acumulado += p
            volume_acumulado += v

    df_res = pd.DataFrame(pedidos_alocados) if pedidos_alocados else df_pedidos.head(3)
    return df_res, peso_acumulado, volume_acumulado

def processar_todos_os_eixos(peso_max_padrao=4000.0, volume_max_padrao=22.0):
    """
    Processa os 5 eixos simultaneamente, calcula o aproveitamento,
    diagnostica o gargalo (Peso vs Volume) e ajuda na alocação da frota.
    """
    path_pedidos = os.path.join('data', 'Pedidos_Limpos.csv')
    if not os.path.exists(path_pedidos):
        return None
    
    df_pedidos = pd.read_csv(path_pedidos, sep=None, engine='python', encoding='utf-8')
    df_pedidos.columns = [col.strip().lower() for col in df_pedidos.columns]
    
    mapeamento_eixos = {
        "1": ["BURITI DOS MONTES"],
        "2": ["SUCESSO", "TAMBORIL", "NOVA RUSSAS", "FAZENDA"],
        "3": ["INDEPENDENCIA"],
        "4": ["IPAPORANGA", "PORANGA", "ARARENDA"],
        "5": ["NOVO ORIENTE", "REALEJO", "SANTANA", "MONTE NEBO", "SANTO ANDRE", "QUITERIANOPOLIS"]
    }
    
    relatorio_geral = []

    for eixo, cidades in mapeamento_eixos.items():
        if 'cidade' in df_pedidos.columns:
            df_pedidos['cidade_limpa'] = df_pedidos['cidade'].astype(str).str.strip().str.upper()
            df_filtrado = df_pedidos[df_pedidos['cidade_limpa'].isin(cidades)].copy()
        else:
            df_filtrado = df_pedidos.head(5).copy()

        if df_filtrado.empty:
            df_filtrado = df_pedidos.head(5).copy()

        resumo_pedidos = []
        for index, row in df_filtrado.iterrows():
            valor = float(row.get('valor_numerico', row.get('valor do pedido', 150.0)))
            cidade = str(row.get('cidade', 'INTERIOR'))
            id_pedido = str(row.get('pedido', index))
            
            peso_estimado = round(max(valor * 0.04, 20.0), 2)  
            volume_estimado = round(max(valor * 0.00025, 0.15), 2) 
            
            resumo_pedidos.append({
                'pedido': id_pedido,
                'cidade': cidade,
                'valor': valor,
                'peso_total': peso_estimado,
                'volume_total': volume_estimado
            })

        df_candidatos = pd.DataFrame(resumo_pedidos)
        df_otimizado, p_total, v_total = otimizar_mochila_carga(df_candidatos, peso_max_padrao, volume_max_padrao)
        
        # Cálculo de Ocupação Percentual
        p_perc = (p_total / peso_max_padrao) * 100
        v_perc = (v_total / volume_max_padrao) * 100
        
        # Diagnóstico de Gargalo (Peso x Volume)
        if p_perc >= v_perc:
            gargalo = "Limitado por PESO (kg)"
            perfil = "Carga Densa / Pisos e Revestimentos"
        else:
            gargalo = "Limitado por VOLUME (m³)"
            perfil = "Carga Volumosa / Materiais Leves"

        relatorio_geral.append({
            'eixo': f"Eixo {eixo}",
            'cidades': ", ".join(cidades),
            'pedidos_alocados': len(df_otimizado),
            'peso_total': round(p_total, 2),
            'ocupacao_peso_perc': round(p_perc, 1),
            'volume_total': round(v_total, 2),
            'ocupacao_volume_perc': round(v_perc, 1),
            'gargalo': gargalo,
            'perfil': perfil
        })

    return pd.DataFrame(relatorio_geral)

# Mantém a compatibilidade com a função individual usada na tela web
def processar_etapa_2(eixo_escolhido="1"):
    df_geral = processar_todos_os_eixos()
    # Retorna uma simulação isolada para o eixo escolhido na interface web
    path_pedidos = os.path.join('data', 'Pedidos_Limpos.csv')
    df_pedidos = pd.read_csv(path_pedidos, sep=None, engine='python', encoding='utf-8')
    df_pedidos.columns = [col.strip().lower() for col in df_pedidos.columns]
    
    mapeamento = {
        "1": ["BURITI DOS MONTES"],
        "2": ["SUCESSO", "TAMBORIL", "NOVA RUSSAS", "FAZENDA"],
        "3": ["INDEPENDENCIA"],
        "4": ["IPAPORANGA", "PORANGA", "ARARENDA"],
        "5": ["NOVO ORIENTE", "REALEJO", "SANTANA", "MONTE NEBO", "SANTO ANDRE", "QUITERIANOPOLIS"]
    }
    cidades = mapeamento.get(str(eixo_escolhido), [])
    df_f = df_pedidos[df_pedidos['cidade'].astype(str).str.strip().str.upper().isin(cidades)].copy()
    if df_f.empty:
        df_f = df_pedidos.head(10).copy()
        
    resumo = []
    for idx, r in df_f.iterrows():
        val = float(r.get('valor_numerico', 100.0))
        resumo.append({
            'pedido': str(r.get('pedido', idx)),
            'cidade': str(r.get('cidade', 'LOC')),
            'valor': val,
            'peso_total': round(max(val * 0.04, 20.0), 2),
            'volume_total': round(max(val * 0.00025, 0.15), 2)
        })
    df_res = pd.DataFrame(resumo)
    df_otimizado, _, _ = otimizar_mochila_carga(df_res)
    if not df_otimizado.empty:
        df_otimizado = df_otimizado.sort_values(by=['cidade', 'pedido']).reset_index(drop=True)
        df_otimizado['ordem_descarga'] = range(1, len(df_otimizado) + 1)
    return df_otimizado

if __name__ == '__main__':
    print("=== RELATÓRIO GLOBAL DOS 5 EIXOS (MULTIEIXO & GARGALOS) ====")
    print(processar_todos_os_eixos())