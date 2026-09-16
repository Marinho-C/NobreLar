from flask import Flask, render_template, request, jsonify
import os
from processamento import processar_etapa_2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/processar-carga', methods=['POST'])
def processar_carga():
    try:
        dados = request.json
        eixo_escolhido = dados.get('eixo', '1')
        data_viagem = dados.get('data')

        # 1. Executa o motor passando o eixo escolhido
        df_res = processar_etapa_2(eixo_escolhido)
        
        if df_res is None or df_res.empty:
            return jsonify({"status": "erro", "mensagem": "Nenhum pedido encontrado para este Eixo."})

        # 2. Soma os totais reais daquela rota
        peso_total_alocado = df_res['peso_total'].sum()
        volume_total_alocado = df_res['volume_total'].sum()

        # Limites reais do caminhão padrão (ex: Caminhão Médio = 4000 kg e 22 m³)
        peso_maximo_veiculo = 4000.0  
        volume_maximo_veiculo = 22.0  

        pct_peso = (peso_total_alocado / peso_maximo_veiculo) * 100
        pct_volume = (volume_total_alocado / volume_maximo_veiculo) * 100

        pedidos_lista = df_res.to_dict(orient='records')

        return jsonify({
            "status": "sucesso",
            "mensagem": f"Carga otimizada gerada para o Eixo {eixo_escolhido} (Data: {data_viagem})!",
            "peso_total": f"{peso_total_alocado:.2f} kg",
            "volume_total": f"{volume_total_alocado:.2f} m³",
            "ocupacao_peso": f"{pct_peso:.1f}%",
            "ocupacao_volume": f"{pct_volume:.1f}%",
            "pedidos": pedidos_lista
        })

    except Exception as e:
        return jsonify({"status": "erro", "mensagem": str(e)})

if __name__ == '__main__':
    app.run(debug=True)