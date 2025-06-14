import pandas as pd
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


# === 1. Envio para o modelo ===
def send_to_model(prompt_text):
    url = "http://localhost:3000/api/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjI2NGExZTkyLWViNGEtNDNiNC1hZjJlLTA3ZDQ4ODZmYmIyYyJ9.8NTqGLaQMSGvJAWoLCiehMaZH277LXuE7MVMd1kmaB4",  # substitua pelo seu token
    }
    payload = {
        "model": "deepseek-r1:8b",
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 100000,
        "stream": False,
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=999999)
        return response.json()
    except Exception as e:
        print("Erro de requisição:", e)
        return None


# === 2. Construção do prompt ===
def build_prompt(batch_df):
    linhas = "\n".join(f"{idx}; {row['text']}" for idx, row in batch_df.iterrows())
    return f"""
Classifique os seguintes textos conforme instruções abaixo.
Cada texto segue o formato, com duas colunas separadas por pipe (|): 
<index>|<text>

Instruções:
1. Identificar se o texto descreve um **crime ocorrido** (responda com S para sim, ou N para não).
2. Se for um crime ocorrido (S), classificar o tipo de crime em uma das seguintes categorias:

*Golpe da troca de cartão;
*Golpe do Cartão Clonado;
*Golpe no whatsapp;
*Golpe falsa central;
*Golpe do PIX;
*Golpe do CPF;

Para cada texto recebido, retorne exatamente uma linha com as seguintes três colunas, separadas por ponto e vírgula (;):
<index do texto original>;flg_golpe;tipo

* Substitua <index do texto original> pelo número da linha do texto que está classificando.
* flg_golpe: S se for um crime ocorrido, N caso contrário.
* tipo: uma das quatro categorias acima, ou deixe em branco se flg_golpe = N.

Importante:
* **Não** adicione cabeçalhos, comentários ou explicações extras.
* **Mantenha a ordem das entradas.**
* Retorne **somente** os dados classificados no formato solicitado acima.
* Se não der pra classificar em nenhuma das categorias, não pule, retorne as categorias em branco.
dados:
{linhas}
""".strip()


# === 3. Função que processa cada lote ===
def processar_lote(lote_df):
    prompt = build_prompt(lote_df)
    resposta = send_to_model(prompt)
    resultados = []

    if resposta and "choices" in resposta:
        try:
            conteudo = resposta["choices"][0]["message"]["content"]
            linhas = conteudo.strip().split("\n")
            for linha in linhas:
                partes = linha.strip().split(";")
                if len(partes) == 3:
                    resultados.append(partes)
        except Exception as e:
            print("Erro ao processar resposta:", e)

    return resultados


# === 4. Pipeline paralelizado ===
def pipeline_paralelo(path_csv, path_saida, batch_size=2, n_threads=4):
    df = pd.read_csv(path_csv, sep="|")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "original_index"}, inplace=True)

    resultados_finais = []
    aux = 0
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i : i + batch_size]
            futures.append(executor.submit(processar_lote, batch_df))

        for future in as_completed(futures):
            lote_resultado = future.result()
            resultados_finais.extend(lote_resultado)

            # Salvamento progressivo (opcional, seguro)
            pd.DataFrame(resultados_finais).to_csv(
                path_saida + f"teste_ollama_batched_{aux}.csv", sep=";", index=False
            )

            aux = aux + 1

    print(f"✅ Finalizado: {len(resultados_finais)} linhas classificadas.")
    return resultados_finais


# === 5. Executar ===
pipeline_paralelo(
    path_csv=r"C:\Users\jhont\Desktop\g05-ia-seguranca-digital\src\data\processed\bronze\extra_text.csv",
    path_saida=r"C:\Users\jhont\Desktop\g05-ia-seguranca-digital\src\data\stage",
    batch_size=2,  # seguro para seu M1
    n_threads=4,  # comece com 4, aumente se o uso de CPU permitir
)
