import pandas as pd
import requests


# 1. Extrair texto do CSV
def extract_text_from_csv(file_path, max_rows=10):
    df = pd.read_csv(file_path, sep=";")
    df = df.head(max_rows)
    return df.to_string(index=False)


# 2. Mandar para o modelo via Open WebUI API
def send_to_model(prompt_text):
    url = "http://localhost:3000/api/chat/completions"
    header = {
        "Content-Type": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6ImU0Y2I2NmNiLWFmZDYtNDQ5My1iNzIwLTI5NDFmMTQ3NDQyMiJ9.RaG4W4gqxRdUhmUTfYplTKNDnRhzjosTG18X_46m7T4",
    }
    payload = {
        "model": "llama3:instruct",
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
    }
    response = requests.post(url, headers=header, json=payload)
    return response.json()


# Exemplo de uso
csv_text = extract_text_from_csv(
    "/Users/jhonatatirloni/Desktop/g05-ia-seguranca-digital/src/data/processed/df_text_news_short.csv"
)
prompt = f"""
1. Identificar se o texto descreve um **crime ocorrido** (responda com S para sim, ou N para não).
2. Se for um crime ocorrido (S), classificar o tipo de crime em uma das seguintes categorias:

* Cybercrime
* Fraude
* Golpe bancário
* Outros

Para cada texto recebido, retorne exatamente uma linha com as seguintes três colunas, separadas por ponto e vírgula (;):

                
<index do texto original>;flg_golpe;tipo


* Substitua <index do texto original> pelo numero da linha do texto que está classificando.
* flg_golpe: S se for um crime ocorrido, N caso contrário.
* tipo: uma das quatro categorias acima, ou deixe em branco se flg_golpe = N.

Importante:

* **Não** adicione cabeçalhos, comentários ou explicações extras.
* **Mantenha a ordem das entradas.**
* Retorne **somente** os dados classificados.
dados: {csv_text}"""
resposta = send_to_model(prompt)

model_text_response = resposta["choices"][0]["message"]["content"]

linhas = model_text_response.strip().split("\n")
dados = [linha.split(";") for linha in linhas]
df_resposta = pd.DataFrame(dados)

df_resposta.to_csv(
    "/Users/jhonatatirloni/Desktop/g05-ia-seguranca-digital/src/data/processed/teste_ollama.csv",
    sep=";",
    index=False,
)
