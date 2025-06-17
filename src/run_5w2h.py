import pandas as pd
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


# === 1. Envio para o modelo ===
def send_to_model(prompt_text):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.3:70b",
        "prompt": prompt_text,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 6080,
        "stream": False,
    }
    try:
        response = requests.post(url, json=payload, timeout=999999)
        return response.json()
    except Exception as e:
        print("❌ Erro de requisição:", e)
        return None


# === 2. Construção do prompt ===
def build_prompt(batch_df):
    linhas = "\n".join(
        f"{row['original_index']}|{row['text']}|{row['flg_crime']}|{row['cat_crime']}"
        for _, row in batch_df.iterrows()
    )
    return f"""
Você receberá entradas no seguinte formato, com colunas separadas por pipe (|):
index|text|flg_crime|cat_crime

Cada linha representa um relato de possível cibercrime bancário.

Tarefa:
Extraia as informações do campo text utilizando a metodologia 5W1H, conforme as instruções abaixo.

Perguntas (5W1H):
Quem (who): Quem foi a vítima? Quem foi o golpista?
O quê (what): O que aconteceu? Tipo de golpe e prejuízo, se houver.
Quando (when): Quando o golpe ocorreu?
Onde (where): Onde o golpe ocorreu (meio ou canal usado)?
Por quê (why): Por que o golpe funcionou? Qual vulnerabilidade foi explorada?
Como (how): Como o golpe foi executado? Etapas ou método.

Formato de saída esperado:
Para cada linha, gere exatamente uma nova linha com os campos separados por pipe (|) **sem espaços**, com as respostas das perguntas do método 5W2H,
salvas nas seguintes colunas correspondentes:
<index>|quem|o_que|quando|onde|por_que|como

Regras obrigatórias:
- Use apenas o campo text como fonte das informações.
- Mantenha o campo index original no início da linha.
- Se alguma informação não estiver claramente presente, deixe o campo vazio, mas mantenha a estrutura com "|"
- Não adicione comentários, títulos ou cabeçalhos.
- Não invente informações.

Entradas:
{linhas}
""".strip()


# === 3. Função que processa cada lote ===
def processar_lote(lote_df):
    prompt = build_prompt(lote_df)
    resposta = send_to_model(prompt)
    resultados = []

    if resposta and "response" in resposta:
        try:
            conteudo = resposta["response"]
            linhas = conteudo.strip().split("\n")
            for linha in linhas:
                partes = linha.strip().split("|")
                if len(partes) == 7:
                    resultados.append([p.strip() for p in partes])
                else:
                    print(f"⚠️ Linha com formato inválido: {linha}")
        except Exception as e:
            print("❌ Erro ao processar resposta:", e)
    else:
        print("⚠️ Resposta inesperada ou vazia:", resposta)

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

            # Salvamento progressivo (seguro em caso de interrupção)
            pd.DataFrame(resultados_finais).to_csv(
                path_saida + f"/saida_lote_{aux}.csv",
                sep="|",
                index=False,
                header=False,
            )
            aux += 1

    print(f"✅ Finalizado: {len(resultados_finais)} linhas classificadas.")
    return resultados_finais


# === 5. Executar ===
pipeline_paralelo(
    path_csv=r"C:\Users\jhont\Desktop\g05-ia-seguranca-digital\src\data\processed\gold\df_concat_bigthree.csv",
    path_saida=r"C:\Users\jhont\Desktop\g05-ia-seguranca-digital\src\data\stage",
    batch_size=1,  # Pode aumentar conforme a capacidade do sistema
    n_threads=4,  # Comece com 4, aumente se o uso de CPU permitir
)
