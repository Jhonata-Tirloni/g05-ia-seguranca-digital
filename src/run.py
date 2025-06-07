import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from modules.get_llm_model import call_update_window
from functools import partial
from os import listdir
import torch


def carregar_modelo(model_path: str):
    try:
        # Força uso de CPU ou GPU com fallback seguro
        device = 0 if torch.cuda.is_available() else -1
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return pipeline(
            "text-generation", model=model, tokenizer=tokenizer, device=device
        )
    except Exception as e:
        messagebox.showerror("Erro ao carregar modelo", str(e))
        return None


# Função para processar o dataset
def processar_dataset(pipe):
    file_path = filedialog.askopenfilename(
        title="Selecione um arquivo CSV", filetypes=[("CSV files", "*.csv")]
    )
    if not file_path:
        return

    try:
        df = pd.read_csv(file_path, sep=";")
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao ler o arquivo: {e}")
        return

    classificacoes = []
    for col in df.columns:
        entrada = (
            f"Classifique a coluna '{col}' com uma fonte, descrição e tipo de crime"
        )
        try:
            resposta = pipe(entrada, max_new_tokens=256)[0]["generated_text"]
        except Exception as e:
            resposta = f"Erro: {e}"
        classificacoes.append(resposta)

    resultado_df = pd.DataFrame(
        {"Coluna": df.columns, "Classificação LLM": classificacoes}
    )

    output_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Salvar resultado",
    )
    if output_path:
        resultado_df.to_csv(output_path, sep=";", index=False)
        messagebox.showinfo("Sucesso", f"Arquivo salvo em:\n{output_path}")


# Inicialização segura do modelo
pipe = None
if len(listdir(r"src/llm_model")) > 1:
    modelPath = r"src/llm_model"
    pipe = carregar_modelo(modelPath)
else:
    messagebox.showwarning(
        title="Atenção",
        message="Não foi encontrado nenhum modelo na pasta /src/models.\
                Baixe algum através da opção Menu > Atualizar na barra superior.",
    )

# Interface gráfica
root = tk.Tk()
root.title("Classificador de Dataset")
root.geometry("500x300")
root.resizable(False, False)

label_info = tk.Label(
    root,
    text="Para tratar um dataset, envie um arquivo no botão abaixo",
    font=("Arial", 12),
    wraplength=480,
    justify="center",
)
label_info.pack(pady=(40, 20))

btn_enviar = tk.Button(
    root,
    text="Inserir arquivo",
    font=("Arial", 12),
    width=20,
    command=lambda: (
        processar_dataset(pipe)
        if pipe
        else messagebox.showwarning(
            "Modelo não carregado", "Carregue um modelo antes de continuar."
        )
    ),
)
btn_enviar.pack()

menu_bar = tk.Menu()
model_menu = tk.Menu(menu_bar, tearoff=False)
model_menu.add_command(
    label="Atualizar/baixar modelo",
    accelerator="Ctrl+U",
    command=partial(call_update_window, root),
)
menu_bar.add_cascade(menu=model_menu, label="Modelo")
root.config(menu=menu_bar)

root.mainloop()
