from huggingface_hub import snapshot_download, configure_http_backend
import requests
import urllib3
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from functools import partial

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def backendFactory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session


configure_http_backend(backend_factory=backendFactory)


def confirm_update(root):
    response = messagebox.askyesno(
        "Confirmar update/download",
        "Você tem certeza que deseja prosseguir? Esta ação\
                                    irá realizar o download de um novo modelo.\
                                    Se já existir um em /src/models ele será substituido e,\
                                    caso não, será salvo neste mesmo caminho.",
    )
    if response:
        request_auth_key(root)
    elif response is False:
        root.destroy()


def request_auth_key(root):
    auth_window = tk.Toplevel(root)
    auth_window.title("Auth Key")
    auth_window.geometry("520x320")

    auth_label = tk.Label(
        auth_window, text="Insira sua key do Hugging Face:", font=("Helvetica", 12)
    )
    auth_label.pack(pady=10)
    auth_key_value = tk.StringVar()
    auth_key_entry = tk.Entry(auth_window, show="*", textvariable=auth_key_value)
    auth_key_entry.pack(pady=5)

    modeluri_label = tk.Label(
        auth_window,
        text="Insira o caminho do modelo no Hugging Face:",
        font=("Helvetica", 12),
    )
    modeluri_label.pack(pady=5)

    modeluri_value = tk.StringVar()
    modeluri_text = tk.Entry(auth_window, textvariable=modeluri_value)
    modeluri_text.pack(pady=5)

    def get_model_key_value():
        authkey_value = auth_key_value.get()
        modeluriValue = modeluri_value.get()

        return download_model(authkey_value, modeluriValue)

    submit_button = tk.Button(auth_window, text="Submit", command=get_model_key_value)
    submit_button.pack(pady=10)


def download_model(auth_token, modeluri):
    print(auth_token)
    print(modeluri)
    try:
        snapshot_download(
            repo_id=modeluri,
            use_auth_token=auth_token,
            local_dir=r"./src/llm_model",
        )
        messagebox.showinfo(
            "Sucesso",
            "Modelo atualizado/baixado com sucesso!\
                            É recomendado reiniciar o aplicativo para garantir o funcionamento.",
        )
    except Exception as e:
        messagebox.showwarning("Erro inesperado", message=str(e))


def call_update_window(root):
    # Obter a posição da janela principal
    x = root.winfo_x()
    y = root.winfo_y()

    new_window = tk.Toplevel(root)

    new_window.geometry(f"+{x}+{y}")

    new_window.resizable(False, False)

    info_label = tk.Label(
        new_window,
        text="Atenção: Essa ação irá baixar um novo modelo para a aplicação.\
                            Pode ser um download de +2.5gb, e substituirá o que tem na /src/models.",
        font=("Helvetica", 12),
        wraplength=250,
        justify="center",
    )
    info_label.pack(pady=20)

    exit_button = tk.Button(
        new_window,
        text="Atualizar/baixar modelo",
        command=partial(confirm_update, new_window),
    )
    exit_button.pack(padx=10, pady=10)
