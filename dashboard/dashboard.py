import streamlit as st
import os

# ==============================================================================
# CÓDIGO DE DEPURAÇÃO DE ARQUIVOS
# ==============================================================================
st.set_page_config(layout="wide")
st.title("Depurando a Estrutura de Arquivos no Servidor Streamlit")

# Pega o diretório onde o script está rodando
try:
    diretorio_script = os.path.dirname(__file__)
    st.write(f"O script 'dashboard.py' está em: **{diretorio_script}**")
except NameError:
    st.write("Não foi possível determinar o diretório do script (normal em alguns ambientes).")

st.header("Listando todos os arquivos e pastas a partir da raiz do repositório:")

file_list = []
try:
    # O ponto "." significa o diretório atual/raiz do repositório
    for root, dirs, files in os.walk("."): 
        for name in files:
            path = os.path.join(root, name).replace("\\", "/") # Normaliza para barras de web
            try:
                size = os.path.getsize(path)
                file_list.append(f"Arquivo: {path}, Tamanho: {size} bytes")
            except OSError:
                file_list.append(f"Arquivo: {path}, Tamanho: Não foi possível ler (pode ser um ponteiro LFS)")
except Exception as e:
    st.error(f"Erro ao listar arquivos: {e}")

# Mostra a lista de arquivos encontrados
if file_list:
    st.code("\n".join(sorted(file_list)))
else:
    st.warning("Nenhum arquivo encontrado na listagem.")

st.info("Se os seus arquivos .pkl ou .csv aparecem com um tamanho muito pequeno (ex: 133 bytes), isso confirma o problema com Git LFS.")
st.info("Se os arquivos não aparecem na pasta 'dashboard', eles não foram enviados corretamente para o GitHub.")

# Interrompe a execução do resto do dashboard para focar na depuração
st.stop()