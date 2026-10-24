
import hashlib
import os
from pathlib import Path

import git
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# GIT_REPOSITORY = ["README.md", "FrontSamplePlugin.gif", "index.css", "index.js", "mockCRM.js", "plugin.html"]
GIT_REPOSITORY = ["index.js"]
CODING_FILES = [".js", ".css", ".py", ".html", ".go", ".rs", ".ipynb"]
USER = os.environ["USER"]
DIR_P = Path(f"/home/{USER}/Downloads")

def init_model() -> init_chat_model:
    return init_chat_model(
        model="gpt-5-nano",
        temperature=1.0
    )

def init_emb() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

def init_vectorstore( embe: OpenAIEmbeddings) -> Chroma:
    return Chroma(
        collection_name="repo_files",
        embedding_function= embe,
        persist_directory="./data/chroma_db"
    )

def download(repo:str, dir_p:Path) -> str:
    repo_name = repo.split('/')[-1].strip().removesuffix(".git")
    git.Git(dir_p).clone(repo)
    full_path = Path(f"{dir_p}/{repo_name}")
    with open("temp_doc_list.txt", "w") as file:
        for f in list(full_path.rglob("*")):
            file.writelines(str(f))
            file.writelines(",")
    return str(full_path)

def filter_docs() -> list[str]:
    wanted_docs = []
    with open("temp_doc_list.txt", "r+") as file:
        files = file.read()
        files = files.split(',')
        wanted_docs = [
            file 
            for file in files
            if file.endswith(tuple(CODING_FILES))
        ]
    return wanted_docs

def chunking(docs_list:list[str], chunk_size:int = 500) -> dict:
    overlap = 100
    chunk_size = chunk_size * 4
    docs_dict = {}
    print("Dividiendo documentos...")
    for doc in docs_list:
        chunks = []
        start = 0
        with open(f"{doc}", "r") as file:
            open_doc = file.read()
            while start < len(open_doc):
                end = start + chunk_size
                chunk = open_doc[start:end]
                if chunk.strip():
                    chunks.append(chunk)
                start = end - overlap*4
        docs_dict[doc] = chunks
        print(f"Documento {doc} ha sido dividido")
    return docs_dict

def embeddings_vectorstore(docs_dict:dict, vectorstore: Chroma):
    for doc, chunks in docs_dict.items():
        print(f"Agregando {doc} a vector DB...")
        vectorstore.add_texts(
            texts=chunks,
            metadatas=[{"file_name":doc} for _ in chunks],
            ids=[ hashlib.sha256(chunk.encode()).hexdigest() for chunk in chunks ]
        )
        print(f"{len(chunks)} chunks agregados...")

def rag(question:str, llm:init_chat_model, vectorstore: Chroma):
    retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k':3}
    )
    prompt = ChatPromptTemplate.from_messages(
        [
        ("system","""Eres un asistente que responde preguntas
basándote ÚNICAMENTE en el contexto proporcionado.

Contexto recuperado de los documentos:
{context}

Instrucciones:
- Si la respuesta está en el contexto, respóndela con precisión.
- Si no está, di: "No encontré esa información en los documentos."
- Cita la fuente cuando sea posible.
- No inventes ni supongas información."""),
        ("human", "{quest}")
    ])
    parser = StrOutputParser()

    chain = ({
        "context":retriever,
        "quest": RunnablePassthrough()
    }|prompt|llm|parser)
    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)

def formatted_answere(un_json, question):
    # se formatea el json
    return f'La respuesta a la pregunta "{question}": ...'

def clean_up(dir_p:Path):
    pass

if __name__ == '__main__':
    # model = init_model()
    # embe = init_emb()
    # vectorstore = init_vectorstore(embe)
    # download()
    # filtered_docs = filter_docs()
    # docs_dict = chunking(filtered_docs)
    # embeddings_vectorstore(docs_dict, vectorstore)
    # question = input("User: ").strip()
    # rag(question, model, vectorstore)
    # repo = input("Ingresa el link del repo: ")
    full_path = download("https://github.com/frontapp/frontapp.github.io.git", DIR_P)
    print(filter_docs())
