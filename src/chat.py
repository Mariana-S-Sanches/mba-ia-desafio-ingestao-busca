import search
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from search import load_config, semantic_search, get_llm

load_dotenv()

def build_context(results) -> str:
    if not results:
        return ""

    parts = []
    for doc, score in results:
        page = doc.metadata.get("page")
        source = doc.metadata.get("source", "document.pdf")
        header = f"[source={source} page={page} score={score}]"
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)

def main():
    cfg = load_config()
    llm = get_llm(cfg)

    prompt = ChatPromptTemplate.from_template(search.PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    print("Faça sua pergunta (digite 'sair' para encerrar):\n")

    while True:
        pergunta = input("PERGUNTA: ").strip()
        if not pergunta:
            continue
        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("Encerrando...")
            break

        results = semantic_search(cfg, pergunta, k=10)
        contexto = build_context(results)

        resposta = chain.invoke({"contexto": contexto, "pergunta": pergunta})
        print(f"RESPOSTA: {resposta}\n")

if __name__ == "__main__":
    main()