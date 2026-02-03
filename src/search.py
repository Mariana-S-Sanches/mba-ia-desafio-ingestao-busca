import os
from dataclasses import dataclass
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_postgres import PGVector

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

load_dotenv()

PROMPT_TEMPLATE = """CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"""

@dataclass(frozen=True)
class AppConfig:
    provider: str
    collection_name: str
    conn_str: str

    openai_embedding_model: str
    openai_llm_model: str

    google_embedding_model: str
    google_llm_model: str


def _build_conn_str() -> str:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    db = os.getenv("POSTGRES_DB", "rag")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"


def load_config() -> AppConfig:
    provider = os.getenv("PROVIDER", "openai").strip().lower()
    collection_name = os.getenv("COLLECTION_NAME", "document_pdf").strip()
    conn_str = os.getenv("DATABASE_URL", "").strip() or _build_conn_str()

    return AppConfig(
        provider=provider,
        collection_name=collection_name,
        conn_str=conn_str,
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        openai_llm_model=os.getenv("OPENAI_LLM_MODEL", "gpt-5-nano"),
        google_embedding_model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001"),
        google_llm_model=os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash-lite"),
    )

def get_embeddings(cfg: AppConfig):
    """Retorna o gerador de embeddings conforme PROVIDER."""
    if cfg.provider == "gemini":
        return GoogleGenerativeAIEmbeddings(model=cfg.google_embedding_model)
    return OpenAIEmbeddings(model=cfg.openai_embedding_model)


def get_llm(cfg: AppConfig):
    """Retorna o chat model conforme PROVIDER."""
    if cfg.provider == "gemini":
        return ChatGoogleGenerativeAI(model=cfg.google_llm_model, temperature=0)
    return ChatOpenAI(model=cfg.openai_llm_model, temperature=0)

def get_vectorstore(cfg: AppConfig) -> PGVector:
    embeddings = get_embeddings(cfg)
    return PGVector(
        embeddings=embeddings,
        collection_name=cfg.collection_name,
        connection=cfg.conn_str,
        use_jsonb=True,
    )


def semantic_search(cfg: AppConfig, query: str, k: int = 10) -> List[Tuple[Document, float]]:
    """Busca semântica no Postgres (pgvector)."""
    vs = get_vectorstore(cfg)
    return vs.similarity_search_with_score(query, k=k)

def build_context(results: List[Tuple[Document, float]]) -> str:
    """Concatena resultados em um contexto legível, incluindo metadados (page/source/score)."""
    if not results:
        return ""

    parts: List[str] = []
    for doc, score in results:
        page = doc.metadata.get("page")
        source = doc.metadata.get("source", "document.pdf")
        header = f"[source={source} page={page} score={score}]"
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


def search_prompt(question: str, k: int = 10) -> Tuple[str, List[Tuple[Document, float]]]:
    """Gera o prompt completo (template + contexto) a partir de uma pergunta.

    Retorna:
      - prompt (str)
      - results (List[(Document, score)])

    Esse helper deixa o `search.py` com uma estrutura bem próxima ao fork (template + função).
    """
    cfg = load_config()
    results = semantic_search(cfg, question, k=k)
    contexto = build_context(results)
    prompt = PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question)
    return prompt, results