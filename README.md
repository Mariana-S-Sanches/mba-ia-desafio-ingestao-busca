# Desafio MBA Engenharia de Software com IA - Full Cycle

Projeto de **ingestão e busca semântica** com **LangChain + PostgreSQL (pgvector)**.

- **Ingestão**: lê `document.pdf`, quebra em chunks (1000 chars, overlap 150), gera embeddings e salva no Postgres.
- **Chat (CLI)**: você pergunta no terminal e a resposta é gerada **somente com base no conteúdo do PDF**.

---

## Pré-requisitos

### Obrigatórios
- **Docker** e **Docker Compose**
- **Python 3.10+** (recomendado 3.11)
- Um arquivo `document.pdf` na raiz do projeto

### Chaves de API (escolha 1 provedor)
- **OpenAI** (recomendado): `OPENAI_API_KEY`
- **Gemini**: `GOOGLE_API_KEY`

> Importante: a ingestão e a busca **devem usar o mesmo provedor de embeddings**. Se você mudar `PROVIDER`, precisa **rodar a ingestão novamente**.

---

## 📁 Estrutura do projeto

```
├── docker-compose.yml
├── docker/
│   └── init.sql
├── requirements.txt
├── .env
├── .env.example
├── src/
│   ├── ingest.py
│   ├── search.py
│   └── chat.py
└── document.pdf
```

---

## 1) Configuração do `.env`

Crie o arquivo `.env` a partir do template:

```bash
cp .env.example .env
```

Edite o `.env` e escolha o provedor:

### Opção A — OpenAI (recomendado)
```env
PROVIDER=openai
OPENAI_API_KEY=SUA_CHAVE_AQUI
```

### Opção B — Gemini
```env
PROVIDER=gemini
GOOGLE_API_KEY=SUA_CHAVE_AQUI
```

---

## 2) Subir o PostgreSQL com pgvector (Docker)

### 2.1 Criar a extensão pgvector
Garanta que existe o arquivo `docker/init.sql` com o conteúdo:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2.2 Subir o container
Na raiz do projeto:

```bash
docker compose up -d
```

Ver logs:

```bash
docker compose logs -f postgres
```

Verificar se está rodando:

```bash
docker ps
```

> O banco sobe em `localhost:5432` (usuário `postgres`, senha `postgres`, db `rag`).

---

## 3) Configurar e instalar dependências (Python)

### 3.1 Criar e ativar venv

```bash
python3 -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3.2 Instalar dependências

```bash
pip install -r requirements.txt
```

### 3.3 VS Code (Pylance) não encontra imports
Se aparecer “Não foi possível resolver a importação ...”:
- `Cmd+Shift+P` → **Python: Select Interpreter**
- selecione o interpretador do `venv`

---

## 4) Ingestão do PDF

Com o banco rodando e o `.env` configurado:

```bash
python src/ingest.py
```

Você deve ver algo como:
- quantidade de páginas
- quantidade de chunks
- nome da coleção

> A ingestão recria a coleção (evita duplicar dados). Se você trocar `PROVIDER`, rode a ingestão novamente.

---

## 5) Rodar o chat (CLI)

```bash
python src/chat.py
```

Exemplo:

```
Faça sua pergunta (digite 'sair' para encerrar):

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: ...
```

---

## 6) Teste rápido de busca (debug)

Se quiser validar se a busca está retornando chunks:

```bash
python -c "from src.search import load_config, semantic_search; cfg=load_config(); r=semantic_search(cfg,'Qual o faturamento da Empresa SuperTechIABrazil?',k=10); print('qtd:',len(r)); print([(d.metadata.get('page'), s, d.page_content[:140].replace('\\n',' ')) for d,s in r[:3]])"
```

- Se `qtd: 0`, a coleção está vazia, a conexão está errada, ou o provider não bate com a ingestão.

---

## Problemas comuns e como resolver

### 1) `qtd: 0` na busca (nenhum resultado)
Causas comuns:
- Você mudou `PROVIDER` depois da ingestão (ex.: ingeriu com OpenAI e buscou com Gemini)
- Nome da coleção diferente no `.env`
- Ingestão não gravou no banco

Solução:
1. Confirme no `.env` que `PROVIDER` está correto.
2. Rode novamente:
   ```bash
   python src/ingest.py
   ```

### 2) `429 ResourceExhausted` (Gemini quota exceeded)
Isso significa que seu projeto/conta está sem quota para embeddings no Gemini.

Solução:
- Use **OpenAI** (recomendado), ou
- Ative billing / ajuste quotas no painel do Google.

### 3) pgvector não foi criado (erro: extensão `vector` não existe)
Isso ocorre quando o volume do Postgres já existia antes do `init.sql`.

Solução (APAGA o banco e recria):

```bash
docker compose down -v
docker compose up -d
```

### 4) Porta 5432 já está em uso
Outro Postgres pode estar rodando na sua máquina.

Soluções:
- Pare o serviço que está usando 5432, ou
- Troque a porta no `docker-compose.yml` (ex.: `"5433:5432"`) e ajuste `POSTGRES_PORT` no `.env`.

### 5) “Pylance: import could not be resolved”
Solução:
- `Cmd+Shift+P` → **Python: Select Interpreter** → selecione o `venv`.

### 6) PDF sem texto (scan)
Se o `PyPDFLoader` extrair texto vazio, seu PDF pode ser imagem/scaneado.

Como verificar:
```bash
python -c "from langchain_community.document_loaders import PyPDFLoader; d=PyPDFLoader('document.pdf').load(); print(d[0].page_content[:500])"
```
Se vier vazio, será necessário OCR (não implementado neste template).

---

## Reset completo (quando tudo parece bagunçado)

1) Derruba containers e apaga volume do banco:

```bash
docker compose down -v
```

2) Sobe o banco de novo:

```bash
docker compose up -d
```

3) Reinstala deps (se necessário) e roda ingestão:

```bash
source venv/bin/activate
pip install -r requirements.txt
python src/ingest.py
python src/chat.py
```

---

## Observações
- O prompt do chat foi construído para **responder somente com base no contexto** e retornar a frase padrão quando não houver informação.
- O banco vetorial usa `langchain-postgres` + `pgvector`.