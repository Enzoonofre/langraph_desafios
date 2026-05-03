# Assistente Pessoal com LangGraph + Groq

Agente conversacional construído com **LangGraph** e **Groq (LLaMA 3.3 70B)** capaz de usar ferramentas para responder perguntas práticas do dia a dia.

Projeto desenvolvido como solução do **Desafio 01** do curso de LangGraph.

---

## O que o agente faz

O assistente recebe perguntas em linguagem natural e decide automaticamente qual ferramenta usar para respondê-las:

| Ferramenta | O que faz |
|---|---|
| `celsius_to_fahrenheit` | Converte temperatura de °C para °F |
| `get_bmi` | Calcula o IMC dado peso (kg) e altura (m) |
| `convert_currency` | Converte Reais para Dólares usando taxa de câmbio em tempo real |

Exemplos de perguntas que o agente responde corretamente:

```
"Qual é meu IMC se peso 75kg e tenho 1,75m?"
"Converta 100 reais para dólares"
"Qual é minha temperatura em Fahrenheit se tenho 37.5°C de febre?"
```

---

## Como foi feito

### Arquitetura do grafo

O agente segue o padrão **ReAct** (Reason + Act) implementado com LangGraph:

```
START → llm → tools_condition → tools → llm → ... → END
```

- O nó `llm` invoca o modelo com as ferramentas vinculadas (`bind_tools`)
- O `tools_condition` verifica se o modelo quer chamar alguma ferramenta
- Se sim, o `ToolNode` executa a ferramenta e devolve o resultado ao `llm`
- O ciclo se repete até o modelo dar a resposta final

### Estado

Usa `MessagesState` do LangGraph — uma lista de mensagens que acumula todo o histórico da conversa (humano, IA, ferramentas).

### Ferramentas

Cada ferramenta é uma função Python decorada com `@tool`. O LLM depende das **docstrings** para decidir qual ferramenta chamar e com quais argumentos.

A ferramenta `convert_currency` consome a [Open Exchange Rates API](https://open.er-api.com) para obter a cotação real do dólar em vez de usar taxas fixas.

### Modelo

**Groq** com o modelo `llama-3.3-70b-versatile` — inferência rápida e gratuita para desenvolvimento.

---

## Estrutura do projeto

```
curso_langraph/
├── src/
│   └── desafios/
│       └── ex1.py        # agente principal
├── pyproject.toml
└── .env                  # GROQ_API_KEY (não versionar)
```

---

## Como rodar

**1. Clone o repositório e instale as dependências**

```bash
# com uv (recomendado)
uv sync

# ou com pip
pip install langchain-groq langgraph python-dotenv requests
```

**2. Configure a chave de API**

Crie um arquivo `.env` na raiz do projeto:

```
GROQ_API_KEY=sua_chave_aqui
```

Obtenha sua chave gratuitamente em [console.groq.com](https://console.groq.com).

**3. Execute o agente**

```bash
python src/desafios/ex1.py
```

**4. Converse**

```
=== Assistente Pessoal ===
Digite 'sair' para encerrar.

Você: Qual é meu IMC se peso 80kg e tenho 1.80m?

[Human]: Qual é meu IMC se peso 80kg e tenho 1.80m?
[AI calls tool]: get_bmi({'weight_kg': 80, 'height_m': 1.8})
[Tool result]: 24.691358024691358
[AI final answer]: Seu IMC é de aproximadamente 24,69, o que se enquadra na faixa de Peso Normal (18,5 a 24,9). Parabéns!
```

---

## Dependências principais

- [langgraph](https://github.com/langchain-ai/langgraph) — orquestração do grafo de agente
- [langchain-groq](https://python.langchain.com/docs/integrations/chat/groq/) — integração com Groq
- [langchain-core](https://python.langchain.com/docs/concepts/) — `@tool`, `MessagesState`, `ToolNode`
- [python-dotenv](https://github.com/theskumar/python-dotenv) — variáveis de ambiente
- [requests](https://docs.python-requests.org/) — chamada à API de câmbio
