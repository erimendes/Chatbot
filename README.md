# 💼 Chatbot de Folha de Pagamento com RAG

Um chatbot inteligente construído com **Streamlit** que combina conversação geral e consultas especializadas em dados de folha de pagamento usando **RAG (Retrieval-Augmented Generation)**.

## 🎯 Funcionalidades

### Requisitos Mínimos (✅ Implementado)
- ✅ **Chat básico com LLM**: Entrada de usuário → resposta do modelo
- ✅ **RAG sobre folha de pagamento**: Carregamento e consulta do dataset
- ✅ **Perguntas por múltiplos critérios**: Nome, competência, pagamento líquido, bônus, INSS, IRRF, datas
- ✅ **Citação de fontes**: Referência às linhas do dataset usadas na resposta
- ✅ **Configuração via `.env`**: Chaves de API, URLs, parâmetros
- ✅ **Tratamento de erros e logs**: Logging estruturado e tratamento de exceções

### Funcionalidades Avançadas (✅ Implementado)
- ✅ **RAG bem feito**: Embeddings multilíngues + chunking semântico + ranking por similaridade
- ✅ **Tolerância a variações**: Suporta diferentes formatos de data e texto
- ✅ **Memória de conversa**: Contexto mantido entre turnos (histórico configurável)
- ✅ **Testes automatizados**: 3 suítes de testes (RAG, Intent, Conversation)
- ✅ **UX moderna**: Interface Streamlit caprichada com histórico, botões, export JSON
- ✅ **Observabilidade**: Logs estruturados com níveis configuráveis
- ✅ **Guardrails**: Proteção contra prompt injection e sanitização de entrada
- ✅ **Classificação de intenções**: Roteamento inteligente de queries
- ✅ **Formatação monetária**: Valores em formato brasileiro (R$ X.XXX,XX)

## 📁 Estrutura do Projeto

```
Chatbot/
├── app.py                          # Aplicação principal Streamlit
├── requirements.txt                # Dependências Python
├── .env_example                    # Exemplo de configuração
├── README.md                       # Esta documentação
├── data/
│   └── payroll.csv                # Dataset de folha de pagamento
├── src/
│   ├── __init__.py
│   ├── rag_engine.py              # Motor RAG (embeddings + busca)
│   ├── llm_interface.py           # Interface LLM (mock/OpenAI/Anthropic)
│   ├── conversation_manager.py    # Gerenciador de histórico
│   └── intent_classifier.py       # Classificador de intenções
├── tests/
│   ├── __init__.py
│   ├── test_rag_engine.py         # Testes do motor RAG
│   ├── test_intent_classifier.py  # Testes do classificador
│   └── test_conversation_manager.py # Testes do gerenciador
└── logs/
    └── chatbot.log                # Logs da aplicação
```

## 🚀 Como Rodar

### 1. Pré-requisitos

- Python 3.9+
- pip ou conda

### 2. Instalação

```bash
# Clonar o repositório
git clone <url-do-repo>
cd Chatbot

# Criar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 3. Configuração

```bash
# Copiar arquivo de exemplo
cp .env_example .env

# Editar .env com suas configurações (opcional)
# Por padrão, usa LLM mock sem necessidade de chaves de API
```

**Configurações disponíveis:**

```env
LLM_PROVIDER=mock                  # opções: mock, openai, anthropic
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2
TOP_K_RESULTS=3                    # Número de resultados RAG
MAX_CONVERSATION_HISTORY=10        # Tamanho do histórico
TEMPERATURE=0.7                    # Temperatura do LLM
LOG_LEVEL=INFO                     # Nível de log
```

### 4. Executar Aplicação

```bash
streamlit run app.py
```

A aplicação abrirá automaticamente em `http://localhost:8501`

### 5. Rodar Testes

```bash
# Executar todos os testes
pytest tests/ -v

# Executar com cobertura
pytest tests/ --cov=src --cov-report=html

# Executar testes específicos
pytest tests/test_rag_engine.py -v
```

## 💡 Como Usar

### Exemplos de Perguntas

**Consultas sobre funcionários específicos:**
```
- Qual o salário líquido da Ana em março?
- Mostre os pagamentos do Bruno Lima
- Quanto a Ana recebeu em maio de 2025?
```

**Consultas sobre componentes:**
```
- Qual foi o bônus em maio de 2025?
- Quais os descontos de INSS da Ana?
- Mostre os benefícios (VT/VR) do Bruno
```

**Consultas sobre datas:**
```
- Quando foi o pagamento de abril?
- Mostre a data de pagamento da Ana em junho
```

**Estatísticas gerais:**
```
- Quantos funcionários temos?
- Qual a média de pagamentos?
- Total pago no período
```

**Conversa geral:**
```
- Olá, bom dia!
- Ajuda
- O que você pode fazer?
- Obrigado
```

## 🏗️ Decisões Técnicas

### 1. RAG (Retrieval-Augmented Generation)

**Embeddings:**
- Modelo: `paraphrase-multilingual-mpnet-base-v2`
- Razão: Suporta português com alta qualidade e é relativamente leve
- Alternativas consideradas: `all-MiniLM-L6-v2` (menor), `all-mpnet-base-v2` (melhor, mas só inglês)

**Chunking:**
- Estratégia: 1 chunk = 1 linha do dataset
- Cada chunk contém texto descritivo + metadados estruturados
- Permite citação precisa da fonte (linha específica)

**Ranking:**
- Método: Similaridade cosseno entre embeddings
- Top-K configurável (padrão: 3 resultados)
- Threshold mínimo para garantir relevância

### 2. LLM Interface

**Arquitetura plugável:**
- Suporta múltiplos providers: Mock, OpenAI, Anthropic
- Mock inteligente para demonstração sem custos de API
- Fácil extensão para outros providers

**Por que Mock é o padrão?**
- Permite demonstração imediata sem chaves de API
- Mock contextual analisa patterns e gera respostas relevantes
- Ideal para desenvolvimento e testes

### 3. Classificação de Intenções

**Abordagem baseada em regras:**
- Padrões regex para identificar intenções
- 4 categorias: Payroll, Statistics, Help, General
- Score de confiança baseado em matches

**Por que não ML para classificação?**
- Dataset pequeno não justifica modelo ML
- Regras são transparentes e facilmente ajustáveis
- Performance adequada para o escopo

### 4. Interface (Streamlit)

**Por que Streamlit?**
- Prototipagem rápida com UI moderna
- Componentes interativos nativos
- Fácil deployment (Streamlit Cloud)
- Cache de recursos para performance

**UX Features:**
- Histórico visual de conversa
- Sidebar com estatísticas e controles
- Export JSON para análise
- Metadados visíveis (intent, confidence, sources)

### 5. Testes

**Estratégia:**
- Testes unitários para componentes isolados
- Fixtures para setup reutilizável
- Cobertura das principais funcionalidades
- Validação de formatos monetários brasileiros

## ⚠️ Limitações Conhecidas

### 1. Modelo de Linguagem
- **Mock LLM**: Respostas baseadas em regras, não contexto profundo
- **Solução**: Configurar OpenAI ou Anthropic API no `.env`
- **Impacto**: Respostas menos naturais, mas funcionais para demonstração

### 2. Dataset
- **Tamanho**: Apenas 12 registros (2 funcionários x 6 meses)
- **Impacto**: Queries fora desse escopo retornarão "não encontrado"
- **Extensão**: Adicionar mais linhas ao `data/payroll.csv`

### 3. Embeddings
- **Download inicial**: ~500MB na primeira execução
- **Tempo**: ~10-30 segundos dependendo da conexão
- **Cache**: Modelos são cacheados após download

### 4. Memória de Conversa
- **Escopo**: Apenas na sessão atual (não persiste)
- **Limite**: Configurável, padrão 10 mensagens
- **Reset**: Ao recarregar a página

### 5. Guardrails
- **Básico**: Proteção contra SQL injection e XSS simples
- **Limitação**: Não cobre todos os ataques possíveis
- **Produção**: Usar WAF e validação mais robusta

### 6. Multilíngua
- **Suporte**: Otimizado para português
- **Inglês**: Funciona, mas com performance reduzida
- **Outros idiomas**: Não testados

### 7. Performance
- **Inicialização**: ~5-10 segundos (carregar modelo + criar embeddings)
- **Queries**: <1 segundo após inicialização
- **Escalabilidade**: Adequado para uso individual/pequeno time
- **Produção**: Considerar cache de embeddings em banco vetorial

## 🧪 Cobertura de Testes

### Testes Implementados

**`test_rag_engine.py` (9 testes):**
- ✅ Inicialização do engine
- ✅ Busca por nome de funcionário
- ✅ Busca por competência
- ✅ Busca por pagamento líquido
- ✅ Filtros estruturados (nome, competência, employee_id)
- ✅ Cálculo de estatísticas
- ✅ Sanitização de queries maliciosas
- ✅ Formatação monetária brasileira

**`test_intent_classifier.py` (6 testes):**
- ✅ Detecção de queries de folha de pagamento
- ✅ Detecção de consultas estatísticas
- ✅ Detecção de pedidos de ajuda
- ✅ Detecção de conversa geral
- ✅ Extração de filtros estruturados
- ✅ Queries com múltiplos indicadores

**`test_conversation_manager.py` (7 testes):**
- ✅ Inicialização do gerenciador
- ✅ Adição de mensagens
- ✅ Limite de mensagens no histórico
- ✅ Armazenamento de metadados
- ✅ Limpeza do histórico
- ✅ Exportação da conversa
- ✅ Timestamps nas mensagens

### Executar Testes

```bash
# Todos os testes
pytest tests/ -v

# Com cobertura
pytest tests/ --cov=src --cov-report=term-missing

# Testes específicos
pytest tests/test_rag_engine.py::test_monetary_formatting -v
```

## 📊 Observabilidade

### Logs

**Níveis disponíveis:**
- `DEBUG`: Detalhes de cada operação
- `INFO`: Eventos importantes (queries, inicialização)
- `WARNING`: Situações anormais (queries maliciosas)
- `ERROR`: Erros que impedem operação

**Configuração:**
```env
LOG_LEVEL=INFO
LOG_FILE=logs/chatbot.log
```

**Visualização:**
```bash
# Tempo real
tail -f logs/chatbot.log

# Filtrar por nível
grep "ERROR" logs/chatbot.log

# Últimas 50 linhas
tail -n 50 logs/chatbot.log
```

### Métricas Disponíveis

**No sidebar da aplicação:**
- Número de mensagens processadas
- Tamanho do histórico atual
- Provider LLM em uso

**Nos metadados (JSON):**
- Intent classificado
- Confidence score
- Número de resultados RAG
- Indices das linhas fonte

## 🔒 Segurança

### Implementações

1. **Sanitização de input**: Remove comandos SQL e scripts
2. **Validação de queries**: Rejeita queries suspeitas
3. **Logging de ataques**: Registra tentativas de injeção
4. **Sem execução de código**: Apenas busca em dados estáticos

### Recomendações para Produção

- [ ] Adicionar rate limiting
- [ ] Implementar autenticação/autorização
- [ ] Usar HTTPS
- [ ] Criptografar dados sensíveis
- [ ] Adicionar WAF (Web Application Firewall)
- [ ] Implementar Content Security Policy
- [ ] Audit logs completos

## 🚀 Deployment

### Streamlit Cloud (Recomendado)

```bash
# 1. Fazer push do código para GitHub
git add .
git commit -m "Initial commit"
git push origin main

# 2. Acessar streamlit.io/cloud
# 3. Conectar repositório GitHub
# 4. Definir variáveis de ambiente no dashboard
# 5. Deploy automático!
```

### Docker (Alternativa)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t payroll-chatbot .
docker run -p 8501:8501 payroll-chatbot
```

## 📈 Melhorias Futuras

### Curto Prazo
- [ ] Adicionar mais funcionários ao dataset
- [ ] Implementar cache de embeddings em disco
- [ ] Adicionar gráficos de visualização de dados
- [ ] Suportar upload de CSV customizado

### Médio Prazo
- [ ] Integrar LLM real (GPT-4, Claude)
- [ ] Implementar histórico persistente (banco de dados)
- [ ] Adicionar autenticação de usuários
- [ ] Dashboard de analytics

### Longo Prazo
- [ ] Multitenancy (múltiplas empresas)
- [ ] Integração com sistemas de folha reais
- [ ] API REST para integração externa
- [ ] App mobile

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto é distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

## 👤 Autor

Desenvolvido como projeto de demonstração de RAG + Chatbot.

## 🙏 Agradecimentos

- **Sentence Transformers**: Por modelos de embedding de alta qualidade
- **Streamlit**: Por tornar interfaces web tão simples
- **Hugging Face**: Por democratizar ML/NLP

---

**💼 Chatbot de Folha de Pagamento v1.0** | Powered by RAG + Streamlit 