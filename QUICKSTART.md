# 🚀 Início Rápido

Este guia vai te colocar de pé e rodando em **menos de 5 minutos**!

## ⚡ Setup Rápido (3 passos)

### 1️⃣ Instalar Dependências

```bash
pip install -r requirements.txt
```

*Tempo estimado: 2-3 minutos*

### 2️⃣ Rodar a Aplicação

```bash
streamlit run app.py
```

*A aplicação abrirá automaticamente em http://localhost:8501*

### 3️⃣ Testar!

Digite na interface:

```
Qual o salário da Ana em março de 2025?
```

**Pronto! 🎉** Você já tem um chatbot RAG funcionando!

## 📱 Primeiros Passos

### Perguntas Sugeridas

Experimente estas perguntas:

1. **"Mostre os pagamentos do Bruno Lima"**
   - Vai listar todos os pagamentos dele

2. **"Qual foi o bônus em maio?"**
   - Mostra bônus de maio de todos os funcionários

3. **"Estatísticas gerais"**
   - Exibe resumo completo do dataset

4. **"Ajuda"**
   - Lista todas as funcionalidades disponíveis

### Recursos da Interface

**Sidebar (lado esquerdo):**
- 📈 **Ver Estatísticas**: Resumo dos dados
- 💾 **Exportar Conversa**: Download JSON da conversa
- 🗑️ **Limpar Histórico**: Resetar a conversa
- 💡 **Exemplos**: Lista de perguntas sugeridas

**Painel Principal:**
- 💬 **Conversa**: Histórico de mensagens
- 📋 **Metadados**: Detalhes técnicos das respostas (intent, confidence, fontes)

## 🧪 Rodar Testes (Opcional)

```bash
pytest tests/ -v
```

Vai executar 22 testes em ~10 segundos.

## ⚙️ Configuração Avançada (Opcional)

### Usar LLM Real (OpenAI/Anthropic)

1. Copie `.env_example` para `.env`:
```bash
copy .env_example .env  # Windows
# ou
cp .env_example .env    # Linux/Mac
```

2. Edite `.env` e adicione sua chave:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-seu-token-aqui
```

3. Reinicie a aplicação

### Adicionar Mais Dados

Edite `data/payroll.csv` e adicione linhas seguindo o formato:

```csv
E003,Carlos Santos,2025-01,10000,1000,700,0,1100.0,600.0,0,9000.0,2025-01-28
```

**Colunas:**
- employee_id
- name
- competency (YYYY-MM)
- base_salary
- bonus
- benefits_vt_vr
- other_earnings
- deductions_inss
- deductions_irrf
- other_deductions
- net_pay
- payment_date

Reinicie a aplicação para recarregar os dados.

## 🐛 Problemas Comuns

### "ModuleNotFoundError"
```bash
# Certifique-se que está no ambiente virtual
pip install -r requirements.txt
```

### "Cannot find data/payroll.csv"
```bash
# Verifique que está no diretório correto
cd Chatbot
```

### Download lento na primeira execução
- Normal! O modelo de embeddings (~500MB) está sendo baixado
- Após o download, fica em cache e é instantâneo

### Streamlit não abre automaticamente
- Abra manualmente: http://localhost:8501
- Ou use: `streamlit run app.py --server.headless=false`

## 📚 Próximos Passos

1. ✅ Leia o [README.md](README.md) completo para entender a arquitetura
2. ✅ Explore o código em `src/`
3. ✅ Rode os testes em `tests/`
4. ✅ Customize o dataset em `data/payroll.csv`
5. ✅ Experimente diferentes configurações no `.env`

## 💬 Dúvidas?

Consulte o [README.md](README.md) para:
- Decisões técnicas detalhadas
- Limitações conhecidas
- Guia de deployment
- Roadmap de melhorias

---

**Divirta-se explorando! 🚀** 