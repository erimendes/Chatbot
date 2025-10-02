# Etapas para instalar e rodar LLM com Ollama no WSL2
curl -fsSL https://ollama.com/install.sh | sh
ollama run mistral
ollama list
echo "Explique o que é inteligência artificial" | ollama run mistral
streamlit run app.py

