# 👨‍🍳 ChefAI: Tu Asistente Culinario Inteligente

ChefAI es una aplicación de inteligencia artificial generativa que sugiere recetas personalizadas
según los ingredientes que tengas disponibles en tu cocina, preferencias dietéticas y estilo culinario.
Utiliza generación aumentada por recuperación (RAG), embeddings (FAISS) y modelos de lenguaje de OpenAI para generar platos creativos,
variantes e incluso responder a preguntas culinarias.

## 🧠 Tecnologías utilizadas

- Python 3.12
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- [FAISS (faiss-cpu)](https://github.com/facebookresearch/faiss)
- Dataset de recetas (train.csv): [Frorozcol/recetas-cocina](https://huggingface.co/datasets/Frorozcol/recetas-cocina)

## ⚙️ Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/ChefAI.git
   cd ChefAI
   ```

2. (Opcional) Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Añade tu clave de OpenAI en un archivo `.env` en la raíz del proyecto:
   ```
   OPENAI_API_KEY="sk-..."
   ```

5. Ejecuta la aplicación:
   ```bash
   streamlit run app.py
   ```

## 📦 Estructura del proyecto

```
.
├── app.py            # Código principal de la aplicación
├── requirements.txt   # Dependencias del proyecto
├── .env               # Clave API de OpenAI
├── train.csv          # Dataset de recetas base (cargado desde Hugging Face)
└── README.md          
```

## 🧪 Ejemplos de uso

- **Ingredientes**: `pollo, brócoli, arroz`
- **Preferencias**: `sin gluten`, `cocina asiática`

ChefAI generará una receta base, una variante, un plato alternativo y preguntas sugeridas.
También puedes hacer preguntas personalizadas en un chat por receta.

## 📝 Notas

- Asegúrate de tener una clave válida de la API de OpenAI.
- El dataset `train.csv` debe descargarse manualmente desde Hugging Face.
