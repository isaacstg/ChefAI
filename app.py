import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv
import hashlib

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import csv

st.set_page_config(page_title="ChefAI", layout="wide", initial_sidebar_state="collapsed")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None 

FAISS_INDEX_PATH = "faiss_recipe_index"
NOMBRE_ARCHIVO_LOCAL_RECETAS = "train.csv"

default_session_state = {
    'rag_chain': None,
    'recipes_data_loaded': False,
    'processed_recipes_data': [], 
    'displayed_recipes': [],      
    'chat_histories': {},         
    'generated_questions_for_recipe': {}, 
    'sugg_question_clicked_info': None
}
for key, default_value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


@st.cache_data
def load_and_prepare_dataset_local(local_file_path):
    try:
        dataset_list_of_dicts = []
        if not os.path.exists(local_file_path):
            st.error(f"Archivo de recetas '{local_file_path}' no encontrado.")
            return []
        with open(local_file_path, mode='r', encoding='utf-8', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row_dict in reader:
                dataset_list_of_dicts.append(row_dict)
        if not dataset_list_of_dicts:
            st.error(f"El archivo de recetas '{local_file_path}' está vacío.")
            return []
        
        processed_recipes = []
        for i, recipe_data in enumerate(dataset_list_of_dicts):
            title = recipe_data.get("title") or recipe_data.get("titulo") or f"Receta Desconocida {i+1}"
            ingredients_raw = recipe_data.get("ingredients") or recipe_data.get("ingredientes")
            steps_raw = recipe_data.get("steps") or recipe_data.get("preparacion")
            uuid_val = recipe_data.get("uuid")
            url_val = recipe_data.get("url")
            recipe_id = uuid_val or url_val or f"local_recipe_{i}"
            
            ingredients_text = ""
            if isinstance(ingredients_raw, str): ingredients_text = ingredients_raw
            elif isinstance(ingredients_raw, list): ingredients_text = "\n".join(filter(None, map(str, ingredients_raw)))
            elif ingredients_raw is not None: ingredients_text = str(ingredients_raw)
            else: ingredients_text = "Ingredientes no detallados"

            steps_text = ""
            if isinstance(steps_raw, str): steps_text = steps_raw
            elif isinstance(steps_raw, list): steps_text = "\n".join(filter(None, map(str, steps_raw)))
            elif steps_raw is not None: steps_text = str(steps_raw)
            else: steps_text = "Pasos no detallados"

            texto_completo_rag = f"Título: {title}\nIngredientes: {ingredients_text}\nPasos: {steps_text}"

            processed_recipes.append({
                "id": recipe_id,
                "title": title, 
                "texto_completo_rag": texto_completo_rag, 
                "source_dataset": "local_file: " + local_file_path
            })
        return processed_recipes
    except Exception as e:
        st.error(f"Error al cargar recetas: {e}")
        return []


def initialize_rag_system(recipes_data_list, embeddings_model_instance, index_path):
    if not OPENAI_API_KEY: return None
    vector_store = None
    if os.path.exists(index_path) and os.path.exists(os.path.join(index_path, "index.faiss")):
        try:
            vector_store = FAISS.load_local(index_path, embeddings_model_instance, allow_dangerous_deserialization=True)
        except Exception: vector_store = None

    if vector_store is None:
        if not recipes_data_list: return None
        documents_for_rag = []
        for recipe_item in recipes_data_list:
            doc_content = recipe_item.get("texto_completo_rag", "") # Usar el texto para RAG
            doc_metadata = {
                "source_id": recipe_item.get("id", "unknown"), 
                "title": recipe_item.get("title", "Sin título"), 
                "source_dataset": recipe_item.get("source_dataset", "unknown_source")
            }
            documents_for_rag.append(Document(page_content=doc_content, metadata=doc_metadata))
        if not documents_for_rag: return None
        try:
            valid_documents = [doc for doc in documents_for_rag if doc.page_content and doc.page_content.strip()]
            if not valid_documents: return None
            batch_size_docs = 500
            first_batch_docs = valid_documents[:batch_size_docs]
            if not first_batch_docs: return None
            vector_store = FAISS.from_documents(first_batch_docs, embeddings_model_instance)
            for i in range(batch_size_docs, len(valid_documents), batch_size_docs):
                current_batch_docs = valid_documents[i:i + batch_size_docs]
                if current_batch_docs: vector_store.add_documents(current_batch_docs)
            if not vector_store: return None
            vector_store.save_local(index_path)
        except Exception: return None
    
    if vector_store is None: return None
        
    llm_for_rag = ChatOpenAI(temperature=0.4, model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY, max_tokens=1000)
    prompt_template_rag = """Eres ChefAI. Dada la 'Pregunta del Usuario' y el 'Contexto Recuperado' de una receta, presenta la receta del contexto.
    LA PRIMERA LÍNEA DE TU RESPUESTA DEBE SER EL TÍTULO DE LA RECETA DEL CONTEXTO, SIN NINGÚN PREFIJO.
    Luego, presenta el resto de la receta de forma clara.

    Contexto Recuperado:
    {context}

    Pregunta del Usuario:
    {question}

    Respuesta de ChefAI (solo la receta, empezando con el título en la primera línea):"""
    PROMPT_RAG = PromptTemplate(template=prompt_template_rag, input_variables=["context", "question"])
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_for_rag, chain_type="stuff", retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT_RAG}, return_source_documents=True
    )
    return qa_chain


def reformat_and_generate_variants(base_recipe_text_from_rag, original_user_query_with_all_prefs, client_openai=None):
    if not client_openai or not OPENAI_API_KEY:
        base_title = base_recipe_text_from_rag.split('\n', 1)[0].strip() if base_recipe_text_from_rag else "Receta Base"
        return [{"label": "Receta Principal", "text": base_recipe_text_from_rag, "unique_id": "base_0"}] # ID simple


    all_recipes_to_display = []
    recipe_counter = 0
    
    prompt_reformat_and_iterate = f"""
    Eres un chef y editor culinario de élite. Considera la 'Consulta Original del Usuario Completa'.
    Tienes dos tareas:

    TAREA 1: RECETA PRINCIPAL (ADAPTADA)
    Toma la 'Receta Base de RAG' y la 'Consulta Original del Usuario Completa'. Adáptala a CUALQUIER restricción o preferencia dietética mencionada (no hay que adaptar si no se menciona).
    Preséntala con el siguiente formato, asegurando que el título sea lo primero:

    **[TÍTULO ATRACTIVO Y CONCISO PARA ESTA RECETA PRINCIPAL. PUEDES USAR EL TÍTULO DE LA RECETA BASE O MEJORARLO LIGERAMENTE. NO USES LA FRASE 'Título de la Receta:'. EL TÍTULO DEBE SER LA PRIMERA LÍNEA.]**

    **Descripción**: (1-3 frases que describan el plato)

    **Porciones Estimadas**: (Ej: Para 4 personas)

    **Tiempo Estimado**: (Ej: Preparación: 20 min, Cocción: 40 min)

    **Ingredientes**:
    - [Cantidad específica] de [Ingrediente 1] (ej: 200g de pechuga de pollo)
    - [Cantidad específica] de [Ingrediente 2]

    **Pasos de Preparación**:
    1. [Paso 1 claro y conciso]
    2. [Paso 2]
    3. ...

    **Consejo del Chef**: (Opcional)

    TAREA 2: VARIANTE DE LA RECETA PRINCIPAL
    Usando la "Receta Principal (Adaptada y Mejorada)" que acabas de crear, genera UNA variante que sea una iteración o mejora directa, manteniendo la coherencia con TODAS las preferencias de la consulta original (incluyendo restricciones).
    Usa el mismo formato detallado que para la TAREA 1. El título debe ser una variante, inventada por ti, y estar en la primera línea sin prefijos. Pon todos los pasos de preparación completos, aunque sean similares.

    Consulta Original del Usuario Completa: "{original_user_query_with_all_prefs}"
    Receta Base de RAG:
    ---
    {base_recipe_text_from_rag}
    ---
    Devuelve AMBAS recetas (Principal y Variante) separadas EXACTAMENTE y CLARAMENTE por la expresión "---SIGUIENTE RECETA---".
    Recuerda, para ambas recetas, el título va en la primera línea, sin el prefijo "**Título de la Receta**:".
    """

    try:
        response_step1 = client_openai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "Eres un chef y editor culinario de élite."},
                {"role": "user", "content": prompt_reformat_and_iterate}
            ],
            temperature=0.4, max_tokens=2300
        )
        step1_output_text = response_step1.choices[0].message.content
        parts = step1_output_text.split("---SIGUIENTE RECETA---")
        
        reformatted_base_recipe_text = parts[0].strip()
        all_recipes_to_display.append({
            "label": "Receta Principal", 
            "text": reformatted_base_recipe_text, 
            "unique_id": f"recipe_{recipe_counter}"
        })
        recipe_counter += 1

        context_for_next_variant = reformatted_base_recipe_text 

        if len(parts) > 1:
            iterative_variant_text = parts[1].strip()
            all_recipes_to_display.append({
                "label": "Variante de la Receta Principal", 
                "text": iterative_variant_text, 
                "unique_id": f"recipe_{recipe_counter}"
            })
            recipe_counter += 1
        else:
            st.warning("ChefAI no pudo generar una variante directa esta vez.")

        prompt_different_dish = f"""
        Eres un chef recursivo y creativo, experto en dietas.
        La 'Consulta Original del Usuario Completa' es tu guía.
        Ya se ha generado una receta principal.

        Tu tarea AHORA es:
        Proponer un TIPO DE PLATO COMPLETAMENTE DIFERENTE, que no se parezca nada a los anteriores, usando los ingredientes principales de la 'Consulta Original del Usuario Completa', respetando TODAS las restricciones y preferencias.
        Sigue el formato detallado:

        **[INVENTA AQUÍ UN TÍTULO ATRACTIVO Y CONCISO PARA ESTE PLATO ALTERNATIVO. NO USES LA FRASE 'Título de la Receta:'. EL TÍTULO DEBE SER LA PRIMERA LÍNEA.]**

        **Descripción Corta**: (Opcional)

        **Porciones Estimadas**: ...

        **Tiempo Estimado**: ...

        **Ingredientes**: (Con cantidades específicas)
        - ...

        **Pasos de Preparación**:
        1. ...

        **Consejo del Chef**: (Opcional)

        Consulta Original del Usuario Completa: "{original_user_query_with_all_prefs}"
        (Contexto de la Receta Principal, para evitar repetir el mismo tipo de plato):
        ---
        {context_for_next_variant} 
        ---
        Genera UNA nueva receta completa para este plato diferente. NO uses el prefijo "**Título de la Receta**:".
        """
        response_step2 = client_openai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "Eres un chef innovador que transforma ingredientes, respetando dietas."},
                {"role": "user", "content": prompt_different_dish}
            ],
            temperature=0.8, max_tokens=1200
        )
        different_dish_variant_text = response_step2.choices[0].message.content
        all_recipes_to_display.append({
            "label": "Plato Alternativo", 
            "text": different_dish_variant_text, 
            "unique_id": f"recipe_{recipe_counter}"
        })
        recipe_counter += 1
        
        return all_recipes_to_display


    except Exception as e:
        st.error(f"ChefAI tuvo un problema al generar las ideas de recetas: {e}")
        return [{"label": "Receta Principal", "text": base_recipe_text_from_rag, "unique_id": "base_fallback_0"}]

def get_follow_up_answer(recipe_context_str, question_text_str, client_openai=None, is_chat=False):
    if not client_openai or not OPENAI_API_KEY:
        return "Lo siento, no puedo responder en este momento."
    
    system_message = "Eres ChefAI, un asistente de cocina servicial y experto."
    if is_chat:
        system_message += " Estás en un chat. Limita tus respuestas estrictamente a preguntas relacionadas con la receta proporcionada o temas culinarios directamente derivados de ella que no estén ya en la información de la receta previamente puesta. Si la pregunta se desvía, indica amablemente que solo puedes ayudar con la receta actual y temas de cocina."

    full_prompt_text = f"""
    Contexto de la Receta Actual:
    ---
    {recipe_context_str}
    ---
    Pregunta del Usuario: {question_text_str}
    Tu Respuesta (concisa, útil y, si es un chat, limitada al tema de la receta y cocina):
    """
    try:
        response = client_openai.chat.completions.create( 
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": full_prompt_text}
            ],
            temperature=0.4, max_tokens=600
        )
        return response.choices[0].message.content
    except Exception:
        return "Lo siento, tuve un problema al procesar tu pregunta."

if not OPENAI_API_KEY or not client:
    st.error("La API Key de OpenAI no está configurada. La aplicación no puede funcionar.")
    st.stop()

if not st.session_state.recipes_data_loaded:
    st.session_state.processed_recipes_data = load_and_prepare_dataset_local(local_file_path=NOMBRE_ARCHIVO_LOCAL_RECETAS)
    if st.session_state.processed_recipes_data:
        st.session_state.recipes_data_loaded = True
    else:
        st.error(f"No se pudieron cargar las recetas desde '{NOMBRE_ARCHIVO_LOCAL_RECETAS}'.")
        st.stop()

if st.session_state.rag_chain is None:
    embeddings_model_global = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    with st.spinner("ChefAI está organizando su conocimiento culinario..."):
        st.session_state.rag_chain = initialize_rag_system(
            st.session_state.processed_recipes_data, 
            embeddings_model_global,
            FAISS_INDEX_PATH
        )
    if not st.session_state.rag_chain:
        st.error("ChefAI no pudo preparar su conocimiento. Intenta recargar.")
        st.stop()

st.title("👨‍🍳 ChefAI: Tu Asistente Culinario Inteligente")
st.markdown("Dime qué ingredientes tienes y tus preferencias, ¡y te daré algunas ideas deliciosas!")

with st.form(key="recipe_search_form_v2"):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ingredientes en tu Cocina")
        ingredients_text_input = st.text_area(
            "Escribe tus ingredientes...", height=123, key="ingredients_input_form_v2",
            placeholder="Ej: pollo, brócoli, arroz, curry")
    with col2:
        st.subheader("Preferencias Adicionales")
        cuisine_type_select = st.selectbox(
            "Estilo de cocina (opcional):",
            ["Cualquiera", "Española", "Italiana", "Mexicana", "Asiática", "Mediterránea", "India", "Fusión"],
            key="cuisine_select_form_v2")
        dietary_restrictions_select = st.multiselect(
            "Restricciones o preferencias dietéticas (opcional):",
            ["Ninguna", "Vegetariano", "Vegano", "Sin Gluten", "Sin Lactosa", "Bajo en Calorías", "Alto en Proteínas"],
            key="dietary_restrictions_form_v2", default=[])
    
    submitted = st.form_submit_button("Encontrar Recetas", type="primary", use_container_width=True)

main_recipe_display_area = st.container()

if submitted:
    actual_dietary_restrictions = [r for r in dietary_restrictions_select if r != "Ninguna"] if dietary_restrictions_select else []
    
    st.session_state.displayed_recipes = []
    st.session_state.chat_histories = {} 
    st.session_state.generated_questions_for_recipe = {}
    st.session_state.sugg_question_clicked_info = None

    if not ingredients_text_input:
        st.warning("Por favor, dime al menos un ingrediente para empezar.")
    elif st.session_state.rag_chain is None:
        st.error("ChefAI no está listo. Intenta recargar la página.")
    else:
        with st.spinner("ChefAI está buscando la inspiración perfecta para ti... 🍳"):
            user_query_for_rag = f"Receta que use: {ingredients_text_input}."
            if cuisine_type_select != "Cualquiera": 
                user_query_for_rag += f" Con un toque de cocina {cuisine_type_select}."
            
            try:
                rag_response = st.session_state.rag_chain({"query": user_query_for_rag})
                base_recipe_from_rag = rag_response.get("result") 
                
                if base_recipe_from_rag:
                    full_user_query_for_variants = f"Ingredientes principales: {ingredients_text_input}."
                    if cuisine_type_select != "Cualquiera": 
                        full_user_query_for_variants += f" Preferencia de cocina: {cuisine_type_select}."
                    if actual_dietary_restrictions: 
                        full_user_query_for_variants += f" Restricciones dietéticas: {', '.join(actual_dietary_restrictions)}."

                    all_recipes_to_display = reformat_and_generate_variants(
                        base_recipe_text_from_rag=base_recipe_from_rag,
                        original_user_query_with_all_prefs=full_user_query_for_variants,
                        client_openai=client
                    )
                    st.session_state.displayed_recipes = all_recipes_to_display

                    if not all_recipes_to_display:
                        st.error("Lo siento, no pude generar ideas de recetas esta vez.")
                        base_title_rag = base_recipe_from_rag.split('\n', 1)[0].strip()
                        st.session_state.displayed_recipes = [{
                            "label": "Receta Principal", 
                            "text": base_recipe_from_rag, 
                            "title_for_key": "".join(filter(str.isalnum, base_title_rag)).lower()[:30]
                        }]
                else:
                    st.error("Lo siento, no encontré una receta base con esos criterios.")
            except Exception as e:
                st.error(f"ChefAI tuvo un contratiempo: {e}")

if st.session_state.get('sugg_question_clicked_info'):
    clicked_info = st.session_state.sugg_question_clicked_info
    recipe_key_clicked = clicked_info["recipe_key"]
    question_to_process = clicked_info["question"]
    context_to_use = clicked_info["context"]
    
    st.session_state.sugg_question_clicked_info = None 
    
    if client and OPENAI_API_KEY: 
        answer_str = get_follow_up_answer(context_to_use, question_to_process, client)
        
        if recipe_key_clicked not in st.session_state.chat_histories:
            st.session_state.chat_histories[recipe_key_clicked] = []
            
        st.session_state.chat_histories[recipe_key_clicked].append({"role": "assistant", "content": answer_str})
        st.session_state.chat_histories[recipe_key_clicked].append({"role": "user", "content": question_to_process})
       
if st.session_state.get('displayed_recipes'):
    recipe_tabs_data = st.session_state.displayed_recipes
    recipe_labels_for_tabs = [recipe["label"] for recipe in recipe_tabs_data]
    
    if recipe_labels_for_tabs:
        tabs_components = st.tabs(recipe_labels_for_tabs)
        
        for i, recipe_info in enumerate(recipe_tabs_data):
            with tabs_components[i]:
                lines = recipe_info['text'].strip().split('\n')
                current_tab_recipe_title = lines[0].replace("**", "").strip() if lines else recipe_info['label']
                current_tab_recipe_context = recipe_info['text']
                recipe_content_without_title = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

                st.markdown(f"## **{current_tab_recipe_title}**")
                st.markdown(recipe_content_without_title)

                recipe_key = recipe_info.get("unique_id", f"fallback_recipe_key_{i}")

                if recipe_key not in st.session_state.chat_histories:
                    st.session_state.chat_histories[recipe_key] = []
                if recipe_key not in st.session_state.generated_questions_for_recipe:
                    st.session_state.generated_questions_for_recipe[recipe_key] = []

                st.markdown("---")

                if not st.session_state.generated_questions_for_recipe.get(recipe_key):
                    if client and OPENAI_API_KEY:
                        prompt_suggest_questions = f"""Para la receta '{current_tab_recipe_title}': {current_tab_recipe_context}... 
                        Genera 3 preguntas cortas (máx 12 palabras) sobre esta receta. Una DEBE ser sobre maridaje.
                        Devuelve solo la lista de preguntas, cada una en una nueva línea, sin numeración/viñetas."""
                        try:
                            response_questions = client.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt_suggest_questions}], temperature=0.6, max_tokens=120)
                            suggested_questions_text = response_questions.choices[0].message.content
                            st.session_state.generated_questions_for_recipe[recipe_key] = [q.strip() for q in suggested_questions_text.split('\n') if q.strip() and '?' in q and 5 < len(q.strip()) <= 80]
                        except: st.session_state.generated_questions_for_recipe[recipe_key] = []
                
                current_suggested_questions = st.session_state.generated_questions_for_recipe.get(recipe_key, [])
                if not current_suggested_questions: 
                    current_suggested_questions = [
                        f"¿Bebida para '{current_tab_recipe_title[:50]}...'?",
                        f"¿Versión vegetariana?",
                        f"¿Conservación?"
                    ]
                
                if current_suggested_questions:
                    num_cols_sugg = min(len(current_suggested_questions), 4)
                    button_cols_sugg = st.columns(num_cols_sugg)
                    for idx, question_str in enumerate(current_suggested_questions):
                        button_key_sugg = f"sugg_q_btn_{recipe_key}_{idx}"
                        if button_cols_sugg[idx % num_cols_sugg].button(question_str, key=button_key_sugg):
                            st.session_state.sugg_question_clicked_info = {
                                "recipe_key": recipe_key, 
                                "question": question_str,
                                "context": current_tab_recipe_context
                            }
                            st.rerun()
                
                st.markdown(f"**Chatea con ChefAI sobre '{current_tab_recipe_title}':**")
                
                chat_input_key = f"chat_input_{recipe_key}"
                if user_chat_this_tab := st.chat_input(f"Tu pregunta sobre '{current_tab_recipe_title}'...", key=chat_input_key):
                    if client and OPENAI_API_KEY:
                        with st.spinner("ChefAI respondiendo..."):
                            assistant_response = get_follow_up_answer(current_tab_recipe_context, user_chat_this_tab, client, is_chat=True)
                            st.session_state.chat_histories[recipe_key].append({"role": "assistant", "content": assistant_response})
                        st.session_state.chat_histories[recipe_key].append({"role": "user", "content": user_chat_this_tab})
                        st.rerun()

                current_chat_history = st.session_state.chat_histories.get(recipe_key, [])
                chat_history_container = st.container()
                with chat_history_container:
                    for message in reversed(current_chat_history): 
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])