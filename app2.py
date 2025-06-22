import streamlit as st
import os
from dotenv import load_dotenv
import json
import requests
import folium
from streamlit_folium import st_folium

# Import LangChain components
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
JSON_PATH = "countries2.json"
VECTOR_STORE_PATH = "faiss_index_final"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HF_MODEL_REPO_ID = "HuggingFaceH4/zephyr-7b-beta"

# ==============================================================================
#      CORE LOGIC FUNCTIONS (No changes in this section)
# ==============================================================================

@st.cache_resource
def create_vector_store():
    """Creates a FAISS vector store with rich metadata from the JSON file."""
    with st.spinner("Performing first-time setup: Creating new vector store..."):
        documents = []
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for country_data in data['countries']:
            page_content = country_data["description"]
            metadata = country_data.get("metadata", {})
            metadata["source_country"] = country_data["name"]
            documents.append(Document(page_content=page_content, metadata=metadata))

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(VECTOR_STORE_PATH)
    st.success(f"New vector store created successfully at: {VECTOR_STORE_PATH}")

def filter_source_data(source_data, interests, budget, travel_months):
    filtered_data = []
    for country in source_data:
        meta = country.get("metadata", {})
        country_budget = str(meta.get("budget_tier", "")).lower()
        passes_budget = not budget or country_budget == budget
        if not passes_budget:
            continue
        try:
            country_best_months = [int(m) for m in meta.get("best_months", [])]
            user_travel_months = [int(m) for m in travel_months]
            passes_date = not user_travel_months or any(month in country_best_months for month in user_travel_months)
        except (ValueError, TypeError):
            passes_date = False
        if not passes_date:
            continue
        country_interests = [str(i).strip().lower() for i in meta.get("interests", [])]
        user_interests = [str(i).strip().lower() for i in interests]
        passes_interests = not user_interests or any(interest in country_interests for interest in user_interests)
        if not passes_interests:
            continue
        filtered_data.append(country)
    return filtered_data

@st.cache_data(ttl=600)
def get_weather(city_name, country_code):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key or not city_name: return None
    query = f"{city_name},{country_code}"
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": query, "appid": api_key, "units": "metric"}
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return {
            "description": data["weather"][0]["description"].title(), 
            "temperature": f"{round(data['main']['temp'])}°C", 
            "icon": f"http://openweathermap.org/img/wn/{data['weather'][0]['icon']}@2x.png"
        }
    except requests.exceptions.RequestException as e:
        print(f"Weather API error for {query}: {e}")
        return None

@st.cache_resource
def load_components():
    load_dotenv()
    if "HUGGINGFACEHUB_API_TOKEN" not in os.environ or "OPENWEATHER_API_KEY" not in os.environ:
        st.error("🚨 API key(s) not found. Please set HUGGINGFACEHUB_API_TOKEN and OPENWEATHER_API_KEY in your .env file.")
        st.stop()
    try:
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            source_of_truth_data = json.load(f)['countries']
    except Exception as e:
        st.error(f"Failed to load or parse {JSON_PATH}: {e}")
        st.stop()
    llm = HuggingFaceEndpoint(repo_id=HF_MODEL_REPO_ID, temperature=0.5, max_new_tokens=512)
    return source_of_truth_data, llm

# ==============================================================================
#                         STREAMLIT FRONTEND
# ==============================================================================

st.set_page_config(page_title="AI Travel Recommender", page_icon="🌍", layout="wide")
st.title("🌍 AI-Powered Travel Recommender")
st.markdown("Find your next destination based on your personal preferences and live weather updates.")

if not os.path.exists(VECTOR_STORE_PATH):
    st.info(f"Vector store not found. Creating a new one from `{JSON_PATH}`.")
    create_vector_store()

try:
    source_data, llm = load_components()
except Exception as e:
    st.error(f"Failed to load AI components: {e}", icon="🚨")
    st.stop()

# --- NEW: Initialize session state to hold results ---
if 'recommendations_generated' not in st.session_state:
    st.session_state.recommendations_generated = False
    st.session_state.final_docs = []
    st.session_state.pre_filtered_data = []

with st.sidebar:
    st.header("Your Travel Preferences")
    available_interests = sorted(["trekking", "skiing", "beaches", "history", "scuba diving", "shopping", "mountains", "culture", "food", "wildlife", "city life"])
    interests = st.multiselect("What are your interests?", options=available_interests)
    budget = st.select_slider("What's your budget level?", options=["low", "medium", "high"])
    st.markdown("**When do you want to travel?**")
    month_map = {name: i+1 for i, name in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
    month_list = list(month_map.keys())
    col1, col2 = st.columns(2)
    with col1:
        start_month_name = st.selectbox("From", options=month_list, index=6)
    with col2:
        start_index = month_list.index(start_month_name)
        end_month_name = st.selectbox("To", options=month_list[start_index:], index=min(2, len(month_list[start_index:])-1))
    start_month_num = month_map[start_month_name]
    end_month_num = month_map[end_month_name]
    travel_months = list(range(start_month_num, end_month_num + 1))
    other_details = st.text_area("Any other details? (e.g., 'I want a quiet place with great food')")
    submit_button = st.button("Get Recommendations")

# --- RESTRUCTURED LOGIC: Part 1 - Calculation ---
# This part only runs when the button is clicked. It calculates results and saves them.
if submit_button:
    if not interests and not other_details:
        st.warning("Please select at least one interest or provide some details.", icon="⚠️")
        st.session_state.recommendations_generated = False # Reset if no input
    else:
        with st.spinner("🧠 Finding the perfect destinations for you..."):
            pre_filtered = filter_source_data(source_data, interests, budget, travel_months)
            
            if not pre_filtered:
                st.error("No destinations found matching your specific criteria.")
                st.info("Try broadening your search (e.g., select fewer interests or a wider date range).")
                st.session_state.recommendations_generated = False # Reset on no results
            else:
                filtered_docs = []
                for country in pre_filtered:
                    metadata = country.get("metadata", {}).copy()
                    metadata["source_country"] = country["name"] 
                    filtered_docs.append(Document(page_content=country["description"], metadata=metadata))
                
                final_results = []
                if len(filtered_docs) > 0:
                    filtered_vector_store = FAISS.from_documents(filtered_docs, HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))
                    retriever = filtered_vector_store.as_retriever(search_kwargs={"k": 3})
                    semantic_query = other_details if other_details else " ".join(interests)
                    final_results = retriever.invoke(semantic_query)
                
                # --- NEW: Store results in session state ---
                st.session_state.final_docs = final_results
                st.session_state.pre_filtered_data = pre_filtered
                st.session_state.recommendations_generated = True

# --- RESTRUCTURED LOGIC: Part 2 - Display ---
# This part runs if results have been generated, ensuring they stay on screen.
if st.session_state.recommendations_generated:
    st.markdown("### ✨ Here are your personalized recommendations:")

    final_docs = st.session_state.final_docs
    pre_filtered_data = st.session_state.pre_filtered_data

    if not final_docs and not pre_filtered_data:
        # This case should ideally not be hit due to checks above, but as a safeguard:
        st.error("Something went wrong. Please try your search again.")
    elif not final_docs:
        st.warning("Could not find a specific match for your text query, but here are some options based on your filters:")
        for country in pre_filtered_data[:3]:
            st.subheader(country["name"])
            st.markdown(country["description"])
            st.divider()
    else:
        for doc in final_docs:
            meta = doc.metadata
            country_name = meta.get("source_country", "N/A")
            capital_city = meta.get("capital", "N/A")
            
            # --- NEW: Two-column layout for text and individual map ---
            col1, col2 = st.columns([3, 2]) # 60% width for text, 40% for map

            with col1:
                st.subheader(country_name)
                st.markdown(doc.page_content)
                
                weather_info = get_weather(capital_city, meta.get("country_code"))
                if weather_info and capital_city:
                    st.markdown(f"**Weather in {capital_city}**")
                    weather_col1, weather_col2 = st.columns([1,4])
                    with weather_col1:
                        st.image(weather_info['icon'], width=60)
                    with weather_col2:
                        st.metric(label=weather_info['description'], value=weather_info['temperature'])
                else:
                    st.caption("Weather not available")

            with col2:
                # --- NEW: Create and display an individual map for each location ---
                if "lat" in meta and "lon" in meta:
                    map_location = [meta["lat"], meta["lon"]]
                    
                    # Create a new map centered on the destination
                    m = folium.Map(location=map_location, zoom_start=6, tiles="CartoDB positron")
                    
                    # Add a marker
                    popup_html = f"<b>{country_name}</b><br>Capital: {capital_city}"
                    folium.Marker(
                        location=map_location,
                        popup=folium.Popup(popup_html, max_width=200),
                        tooltip=country_name,
                        icon=folium.Icon(color='blue', icon='globe', prefix='fa')
                    ).add_to(m)
                    
                    # Render the map in Streamlit, ensuring a unique key for each map
                    st_folium(m, use_container_width=True, height=350, key=f"map_{country_name}")
                else:
                    st.info("No location data to display on map.")
            
            st.divider()