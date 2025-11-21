import streamlit as st
import re
import sys
import os
import pandas as pd
import altair as alt
import time
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.search_engine import SearchEngine
from src.config import RAW_DATA_DIR

# Page Config
st.set_page_config(
    page_title="SemanticCache", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stTextInput > div > div > input {
        font-size: 1.2rem;
        padding: 12px;
        border-radius: 10px;
    }
    .result-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .result-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #fafafa;
        margin-bottom: 5px;
    }
    .result-meta {
        font-size: 0.85rem;
        color: #a0a0a0;
        margin-bottom: 10px;
    }
    .highlight {
        background-color: #ff4b4b40;
        color: #ff4b4b;
        padding: 0 2px;
        border-radius: 3px;
        font-weight: bold;
    }
    .score-badge {
        background-color: #1f77b4;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-right: 5px;
    }
    .keyword-badge {
        background-color: #2ca02c;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Engine
@st.cache_resource
def load_engine():
    return SearchEngine()

try:
    engine = load_engine()
except Exception as e:
    st.error(f"Failed to load search engine: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.title("🧠 SemanticCache")
    st.markdown("---")
    
    st.subheader("⚙️ Search Configuration")
    k = st.slider("Results to Retrieve (k)", 1, 50, 5)
    alpha = st.slider("Hybrid Weight (Alpha)", 0.0, 1.0, 0.5, 0.1, help="0.0 = Keyword (BM25), 1.0 = Vector (FAISS)")
    rerank = st.toggle("Enable Cross-Encoder Re-ranking", value=True)
    
    st.markdown("---")
    st.subheader("📊 Index Stats")
    if engine.index:
        st.metric("Total Documents", engine.index.ntotal)
        st.metric("Embedding Dimension", engine.index.d)
    else:
        st.warning("Index not loaded")
        
    st.markdown("---")
    st.info("Built with FastAPI, FAISS, Sentence-Transformers & Streamlit")

# Main Content
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100) # Generic AI icon
with col2:
    st.title("Intelligent Document Search")
    st.markdown("Experience the power of **Hybrid Search** combining dense vector embeddings with sparse keyword matching.")

# Search Bar
query = st.text_input("", placeholder="Ask a question or search for a topic...", help="Type your query and hit enter")

def highlight_text(text, query):
    # Escape HTML characters in the original text first to prevent injection
    import html
    text = html.escape(text)
    
    terms = query.split()
    for term in terms:
        if len(term) < 2: continue
        escaped_term = re.escape(html.escape(term))
        # Use a unique placeholder to avoid recursive replacement issues
        text = re.sub(f"({escaped_term})", r'__HIGHLIGHT_START__\1__HIGHLIGHT_END__', text, flags=re.IGNORECASE)
    
    # Replace placeholders with actual HTML
    text = text.replace('__HIGHLIGHT_START__', '<span class="highlight">').replace('__HIGHLIGHT_END__', '</span>')
    return text

if query:
    start_time = time.time()
    with st.spinner("Running Hybrid Search & Re-ranking..."):
        results = engine.search(query, k=k, alpha=alpha, rerank=rerank)
    end_time = time.time()
    duration = end_time - start_time
    
    if results:
        st.markdown(f"Found **{len(results)}** results in **{duration:.3f}s**")
        
        tab1, tab2, tab3 = st.tabs(["📄 Results", "📈 Analytics", "🔍 Debug Data"])
        
        with tab1:
            # Export Options
            if results:
                df_results = pd.DataFrame(results)
                # Select relevant columns for export
                export_cols = ['filename', 'score', 'bm25_score', 'vector_score', 'content', 'explanation']
                df_export = df_results[export_cols] if all(col in df_results.columns for col in export_cols) else df_results
                
                c1, c2 = st.columns([1, 5])
                with c1:
                    st.download_button(
                        label="📥 Export CSV",
                        data=df_export.to_csv(index=False).encode('utf-8'),
                        file_name='search_results.csv',
                        mime='text/csv'
                    )
                with c2:
                    st.download_button(
                        label="📥 Export JSON",
                        data=df_export.to_json(orient='records', indent=2),
                        file_name='search_results.json',
                        mime='application/json'
                    )

            for res in results:
                highlighted_content = highlight_text(res['content'], query)
                explanation = res.get('explanation', 'No explanation available.')
                
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">📄 {res['filename']}</div>
                    <div class="result-meta">
                        <span class="score-badge">Score: {res['score']:.4f}</span>
                        <span class="keyword-badge">BM25: {res['bm25_score']:.4f}</span>
                        <span class="score-badge" style="background-color: #9467bd">Vector: {res['vector_score']:.4f}</span>
                    </div>
                    <div style="font-size: 0.9rem; color: #4caf50; margin-bottom: 8px;">💡 {explanation}</div>
                    <div style="color: #dcdcdc; line-height: 1.6;">{highlighted_content}</div>
                </div>
                """, unsafe_allow_html=True)

                # Document Preview Modal (Expander)
                with st.expander(f"👁️ View Full Document: {res['filename']}"):
                    try:
                        file_path = RAW_DATA_DIR / res['filename']
                        if file_path.exists():
                            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                full_content = f.read()
                            st.code(full_content, language='text')
                        else:
                            st.warning("File not found on disk.")
                    except Exception as e:
                        st.error(f"Could not read file: {e}")
        
        with tab2:
            st.subheader("Score Distribution")
            
            # Prepare data for chart
            data = []
            for res in results:
                data.append({"Filename": res['filename'], "Type": "Vector Score", "Score": res['vector_score']})
                data.append({"Filename": res['filename'], "Type": "BM25 Score", "Score": res['bm25_score']})
                data.append({"Filename": res['filename'], "Type": "Final Score", "Score": res['score']})
            
            df = pd.DataFrame(data)
            
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('Score:Q'),
                y=alt.Y('Filename:N', sort='-x'),
                color='Type:N',
                tooltip=['Filename', 'Type', 'Score']
            ).properties(height=300)
            
            st.altair_chart(chart, use_container_width=True)
            
            st.divider()
            st.subheader("☁️ Word Cloud")
            
            # Aggregate text from results
            all_text = " ".join([res['content'] for res in results])
            
            if all_text:
                try:
                    wordcloud = WordCloud(width=800, height=400, background_color='#0e1117', colormap='viridis').generate(all_text)
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    # Set background to match Streamlit dark theme
                    fig.patch.set_facecolor('#0e1117')
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.error(f"Could not generate word cloud: {e}")
            else:
                st.info("Not enough text to generate word cloud.")
            
        with tab3:
            st.subheader("Raw JSON Response")
            st.json(results)
            
    else:
        st.warning("No results found. Try adjusting your query or the hybrid weight.")
        st.image("https://cdn-icons-png.flaticon.com/512/6134/6134065.png", width=200)

else:
    # Empty State
    st.markdown("### Try searching for:")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Artificial Intelligence"):
            pass # In a real app, this would trigger search
    with c2:
        if st.button("World War II"):
            pass
    with c3:
        if st.button("Neural Networks"):
            pass

