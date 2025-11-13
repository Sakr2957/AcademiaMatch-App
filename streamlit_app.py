import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
import re
import io

# Page config
st.set_page_config(
    page_title="Research Alignment Matcher",
    page_icon="🔬",
    layout="wide"
)

# Initialize model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_text(text):
    """Preprocess text for matching"""
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
    tokens = [word for word in tokens if word not in stopwords and len(word) > 2]
    
    return ' '.join(tokens)

def create_sample_data():
    """Create sample data for demonstration"""
    internal_data = {
        'internal_name': [
            'Dr. Sarah Thompson',
            'Dr. Michael Lee',
            'Dr. Priya Nair',
            'Dr. James O\'Connell',
            'Dr. Aisha Rahman'
        ],
        'department': [
            'Chemistry',
            'Computer Science',
            'Environmental Science',
            'Psychology',
            'Mechanical Engineering'
        ],
        'expertise_summary': [
            'Sustainable catalysis and green chemistry approaches for industrial applications',
            'Machine learning algorithms and ethical AI development for social good',
            'Climate change impacts on biodiversity and ecosystem conservation strategies',
            'Cognitive neuroscience and behavioral modeling in decision-making processes',
            'Advanced materials for robotics and autonomous manufacturing systems'
        ]
    }
    
    external_data = {
        'external_name': [
            'Dr. Emily Chen',
            'Dr. Omar Yusuf',
            'Dr. Rachel Stein',
            'Dr. David Park',
            'Dr. Lina Morales'
        ],
        'affiliation': [
            'GreenTech Research Institute',
            'AI for Humanity Lab',
            'Global Climate Alliance',
            'Cognitive Systems Research Centre',
            'Advanced Robotics Group'
        ],
        'research_interest_summary': [
            'Developing novel catalysts for hydrogen production and sustainable chemical reactions',
            'AI-driven data analytics and the development of fair machine learning systems',
            'Environmental policy development and conservation of endangered ecosystems',
            'Modeling human cognition and understanding neural processes in behavior',
            'Intelligent robotic systems for advanced manufacturing and automation'
        ]
    }
    
    return pd.DataFrame(internal_data), pd.DataFrame(external_data)

def match_researchers(internal_df, external_df, method='embeddings', top_n=3, threshold=0.0):
    """Match external researchers with internal researchers"""
    
    # Preprocess text
    internal_df['processed_text'] = internal_df['expertise_summary'].apply(preprocess_text)
    external_df['processed_text'] = external_df['research_interest_summary'].apply(preprocess_text)
    
    # Generate embeddings
    if method == 'embeddings':
        model = load_model()
        with st.spinner('Generating semantic embeddings...'):
            internal_embeddings = model.encode(internal_df['expertise_summary'].tolist())
            external_embeddings = model.encode(external_df['research_interest_summary'].tolist())
    else:  # TF-IDF
        with st.spinner('Computing TF-IDF vectors...'):
            all_texts = (internal_df['processed_text'].tolist() + 
                        external_df['processed_text'].tolist())
            vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            vectorizer.fit(all_texts)
            internal_embeddings = vectorizer.transform(internal_df['processed_text']).toarray()
            external_embeddings = vectorizer.transform(external_df['processed_text']).toarray()
    
    # Calculate similarity
    with st.spinner('Calculating similarity scores...'):
        similarity_matrix = cosine_similarity(external_embeddings, internal_embeddings)
    
    # Extract top matches
    results = []
    for ext_idx, ext_row in external_df.iterrows():
        similarities = similarity_matrix[ext_idx]
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        for rank, int_idx in enumerate(top_indices, 1):
            int_row = internal_df.iloc[int_idx]
            score = float(similarities[int_idx])
            
            if score >= threshold:
                results.append({
                    'external_name': ext_row['external_name'],
                    'external_affiliation': ext_row['affiliation'],
                    'external_research_interest': ext_row['research_interest_summary'],
                    'match_rank': rank,
                    'internal_name': int_row['internal_name'],
                    'internal_department': int_row['department'],
                    'internal_expertise': int_row['expertise_summary'],
                    'similarity_score': score
                })
    
    results_df = pd.DataFrame(results)
    avg_similarity = results_df['similarity_score'].mean() if len(results_df) > 0 else 0.0
    
    return results_df, similarity_matrix, avg_similarity

# Header
st.title("🔬 Research Alignment Matcher")
st.markdown("**AI-Powered Research Collaboration Matching**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    method = st.selectbox(
        "Matching Method",
        ["embeddings", "tfidf"],
        format_func=lambda x: "Sentence Transformers (Recommended)" if x == "embeddings" else "TF-IDF",
        help="Sentence Transformers uses semantic embeddings for better matching"
    )
    
    top_n = st.slider("Top N Matches", 1, 10, 3, help="Number of matches to show per external researcher")
    threshold = st.slider("Similarity Threshold (%)", 0, 100, 0, step=5, help="Minimum similarity score to include") / 100
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info(
        "This tool uses advanced NLP to match external researchers with internal faculty based on research interests. "
        "Upload CSV files or use sample data to get intelligent matching results instantly."
    )

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload & Match", "📊 Results", "📈 Visualizations"])

with tab1:
    st.header("Upload CSV Files")
    
    # Option to use sample data
    use_sample = st.checkbox("Use sample data for demonstration", value=True)
    
    if use_sample:
        internal_df, external_df = create_sample_data()
        st.success(f"✅ Using sample data: {len(internal_df)} internal researchers, {len(external_df)} external researchers")
        
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("Preview Internal Researchers"):
                st.dataframe(internal_df)
        with col2:
            with st.expander("Preview External Researchers"):
                st.dataframe(external_df)
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Internal Researchers")
            st.caption("Required columns: internal_name, department, expertise_summary")
            
            internal_file = st.file_uploader(
                "Upload Internal Researchers CSV",
                type=['csv'],
                key='internal',
                help="CSV file containing internal researcher information"
            )
            
            if internal_file is not None:
                try:
                    internal_df = pd.read_csv(internal_file)
                    st.success(f"✅ Loaded {len(internal_df)} internal researchers")
                    with st.expander("Preview Data"):
                        st.dataframe(internal_df.head())
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                    internal_df = None
            else:
                internal_df = None
        
        with col2:
            st.subheader("External Researchers")
            st.caption("Required columns: external_name, affiliation, research_interest_summary")
            
            external_file = st.file_uploader(
                "Upload External Researchers CSV",
                type=['csv'],
                key='external',
                help="CSV file containing external researcher information"
            )
            
            if external_file is not None:
                try:
                    external_df = pd.read_csv(external_file)
                    st.success(f"✅ Loaded {len(external_df)} external researchers")
                    with st.expander("Preview Data"):
                        st.dataframe(external_df.head())
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                    external_df = None
            else:
                external_df = None
    
    st.markdown("---")
    
    # Run matching
    if st.button("🚀 Run Matching Algorithm", type="primary", use_container_width=True):
        if internal_df is not None and external_df is not None:
            try:
                # Validate columns
                required_internal = ['internal_name', 'department', 'expertise_summary']
                required_external = ['external_name', 'affiliation', 'research_interest_summary']
                
                missing_internal = [col for col in required_internal if col not in internal_df.columns]
                missing_external = [col for col in required_external if col not in external_df.columns]
                
                if missing_internal:
                    st.error(f"❌ Missing columns in internal file: {', '.join(missing_internal)}")
                elif missing_external:
                    st.error(f"❌ Missing columns in external file: {', '.join(missing_external)}")
                else:
                    # Run matching
                    results_df, similarity_matrix, avg_similarity = match_researchers(
                        internal_df, external_df, method, top_n, threshold
                    )
                    
                    # Store in session state
                    st.session_state['results_df'] = results_df
                    st.session_state['similarity_matrix'] = similarity_matrix
                    st.session_state['avg_similarity'] = avg_similarity
                    st.session_state['internal_df'] = internal_df
                    st.session_state['external_df'] = external_df
                    
                    st.success(f"✅ Matching complete! Found {len(results_df)} matches with average similarity of {avg_similarity:.1%}")
                    st.balloons()
                    
            except Exception as e:
                st.error(f"❌ Error during matching: {e}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Please upload both CSV files or enable sample data")

with tab2:
    st.header("Matching Results")
    
    if 'results_df' in st.session_state and len(st.session_state['results_df']) > 0:
        results_df = st.session_state['results_df']
        avg_similarity = st.session_state['avg_similarity']
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Matches", len(results_df))
        with col2:
            st.metric("Avg Similarity", f"{avg_similarity:.1%}")
        with col3:
            high_quality = len(results_df[results_df['similarity_score'] > 0.5])
            st.metric("High Quality (>50%)", high_quality)
        with col4:
            unique_external = results_df['external_name'].nunique()
            st.metric("External Researchers", unique_external)
        
        st.markdown("---")
        
        # Detailed results
        st.subheader("Detailed Match Results")
        
        # Format for display
        display_df = results_df.copy()
        display_df['similarity_score'] = display_df['similarity_score'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="matching_results.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Top matches summary
        st.markdown("---")
        st.subheader("Top Matches Summary")
        
        top_matches = results_df[results_df['match_rank'] == 1].sort_values('similarity_score', ascending=False)
        
        for idx, row in top_matches.iterrows():
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{row['external_name']}** → **{row['internal_name']}**")
                    st.caption(f"{row['internal_department']} | {row['external_affiliation']}")
                with col2:
                    score_color = "green" if row['similarity_score'] > 0.7 else "orange" if row['similarity_score'] > 0.5 else "red"
                    st.markdown(f":{score_color}[**{row['similarity_score']:.1%}**]")
                st.markdown("---")
    else:
        st.info("👆 Run the matching algorithm in the 'Upload & Match' tab to see results here")

with tab3:
    st.header("Visualizations")
    
    if 'similarity_matrix' in st.session_state:
        similarity_matrix = st.session_state['similarity_matrix']
        internal_df = st.session_state['internal_df']
        external_df = st.session_state['external_df']
        results_df = st.session_state['results_df']
        
        # Heatmap
        st.subheader("Similarity Heatmap")
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=[name.split()[-1] for name in internal_df['internal_name']],
            y=[name.split()[-1] for name in external_df['external_name']],
            colorscale='RdYlGn',
            text=similarity_matrix,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Similarity")
        ))
        
        fig_heatmap.update_layout(
            title="Research Alignment Similarity Matrix",
            xaxis_title="Internal Researchers",
            yaxis_title="External Researchers",
            height=500
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Distribution
        st.subheader("Similarity Score Distribution")
        
        fig_dist = px.histogram(
            results_df,
            x='similarity_score',
            nbins=20,
            title="Distribution of Similarity Scores",
            labels={'similarity_score': 'Similarity Score', 'count': 'Frequency'},
            color_discrete_sequence=['#636EFA']
        )
        
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Top matches bar chart
        st.subheader("Top Matches by External Researcher")
        
        top_matches = results_df[results_df['match_rank'] == 1].sort_values('similarity_score', ascending=True)
        
        fig_bar = px.bar(
            top_matches,
            y='external_name',
            x='similarity_score',
            orientation='h',
            title="Best Match Similarity for Each External Researcher",
            labels={'similarity_score': 'Similarity Score', 'external_name': 'External Researcher'},
            color='similarity_score',
            color_continuous_scale='RdYlGn'
        )
        
        fig_bar.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    else:
        st.info("👆 Run the matching algorithm in the 'Upload & Match' tab to see visualizations here")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "© 2025 Research Alignment Matcher | Powered by Sentence Transformers & Advanced NLP"
    "</div>",
    unsafe_allow_html=True
)
