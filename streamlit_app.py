import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import base64
from pathlib import Path

# Page config
st.set_page_config(
    page_title="ScholarSync - Academic Collaboration Platform",
    page_icon="🎓",
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

def match_datasets(internal_df, external_df, method='embeddings', top_n=3, threshold=0.0):
    """Match external dataset with internal dataset - returns EXACTLY top_n matches per external item"""
    
    # Preprocess text
    internal_df['processed_text'] = internal_df['expertise_summary'].apply(preprocess_text)
    external_df['processed_text'] = external_df['research_interest_summary'].apply(preprocess_text)
    
    # Generate embeddings
    if method == 'embeddings':
        model = load_model()
        with st.spinner('Generating semantic embeddings...'):
            internal_embeddings = model.encode(internal_df['expertise_summary'].tolist())
            external_embeddings = model.encode(external_df['research_interest_summary'].tolist())
    else:
        st.warning("LLM method is under construction. Using Sentence Transformers instead.")
        model = load_model()
        internal_embeddings = model.encode(internal_df['expertise_summary'].tolist())
        external_embeddings = model.encode(external_df['research_interest_summary'].tolist())
    
    # Calculate similarity
    with st.spinner('Calculating similarity scores...'):
        similarity_matrix = cosine_similarity(external_embeddings, internal_embeddings)
    
    # FIXED: Extract EXACTLY top_n matches per external item
    results = []
    for ext_idx, ext_row in external_df.iterrows():
        similarities = similarity_matrix[ext_idx]
        
        # Get exactly top_n matches (or all available if less than top_n)
        actual_top_n = min(top_n, len(internal_df))
        top_indices = np.argsort(similarities)[-actual_top_n:][::-1]
        
        # Add all top_n matches, regardless of threshold
        for rank, int_idx in enumerate(top_indices, 1):
            int_row = internal_df.iloc[int_idx]
            score = float(similarities[int_idx])
            
            # Only filter by threshold if score is above it, but always include at least top_n
            # This ensures we get exactly top_n matches per external item
            results.append({
                'external_name': ext_row['external_name'],
                'best_internal_match': int_row['internal_name'],
                'similarity_score': score,
                'internal_department': int_row['department']
            })
    
    # If threshold is set, filter AFTER getting top_n matches
    # But keep at least 1 match per external item
    if threshold > 0:
        filtered_results = []
        for ext_name in external_df['external_name'].unique():
            ext_matches = [r for r in results if r['external_name'] == ext_name]
            # Keep matches above threshold, or at least the best match
            above_threshold = [r for r in ext_matches if r['similarity_score'] >= threshold]
            if above_threshold:
                filtered_results.extend(above_threshold)
            elif ext_matches:  # Keep at least the best match
                filtered_results.append(ext_matches[0])
        results = filtered_results
    
    results_df = pd.DataFrame(results)
    avg_similarity = results_df['similarity_score'].mean() if len(results_df) > 0 else 0.0
    
    return results_df, similarity_matrix, avg_similarity

# Function to load and encode logo
def get_logo_base64():
    """Try multiple paths to find and encode the logo"""
    possible_paths = [
        'humber_logo.png',
        './humber_logo.png',
        'images/humber_logo.png',
        '/mount/src/research-alignment-matcher/humber_logo.png'
    ]
    
    for path in possible_paths:
        try:
            if Path(path).exists():
                with open(path, "rb") as img_file:
                    return base64.b64encode(img_file.read()).decode()
        except:
            continue
    return None

# Try to load Humber logo
logo_base64 = get_logo_base64()

# Header with branding
if logo_base64:
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem 0;'>
        <img src='data:image/png;base64,{logo_base64}' style='height: 60px; margin-bottom: 1.5rem;' alt='Humber Polytechnic'>
        <h1 style='color: #1f77b4; font-size: 3rem; margin-bottom: 0.5rem;'>ScholarSync</h1>
        <h3 style='color: #666; font-weight: 400; margin-bottom: 0.5rem;'>Academic Collaboration Platform</h3>
        <p style='color: #888; font-size: 1.1rem; font-style: italic;'>Bridging Academic Minds Through Intelligent Matching</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Fallback without logo
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #1f77b4; font-size: 3rem; margin-bottom: 0.5rem;'>ScholarSync</h1>
        <h3 style='color: #666; font-weight: 400; margin-bottom: 0.5rem;'>Academic Collaboration Platform</h3>
        <p style='color: #888; font-size: 1.1rem; font-style: italic;'>Bridging Academic Minds Through Intelligent Matching</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Simplified professional About section
st.markdown("""
<div style='background-color: #f0f8ff; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #1f77b4; margin-bottom: 2rem;'>
    <p style='margin: 0; color: #333; font-size: 1.05rem;'>
    <strong>ScholarSync</strong> matches researchers and faculty members based on their expertise and research interests using AI-powered semantic analysis.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    method = st.selectbox(
        "Matching Method",
        ["embeddings", "llm"],
        format_func=lambda x: "Sentence Transformers (AI-Powered)" if x == "embeddings" else "LLM (Under Construction)",
        help="Sentence Transformers uses semantic embeddings for intelligent matching"
    )
    
    top_n = st.number_input(
        "Top N Matches", 
        min_value=1, 
        max_value=100, 
        value=3, 
        step=1,
        help="Number of matches to show per external item"
    )
    
    threshold = st.slider(
        "Similarity Threshold (%)", 
        0, 
        100, 
        0, 
        step=5, 
        help="Minimum similarity score to include (0-30% recommended)"
    ) / 100
    
    st.markdown("---")
    
    # Simplified About AI Technology section - removed clustering
    st.markdown("### 📊 About AI Technology")
    st.info(
        "✅ **Yes, this is AI-Powered!**\n\n"
        "ScholarSync uses **Sentence Transformers**, a deep learning model that:\n\n"
        "• Creates semantic embeddings\n\n"
        "• Understands context and meaning\n\n"
        "• Captures concept relationships\n\n"
        "• Delivers intelligent matching"
    )

# Main content - Only 2 tabs
tab1, tab2 = st.tabs(["📤 Upload & Match", "📊 Results"])

with tab1:
    st.header("Upload CSV Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Internal Dataset")
        
        internal_file = st.file_uploader(
            "Upload Internal Dataset",
            type=['csv'],
            key='internal',
            help="CSV file with columns: internal_name, department, expertise_summary"
        )
        
        if internal_file is not None:
            try:
                internal_df = pd.read_csv(internal_file)
                st.success(f"✅ Loaded {len(internal_df)} records")
                with st.expander("Preview Data"):
                    st.dataframe(internal_df.head())
            except Exception as e:
                st.error(f"Error loading file: {e}")
                internal_df = None
        else:
            internal_df = None
    
    with col2:
        st.subheader("External Dataset")
        
        external_file = st.file_uploader(
            "Upload External Dataset",
            type=['csv'],
            key='external',
            help="CSV file with columns: external_name, affiliation, research_interest_summary"
        )
        
        if external_file is not None:
            try:
                external_df = pd.read_csv(external_file)
                st.success(f"✅ Loaded {len(external_df)} records")
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
                    results_df, similarity_matrix, avg_similarity = match_datasets(
                        internal_df, external_df, method, top_n, threshold
                    )
                    
                    # Store in session state
                    st.session_state['results_df'] = results_df
                    st.session_state['similarity_matrix'] = similarity_matrix
                    st.session_state['avg_similarity'] = avg_similarity
                    st.session_state['internal_df'] = internal_df
                    st.session_state['external_df'] = external_df
                    
                    st.success(f"✅ Matching complete! Found {len(results_df)} matches with average similarity of {avg_similarity:.1%}")
                    # REMOVED: st.balloons()
                    
            except Exception as e:
                st.error(f"❌ Error during matching: {e}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("⚠️ Please upload both CSV files")

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
            st.metric("External Items", unique_external)
        
        st.markdown("---")
        
        # Results table with exact format requested
        st.subheader("Match Results Table")
        
        # Format the display dataframe
        display_df = results_df[['external_name', 'best_internal_match', 'similarity_score', 'internal_department']].copy()
        display_df['similarity_score'] = display_df['similarity_score'].apply(lambda x: f"{x:.3f}")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "external_name": st.column_config.TextColumn("external_name", width="medium"),
                "best_internal_match": st.column_config.TextColumn("best_internal_match", width="medium"),
                "similarity_score": st.column_config.TextColumn("similarity_score", width="small"),
                "internal_department": st.column_config.TextColumn("internal_department", width="medium")
            }
        )
        
        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="scholarsync_matches.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    else:
        st.info("👆 Run the matching algorithm in the 'Upload & Match' tab to see results here")

# Footer - Simple and clean
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 2rem 0;'>"
    "© 2025 ScholarSync - Academic Collaboration Platform"
    "</div>",
    unsafe_allow_html=True
)
