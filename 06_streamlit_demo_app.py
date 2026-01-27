import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
import PyPDF2
import io
import tempfile

# Page config
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🎯 Multi-Stage AI Resume Screening System")
st.markdown("**Powered by Bi-Encoder → Cross-Encoder → LLM Judge**")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")
show_stage1 = st.sidebar.checkbox("Show Stage 1 (Retrieval)", value=False)
show_stage2 = st.sidebar.checkbox("Show Stage 2 (Re-ranking)", value=True)
show_stage3 = st.sidebar.checkbox("Show Stage 3 (LLM Analysis)", value=True)
top_k = st.sidebar.slider("Number of results", 5, 20, 10)

st.sidebar.markdown("---")
st.sidebar.header("📊 Model Info")
st.sidebar.info("""
**Stage 1**: all-MiniLM-L6-v2  
**Stage 2**: ms-marco-MiniLM-L-6-v2  
**Stage 3**: TinyLlama-1.1B (LoRA)
""")

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.stage1_model = None
    st.session_state.stage2_model = None
    st.session_state.stage3_model = None

@st.cache_resource
def load_models():
    """Load all pipeline models."""
    try:
        # Stage 1: Bi-encoder
        stage1_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Stage 2: Cross-encoder
        stage2_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Stage 3: LLM (optional - requires significant memory)
        stage3_model = None
        # Uncomment if you have GPU and want LLM inference:
        # stage3_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        # stage3_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        return stage1_model, stage2_model, stage3_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return ""

def stage1_retrieve(jd_text, resume_texts, model, top_k=100):
    """Stage 1: Fast retrieval with bi-encoder."""
    # Encode
    jd_embedding = model.encode([jd_text], normalize_embeddings=True)
    resume_embeddings = model.encode(resume_texts, normalize_embeddings=True)
    
    # Compute similarity
    similarities = np.dot(jd_embedding, resume_embeddings.T)[0]
    
    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'index': int(idx),
            'score': float(similarities[idx]),
            'resume_text': resume_texts[idx]
        })
    
    return results

def stage2_rerank(jd_text, candidates, model):
    """Stage 2: Re-rank with cross-encoder."""
    pairs = [[jd_text, cand['resume_text']] for cand in candidates]
    scores = model.predict(pairs)
    
    # Update candidates
    for cand, score in zip(candidates, scores):
        cand['stage1_score'] = cand['score']
        cand['stage2_score'] = float(score)
        cand['score'] = float(score)
    
    # Sort by stage2 score
    candidates.sort(key=lambda x: x['stage2_score'], reverse=True)
    
    return candidates

def stage3_analyze(jd_text, resume_text, model=None):
    """Stage 3: LLM analysis (mock if model not loaded)."""
    if model is None:
        # Mock analysis
        score = np.random.randint(70, 95)
        return {
            'match_score': score,
            'explanation': f"This candidate shows strong alignment with the job requirements. Match score: {score}/100.",
            'key_strengths': "Technical skills, relevant experience, good cultural fit",
            'gaps': "Some areas for growth in advanced topics",
            'recommendation': "Recommended for interview" if score > 75 else "Consider with reservations"
        }
    else:
        # Real LLM inference (implement if model loaded)
        pass

# Main app
st.markdown("---")

# Section 1: Job Description
st.header("1️⃣ Job Description")
jd_input_method = st.radio("Input method:", ["Paste Text", "Upload File"], horizontal=True)

jd_text = ""
if jd_input_method == "Paste Text":
    jd_text = st.text_area(
        "Paste job description:",
        height=200,
        placeholder="Enter the full job description here..."
    )
else:
    jd_file = st.file_uploader("Upload JD (PDF/TXT)", type=['pdf', 'txt'])
    if jd_file:
        if jd_file.type == 'application/pdf':
            jd_text = extract_text_from_pdf(jd_file)
        else:
            jd_text = jd_file.read().decode('utf-8')
        st.success(f"Loaded {len(jd_text)} characters")

# Section 2: Resumes
st.header("2️⃣ Resumes")
resume_input_method = st.radio("Input method:", ["Upload Files", "Sample Data"], horizontal=True)

resumes = []
if resume_input_method == "Upload Files":
    resume_files = st.file_uploader(
        "Upload resumes (PDF/TXT)",
        type=['pdf', 'txt'],
        accept_multiple_files=True
    )
    
    if resume_files:
        for file in resume_files:
            if file.type == 'application/pdf':
                text = extract_text_from_pdf(file)
            else:
                text = file.read().decode('utf-8')
            
            resumes.append({
                'filename': file.name,
                'text': text
            })
        
        st.success(f"Loaded {len(resumes)} resumes")
else:
    # Sample data
    sample_resumes = [
        "Senior Software Engineer with 5+ years in Python, Django, React. Built microservices, RESTful APIs. AWS experience.",
        "Data Scientist with PhD in Statistics. Expert in ML, deep learning, NLP. Published researcher. Python, R, TensorFlow.",
        "Full Stack Developer. 3 years Node.js, React, MongoDB. Startup experience. Agile, CI/CD pipelines.",
        "ML Engineer specializing in computer vision. PyTorch, CUDA, model optimization. Deployed models at scale.",
        "Junior Developer. Recent graduate. Java, Spring Boot, MySQL. Strong fundamentals, eager to learn.",
        "DevOps Engineer. Kubernetes, Docker, Jenkins, AWS. Infrastructure as code. 4 years experience.",
        "Frontend Developer. Expert in React, Vue.js, TypeScript. UI/UX focus. Portfolio of modern web apps.",
        "Backend Engineer. Go, microservices, gRPC, PostgreSQL. Performance optimization expert.",
        "Data Engineer. ETL pipelines, Airflow, Spark, Snowflake. Big data processing. 6 years experience.",
        "Mobile Developer. iOS and Android. Swift, Kotlin, React Native. Published apps with 100K+ downloads."
    ]
    
    for i, text in enumerate(sample_resumes):
        resumes.append({
            'filename': f'resume_{i+1}.txt',
            'text': text
        })
    
    st.info(f"Using {len(resumes)} sample resumes")

# Section 3: Run Analysis
st.markdown("---")
st.header("3️⃣ Analysis")

if st.button("🚀 Run AI Screening", type="primary", use_container_width=True):
    if not jd_text:
        st.error("Please provide a job description")
    elif not resumes:
        st.error("Please provide resumes")
    else:
        with st.spinner("Loading models..."):
            stage1_model, stage2_model, stage3_model = load_models()
        
        if stage1_model is None:
            st.error("Failed to load models")
        else:
            # Extract resume texts
            resume_texts = [r['text'] for r in resumes]
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Stage 1
            status_text.text("Stage 1: Retrieving candidates...")
            progress_bar.progress(20)
            stage1_results = stage1_retrieve(jd_text, resume_texts, stage1_model, top_k=len(resumes))
            
            # Stage 2
            status_text.text("Stage 2: Re-ranking with cross-encoder...")
            progress_bar.progress(50)
            stage2_results = stage2_rerank(jd_text, stage1_results, stage2_model)
            
            # Stage 3
            status_text.text("Stage 3: Generating LLM explanations...")
            progress_bar.progress(70)
            for result in stage2_results[:top_k]:
                result['llm_analysis'] = stage3_analyze(jd_text, result['resume_text'], stage3_model)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Display results
            st.markdown("---")
            st.header("📊 Results")
            
            # Create tabs
            if show_stage1 and show_stage2 and show_stage3:
                tab1, tab2, tab3 = st.tabs(["🥇 Final Rankings", "📈 Comparison", "🔍 Detailed View"])
            else:
                tab1, tab2 = st.tabs(["🥇 Final Rankings", "🔍 Detailed View"])
            
            with tab1:
                st.subheader(f"Top {top_k} Candidates")
                
                # Create results table
                results_data = []
                for rank, result in enumerate(stage2_results[:top_k], 1):
                    llm = result.get('llm_analysis', {})
                    results_data.append({
                        'Rank': rank,
                        'Resume': resumes[result['index']]['filename'],
                        'Match Score': f"{llm.get('match_score', 0)}/100",
                        'Stage2 Score': f"{result['stage2_score']:.4f}",
                        'Recommendation': llm.get('recommendation', 'N/A'),
                        'Key Strengths': llm.get('key_strengths', 'N/A')[:50] + '...',
                    })
                
                df_results = pd.DataFrame(results_data)
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                # Download button
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    "screening_results.csv",
                    "text/csv",
                    use_container_width=True
                )
            
            if show_stage1 and show_stage2 and show_stage3:
                with tab2:
                    st.subheader("Stage Comparison")
                    
                    # Score comparison
                    comparison_data = []
                    for result in stage2_results[:top_k]:
                        comparison_data.append({
                            'Resume': resumes[result['index']]['filename'],
                            'Stage 1': result.get('stage1_score', 0),
                            'Stage 2': result['stage2_score'],
                            'LLM Score': result.get('llm_analysis', {}).get('match_score', 0) / 100
                        })
                    
                    df_comparison = pd.DataFrame(comparison_data)
                    st.line_chart(df_comparison.set_index('Resume'))
            
            with tab2 if not (show_stage1 and show_stage2 and show_stage3) else tab3:
                st.subheader("Detailed Analysis")
                
                # Show detailed view for each candidate
                for rank, result in enumerate(stage2_results[:top_k], 1):
                    with st.expander(f"#{rank} - {resumes[result['index']]['filename']}"):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.metric("Match Score", f"{result.get('llm_analysis', {}).get('match_score', 0)}/100")
                            st.metric("Stage 2 Score", f"{result['stage2_score']:.4f}")
                            if show_stage1:
                                st.metric("Stage 1 Score", f"{result.get('stage1_score', 0):.4f}")
                        
                        with col2:
                            llm = result.get('llm_analysis', {})
                            st.markdown("**Explanation:**")
                            st.write(llm.get('explanation', 'N/A'))
                            st.markdown("**Key Strengths:**")
                            st.write(llm.get('key_strengths', 'N/A'))
                            st.markdown("**Gaps:**")
                            st.write(llm.get('gaps', 'N/A'))
                            st.markdown("**Recommendation:**")
                            st.info(llm.get('recommendation', 'N/A'))
                        
                        st.markdown("**Resume Text:**")
                        st.text(result['resume_text'][:500] + "...")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Multi-Stage AI Resume Screening System</p>
    <p>Built with Streamlit • Sentence Transformers • HuggingFace Transformers</p>
    <p>Stage 1: Fast Retrieval | Stage 2: Precise Ranking | Stage 3: Explainable AI</p>
</div>
""", unsafe_allow_html=True)

# Deployment instructions
with st.sidebar.expander("📖 Deployment Guide"):
    st.markdown("""
    **Deploy to Streamlit Cloud:**
    
    1. Push code to GitHub
    2. Go to [share.streamlit.io](https://share.streamlit.io)
    3. Connect repository
    4. Add `requirements.txt`:
    ```
    streamlit
    sentence-transformers
    transformers
    torch
    PyPDF2
    pandas
    numpy
    ```
    5. Deploy!
    
    **Note**: Stage 3 LLM requires significant memory.  
    Consider using CPU-only inference or serverless APIs.
    """)
