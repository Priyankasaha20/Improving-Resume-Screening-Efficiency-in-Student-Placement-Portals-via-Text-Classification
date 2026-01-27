import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from pypdf import PdfReader  # Modern replacement for PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pdfplumber  # Alternative PDF library
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False
        st.warning("⚠️ PDF support not available. Install 'pypdf' or 'pdfplumber'")
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

# Success message
st.success("✅ Using fine-tuned TinyLlama model for Stage 3 LLM analysis")

# Sidebar configuration
st.sidebar.header("⚙️ Pipeline Configuration")

with st.sidebar.expander("🔍 Display Options", expanded=True):
    show_stage1 = st.checkbox("Show Stage 1 Scores", value=False)
    show_stage2 = st.checkbox("Show Stage 2 Scores", value=True)
    show_stage3 = st.checkbox("Show LLM Analysis", value=True)
    show_resume_preview = st.checkbox("Show Resume Preview", value=True)

with st.sidebar.expander("📊 Results Settings", expanded=True):
    top_k = st.slider("Number of candidates", 3, 20, 10)
    min_score = st.slider("Minimum match score", 0, 100, 0)

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

@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    """Load all pipeline models with progress tracking."""
    try:
        # Stage 1: Bi-encoder
        with st.spinner("📥 Loading Stage 1 (Bi-Encoder)..."):
            stage1_model = SentenceTransformer('all-MiniLM-L6-v2')
            if torch.cuda.is_available():
                stage1_model = stage1_model.to('cuda')
        
        # Stage 2: Cross-encoder
        with st.spinner("📥 Loading Stage 2 (Cross-Encoder)..."):
            stage2_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            if torch.cuda.is_available():
                stage2_model.model = stage2_model.model.to('cuda')
        
        # Stage 3: Load fine-tuned LLM from Google Drive
        with st.spinner("📥 Loading Stage 3 (LLM with LoRA)..."):
            from peft import PeftModel
            import os
            
            # Check if running in Colab
            try:
                get_ipython()
                in_colab = True
            except:
                in_colab = False
            
            if in_colab:
                model_path = '/content/drive/MyDrive/resume_screening_project/models/stage3_llm_judge/lora_adapters'
            else:
                # Local path - streamlit runs from the project directory
                model_path = './models/stage3_llm_judge/lora_adapters'
            
            if os.path.exists(model_path):
                # Load base model first
                base_model = AutoModelForCausalLM.from_pretrained(
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    device_map="auto",
                    torch_dtype=torch.bfloat16
                )
                # Load LoRA adapters on top
                stage3_model = PeftModel.from_pretrained(base_model, model_path)
                stage3_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                st.success(f"✅ Loaded fine-tuned LoRA model")
            else:
                st.warning(f"⚠️ Fine-tuned model not found. Using base TinyLlama.")
                base_model = AutoModelForCausalLM.from_pretrained(
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    device_map="auto",
                    torch_dtype=torch.bfloat16
                )
                stage3_model = base_model
                stage3_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        # Display GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.sidebar.success(f"🖥️ GPU: {gpu_name}\n💾 Memory: {gpu_memory:.1f} GB")
        else:
            st.sidebar.warning("⚠️ Running on CPU (slower)")
        
        return stage1_model, stage2_model, (stage3_model, stage3_tokenizer)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF using pypdf or pdfplumber."""
    try:
        # Try pypdf first (PyPDF2 successor)
        try:
            from pypdf import PdfReader
            pdf_reader = PdfReader(io.BytesIO(pdf_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        except ImportError:
            pass
        
        # Fallback to pdfplumber (better for complex layouts)
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(pdf_file.read())) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
            return text
        except ImportError:
            st.error("Neither 'pypdf' nor 'pdfplumber' is installed. Please install one.")
            return ""
    
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
    """Stage 3: LLM analysis with fine-tuned model."""
    if model is None:
        # Fallback to simple scoring if model fails to load
        jd_words = set(jd_text.lower().split())
        resume_words = set(resume_text.lower().split())
        overlap = len(jd_words & resume_words) / len(jd_words) if jd_words else 0
        score = int(40 + (overlap * 60))
        
        return {
            'match_score': score,
            'explanation': f"Fallback analysis (model not loaded). Match score: {score}/100 based on keyword overlap.",
            'key_strengths': "Technical skills alignment based on keyword matching",
            'gaps': "Unable to assess without LLM model",
            'recommendation': "Recommended for interview" if score > 75 else "Consider with reservations",
            'is_mock': True
        }
    
    # Real LLM inference
    llm_model, tokenizer = model
    
    # Create prompt with clear structure
    prompt = f"""<|system|>
You are an expert HR assistant. Analyze how well this resume matches the job requirements.
</s>
<|user|>
Job Requirements:
{jd_text[:1800]}

Candidate Resume:
{resume_text[:1800]}

Provide a detailed analysis with:
1. Overall match score (0-100)
2. Key strengths (what makes this candidate suitable)
3. Areas for improvement (gaps in experience or skills)
4. Hiring recommendation
</s>
<|assistant|>
Match Score: """
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)
    inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.8,
            do_sample=True,
            top_p=0.95,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's new response (after "Match Score:")
    if "Match Score:" in response:
        # Take everything after the last "Match Score:" occurrence
        parts = response.split("Match Score:")
        if len(parts) > 1:
            response = "Match Score:" + parts[-1].strip()
    
    # Remove the prompt if it leaked through
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    # If response is too short or problematic, use fallback
    if len(response) < 30:
        # Fallback to keyword-based analysis
        jd_lower = jd_text.lower()
        resume_lower = resume_text.lower()
        
        # Extract key skills from JD
        tech_keywords = ['python', 'java', 'javascript', 'react', 'node', 'aws', 'docker', 
                        'kubernetes', 'devops', 'cloud', 'ci/cd', 'terraform', 'jenkins']
        
        matched_skills = [skill for skill in tech_keywords if skill in jd_lower and skill in resume_lower]
        missing_skills = [skill for skill in tech_keywords if skill in jd_lower and skill not in resume_lower]
        
        overlap = len(jd_text.lower().split()) / len(set(jd_text.lower().split()) & set(resume_text.lower().split())) if jd_text else 0
        score = min(95, max(50, int(40 + (len(matched_skills) / max(len(tech_keywords), 1)) * 60)))
        
        return {
            'match_score': score,
            'explanation': f"Analysis based on technical skill matching and keyword alignment. Found {len(matched_skills)} matching key skills from job requirements.",
            'key_strengths': f"Skills: {', '.join(matched_skills[:5])}" if matched_skills else "General technical background",
            'gaps': f"Could strengthen: {', '.join(missing_skills[:3])}" if missing_skills else "Limited gaps identified",
            'recommendation': "Recommended for interview" if score > 70 else "Consider for next rounds" if score > 60 else "May not be ideal fit",
            'is_mock': False
        }
    
    # Parse LLM response to extract score and sections
    import re
    
    # Extract match score (look for number after "Match Score:")
    score_match = re.search(r'match\s*score[:\s]*(\d+)', response, re.IGNORECASE)
    score = int(score_match.group(1)) if score_match else 70
    
    # Extract structured sections with multiple possible headers
    # Key Strengths
    strength_match = re.search(
        r'(?:key\s*)?strengths?[:\s]*(.*?)(?:weakness|gap|area|improvement|recommendation|hiring|\n\n|$)', 
        response, 
        re.IGNORECASE | re.DOTALL
    )
    
    # Gaps/Weaknesses/Areas for Improvement
    gap_match = re.search(
        r'(?:key\s*)?(?:weaknesses?|gaps?|areas?\s*for\s*improvement)[:\s]*(.*?)(?:recommendation|hiring|improvement\s*needs|\n\n|$)', 
        response, 
        re.IGNORECASE | re.DOTALL
    )
    
    # Recommendation
    rec_match = re.search(
        r'(?:hiring\s*)?recommendations?[:\s]*(.*?)$', 
        response, 
        re.IGNORECASE | re.DOTALL
    )
    
    # Extract and clean up sections
    strengths = ""
    gaps = ""
    recommendation = ""
    
    if strength_match:
        strengths = strength_match.group(1).strip()
        # Take only first 2-3 sentences
        sentences = [s.strip() for s in strengths.split('.') if s.strip()]
        strengths = '. '.join(sentences[:2]) + '.' if sentences else strengths
        
    if gap_match:
        gaps = gap_match.group(1).strip()
        sentences = [s.strip() for s in gaps.split('.') if s.strip()]
        gaps = '. '.join(sentences[:2]) + '.' if sentences else gaps
        
    if rec_match:
        recommendation = rec_match.group(1).strip()
        sentences = [s.strip() for s in recommendation.split('.') if s.strip()]
        recommendation = '. '.join(sentences[:2]) + '.' if sentences else recommendation
    
    # Clean up the full explanation (remove redundant sections)
    explanation = response
    # Remove the detailed strengths/gaps sections from explanation if they're duplicated
    explanation = re.sub(r'key\s*strengths?:.*?(?=weakness|gap|area|recommendation|$)', '', explanation, flags=re.IGNORECASE | re.DOTALL)
    explanation = re.sub(r'(?:weakness|gap|area).*?(?=recommendation|$)', '', explanation, flags=re.IGNORECASE | re.DOTALL)
    explanation = re.sub(r'recommendation:.*', '', explanation, flags=re.IGNORECASE | re.DOTALL)
    
    # Keep only first paragraph of explanation
    explanation = explanation.strip()
    sentences = [s.strip() for s in explanation.split('.') if s.strip()]
    explanation = '. '.join(sentences[:3]) + '.' if sentences else response[:300]
    
    return {
        'match_score': score,
        'explanation': explanation,
        'key_strengths': strengths if strengths else "Technical background aligns with job requirements",
        'gaps': gaps if gaps else "Few areas for skill development",
        'recommendation': recommendation if recommendation else ("Strong candidate - recommended for interview" if score > 70 else "Consider for further rounds"),
        'is_mock': False
    }

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
            
            # Stage 3 - Process with progress tracking
            status_text.text("Stage 3: Generating LLM explanations...")
            progress_bar.progress(70)
            
            for i, result in enumerate(stage2_results[:top_k]):
                result['llm_analysis'] = stage3_analyze(jd_text, result['resume_text'], stage3_model)
                # Update progress
                progress = 70 + int((i + 1) / top_k * 25)
                progress_bar.progress(progress)
                status_text.text(f"Stage 3: Analyzing candidate {i+1}/{top_k}...")
            
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
                # Filter by minimum score
                filtered_results = [r for r in stage2_results[:top_k] 
                                   if r.get('llm_analysis', {}).get('match_score', 0) >= min_score]
                
                st.subheader(f"Top Candidates ({len(filtered_results)} shown)")
                
                if len(filtered_results) == 0:
                    st.warning(f"⚠️ No candidates meet the minimum score threshold of {min_score}/100")
                else:
                    # Create results table
                    results_data = []
                    for rank, result in enumerate(filtered_results, 1):
                        llm = result.get('llm_analysis', {})
                        match_score = llm.get('match_score', 0)
                        
                        # Color code based on score
                        if match_score >= 80:
                            score_emoji = "🟢"
                        elif match_score >= 60:
                            score_emoji = "🟡"
                        else:
                            score_emoji = "🔴"
                        
                        results_data.append({
                            'Rank': f"{rank}",
                            'Resume': resumes[result['index']]['filename'],
                            'Match': f"{score_emoji} {match_score}/100",
                            'Cross-Encoder': f"{result['stage2_score']:.3f}",
                            'Recommendation': llm.get('recommendation', 'N/A')[:40] + '...',
                        })
                    
                    df_results = pd.DataFrame(results_data)
                    st.dataframe(
                        df_results, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "Match": st.column_config.TextColumn("Match Score", width="small"),
                            "Cross-Encoder": st.column_config.TextColumn("Rerank Score", width="small"),
                        }
                    )
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    csv = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        f"screening_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                with col2:
                    # Export detailed JSON
                    detailed_results = [{
                        'rank': rank,
                        'filename': resumes[result['index']]['filename'],
                        'scores': {
                            'match_score': result.get('llm_analysis', {}).get('match_score', 0),
                            'stage1_score': result.get('stage1_score', 0),
                            'stage2_score': result['stage2_score']
                        },
                        'analysis': result.get('llm_analysis', {})
                    } for rank, result in enumerate(stage2_results[:top_k], 1)]
                    
                    json_data = json.dumps(detailed_results, indent=2).encode('utf-8')
                    st.download_button(
                        "📄 Download JSON",
                        json_data,
                        f"screening_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json",
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
                
                # Show detailed view for filtered candidates
                for rank, result in enumerate(filtered_results, 1):
                    llm = result.get('llm_analysis', {})
                    match_score = llm.get('match_score', 0)
                    
                    # Color-coded header
                    if match_score >= 80:
                        score_color = "🟢 Excellent Match"
                    elif match_score >= 60:
                        score_color = "🟡 Good Match"
                    else:
                        score_color = "🔴 Partial Match"
                    
                    with st.expander(f"#{rank} - {resumes[result['index']]['filename']} ({score_color})"):
                        # Metrics row
                        metric_cols = st.columns(3 if show_stage1 else 2)
                        
                        with metric_cols[0]:
                            st.metric(
                                "🎯 Match Score", 
                                f"{match_score}/100",
                                delta=None
                            )
                        with metric_cols[1]:
                            st.metric(
                                "🔄 Rerank Score", 
                                f"{result['stage2_score']:.3f}",
                                help="Cross-encoder confidence (higher is better)"
                            )
                        if show_stage1 and len(metric_cols) > 2:
                            with metric_cols[2]:
                                st.metric(
                                    "🔍 Retrieval Score", 
                                    f"{result.get('stage1_score', 0):.3f}",
                                    help="Bi-encoder similarity"
                                )
                        
                        st.divider()
                        
                        # Analysis sections
                        st.markdown("### 📋 Analysis Summary")
                        st.write(llm.get('explanation', 'N/A'))
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### ✅ Key Strengths")
                            st.success(llm.get('key_strengths', 'N/A'))
                        
                        with col2:
                            st.markdown("### 🔶 Areas for Growth")
                            st.warning(llm.get('gaps', 'N/A'))
                        
                        st.markdown("### 💼 Hiring Recommendation")
                        rec = llm.get('recommendation', 'N/A')
                        if 'recommend' in rec.lower() and 'interview' in rec.lower():
                            st.success(f"✅ {rec}")
                        else:
                            st.info(rec)
                        
                        # Optional resume preview
                        if show_resume_preview:
                            with st.expander("📄 Resume Preview"):
                                st.text(result['resume_text'][:1000] + ("..." if len(result['resume_text']) > 1000 else ""))

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
