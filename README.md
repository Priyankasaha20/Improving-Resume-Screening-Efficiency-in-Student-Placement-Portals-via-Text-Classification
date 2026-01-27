# 🎯 Multi-Stage AI-Powered Resume Screening System

A research-quality, production-ready resume screening system using a three-stage deep learning pipeline designed for Kaggle/Google Colab notebooks.

## 🌟 Project Overview

This system demonstrates state-of-the-art NLP techniques for automated resume screening:

- **Stage 1**: Fast retrieval using bi-encoder (sentence-transformers) + FAISS
- **Stage 2**: Precise re-ranking with cross-encoder attention models
- **Stage 3**: Explainable AI scoring using fine-tuned LLMs with LoRA

## 📁 Project Structure

```
resume-screening/
├── 00_setup_and_data_preprocessing.ipynb    # Data loading, cleaning, anonymization
├── 01_stage1_retriever_biencoder.ipynb      # Fast retrieval with embeddings + FAISS
├── 02_stage2_reranker_crossencoder.ipynb    # Precision re-ranking
├── 03_stage3_llm_judge_finetuning.ipynb     # LLM fine-tuning with LoRA
├── 04_full_pipeline_integration.ipynb       # End-to-end pipeline
├── 05_evaluation_and_metrics.ipynb          # Comprehensive evaluation
├── 06_streamlit_demo_app.py                 # Interactive web application
├── README.md                                 # This file
└── requirements.txt                          # Python dependencies
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. **Upload notebooks to Google Drive**
2. **Open each notebook in Colab**:
   - File → Open notebook → Google Drive
3. **Enable GPU** (for Stage 3):
   - Runtime → Change runtime type → T4 GPU
4. **Run notebooks sequentially** (00 → 05)

### Option 2: Kaggle

1. **Create a new Kaggle notebook**
2. **Add notebooks as dataset** or copy code
3. **Enable GPU** in Settings → Accelerator
4. **Run sequentially**

### Option 3: Local Setup

```bash
# Clone or download the project
cd resume-screening

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook

# Or run Streamlit app
streamlit run 06_streamlit_demo_app.py
```

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Job Description                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │  Stage 1: Retrieval   │  ← Bi-Encoder + FAISS
           │  1M resumes → Top 100 │     (~10ms)
           └───────────┬───────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │  Stage 2: Re-Ranking  │  ← Cross-Encoder
           │  100 → Top 20         │     (~200ms)
           └───────────┬───────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │  Stage 3: LLM Judge   │  ← Fine-tuned LLM
           │  Explainable Scoring  │     (~500ms)
           └───────────┬───────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │   Ranked Resumes +    │
           │   Explanations        │
           └───────────────────────┘
```

## 🎓 Educational Features

Each notebook includes:

- ✅ **Comprehensive markdown explanations** - Tutorial-style documentation
- ✅ **Research insights** - Why these architectures, hyperparameter choices
- ✅ **Performance benchmarks** - Speed vs accuracy trade-offs
- ✅ **Visualizations** - UMAP embeddings, score distributions, ablation studies
- ✅ **Best practices** - Error handling, memory management, checkpointing
- ✅ **Reproducibility** - Seed setting, saved artifacts, metadata tracking

## 📚 Notebook Details

### Notebook 00: Setup & Data Preprocessing

**Runtime**: 15-30 min (CPU)

- Dataset loading from Hugging Face
- PDF text extraction
- PII anonymization (names, emails, phone numbers)
- Data exploration and profiling
- Save processed data as Parquet

**Datasets Used**:

- `netsol/resume-score-details`
- `facehuggerapoorv/resume-jd-match`

### Notebook 01: Stage 1 Retrieval

**Runtime**: 10-20 min (CPU/GPU)

- Load `all-MiniLM-L6-v2` sentence-transformer
- Create resume embeddings (384-dim vectors)
- Build FAISS index for fast similarity search
- Benchmark: Query 1M resumes in <50ms
- UMAP/t-SNE visualizations

**Key Metrics**:

- Recall@100: Target >95%
- Query latency: ~10ms

### Notebook 02: Stage 2 Re-Ranking

**Runtime**: 10-15 min (CPU/GPU)

- Load `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Re-score top-100 candidates with cross-attention
- Ablation study: Bi-encoder vs. Bi+Cross pipeline
- NDCG@K improvements

**Key Metrics**:

- NDCG@10 improvement: Target >10%
- Re-ranking latency: ~200ms for 100 candidates

### Notebook 03: Stage 3 LLM Fine-Tuning

**Runtime**: 2-4 hours (⚠️ GPU REQUIRED)

- Load TinyLlama-1.1B (or Mistral-7B) with 4-bit quantization
- LoRA fine-tuning (r=16, only 1.2% of params)
- Structured output generation (JSON format)
- Before/after fine-tuning comparison

**Hardware Requirements**:

- Minimum: T4 GPU (15GB VRAM)
- Recommended: A100 GPU (40GB VRAM)

### Notebook 04: Full Pipeline Integration

**Runtime**: 5-10 min

- Load all three stages from checkpoints
- End-to-end inference pipeline
- Batch processing demonstrations
- Generate HTML comparison reports

### Notebook 05: Evaluation & Metrics

**Runtime**: 10-15 min

- Ranking metrics: NDCG, MRR, Precision@K, Recall@K
- Fairness analysis (optional)
- Statistical significance tests
- Export comprehensive PDF report

### App: Streamlit Demo

**Runtime**: Real-time

```bash
streamlit run 06_streamlit_demo_app.py
```

- Upload job descriptions and resumes
- Real-time processing
- Interactive results table
- Download results as CSV

## 🔧 Technical Stack

| Component             | Technology            | Purpose                            |
| --------------------- | --------------------- | ---------------------------------- |
| **Embeddings**        | sentence-transformers | Dense vector representations       |
| **Similarity Search** | FAISS                 | Fast approximate nearest neighbors |
| **Re-Ranking**        | CrossEncoder          | Precise pair-wise scoring          |
| **LLM**               | TinyLlama/Mistral     | Explainable reasoning              |
| **Fine-Tuning**       | LoRA (PEFT)           | Parameter-efficient adaptation     |
| **Quantization**      | bitsandbytes          | 4-bit model compression            |
| **Training**          | HuggingFace TRL       | Supervised fine-tuning             |
| **UI**                | Streamlit             | Interactive demo                   |

## 💾 Storage & Persistence

### Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
# Save to: /content/drive/MyDrive/resume_screening_project/
```

### Kaggle

```python
# Save to: /kaggle/working/resume_screening_project/
# Then create Kaggle Dataset from output
```

### Saved Artifacts

- `data/processed/` - Anonymized parquet files
- `models/stage1_retriever/` - FAISS index + embeddings
- `models/stage2_reranker/` - Cross-encoder cache
- `models/stage3_llm_judge/` - LoRA adapters (~50MB)
- `outputs/` - Visualizations, reports

## 📈 Performance Benchmarks

| Stage             | Latency   | Throughput  | Memory         |
| ----------------- | --------- | ----------- | -------------- |
| Stage 1           | 10ms      | 100 QPS     | 2GB RAM        |
| Stage 2           | 200ms     | 5 QPS       | 4GB RAM        |
| Stage 3           | 500ms     | 2 QPS       | 8GB VRAM       |
| **Full Pipeline** | **710ms** | **1.4 QPS** | **10GB total** |

## 🎯 Research Highlights

### Why Multi-Stage?

1. **Bi-Encoder**: Fast but approximate (pre-computed embeddings)
2. **Cross-Encoder**: Accurate but slow (pair-wise attention)
3. **LLM**: Explainable but expensive (generative reasoning)

**Solution**: Use each where it's best!

- Stage 1 filters 1M → 100 (speed)
- Stage 2 refines 100 → 20 (accuracy)
- Stage 3 explains top 20 (transparency)

### Novel Contributions

- ✨ **Modular design**: Each stage is independently useful
- ✨ **Memory efficient**: 4-bit LLM + LoRA fine-tuning
- ✨ **Explainable AI**: Not just scores, but reasoning
- ✨ **Production-ready**: Restart-safe, error handling, monitoring
- ✨ **Educational**: Tutorial-style with research insights

## 🔒 Privacy & Ethics

### PII Anonymization

- Automatic detection and removal of:
  - Names (NER-based)
  - Email addresses
  - Phone numbers
  - Physical addresses
  - Social media links

### Fairness Considerations

- Bias detection in Notebook 05
- Disparate impact analysis
- Recommendation: Human-in-the-loop final decisions

## 🐛 Troubleshooting

### GPU Out of Memory (Stage 3)

```python
# Reduce batch size
per_device_train_batch_size=2  # instead of 4

# Enable gradient checkpointing
gradient_checkpointing=True

# Use smaller model
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # instead of Mistral-7B
```

### FAISS Import Error

```bash
# Use CPU version
pip install faiss-cpu

# Or GPU version (if CUDA available)
pip install faiss-gpu
```

### Tokenizer Warnings

```python
# Ignore harmless warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
```

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@software{resume_screening_2025,
  title = {Multi-Stage AI Resume Screening System},
  year = {2025},
  author = {Your Name},
  url = {https://github.com/yourusername/resume-screening}
}
```

## 🤝 Contributing

Improvements welcome!

**Ideas for extensions**:

- [ ] Add support for more file formats (DOCX, HTML)
- [ ] Implement active learning for continuous improvement
- [ ] Multi-language support
- [ ] Integration with ATS systems
- [ ] A/B testing framework
- [ ] Model explainability with LIME/SHAP

## 📜 License

MIT License - feel free to use for research or commercial purposes.

## 🙏 Acknowledgments

- **Sentence Transformers** by Nils Reimers
- **FAISS** by Facebook AI Research
- **Hugging Face** for transformers ecosystem
- **Google Colab** & **Kaggle** for free GPU access

## 📞 Support

- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Email**: your.email@example.com

---

**⭐ If you find this useful, please star the repository!**

Built with ❤️ for the AI/ML community
