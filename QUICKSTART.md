# 🚀 Quick Start Guide - Resume Screening System

## What You Just Got

A complete, research-quality AI resume screening system with **7 files**:

### 📓 Jupyter Notebooks (6 files)

1. **00_setup_and_data_preprocessing.ipynb** - Data loading & cleaning
2. **01_stage1_retriever_biencoder.ipynb** - Fast vector search
3. **02_stage2_reranker_crossencoder.ipynb** - Precise ranking
4. **03_stage3_llm_judge_finetuning.ipynb** - Explainable AI (GPU needed!)
5. **04_full_pipeline_integration.ipynb** - Complete pipeline
6. **05_evaluation_and_metrics.ipynb** - Performance metrics

### 🌐 Web App

7. **06_streamlit_demo_app.py** - Interactive demo UI

### 📄 Documentation

- **README.md** - Complete project documentation
- **requirements.txt** - All Python dependencies

---

## ⚡ 3-Minute Setup

### Option A: Google Colab (Easiest)

```
1. Upload all .ipynb files to your Google Drive
2. Open 00_setup_and_data_preprocessing.ipynb in Colab
3. Run all cells (Runtime → Run all)
4. Continue with notebooks 01 → 05 in order
```

**For notebook 03 (LLM)**: Enable GPU

- Runtime → Change runtime type → T4 GPU → Save

### Option B: Kaggle

```
1. Create new Kaggle notebook
2. Upload notebooks or copy-paste code
3. Enable GPU in Settings (for notebook 03)
4. Run sequentially
```

### Option C: Local (Your Computer)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Jupyter
jupyter notebook

# 3. Open notebooks and run in order: 00 → 05

# 4. For the web app:
streamlit run 06_streamlit_demo_app.py
```

---

## 🎯 What Each Notebook Does

| Notebook | What It Does                       | Time    | GPU?     |
| -------- | ---------------------------------- | ------- | -------- |
| **00**   | Loads data, removes personal info  | 15 min  | No       |
| **01**   | Creates searchable resume database | 10 min  | Optional |
| **02**   | Improves ranking accuracy          | 10 min  | Optional |
| **03**   | Adds AI explanations               | 2-4 hrs | **YES**  |
| **04**   | Combines everything                | 5 min   | No       |
| **05**   | Measures performance               | 10 min  | No       |

**Total time**: ~3-4 hours (mostly waiting for notebook 03)

---

## ⚠️ Important Notes

### GPU Requirement

- **Notebooks 00, 01, 02, 04, 05**: Run fine on CPU
- **Notebook 03**: NEEDS GPU (T4 minimum, 15GB VRAM)

### Data Storage

The notebooks save data to:

- **Colab**: `/content/drive/MyDrive/resume_screening_project/`
- **Kaggle**: `/kaggle/working/resume_screening_project/`
- **Local**: `./resume_screening_project/`

### Expected Outputs

After running all notebooks, you'll have:

```
resume_screening_project/
├── data/
│   ├── processed/
│   │   ├── resume_scores_anonymized.parquet
│   │   └── jd_resume_match_anonymized.parquet
│   └── raw/
├── models/
│   ├── stage1_retriever/
│   │   ├── faiss_index.bin (FAISS search index)
│   │   └── resume_embeddings.npy
│   ├── stage2_reranker/
│   │   └── reranking_cache.pkl
│   └── stage3_llm_judge/
│       └── lora_adapters/ (Fine-tuned model)
└── outputs/
    ├── embeddings_umap.html
    ├── stage2_score_analysis.png
    └── evaluation_report.pdf
```

---

## 🎓 Learning Path

### Beginner

1. Run notebooks 00-02 only
2. Skip notebook 03 (no GPU needed)
3. Run notebook 04 with mock LLM
4. Focus on understanding the pipeline

### Intermediate

1. Run all notebooks sequentially
2. Modify hyperparameters
3. Try different models
4. Deploy Streamlit app

### Advanced

1. Fine-tune on your own data
2. Implement custom evaluation metrics
3. Add fairness analysis
4. Optimize for production

---

## 🐛 Common Issues

### "No module named 'sentence_transformers'"

```bash
pip install sentence-transformers
```

### "CUDA out of memory" (Notebook 03)

```python
# In notebook 03, reduce batch size:
per_device_train_batch_size=2  # Change from 4 to 2
```

### "GPU not detected" (Notebook 03)

- **Colab**: Runtime → Change runtime type → GPU
- **Kaggle**: Settings → Accelerator → GPU T4 x2

### Notebooks run slow

- Use GPU for notebooks 01, 02, 03
- Reduce dataset size in notebook 00
- Lower `max_examples` in notebook 03

---

## 💡 Quick Tips

### Speed Up Development

1. **Use sample data**: Notebooks generate sample data if real data unavailable
2. **Skip LLM training**: Use pre-trained model for testing
3. **Cache results**: Notebooks save checkpoints automatically

### Best Practices

1. **Run sequentially**: Each notebook depends on previous outputs
2. **Check GPU**: Verify GPU availability before notebook 03
3. **Save often**: Google Colab can timeout (save to Drive)
4. **Monitor memory**: Keep an eye on RAM/VRAM usage

### Customization Points

- **Notebook 01**: Change `MODEL_NAME` for different embeddings
- **Notebook 02**: Adjust `top_k` for more/fewer candidates
- **Notebook 03**: Switch between TinyLlama (fast) and Mistral-7B (accurate)
- **Notebook 04**: Modify scoring weights for stages

---

## 📊 Testing the System

### Test with Sample Data

All notebooks include sample data fallbacks. Just run them!

### Test with Your Own Data

```python
# In notebook 00, replace dataset loading with:
df_resumes = pd.read_csv('your_resumes.csv')
df_jd_match = pd.read_csv('your_jd_resume_pairs.csv')
```

### Test the Streamlit App

```bash
streamlit run 06_streamlit_demo_app.py
```

Then:

1. Paste a job description
2. Upload resumes (PDF or TXT)
3. Click "Run AI Screening"
4. Download results as CSV

---

## 🎯 Next Steps

After completing the setup:

1. **Read the README.md** for detailed documentation
2. **Explore visualizations** in the `outputs/` folder
3. **Try the Streamlit app** for interactive testing
4. **Modify and experiment** with different configurations
5. **Share your results** or contribute improvements

---

## 🆘 Need Help?

### Resources

- **Documentation**: See README.md
- **Code comments**: Every notebook has detailed explanations
- **Error messages**: Read carefully - they usually point to the issue

### Debugging Checklist

- [ ] Installed all requirements?
- [ ] Running notebooks in order (00 → 05)?
- [ ] GPU enabled for notebook 03?
- [ ] Enough disk space (need ~5GB)?
- [ ] Enough RAM (need ~8GB)?

---

## ✅ Success Criteria

You'll know it's working when:

1. **Notebook 00**: Creates `resume_scores_anonymized.parquet`
2. **Notebook 01**: Builds FAISS index successfully
3. **Notebook 02**: Shows re-ranking improvements
4. **Notebook 03**: Saves LoRA adapters (~50MB)
5. **Notebook 04**: Generates comparison report
6. **Notebook 05**: Creates evaluation metrics
7. **Streamlit**: Shows ranked resumes with explanations

---

## 🎉 You're Ready!

Start with notebook 00 and work through sequentially. Each notebook is self-contained with clear instructions.

**Estimated total time**: 3-4 hours  
**Effort level**: Intermediate  
**GPU requirement**: Only for notebook 03

Good luck! 🚀
