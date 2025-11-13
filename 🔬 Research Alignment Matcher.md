# 🔬 Research Alignment Matcher

AI-Powered Research Collaboration Matching using Advanced NLP

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

## 🌟 Features

- **Advanced NLP Matching**: Uses Sentence Transformers for semantic similarity
- **Interactive Visualizations**: Heatmaps, distributions, and bar charts
- **Flexible Configuration**: Choose matching method, top N matches, and similarity threshold
- **CSV Upload**: Upload your own researcher data or use sample data
- **Downloadable Results**: Export matching results as CSV
- **Real-time Processing**: Get results in seconds

## 🚀 Quick Start

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Deploy from your forked repository

## 📊 How It Works

1. **Upload CSV Files** with researcher information
2. **Configure Parameters** (matching method, top N, threshold)
3. **Run Algorithm** to find research alignments
4. **View Results** in interactive tables and visualizations
5. **Download** matching results as CSV

## 📁 CSV Format

### Internal Researchers
```csv
internal_name,department,expertise_summary
Dr. Sarah Thompson,Chemistry,Sustainable catalysis and green chemistry
```

### External Researchers
```csv
external_name,affiliation,research_interest_summary
Dr. Emily Chen,GreenTech Research Institute,Hydrogen production and sustainable chemical reactions
```

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **NLP Engine**: Sentence Transformers (all-MiniLM-L6-v2)
- **Similarity Metric**: Cosine Similarity
- **Visualizations**: Plotly
- **Data Processing**: Pandas, NumPy, scikit-learn

## 📈 Matching Methods

### Sentence Transformers (Recommended)
- 384-dimensional semantic embeddings
- Captures contextual meaning
- Better accuracy for research domain matching

### TF-IDF (Traditional)
- 500-dimensional sparse vectors
- Faster processing
- Good for keyword-based matching

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**© 2025 Research Alignment Matcher | Powered by Sentence Transformers & Advanced NLP**
