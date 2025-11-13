# 🎓 AcademiaMatch

**Academic Collaboration Platform**

*Bridging Academic Minds Through Intelligent Matching*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg )](https://academiamatch.streamlit.app )

---

## 📖 About

AcademiaMatch is an AI-powered platform that matches researchers and faculty members based on their expertise and research interests using advanced semantic analysis. Find your next collaborator, co-author, or research partner with precision matching.

## ✨ Features

- 🤖 **AI-Powered Matching** - Uses Sentence Transformers for semantic similarity
- 📊 **Flexible CSV Upload** - Works with any researcher dataset
- 🎯 **Top N Matching** - Get exactly N matches per external researcher
- 🔍 **Similarity Threshold** - Filter matches by quality score
- 📥 **Export Results** - Download matches as CSV
- ⚡ **Real-time Processing** - Get results in seconds

## 🚀 Quick Start

### Run Locally

```bash
# Clone the repository
git clone https://github.com/Sakr2957/AcademiaMatch-App.git
cd AcademiaMatch-App

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

### Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Create new app and select your forked repository
4. Deploy!

## 📁 CSV Format

### Internal Dataset (Your Institution)

```csv
internal_name,department,expertise_summary
Dr. Sarah Thompson,Chemistry,Sustainable catalysis and green chemistry
Dr. Michael Lee,Computer Science,Machine learning and AI ethics
```

### External Dataset (External Researchers)

```csv
external_name,affiliation,research_interest_summary
Dr. Emily Chen,GreenTech Institute,Hydrogen production and sustainable reactions
Dr. Omar Yusuf,AI for Humanity Lab,Fair machine learning systems
```

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Engine**: Sentence Transformers (all-MiniLM-L6-v2)
- **Similarity**: Cosine Similarity
- **Data Processing**: Pandas, NumPy, scikit-learn

## 🎯 Use Cases

- **Research Collaboration** - Match researchers for joint projects
- **Faculty Recruitment** - Match candidates to positions
- **Student-Advisor Matching** - Pair students with suitable advisors
- **Grant Partnerships** - Find collaborators for funding opportunities
- **Conference Networking** - Connect attendees with similar interests

## 📊 How It Works

1. Upload two CSV files (internal and external datasets)
2. Configure matching parameters (Top N, Threshold)
3. Run the AI matching algorithm
4. View results in formatted table
5. Download matches as CSV

## 🤖 AI Technology

AcademiaMatch uses **Sentence Transformers**, a state-of-the-art deep learning model that:

- Creates 384-dimensional semantic embeddings
- Understands context and meaning beyond keywords
- Captures relationships between research concepts
- Achieves high accuracy in semantic matching

## 📈 Output Format

Results include:

- `external_name` - External researcher name
- `best_internal_match` - Matched internal researcher
- `similarity_score` - Match quality (0.0 - 1.0)
- `internal_department` - Department of matched researcher

## 📄 License

© 2025 Humber Polytechnic. All rights reserved.

This project was developed at Humber Polytechnic for academic research collaboration purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**© 2025 AcademiaMatch | Powered by Humber Polytechnic Research & Innovation
