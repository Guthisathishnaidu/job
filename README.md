# 🎯 RoleMatch AI — Job Title Recommendation System

> An AI-powered web application that reads your resume and recommends the most relevant job titles using **TF-IDF Vectorization** and **Cosine Similarity**.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-PythonAnywhere-brightgreen)](https://guthisathish.pythonanywhere.com)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey)](https://flask.palletsprojects.com/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)](https://scikit-learn.org/)

---

## 🌐 Live Demo

**[https://guthisathish.pythonanywhere.com](https://guthisathish.pythonanywhere.com)**

Upload your resume and get instant job title recommendations in under 1 second.

---

## 📌 What is This Project?

Most job seekers don't know which job titles best match their skills. They spend hours browsing job portals and applying for wrong roles. **RoleMatch AI** solves this by:

1. Reading your uploaded resume (PDF, DOCX, or TXT)
2. Analysing your skills and experience using NLP
3. Matching your profile against **55,068 real job records**
4. Returning the **Top 8 most relevant job titles** ranked by similarity score

---

## ✨ Features

- 📄 **Resume Upload** — Supports PDF, DOCX, and TXT formats
- 📂 **Multi-File Upload** — Upload multiple resumes at once, get separate results per file
- ✍️ **Paste Text** — Paste your skills or resume text directly into the text box
- ⚡ **Real-Time Results** — Recommendations returned in under 100ms
- 📊 **Similarity Score** — Each job title shows a percentage match score
- 🏆 **Ranked Results** — Top 8 titles ranked with Gold/Silver/Bronze medals
- 🔄 **Auto Model Rebuild** — Model auto-rebuilds if dataset changes, no manual steps needed
- 🌐 **Live Deployment** — Hosted on PythonAnywhere, accessible via public URL
- ⌨️ **Keyboard Shortcut** — Press `Ctrl + Enter` to submit instantly

---

## 🧠 How It Works

```
Your Resume (PDF/DOCX/TXT)
        ↓
Text Extraction (pypdf / python-docx)
        ↓
Text Cleaning (lowercase, remove stopwords, punctuation, URLs)
        ↓
TF-IDF Vectorization (1 × 20,000 dimensional vector)
        ↓
Cosine Similarity vs 55,068 Job Profiles
        ↓
Aggregate Scores by Job Title (average + best score)
        ↓
Top 8 Job Titles Ranked by Similarity Score
```

### Algorithm Details

| Step | Technique | Purpose |
|------|-----------|---------|
| Feature Extraction | TF-IDF (Term Frequency–Inverse Document Frequency) | Convert text to numerical vector |
| Similarity Matching | Cosine Similarity | Measure closeness between resume and job profiles |
| Model Storage | Pickle (.pkl) | Save trained model for real-time inference |
| Text Extraction | pypdf + python-docx | Read PDF and DOCX resume files |

---

## 📁 Project Structure

```
job_Recommendation_Platform/
│
├── app.py                        # Flask backend — routes, ML engine, startup logic
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── templates/
│   └── index.html                # Frontend HTML dashboard
│
├── static/
│   ├── style.css                 # Styling (light theme, Playfair Display + Outfit fonts)
│   └── script.js                 # Frontend interactions, drag-drop, results rendering
│
├── model/
│   ├── preprocess.py             # Text cleaning utilities
│   ├── tfidf_vectorizer.pkl      # Saved TF-IDF vectorizer (auto-generated)
│   ├── tfidf_matrix.pkl          # Saved corpus matrix (auto-generated)
│   ├── corpus.pkl                # Cleaned corpus with job titles (auto-generated)
│   └── model_version.txt         # Version check file (auto-generated)
│
└── dataset/
    ├── Experience.csv            # Applicant past job positions (8,653 rows)
    ├── Job_Views.csv             # Jobs viewed with titles (12,370 rows)
    ├── Positions_Of_Interest.csv # Desired job roles per applicant (6,560 rows)
    └── job_data.csv              # Pre-processed job descriptions (5,000 rows)
```

---

## 🗃️ Dataset

**Source:** [kandij/job-recommendation-datasets](https://www.kaggle.com/datasets/kandij/job-recommendation-datasets) — Kaggle

| File | Rows | Description |
|------|------|-------------|
| `Experience.csv` | 8,653 | Applicant past job positions and descriptions |
| `Job_Views.csv` | 12,370 | Jobs viewed by applicants with position titles |
| `Positions_Of_Interest.csv` | 6,560 | Desired job roles per applicant |
| `job_data.csv` | 5,000 | Pre-processed job description text |

**After merging and cleaning:** 55,068 records · 4,768 unique job titles

> ⚠️ Dataset files are not committed to GitHub due to size limits. Download from Kaggle and place in `/dataset/` folder.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or above
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/Guthisathishnaidu/job_Recommendation_Platform.git
cd job_Recommendation_Platform
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Dataset

Download the 4 CSV files from [Kaggle](https://www.kaggle.com/datasets/kandij/job-recommendation-datasets) and place them inside the `dataset/` folder:

```
dataset/
├── Experience.csv
├── Job_Views.csv
├── Positions_Of_Interest.csv
└── job_data.csv
```

### 5. Run the Application

```bash
python app.py
```

### 6. Open in Browser

```
http://127.0.0.1:5000
```

> 💡 On first run, the app automatically builds the TF-IDF model and saves PKL files to `/model/`. This takes about 30–60 seconds. Subsequent starts load instantly from the saved PKLs.

---

## 📦 Dependencies

```
flask>=3.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
pypdf>=3.0.0
python-docx>=1.0.0
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🔌 API Endpoints

| Endpoint | Method | Input | Description |
|----------|--------|-------|-------------|
| `/` | GET | — | Serves the main dashboard |
| `/api/recommend` | POST | `file` (FormData) or `{"text": "..."}` (JSON) | Returns top 8 job recommendations |
| `/api/stats` | GET | — | Returns top 10 job title distribution |
| `/api/debug` | POST | File or text | Diagnostic — shows extracted text and raw scores |

### Example API Usage

**Text input (JSON):**
```bash
curl -X POST http://127.0.0.1:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"text": "Python machine learning TensorFlow data science scikit-learn"}'
```

**File upload:**
```bash
curl -X POST http://127.0.0.1:5000/api/recommend \
  -F "resumes=@Resume.pdf"
```

**Response:**
```json
{
  "best_title": "Machine Learning Engineer",
  "results": [
    { "title": "Machine Learning Engineer", "score": 0.512, "best_score": 0.612, "matches": 42 },
    { "title": "Data Scientist",            "score": 0.384, "best_score": 0.501, "matches": 31 },
    { "title": "Python Developer",          "score": 0.271, "best_score": 0.389, "matches": 18 }
  ],
  "total_matched": 8
}
```

---

## 🖥️ Screenshots

### Homepage
Clean upload interface with drag-drop zone, text paste area, and animated submit button.

### Results Panel
Best match shown with large job title, similarity percentage, and gradient score bar.
Full ranked list with Gold 🥇 Silver 🥈 Bronze 🥉 medals and animated fill bars.

### Loading State
Four-step animated progress indicator:
`Extracting text → TF-IDF vectorization → Cosine similarity → Ranking results`

---

## ⚙️ Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | TF-IDF + Cosine Similarity |
| max_features | 20,000 |
| ngram_range | (1, 2) — unigrams + bigrams |
| sublinear_tf | True (log scaling) |
| min_df | 1 (keeps rare tech terms like TensorFlow, React) |
| max_df | 0.95 (removes near-universal terms) |
| Similarity threshold | ≥ 0.001 |
| Response time | < 100ms |
| Matrix size | 55,068 × 20,000 (sparse, ~0.08% density) |

---

## 🔄 Auto Model Versioning

The app includes a smart model versioning system:

- A `MODEL_VERSION` constant is stored in code (currently `v3`)
- On every startup, the saved version is compared against the current version
- If they don't match → model is **automatically rebuilt** from dataset
- If corpus contains numeric job titles (stale data) → model is **automatically rebuilt**
- No manual PKL deletion ever needed

---

## ☁️ Deployment (PythonAnywhere)

### Steps

1. Log in to [PythonAnywhere](https://www.pythonanywhere.com)
2. Open a Bash console and clone the repo:
   ```bash
   git clone https://github.com/Guthisathishnaidu/job_Recommendation_Platform.git
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt --user
   ```
4. Upload dataset CSVs to `/home/GuthiSathish/job_Recommendation_Platform/dataset/` via the Files tab
5. Set up WSGI file (Web tab → WSGI configuration file):
   ```python
   import sys, os
   project_home = '/home/GuthiSathish/job_Recommendation_Platform'
   if project_home not in sys.path:
       sys.path.insert(0, project_home)
   os.chdir(project_home)
   from app import app as application
   ```
6. Click **Reload** on the Web tab
7. Visit `https://guthisathish.pythonanywhere.com`

### Update After Code Changes

```bash
# On PythonAnywhere Bash console
cd ~/job_Recommendation_Platform
git pull
```
Then click **Reload** on the Web tab.

---

## 👥 Team

| Name | Roll Number |
|------|-------------|
| G. Sathish | 24R05A0515 |
| K. Akhil | 24R05A0518 |
| P. Zaid | 24R05A0536 |

---

## 📄 License

This project is developed for academic purposes as part of the Industry Oriented Mini Project at CMR Institute of Technology, Hyderabad.

---

## 🙏 Acknowledgements

- [Kaggle](https://www.kaggle.com/datasets/kandij/job-recommendation-datasets) — Dataset
- [Scikit-learn](https://scikit-learn.org) — TF-IDF and Cosine Similarity
- [PythonAnywhere](https://www.pythonanywhere.com) — Free hosting platform
- [Flask](https://flask.palletsprojects.com) — Web framework
