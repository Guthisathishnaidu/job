# 🎯 RoleMatch — Job Title Recommendation System

Upload a resume (PDF/DOCX/TXT) or paste skills text → get ranked job title recommendations.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your dataset CSV(s) to /dataset/
#    (works with kandij/job-recommendation-datasets — all 4 files)

# 3. Run
python app.py

# 4. Open browser
http://localhost:5000
```

## How It Works

```
Resume (PDF/DOCX/TXT/text)
        ↓
   Text Extraction
        ↓
   Clean + Normalize   (lowercase, remove stopwords, punctuation)
        ↓
   TF-IDF Transform    (using saved vectorizer)
        ↓
   Cosine Similarity   (against all corpus documents)
        ↓
   Aggregate by Job Title  (avg score per title)
        ↓
   Top-N Ranked Results
```

## API Endpoints

| Route | Method | Body | Description |
|---|---|---|---|
| `/` | GET | — | Main UI |
| `/api/recommend` | POST | `file` (FormData) or `{text}` (JSON) | Get job title recommendations |
| `/api/stats` | GET | — | Dataset overview for stats bar |

## Model Files (auto-generated on first run)

```
model/
  tfidf_vectorizer.pkl   — fitted TF-IDF vectorizer
  tfidf_matrix.pkl       — transformed corpus matrix  
  corpus.pkl             — cleaned dataset (text + titles)
```

Delete PKL files to force retraining (e.g. after updating dataset).

## Dataset

Place any job CSV files in `/dataset/`. Columns are auto-detected by keyword:
- Text column: any col containing `skill / description / resume / experience / summary`
- Title column: any col containing `title / role / position / job / category`