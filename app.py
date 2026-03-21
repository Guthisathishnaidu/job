"""
app.py  —  Job Title Recommendation System
Flask + TF-IDF + Cosine Similarity

Dataset: kandij/job-recommendation-datasets  (4 CSV files)
  Experience.csv              → Applicant.ID, Position.Name, Job.Description ...
  Job_Views.csv               → Applicant.ID, Job.ID, Position ...
  Positions_Of_Interest.csv   → Applicant.ID, Position.Of.Interest
  job_data.csv                → Job.ID, text (pre-processed job descriptions)

Corpus built from 4 segments — NO column auto-detection (hardcoded, correct):
  1. job_data.csv JOIN Job_Views.csv on Job.ID  → text + job title
  2. Experience.csv (grouped) JOIN POI → experience text + desired title
  3. Positions_Of_Interest direct  → title keywords
  4. Experience Position.Name direct → past roles as labels

Supports: single OR multiple resume file uploads (PDF / DOCX / TXT)
"""

from flask import Flask, render_template, request, jsonify
import os, re, pickle, glob, io
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

BASE_DIR  = os.path.dirname(__file__)
DATA_DIR  = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
VEC_PATH  = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
MAT_PATH  = os.path.join(MODEL_DIR, 'tfidf_matrix.pkl')
CORP_PATH = os.path.join(MODEL_DIR, 'corpus.pkl')
os.makedirs(MODEL_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════
#  FILE TEXT EXTRACTION
# ═══════════════════════════════════════════════════════
def extract_pdf(file_bytes):
    try:
        import pypdf
        r = pypdf.PdfReader(io.BytesIO(file_bytes))
        t = ' '.join(p.extract_text() or '' for p in r.pages)
        if t.strip(): return t
    except Exception: pass
    try:
        import PyPDF2
        r = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        t = ' '.join(p.extract_text() or '' for p in r.pages)
        if t.strip(): return t
    except Exception: pass
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            t = ' '.join(p.extract_text() or '' for p in pdf.pages)
        if t.strip(): return t
    except Exception: pass
    return ''

def extract_docx(file_bytes):
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        return ' '.join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception: return ''

def extract_file(filename, file_bytes):
    name = (filename or '').lower()
    if   name.endswith('.pdf'):  return extract_pdf(file_bytes)
    elif name.endswith('.docx'): return extract_docx(file_bytes)
    elif name.endswith('.txt'):  return file_bytes.decode('utf-8', errors='ignore')
    return ''


# ═══════════════════════════════════════════════════════
#  TEXT CLEANING
# ═══════════════════════════════════════════════════════
STOPWORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with',
    'by','from','is','was','are','were','be','been','have','has','had',
    'do','does','did','will','would','could','should','may','might','i',
    'you','he','she','it','we','they','this','that','my','our','your',
    'also','not','no','as','if','into','so','just','very','more','most',
    'about','up','out','then','than','its','their','there','here',
    'per','etc','via','inc','llc','ltd','co','corp',
}

def clean(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\+?\d[\d\s\-\(\)]{6,}', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(w for w in text.split() if w not in STOPWORDS and len(w) > 1)


# ═══════════════════════════════════════════════════════
#  CORPUS BUILDER  (hardcoded for this dataset)
# ═══════════════════════════════════════════════════════
def _read(filename, **kwargs):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  ⚠  Not found: {filename}")
        return None
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines='skip', **kwargs)
        except Exception:
            continue
    print(f"  ⚠  Could not read: {filename}")
    return None

def build_corpus():
    segments = []

    # ── 1. job_data  JOIN  Job_Views  →  description text + job title ──
    job_data  = _read('job_data.csv')
    job_views = _read('Job_Views.csv')
    if job_data is not None and job_views is not None:
        try:
            jd = job_data[['Job.ID', 'text']].dropna(subset=['text'])
            jv = job_views[['Job.ID', 'Position']].dropna()
            jv = jv.drop_duplicates('Job.ID')
            mg = jd.merge(jv, on='Job.ID', how='inner')
            mg = mg[mg['Position'].str.strip() != '']
            seg = pd.DataFrame({
                'resume_text': mg['text'].astype(str),
                'job_title'  : mg['Position'].astype(str).str.strip().str.title(),
            })
            segments.append(seg)
            print(f"  ✅ Seg 1 (job_data + Job_Views): {len(seg)} rows")
        except Exception as e:
            print(f"  ⚠  Seg 1 failed: {e}")

    # ── 2. Experience (grouped)  JOIN  Positions_Of_Interest ───────────
    experience = _read('Experience.csv')
    poi        = _read('Positions_Of_Interest.csv')
    if experience is not None and poi is not None:
        try:
            # Combine Position.Name + Job.Description per applicant into one text blob
            def combine_applicant(grp):
                parts = grp['Position.Name'].dropna().astype(str).tolist()
                if 'Job.Description' in grp.columns:
                    parts += grp['Job.Description'].dropna().astype(str).tolist()
                return ' '.join(parts)

            exp_grp = experience.groupby('Applicant.ID').apply(
                combine_applicant
            ).reset_index(name='experience_text')

            poi_grp = poi.groupby('Applicant.ID')['Position.Of.Interest'] \
                         .first().reset_index()

            merged = exp_grp.merge(poi_grp, on='Applicant.ID', how='inner')
            merged = merged[merged['Position.Of.Interest'].str.strip() != '']

            seg = pd.DataFrame({
                'resume_text': merged['experience_text'].astype(str),
                'job_title'  : merged['Position.Of.Interest'].astype(str)
                                 .str.strip().str.title(),
            })
            segments.append(seg)
            print(f"  ✅ Seg 2 (Experience + POI): {len(seg)} rows")
        except Exception as e:
            print(f"  ⚠  Seg 2 failed: {e}")

    # ── 3. Positions_Of_Interest  →  direct keyword index ──────────────
    if poi is not None:
        try:
            p = poi['Position.Of.Interest'].dropna()
            p = p[p.str.strip() != '']
            seg = pd.DataFrame({
                'resume_text': p.astype(str),
                'job_title'  : p.astype(str).str.strip().str.title(),
            })
            segments.append(seg)
            print(f"  ✅ Seg 3 (POI direct): {len(seg)} rows")
        except Exception as e:
            print(f"  ⚠  Seg 3 failed: {e}")

    # ── 4. Experience Position.Name  →  direct role index ──────────────
    if experience is not None:
        try:
            p = experience['Position.Name'].dropna()
            p = p[p.str.strip() != '']
            p = p[~p.str.match(r'^\s*\d+\s*$')]   # drop any numeric IDs
            seg = pd.DataFrame({
                'resume_text': p.astype(str),
                'job_title'  : p.astype(str).str.strip().str.title(),
            })
            segments.append(seg)
            print(f"  ✅ Seg 4 (Experience positions): {len(seg)} rows")
        except Exception as e:
            print(f"  ⚠  Seg 4 failed: {e}")

    if not segments:
        raise ValueError("No corpus segments built — check dataset/ folder.")

    corpus = pd.concat(segments, ignore_index=True)

    # Hard filter: remove ANY row where job_title is purely numeric
    corpus = corpus[~corpus['job_title'].str.match(r'^\s*[\d\s]+$')]
    corpus = corpus.dropna().drop_duplicates(
        subset=['resume_text', 'job_title']
    ).reset_index(drop=True)
    corpus['cleaned'] = corpus['resume_text'].apply(clean)
    corpus = corpus[corpus['cleaned'].str.split().str.len() >= 2].reset_index(drop=True)

    print(f"\n  📊 Corpus ready: {len(corpus)} rows | "
          f"{corpus['job_title'].nunique()} unique titles")
    print("  Top titles:", corpus['job_title'].value_counts().head(8).to_dict())
    return corpus


# ═══════════════════════════════════════════════════════
#  TF-IDF MODEL
# ═══════════════════════════════════════════════════════
def build_model(corpus):
    print("\n  Fitting TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features = 20000,
        ngram_range  = (1, 2),
        stop_words   = 'english',
        sublinear_tf = True,
        min_df       = 1,
        max_df       = 0.95,
    )
    matrix = tfidf.fit_transform(corpus['cleaned'])
    with open(VEC_PATH,  'wb') as f: pickle.dump(tfidf,  f)
    with open(MAT_PATH,  'wb') as f: pickle.dump(matrix, f)
    with open(CORP_PATH, 'wb') as f:
        pickle.dump(corpus[['cleaned', 'job_title']].reset_index(drop=True), f)
    print(f"  💾 Saved | vocab={len(tfidf.vocabulary_):,} | matrix={matrix.shape}")
    return tfidf, matrix, corpus

def load_model():
    with open(VEC_PATH,  'rb') as f: tfidf  = pickle.load(f)
    with open(MAT_PATH,  'rb') as f: matrix = pickle.load(f)
    with open(CORP_PATH, 'rb') as f: corpus = pickle.load(f)
    print(f"  ✅ Loaded | vocab={len(tfidf.vocabulary_):,} | matrix={matrix.shape}")
    return tfidf, matrix, corpus


# ── App startup ──────────────────────────────────────────
# MODEL_VERSION: bump this number any time you change corpus logic.
# A mismatch forces an automatic rebuild — no manual PKL deletion needed.
MODEL_VERSION = 3

VERSION_PATH = os.path.join(MODEL_DIR, 'model_version.txt')

def _pkls_valid():
    """Return True only if PKLs exist AND were built with the current MODEL_VERSION."""
    if not all(os.path.exists(p) for p in [VEC_PATH, MAT_PATH, CORP_PATH, VERSION_PATH]):
        return False
    try:
        with open(VERSION_PATH) as f:
            saved = int(f.read().strip())
        return saved == MODEL_VERSION
    except Exception:
        return False

def _save_version():
    with open(VERSION_PATH, 'w') as f:
        f.write(str(MODEL_VERSION))

def _delete_pkls():
    for p in [VEC_PATH, MAT_PATH, CORP_PATH, VERSION_PATH]:
        try: os.remove(p)
        except FileNotFoundError: pass

print("\n🚀 Starting Job Recommendation Engine...")
corpus_df = build_corpus()

if _pkls_valid():
    try:
        TFIDF, MATRIX, CORPUS = load_model()
        # Final sanity check: titles in loaded corpus must NOT be purely numeric
        numeric_pct = CORPUS['job_title'].str.match(r'^\s*\d+\s*$').mean()
        if numeric_pct > 0.05 or len(TFIDF.vocabulary_) < 100:
            raise ValueError(
                f"Stale corpus detected — {numeric_pct:.0%} numeric titles. Rebuilding..."
            )
        print(f"  ✅ PKLs valid (version={MODEL_VERSION})")
    except Exception as e:
        print(f"  ⚠  {e}")
        _delete_pkls()
        TFIDF, MATRIX, CORPUS = build_model(corpus_df)
        _save_version()
else:
    print(f"  ℹ  PKLs missing or version mismatch → rebuilding (v{MODEL_VERSION})...")
    _delete_pkls()
    TFIDF, MATRIX, CORPUS = build_model(corpus_df)
    _save_version()

TOTAL_JOBS   = len(CORPUS)
TOTAL_TITLES = CORPUS['job_title'].nunique()
print(f"\n  ✅ Ready — {TOTAL_JOBS} records | {TOTAL_TITLES} job titles\n")


# ═══════════════════════════════════════════════════════
#  RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════
def recommend(resume_text, top_n=8):
    cleaned = clean(resume_text)
    if len(cleaned.split()) < 2:
        return []

    vec    = TFIDF.transform([cleaned])
    scores = cosine_similarity(vec, MATRIX).flatten()

    title_scores = defaultdict(list)
    for idx, score in enumerate(scores):
        if score >= 0.001:
            title_scores[CORPUS.iloc[idx]['job_title']].append(float(score))

    if not title_scores:   # fallback: any non-zero
        for idx, score in enumerate(scores):
            if score > 0:
                title_scores[CORPUS.iloc[idx]['job_title']].append(float(score))

    if not title_scores:
        return []

    return sorted([
        {
            'title'      : t,
            'score'      : round(float(np.mean(sc)), 4),
            'best_score' : round(float(max(sc)), 4),
            'matches'    : len(sc),
        }
        for t, sc in title_scores.items()
    ], key=lambda x: x['score'], reverse=True)[:top_n]


# ═══════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════
@app.route('/')
def index():
    return render_template('index.html',
        total_jobs   = f"{TOTAL_JOBS:,}",
        total_titles = TOTAL_TITLES,
        pdf_ok=True, docx_ok=True)


@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """
    Accepts:
      A) Multiple files  FormData field: 'resumes'  (multiple=true)
      B) Single file     FormData field: 'resume'
      C) Plain text      JSON { "text": "..." }

    Returns:
      A/B) { results_per_file: [ { filename, results, best_title, ... }, ... ] }
      C)   { results, best_title, resume_preview, total_matched }
    """
    # ── Multi-file or single-file upload ───────────────
    files = request.files.getlist('resumes') or \
            ([request.files['resume']] if request.files.get('resume') else [])

    if files:
        all_results = []
        for f in files:
            fname = f.filename or 'unknown'
            data  = f.read()
            text  = extract_file(fname, data)

            if not text.strip():
                all_results.append({
                    'filename': fname,
                    'error'   : f'Could not extract text from "{fname}". '
                                f'Install pypdf: pip install pypdf'
                })
                continue

            results = recommend(text, top_n=8)
            preview = ' '.join(text.split()[:40]) + \
                      ('...' if len(text.split()) > 40 else '')

            if not results:
                all_results.append({
                    'filename': fname,
                    'error'   : 'No matching roles found. Try a more detailed resume.'
                })
            else:
                all_results.append({
                    'filename'      : fname,
                    'results'       : results,
                    'best_title'    : results[0]['title'],
                    'resume_preview': preview,
                    'word_count'    : len(clean(text).split()),
                })

        return jsonify({'results_per_file': all_results})

    # ── Plain text ─────────────────────────────────────
    text = ''
    if request.is_json:
        text = (request.get_json(silent=True) or {}).get('text', '')
    else:
        text = request.form.get('text', '')

    text = text.strip()
    if not text:
        return jsonify({'error': 'No resume provided.'}), 400

    results = recommend(text, top_n=8)
    if not results:
        return jsonify({'error': 'No matches found. Add more skill keywords.'}), 200

    preview = ' '.join(text.split()[:40]) + ('...' if len(text.split()) > 40 else '')
    return jsonify({
        'results'        : results,
        'best_title'     : results[0]['title'],
        'resume_preview' : preview,
        'total_matched'  : len(results),
    })


@app.route('/api/stats')
def api_stats():
    top = corpus_df['job_title'].value_counts().head(10)
    return jsonify({
        'labels'       : top.index.tolist(),
        'counts'       : top.values.tolist(),
        'total_jobs'   : TOTAL_JOBS,
        'total_titles' : TOTAL_TITLES,
    })


@app.route('/api/debug', methods=['POST'])
def api_debug():
    text = ''
    if request.files.get('resume'):
        f = request.files['resume']
        text = extract_file(f.filename or '', f.read())
    elif request.is_json:
        text = (request.get_json(silent=True) or {}).get('text', '')
    else:
        text = request.form.get('text', '')

    cleaned = clean(text)
    vec     = TFIDF.transform([cleaned])
    scores  = cosine_similarity(vec, MATRIX).flatten()
    top10   = scores.argsort()[::-1][:10]
    return jsonify({
        'raw_chars'     : len(text),
        'cleaned_words' : len(cleaned.split()),
        'preview'       : cleaned[:300],
        'non_zero'      : int(np.sum(scores > 0)),
        'max_score'     : round(float(scores.max()), 5),
        'top_10'        : [
            {'title': CORPUS.iloc[i]['job_title'], 'score': round(float(scores[i]), 5)}
            for i in top10
        ],
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)