import os
import re
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from newspaper import Article
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import logging
import sys

# ==============================
# LOGGING SETUP
# ==============================
logging.basicConfig(
    filename='warnings.log',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class StderrToLogger(object):
    def write(self, message):
        if message.strip():
            logging.warning(message)
    def flush(self): pass

sys.stderr = StderrToLogger()

# ==============================
# PATHS
# ==============================
CSV_PATH = "articles_database.csv"
EMB_PATH = "article_embeddings.npy"
MODEL_PATH = "./models/summarizer"

# ==============================
# MODELS
# ==============================
summarizer = pipeline("summarization", model=MODEL_PATH, device=-1)
retriever = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# ==============================
# EXISTING FUNCTION NAMES (REFINED)
# ==============================

def chunked_summary(text, max_chunk_size=500):
    """
    Splits long text into readable chunks and summarizes each.
    Keeps punctuation and avoids short (<5 words) sentences.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, current_chunk = [], ""

    for sentence in sentences:
        if len(sentence.split()) < 5:
            continue
        if len(current_chunk.split()) + len(sentence.split()) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())

    summaries = []
    for chunk in chunks:
        try:
            input_len = len(chunk.split())
            max_len = min(130, max(10, input_len // 2))
            summary = summarizer(
                chunk,
                max_new_tokens=max_len,
                min_length=10,
                do_sample=False
            )[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            logging.warning(f"Summarization failed for chunk: {e}")
            continue

    return " ".join(summaries)


def fetch_article_content(url):
    """
    Fetches and cleans full article text from a given URL.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text.strip()
        return text if text else "Failed to fetch content"
    except Exception as e:
        logging.warning(f"Error fetching article content: {e}")
        return "Failed to fetch content"


def fetch_articles(headline, top_n=5):
    """
    Searches local database for related articles using embeddings.
    Falls back to web scraping via Bing if none found.
    Automatically updates the CSV and embedding database.
    """
    # Step 1 — Try Local Database Search First
    print(f"🔍 Searching local database for: {headline}")
    local_results, score = local_article_search(headline)

    if local_results is not None:
        print(f"✅ Found related articles locally (similarity={score:.2f})")
        return local_results.to_dict(orient="records")

    # Step 2 — Fallback: Web Scraping
    print("⚠️ No similar articles found locally. Fetching from the web...")
    web_results = fetch_articles_from_web(headline, top_n)
    if web_results:
        print(f"🌐 {len(web_results)} new articles added to local database.")
        return web_results
    else:
        print("❌ No relevant articles found online either.")
        return []


def fetch_articles_from_web(headline, top_n=5):
    """
    Scrapes Bing News for new articles about the given headline.
    Summarizes and stores them in CSV with updated embeddings.
    """
    query = headline.replace(' ', '+')
    url = f"https://www.bing.com/news/search?q={query}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    results = soup.find_all('a', {'class': 'title'})[:top_n]
    articles = []

    for result in results:
        title = result.text.strip()
        link = result['href']

        # Avoid duplicates
        if os.path.exists(CSV_PATH):
            existing_df = pd.read_csv(CSV_PATH)
            if link in existing_df['link'].values:
                continue

        content = fetch_article_content(link)
        if content != "Failed to fetch content":
            summary = chunked_summary(content)
            articles.append({
                'title': title,
                'link': link,
                'content': content,
                'summary': summary
            })

    if not articles:
        return []

    # Save to CSV
    df_new = pd.DataFrame(articles)
    if os.path.exists(CSV_PATH):
        df_existing = pd.read_csv(CSV_PATH)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(CSV_PATH, index=False)

    # Update embeddings
    new_embeddings = retriever.encode(df_new['summary'].tolist(), convert_to_numpy=True)
    if os.path.exists(EMB_PATH):
        old_embs = np.load(EMB_PATH)
        combined_embs = np.vstack((old_embs, new_embeddings))
    else:
        combined_embs = new_embeddings
    np.save(EMB_PATH, combined_embs)

    return articles


def local_article_search(claim, threshold=0.5, top_k=5):
    """
    Searches the local CSV + FAISS index for semantically similar articles.
    Returns (DataFrame, best_score) if found, else (None, score).
    """
    if not (os.path.exists(CSV_PATH) and os.path.exists(EMB_PATH)):
        return None, 0

    df = pd.read_csv(CSV_PATH)
    embeddings = np.load(EMB_PATH)

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    claim_emb = retriever.encode([claim], convert_to_numpy=True)
    faiss.normalize_L2(claim_emb)

    scores, ids = index.search(claim_emb, top_k)
    top_ids, top_scores = ids[0], scores[0]
    best_match_score = np.max(top_scores)

    if best_match_score < threshold:
        return None, best_match_score

    top_articles = df.iloc[top_ids]
    return top_articles, best_match_score
