import re
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from transformers import pipeline
import logging
import sys

logging.basicConfig(
    filename='warnings.log',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class StderrToLogger(object):
    def write(self, message):
        if message.strip():
            logging.warning(message)
    def flush(self):
        pass

sys.stderr = StderrToLogger()

save_directory = "./models/summarizer"
summarizer = pipeline("summarization", model=save_directory, device=-1)



def chunked_summary(text, max_chunk_size=500):
    """
    Splits text into sentences, groups them into chunks, and summarizes each chunk.
    Ensures only relevant sentences (>5 words) are included and punctuation is preserved.
    """
    
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = ""

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
        except:
            continue 

    
    return " ".join(summaries)


def fetch_articles(headline, top_n=5):
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
        content = fetch_article_content(link)
        if content != "Failed to fetch content":
            summary = chunked_summary(content)
            articles.append({
                'title': title,
                'link': link,
                'content': content,
                'summary': summary
            })
    return articles

def fetch_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text.strip()
        return text if text else "Failed to fetch content"
    except:
        return "Failed to fetch content"
