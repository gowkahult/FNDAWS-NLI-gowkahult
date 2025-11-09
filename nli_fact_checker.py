
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import logging
import sys
from collections import OrderedDict

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



def split_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.split()) > 4]  # skip very short sentences


flan_model_name = "google/flan-t5-small"
save_directory_norm = "./models/normalizer"
norm_tokenizer = AutoTokenizer.from_pretrained(save_directory_norm)
print("BRUHHHHHHH")
norm_model = AutoModelForSeq2SeqLM.from_pretrained(save_directory_norm)



def normalize_text(text):
    prompt = f"Normalize this text: {text}"
    inputs = norm_tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = norm_model.generate(**inputs, max_length=256)
    return norm_tokenizer.decode(outputs[0], skip_special_tokens=True)

save_directory_sent= './models/sent_transformer'
retriever = SentenceTransformer(save_directory_sent)

def clean_sentence(s):
    """Clean and normalize text."""
    s = re.sub(r'\s+', ' ', s.strip())
    s = re.sub(r'[“”"]', '', s)
    s = s.replace("’", "'")
    if not s.endswith(('.', '!', '?')):
        s += '.'
    return s[0].upper() + s[1:] if s else s

def retrieve_top_k(claim, sources, top_k=3):
    """
    Retrieve top-k coherent, high-quality evidence sentences related to the claim.
    Produces a human-readable summary from multi-source context.
    """
    all_sentences = []
    for src in sources:
        sentences = split_sentences(src)
        all_sentences.extend(sentences)

    # Deduplicate sentences (preserve order)
    all_sentences = list(OrderedDict.fromkeys(all_sentences))

    if not all_sentences:
        return "No relevant sentences found.", 0.0

    # Encode claim and sentences
    claim_emb = retriever.encode(claim, convert_to_tensor=True)
    sent_embs = retriever.encode(all_sentences, convert_to_tensor=True)

    # Compute similarity
    scores = util.cos_sim(claim_emb, sent_embs)[0].cpu().numpy()

    # Get top-k by similarity
    top_idx = np.argsort(scores)[-top_k * 2:][::-1]  # fetch slightly more for filtering

    # Select diverse top sentences (avoid same article or rephrases)
    selected = []
    seen_phrases = set()
    for i in top_idx:
        sent = clean_sentence(all_sentences[i])
        if any(p in sent.lower() for p in seen_phrases):
            continue
        seen_phrases.update(sent.lower().split()[:4])  # block duplicates
        selected.append(sent)
        if len(selected) >= top_k:
            break

    # Merge into coherent paragraph
    combined_evidence = " ".join(selected)

    # Optional: fix flow with slight reordering by similarity descending
    ordered = sorted(zip(selected, scores[top_idx][:len(selected)]), key=lambda x: -x[1])
    combined_evidence = " ".join([s for s, _ in ordered])

    semantic_sim = float(np.mean(scores[top_idx][:len(selected)]))

    return combined_evidence.strip(), semantic_sim


nli_model_name = "MoritzLaurer/deberta-v3-base-mnli-fever-anli"
save_directory_nli = "./models/nli_model"
nli_tokenizer = AutoTokenizer.from_pretrained(save_directory_nli)
print("BRUH")
nli_model = AutoModelForSequenceClassification.from_pretrained(save_directory_nli)


nli_model.eval()
labels = nli_model.config.id2label  # entailment=2, neutral=1, contradiction=0

def nli_probs(claim, evidence):
    inputs = nli_tokenizer(claim, evidence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
    return {labels[i]: float(probs[i]) for i in range(len(probs))}


def compute_verdict(probs, semantic_sim, sem_threshold=0.85, ent_threshold=0.6, contr_threshold=0.44):
    entail = probs['entailment']
    neutral = probs['neutral']
    contr = probs['contradiction']

    if semantic_sim >= sem_threshold:
        return "LIKELY TRUE (semantic override)" if entail > contr else "LIKELY FALSE (semantic override)"
    if entail >= ent_threshold:
        return "LIKELY TRUE"
    elif contr >= contr_threshold:
        return "LIKELY FALSE"
    elif neutral >= 0.5:
        return "UNCERTAIN"
    else:
        return "UNCERTAIN"
