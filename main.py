from retrieval_module import fetch_articles
from nli_fact_checker import retrieve_top_k, nli_probs, compute_verdict

headline = input("Enter a news headline: ")
print(f"\n--- HEADLINE: {headline} ---")


articles = fetch_articles(headline)
if not articles:
    print("No articles found regarding this headline so high chance of being false.")
    exit()


sources = [a['summary'] for a in articles if a['summary']]
if not sources:
    print("No valid summaries found for this headline.")
    exit()

evidence, sem_sim = retrieve_top_k(headline, sources)


probs = nli_probs(headline, evidence)


verdict = compute_verdict(probs, sem_sim)


print(f"Claim: {headline}")
print(f"Combined Evidence:\n{evidence}\n")
print(f"NLI Probabilities: {probs}")
print(f"Semantic Similarity: {sem_sim:.3f}")
print(f"Final Verdict → {verdict}")

