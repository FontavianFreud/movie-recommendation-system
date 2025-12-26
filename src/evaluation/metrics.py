def precision_at_k(recommended_ids, relevant_ids, k):
    """
    Precision@K = (# recommended items that are relevant) / K
    """
    if k == 0:
        return 0.0
    rec_k = recommended_ids[:k]
    hits = sum(1 for mid in rec_k if mid in relevant_ids)
    return hits / k


def recall_at_k(recommended_ids, relevant_ids, k):
    """
    Recall@K = (# relevant items that were recommended) / (# relevant items)
    """
    if not relevant_ids:
        return 0.0
    rec_k = recommended_ids[:k]
    hits = sum(1 for mid in rec_k if mid in relevant_ids)
    return hits / len(relevant_ids)
