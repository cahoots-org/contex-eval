def retrieval_prf(retrieved: list[str], gold: list[str]) -> tuple[float, float, float]:
    R, G = set(retrieved), set(gold)
    tp = len(R & G)
    precision = tp / len(R) if R else 0.0
    recall = tp / len(G) if G else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1
