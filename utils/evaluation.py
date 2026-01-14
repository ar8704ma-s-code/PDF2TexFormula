def evaluate(preds, golds):
    correct = 0
    total = len(preds)

    for p, g in zip(preds, golds):
        if p.get("correct") == g.get("correct"):
            correct += 1

    return {
        "accuracy": correct / total,
        "total": total
    }
