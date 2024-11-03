import time
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and move to appropriate device
tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta")
model = AutoModelForMaskedLM.from_pretrained("EMBEDDIA/sloberta").to(device)


def embedd_sequences(sequences) -> torch.Tensor:
    # Tokenize all sequences at once with padding
    embeddings = tokenizer(
        sequences, padding=True, truncation=True, return_tensors="pt"
    )

    # Move input tensors to device
    input_ids = embeddings["input_ids"].to(device)
    attention_mask = embeddings["attention_mask"].to(device)

    # Get embeddings
    with torch.no_grad():  # disable gradient calculation for inference
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Mean pooling
    token_embeddings = outputs.logits  # Shape: [batch_size, seq_length, vocab_size]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

    # If you need the embeddings on CPU for further processing, uncomment the following line:
    embeddings = embeddings.cpu()

    return embeddings


if __name__ == "__main__":

    sequences = [
        "Severni medved je eden njvečjih medvedov na svetu.",
        "Severni medved je zelo velik.",
        "Severni medved je zelo velik medved.",
        "Severni medved je zelo velik medved, ki živi na severu.",
        "Severni medved je zelo velik medved, ki živi na severu in se prehranjuje s tjulni.",
        "Severni medved je zelo velik medved, ki živi na severu in se prehranjuje s tjulni. Je zelo nevaren.",
        "Severni medved je zelo velik medved, ki živi na severu in se prehranjuje s tjulni. Je zelo nevaren za ljudi.",
        "Severni medved je zelo velik medved, ki živi na severu in se prehranjuje s tjulni. Je zelo nevaren za ljudi. Ljudje se ga bojijo.",
        "Tjulen je zelo okusna hrana za severnega medveda.",
        "Tjulen je zelo okusna hrana za severnega medveda. Severni medved se prehranjuje s tjulni.",
        "Tjulen je zelo okusna hrana za severnega medveda. Severni medved se prehranjuje s tjulni. Tjuleni so zelo okusni.",
        "Tjulen je zelo okusna hrana za severnega medveda. Severni medved se prehranjuje s tjulni. Tjuleni so zelo okusni. Tjuleni so zelo okusni za severnega medveda.",
        "Mraz je zelo velik problem za severnega medveda.",
        "Mraz je zelo velik problem za severnega medveda. Severni medved se ne mara mraza.",
        "Mraz je zelo velik problem za severnega medveda. Severni medved se ne mara mraza. Severni medved se ne mara mraza.",
        "Mraz je zelo velik problem za severnega medveda. Severni medved se ne mara mraza. Severni medved se ne mara mraza. Severni medved se ne mara mraza.",
        "Mraz je zelo velik problem za severnega medveda. Severni medved se ne mara mraza. Severni medved se ne mara mraza. Severni medved se ne mara mraza. Severni medved se ne mara mraza.",
        "Severni medved je eden njvečjih medvedov na svetu.",
        "Severni medved je zelo velik.",
        "Severni medved je zelo velik medved.",
        "Severni medved je zelo velik medved, ki živi na severu.",
        "Severni medved je zelo velik medved, ki živi na severu in se prehranjuje s tjulni.",
        "Severni medved je zelo velik medved, ki živi na severu in se prehranjuje s tjulni. Je zelo nevaren.",
        "Severni medved je zelo velik medved, ki živi na severu in se prehranjuje s tjulni. Je zelo nevaren za ljudi.",
        "Severni medved je zelo velik medved, ki živi na severu in se prehranjuje s tjulni. Je zelo nevaren za ljudi. Ljudje se ga bojijo.",
        "Tjulen je zelo okusna hrana za severnega medveda.",
        "Tjulen je zelo okusna hrana za severnega medveda. Severni medved se prehranjuje s tjulni.",
        "Tjulen je zelo okusna hrana za severnega medveda. Severni medved se prehranjuje s tjulni. Tjuleni so zelo okusni.",
        "Tjulen je zelo okusna hrana za severnega medveda. Severni medved se prehranjuje s tjulni. Tjuleni so zelo okusni. Tjuleni so zelo okusni za severnega medveda.",
        "Mraz je zelo velik problem za severnega medveda.",
        "Mraz je zelo velik problem za severnega medveda. Severni medved se ne mara mraza.",
        "Mraz je zelo velik problem za severnega medveda. Severni medved se ne mara mraza. Severni medved se ne mara mraza.",
        "Mraz je zelo velik problem za severnega medveda. Severni medved se ne mara mraza. Severni medved se ne mara mraza. Severni medved se ne mara mraza.",
        "Mraz je zelo velik problem za severnega medveda. Severni medved se ne mara mraza. Severni medved se ne mara mraza. Severni medved se ne mara mraza. Severni medved se ne mara mraza.",
        "Severni medved je eden njvečjih medvedov na svetu.",
        "Severni medved je zelo velik.",
        "Severni medved je zelo velik medved.",
        "Severni medved je zelo velik medved, ki živi na severu.",
        "Severni medved je zelo velik medved, ki živi na severu in se prehranjuje s tjulni.",
        "Severni medved je zelo velik medved, ki živi na severu in se prehranjuje s tjulni. Je zelo nevaren.",
        "Severni medved je zelo velik medved, ki živi na severu in se prehranjuje s tjulni. Je zelo nevaren za ljudi.",
        "Severni medved je zelo velik medved, ki živi na severu in se prehranjuje s tjulni. Je zelo nevaren za ljudi. Ljudje se ga bojijo.",
        "Tjulen je zelo okusna hrana za severnega medveda.",
        "Tjulen je zelo okusna hrana za severnega medveda. Severni medved se prehranjuje s tjulni.",
        "Tjulen je zelo okusna hrana za severnega medveda. Severni medved se prehranjuje s tjulni. Tjuleni so zelo okusni.",
        "Tjulen je zelo okusna hrana za severnega medveda. Severni medved se prehranjuje s tjulni. Tjuleni so zelo okusni. Tjuleni so zelo okusni za severnega medveda.",
        "Mraz je zelo velik problem za severnega medveda.",
        "Mraz je zelo velik problem za severnega medveda. Severni medved se ne mara mraza.",
        "Mraz je zelo velik problem za severnega medveda. Severni medved se ne mara mraza. Severni medved se ne mara mraza.",
        "Mraz je zelo velik problem za severnega medveda. Severni medved se ne mara mraza. Severni medved se ne mara mraza. Severni medved se ne mara mraza.",
        "Mraz je zelo velik problem za severnega medveda. Severni medved se ne mara mraza. Severni medved se ne mara mraza. Severni medved se ne mara mraza. Severni medved se ne mara mraza.",
    ]

    t1 = time.time()
    embeddings = embedd_sequences(sequences)
    print(embeddings)
    print(f"Time taken: {time.time() - t1:.2f}s")
