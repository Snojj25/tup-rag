from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta")
model = AutoModelForMaskedLM.from_pretrained("EMBEDDIA/sloberta")


sequence = "Severni medved je eden njvečjih medvedov na svetu."


def embedd_sentence(sequence):
    embedding = tokenizer(sequence)

    input_ids = torch.tensor([embedding["input_ids"]])
    attention_mask = torch.tensor([embedding["attention_mask"]])

    # Get embeddings
    with torch.no_grad():  # disable gradient calculation for inference
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Mean pooling
    token_embeddings = outputs.logits  # torch.Size([1, 27, 32005])
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    return embeddings  # torch.Size([1, 32005])


embedding = embedd_sentence(sequence)

print(embedding)
