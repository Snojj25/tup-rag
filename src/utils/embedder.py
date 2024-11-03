import time
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)


# Mean Pooling - Take attention mask into account for correct averaging
def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    ).to(device)
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def embedd_sequences(sequences) -> torch.Tensor:
    # Tokenize all sequences at once with padding
    encoded_input = tokenizer(
        sequences, padding=True, truncation=True, return_tensors="pt"
    )

    # Move input tensors to device
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)

    # Get embeddings
    with torch.no_grad():  # disable gradient calculation for inference
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Perform pooling
    sentence_embeddings = _mean_pooling(outputs, encoded_input["attention_mask"])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    # If you need the embeddings on CPU for further processing, uncomment the following line:
    sentence_embeddings = sentence_embeddings.cpu()

    return sentence_embeddings


if __name__ == "__main__":

    sequences = [
        "Polar bears are very dangerous",
        "Polar bears kill and eat seals and penguins",
        "You should not approach a polar bear in the wild",
        "Dogs are friendly animals",
        "Dogs are loyal animals",
        "Dogs eat bones",
        "Dogs love to play fetch",
        "Birds are chirping animals",
        "Birds can fly",
        "Birds build nests",
        "Birds lay eggs",
        "Birds are beautiful animals",
        "Elephants are large animals",
        "Elephants have long trunks",
        "Elephants have big ears",
        "Elephants are herbivores",
        "Elephants are intelligent animals",
        "Cats are independent animals",
        "Cats are agile",
        "Cats are good hunters",
        "Cats are playful animals",
        "Cats are clean animals",
        "Cats are domesticated animals",
        "Cats are popular pets",
        "Cats are curious animals",
        "Cats are territorial animals",
        "Cats are nocturnal animals",
        "Potatoes are starchy vegetables",
        "Potatoes are tubers",
        "Potatoes are a staple food",
        "Potatoes are versatile vegetables",
        "Potatoes are grown underground",
        "Potatoes are rich in carbohydrates",
        "Potatoes are a good source of energy",
        "Polar bears are very dangerous",
        "Polar bears kill and eat seals and penguins",
        "You should not approach a polar bear in the wild",
        "Dogs are friendly animals",
        "Dogs are loyal animals",
        "Dogs eat bones",
        "Dogs love to play fetch",
        "Birds are chirping animals",
        "Birds can fly",
        "Birds build nests",
        "Birds lay eggs",
        "Birds are beautiful animals",
        "Elephants are large animals",
        "Elephants have long trunks",
        "Elephants have big ears",
        "Elephants are herbivores",
        "Elephants are intelligent animals",
    ]

    t1 = time.time()
    embeddings = embedd_sequences(sequences)
    print(embeddings)
    print(f"Time taken: {time.time() - t1:.2f}s")
