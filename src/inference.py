# Load model directly
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Additional setup to handle padding properly
def setup_tokenizer_and_model():

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype=torch.float16,  # Use half precision to save memory
        device_map="auto",  # Automatically handle device placement
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Configure model padding settings
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    return tokenizer, model


def get_model_output(prompt: str, model, tokenizer):

    # Encode with padding and attention mask
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        return_attention_mask=True,
    )

    # Make sure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Move input tensors to GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=512,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Move output back to CPU for decoding
    outputs = outputs.cpu()
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def optimize_memory():
    torch.cuda.empty_cache()  # Clear GPU cache

    # print(torch.cuda.is_available())
    # if torch.cuda.is_available():
    #     print(f"GPU Memory Usage:")
    #     print(f"Allocated: {torch.cuda.memory_allocated(0) // 1024**2}MB")
    #     print(f"Cached: {torch.cuda.memory_reserved(0) // 1024**2}MB")


# if __name__ == "__main__":

#     optimize_memory()

#     sequence = "What's a polar bear?"
#     tokenizer, model = setup_tokenizer_and_model()

#     try:
#         response = get_model_output(sequence, model=model, tokenizer=tokenizer)

#         print("res1: ", response)

#         # response = get_model_output_with_rag(
#         #     "Kaj je severni medved?",
#         #     "Severni medved je velika zver, ki živi na severu in se prehranjuje s tjulni.",
#         # )

#         # print("\nres2: ", response)
#     except RuntimeError as e:
#         if "out of memory" in str(e):
#             torch.cuda.empty_cache()
#             print("GPU out of memory, trying again with cleared cache...")
#             response = get_model_output_without_rag(sequence)
#         else:
#             raise e


# tokenizer.decode(output.logits[0].argmax(dim=-1).tolist())
