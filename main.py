import time
import os
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM


def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=15,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id  # Ensure pad_token_id is set to eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def generate_response_with_time(model, tokenizer, prompt):
    start_time = time.time()
    response = generate_response(model, tokenizer, prompt)
    end_time = time.time()
    response_time = end_time - start_time
    return response, response_time


def chatbot(model, tokenizer):
    print("Welcome to the Chatbot! Type 'exit' to quit.")
    while True:
        prompt = input("You: ")
        if prompt.lower() == "exit":
            break
        response = generate_response(model, tokenizer, prompt)
        print(f"Bot: {response}")


def chatbot_with_performance(model, tokenizer):
    print("Welcome to the Chatbot! Type 'exit' to quit.")
    while True:
        prompt = input("You: ")
        if prompt.lower() == "exit":
            break
        response, response_time = generate_response_with_time(model, tokenizer, prompt)
        print(f"Bot: {response}")
        print(f"Response Time: {response_time:.2f} seconds")


if __name__ == "__main__":
    token = os.getenv("HUGGINGFACE_HUB_TOKEN")

    # Load LLaMA 3.1 Model
    llama_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name, token=token)
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, token=token)

    # Set the pad token if it's not already set
    if llama_tokenizer.pad_token is None:
        llama_tokenizer.add_special_tokens({'pad_token': llama_tokenizer.eos_token})
        llama_model.resize_token_embeddings(len(llama_tokenizer))

    # Chatbot with LLaMA 3.1
    print("Chatting with LLaMA 3.1")
    chatbot_with_performance(llama_model, llama_tokenizer)

    '''
    # Load Apple/DCLM-7B Model
    dclm_model_name = "apple/DCLM-7B"
    dclm_model = AutoModelForCausalLM.from_pretrained(dclm_model_name, token=token)
    dclm_tokenizer = AutoTokenizer.from_pretrained(dclm_model_name, token=token)

    # Set the pad token if it's not already set
    if dclm_tokenizer.pad_token is None:
        dclm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        dclm_model.resize_token_embeddings(len(dclm_tokenizer))

    # Chatbot with Apple/DCLM-7B
    print("Chatting with Apple/DCLM-7B")
    chatbot_with_performance(dclm_model, dclm_tokenizer)
    '''