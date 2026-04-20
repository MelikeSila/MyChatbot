import time
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import MODEL_NAME


class ChatbotTextToText(object):
    def __init__(self):
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")

        # Device Pick
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        print("Using device:", self.device)

        self.llama_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            token=token,
            torch_dtype=torch.float16
        ).to(self.device)
        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            token=token
        )

        # Set the pad token if it's not already set
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.add_special_tokens({'pad_token': self.llama_tokenizer.eos_token})
            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

    def generate_response(self, prompt):
        inputs = self.llama_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        # Send inouts to GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.llama_model.generate(
            inputs['input_ids'],
            max_length=50,
            num_return_sequences=1,
            pad_token_id=self.llama_tokenizer.eos_token_id # Ensure pad_token_id is set to eos_token_id
        )
        response = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


    def generate_response_with_time(self, prompt):
        start_time = time.time()
        response = self.generate_response(prompt)
        end_time = time.time()
        response_time = end_time - start_time
        return response, response_time


    def chatbot(self):
        print("Welcome to the Chatbot! Type 'exit' to quit.")
        while True:
            prompt = input("You: ")
            if prompt.lower() == "exit":
                break
            response = self.generate_response(prompt)
            print(f"Bot: {response}")


    def chatbot_with_performance(self, message):
        response, response_time = self.generate_response_with_time(message)
        print(f"Bot: {response}")
        print(f"Response Time: {response_time:.2f} seconds")
        return response + " ( in " + str(response_time) + " seconds )"


    def ask_to_chatbot(self, message):
        # Chatbot with LLaMA 3.1
        return self.chatbot_with_performance(message)
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