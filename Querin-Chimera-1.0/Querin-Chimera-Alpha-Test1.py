from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
import os

base_model_name = "meta-llama/Llama-2-7b-chat-hf"
lora_model_name = "utkmst/chimera-alpha-test1"

def load_model():
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=False)
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.float16)
            model = PeftModel.from_pretrained(base_model, lora_model_name)
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.pad_token = tokenizer.eos_token
            return model, tokenizer
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None
    else:
        print("HF_TOKEN environment variable not set.")
        return None, None

model, tokenizer = load_model()

if model is None or tokenizer is None:
    exit(1)

def generate_response(prompt, chat_history):
    full_prompt = ""
    for user_msg, bot_msg in chat_history:
        full_prompt += f"[INST] {user_msg.strip()} [/INST] {bot_msg.strip()} </s>\n"
    full_prompt += f"[INST] {prompt.strip()} [/INST]"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device) 

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(full_prompt):].strip()
    return response   

def chatbot():
    chat_history = []
    print("Welcome to Querin Chimera-Alpha Test 1.0! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = generate_response(user_input, chat_history)
        print("Chimera: ", response)
        chat_history.append((user_input, response))

if __name__ == "__main__":
    chatbot()