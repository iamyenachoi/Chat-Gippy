from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(prompt, max_length=100):
    try:
        if len(prompt) > max_length:
            prompt = prompt[:max_length]

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=device)

        output = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_length=max_length, 
            num_return_sequences=1, 
            no_repeat_ngram_size=2, 
            temperature=0.7
        )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response.strip()
    
    except Exception as e:
        return f"⚠️ Error Generating Response: {str(e)}"
