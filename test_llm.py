from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
expected_tensor = [[ 50377,  99511,  23990,   3837,  50377,  99511,  23990,   3837,  50377,
          99511,  23990,   3837,  50377,  99511,  23990,   1773, 151645]]

model_path = "saved_models/llm/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
llm = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

inputs_embeds = torch.load("saved_tensors/llm_inputs_embeds.pt", map_location=device)
inputs_embeds = inputs_embeds.to(device)


generated_ids = llm.generate(inputs_embeds=inputs_embeds, max_new_tokens=512)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)