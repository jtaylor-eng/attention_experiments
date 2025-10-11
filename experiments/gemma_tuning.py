from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")

# input_text = "Michael Jordan plays for the "
# input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
# outputs = model.generate(**input_ids)
# print(tokenizer.decode(outputs[0]))

print(type(model))
print(model)
