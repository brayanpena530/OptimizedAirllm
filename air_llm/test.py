from airllm import AutoModel

MAX_Length = 128

model = AutoModel.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

input_text = [
        'What is the capital of the United States?',
        #'I like',
    ]

input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_Length, 
    padding=False)
           
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=1,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)