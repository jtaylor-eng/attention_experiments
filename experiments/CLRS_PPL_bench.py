import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM #type:ignore
from clrs._src.clrs_text import huggingface_generators
from datasets import Dataset
from training import architecture
import numpy as np

m1 = torch.load('/fs/ess/PAS2836/jacktaylor/test/checkpoints/checkpoint_step_1000_1760122203.693063.pt')
m2 = torch.load('/fs/ess/PAS2836/jacktaylor/test/checkpoints/checkpoint_step_5000_1760123507.6963952.pt')

device = torch.device('cuda') #need cuda to load models
m1.to(device) ; m2.to(device)
m1.eval() ; m2.eval()

tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
tokenizer.pad_token = tokenizer.eos_token

def compute_perplexity(model, texts):
    losses = []
    for text in texts:
        enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = enc['input_ids'].to(device)

        with torch.no_grad():
            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        losses.append(loss.item())

    return float(np.exp(np.mean(losses)))


if __name__ == '__main__':
    #TODO: greater selection
    tasks = ['bfs', 'dfs', 'insertion_sort', 'binary_search', 'quicksort']
    results = {}

    for task_name in tasks:
        print(f'evaluating task: {task_name}')
        
        #generate dataset for task
        task_dataset = Dataset.from_generator(
            huggingface_generators.clrs_generator,
            gen_kwargs={
                'algos_and_lengths': {task_name: [16]},
                'num_samples': 50,
                'use_hints': False,
                'seed': 42
            }
        )
        
        task_texts = task_dataset['text']
        m1_ppl = compute_perplexity(m1, task_texts)
        m2_ppl = compute_perplexity(m2, task_texts)
        
        results[task_name] = {'m1': m1_ppl, 'm2': m2_ppl}
        print(f'm1 PPL: {m1_ppl:.2f}')
        print(f'm2 PPL: {m2_ppl:.2f}')

    print('summary of results:')
    for task_name, ppls in results.items(): print(f"{task_name}: m1={ppls['m1']:.2f}, m2={ppls['m2']:.2f}")
    
    #TODO: plotting / viz