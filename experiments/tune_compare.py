import inspect
import matplotlib.pyplot as plt
import re

from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers.models.gemma.modeling_gemma import GemmaAttention
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset, concatenate_datasets
from typing import Tuple

from clrs._src import algorithms
from clrs._src.clrs_text import huggingface_generators

import absl.logging ; absl.logging.set_verbosity(absl.logging.ERROR)


class AdaptiveTemperatureAttn(GemmaAttention):
    """Drop adaptive_temperature_softmax in for F.softmax so model tunes using it in attention."""
    def forward(self, *args, **kwargs):
        orig_softmax = F.softmax

        F.softmax = adaptive_temperature_softmax #compute output using our attn
        out = super().forward(*args, **kwargs)

        F.softmax = orig_softmax #ensure softmax is set back to traditional

        return out

 
def adaptive_temperature_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Implements the adaptive temperature softmax from the paper translated to pytorch."""
    poly_fit = torch.tensor([-0.037, 0.481, -2.3, 4.917, -1.791], device=logits.device)
    
    #calculate initial probs & entropy
    with torch.no_grad():
        original_probs = F.softmax(logits, dim=-1)
        entropy = torch.sum(-original_probs * torch.log(original_probs + 1e-9), dim=-1, keepdim=True)

    #Gemini: TODO: check
    # Calculate beta (1/theta) based on the polynomial fit [cite: 224, 226]
    # The paper's JAX code uses polyval, which evaluates from the highest power.
    # PyTorch's polyval needs the tensor of powers, so we build it.
    pows = torch.arange(len(poly_fit) - 1, -1, -1, device=logits.device)
    entropy_pows = entropy ** pows
    beta = torch.sum(poly_fit * entropy_pows, dim=-1, keepdim=True)
    
    beta = torch.where(
        entropy > 0.5,
        torch.maximum(beta, torch.tensor(1.0, device=logits.device)),
        torch.tensor(1.0, device=logits.device)
    )
    
    return F.softmax(logits * beta, dim=-1, dtype=torch.float32).to(logits.dtype)


def swap_gemma_attention_layers(model):
    """Recursively traverses model and replaces each GemmaAttention w/ AdaptiveTempAttn"""
    for name, module in model.named_children():
        if isinstance(module, GemmaAttention): #is gemma attn, replace
            adaptive_layer = AdaptiveTemperatureAttn(
                config=module.config,
                layer_idx=module.layer_idx
            ).to(device=module.q_proj.weight.device, dtype=module.q_proj.weight.dtype)

            adaptive_layer.load_state_dict(module.state_dict())
            setattr(model, name, adaptive_layer)

        else: swap_gemma_attention_layers(module) #recurse
    return model

    
def test_gen(model, tokenizer):
    """Ensure functionality to generate from model w/ tokenizer."""
    input_text = 'Michael Jordan plays for the '
    input_ids = tokenizer(input_text, return_tensors='pt').to('cuda')
    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))


def get_train_val_splits(
    algorithm: str,
    train_range: Tuple[int,int] = (4, 9),
    val_range: Tuple[int,int] = (8, 17),
    train_samples: int = 50,
    val_samples: int = 16
):
    """
    given CLRS algorithm (dfs, bfs, dijkstra, ...) generate train_samples trajectories and store in train split,
    val_samples and store in val split.

    train_range, val_range is sequence length for trajectories in the respective sets. To test OOD generations,
    the upper bound of the val range should exceed the UB of train range.

    Returns: dict with keys 'train' and 'val' of HF Datasets, or None if generation fails.
    """
    assert train_range[0] >= 4 and train_range[1] <= 64, 'Train range must be within (4, 64)'
    assert val_range[0] >= 4 and val_range[1] <= 128, 'Validation range must be within (4, 128)'
    assert val_range[1] > train_range[1], 'Validation upper bound should be larger than training upper bound to test OOD sequences'

    # The CLRS generator uses range(start, end), so upper bound is exclusive
    train_lens = list(range(train_range[0], train_range[1]))
    val_lens = list(range(val_range[0], val_range[1]))

    splits = {}
    for split_name, lens in [('train', train_lens), ('val', val_lens)]:
        samples = train_samples if split_name == 'train' else val_samples
        try:
            ds = Dataset.from_generator(
                huggingface_generators.clrs_generator,
                gen_kwargs={
                    'algos_and_lengths': {algorithm: lens},
                    'num_samples': samples,
                    'use_hints': False,
                    'seed': 1337
                }
            )
            splits[split_name] = ds
        except Exception as e:
            print(f'[WARN] Failed to generate {split_name} split for algo {algorithm}, lens {lens[:5]}...: {e}')
            return

    return splits


def preprocess_dataset(dataset, tokenizer):
    """Format & Tokenize CLRS trajectories to use in tuning."""
    def format_example(example):
        example['formatted_text'] = f"{example['text']}\n{example['question']}\nAnswer: {example['answer']}"
        return example
    
    dataset = dataset.map(format_example)

    def tokenize(batch):
        tokenized = tokenizer(
            batch['formatted_text'],
            truncation=True,
            max_length=2048,
            padding='max_length'
        )
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized

    return dataset.map(tokenize, batched=True, remove_columns=['formatted_text'])


def fine_tune_model(model, tokenizer, dataset, output_dir='./checkpoints', epochs=3):
    """Use HF Trainer to tune model given dataset."""
    tokenized = preprocess_dataset(dataset, tokenizer)
    
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=epochs,
        logging_steps=50,
        save_strategy='no',
        report_to='none',
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    trainer.train()
    return model


def evaluate_accuracy(model, tokenizer, dataset, batch_size=128):
    """Computes prediction accuracy over a given dataset using batching for speed."""
    model.eval()
    correct = 0
    total = 0

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating Accuracy", leave=False):
        batch_examples = dataset[i : i + batch_size]
        
        prompts = [f"{text}\n{question}\nAnswer: " for text, question in zip(batch_examples['text'], batch_examples['question'])]
        
        inputs = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )

        # Decode all generated sequences in the batch at once
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ground_truths = batch_examples['answer']

        # Compare results
        for j, decoded_output in enumerate(decoded_outputs):
            answer_part = decoded_output.split("Answer: ")[-1]
            prediction = answer_part.strip().split('\n')[0]
            ground_truth = ground_truths[j].strip()

            norm_prediction = re.sub(r'[^0-9\s]', '', prediction).strip()
            norm_ground_truth = re.sub(r'[^0-9\s]', '', ground_truth).strip()
            
            if norm_prediction == norm_ground_truth:
                correct += 1
            total += 1

    return (correct / total) if total > 0 else 0

def run_evaluation(algorithms, validation_datasets, model, tokenizer):
    """
    Runs evaluation for a given model and returns the accuracies.
    This replaces the evaluation loop inside the plotting function.
    """
    accuracies_by_algo = {}
    for algo, val_set in tqdm(zip(algorithms, validation_datasets), total=len(algorithms), desc=f"Evaluating model"):
        unique_lengths = sorted(list(set(ex['length'] for ex in val_set)))
        accuracies = []
        for length in unique_lengths:
            subset = val_set.filter(lambda example: example['length'] == length, num_proc=8)
            if len(subset) > 0:
                acc = evaluate_accuracy(model, tokenizer, subset)
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        accuracies_by_algo[algo] = {
            'lengths': unique_lengths,
            'accuracies': accuracies
        }
    return accuracies_by_algo


def plot_ood_accuracy(algorithms, all_results, n_cols=6):
    """
    For each algorithm, plots the accuracy vs. test length for all models.
    'all_results' is a dict like {'baseline': baseline_results, 'adaptive': adaptive_results}
    """
    num_algos = len(algorithms)
    n_rows = (num_algos + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    plot_configs = {
        'baseline': {'label': 'Gemma 2B (Tuned)', 'color': 'tab:blue'},
        'adaptive': {'label': 'Gemma 2B (Tuned + Entropy)', 'color': 'tab:red'}
    }

    for i, algo in enumerate(tqdm(algorithms, total=num_algos, desc="Plotting algos")):
        ax = axes[i]
        for model_name, results_for_model in all_results.items():
            config = plot_configs[model_name]
            data = results_for_model[algo]
            ax.plot(data['lengths'], data['accuracies'], marker='o', label=config['label'], color=config['color'])

        ax.set_title(algo.replace('_', ' ').title())
        ax.set_xlabel('Test length')
        ax.set_ylabel('Accuracy')
        ax.grid(True)
        ax.set_ylim(-0.05, 1.05)
    
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=len(all_results))
    
    for j in range(num_algos, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('accuracy_ood_comparison.png')
    print("\nSaved OOD accuracy comparison plot to accuracy_ood_comparison.png")


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    all_algos = ['bfs', 'dfs', 'insertion_sort', 'binary_search', 'quicksort', 'find_maximum_subarray_kadane', 'activity_selector', 'kmp_matcher', 'minimum', 'naive_string_matcher']
    
    tune_sets, val_sets, valid_algos_list = [], [], []
    for algo in tqdm(all_algos, desc="Generating Datasets"):
        split = get_train_val_splits(algo)
        if split is None: continue
        tune_sets.append(split['train'])
        val_sets.append(split['val'])
        valid_algos_list.append(algo)
        
    tune_sets = concatenate_datasets(tune_sets).shuffle(seed=1337)

    all_results = {}
    model_configs = ['baseline', 'adaptive']

    for config in model_configs:
        print(f"\n{'='*20} RUNNING EXPERIMENT: {config.upper()} {'='*20}")
        
        # model = AutoModelForCausalLM.from_pretrained('google/gemma-2b', device_map='auto', torch_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained('google/gemma-2b', torch_dtype=torch.bfloat16)
        if config == 'adaptive': model = swap_gemma_attention_layers(model)

        print(f'Fine-tuning on {len(tune_sets)} examples...')
        model_tuned = fine_tune_model(model, tokenizer, tune_sets)

        all_results[config] = run_evaluation(valid_algos_list, val_sets, model_tuned, tokenizer)

    plot_ood_accuracy(valid_algos_list, all_results)