#asked GPT to fix the hellish CLRS / HF Dataset interface issues
import inspect
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset, concatenate_datasets
from typing import Tuple

import clrs
from clrs._src import algorithms
from clrs._src.clrs_text import huggingface_generators

import absl.logging ; absl.logging.set_verbosity(absl.logging.ERROR)


def get_train_val_splits(
    algorithm: str,
    train_range: Tuple[int,int] = (4, 16),
    val_range: Tuple[int,int] = (4, 32),
    train_samples: int = 50,
    val_samples: int = 10
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
        # This function now just adds a new column 'formatted_text'
        example['formatted_text'] = f"{example['text']}\n{example['question']}\nAnswer: {example['answer']}"
        return example

    # Keep the original 'text', 'question', and 'answer' columns
    dataset = dataset.map(format_example)

    def tokenize(batch):
        # Tokenize the newly created 'formatted_text' for training
        tokenized = tokenizer(
            batch['formatted_text'], # Use the formatted text here
            padding='max_length',
            truncation=True,
            max_length=512,
        )
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized

    # We tokenize a different column, so we need to tell map to keep the old ones
    return dataset.map(tokenize, batched=True, remove_columns=['formatted_text'])


def fine_tune_model(model, tokenizer, dataset, output_dir='./checkpoints', epochs=3):
    """Use HF Trainer to tune model given dataset."""
    tokenized = preprocess_dataset(dataset, tokenizer)
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        num_train_epochs=epochs,
        logging_steps=50,
        save_strategy='no',
        report_to='none',
        bf16=True,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    trainer.train()
    return model


def evaluate_accuracy(model, tokenizer, dataset):
    """Computes prediction accuracy over a given dataset with normalization."""
    model.eval()
    correct = 0
    total = 0

    for example in tqdm(dataset, desc="Evaluating Accuracy", leave=False):
        prompt = f"{example['text']}\n{example['question']}\nAnswer: "
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                pad_token_id=tokenizer.eos_token_id,
                # do_sample=False
            )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer_part = decoded_output.split("Answer: ")[-1]
        prediction = answer_part.strip().split('\n')[0]
        ground_truth = example['answer'].strip()

        # --- NORMALIZATION STEP ---
        norm_prediction = re.sub(r'[^0-9\s]', '', prediction).strip()
        norm_ground_truth = re.sub(r'[^0-9\s]', '', ground_truth).strip()
        
        if norm_prediction == norm_ground_truth:
            correct += 1
        total += 1

    return (correct / total) if total > 0 else 0


def plot_ood_accuracy(algorithms, validation_datasets, model, tokenizer, n_cols=6):
    """For each algorithm, compute and plot accuracy vs. test length."""
    num_algos = len(algorithms)
    n_rows = (num_algos + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten()

    for i, (algo, val_set) in enumerate(tqdm(zip(algorithms, validation_datasets), total=num_algos, desc="Plotting algos")):
        ax = axes[i]
        
        unique_lengths = sorted(list(set(ex['length'] for ex in val_set)))
        accuracies = []

        for length in unique_lengths:
            subset = val_set.filter(lambda example: example['length'] == length, num_proc=4)
            if len(subset) > 0:
                acc = evaluate_accuracy(model, tokenizer, subset)
                accuracies.append(acc)
            else:
                accuracies.append(0)

        ax.plot(unique_lengths, accuracies, marker='o', label='Gemma 2B Tuned')
        ax.set_title(algo.replace('_', ' ').title())
        ax.set_xlabel('Test length')
        ax.set_ylabel('Accuracy')
        ax.grid(True)
        ax.set_ylim(-0.05, 1.05)
    
    # Hide unused subplots
    for j in range(len(validation_datasets), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig('accuracy_ood.png')
    print("\nSaved OOD accuracy plot to accuracy_ood.png")


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
    model = AutoModelForCausalLM.from_pretrained('google/gemma-2b', device_map='auto', torch_dtype=torch.bfloat16)

    # Add a padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_algos = [s for s, o in inspect.getmembers(algorithms) if inspect.isfunction(o)]
    all_algos = ['bfs', 'dfs']

    tune_sets, val_sets, valid_algos_list = [], [], []
    for algo in tqdm(all_algos, desc="Generating Datasets"):
        split = get_train_val_splits(algo)
        if split is None:
            print(f"[SKIP] algorithm {algo}")
            continue
        tune_sets.append(split['train'])
        val_sets.append(split['val'])
        valid_algos_list.append(algo) #plot_ood_acc wants this format

    tune_sets = concatenate_datasets(tune_sets).shuffle(seed=1337)

    print(f"\nFine-tuning on {len(tune_sets)} examples from {len(valid_algos_list)} algorithms...")
    model_tuned = fine_tune_model(model, tokenizer, tune_sets)

    print("\nStarting evaluation for OOD plot...")
    plot_ood_accuracy(valid_algos_list, val_sets, model_tuned, tokenizer)