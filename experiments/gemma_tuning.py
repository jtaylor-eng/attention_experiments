import inspect
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset, concatenate_datasets
from typing import Tuple

import clrs
from clrs._src import algorithms
from clrs._src.clrs_text import huggingface_generators

import absl.logging ; absl.logging.set_verbosity(absl.logging.ERROR) #dataset kwargs implicitely passing 'use padding' -- gives phat error msg

#MONKEY PATCHING CLRS / HF Dataset interface problem
#Gemini:
# The clrs library has a bug where its text formatting utility fails on
# algorithms that use scalar inputs. This patch fixes the issue by:
# 1. Matching the original function's signature to accept all arguments.
# 2. Using a try-except block to handle both scalar and iterable data.
import clrs._src.clrs_text.clrs_utils as clrs_utils
def monkey(x, **kwargs):
    try: content = clrs_utils.SEQUENCE_SEPARATOR.join([str(a) for a in x])
    except TypeError: content = str(x)

    use_brackets = kwargs.get('brackets', True)
    return clrs_utils._bracket(content) if use_brackets else content
clrs_utils._convert_node_features_to_str = monkey
#


def test_gen(model, tokenizer):
    """Ensure functionality to generate from model w/ tokenizer."""
    input_text = 'Michael Jordan plays for the '
    input_ids = tokenizer(input_text, return_tensors='pt').to('cuda')
    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))


def get_train_val_splits(
    algorithm: str,
    train_range: Tuple[int,int] = (1,16),
    val_range: Tuple[int,int] = (1,32),
    train_samples: int = 500,
    val_samples: int = 50
):
    """
    given CLRS algorithm (dfs, bfs, dijkstra, ...) generate train_samples trajectories and store in train split,
    val_samples and store in val split.

    train_range, val_range is sequence length for trajectories in the respective sets. To test OOD generations,
    the upper bound of the val range should exceed the UB of train range.

    Returns: HF Dataset object of the form:
        {'train': Dataset({
            features: ['text', 'question', 'answer', 'algo_name', 'length', 'use_hints'],
            num_rows: train_samples
        }), 'val': Dataset({
            features: ['text', 'question', 'answer', 'algo_name', 'length', 'use_hints'],
            num_rows: val_samples
        })}
    """
    assert train_range[0] >= 1 and train_range[1] <= 64, 'Train range must be within (1, 64)'
    assert val_range[0] >= 1 and val_range[1] <= 128, 'Validation range must be within (1, 128)'
    assert val_range[1] > train_range[1], 'Validation upper bound should be larger than training upper bound to test OOD sequences'
    
    train_lens = list(range(train_range[0], train_range[1]))
    val_lens = list(range(val_range[0], val_range[1]))

    it = [
        ('train', train_lens),
        ('val', val_lens)
    ]

    splits = {}

    for split, lens in it:
        samples = train_samples if split == 'train' else val_samples
        splits[split] = Dataset.from_generator(
            huggingface_generators.clrs_generator,
            gen_kwargs={
                'algos_and_lengths': {algorithm: lens},
                'num_samples': samples,
                'use_hints': False,
                'seed': 1337
            }
        )

    # print(splits, splits['train'][0], splits['val']) ; exit() #look at data gen
    return splits


def preprocess_dataset(dataset, tokenizer):
    def format_example(example):
        text = f"{example['text']}\n{example['question']}\nAnswer: {example['answer']}"
        return {'text': text}
    
    dataset = dataset.map(format_example)
    
    def tokenize(batch):
        return tokenizer(
            batch['text'],
            padding='max_length',
            truncation=True,
            max_length=512,
        )
    
    return dataset.map(tokenize, batched=True)


def fine_tune_model(model, tokenizer, dataset, output_dir='./checkpoints', epochs=1):
    tokenized = preprocess_dataset(dataset, tokenizer)
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=epochs,
        logging_steps=5,
        save_strategy='no',
        report_to='none',
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    return model


def evaluate_perplexity(model, tokenizer, dataset):
    model.eval()
    losses = []
    for example in tqdm(dataset):
        inputs = tokenizer(example['text'], return_tensors='pt', truncation=True, max_length=512).to(model.device)
        with torch.no_grad(): loss = model(**inputs, labels=inputs['input_ids']).loss
        losses.append(loss.item())
    return np.exp(np.mean(losses))


def plot_validation_OOD_performance(validation_datasets, model):
    lens, ppls = [], []
    
    for val_set in val_sets:
        avg_len = np.mean([ex['length'] for ex in val_set])
        ppl = evaluate_perplexity(model, tokenizer, val_set)
        lens.append(avg_len)
        ppls.append(ppl)
    
    plt.plot(lens, ppls) ; plt.grid(True) ; plt.show()


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
    model = AutoModelForCausalLM.from_pretrained('google/gemma-2b', device_map='auto')

    # test_gen(model, tokenizer)

    print(type(model))
    print(model)

    #probably a convenience method for this, but couldn't find
    #list of strings for generating the 30 algorithm datasets
    algos = [s for s, o in inspect.getmembers(algorithms) if inspect.isfunction(o)]

    tune_sets, val_sets = [], []
    for algo in algos:
        split = get_train_val_splits(algo)
        tune_sets.append(split['train'])
        val_sets.append(split['val']) #store for future comparison

    tune_sets = concatenate_datasets(tune_sets).shuffle(seed=1337)

    model_tuned = fine_tune_model(model, tokenizer, tune_sets)

    plot_validation_OOD_performance(val_sets, model_tuned) 