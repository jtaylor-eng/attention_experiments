#Modified from karpathy fineweb dataset and gpt2 tokenizer, to use C4 english dataset and Llama2 tokenizer
import os
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import LlamaTokenizer

#set to true to test download on small amount of data, runs <5s with decent num cpu cores
TEST_MODE = True

if TEST_MODE:
    TARGET_TOKENS = int(1e7) 
    SHARD_SIZE = int(1e5)
    OUTPUT_DIR = './test_download'
else:
    TARGET_TOKENS = int(10e9) 
    SHARD_SIZE = int(1e8)
    OUTPUT_DIR = './c4_en_llama2_tokens'

TOKENIZER_ID = 'meta-llama/Llama-2-7b-hf'
NPROCS = max(1, os.cpu_count() // 2)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f'Loading tokenizer: {TOKENIZER_ID}')
tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_ID)
eos_id = tokenizer.eos_token_id
bos_id = tokenizer.bos_token_id

def tokenize(doc): return np.array([bos_id] + tokenizer.encode(doc['text']) + [eos_id], dtype=np.uint16)

def main():
    # Load the C4 dataset in streaming mode to avoid full download
    fw = load_dataset('allenai/c4', name='en', split='train', streaming=True)

    with mp.Pool(NPROCS) as pool:
        shard_index = 0
        all_tokens_np = np.empty((SHARD_SIZE,), dtype=np.uint16)
        token_count = 0
        total_tokens_processed = 0
        progress_bar = None

        print(f'Starting tokenization with {NPROCS} processes')
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            if progress_bar is None: progress_bar = tqdm(total=SHARD_SIZE, unit='tokens', desc=f'Shard {shard_index}')
            #check if there's enough space in the current shard
            if token_count + len(tokens) < SHARD_SIZE:
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                total_tokens_processed += len(tokens)
                progress_bar.update(len(tokens))

            else:
                #the shard is full,write to file
                remainder = SHARD_SIZE - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
                
                split = 'val' if shard_index == 0 else 'train'
                filename = os.path.join(OUTPUT_DIR, f'c4_en_{split}_{shard_index:06d}.npy')
                np.save(filename, all_tokens_np)
                
                shard_index += 1
                progress_bar.close()
                progress_bar = None
                
                #start next shard with remaining tokens
                all_tokens_np[0 : len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder
                total_tokens_processed += len(tokens)

            #stop cond 10BTs
            if total_tokens_processed >= TARGET_TOKENS:
                print(f'target of {TARGET_TOKENS} tokens reached.')
                break
        
        #write remaining tokens as last shard
        if token_count > 0:
            split = 'val' if shard_index == 0 else 'train'
            filename = os.path.join(OUTPUT_DIR, f'c4_en_{split}_{shard_index:06d}.npy')
            np.save(filename, all_tokens_np[:token_count])

if __name__ == '__main__': main()