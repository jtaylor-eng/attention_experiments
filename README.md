# Project Description:  
 - Seeks to answer: can language models get better performance with alternative functions to softmax.
 - Particularly: reconstruct and expand on experiments done in the Softmax is Not Enough paper. (arXiv:2410.01104)

# Repository Structure:
 - checkpoints: Model checkpoints, automatically written every k steps, and once training completes.
 - data: Code to download dataset (c4_en with llama2 tokenizer). Can easily switch dataset and tokenizer to any HF dataset / tokenizer.
 - experiments: Code to compare quality of pretrained models. Namely perplexity comparison using CLRS benchmark between 2 checkpoints.
 - training: Pretraining code: LM architecture, train loop, dataloaders, lr scheduling, etc.

# TODO:
 - [x] seperate files / better repo structure (train loop, architecture, utils,  experiments, logging)
 - [x] fix logging pathing (sbatch)
 - [ ] fix DDP
 - [ ] consider fp precision for fast train / inference

