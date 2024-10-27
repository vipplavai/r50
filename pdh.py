import os
import sys
import random
import logging
import math
import time
import subprocess
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from tqdm import tqdm

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step()` before `optimizer.step()`")

# Set NCCL environment variables based on your network capabilities
os.environ['NCCL_SOCKET_IFNAME'] = 'enp0s31f6'  # Replace with your network interface
os.environ['NCCL_BUFFSIZE'] = '1048576'  # 1MB buffer size
os.environ['NCCL_MIN_NRINGS'] = '4'  # Increase number of rings for parallelism

# Set environment variables for NCCL debugging if needed
# os.environ['NCCL_DEBUG'] = 'INFO'
# os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

def get_gpu_metrics():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total,power.draw', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        lines = result.strip().split('\n')
        total_memory = 0
        total_power = 0.0
        for line in lines:
            memory_str, power_str = line.split(',')
            total_memory += int(memory_str.strip())
            total_power += float(power_str.strip())
        return total_memory, total_power
    except Exception as e:
        logging.error(f"Error getting GPU metrics: {e}")
        return 0, 0.0

def setup(rank, world_size):
    try:
        # Get MASTER_ADDR and MASTER_PORT from environment variables set by torchrun
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']

        # Log the master address and port
        logging.info(f"[Rank {rank}] MASTER_ADDR: {master_addr}, MASTER_PORT: {master_port}")

        # Log NCCL_SOCKET_IFNAME if set
        nccl_socket_ifname = os.environ.get('NCCL_SOCKET_IFNAME')
        logging.info(f"[Rank {rank}] NCCL_SOCKET_IFNAME: {nccl_socket_ifname}")

        # Initialize the process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )
        torch.cuda.set_device(0)  # All nodes have a single GPU at index 0
        logging.info(f"[Rank {rank}] Distributed environment initialized with world size {world_size}.")

        # Barrier to synchronize processes
        dist.barrier()
    except Exception as e:
        logging.error(f"[Rank {rank}] Error initializing distributed process group: {e}")
        sys.exit(1)

def cleanup():
    dist.destroy_process_group()
    logging.info("Distributed environment cleaned up.")

def set_device():
    device = torch.device('cuda:0')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    return device

def clear_cuda_memory():
    torch.cuda.empty_cache()
    logging.info("Cleared CUDA memory cache.")

def check_and_log_gpu_processes():
    logging.info("Checking for processes running on the CUDA device...")
    try:
        result = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
        logging.info("nvidia-smi output:")
        logging.info(result)
    except Exception as e:
        logging.error(f"Error running nvidia-smi: {e}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setup_logging(log_dir, rank):
    if rank == 0:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logging.basicConfig(
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ],
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
    else:
        logging.basicConfig(
            handlers=[logging.NullHandler()],
            level=logging.ERROR
        )

def create_directories(dirs):
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def load_tokenizer(tokenizer_path):
    if os.path.isfile(tokenizer_path):
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        logging.info(f"Loaded tokenizer from file: {tokenizer_path}")
    else:
        raise ValueError(f"Tokenizer file {tokenizer_path} does not exist.")

    # Add pad_token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        logging.info("Added pad_token to tokenizer")
    return tokenizer

def load_dataset_phase1(dataset_name):
    logging.info(f"Loading dataset for Phase 1: {dataset_name}")

    # Load dataset from Hugging Face Hub
    raw_datasets = load_dataset(dataset_name)
    logging.info("Dataset loaded successfully.")

    # Get train and validation datasets
    train_dataset = raw_datasets['train']
    val_dataset = raw_datasets['validation']

    return train_dataset, val_dataset

def initialize_model(tokenizer, device=None):
    model_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=512,
        n_embd=528,
        n_layer=7,
        n_head=8,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel(model_config)

    # Resize model embeddings to match tokenizer
    model.resize_token_embeddings(len(tokenizer))
    logging.info("Resized model embeddings to match tokenizer")

    logging.info("Initialized model")

    model.to(device)
    # No need to wrap with DDP; Trainer will handle it
    return model

class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Extract input_tokens and target_token from each feature
        input_texts = [feature['original'] for feature in features]
        target_texts = [feature['target_token'] for feature in features]
        # Combine input and target for tokenization
        combined_texts = [text + tgt for text, tgt in zip(input_texts, target_texts)]

        # Tokenize and encode inputs and labels together
        encoding = self.tokenizer(
            combined_texts,
            padding=True,
            truncation=True,
            max_length=512,  # Specify max_length to avoid warnings
            return_tensors='pt'
        )

        input_ids = encoding['input_ids']
        labels = input_ids.clone()

        # Shift labels to align with inputs
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # We can ignore the last token

        return {'input_ids': input_ids, 'labels': labels}

def main():
    # Get rank and world size from environment variables
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # Set up logging before calling setup()
    setup_logging('logs/', rank)

    # Initialize process group
    setup(rank, world_size)

    device = set_device()

    # Get GPU metrics on this node
    total_memory, total_power = get_gpu_metrics()

    # Gather metrics from all nodes
    metrics = (total_memory, total_power)
    gathered_metrics = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_metrics, metrics)

    if rank == 0:
        total_memory_all_nodes = sum(mem for mem, power in gathered_metrics)
        total_power_all_nodes = sum(power for mem, power in gathered_metrics)
        logging.info(f"Total GPU Memory across all nodes: {total_memory_all_nodes} MB")
        logging.info(f"Total GPU Power Draw across all nodes: {total_power_all_nodes} W")

    # Set parameters directly in the code
    script_dir = os.path.dirname(os.path.abspath(__file__))
    epochs = 1
    batch_size = 8
    learning_rate = 5e-4
    checkpoint_dir = 'checkpoints/'
    log_dir = 'logs/'
    tokenizer_path = os.path.join(script_dir, 'tokenizer', 'custom_bpe_tokenizer.json')
    dataset_name = 'Vipplav/phase_1_3M'
    resume_from_checkpoint = True  # Set to True to resume training
    accumulation_steps = 4

    set_seed()

    # Create necessary directories
    if rank == 0:
        create_directories([checkpoint_dir, log_dir])

    # Initialize W&B logging only on Rank 0
    if rank == 0:
        import wandb
        wandb.init(project="ddp_training_project", mode="online")

    # Barrier to ensure all processes have finished setup
    dist.barrier()

    # Load tokenizer and dataset
    tokenizer_path = os.path.abspath(tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    train_dataset, val_dataset = load_dataset_phase1(dataset_name)

    # Prepare data collator
    data_collator = CustomDataCollator(tokenizer)

    # Initialize model
    model = initialize_model(tokenizer, device=device)

    # Adjust TrainingArguments
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=500,  # Adjust as needed
        save_strategy="steps",
        save_steps=500,  # Must match eval_steps
        logging_dir=log_dir,
        logging_steps=100,
        report_to="wandb",
        ddp_find_unused_parameters=False,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        warmup_steps=500,  # Fixed warmup steps
        optim="adamw_torch",
        lr_scheduler_type="linear",
        save_total_limit=1,  # Save only the best checkpoint
    )

    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Start training
    if resume_from_checkpoint:
        last_checkpoint = None
        if os.path.isdir(checkpoint_dir):
            checkpoints = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
            if checkpoints:
                # Find the latest checkpoint
                last_checkpoint = max(checkpoints, key=os.path.getctime)
        if last_checkpoint is not None:
            logging.info(f"Resuming training from checkpoint: {last_checkpoint}")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            logging.info("No checkpoint found. Starting training from scratch.")
            trainer.train()
    else:
        trainer.train()

    # Final evaluation
    eval_results = trainer.evaluate()
    if rank == 0:
        wandb.log(eval_results)
        logging.info(f"Final Evaluation Results: {eval_results}")

    # Save final model
    if rank == 0:
        trainer.save_model("./final_model")
        logging.info("Model saved to ./final_model")

    cleanup()

if __name__ == '__main__':
    main()
