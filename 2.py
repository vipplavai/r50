import os
import argparse
import random
import logging
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    AdamW,
    get_linear_schedule_with_warmup,
    PreTrainedTokenizerFast,
)
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu
from datasets.utils.filelock import FileLock


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Telugu Language Model with DDP")
    parser.add_argument('--phase', type=int, required=True, help='Training phase (1-4)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs per phase')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory to save logs')
    parser.add_argument('--tokenizer_path', type=str, default='tokenizer/custom_bpe_tokenizer.json', help='Path to the tokenizer file')
    parser.add_argument('--resume_from_checkpoint', action='store_true', help='Resume training from the last checkpoint')
    args = parser.parse_args()
    return args

def clear_cuda_memory():
    logging.info("Checking for existing CUDA processes...")
    gpu_id = 0  # Since all machines have only one GPU
    # Use nvidia-smi to check for processes
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
        output = result.stdout.decode()
        if f'No running processes found' not in output:
            logging.info("Existing CUDA processes found. Clearing CUDA memory...")
            torch.cuda.empty_cache()
            # Additional cleanup if necessary
        else:
            logging.info("No existing CUDA processes found.")
    except Exception as e:
        logging.warning(f"Failed to check CUDA processes: {e}")

def setup_distributed():
    print("Before dist.init_process_group()")
    dist.init_process_group(backend='nccl')
    print("After dist.init_process_group()")
    logging.info("Initialized distributed training")


def cleanup():
    dist.destroy_process_group()

def set_device(local_rank):
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    print(f"Using GPU: {torch.cuda.get_device_name(local_rank)} on local_rank {local_rank}")
    return device

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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
            handlers=[logging.StreamHandler()],
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
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

def load_datasets(phase, tokenizer, batch_size, rank, world_size):
    # Define dataset names based on phase
    dataset_map = {
        1: 'Vipplav/phase_1_3M',
        2: 'Vipplav/phase_2_3M',
        3: 'Vipplav/phase_3_1M',
        4: 'Vipplav/phase_4_50k'
    }
    dataset_name = dataset_map[phase]

    if rank == 0:
        logging.info(f"Loading dataset for Phase {phase}: {dataset_name}")

    # Ensure only one process downloads the dataset
    with FileLock(".lock"):
        raw_datasets = load_dataset(dataset_name)
    
    dist.barrier()
    
    if rank == 0:
        logging.info("Dataset loaded successfully.")

    # Define data collator and processing based on phase
    # (Assuming data collator is correctly defined)
    data_collator = DataCollatorWithPadding(tokenizer, return_tensors='pt')

    train_dataset = raw_datasets['train']
    val_dataset = raw_datasets['validation']

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=data_collator)

    if rank == 0:
        logging.info(f"DataLoaders created. Training batches: {len(train_dataloader)}, Validation batches: {len(val_dataloader)}")

    return train_dataloader, val_dataloader

def initialize_model(tokenizer, checkpoint_path=None):
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

    if checkpoint_path and os.path.exists(checkpoint_path):
        model_state = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(model_state['model_state_dict'])
        logging.info(f"Loaded model from checkpoint: {checkpoint_path}")
    else:
        logging.info("Initialized model from scratch")

    # Resize model embeddings to match tokenizer
    model.resize_token_embeddings(len(tokenizer))
    logging.info("Resized model embeddings to match tokenizer")

    return model

def setup_optimizer(model, learning_rate):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    return optimizer

def setup_scheduler(optimizer, num_warmup_steps, num_training_steps):
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    return scheduler

def save_checkpoint(model, optimizer, scheduler, epoch, phase, checkpoint_dir, rank, is_best=False):
    if rank == 0:
        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': model.module.state_dict(),  # Use model.module for DDP
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        filename = f'phase_{phase}_epoch_{epoch}.pt'
        if is_best:
            filename = f'phase_{phase}_best.pt'
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint: {checkpoint_path}")
    dist.barrier()

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    phase = checkpoint['phase']
    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    return model, optimizer, scheduler, epoch

def setup_tensorboard(log_dir, rank):
    if rank == 0:
        writer = SummaryWriter(log_dir)
    else:
        writer = None
    return writer

def log_metrics(writer, phase, epoch, training_loss, validation_loss, tokens_processed, rank, perplexity=None, bleu_score=None):
    if rank == 0 and writer is not None:
        writer.add_scalar(f'Phase_{phase}/Training_Loss', training_loss, epoch)
        writer.add_scalar(f'Phase_{phase}/Validation_Loss', validation_loss, epoch)
        writer.add_scalar(f'Phase_{phase}/Tokens_Processed', tokens_processed, epoch)
        if perplexity is not None:
            writer.add_scalar(f'Phase_{phase}/Perplexity', perplexity, epoch)
        if bleu_score is not None:
            writer.add_scalar(f'Phase_{phase}/BLEU_Score', bleu_score, epoch)

def train(model, dataloader, optimizer, scheduler, device, epoch, rank):
    model.train()
    total_loss = 0
    tokens_processed = 0
    dataloader.sampler.set_epoch(epoch)
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch} Rank {rank}", disable=rank != 0)
    for batch in progress_bar:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        batch_loss = loss.item()
        total_loss += batch_loss * inputs.size(0)  # Multiply by batch size
        tokens_processed += inputs.numel()

        if rank == 0:
            progress_bar.set_postfix({'loss': f'{batch_loss:.4f}'})

    # Compute average loss across all processes
    total_loss_tensor = torch.tensor(total_loss, device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    total_loss = total_loss_tensor.item() / dist.get_world_size()

    dataset_size_tensor = torch.tensor(len(dataloader.dataset), device=device)
    dist.all_reduce(dataset_size_tensor, op=dist.ReduceOp.SUM)
    dataset_size = dataset_size_tensor.item()

    avg_loss = total_loss / dataset_size
    return avg_loss, tokens_processed

def evaluate(model, dataloader, device, rank):
    model.eval()
    total_loss = 0
    tokens_processed = 0
    progress_bar = tqdm(dataloader, desc=f"Evaluating Rank {rank}", disable=rank != 0)
    with torch.no_grad():
        for batch in progress_bar:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            batch_loss = loss.item()
            total_loss += batch_loss * inputs.size(0)  # Multiply by batch size
            tokens_processed += inputs.numel()

            if rank == 0:
                progress_bar.set_postfix({'loss': f'{batch_loss:.4f}'})

    # Compute average loss across all processes
    total_loss_tensor = torch.tensor(total_loss, device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    total_loss = total_loss_tensor.item() / dist.get_world_size()

    dataset_size_tensor = torch.tensor(len(dataloader.dataset), device=device)
    dist.all_reduce(dataset_size_tensor, op=dist.ReduceOp.SUM)
    dataset_size = dataset_size_tensor.item()

    avg_loss = total_loss / dataset_size
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, perplexity, tokens_processed

def calculate_bleu(model, dataloader, tokenizer, device, rank):
    model.eval()
    bleu_scores = []
    progress_bar = tqdm(dataloader, desc=f"Calculating BLEU Rank {rank}", disable=rank != 0)
    with torch.no_grad():
        for batch in progress_bar:
            inputs = batch['input_ids'].to(device)
            outputs = model.module.generate(inputs, max_length=50)
            for i in range(len(outputs)):
                reference = tokenizer.decode(batch['labels'][i], skip_special_tokens=True)
                hypothesis = tokenizer.decode(outputs[i], skip_special_tokens=True)
                bleu_score = sentence_bleu([reference.split()], hypothesis.split())
                bleu_scores.append(bleu_score)
    # Gather BLEU scores from all processes
    bleu_scores_tensor = torch.tensor(bleu_scores, device=device)
    dist.barrier()
    dist.all_reduce(bleu_scores_tensor, op=dist.ReduceOp.SUM)
    avg_bleu = bleu_scores_tensor.sum().item() / len(bleu_scores_tensor)
    return avg_bleu

def load_model_for_phase(phase, tokenizer, checkpoint_dir, resume_from_checkpoint, rank):
    if resume_from_checkpoint:
        checkpoint_path = os.path.join(checkpoint_dir, f'phase_{phase}_last.pt')
        if os.path.exists(checkpoint_path):
            model = initialize_model(tokenizer, checkpoint_path=checkpoint_path)
            if rank == 0:
                logging.info(f"Resuming training from checkpoint: {checkpoint_path}")
        else:
            if rank == 0:
                logging.warning(f"No checkpoint found at {checkpoint_path}, initializing model from scratch.")
            model = initialize_model(tokenizer)
    elif phase > 1:
        prev_phase = phase - 1
        checkpoint_path = os.path.join(checkpoint_dir, f'phase_{prev_phase}_best.pt')
        if os.path.exists(checkpoint_path):
            model = initialize_model(tokenizer, checkpoint_path=checkpoint_path)
            if rank == 0:
                logging.info(f"Loaded model from Phase {prev_phase} checkpoint: {checkpoint_path}")
        else:
            if rank == 0:
                logging.warning(f"No checkpoint found for Phase {prev_phase}, initializing model from scratch.")
            model = initialize_model(tokenizer)
    else:
        model = initialize_model(tokenizer)
    return model

def generate_text(model, tokenizer, prompts, device, max_length=50, rank=0):
    if rank != 0:
        return
    model.eval()
    generated_texts = []
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            output_ids = model.module.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
    return generated_texts

def main_worker(args):
    try:
        print("Starting main_worker")
        # Get rank and local_rank from environment variables
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"Rank: {rank}, Local rank: {local_rank}, World size: {world_size}")

        # Set up logging immediately
        setup_logging(args.log_dir, rank)
        logging.info("Logging is set up.")
        clear_cuda_memory()
        setup_distributed()
        set_seed()

        device = set_device(local_rank)

        # Adjust learning rates based on phase if not provided
        default_learning_rates = {1: 5e-4, 2: 3e-4, 3: 2e-4, 4: 1e-4}
        if not args.learning_rate:
            args.learning_rate = default_learning_rates.get(args.phase, 5e-4)

        # Create necessary directories
        if rank == 0:
            create_directories([args.checkpoint_dir, args.log_dir, 'inference_outputs/'])
        writer = setup_tensorboard(args.log_dir, rank)

        if rank == 0:
            logging.info(f"Starting Phase {args.phase} Training")
            logging.info(f"Parameters: Epochs={args.epochs}, Batch Size={args.batch_size}, Learning Rate={args.learning_rate}")

        # Load tokenizer and datasets
        tokenizer_path = os.path.abspath(args.tokenizer_path)
        tokenizer = load_tokenizer(tokenizer_path)
        train_dataloader, val_dataloader = load_datasets(args.phase, tokenizer, args.batch_size, rank, world_size)

        # Initialize model
        model = load_model_for_phase(args.phase, tokenizer, args.checkpoint_dir, args.resume_from_checkpoint, rank)
        model.to(device)
        model = DDP(model, device_ids=[local_rank])

        # Set up optimizer and scheduler
        num_training_steps = args.epochs * len(train_dataloader)
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
        optimizer = setup_optimizer(model, args.learning_rate)
        scheduler = setup_scheduler(optimizer, num_warmup_steps, num_training_steps)

        # Resume from checkpoint if needed
        start_epoch = 1
        if args.resume_from_checkpoint:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'phase_{args.phase}_last.pt')
            if os.path.exists(checkpoint_path):
                model, optimizer, scheduler, start_epoch = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
                if rank == 0:
                    logging.info(f"Resumed training from checkpoint {checkpoint_path}")
            else:
                if rank == 0:
                    logging.warning(f"No checkpoint found at {checkpoint_path}, starting from scratch")

        best_val_loss = float('inf')
        for epoch in range(start_epoch, args.epochs + 1):
            if rank == 0:
                logging.info(f"Starting epoch {epoch}/{args.epochs} for Phase {args.phase}")
            epoch_start_time = time.time()

            train_loss, tokens_processed_train = train(model, train_dataloader, optimizer, scheduler, device, epoch, rank)
            val_loss, perplexity, tokens_processed_val = evaluate(model, val_dataloader, device, rank)

            epoch_time = time.time() - epoch_start_time
            if rank == 0:
                logging.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
                logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Perplexity = {perplexity:.4f}")

            if args.phase >= 3:
                bleu_score = calculate_bleu(model, val_dataloader, tokenizer, device, rank)
                if rank == 0:
                    logging.info(f"Epoch {epoch}: BLEU Score = {bleu_score:.4f}")
                    log_metrics(writer, args.phase, epoch, train_loss, val_loss, tokens_processed_train, rank, bleu_score=bleu_score)
            else:
                if rank == 0:
                    log_metrics(writer, args.phase, epoch, train_loss, val_loss, tokens_processed_train, rank, perplexity=perplexity)

            # Save the best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, args.phase, args.checkpoint_dir, rank, is_best=is_best)

        # Generate outputs after training
        if rank == 0:
            prompts = ["<s> మీరు", "<s> ఒకసారి", "<s> చాలా సంతోషం"]
            generated_texts = generate_text(model, tokenizer, prompts, device, rank=rank)
            for prompt, gen_text in zip(prompts, generated_texts):
                print(f"Prompt: {prompt}")
                print(f"Generated: {gen_text}")
                print("-" * 20)
                # Optionally save to file
                output_file = os.path.join('inference_outputs', 'generated_texts.txt')
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Generated: {gen_text}\n")
                    f.write('-' * 20 + '\n')

        cleanup()
    except Exception as e:
        print(f"Exception in main_worker: {e}")
        import traceback
        traceback.print_exc()

def main():
    args = parse_args()
    main_worker(args)

if __name__ == '__main__':
    main()
