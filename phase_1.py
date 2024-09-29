import os
import random
import logging
import math
import time
import subprocess
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW  # Use PyTorch's AdamW optimizer
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
    PreTrainedTokenizerFast,
)
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set")
warnings.filterwarnings("ignore", category=FutureWarning)

def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
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

def setup_logging(log_dir):
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

    # Resize model embeddings to match tokenizer
    model.resize_token_embeddings(len(tokenizer))
    logging.info("Resized model embeddings to match tokenizer")

    if checkpoint_path and os.path.exists(checkpoint_path):
        model_state = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(model_state['model_state_dict'])
        logging.info(f"Loaded model from checkpoint: {checkpoint_path}")
    else:
        logging.info("Initialized model from scratch")

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

def save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    filename = f'phase_1_epoch_{epoch}.pt'
    if is_best:
        filename = f'phase_1_best.pt'
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    logging.info(f"Saved checkpoint: {checkpoint_path}")

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    return model, optimizer, scheduler, start_epoch

def setup_tensorboard(log_dir):
    writer = SummaryWriter(log_dir)
    return writer

def log_metrics(writer, epoch, training_loss, validation_loss, train_perplexity, val_perplexity, tokens_processed_train, tokens_processed_val, epoch_time):
    writer.add_scalar('Loss/Training', training_loss, epoch)
    writer.add_scalar('Loss/Validation', validation_loss, epoch)
    writer.add_scalar('Perplexity/Training', train_perplexity, epoch)
    writer.add_scalar('Perplexity/Validation', val_perplexity, epoch)
    writer.add_scalar('Tokens_Processed/Training', tokens_processed_train, epoch)
    writer.add_scalar('Tokens_Processed/Validation', tokens_processed_val, epoch)
    writer.add_scalar('Time/Epoch', epoch_time, epoch)

from torch.cuda.amp import autocast, GradScaler

def train(model, dataloader, optimizer, scheduler, device, accumulation_steps):
    model.train()
    total_loss = 0
    tokens_processed = 0
    scaler = GradScaler()
    optimizer.zero_grad()
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    for step, batch in progress_bar:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        with autocast():
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        batch_loss = loss.item() * accumulation_steps
        total_loss += batch_loss * inputs.size(0)
        tokens_processed += inputs.numel()

        progress_bar.set_postfix({'loss': f'{batch_loss:.4f}'})

    avg_loss = total_loss / len(dataloader.dataset)
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, perplexity, tokens_processed

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    tokens_processed = 0
    progress_bar = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for batch in progress_bar:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss

            batch_loss = loss.item()
            total_loss += batch_loss * inputs.size(0)
            tokens_processed += inputs.numel()

            progress_bar.set_postfix({'loss': f'{batch_loss:.4f}'})

    avg_loss = total_loss / len(dataloader.dataset)
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, perplexity, tokens_processed

class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Extract input_tokens and target_token from each feature
        input_texts = [feature['original'] for feature in features]
        # Combine input and target for tokenization
        combined_texts = [text + feature['target_token'] for text, feature in zip(input_texts, features)]

        # Tokenize and encode inputs and labels together
        encoding = self.tokenizer(
            combined_texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids']
        labels = input_ids.clone()

        # Shift labels to align with inputs
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # We can ignore the last token

        return {'input_ids': input_ids, 'labels': labels}

def main():
    # Set parameters directly in the code
    epochs = 10
    batch_size = 8  # Reduced batch size to avoid memory errors
    learning_rate = 5e-4
    checkpoint_dir = 'checkpoints/'
    log_dir = 'logs/'
    tokenizer_path = 'tokenizer/custom_bpe_tokenizer.json'
    dataset_name = 'Vipplav/phase_1_3M'
    resume_from_checkpoint = False  # Set to True if you want to resume training
    accumulation_steps = 4  # Increased accumulation steps

    device = set_device()
    set_seed()

    # Create necessary directories
    create_directories([checkpoint_dir, log_dir])
    setup_logging(log_dir)
    writer = setup_tensorboard(log_dir)

    logging.info("Starting Phase 1 Training")
    logging.info(f"Parameters: Epochs={epochs}, Batch Size={batch_size}, Learning Rate={learning_rate}")
    logging.info(f"Accumulation Steps: {accumulation_steps}")

    # Check and log GPU processes
    check_and_log_gpu_processes()

    # Clear CUDA memory cache
    clear_cuda_memory()

    # Load tokenizer and dataset
    tokenizer_path = os.path.abspath(tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    train_dataset, val_dataset = load_dataset_phase1(dataset_name)

    # Initialize the custom data collator
    data_collator = CustomDataCollator(tokenizer)

    # Prepare data collator and dataloaders
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=data_collator
    )

    logging.info(f"DataLoaders created. Training batches: {len(train_dataloader)}, Validation batches: {len(val_dataloader)}")

    # Initialize model
    checkpoint_path = None
    if resume_from_checkpoint:
        checkpoint_path = os.path.join(checkpoint_dir, 'phase_1_last.pt')
    model = initialize_model(tokenizer, checkpoint_path=checkpoint_path)
    model.to(device)

    # Set up optimizer and scheduler
    num_training_steps = epochs * len(train_dataloader) // accumulation_steps
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    optimizer = setup_optimizer(model, learning_rate)
    scheduler = setup_scheduler(optimizer, num_warmup_steps, num_training_steps)

    # Resume from checkpoint if needed
    start_epoch = 1
    if resume_from_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
        model, optimizer, scheduler, start_epoch = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
        logging.info(f"Resumed training from checkpoint {checkpoint_path}")
    else:
        logging.info("Starting training from scratch")

    best_val_loss = float('inf')
    for epoch in range(start_epoch, epochs + 1):
        logging.info(f"Starting epoch {epoch}/{epochs} for Phase 1")
        epoch_start_time = time.time()

        train_loss, train_perplexity, tokens_processed_train = train(
            model, train_dataloader, optimizer, scheduler, device, accumulation_steps
        )
        val_loss, val_perplexity, tokens_processed_val = evaluate(model, val_dataloader, device)

        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        logging.info(f"Epoch {epoch}: Train Perplexity = {train_perplexity:.4f}, Val Perplexity = {val_perplexity:.4f}")
        logging.info(f"Epoch {epoch}: Tokens Processed (Train) = {tokens_processed_train}, Tokens Processed (Val) = {tokens_processed_val}")

        # Log metrics
        log_metrics(
            writer, epoch, train_loss, val_loss, train_perplexity, val_perplexity,
            tokens_processed_train, tokens_processed_val, epoch_time
        )

        # Save the best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir, is_best=is_best)

    logging.info("Training completed.")

if __name__ == '__main__':
    main()
