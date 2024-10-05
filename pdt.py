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
from torch.optim import AdamW
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    get_linear_schedule_with_warmup,
    PreTrainedTokenizerFast,
)
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", message="`clean_up_tokenization_spaces` was not set")
warnings.filterwarnings("ignore", category=FutureWarning)

# Set environment variables for NCCL debugging
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


def initialize_model(tokenizer, checkpoint_path=None, device=None):
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

    model.to(device)
    model = DDP(model, device_ids=[0], output_device=0)
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
    if dist.get_rank() == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        filename = f'phase_1_epoch_{epoch}.pt'
        if is_best:
            filename = f'phase_1_best.pt'
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Saved checkpoint: {checkpoint_path}")

    # Synchronize after saving
    dist.barrier()


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.module.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    return model, optimizer, scheduler, start_epoch


def setup_tensorboard(log_dir):
    writer = SummaryWriter(log_dir)
    return writer


def log_metrics(writer, epoch, training_loss, validation_loss, train_perplexity, val_perplexity,
                tokens_processed_train, tokens_processed_val, epoch_time, avg_tokens_per_node, comm_time):
    writer.add_scalar('Loss/Training', training_loss, epoch)
    writer.add_scalar('Loss/Validation', validation_loss, epoch)
    writer.add_scalar('Perplexity/Training', train_perplexity, epoch)
    writer.add_scalar('Perplexity/Validation', val_perplexity, epoch)
    writer.add_scalar('Tokens_Processed/Training', tokens_processed_train, epoch)
    writer.add_scalar('Tokens_Processed/Validation', tokens_processed_val, epoch)
    writer.add_scalar('Tokens_Processed/Avg_per_Node', avg_tokens_per_node, epoch)
    writer.add_scalar('Time/Epoch', epoch_time, epoch)
    writer.add_scalar('Time/Communication', comm_time, epoch)
    writer.flush()


def train(model, dataloader, optimizer, scheduler, device, accumulation_steps, epoch, sampler):
    model.train()
    sampler.set_epoch(epoch)  # Shuffle data differently each epoch
    total_loss = 0
    tokens_processed = 0
    scaler = GradScaler()
    optimizer.zero_grad()
    comm_time = 0.0  # To measure communication time
    compute_time = 0.0  # To measure computation time

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Training Epoch {epoch}", disable=(dist.get_rank() != 0))

    for step, batch in progress_bar:
        start_compute = time.time()
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        with autocast():
            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss / accumulation_steps

        scaler.scale(loss).backward()
        end_compute = time.time()
        compute_time += end_compute - start_compute

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            torch.cuda.synchronize()
            start_comm = time.time()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            torch.cuda.synchronize()
            end_comm = time.time()
            comm_time += end_comm - start_comm

        batch_loss = loss.item() * accumulation_steps
        total_loss += batch_loss * inputs.size(0)
        tokens_processed += inputs.numel()

        if dist.get_rank() == 0:
            progress_bar.set_postfix({'loss': f'{batch_loss:.4f}'})

    # Aggregate losses and tokens across processes
    total_loss_tensor = torch.tensor(total_loss, device=device)
    tokens_processed_tensor = torch.tensor(tokens_processed, device=device)
    dist.reduce(total_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(tokens_processed_tensor, dst=0, op=dist.ReduceOp.SUM)

    # Synchronize after training epoch
    dist.barrier()

    if dist.get_rank() == 0:
        avg_loss = total_loss_tensor.item() / (tokens_processed_tensor.item() / inputs.size(-1))
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    else:
        avg_loss = None
        perplexity = None

    return avg_loss, perplexity, tokens_processed_tensor.item(), comm_time


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    tokens_processed = 0
    progress_bar = tqdm(dataloader, desc="Evaluating", disable=(dist.get_rank() != 0))

    with torch.no_grad():
        for batch in progress_bar:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=inputs, labels=labels)
            loss = outputs.loss

            batch_loss = loss.item()
            total_loss += batch_loss * inputs.size(0)
            tokens_processed += inputs.numel()

            if dist.get_rank() == 0:
                progress_bar.set_postfix({'loss': f'{batch_loss:.4f}'})

    total_loss_tensor = torch.tensor(total_loss, device=device)
    tokens_processed_tensor = torch.tensor(tokens_processed, device=device)
    dist.reduce(total_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
    dist.reduce(tokens_processed_tensor, dst=0, op=dist.ReduceOp.SUM)

    # Synchronize after evaluation
    dist.barrier()

    if dist.get_rank() == 0:
        avg_loss = total_loss_tensor.item() / (tokens_processed_tensor.item() / inputs.size(-1))
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    else:
        avg_loss = None
        perplexity = None

    return avg_loss, perplexity, tokens_processed_tensor.item()


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


def get_dataloaders(train_dataset, val_dataset, batch_size, tokenizer):
    # Initialize samplers
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    # Initialize data collator
    data_collator = CustomDataCollator(tokenizer)

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, train_sampler, val_sampler


def main():
    # Get rank and world size from environment variables
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

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
    batch_size = 16
    learning_rate = 5e-4
    checkpoint_dir = 'checkpoints/'
    log_dir = 'logs/'
    tokenizer_path = os.path.join(script_dir, 'tokenizer', 'custom_bpe_tokenizer.json')
    dataset_name = 'Vipplav/phase_1_3M'
    resume_from_checkpoint = False
    accumulation_steps = 4

    set_seed()

    # Create necessary directories
    if rank == 0:
        create_directories([checkpoint_dir, log_dir])

    writer = setup_tensorboard(log_dir) if rank == 0 else None

    if rank == 0:
        logging.info("Starting Phase 1 Training")
        logging.info(f"Parameters: Epochs={epochs}, Batch Size={batch_size}, Learning Rate={learning_rate}")
        logging.info(f"Accumulation Steps: {accumulation_steps}")

        # Check and log GPU processes
        check_and_log_gpu_processes()

        # Clear CUDA memory cache
        clear_cuda_memory()

    # Barrier to ensure all processes have finished setup
    dist.barrier()

    # Load tokenizer and dataset
    tokenizer_path = os.path.abspath(tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    train_dataset, val_dataset = load_dataset_phase1(dataset_name)

    # Barrier to ensure all processes have loaded the dataset
    dist.barrier()

    # Prepare data collator and dataloaders
    train_dataloader, val_dataloader, train_sampler, val_sampler = get_dataloaders(
        train_dataset, val_dataset, batch_size, tokenizer
    )

    if rank == 0:
        logging.info(f"DataLoaders created. Training batches: {len(train_dataloader)}, Validation batches: {len(val_dataloader)}")

    # Adjust accumulation steps to maintain effective batch size
    accumulation_steps = max(1, accumulation_steps // world_size)
    if rank == 0:
        logging.info(f"Adjusted Accumulation Steps: {accumulation_steps}")

    # Initialize model
    checkpoint_path = None
    if resume_from_checkpoint:
        checkpoint_path = os.path.join(checkpoint_dir, 'phase_1_last.pt')
    model = initialize_model(tokenizer, checkpoint_path=checkpoint_path, device=device)

    # Set up optimizer and scheduler
    num_training_steps = epochs * (len(train_dataloader) * world_size) // accumulation_steps
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    optimizer = setup_optimizer(model, learning_rate)
    scheduler = setup_scheduler(optimizer, num_warmup_steps, num_training_steps)

    if rank == 0:
        logging.info(f"Number of training steps: {num_training_steps}")
        logging.info(f"Number of warmup steps: {num_warmup_steps}")

    # Barrier to ensure all processes have initialized the model, optimizer, and scheduler
    dist.barrier()

    # Resume from checkpoint if needed
    start_epoch = 1
    if resume_from_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
        model, optimizer, scheduler, start_epoch = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
        if rank == 0:
            logging.info(f"Resumed training from checkpoint {checkpoint_path}")
    else:
        if rank == 0:
            logging.info("Starting training from scratch")

    best_val_loss = float('inf')
    for epoch in range(start_epoch, epochs + 1):
        if rank == 0:
            logging.info(f"Starting epoch {epoch}/{epochs} for Phase 1")
        epoch_start_time = time.time()

        train_loss, train_perplexity, tokens_processed_train, comm_time = train(
            model, train_dataloader, optimizer, scheduler, device, accumulation_steps, epoch, train_sampler
        )

        val_loss, val_perplexity, tokens_processed_val = evaluate(model, val_dataloader, device)

        epoch_time = time.time() - epoch_start_time

        avg_tokens_per_node = tokens_processed_train / world_size

        if rank == 0:
            logging.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            logging.info(f"Epoch {epoch}: Train Perplexity = {train_perplexity:.4f}, Val Perplexity = {val_perplexity:.4f}")
            logging.info(f"Epoch {epoch}: Tokens Processed (Train) = {tokens_processed_train}, Tokens Processed (Val) = {tokens_processed_val}")
            logging.info(f"Epoch {epoch}: Average Tokens per Node = {avg_tokens_per_node}")
            logging.info(f"Epoch {epoch}: Communication Time = {comm_time:.2f}s")

            # Log metrics
            log_metrics(
                writer, epoch, train_loss, val_loss, train_perplexity, val_perplexity,
                tokens_processed_train, tokens_processed_val, epoch_time, avg_tokens_per_node, comm_time
            )

            # Save the best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, checkpoint_dir, is_best=is_best)

    if rank == 0:
        logging.info("Training completed.")

    cleanup()


if __name__ == '__main__':
    main()
