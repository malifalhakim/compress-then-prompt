from tqdm import tqdm
from itertools import chain
import torch
import argparse
import numpy as np
import time
import torch.nn.functional as F
import os
from collections.abc import Mapping
from transformers import AutoTokenizer, set_seed, default_data_collator
from datasets import load_dataset, Dataset
from typing import Any, Union, Dict, List, Optional, Tuple
from prompt import LLamaPromptTuningLM, OPTPromptTuningLM, llama_loader, TextDataset, BloomPromptTuningLM
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, LlamaTokenizer
from transformers.optimization import Adafactor, AdafactorSchedule


def parse_arguments():
    """Parse command line arguments for soft prompt learning."""
    parser = argparse.ArgumentParser(description="Soft Prompt Learning for Language Models")
    parser.add_argument("--model", type=str, required=True, 
                        help="Model identifier (e.g., facebook/opt, decapoda-research/llama, etc.)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save model checkpoints and results")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name (wikitext2, ptb, c4)")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model or model identifier")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--max_steps", default=20000, type=int,
                        help="Maximum number of training steps")
    parser.add_argument("--prompt_lr", type=float, default=0.3,
                        help="Learning rate for the soft prompt parameters")
    parser.add_argument("--warmup_step_prompt", type=int, default=500,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length for the model")
    parser.add_argument("--init_from_vocab", action="store_false",
                        help="Initialize soft prompt from vocabulary")
    parser.add_argument("--eval_every_steps", type=int, default=500,
                        help="Evaluate model every N steps")
    parser.add_argument("--soft_token_num", type=int, default=20,
                        help="Number of soft tokens to use")
    parser.add_argument("--optimizer", type=str, default="Adafactor",
                        help="Optimizer to use (Adafactor or AdamW)")
    parser.add_argument("--dataloader_num_workers", type=int, default=16,
                        help="Number of workers for data loading")
    parser.add_argument("--dataloader_pin_memory", action="store_true",
                        help="Pin memory for data loader")
    parser.add_argument("--seqlen", type=int, default=1024,
                        help="Sequence length for training")
    parser.add_argument("--root", type=str, required=True,
                        help="Root directory for saving results")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Batch size per device for evaluation")
    return parser.parse_args()


def freeze_model(model):
    """
    Freeze all model parameters except the soft prompt parameters.
    
    Args:
        model: The model to freeze
        
    Returns:
        model: The model with frozen parameters
    """
    for n, m in model.named_parameters():
        if "soft_prompt" in n:
            m.requires_grad = True 
        else:
            m.requires_grad = False

    tot_params = sum(p.numel() for p in model.parameters())
    print(f"***** Model Total Parameters: {tot_params} *****")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"***** Model Trainable Parameters: {trainable_params} *****")

    print(f"***** Trainable Parameters Ratio: {trainable_params/tot_params * 100:.4f}% *****")

    return model


def prepare_input(data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
    """
    Prepare inputs for model by moving tensors to the correct device.
    
    Args:
        data: Input data, can be tensor or nested structure of tensors
        
    Returns:
        Prepared input on the correct device
    """
    if isinstance(data, Mapping):
        return type(data)({k: prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device='cuda')
        return data.to(**kwargs)
    return data


def loss_func(logits, inputs_ids, model, loss_fct):
    """
    Calculate the loss for the model.
    
    Args:
        logits: Model output logits
        inputs_ids: Input token IDs
        model: The model
        loss_fct: Loss function
        
    Returns:
        loss: The calculated loss
    """
    labels = prepare_input_and_label(model, inputs_ids)
    logits = logits[..., :-1, :].contiguous()
    batch_size, seq_len, vocab_size = logits.shape
    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss = loss.view(batch_size, -1).sum(dim=-1)
    loss = loss.mean()
    return loss


def prepare_input_and_label(model, inputs_ids):
    """
    Prepare input and labels for autoregressive language modeling.
    
    Shifts input right to create labels: input=A,B,C -> labels=B,C,D
    
    Args:
        model: The language model
        inputs_ids: Input token IDs
        
    Returns:
        labels: Labels for computing loss
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    padded_input_tokens = model._extend_labels(inputs_ids)
    labels = padded_input_tokens[..., 1:].contiguous()
    input_tokens = padded_input_tokens[..., :-1].contiguous()
    labels[input_tokens < 0] = -100
    return labels


@torch.no_grad()
def evaluate(prompt_model, val_loader, loss_fct, is_llama=True):
    """
    Evaluate the model on validation data.
    
    Args:
        prompt_model: The model to evaluate
        val_loader: Validation data loader
        loss_fct: Loss function
        is_llama: Whether the model is LLaMA
        
    Returns:
        ppl: Perplexity on validation set
    """
    prompt_model.eval()
    nlls = []
    total_samples = 0
    for idx, inputs_ids in tqdm(enumerate(val_loader), desc="Evaluating"):
        if torch.cuda.device_count() == 1:
            inputs_ids = inputs_ids.cuda()
        bs, seqlen = inputs_ids.shape
        total_samples += bs
        labels = prepare_input_and_label(prompt_model, inputs_ids)
        
        output = prompt_model(inputs_ids)
        shift_logits = output.logits[:, :-1, :]
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.shape[-1]), labels.view(-1))
        neg_log_likelihood = loss.float().reshape(bs, -1).mean(dim=-1) * seqlen
        
        if torch.cuda.device_count() > 1:
            nll = accelerator.gather(neg_log_likelihood.view(1, -1))
        else:
            nll = neg_log_likelihood.view(1, -1)
        nlls.append(nll)
    
    nlls = torch.hstack(nlls).view(-1)
    ppl = torch.exp(nlls.sum() / (nlls.numel() * seqlen))
    print(f"Perplexity: {ppl.item():.3f}")
    return ppl.item()


def load_model_and_tokenizer(args):
    """
    Load the model and tokenizer based on model type.
    
    Args:
        args: Command line arguments
        
    Returns:
        prompt_model: The loaded and frozen model with soft prompt
        tokenizer: The tokenizer
        is_llama: Flag indicating if model is LLaMA
    """
    if 'opt' in args.model:
        prompt_model = OPTPromptTuningLM.from_pretrained(
            args.model_name_or_path,
            soft_prompt_path=None,
            n_tokens=args.soft_token_num,
            initialize_from_vocab=args.init_from_vocab,
            torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        is_llama = False
    
    elif 'llama' in args.model:
        prompt_model = LLamaPromptTuningLM.from_pretrained(
            args.model_name_or_path,
            soft_prompt_path=None,
            n_tokens=args.soft_token_num,
            initialize_from_vocab=args.init_from_vocab,
            torch_dtype=torch.bfloat16
        )
        tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)
        is_llama = True
    
    elif 'bloom' in args.model:
        prompt_model = BloomPromptTuningLM.from_pretrained(
            args.model_name_or_path,
            soft_prompt_path=None,
            n_tokens=args.soft_token_num,
            initialize_from_vocab=args.init_from_vocab,
            torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        is_llama = False
    
    else:
        raise NotImplementedError("Currently only OPT, LLaMA, and BLOOM models are supported")
    
    prompt_model = freeze_model(prompt_model)
    print(prompt_model.soft_prompt)
    
    return prompt_model, tokenizer, is_llama


def load_dataset_and_prepare_dataloaders(args, tokenizer):
    """
    Load and prepare dataset and dataloaders.
    
    Args:
        args: Command line arguments
        tokenizer: Tokenizer for the model
        
    Returns:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
    """
    if args.dataset == "wikitext2":
        raw_tra_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        raw_val_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
        
        train_dataset = TextDataset(raw_tra_data, tokenizer, args, mode="train", col_key='text')
        val_dataset = TextDataset(raw_val_data, tokenizer, args, mode="val", col_key='text')

    elif args.dataset == 'ptb':
        raw_tra_data = load_dataset('ptb_text_only', 'penn_treebank', split='train', trust_remote_code=True)
        raw_val_data = load_dataset('ptb_text_only', 'penn_treebank', split='validation', trust_remote_code=True)
        
        train_dataset = TextDataset(raw_tra_data, tokenizer, args, mode="train", col_key='sentence')
        val_dataset = TextDataset(raw_val_data, tokenizer, args, mode="val", col_key='sentence')

    elif args.dataset == 'c4':
        raw_tra_data = load_dataset(
            'allenai/c4',
            data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, 
            split='train'
        )
        raw_val_data = load_dataset(
            'allenai/c4',
            data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, 
            split='validation'
        )
        
        train_dataset = TextDataset(raw_tra_data, tokenizer, args, mode="train", col_key='text', cutoff=5000)
        val_dataset = TextDataset(raw_val_data, tokenizer, args, mode="val", col_key='text', cutoff=1100)

    else:
        raise NotImplementedError("Currently only wikitext2, ptb, and c4 datasets are supported")
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.per_device_train_batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.per_device_eval_batch_size, 
        shuffle=False
    )
    
    return train_dataloader, val_dataloader


def setup_optimizer_and_scheduler(args, prompt_model):
    """
    Set up optimizer and learning rate scheduler.
    
    Args:
        args: Command line arguments
        prompt_model: The model with soft prompt
        
    Returns:
        optimizer: The configured optimizer
        scheduler: The learning rate scheduler
    """
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in prompt_model.named_parameters() if n == "soft_prompt.weight"],
            "weight_decay": 0.0,  # Following Openprompt package, we do not use weight decay for soft prompt
        }
    ]

    if args.optimizer.lower() == "adafactor":
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.prompt_lr,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
            weight_decay=1e-5
        )
        scheduler = get_constant_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=args.warmup_step_prompt
        )
    
    elif args.optimizer.lower() == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, 
            lr=args.prompt_lr, 
            weight_decay=1e-5
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_step_prompt, 
            num_training_steps=args.max_steps
        )
    
    else:
        raise NotImplementedError("Currently only AdamW and Adafactor optimizers are supported")
    
    return optimizer, scheduler


def save_model_checkpoint(args, prompt_model, optimizer, accelerator=None):
    """
    Save model checkpoint.
    
    Args:
        args: Command line arguments
        prompt_model: The model to save
        optimizer: The optimizer state to save
        accelerator: Accelerator for distributed training
    """
    save_path = f"{args.root}/{args.output_dir}/best.pth"
    
    if accelerator is not None and isinstance(prompt_model, torch.nn.parallel.DistributedDataParallel):
        unwrapped_model = accelerator.unwrap_model(prompt_model)
        accelerator.save({
            "model": unwrapped_model.soft_prompt.state_dict(),
            "optimizer": optimizer.optimizer.state_dict()  # optimizer is an AcceleratedOptimizer object
        }, save_path)
    else:
        torch.save({
            "model": prompt_model.soft_prompt.state_dict(), 
            "optimizer": optimizer.state_dict(),
        }, save_path)
    
    print(f"Model saved to {save_path}")


def train(args, prompt_model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fct, is_llama):
    """
    Train the soft prompt model.
    
    Args:
        args: Command line arguments
        prompt_model: The model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        loss_fct: Loss function
        is_llama: Whether the model is LLaMA
        
    Returns:
        best_val_ppl: Best validation perplexity
        acc_traces: List of validation perplexities during training
    """
    tot_loss = 0
    log_loss = 0
    best_val_ppl = float('inf')
    glb_step = 0
    actual_step = 0
    acc_traces = []
    tot_train_time = 0
    pbar_update_freq = 10
    
    # Set up accelerator for distributed training if needed
    if torch.cuda.device_count() > 1:
        accelerator = Accelerator()
        prompt_model, optimizer, train_dataloader, scheduler = accelerator.prepare(
            prompt_model, optimizer, train_dataloader, scheduler)
        device = accelerator.device
        val_dataloader = accelerator.prepare(val_dataloader)
    else:
        device = "cuda"
        accelerator = None
    
    prompt_model = prompt_model.to(device)
    
    pbar = tqdm(total=args.max_steps, desc="Training")
    
    for epoch in range(1000000):  # Very large number, we'll break based on max_steps
        print(f"Begin epoch {epoch}")
        prompt_model.train()
        
        for step, batch in enumerate(train_dataloader):
            if torch.cuda.device_count() > 1:
                input_ids = batch
            else:
                input_ids = batch.cuda()
                
            start_time = time.time()
            
            output = prompt_model(input_ids)
            logits = output.logits
            loss = loss_func(logits, input_ids, prompt_model, loss_fct)
            
            if accelerator is not None:
                accelerator.backward(loss)
            else:
                loss.backward()
                
            tot_loss += loss.item()
            actual_step += 1
            
            # Clip gradient
            torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
            glb_step += 1

            # Update progress bar
            if glb_step % pbar_update_freq == 0:
                aveloss = (tot_loss - log_loss) / pbar_update_freq
                pbar.update(pbar_update_freq)
                pbar.set_postfix({'loss': aveloss})
                log_loss = tot_loss
                
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            tot_train_time += time.time() - start_time

            # Evaluate periodically
            if glb_step > 0 and glb_step % args.eval_every_steps == 0:
                val_ppl = evaluate(prompt_model, val_dataloader, loss_fct, is_llama)
                print(f'Step {glb_step}, Validation PPL: {val_ppl}')
                
                if val_ppl <= best_val_ppl:
                    save_model_checkpoint(args, prompt_model, optimizer, accelerator)
                    best_val_ppl = val_ppl
                    print(f"New best validation PPL: {best_val_ppl}")

                acc_traces.append(val_ppl)
                print(f"Step {glb_step}, Val PPL {val_ppl}, Avg step time {tot_train_time/actual_step:.4f}s")
                prompt_model.train()
                
            if glb_step >= args.max_steps:
                print(f"Reached max steps ({args.max_steps}). Training complete.")
                return best_val_ppl, acc_traces

    return best_val_ppl, acc_traces


def main():
    """Main function for soft prompt tuning."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(f"{args.root}/{args.output_dir}", exist_ok=True)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Log configuration
    config_str = "="*20 + "\n"
    config_str += f"dataset: {args.dataset}\t"
    config_str += f"model: {args.model}\t"
    config_str += f"seed: {args.seed}\t"
    config_str += f"init_from_vocab: {args.init_from_vocab}\t"
    config_str += f"eval_every_steps: {args.eval_every_steps}\t"
    config_str += f"prompt_lr: {args.prompt_lr}\t"
    config_str += f"optimizer: {args.optimizer}\t"
    config_str += f"seqlen: {args.seqlen}\t"
    config_str += f"warmup_step_prompt: {args.warmup_step_prompt}\t"
    config_str += f"soft_token_num: {args.soft_token_num}\t"
    config_str += "\n"
    print(config_str)
    
    # Load model and tokenizer
    prompt_model, tokenizer, is_llama = load_model_and_tokenizer(args)
    
    # Load dataset and prepare dataloaders
    train_dataloader, val_dataloader = load_dataset_and_prepare_dataloaders(args, tokenizer)
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(args, prompt_model)
    
    # Setup loss function
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    
    # Train the model
    best_val_ppl, acc_traces = train(
        args,
        prompt_model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        loss_fct,
        is_llama
    )
    
    # Save final results
    print(f"Training completed. Best validation PPL: {best_val_ppl}")
    
    # Write validation performance trace to file
    with open(f"{args.root}/{args.output_dir}/val_performance.txt", "w") as f:
        f.write("\n".join([str(x) for x in acc_traces]))


if __name__ == "__main__":
    main()