# Modified from:
#  ./LlamaGen/autoregressive/train/train_c2i_fsdp.py

# Include LlamaGen repo as a library
import sys
sys.path.append("./LlamaGen")

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy, MixedPrecision, StateDictType, FullStateDictConfig
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy, size_based_auto_wrap_policy
import torch.nn.functional as F

import os
import time
import inspect
import functools
import argparse
import contextlib
from glob import glob
import wandb
from copy import deepcopy

from utils.logger import create_logger
from utils.ema import update_ema, requires_grad
from dataset.build import build_dataset
from autoregressive.models.gpt import GPT_models



def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace, device) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        # auto_wrap_policy=size_based_auto_wrap_policy,
        # process_group=fs_init.get_data_parallel_group(),
        device_id=device,
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
            "hsdp": ShardingStrategy.HYBRID_SHARD,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[args.mixed_precision],
            reduce_dtype={
                "fp32": torch.float, "tf32": torch.float,
                "bf16": torch.bfloat16, "fp16": torch.float16,
            }[args.grad_precision or args.mixed_precision],
        ),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )

    torch.cuda.synchronize()

    return model


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

def creat_optimizer_by_name(model, weight_decay, learning_rate, betas, global_rank, logger):
    # start with all of the candidate parameters
    all_param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in all_param_dict.items() if p.requires_grad}
    
    # create optim groups. 
    # Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    
    # decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    # nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    
    # model params are flatten by fsdp, we need to set the params by its name
    decay_params = [p for n, p in param_dict.items() if 'norm' not in n]
    nodecay_params = [p for n, p in param_dict.items() if 'norm' in n]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"(rank {global_rank}) num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"(rank {global_rank}) num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    print(f"(rank {global_rank}) num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"(rank {global_rank}) num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer



def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert args.gpt_type == 'c2i', "FSDP only supports c2i currently."
    # =======================================
    #    Initialize Distributed Training
    # =======================================
    dist.init_process_group("nccl")
    # init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    global_rank = dist.get_rank()
    device = global_rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + global_rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={global_rank}, device={device}, seed={seed}, world_size={dist.get_world_size()}.")
    
    if global_rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.gpt_model.replace("/", "-")  # e.g., GPT-XL/2 --> GPT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{args.expid}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")  

    # ======================================================
    #     Initialize model and resume
    # ======================================================
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    latent_size = args.image_size // args.downsample_size
    if args.keep_dropout:
        model = GPT_models[args.gpt_model](
            vocab_size=args.vocab_size,
            block_size=latent_size ** 2,
            num_classes=args.num_classes,
            cls_token_num=args.cls_token_num,
            model_type=args.gpt_type,
            resid_dropout_p=dropout_p,
            ffn_dropout_p=dropout_p,
            drop_path_rate=args.drop_path_rate,
            token_dropout_p=args.token_dropout_p,
        ).to(device)
    else:
        model = GPT_models[args.gpt_model](
            vocab_size=args.vocab_size,
            block_size=latent_size ** 2,
            num_classes=args.num_classes,
            cls_token_num=args.cls_token_num,
            model_type=args.gpt_type,
            class_dropout_prob=0.0,
            token_dropout_p = 0.0,
            resid_dropout_p = 0.0,
            ffn_dropout_p = 0.0,
            drop_path_rate = 0.0,
        ).to(device)
        disable_dropout_in_model(model)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    ref_model = deepcopy(model).to(device)
    if global_rank == 0:  # other ranks receive weights in setup_fsdp_sync
        checkpoint = torch.load(args.ref_ckpt, map_location="cpu")
        weight = checkpoint["model"] if ("XXL" not in args.ref_ckpt and "3B" not in args.ref_ckpt) else checkpoint
        if "freqs_cis" in weight:
            weight.pop("freqs_cis")
        model.load_state_dict(weight)
        ref_model.load_state_dict(weight) # TODO strict = True
        logger.info(f"Ref ckpt loaded.")
        train_steps = 0
        start_epoch = 0
    requires_grad(ref_model, False)
    logger.info(f"Ref Parameters: {sum(p.numel() for p in ref_model.parameters()):,}")
    model = setup_fsdp_sync(model, args, device)
    ref_model = setup_fsdp_sync(ref_model, args, device)



    # ======================================================
    #     Initialize optimizer and resume
    # ======================================================
    optimizer = creat_optimizer_by_name(model, args.weight_decay, args.lr, (args.beta1, args.beta2), global_rank, logger)

    # ======================================================
    #     Initialize Dataloader
    # ======================================================
    dataset = build_dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=global_rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int((args.global_batch_size // dist.get_world_size()) // args.gradient_accumulation_steps),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Mini batch size is : {int((args.global_batch_size // dist.get_world_size()) // args.gradient_accumulation_steps)}")

    flip_info = 'with' if dataset.flip else 'without'
    aug_info = 10 if 'ten_crop' in dataset.feature_dir else 1
    aug_info = 2 * aug_info if dataset.aug_feature_dir is not None else aug_info
    logger.info(f"Dataset contains {len(dataset):,} images ({args.code_path}) "
                f"{flip_info} flip augmentation and {aug_info} crop augmentation")
     
    train_steps = 0
    start_epoch = 0

    model.train()
    if args.keep_dropout:
        ref_model.train()
    else:
        ref_model.eval()
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_acc = 0
    running_margin = 0
    running_chosen_rew = 0
    running_rejected_rew = 0
    running_sft_loss = 0
    running_shuffled_sft_loss = 0
    running_ref_sft_loss = 0
    running_shuffled_ref_sft_loss = 0
    start_time = time.time()
    acc_step = 0

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device, non_blocking=True) # <bz, 1, 256>
            y = y.to(device, non_blocking=True) # <bz, 1>
            z_indices = x.reshape(x.shape[0], -1)
            c_indices = y.reshape(-1)
            assert z_indices.shape[0] == c_indices.shape[0]

            # generate negative conditions
            shuffled_c_indices = torch.roll(c_indices, shifts=-1, dims=0)
            # The implementation below is not preferred because it may cause sampling bias when batch size is small
            # shuffle_label_B = c_indices[torch.randperm(c_indices.shape[0])]
            if args.uncond_ratio > 0.0:
                # randomly mask conditions to maintain unconditional distribution and thus enable CFG sampling
                TBM = torch.rand(c_indices.shape[0]) < args.uncond_ratio
                shuffled_c_indices[TBM] = args.num_classes
                c_indices[TBM] = args.num_classes

            # concat positive and negative data
            bz = c_indices.shape[0]
            all_c_indices = torch.cat([c_indices, shuffled_c_indices])
            all_z_indices = torch.cat([z_indices, z_indices])
            with {
                "bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
                "fp16": torch.cuda.amp.autocast(dtype=torch.float16),
                "fp32": contextlib.nullcontext(),
                "tf32": contextlib.nullcontext(),
            }[args.mixed_precision]: 
                with torch.no_grad():
                    ref_all_logits, _ = ref_model(cond_idx=all_c_indices, idx=all_z_indices[:,:-1], targets=all_z_indices, training_behavior=True)
                    ref_logits = ref_all_logits[:bz]
                    shuffled_ref_logits = ref_all_logits[bz:]
                    # ref_logits, ref_sft_loss = ref_model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices, training_behavior=True)
                    # shuffled_ref_logits, shuffled_ref_sft_loss = ref_model(cond_idx=shuffled_c_indices, idx=z_indices[:,:-1], targets=z_indices, training_behavior=True)                     
                    ref_sft_loss = shuffled_ref_sft_loss = torch.Tensor([0.0])

                all_logits, _ = model(cond_idx=all_c_indices, idx=all_z_indices[:,:-1], targets=all_z_indices)
                logits = all_logits[:bz]
                shuffled_logits = all_logits[bz:]
                # logits, sft_loss = model(cond_idx=c_indices, idx=z_indices[:,:-1], targets=z_indices)
                # shuffled_logits, shuffled_sft_loss = model(cond_idx=shuffled_c_indices, idx=z_indices[:,:-1], targets=z_indices)
                sft_loss = shuffled_sft_loss = torch.Tensor([0.0])

            img_logps = torch.gather(logits.log_softmax(-1), dim=2, index=z_indices.unsqueeze(2)).squeeze(2).sum(-1)
            negative_img_logps = torch.gather(shuffled_logits.log_softmax(-1), dim=2, index=z_indices.unsqueeze(2)).squeeze(2).sum(-1)
            img_ref_logps = torch.gather(ref_logits.log_softmax(-1), dim=2, index=z_indices.unsqueeze(2)).squeeze(2).sum(-1)
            negative_img_ref_logps = torch.gather(shuffled_ref_logits.log_softmax(-1), dim=2, index=z_indices.unsqueeze(2)).squeeze(2).sum(-1)
            img_logp_gap = img_logps - img_ref_logps
            negative_img_logp_gap = negative_img_logps - negative_img_ref_logps
            
            acc = (img_logp_gap > negative_img_logp_gap).float().mean().detach()
            reward_margin = (img_logp_gap - negative_img_logp_gap).mean().detach()
            
            
            if args.loss_type == "CCA":
                if args.uncond_ratio > 0:
                    # treat unconditional data as positive data
                    neg_weight=((TBM).to(img_logp_gap.device)|(shuffled_c_indices==c_indices).to(img_logp_gap.device))
                    # neg_weight=((TBM).to(img_logp_gap.device)) # shuffled_c_indices==c_indices can be removed (would not affect CCA performance too much)
                    mixed_weight = torch.ones_like(neg_weight) * args.lambda_
                    mixed_weight[neg_weight] = 1.0
                    loss = -F.logsigmoid((img_logp_gap)*args.beta).mean() - (mixed_weight * F.logsigmoid(((neg_weight.float()*2-1) * negative_img_logp_gap)*args.beta)).mean()
                    loss = loss / max(args.negw, 1.0)
                else:
                    loss = - img_logp_gap.mean() + args.lambda_ * negative_img_logp_gap.mean()
            elif args.loss_type == "DPO":
                loss = -F.logsigmoid((img_logp_gap - negative_img_logp_gap)*args.beta).mean()
            elif args.loss_type == "unlearning":
                if args.uncond_ratio > 0:
                    neg_weight=((TBM).to(img_logp_gap.device)|(shuffled_c_indices==c_indices).to(img_logp_gap.device))
                    mixed_weight = torch.ones_like(neg_weight) * args.lambda_
                    mixed_weight[neg_weight] = 1.0
                    loss = - img_logp_gap.mean() - (mixed_weight * (neg_weight.float()*2-1) *negative_img_logp_gap).mean()
                    loss = loss / max(args.lambda_, 1.0)
                else:
                    loss = - img_logp_gap.mean() + args.lambda_ * negative_img_logp_gap.mean()
            else:
                raise NotImplementedError
            loss.backward()
            if (acc_step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm != 0.0:
                #   according to https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_
                #   torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    model.clip_grad_norm_(args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                acc_step = 0
            else:
                acc_step += 1
                continue

            # Log loss values:
            running_loss += loss.item()
            running_acc += acc.item()
            running_margin += reward_margin.item()
            running_chosen_rew += img_logp_gap.mean().detach().item()
            running_rejected_rew += negative_img_logp_gap.mean().detach().item()
            running_sft_loss += sft_loss.mean().detach().item()
            running_shuffled_sft_loss += shuffled_sft_loss.mean().detach().item()
            running_ref_sft_loss += ref_sft_loss.mean().detach().item()
            running_shuffled_ref_sft_loss += shuffled_ref_sft_loss.mean().detach().item()

            log_steps += 1
            train_steps += 1
            if (train_steps % args.log_every == 0) or (train_steps < 1000):
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_acc = torch.tensor(running_acc / log_steps, device=device)
                avg_margin = torch.tensor(running_margin / log_steps, device=device)
                avg_chosen_rew = torch.tensor(running_chosen_rew / log_steps, device=device)
                avg_rejected_rew = torch.tensor(running_rejected_rew / log_steps, device=device)

                avg_sft_loss = torch.tensor(running_sft_loss / log_steps, device=device)
                avg_shuffled_sft_loss = torch.tensor(running_shuffled_sft_loss / log_steps, device=device)
                avg_ref_sft_loss = torch.tensor(running_ref_sft_loss / log_steps, device=device)
                avg_shuffled_ref_sft_loss = torch.tensor(running_shuffled_ref_sft_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_acc, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_margin, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_chosen_rew, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_rejected_rew, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                avg_acc = avg_acc.item() / dist.get_world_size()
                avg_margin = avg_margin.item() / dist.get_world_size()
                avg_chosen_rew = avg_chosen_rew.item() / dist.get_world_size()
                avg_rejected_rew = avg_rejected_rew.item() / dist.get_world_size()
                avg_sft_loss = avg_sft_loss.item() / dist.get_world_size()
                avg_shuffled_sft_loss = avg_shuffled_sft_loss.item() / dist.get_world_size()
                avg_ref_sft_loss = avg_ref_sft_loss.item() / dist.get_world_size()
                avg_shuffled_ref_sft_loss = avg_shuffled_ref_sft_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f} Train Acc: {avg_acc:.4f} Rew Margin: {avg_margin:.4f}={avg_chosen_rew:.2f}  -  {avg_rejected_rew:.2f}, Reg loss: ({avg_sft_loss:.2f},{avg_shuffled_sft_loss:.2f},{avg_ref_sft_loss:.2f},{avg_shuffled_ref_sft_loss:.2f}), Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                running_acc=0
                running_margin=0
                running_chosen_rew=0
                running_rejected_rew=0
                running_sft_loss = 0
                running_shuffled_sft_loss = 0
                running_ref_sft_loss = 0
                running_shuffled_ref_sft_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if (train_steps > 0) and ((train_steps % args.ckpt_every == 0) or (train_steps==5000)):
                ### saving model parameters
                with FSDP.state_dict_type(
                    model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                ):
                    consolidated_model_state_dict = model.state_dict()
                    if global_rank == 0:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(consolidated_model_state_dict, checkpoint_path)
                dist.barrier()
                del consolidated_model_state_dict
                

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # CCA parameters
    parser.add_argument("--expid", type=str, required=True, help='Identifier')
    parser.add_argument("--ref_ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--uncond_ratio", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--lambda_", type=float, default=1000.0, help="CCA lambda")
    parser.add_argument("--loss_type", type=str, choices=["DPO", "Unlearning", "CCA"], default="CCA")
    parser.add_argument("--beta", type=float, default=0.02, help="CCA beta")
    parser.add_argument("--keep_dropout", type=int, default=1, help="Whether enable dropout during training.")
    # LlamaGen parameters
    parser.add_argument("--code-path", type=str, required=True)
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-B")
    parser.add_argument("--gpt-resume", type=str, default=None, help="model, optimizer and argument path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--dataset", type=str, default='imagenet_code')
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 parameter for the Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="beta2 parameter for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, choices=["fp32", "tf32", "fp16", "bf16"], default='bf16') 
    parser.add_argument("--data-parallel", type=str, choices=["sdp", "fsdp", "hsdp"], default="fsdp")
    parser.add_argument("--grad-precision", type=str, choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--wandb-project", type=str, default='c2i_fsdp')
    parser.add_argument("--no-wandb", action='store_true')
    args = parser.parse_args()
    main(args)
