# Modified from:
#   ./LlamaGen/autoregressive/train/train_c2i.py
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from glob import glob
from copy import deepcopy
import os
import time
import inspect
import argparse

from LlamaGen.utils.logger import create_logger
from LlamaGen.utils.distributed import init_distributed_mode
from LlamaGen.utils.ema import update_ema, requires_grad
from LlamaGen.dataset.build import build_dataset
from LlamaGen.autoregressive.models.gpt import GPT_models


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer



#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Setup an experiment folder:
    if rank == 0:
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

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")


    # Setup model
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
    requires_grad(ref_model, False)
    logger.info(f"Ref Parameters: {sum(p.numel() for p in ref_model.parameters()):,}")
    if args.ema:
        ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")

    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    # Setup data:
    dataset = build_dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
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

    # Prepare models for training:
    if args.ref_ckpt:
        checkpoint = torch.load(args.ref_ckpt, map_location="cpu")
        weight = checkpoint["model"] if "XXL" not in args.ref_ckpt and "3B" not in args.ref_ckpt else checkpoint
        if "freqs_cis" in weight:
            weight.pop("freqs_cis")
        model.load_state_dict(weight)
        ref_model.load_state_dict(weight)
        if args.ema:
            ema.load_state_dict(weight)
        logger.info(f"Ref ckpt loaded.")
        del checkpoint
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    else:
        raise NotImplementedError

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model) # requires PyTorch 2.0        
        ref_model = torch.compile(ref_model) # requires PyTorch 2.0        

    model = DDP(model.to(device), device_ids=[args.gpu])
    model.train()
    if args.keep_dropout:
        ref_model.train()
    else:
        ref_model.eval() 
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
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
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
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
            with torch.cuda.amp.autocast(dtype=ptdtype):
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
            # backward pass, with gradient scaling if training in fp16         
            scaler.scale(loss).backward()
            if (acc_step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # step the optimizer and scaler if training in fp16
                scaler.step(optimizer)
                scaler.update()
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)
                acc_step = 0
            else:
                acc_step += 1
                continue
            
            # flush the gradients as soon as we can, no need for this memory anymore
            if args.ema:
                update_ema(ema, model.module._orig_mod if not args.no_compile else model.module)

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
                # 5000 steps = 1 epoch  is batch size is 256
                if rank == 0:
                    if not args.no_compile:
                        model_weight = model.module._orig_mod.state_dict()
                    else:
                        model_weight = model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    dist.destroy_process_group()


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
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path-rate", type=float, default=0.0, help="using stochastic depth decay")
    parser.add_argument("--no-compile", action='store_true')
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
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    args = parser.parse_args()
    main(args)
