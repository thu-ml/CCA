# Modified from ./VAR/trainer.py

# Include VAR repo as a library
import sys
sys.path.append("./VAR")

import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.nn.functional as F

import dist
from VAR.models import VAR, VQVAE, VectorQuantizer2
from VAR.utils.amp_sc import AmpOptimizer
from VAR.utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class VAR_CCATrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, ref_var_wo_ddp: VAR, var_wo_ddp: VAR, var: DDP,
        var_opt: AmpOptimizer, label_smooth: float,
        loss_type: str,
        beta: float, lambda_: float, uncond_ratio: float, # CCA parameters
    ):
        super(VAR_CCATrainer, self).__init__()
        self.beta = beta
        self.lambda_ = lambda_
        self.uncond_ratio = uncond_ratio
        self.loss_type = loss_type

        self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.ref_var_wo_ddp: VAR = ref_var_wo_ddp  # after torch.compile
        self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt
        
        del self.var_wo_ddp.rng
        del self.ref_var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)
        self.ref_var_wo_ddp.rng = torch.Generator(device=device)
        
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_type_weight = torch.ones(1, self.L, device=device) / self.L
        
        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn*pn
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
    
    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        for inp_B3HW, label_B in ld_val:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)
            
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            
            self.var_wo_ddp.forward
            logits_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l)
            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)
            tot += B
        self.var_wo_ddp.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # if progressive training
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1    # no prog warmup at first prog stage, as it's already solved in wp
        if prog_si == len(self.patch_nums) - 1: prog_si = -1    # max prog, as if no prog
        
        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping
        
        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)# inp_B3HW <bz,3,256,256> --> gt_idx_Bl [[x]*bz, [x,x,x,x]*bz, ....  ] 
        gt_BL = torch.cat(gt_idx_Bl, dim=1) # <bz, 680>
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl) # <bz, 679, 32>
        
        # generate negative conditions
        shuffle_label_B = torch.roll(label_B, shifts=-1, dims=0)
        # The implementation below is not preferred because it may cause sampling bias when batch size is small
        # shuffle_label_B = label_B[torch.randperm(label_B.shape[0])]
        
        if self.uncond_ratio > 0:
            # randomly mask conditions to maintain unconditional distribution and thus enable CFG sampling
            TBM = torch.rand(label_B.shape[0], device=label_B.device) < self.uncond_ratio
            label_B = torch.where(TBM, 1000, label_B) # 1000 is the uncond mask
            shuffle_label_B = torch.where(TBM, 1000, shuffle_label_B) # 1000 is the uncond mask

        # concat positive and negative data
        bz = label_B.shape[0]
        all_label_B = torch.cat([label_B, shuffle_label_B])
        all_x_BLCv_wo_first_l = torch.cat([x_BLCv_wo_first_l, x_BLCv_wo_first_l])
        
        self.ref_var_wo_ddp.train() # enable dropout for reference model

        with self.var_opt.amp_ctx:
            all_logits = self.var(all_label_B, all_x_BLCv_wo_first_l) # <2bz, 680, 4096>
            logits_BLV = all_logits[:bz]
            shuffled_logits_BLV = all_logits[bz:]
            with torch.no_grad():
                ref_all_logits = self.ref_var_wo_ddp(all_label_B, all_x_BLCv_wo_first_l) # <2bz, 680, 4096>
                ref_logits_BLV = ref_all_logits[:bz]
                shuffled_ref_logits_BLV = ref_all_logits[bz:]

        img_logps = torch.gather(logits_BLV.log_softmax(-1), dim=2, index=gt_BL.unsqueeze(2)).squeeze(2).sum(-1)
        negative_img_logps = torch.gather(shuffled_logits_BLV.log_softmax(-1), dim=2, index=gt_BL.unsqueeze(2)).squeeze(2).sum(-1)
        img_ref_logps = torch.gather(ref_logits_BLV.log_softmax(-1), dim=2, index=gt_BL.unsqueeze(2)).squeeze(2).sum(-1)
        negative_img_ref_logps = torch.gather(shuffled_ref_logits_BLV.log_softmax(-1), dim=2, index=gt_BL.unsqueeze(2)).squeeze(2).sum(-1)
        img_logp_gap = img_logps - img_ref_logps
        negative_img_logp_gap = negative_img_logps - negative_img_ref_logps

        if self.loss_type == "CCA":
            if self.uncond_ratio > 0:
                # treat unconditional data as positive data
                neg_weight=((TBM).to(img_logp_gap.device)|(shuffle_label_B==label_B).to(img_logp_gap.device)) # default paper settings
                # neg_weight=((TBM).to(img_logp_gap.device)) # shuffle_label_B==label_B can be removed (would not affect CCA performance too much)
                mixed_weight = torch.ones_like(neg_weight) * self.lambda_
                mixed_weight[neg_weight] = 1.0
                loss = - F.logsigmoid((img_logp_gap)*self.beta).mean() - (mixed_weight * F.logsigmoid(((neg_weight.float()*2-1) * negative_img_logp_gap)*self.beta)).mean()
                loss = loss / max(self.lambda_, 1.0) # stablize training
            else:
                # Simplified CCA loss
                loss = - F.logsigmoid((img_logp_gap)*self.beta).mean() - self.lambda_ * F.logsigmoid( - negative_img_logp_gap*self.beta).mean()
        elif self.loss_type == "DPO":
            loss = -F.logsigmoid((img_logp_gap - negative_img_logp_gap)*self.beta).mean()
        elif self.loss_type == "unlearning":
            if self.uncond_ratio > 0:
                neg_weight=((TBM).to(img_logp_gap.device)|(shuffle_label_B==label_B).to(img_logp_gap.device))
                mixed_weight = torch.ones_like(neg_weight) * self.lambda_
                mixed_weight[neg_weight] = 1.0
                loss = - img_logp_gap.mean() - (mixed_weight * (neg_weight.float()*2-1) *negative_img_logp_gap).mean()
                loss = loss / max(self.lambda_, 1.0)
            else:
                loss = - img_logp_gap.mean() + self.lambda_ * negative_img_logp_gap.mean()
        else:
            raise NotImplementedError

        with torch.no_grad():
            acc = (img_logp_gap > negative_img_logp_gap).float().mean().detach()
            reward_margin = (img_logp_gap - negative_img_logp_gap).mean().detach()
            sftloss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1).mean()
            shuffle_sftloss = self.train_loss(shuffled_logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1).mean()
            ref_sftloss = self.train_loss(ref_logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1).mean()
            ref_shuffle_sftloss = self.train_loss(shuffled_ref_logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1).mean()

        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            if prog_si >= 0:    # in progressive training
                raise NotImplementedError
            else:               # not in progressive training
                Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item()
                acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
            grad_norm = grad_norm.item() if grad_norm is not None else grad_norm
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
        
        if g_it < 50 or (g_it + 1) % 100 == 0:
            chosen_reward = img_logp_gap.float().mean().detach()
            rejected_reward = negative_img_logp_gap.float().mean().detach()
            uncond_reward = negative_img_logp_gap[TBM].float().mean().detach()

            dist.allreduce(acc)
            dist.allreduce(chosen_reward)
            dist.allreduce(uncond_reward)
            dist.allreduce(rejected_reward)
            dist.allreduce(reward_margin)
            dist.allreduce(sftloss)
            dist.allreduce(shuffle_sftloss)
            dist.allreduce(ref_sftloss)
            dist.allreduce(ref_shuffle_sftloss)

            if dist.is_master():
                tb_lg.update(head='AR_iter_loss', 
                            acc=acc.float().mean().detach().item() / dist.get_world_size(),
                            chosen_reward=chosen_reward.float().mean().detach().item() / dist.get_world_size(),
                            rejected_reward=rejected_reward.float().mean().detach().item() / dist.get_world_size(),
                            uncond_reward=uncond_reward.float().mean().detach().item() / dist.get_world_size(),
                            reward_margin=reward_margin.float().mean().detach().item() / dist.get_world_size(),
                            sftloss=sftloss.float().mean().detach().item() / dist.get_world_size(),
                            shuffle_sftloss=shuffle_sftloss.float().mean().detach().item() / dist.get_world_size(),
                            ref_sftloss=ref_sftloss.float().mean().detach().item() / dist.get_world_size(),
                            ref_shuffle_sftloss=ref_shuffle_sftloss.float().mean().detach().item() / dist.get_world_size(),
                            step=g_it)
    
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2
    
    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp',):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state['var_wo_ddp']
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        raise NotImplementedError