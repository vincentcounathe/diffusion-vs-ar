"""Stage-2 training loop for the information-gain critic.

This script freezes the diffusion denoiser and fits a regression model that
predicts the counterfactual drop in cross-entropy when revealing a masked
position.  It reuses the existing LLaMAFactory-style argument parser so the
critic can be trained with the same dataset/config files as the base model.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import HfArgumentParser, Seq2SeqTrainingArguments, set_seed

from llmtuner.dsets import get_dataset, preprocess_dataset, split_dataset
from llmtuner.hparams import (
    DataArguments,
    DiffusionArguments,
    FinetuningArguments,
    ModelArguments,
)
from llmtuner.tuner.core.loader import load_model_and_tokenizer
from models.info_gain_critic import CriticConfig, InfoGainCritic
from utils.feature_extractor import FeatureExtractorConfig, extract_token_features


def _rankdata(a: np.ndarray) -> np.ndarray:
    if a.size == 0:
        return a
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    sorted_vals = a[order]
    start = 0
    n = len(a)
    while start < n:
        end = start
        while end + 1 < n and np.isclose(sorted_vals[end + 1], sorted_vals[start]):
            end += 1
        avg_rank = 0.5 * (start + end) + 1.0
        ranks[order[start : end + 1]] = avg_rank
        start = end + 1
    return ranks


def _safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _safe_spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    return _safe_pearsonr(_rankdata(x), _rankdata(y))


@dataclass
class CriticTrainingArguments:
    critic_hidden_sizes: str = field(
        default="512,256",
        metadata={"help": "Comma-separated hidden layer sizes for the critic."},
    )
    critic_dropout: float = field(
        default=0.05,
        metadata={"help": "Dropout applied between critic layers."},
    )
    candidate_sample_size: int = field(
        default=40,
        metadata={"help": "How many masked positions per sequence to supervise."},
    )
    feature_include_propagation: bool = field(default=True, metadata={"help": "Include propagation features."})
    feature_include_entropy: bool = field(default=True, metadata={"help": "Include entropy feature."})
    feature_include_margin: bool = field(default=True, metadata={"help": "Include margin feature."})
    feature_include_global: bool = field(default=True, metadata={"help": "Include timestep & mask ratio."})
    feature_include_context: bool = field(default=True, metadata={"help": "Include layer-normalized hidden state."})
    log_interval: int = field(default=10, metadata={"help": "Steps between critic loss logs."})

    def hidden_dims(self) -> Tuple[int, ...]:
        parts = [p.strip() for p in self.critic_hidden_sizes.split(",") if p.strip()]
        if not parts:
            raise ValueError("critic_hidden_sizes must include at least one dimension")
        return tuple(int(p) for p in parts)


def build_dataloader(dataset, training_args: Seq2SeqTrainingArguments, shuffle: bool = True, sampler=None) -> DataLoader:
    columns = ["input_ids", "attention_mask", "src_mask"]
    dataset.set_format(type="torch", columns=columns)
    batch_size = (
        training_args.per_device_train_batch_size
        if shuffle or training_args.per_device_eval_batch_size is None
        else training_args.per_device_eval_batch_size
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
    )


def q_sample(x0: torch.Tensor, t: torch.Tensor, maskable_mask: torch.Tensor, mask_token_id: int, total_steps: int):
    u = torch.rand_like(x0, dtype=torch.float, device=x0.device)
    mask = (u < ((t + 1) / total_steps)[:, None]) & maskable_mask
    xt = x0.masked_fill(mask, mask_token_id)
    return xt, mask


def run_denoiser(model, input_ids, attention_mask):
    embeds = model.get_embeds(input_ids)
    outputs = model.denoise_model(
        inputs_embeds=embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    logits = model.get_logits(hidden)
    return logits, hidden, outputs.attentions


def nll_per_position(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)


def masked_ce_mean(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = nll_per_position(logits, targets)
    mask_float = mask.float()
    total = (losses * mask_float).sum(dim=-1)
    denom = mask_float.sum(dim=-1).clamp(min=1.0)
    return total / denom


def sample_candidate_positions(mask: torch.Tensor, max_candidates: int) -> List[Tuple[int, int]]:
    candidates: List[Tuple[int, int]] = []
    if max_candidates <= 0:
        mask_positions = mask.nonzero(as_tuple=False)
        return [(int(b.item()), int(i.item())) for b, i in mask_positions]

    for batch_idx in range(mask.size(0)):
        positions = mask[batch_idx].nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            continue
        perm = torch.randperm(positions.numel(), device=mask.device)
        take = min(max_candidates, positions.numel())
        chosen = positions[perm[:take]]
        for pos in chosen:
            candidates.append((batch_idx, int(pos.item())))
    return candidates


def evaluate_critic(
    critic: InfoGainCritic,
    model,
    dataloader: Optional[DataLoader],
    tokenizer,
    diffusion_args: DiffusionArguments,
    critic_args,
    device: torch.device,
    feature_cfg: FeatureExtractorConfig,
    world_size: int = 1,
    rank: int = 0,
) -> Optional[Dict[str, float]]:
    if dataloader is None:
        return None

    critic.eval()
    total_loss = 0.0
    total_count = 0
    all_preds: List[float] = []
    all_targets: List[float] = []
    group_matches = 0
    group_total = 0
    topk_matches = 0
    predicted_top_true_gain_sum = 0.0
    candidate_timesteps_all: List[int] = []
    candidate_mask_ratios_all: List[float] = []
    low_bucket_preds: List[float] = []
    low_bucket_targets: List[float] = []
    high_bucket_preds: List[float] = []
    high_bucket_targets: List[float] = []

    with torch.no_grad():
        for step_idx, batch in enumerate(dataloader):
            x0 = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            src_mask = batch["src_mask"].bool().to(device)
            maskable_mask = (~src_mask) & attention_mask.bool()
            batch_size = x0.size(0)

            t = torch.randint(0, diffusion_args.diffusion_steps, (batch_size,), device=device)
            xt, loss_mask = q_sample(
                x0,
                t,
                maskable_mask=maskable_mask,
                mask_token_id=tokenizer.mask_token_id,
                total_steps=diffusion_args.diffusion_steps,
            )
            if loss_mask.sum() == 0:
                continue

            logits, hidden, attentions = run_denoiser(model, xt, attention_mask)
            ce_pre = masked_ce_mean(logits, x0, loss_mask)
            feature_output = extract_token_features(
                logits=logits,
                hidden_states=hidden,
                attentions=attentions,
                mask=loss_mask,
                timestep=t,
                total_steps=diffusion_args.diffusion_steps,
                config=feature_cfg,
            )

            candidate_positions = sample_candidate_positions(loss_mask, critic_args.candidate_sample_size)
            if not candidate_positions:
                continue

            index_map: Dict[Tuple[int, int], int] = {
                (int(b), int(i)): idx
                for idx, (b, i) in enumerate(feature_output.masked_indices.tolist())
            }

            batch_features: List[torch.Tensor] = []
            batch_targets: List[torch.Tensor] = []
            group_ids: List[int] = []
            batch_timesteps_meta: List[int] = []
            batch_mask_ratios_meta: List[float] = []
            for batch_idx, pos_idx in candidate_positions:
                key = (batch_idx, pos_idx)
                feat_index = index_map.get(key)
                if feat_index is None:
                    continue

                z_i = feature_output.features[feat_index].to(device)
                cf_inputs = xt.clone()
                cf_mask = loss_mask.clone()
                cf_inputs[batch_idx, pos_idx] = x0[batch_idx, pos_idx]
                cf_mask[batch_idx, pos_idx] = False
                cf_logits, _, _ = run_denoiser(model, cf_inputs, attention_mask)
                ce_post = masked_ce_mean(cf_logits, x0, cf_mask)
                delta = torch.clamp(ce_pre[batch_idx] - ce_post[batch_idx], min=0.0).detach()

                batch_features.append(z_i)
                batch_targets.append(delta)
                global_idx = step_idx * dataloader.batch_size + batch_idx
                group_ids.append(global_idx)
                batch_timesteps_meta.append(int(t[batch_idx].item()))
                batch_mask_ratios_meta.append(float(loss_mask[batch_idx].float().mean().item()))

            if not batch_features:
                continue

            feature_tensor = torch.stack(batch_features, dim=0)
            target_tensor = torch.stack(batch_targets, dim=0)
            preds = critic(feature_tensor)
            smooth_l1 = F.smooth_l1_loss(preds, target_tensor)
            total_loss += smooth_l1.item() * feature_tensor.size(0)
            total_count += feature_tensor.size(0)

            preds_cpu = preds.detach().cpu().numpy()
            targets_cpu = target_tensor.detach().cpu().numpy()
            all_preds.extend(preds_cpu.tolist())
            all_targets.extend(targets_cpu.tolist())
            candidate_timesteps_all.extend(batch_timesteps_meta)
            candidate_mask_ratios_all.extend(batch_mask_ratios_meta)

            half_step = diffusion_args.diffusion_steps / 2.0
            for pred_val, tgt_val, t_val in zip(preds_cpu, targets_cpu, batch_timesteps_meta):
                if t_val < half_step:
                    low_bucket_preds.append(pred_val)
                    low_bucket_targets.append(tgt_val)
                else:
                    high_bucket_preds.append(pred_val)
                    high_bucket_targets.append(tgt_val)

            groups: Dict[int, List[Tuple[float, float]]] = {}
            for g, p, tgt in zip(group_ids, preds_cpu, targets_cpu):
                groups.setdefault(g, []).append((p, tgt))

            for entries in groups.values():
                if not entries:
                    continue
                pred_scores = np.array([p for p, _ in entries])
                target_scores = np.array([t for _, t in entries])
                pred_top = int(np.argmax(pred_scores))
                target_top = int(np.argmax(target_scores))
                if pred_top == target_top:
                    group_matches += 1
                group_total += 1
                top_k = min(3, len(entries))
                top_true_indices = np.argsort(target_scores)[-top_k:]
                if pred_top in top_true_indices:
                    topk_matches += 1
                predicted_top_true_gain_sum += float(target_scores[pred_top])

    critic.train()
    if total_count == 0:
        return None

    if dist.is_initialized():
        payload = (
            all_preds,
            all_targets,
            candidate_timesteps_all,
            candidate_mask_ratios_all,
            low_bucket_preds,
            low_bucket_targets,
            high_bucket_preds,
            high_bucket_targets,
            total_loss,
            total_count,
            group_matches,
            group_total,
            topk_matches,
            predicted_top_true_gain_sum,
        )
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, payload)

        all_preds, all_targets = [], []
        candidate_timesteps_all, candidate_mask_ratios_all = [], []
        low_bucket_preds, low_bucket_targets = [], []
        high_bucket_preds, high_bucket_targets = [], []
        total_loss = total_count = group_matches = group_total = topk_matches = 0.0
        predicted_top_true_gain_sum = 0.0

        for (
            preds_i,
            targets_i,
            t_i,
            mr_i,
            low_p_i,
            low_t_i,
            high_p_i,
            high_t_i,
            loss_i,
            count_i,
            gm_i,
            gt_i,
            topk_i,
            gain_i,
        ) in gathered:
            all_preds.extend(preds_i)
            all_targets.extend(targets_i)
            candidate_timesteps_all.extend(t_i)
            candidate_mask_ratios_all.extend(mr_i)
            low_bucket_preds.extend(low_p_i)
            low_bucket_targets.extend(low_t_i)
            high_bucket_preds.extend(high_p_i)
            high_bucket_targets.extend(high_t_i)
            total_loss += loss_i
            total_count += count_i
            group_matches += gm_i
            group_total += gt_i
            topk_matches += topk_i
            predicted_top_true_gain_sum += gain_i

    mean_loss = total_loss / total_count if total_count > 0 else 0.0
    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)
    corr = _safe_pearsonr(preds_arr, targets_arr)
    spearman_corr = _safe_spearmanr(preds_arr, targets_arr)
    rmse = float(np.sqrt(np.mean((preds_arr - targets_arr) ** 2))) if targets_arr.size > 0 else 0.0
    ranking_acc = group_matches / group_total if group_total > 0 else 0.0
    topk_acc = topk_matches / group_total if group_total > 0 else 0.0
    true_gain_avg = predicted_top_true_gain_sum / group_total if group_total > 0 else 0.0
    target_mean = float(targets_arr.mean()) if targets_arr.size > 0 else 0.0
    target_std = float(targets_arr.std()) if targets_arr.size > 0 else 0.0
    target_min = float(targets_arr.min()) if targets_arr.size > 0 else 0.0
    target_max = float(targets_arr.max()) if targets_arr.size > 0 else 0.0
    target_frac_zero = float(np.mean(np.isclose(targets_arr, 0.0))) if targets_arr.size > 0 else 0.0
    timestep_arr = np.array(candidate_timesteps_all, dtype=float)
    mask_ratio_arr = np.array(candidate_mask_ratios_all, dtype=float)
    corr_timestep_delta = _safe_pearsonr(timestep_arr, targets_arr) if timestep_arr.size > 0 else 0.0
    corr_maskratio_delta = _safe_pearsonr(mask_ratio_arr, targets_arr) if mask_ratio_arr.size > 0 else 0.0
    low_corr = _safe_pearsonr(np.array(low_bucket_preds), np.array(low_bucket_targets)) if len(low_bucket_preds) > 1 else 0.0
    high_corr = _safe_pearsonr(np.array(high_bucket_preds), np.array(high_bucket_targets)) if len(high_bucket_preds) > 1 else 0.0
    metrics = {
        "eval/loss": mean_loss,
        "eval/acc_corr": corr,
        "eval/acc_rank": ranking_acc,
        "eval_corr_pearson": corr,
        "eval_corr_spearman": spearman_corr,
        "eval_rmse": rmse,
        "eval_top1_in_topk_true": topk_acc,
        "eval_true_gain_at_pred_top1": true_gain_avg,
        "eval_target_mean": target_mean,
        "eval_target_std": target_std,
        "eval_target_min": target_min,
        "eval_target_max": target_max,
        "eval_target_frac_zero": target_frac_zero,
        "eval_corr_timestep_delta": corr_timestep_delta,
        "eval_corr_maskratio_delta": corr_maskratio_delta,
        "eval_corr_pred_delta_low_t": low_corr,
        "eval_corr_pred_delta_high_t": high_corr,
    }
    # Only rank 0 returns metrics in distributed mode
    if dist.is_initialized() and rank != 0:
        return None
    return metrics


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_dist = local_rank != -1

    if use_dist and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    parser = HfArgumentParser(
        (
            ModelArguments,
            DiffusionArguments,
            DataArguments,
            Seq2SeqTrainingArguments,
            FinetuningArguments,
            CriticTrainingArguments,
        )
    )
    (
        model_args,
        diffusion_args,
        data_args,
        training_args,
        finetuning_args,
        critic_args,
    ) = parser.parse_args_into_dataclasses()

    report_to = training_args.report_to
    if isinstance(report_to, str):
        report_to = [report_to]
    use_wandb = wandb is not None and report_to and "wandb" in report_to
    if use_wandb and (not use_dist or dist.get_rank() == 0):
        run_name = os.path.basename(training_args.output_dir.rstrip("/")) or "critic-train"
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "diffusion-vs-ar"),
            name=run_name,
            config={
                "dataset": data_args.dataset,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "epochs": training_args.num_train_epochs,
            },
        )

    if training_args.seed is not None:
        set_seed(training_args.seed)

    if finetuning_args.stage == "sft":
        raise ValueError("Critic training requires a diffusion/MDM checkpoint (stage != 'sft').")

    model, tokenizer = load_model_and_tokenizer(
        model_args,
        finetuning_args,
        is_trainable=False,
        diffusion_args=diffusion_args,
    )
    if use_dist:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    data_args.init_for_training(training_args.seed)
    dataset = get_dataset(model_args, data_args)
    dataset = preprocess_dataset(dataset, tokenizer, data_args, training_args, stage=finetuning_args.stage)
    split = split_dataset(dataset, data_args, training_args)
    train_dataset = split.get("train_dataset")
    eval_dataset = split.get("eval_dataset")

    if train_dataset is None and eval_dataset is not None:
        train_dataset = eval_dataset
        eval_dataset = None

    if train_dataset is None:
        raise ValueError("training dataset is required for critic fitting.")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=dist.get_rank()) if use_dist else None
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=dist.get_rank(), shuffle=False) if (use_dist and eval_dataset is not None) else None

    dataloader = build_dataloader(train_dataset, training_args, shuffle=True, sampler=train_sampler)
    eval_dataloader = build_dataloader(eval_dataset, training_args, shuffle=False, sampler=eval_sampler) if eval_dataset is not None else None
    critic_config: Optional[CriticConfig] = None
    critic: Optional[InfoGainCritic] = None
    optimizer: Optional[AdamW] = None

    feature_cfg = FeatureExtractorConfig(
        include_entropy=critic_args.feature_include_entropy,
        include_margin=critic_args.feature_include_margin,
        include_propagation=critic_args.feature_include_propagation,
        include_global=critic_args.feature_include_global,
        include_context=critic_args.feature_include_context,
    )

    global_step = 0
    total_steps = diffusion_args.diffusion_steps
    gradient_accumulation = max(1, training_args.gradient_accumulation_steps)
    os.makedirs(training_args.output_dir, exist_ok=True)

    def save_checkpoint(base_name: str):
        if critic is None:
            return
        state = {
            "critic_state_dict": critic.module.state_dict() if isinstance(critic, DDP) else critic.state_dict(),
            "critic_config": critic_config.__dict__,
            "feature_config": feature_cfg.__dict__,
        }
        pt_path = os.path.join(training_args.output_dir, f"{base_name}.pt")
        cfg_path = os.path.join(training_args.output_dir, f"{base_name}_config.json")
        torch.save(state, pt_path)
        with open(cfg_path, "w") as f:
            json.dump(state["critic_config"], f, indent=2)
        print(f"Saved critic checkpoint to {pt_path}")

    for epoch in range(math.ceil(training_args.num_train_epochs)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        iter_data = dataloader
        progress = tqdm(iter_data, desc=f"epoch {epoch+1}") if (not use_dist or dist.get_rank() == 0) else iter_data
        optimizer.zero_grad(set_to_none=True) if optimizer else None
        for batch in progress:
            x0 = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            src_mask = batch["src_mask"].bool().to(device)
            maskable_mask = (~src_mask) & attention_mask.bool()
            batch_size = x0.size(0)
            t = torch.randint(0, total_steps, (batch_size,), device=device)
            xt, loss_mask = q_sample(
                x0,
                t,
                maskable_mask=maskable_mask,
                mask_token_id=tokenizer.mask_token_id,
                total_steps=total_steps,
            )

            if loss_mask.sum() == 0:
                continue

            logits, hidden, attentions = run_denoiser(model, xt, attention_mask)
            ce_pre = masked_ce_mean(logits, x0, loss_mask)
            feature_output = extract_token_features(
                logits=logits,
                hidden_states=hidden,
                attentions=attentions,
                mask=loss_mask,
                timestep=t,
                total_steps=total_steps,
                config=feature_cfg,
            )

            if critic is None:
                critic_config = CriticConfig(
                    input_dim=feature_output.features.size(-1),
                    hidden_dims=critic_args.hidden_dims(),
                    dropout=critic_args.critic_dropout,
                )
                critic = InfoGainCritic(critic_config).to(device)
                if use_dist:
                    critic = DDP(
                        critic,
                        device_ids=[local_rank] if device.type == "cuda" else None,
                        output_device=local_rank if device.type == "cuda" else None,
                        broadcast_buffers=False,
                    )
                optimizer = AdamW(
                    critic.parameters(),
                    lr=training_args.learning_rate,
                    weight_decay=training_args.weight_decay,
                )
                optimizer.zero_grad(set_to_none=True)

            assert critic is not None and optimizer is not None

            candidate_positions = sample_candidate_positions(loss_mask, critic_args.candidate_sample_size)
            if not candidate_positions:
                continue

            index_map: Dict[Tuple[int, int], int] = {
                (int(b), int(i)): idx
                for idx, (b, i) in enumerate(feature_output.masked_indices.tolist())
            }

            batch_features: List[torch.Tensor] = []
            batch_targets: List[torch.Tensor] = []

            for batch_idx, pos_idx in candidate_positions:
                key = (batch_idx, pos_idx)
                feat_index = index_map.get(key)
                if feat_index is None:
                    continue
                z_i = feature_output.features[feat_index].to(device).detach()

                cf_inputs = xt.clone()
                cf_mask = loss_mask.clone()
                cf_inputs[batch_idx, pos_idx] = x0[batch_idx, pos_idx]
                cf_mask[batch_idx, pos_idx] = False
                cf_logits, _, _ = run_denoiser(model, cf_inputs, attention_mask)
                ce_post = masked_ce_mean(cf_logits, x0, cf_mask)
                delta = torch.clamp(ce_pre[batch_idx] - ce_post[batch_idx], min=0.0).detach()

                batch_features.append(z_i)
                batch_targets.append(delta)

            if not batch_features:
                continue

            feature_tensor = torch.stack(batch_features, dim=0)
            target_tensor = torch.stack(batch_targets, dim=0)

            critic.train()
            preds = critic(feature_tensor)
            loss = F.smooth_l1_loss(preds, target_tensor)
            (loss / gradient_accumulation).backward()

            if (global_step + 1) % gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), training_args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            if critic_args.log_interval > 0 and global_step % critic_args.log_interval == 0:
                if not use_dist or dist.get_rank() == 0:
                    progress.set_postfix({"critic_loss": loss.item()})
                if use_wandb and (not use_dist or dist.get_rank() == 0):
                    all_targets = torch.stack(batch_targets)
                    target_min = float(all_targets.min().item())
                    target_max = float(all_targets.max().item())
                    target_frac_zero = float((all_targets == 0).float().mean().item())
                    feature_stats = {
                        "feature/entropy_mean": float(feature_output.entropy[loss_mask].mean().item()),
                        "feature/entropy_std": float(feature_output.entropy[loss_mask].std().item()),
                        "feature/margin_mean": float(feature_output.margin[loss_mask].mean().item()),
                        "feature/margin_std": float(feature_output.margin[loss_mask].std().item()),
                    }
                    target_stats = {
                        "target_mean": float(all_targets.mean().item()),
                        "target_std": float(all_targets.std().item()),
                        "target_min": target_min,
                        "target_max": target_max,
                        "target_frac_zero": target_frac_zero,
                    }
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            **feature_stats,
                            **target_stats,
                        },
                        step=global_step,
                    )

            should_eval = (
                eval_dataloader is not None
                and training_args.do_eval
                and training_args.evaluation_strategy == "steps"
                and training_args.eval_steps
                and global_step % training_args.eval_steps == 0
            )
            if should_eval:
                metrics = evaluate_critic(
                    critic,
                    model,
                    eval_dataloader,
                    tokenizer,
                    diffusion_args,
                    critic_args,
                    device,
                    feature_cfg,
                    world_size=world_size,
                    rank=dist.get_rank() if use_dist else 0,
                )
                if metrics and (not use_dist or dist.get_rank() == 0):
                    progress.write(f"Eval metrics: {metrics}")
                    if use_wandb and (not use_dist or dist.get_rank() == 0):
                        wandb.log(metrics, step=global_step)

        if use_wandb and (not use_dist or dist.get_rank() == 0):
            wandb.log({"train/epoch": epoch + 1}, step=global_step)

        if not use_dist or dist.get_rank() == 0:
            job_suffix = os.environ.get("SLURM_JOB_ID")
            epoch_base = f"info_gain_critic_epoch{epoch+1}"
            if job_suffix:
                epoch_base = f"{epoch_base}_{job_suffix}"
            save_checkpoint(epoch_base)

    if critic is not None and (not use_dist or dist.get_rank() == 0):
        job_suffix = os.environ.get("SLURM_JOB_ID")
        base_name = f"info_gain_critic_{job_suffix}" if job_suffix else "info_gain_critic"
        save_checkpoint(base_name)
    else:
        print("No critic was trained. Check if the dataset produces masked tokens.")

    if eval_dataloader is not None and training_args.do_eval:
        metrics = evaluate_critic(
            critic,
            model,
            eval_dataloader,
            tokenizer,
            diffusion_args,
            critic_args,
            device,
            feature_cfg,
            world_size=world_size,
            rank=dist.get_rank() if use_dist else 0,
        )
        if metrics and (not use_dist or dist.get_rank() == 0):
            print(f"Final eval metrics: {metrics}")
            if use_wandb:
                wandb.log(metrics, step=global_step if global_step > 0 else None)

    if use_wandb and (not use_dist or dist.get_rank() == 0):
        wandb.finish()


if __name__ == "__main__":
    main()
