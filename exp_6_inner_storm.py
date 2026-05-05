# %%
# Experiment 6: inner-STORM for SCENT
# Exported from exp_6_inner_storm.ipynb.


# %% [markdown]
# # Experiment 6: inner-STORM for SCENT

# %% [markdown]
# ## 1. One-time setup

# %% [markdown]
# ## 2. Imports

# %%
import os
import logging
import pathlib
import json
import sys
import random
import math
import subprocess

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from libauc.losses import EntLossClassification
from libauc.optimizers import SCENT
# %% [markdown]
# ## 3. Data and model definitions

# %%
class LinearClassifier(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(feature_dim, num_classes, bias=False)
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        classes: torch.Tensor | str | None = None,
        return_classes: bool = False,
    ):
        w_pos = self.fc.weight[labels]  # (B, D)

        mask = None
        sampled_classes = None
        if isinstance(classes, str):
            assert classes == "all"
            sampled_classes = torch.arange(self.num_classes, device=labels.device)
            w_sampled = self.fc.weight
        elif isinstance(classes, torch.Tensor):
            sampled_classes = classes.to(labels.device)
            w_sampled = self.fc.weight[sampled_classes]
        elif classes is None:
            sampled_classes = torch.unique(labels)
            w_sampled = self.fc.weight[sampled_classes]
            mask = labels.unsqueeze(1) == sampled_classes.unsqueeze(0)
        else:
            raise ValueError(f"Unknown classes type: {type(classes)}")

        logits = x @ w_sampled.T - torch.sum(x * w_pos, dim=1, keepdim=True)

        if mask is not None:
            logits = logits.masked_fill(mask.to(logits.device), float("-inf"))

        if return_classes:
            return logits, sampled_classes
        return logits
# %%
class FeaturesDataset(Dataset):
    """Dataset for precomputed features.

    Expects features_path (N, D) and labels_path (N,) where labels are ints 0..C-1.
    """

    def __init__(self, features_path: str, labels_path: str):
        self.features = torch.load(features_path, map_location="cpu")
        self.labels = torch.load(labels_path, map_location="cpu")
        assert self.features.shape[0] == self.labels.shape[0], "features/labels length mismatch"

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y, idx
# %% [markdown]
# ## 4. Shared training utilities

# %%
def setup_logging(out_log_file=None):
    logging.root.handlers = []
    logging.root.setLevel(level=logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d,%H:%M:%S"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if out_log_file is not None:
        file_handler = logging.FileHandler(out_log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)

def build_dataloaders(root_data_dir, batch_size, num_workers=0):
    dataloader_list = []
    for split, shuffle in zip(["train", "val", "test"], [True, False, False]):
        data_dir = os.path.join(root_data_dir, split)
        if not os.path.exists(data_dir):
            if split in ["train", "val"]:
                raise FileNotFoundError(f"Data directory {data_dir} does not exist.")
            dataloader_list.append(None)
            continue

        features = os.path.join(data_dir, "features.pt")
        labels = os.path.join(data_dir, "labels.pt")
        ds = FeaturesDataset(features, labels)
        dataloader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        dataloader_list.append(dataloader)

    train_loader, val_loader, test_loader = dataloader_list
    return train_loader, val_loader, test_loader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Set random seed to {seed}")

def compute_log_priors(train_loader, num_classes):
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, labels, _ in train_loader:
        counts += torch.bincount(labels, minlength=num_classes).float()
    priors = counts / counts.sum()
    priors = priors.clamp_min(1e-12)
    return priors.log()

def apply_la_to_sampled_logits(logits, sampled_classes, labels, log_priors, tau):
    if tau <= 0:
        return logits
    neg_shift = log_priors[sampled_classes].unsqueeze(0)
    pos_shift = log_priors[labels].unsqueeze(1)
    return logits + tau * (neg_shift - pos_shift)

# %%
def evaluate(model, loader, device, log_priors=None, eval_tau=0.0):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            feats, labels, _ = batch
            feats = feats.to(device)
            labels = labels.to(device, dtype=torch.long)

            logits = model.fc(feats)

            if log_priors is not None and eval_tau > 0:
                logits = logits + eval_tau * log_priors.to(device).unsqueeze(0)

            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * feats.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += feats.size(0)

            if i % 100 == 0:
                logging.info(f"  Batch {i} / {len(loader)}: loss={loss.item():.6f}")

    return total_loss / total, correct / total

# %%
def train_one_epoch(model, loader, optimizer, criterion, device, log_priors=None, train_tau=0.0):
    model.train()
    total_loss = 0.0
    total = 0

    for i, batch in enumerate(loader):
        feats, labels, indices = batch
        feats = feats.to(device)
        labels = labels.to(device, dtype=torch.long)

        logits, sampled_classes = model(feats, labels, return_classes=True)
        logits = apply_la_to_sampled_logits(logits, sampled_classes, labels, log_priors, train_tau)

        loss_dict = criterion(logits, indices)
        loss = loss_dict["loss"]

        with torch.no_grad():
            model.eval()
            base_logits = model.fc(feats)
            cross_entropy_loss = F.cross_entropy(base_logits, labels)
            loss_dict["cross_entropy_loss"] = cross_entropy_loss
            model.train()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += cross_entropy_loss.item() * feats.size(0)
        total += feats.size(0)

        if i % 100 == 0:
            log_str = f"  Batch {i} / {len(loader)}:"
            for key, value in loss_dict.items():
                log_str += f" {key}={value.item():.6f}"
            logging.info(log_str)

    return total_loss / total

# %%
def make_scheduler(optimizer, epochs, warmup_epochs=0):
    def lr_lambda(epoch_idx):
        epoch_num = epoch_idx + 1
        if warmup_epochs > 0 and epoch_num <= warmup_epochs:
            return epoch_num / warmup_epochs

        if epochs == warmup_epochs:
            return 1.0

        progress = (epoch_num - warmup_epochs) / max(1, epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# %%
def forward_with_weight_snapshot(weight_snapshot_cpu, feats, labels, sampled_classes):
    """Recompute sampled logits using a CPU snapshot for sampled class rows."""
    rows = sampled_classes.detach().cpu()
    local_weight = weight_snapshot_cpu[rows].to(feats.device)

    mask = labels.unsqueeze(1) == sampled_classes.unsqueeze(0)
    pos_idx = mask.float().argmax(dim=1)

    w_pos = local_weight[pos_idx]
    logits = feats @ local_weight.t() - torch.sum(feats * w_pos, dim=1, keepdim=True)
    logits = logits.masked_fill(mask, float("-inf"))
    return logits


def batch_inner_exp_from_logits(logits):
    """Approximate the inner E[exp(s)] term over sampled negative logits."""
    finite_mask = torch.isfinite(logits)
    exp_logits = torch.where(finite_mask, torch.exp(logits), torch.zeros_like(logits))
    counts = finite_mask.sum(dim=1).clamp_min(1)
    return exp_logits.sum(dim=1) / counts


def init_inner_storm_state(model, data_size):
    return {
        "u": torch.ones(data_size, dtype=torch.float32),
        "prev_weight_cpu": model.fc.weight.detach().cpu().clone(),
        "step": 0,
    }


def update_inner_storm_state(
    feats,
    labels,
    indices,
    sampled_classes,
    logits,
    criterion,
    storm_state,
    log_priors=None,
    train_tau=0.0,
    beta=0.10,
    correction_scale=0.25,
    eps=1e-8,
):
    """Update criterion.nu with a STORM-style estimate of the inner exp term.

    This leaves model gradients untouched. The only state injected into SCENT is
    criterion.nu for the current sample indices.
    """
    idx_cpu = indices.long().cpu()

    with torch.no_grad():
        g_curr = batch_inner_exp_from_logits(logits.detach())

        if storm_state["step"] == 0:
            g_prev = g_curr
        else:
            prev_logits = forward_with_weight_snapshot(
                storm_state["prev_weight_cpu"], feats, labels, sampled_classes
            )
            prev_logits = apply_la_to_sampled_logits(
                prev_logits, sampled_classes, labels, log_priors, train_tau
            )
            g_prev = batch_inner_exp_from_logits(prev_logits)

        u_prev = storm_state["u"][idx_cpu].to(feats.device)

        if storm_state["step"] == 0:
            u_new = g_curr
        else:
            # Conservative STORM correction at the inner estimator level.
            u_new = (
                (1.0 - beta) * u_prev
                + beta * g_curr
                + correction_scale * (1.0 - beta) * (g_curr - g_prev)
            )

        u_new = torch.clamp(u_new, min=eps)
        storm_state["u"][idx_cpu] = u_new.detach().cpu()
        storm_state["step"] += 1

        if hasattr(criterion, "nu"):
            criterion.nu[idx_cpu] = torch.log(u_new).detach().cpu().unsqueeze(1)


def train_one_epoch_inner_storm(
    model,
    loader,
    optimizer,
    criterion,
    device,
    storm_state,
    beta=0.10,
    correction_scale=0.25,
    log_priors=None,
    train_tau=0.0,
):
    model.train()
    total_loss = 0.0
    total = 0

    for i, batch in enumerate(loader):
        feats, labels, indices = batch
        feats = feats.to(device)
        labels = labels.to(device, dtype=torch.long)

        logits, sampled_classes = model(feats, labels, return_classes=True)
        logits = apply_la_to_sampled_logits(logits, sampled_classes, labels, log_priors, train_tau)

        update_inner_storm_state(
            feats,
            labels,
            indices,
            sampled_classes,
            logits,
            criterion,
            storm_state,
            log_priors=log_priors,
            train_tau=train_tau,
            beta=beta,
            correction_scale=correction_scale,
        )

        optimizer.zero_grad()
        loss_dict = criterion(logits, indices)
        loss = loss_dict["loss"]
        loss.backward()

        rows_cpu = sampled_classes.detach().cpu()
        rows_gpu = sampled_classes.detach().to(device)
        weight_before_step_rows = model.fc.weight.detach()[rows_gpu].cpu().clone()

        optimizer.step()
        storm_state["prev_weight_cpu"][rows_cpu] = weight_before_step_rows

        with torch.no_grad():
            model.eval()
            base_logits = model.fc(feats)
            cross_entropy_loss = F.cross_entropy(base_logits, labels)
            loss_dict["cross_entropy_loss"] = cross_entropy_loss
            model.train()

        total_loss += cross_entropy_loss.item() * feats.size(0)
        total += feats.size(0)

        if i % 100 == 0:
            log_str = f"  Batch {i} / {len(loader)}:"
            for key, value in loss_dict.items():
                log_str += f" {key}={value.item():.6f}"
            logging.info(log_str)

    return total_loss / total, storm_state


# %% [markdown]
# ## 5. Global experiment settings

# %%
data_dir = "./features/treeoflife10m_subset"
root_out_dir = "./outputs_exp6_inner_storm/"
feature_folder_url = "https://drive.google.com/drive/folders/10cY2Azqz9Gnci-r4fXcGIpH_e5YaKXH_?usp=sharing"

seed = 2026
device = "cuda"
feature_dim = 512
num_classes = 163002
data_size = 762654
save_frequency = 1

epochs = 20
batch_size = 128

setup_logging()


def download_features_if_needed():
    data_root = pathlib.Path(data_dir)
    if (data_root / "train" / "features.pt").exists() and (data_root / "val" / "features.pt").exists():
        logging.info(f"Found feature data at {data_root}")
        return

    logging.info(f"Feature data not found at {data_root}; downloading with gdown.")
    try:
        import gdown  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "gdown is required to download the feature folder. Install it with:\n"
            "  python -m pip install gdown\n"
            "Then rerun with --download-data."
        ) from exc

    cmd = [
        sys.executable,
        "-m",
        "gdown",
        "--folder",
        feature_folder_url,
        "-O",
        "./features",
    ]
    subprocess.run(cmd, check=True)

    if not (data_root / "train" / "features.pt").exists():
        raise FileNotFoundError(
            f"Download completed, but expected data is still missing at {data_root}. "
            "Check the downloaded folder layout under ./features."
        )

# %% [markdown]
# ## 6. Experiment grid

# %%
EXPERIMENTS = [
    dict(
        name="exp6_cfg01_scent_ref",
        algorithm="scent",
        lr=0.002,
        alpha=0.0,
        alpha_multiplier=0.03,
        momentum=0.9,
        weight_decay=0.0,
        train_tau=0.0,
        eval_tau=0.0,
        warmup_epochs=0,
        use_inner_storm=False,
    ),
    dict(
        name="exp6_cfg02_scent_warmup_ref",
        algorithm="scent",
        lr=0.002,
        alpha=0.0,
        alpha_multiplier=0.03,
        momentum=0.9,
        weight_decay=0.0,
        train_tau=0.0,
        eval_tau=0.0,
        warmup_epochs=2,
        use_inner_storm=False,
    ),
    dict(
        name="exp6_cfg03_inner_storm",
        algorithm="scent",
        lr=0.002,
        alpha=0.0,
        alpha_multiplier=0.03,
        momentum=0.9,
        weight_decay=0.0,
        train_tau=0.0,
        eval_tau=0.0,
        warmup_epochs=0,
        use_inner_storm=True,
        storm_start_epoch=1,
        inner_storm_beta=0.10,
        inner_storm_correction_scale=0.25,
    ),
    dict(
        name="exp6_cfg04_warmup_inner_storm_delayed",
        algorithm="scent",
        lr=0.002,
        alpha=0.0,
        alpha_multiplier=0.03,
        momentum=0.9,
        weight_decay=0.0,
        train_tau=0.0,
        eval_tau=0.0,
        warmup_epochs=2,
        use_inner_storm=True,
        storm_start_epoch=3,
        inner_storm_beta=0.10,
        inner_storm_correction_scale=0.25,
    ),
    dict(
        name="exp6_cfg05_warmup_inner_storm_weak",
        algorithm="scent",
        lr=0.002,
        alpha=0.0,
        alpha_multiplier=0.03,
        momentum=0.9,
        weight_decay=0.0,
        train_tau=0.0,
        eval_tau=0.0,
        warmup_epochs=2,
        use_inner_storm=True,
        storm_start_epoch=3,
        inner_storm_beta=0.05,
        inner_storm_correction_scale=0.10,
    ),
]

# %% [markdown]
# ## 7. Experiment construction and execution helpers

# %%
def build_experiment(cfg):
    name = cfg["name"]
    out_dir = pathlib.Path(root_out_dir) / name
    os.makedirs(out_dir / "checkpoints", exist_ok=True)
    out_log_file = out_dir / "out.log"

    setup_logging(out_log_file)
    set_seed(seed)

    train_loader, val_loader, test_loader = build_dataloaders(data_dir, batch_size, num_workers=0)
    model = LinearClassifier(feature_dim, num_classes).to(device)

    optimizer = SCENT(
        model.parameters(),
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )

    lr_scheduler = make_scheduler(
        optimizer,
        epochs=epochs,
        warmup_epochs=cfg["warmup_epochs"],
    )

    if cfg["algorithm"] == "scent":
        criterion = EntLossClassification(
            data_size=data_size,
            alpha=cfg["alpha"],
            is_scent=True,
            alpha_multiplier=cfg["alpha_multiplier"],
        )
    elif cfg["algorithm"] == "sox":
        criterion = EntLossClassification(
            data_size=data_size,
            gamma=cfg["gamma"],
            is_scent=False,
        )
    else:
        raise ValueError(cfg["algorithm"])

    log_priors = compute_log_priors(train_loader, num_classes).to(device)

    return {
        "name": name,
        "out_dir": out_dir,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
        "criterion": criterion,
        "log_priors": log_priors,
    }

# %%
def run_experiment(cfg):
    exp = build_experiment(cfg)

    name = exp["name"]
    out_dir = exp["out_dir"]
    train_loader = exp["train_loader"]
    val_loader = exp["val_loader"]
    test_loader = exp["test_loader"]
    model = exp["model"]
    optimizer = exp["optimizer"]
    lr_scheduler = exp["lr_scheduler"]
    criterion = exp["criterion"]
    log_priors = exp["log_priors"]

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_test_loss = float("inf")
    all_rows = []

    eval_path = out_dir / f"eval_{name}.jsonl"
    if eval_path.exists():
        eval_path.unlink()

    inner_storm_state = (
        init_inner_storm_state(model, data_size)
        if cfg.get("use_inner_storm", False)
        else None
    )

    for epoch in range(1, epochs + 1):
        logging.info(
            f"{name} | Epoch {epoch}/{epochs}, learning_rate={lr_scheduler.get_last_lr()[0]:.6f}"
        )

        if hasattr(criterion, "adjust_gamma"):
            criterion.adjust_gamma(epoch, epochs)

        if cfg.get("use_inner_storm", False) and epoch >= cfg.get("storm_start_epoch", 1):
            train_ce, inner_storm_state = train_one_epoch_inner_storm(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                inner_storm_state,
                beta=cfg.get("inner_storm_beta", 0.10),
                correction_scale=cfg.get("inner_storm_correction_scale", 0.25),
                log_priors=log_priors,
                train_tau=cfg["train_tau"],
            )
        else:
            train_ce = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                log_priors=log_priors,
                train_tau=cfg["train_tau"],
            )

        lr_scheduler.step()

        logging.info("Evaluating on validation set")
        val_loss, val_acc = evaluate(
            model,
            val_loader,
            device,
            log_priors=log_priors,
            eval_tau=cfg["eval_tau"],
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if test_loader is not None:
                logging.info("Evaluating on test set")
                best_test_loss, best_test_acc = evaluate(
                    model,
                    test_loader,
                    device,
                    log_priors=log_priors,
                    eval_tau=cfg["eval_tau"],
                )

        row = {
            "epoch": epoch,
            "cross_entropy_loss": train_ce,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc,
            "best_test_acc": best_test_acc,
            "best_test_loss": best_test_loss,
        }
        all_rows.append(row)

        logging.info(
            f"{name} | epoch={epoch} train_ce={train_ce:.6f} "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.6f} "
            f"best_val_acc={best_val_acc:.6f} "
            f"best_test_acc={best_test_acc:.6f}"
        )

        with open(eval_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        if epoch % save_frequency == 0 or epoch == epochs:
            save_dict = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "config": cfg,
            }
            if hasattr(criterion, "nu"):
                save_dict["criterion_nu"] = criterion.nu.cpu()
            torch.save(save_dict, out_dir / "checkpoints" / f"epoch_{epoch}.pt")

    return all_rows

# %% [markdown]
# ## 8. Run experiments

# %%
def main(selected=None, download_data=False):
    if download_data:
        download_features_if_needed()

    if selected is None:
        configs = EXPERIMENTS
    else:
        selected = set(selected)
        configs = [cfg for i, cfg in enumerate(EXPERIMENTS) if str(i) in selected or cfg["name"] in selected]
        if not configs:
            raise ValueError(f"No experiment matched: {sorted(selected)}")

    for cfg in configs:
        run_experiment(cfg)


# %% [markdown]
# ## 9. Load and plot saved results

# %%
import glob
import matplotlib.pyplot as plt


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_run(name):
    path = pathlib.Path(root_out_dir) / name / f"eval_{name}.jsonl"
    return load_jsonl(path)


def plot_metric(names, metric, title, *, ylabel=None, yscale="linear"):
    plt.figure(figsize=(9, 5))
    for name in names:
        rows = load_run(name)
        xs = [r["epoch"] for r in rows]
        ys = [r[metric] for r in rows]
        plt.plot(xs, ys, marker="o", linewidth=2, markersize=3, label=name)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel or metric)
    plt.yscale(yscale)
    plt.title(title)
    plt.legend(fontsize=8, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Exp 6 inner-STORM experiments.")
    parser.add_argument(
        "--only",
        nargs="*",
        help="Optional experiment indices or names. Example: --only 1 4",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot metrics after running. Best used after all selected configs have finished.",
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download the TreeOfLife feature folder with the original gdown link before running.",
    )
    args = parser.parse_args()
    main(args.only, download_data=args.download_data)
    if args.plot:
        selected = set(args.only or [])
        names = [
            cfg["name"]
            for i, cfg in enumerate(EXPERIMENTS)
            if args.only is None or str(i) in selected or cfg["name"] in selected
        ]
        plot_metric(names, "cross_entropy_loss", "Training Cross-Entropy", ylabel="Cross-entropy")
        plot_metric(names, "val_loss", "Validation Loss", ylabel="Validation loss")
        plot_metric(names, "val_acc", "Validation Accuracy", ylabel="Validation accuracy")
