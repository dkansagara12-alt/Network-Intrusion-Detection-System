"""
Training loop for the FT-Transformer NIDS classifier.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from . import config
from .ft_transformer import FTTransformer


def set_seed(seed: int):
    """Seed every RNG that affects training so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast():
    """bf16 mixed precision on GPU when config.USE_AMP is set; a no-op otherwise."""
    device_type = config.DEVICE.type
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                          enabled=config.USE_AMP and device_type == "cuda")


def _batches(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool):
    """
    Yields mini-batches by slicing tensors that already live on the device.
    Much faster than a per-row Dataset/DataLoader for millions of rows.
    """
    n = len(X)
    order = torch.randperm(n, device=X.device) if shuffle else torch.arange(n, device=X.device)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield X[idx], y[idx]


def class_weights(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Per-class weights shared by every model (FT-Transformer loss, baselines'
    sample weights), chosen by config.CLASS_WEIGHTING[config.TASK]:
      "balanced" - n / (k * n_c), the same as sklearn's class_weight="balanced"
      "sqrt"     - square root of that, a milder correction for very rare classes
      "none"     - all ones
    """
    mode = config.CLASS_WEIGHTING[config.TASK]
    counts = np.bincount(y_train, minlength=num_classes)
    weights = counts.sum() / (num_classes * np.maximum(counts, 1))
    if mode == "sqrt":
        weights = np.sqrt(weights)
    elif mode == "none":
        weights = np.ones(num_classes)
    return weights.astype(np.float32)


def train_model(data: dict, verbose: bool = True, seed: int = config.RANDOM_STATE):
    set_seed(seed)
    device = config.DEVICE
    num_features = data["X_train"].shape[1]
    num_classes = len(data["class_names"])
    print(f"[train] Device: {device}"
          + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    model = FTTransformer(
        num_features=num_features,
        d_token=config.FT_TOKEN_DIM,
        num_layers=config.FT_NUM_LAYERS,
        num_heads=config.FT_NUM_HEADS,
        ffn_dim=config.FT_FFN_DIM,
        dropout=config.FT_DROPOUT,
        attn_dropout=config.FT_ATTN_DROPOUT,
        num_classes=num_classes,
        cat_groups=data.get("cat_groups", []),
    ).to(device)
    print(f"[train] {model.tokenizer.num_tokens} feature tokens "
          f"({len(data.get('cat_groups', []))} categorical), "
          f"{sum(p.numel() for p in model.parameters()):,} parameters")

    X_train = torch.from_numpy(data["X_train"]).float().to(device)
    y_train = torch.from_numpy(data["y_train"]).long().to(device)
    X_val = torch.from_numpy(data["X_val"]).float().to(device)
    y_val = torch.from_numpy(data["y_val"]).long().to(device)
    num_train_batches = (len(X_train) + config.BATCH_SIZE - 1) // config.BATCH_SIZE

    class_w = torch.from_numpy(class_weights(data["y_train"], num_classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                   weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_path = os.path.join(config.CHECKPOINT_DIR, f"ft_transformer_seed{seed}.pt")

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        loop = tqdm(_batches(X_train, y_train, config.BATCH_SIZE, shuffle=True),
                    total=num_train_batches,
                    desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [train]",
                    disable=not verbose)
        for X_batch, y_batch in loop:
            optimizer.zero_grad()
            with autocast():
                logits = model(X_batch)
            loss = criterion(logits.float(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(X_train)

        # validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in _batches(X_val, y_val, config.EVAL_BATCH_SIZE, shuffle=False):
                with autocast():
                    logits = model(X_batch)
                logits = logits.float()
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()

        val_loss /= len(X_val)
        val_acc = correct / len(X_val)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        if verbose:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"[train] Early stopping at epoch {epoch}")
                break

    # load best checkpoint before returning
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, history


if __name__ == "__main__":
    from .data_preprocessing import load_dataset
    data = load_dataset()
    model, history = train_model(data)
    print(f"Training complete. Best model saved to {config.CHECKPOINT_DIR}")
