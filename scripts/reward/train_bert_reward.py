import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_default_hf_home = str(Path.cwd() / "hf_cache")
os.environ.setdefault("HF_HOME", _default_hf_home)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(_default_hf_home) / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(_default_hf_home) / "transformers"))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(_default_hf_home) / "datasets"))


TASKS = {
    "ag_news": {
        "dataset": ("ag_news", None),
        "text_col": "text",
        "label_col": "label",
        "num_labels": 4,
        "target_label_id": 1,  # Sports in HF ag_news
        "target_label_name": "Sports",
        "make_val_from_train": True,
        "val_fraction": 0.02,
        "score_metric": "macro_f1",
    },
    "cola": {
        "dataset": ("glue", "cola"),
        "text_col": "sentence",
        "label_col": "label",
        "num_labels": 2,
        "target_label_id": 1,  # acceptable
        "target_label_name": "acceptable",
        "make_val_from_train": False,
        "score_metric": "mcc",
    },
    "imdb": {
        "dataset": ("imdb", None),
        "text_col": "text",
        "label_col": "label",
        "num_labels": 2,
        "target_label_id": 1,  # positive
        "target_label_name": "positive",
        "make_val_from_train": True,
        "val_fraction": 0.02,
        "score_metric": "macro_f1",
    },
    "tweet_offensive": {
        "dataset": ("tweet_eval", "offensive"),
        "text_col": "text",
        "label_col": "label",
        "num_labels": 2,
        "target_label_id": 0,  # non-offensive in TweetEval
        "target_label_name": "non-offensive",
        "make_val_from_train": False,
        "score_metric": "macro_f1",
    },
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def macro_f1_score(y_true, y_pred, num_labels: int) -> float:
    f1s = []
    for c in range(num_labels):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1s.append(f1)
    return float(np.mean(f1s))


def matthews_corrcoef_binary(y_true, y_pred) -> float:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    return float((tp * tn - fp * fn) / denom)


def load_task_dataset(task: str, seed: int):
    spec = TASKS[task]
    name, config = spec["dataset"]

    if config is None:
        ds = load_dataset(name)
    else:
        ds = load_dataset(name, config)

    if spec["make_val_from_train"]:
        split = ds["train"].train_test_split(
            test_size=spec.get("val_fraction", 0.02),
            seed=seed,
            stratify_by_column=spec["label_col"],
        )
        train_ds = split["train"]
        val_ds = split["test"]
    else:
        train_ds = ds["train"]
        if "validation" in ds:
            val_ds = ds["validation"]
        else:
            val_ds = ds["test"]

    return train_ds, val_ds, ds


def tokenize_dataset(dataset, tokenizer, text_col: str, label_col: str, max_length: int):
    def preprocess(batch):
        out = tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_token_type_ids=False,
        )
        out["labels"] = batch[label_col]
        return out

    keep_cols = [text_col, label_col]
    remove_cols = [c for c in dataset.column_names if c not in keep_cols]
    tok = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    return tok


@torch.no_grad()
def evaluate(model, dataloader, device, num_labels: int):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_n = 0

    for batch in dataloader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch, labels=labels)
        logits = out.logits
        loss = out.loss

        preds = logits.argmax(dim=-1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        bs = labels.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    acc = float((y_pred == y_true).mean())
    macro_f1 = macro_f1_score(y_true, y_pred, num_labels)
    mcc = matthews_corrcoef_binary(y_true, y_pred) if num_labels == 2 else 0.0
    loss = total_loss / max(total_n, 1)

    model.train()
    return {
        "eval_loss": loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "mcc": mcc,
    }


def save_model(model, tokenizer, out_dir: str, args, task_info, metrics, label_names):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    cfg = {
        "task": args.task,
        "base_model": args.base_model,
        "vocab_size": len(tokenizer),
        "max_length": args.max_length,
        "num_labels": task_info["num_labels"],
        "label_names": label_names,
        "target_label_id": task_info["target_label_id"],
        "target_label_name": task_info["target_label_name"],
        "metrics": metrics,
        "note": "This model is used as differentiable guidance reward only. Final paper rewards should be computed by independent verifiers.",
    }
    with open(out / "reward_config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS.keys()))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_model", default="bert-base-uncased")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    task_info = TASKS[args.task]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    print("task:", args.task)
    print("task_info:", task_info)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        local_files_only=args.local_files_only,
    )

    # Keep BERT vocab exactly aligned with LM1B-DiT endpoint.
    assert len(tokenizer) == 30522, f"Expected bert-base-uncased vocab_size=30522, got {len(tokenizer)}"

    train_raw, val_raw, full_ds = load_task_dataset(args.task, args.seed)

    label_feature = train_raw.features[task_info["label_col"]]
    if hasattr(label_feature, "names") and label_feature.names is not None:
        label_names = list(label_feature.names)
    else:
        label_names = [str(i) for i in range(task_info["num_labels"])]

    print("label_names:", label_names)
    print("target:", task_info["target_label_id"], task_info["target_label_name"])
    print("train size:", len(train_raw), "val size:", len(val_raw))

    train_ds = tokenize_dataset(
        train_raw,
        tokenizer,
        task_info["text_col"],
        task_info["label_col"],
        args.max_length,
    )
    val_ds = tokenize_dataset(
        val_raw,
        tokenizer,
        task_info["text_col"],
        task_info["label_col"],
        args.max_length,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=task_info["num_labels"],
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = math.ceil(len(train_loader) * args.epochs / args.grad_accum)

    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device == "cuda"))

    best_score = -1e9
    best_metrics = {}
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    print("total_steps:", total_steps, "warmup_steps:", warmup_steps)

    for epoch in range(args.epochs):
        print(f"\n===== epoch {epoch + 1}/{args.epochs} =====")

        for batch_idx, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=(args.fp16 and device == "cuda")):
                out = model(**batch, labels=labels)
                loss = out.loss / args.grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 50 == 0:
                    print(f"step={global_step}/{total_steps} loss={loss.item() * args.grad_accum:.4f}")

                if global_step % args.eval_every == 0 or global_step == total_steps:
                    metrics = evaluate(model, val_loader, device, task_info["num_labels"])
                    score_name = task_info["score_metric"]
                    score = metrics[score_name]
                    print(f"[eval step={global_step}] {metrics}, score={score_name}:{score:.4f}")

                    if score > best_score:
                        best_score = score
                        best_metrics = metrics
                        print("new best, saving to", args.out_dir)
                        save_model(model, tokenizer, args.out_dir, args, task_info, metrics, label_names)

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    metrics = evaluate(model, val_loader, device, task_info["num_labels"])
    score_name = task_info["score_metric"]
    score = metrics[score_name]
    print(f"[final eval] {metrics}, score={score_name}:{score:.4f}")

    if score > best_score:
        best_score = score
        best_metrics = metrics
        save_model(model, tokenizer, args.out_dir, args, task_info, metrics, label_names)

    print("best_score:", best_score)
    print("best_metrics:", best_metrics)
    print("saved:", args.out_dir)


if __name__ == "__main__":
    main()
