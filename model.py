"""
model.py — Model definition and shared plotting utilities for MNIST training.

FIX 1: Removed dead load_data() which duplicated dataset.py and had no fallback.
FIX 2: Added matplotlib.use('Agg') guard for headless environments.
FIX 3: __main__ now uses dataset.get_data_loaders() with fallback support.
FIX 4: plot_metrics, plot_predictions, plot_confusion all improved for clarity.
FIX 5: Added plot_comparison() for side-by-side single vs distributed charts.
"""
import os
import torch
import torch.nn as nn
import numpy as np

import matplotlib
matplotlib.use('Agg')   # FIX: headless-safe; must come before pyplot import
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Model ─────────────────────────────────────────────────────────────────────

class LinearRegressionModel(nn.Module):
    """Linear model for 10-class digit classification (0-9)."""
    def __init__(self, input_size=784, output_size=10):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x.view(-1, 784))


def create_model():
    return LinearRegressionModel()


def save_model(model, path="model.pth"):
    torch.save({"model_state": model.state_dict()}, path)
    print(f"Model saved to {path}")


def load_model(path="model.pth"):
    model = create_model()
    if os.path.exists(path):
        checkpoint = torch.load(path, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Model loaded from {path}")
    else:
        print("No saved model found, starting fresh.")
    return model


# ── Plot palette ───────────────────────────────────────────────────────────────
_SINGLE_TRAIN  = "#7F77DD"   # purple  — single machine train
_SINGLE_TEST   = "#534AB7"   # dark purple — single machine test
_DIST_TRAIN    = "#1D9E75"   # teal    — distributed train (server test loss)
_DIST_TEST     = "#0F6E56"   # dark teal — distributed test

_STYLE = dict(linewidth=2, markersize=5)


# ── Single-run plots ───────────────────────────────────────────────────────────

def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies,
                 title="Single-Machine Training — MNIST",
                 save_path="training_metrics.png"):
    """Loss and accuracy curves for one training run."""
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(epochs, train_losses, "o-",  color=_SINGLE_TRAIN, label="Train loss",  **_STYLE)
    axes[0].plot(epochs, test_losses,  "s--", color=_SINGLE_TEST,  label="Test loss",   **_STYLE)
    axes[0].set_title("Loss per epoch"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, train_accuracies, "o-",  color=_SINGLE_TRAIN, label="Train accuracy", **_STYLE)
    axes[1].plot(epochs, test_accuracies,  "s--", color=_SINGLE_TEST,  label="Test accuracy",  **_STYLE)
    axes[1].set_title("Accuracy per epoch"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)"); axes[1].set_ylim(0, 100)
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_predictions(model, test_loader, save_path="predictions.png"):
    """Show a grid of sample predictions with correct/incorrect colouring."""
    model.eval()
    images, labels = next(iter(test_loader))
    with torch.no_grad():
        _, predicted = torch.max(model(images), 1)

    n = min(18, len(images))
    fig, axes = plt.subplots(3, 6, figsize=(14, 7))
    fig.suptitle("Sample predictions  (green = correct, red = wrong)",
                 fontsize=13, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off"); continue
        img = images[i].squeeze().numpy()
        # synthetic data may have negative values — clip for display
        img = np.clip((img - img.min()) / (img.max() - img.min() + 1e-8), 0, 1)
        ax.imshow(img, cmap="gray")
        correct = predicted[i].item() == labels[i].item()
        color = "#1D9E75" if correct else "#D85A30"
        ax.set_title(f"pred: {predicted[i].item()}\ntrue: {labels[i].item()}",
                     fontsize=9, color=color)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_confusion(model, test_loader, save_path="confusion_matrix.png"):
    """Confusion matrix over the full test set."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            _, predicted = torch.max(model(images), 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    matrix = np.zeros((10, 10), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        matrix[true][pred] += 1

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(range(10)); ax.set_yticklabels(range(10))
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title("Confusion matrix — test set", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax)

    thresh = matrix.max() * 0.5
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                    fontsize=8, color="white" if matrix[i][j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved {save_path}")


# ── Comparison plot: single vs distributed ─────────────────────────────────────

def plot_comparison(single_results: dict, dist_results: dict,
                    save_path="comparison_single_vs_distributed.png"):
    """
    Side-by-side comparison of single-machine vs distributed training.

    single_results keys: train_losses, test_losses, train_accuracies,
                         test_accuracies, total_time, epoch_times (optional)
    dist_results   keys: test_losses, test_accuracies, total_time,
                         epoch_times (optional), num_workers
    """
    n_single = len(single_results["test_losses"])
    n_dist   = len(dist_results["test_losses"])
    ep_s     = range(1, n_single + 1)
    ep_d     = range(1, n_dist   + 1)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Single-Machine  vs  Distributed Training — MNIST LinearRegressionModel",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── Row 0: Loss, Accuracy, Speed ──────────────────────────────────────────
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_acc  = fig.add_subplot(gs[0, 1])
    ax_spd  = fig.add_subplot(gs[0, 2])

    # Loss
    ax_loss.plot(ep_s, single_results["train_losses"], "o-",
                 color=_SINGLE_TRAIN, label="Single — train", **_STYLE)
    ax_loss.plot(ep_s, single_results["test_losses"],  "s--",
                 color=_SINGLE_TEST,  label="Single — test",  **_STYLE)
    ax_loss.plot(ep_d, dist_results["test_losses"],    "^-",
                 color=_DIST_TEST,    label="Distributed — test", **_STYLE)
    ax_loss.set_title("Test loss comparison", fontweight="bold")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.legend(fontsize=8); ax_loss.grid(True, alpha=0.3)

    # Accuracy
    ax_acc.plot(ep_s, single_results["train_accuracies"], "o-",
                color=_SINGLE_TRAIN, label="Single — train", **_STYLE)
    ax_acc.plot(ep_s, single_results["test_accuracies"],  "s--",
                color=_SINGLE_TEST,  label="Single — test",  **_STYLE)
    ax_acc.plot(ep_d, dist_results["test_accuracies"],    "^-",
                color=_DIST_TEST,    label="Distributed — test", **_STYLE)
    ax_acc.set_title("Test accuracy comparison", fontweight="bold")
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 100); ax_acc.legend(fontsize=8); ax_acc.grid(True, alpha=0.3)

    # Speed — total time bar chart
    labels_spd  = ["Single\nmachine", f"Distributed\n({dist_results.get('num_workers',3)} workers)"]
    times_spd   = [single_results["total_time"], dist_results["total_time"]]
    colors_spd  = [_SINGLE_TRAIN, _DIST_TEST]
    bars = ax_spd.bar(labels_spd, times_spd, color=colors_spd, edgecolor="black", width=0.5)
    for bar, t in zip(bars, times_spd):
        ax_spd.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{t:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_spd.set_title("Total training time", fontweight="bold")
    ax_spd.set_ylabel("Seconds"); ax_spd.grid(axis="y", alpha=0.3)

    # ── Row 1: Per-epoch time, Final metric bar, Convergence delta ────────────
    ax_ept  = fig.add_subplot(gs[1, 0])
    ax_bar  = fig.add_subplot(gs[1, 1])
    ax_delt = fig.add_subplot(gs[1, 2])

    # Per-epoch time (line if available, else skip)
    et_s = single_results.get("epoch_times", [])
    et_d = dist_results.get("epoch_times",   [])
    if et_s and et_d:
        ax_ept.plot(range(1, len(et_s)+1), et_s, "o-",  color=_SINGLE_TRAIN,
                    label="Single",      **_STYLE)
        ax_ept.plot(range(1, len(et_d)+1), et_d, "^--", color=_DIST_TEST,
                    label="Distributed", **_STYLE)
        ax_ept.set_title("Time per epoch", fontweight="bold")
        ax_ept.set_xlabel("Epoch"); ax_ept.set_ylabel("Seconds")
        ax_ept.legend(fontsize=8); ax_ept.grid(True, alpha=0.3)
    else:
        ax_ept.text(0.5, 0.5, "Epoch timing\nnot recorded",
                    ha="center", va="center", transform=ax_ept.transAxes,
                    fontsize=12, color="gray")
        ax_ept.set_title("Time per epoch", fontweight="bold")
        ax_ept.axis("off")

    # Final accuracy grouped bar
    final_metrics = {
        "Train acc (%)": [single_results["train_accuracies"][-1], None],
        "Test acc (%)":  [single_results["test_accuracies"][-1],
                          dist_results["test_accuracies"][-1]],
        "Test loss":     [single_results["test_losses"][-1],
                          dist_results["test_losses"][-1]],
    }
    x     = np.arange(len(final_metrics))
    width = 0.32
    metric_labels = list(final_metrics.keys())
    s_vals = [v[0] for v in final_metrics.values()]
    d_vals = [v[1] for v in final_metrics.values()]

    bars_s = ax_bar.bar(x - width/2, s_vals, width, color=_SINGLE_TRAIN,
                        label="Single", edgecolor="black")
    bars_d = ax_bar.bar(x + width/2,
                        [v if v is not None else 0 for v in d_vals],
                        width, color=_DIST_TEST, label="Distributed", edgecolor="black")

    for bar, val in zip(bars_s, s_vals):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for bar, val in zip(bars_d, d_vals):
        if val is None: continue
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # Mark N/A for distributed train acc
    ax_bar.text(x[0] + width/2, 1, "N/A", ha="center", va="bottom",
                fontsize=8, color="gray")

    ax_bar.set_title("Final metrics — side by side", fontweight="bold")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(metric_labels, fontsize=9)
    ax_bar.legend(fontsize=9); ax_bar.grid(axis="y", alpha=0.3)

    # Accuracy gap per epoch (single test - distributed test, aligned to shorter run)
    n_common = min(n_single, n_dist)
    delta = [single_results["test_accuracies"][i] - dist_results["test_accuracies"][i]
             for i in range(n_common)]
    colors_delta = [_SINGLE_TRAIN if d > 0 else _DIST_TEST for d in delta]
    ax_delt.bar(range(1, n_common+1), delta, color=colors_delta, edgecolor="black", linewidth=0.5)
    ax_delt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_delt.set_title("Accuracy gap per epoch\n(Single − Distributed, %)", fontweight="bold")
    ax_delt.set_xlabel("Epoch"); ax_delt.set_ylabel("Accuracy difference (%)")
    ax_delt.grid(axis="y", alpha=0.3)

    # Annotation: purple = single leads, teal = distributed leads
    ax_delt.text(0.02, 0.96, "Purple = single leads",  transform=ax_delt.transAxes,
                 fontsize=8, color=_SINGLE_TRAIN, va="top")
    ax_delt.text(0.02, 0.88, "Teal   = distributed leads", transform=ax_delt.transAxes,
                 fontsize=8, color=_DIST_TEST,    va="top")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # FIX: use dataset.get_data_loaders() so the synthetic fallback works
    from dataset import get_data_loaders
    import torch.optim as optim

    EPOCHS     = 10
    BATCH_SIZE = 64
    LR         = 0.01
    MODEL_PATH = "model.pth"

    print("Loading dataset...")
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)

    model     = load_model(MODEL_PATH)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR)

    train_losses, test_losses         = [], []
    train_accuracies, test_accuracies = [], []

    print(f"\nTraining for {EPOCHS} epochs...\n")
    for epoch in range(1, EPOCHS + 1):
        # train
        model.train()
        tot_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            tot_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item(); total += labels.size(0)
        tr_loss, tr_acc = tot_loss / len(train_loader), 100 * correct / total
        train_losses.append(tr_loss); train_accuracies.append(tr_acc)

        # evaluate
        model.eval()
        tot_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                tot_loss += criterion(outputs, labels).item()
                _, pred = torch.max(outputs, 1)
                correct += (pred == labels).sum().item(); total += labels.size(0)
        te_loss, te_acc = tot_loss / len(test_loader), 100 * correct / total
        test_losses.append(te_loss); test_accuracies.append(te_acc)

        print(f"Epoch {epoch:02d} | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.2f}% | "
              f"Test Loss: {te_loss:.4f} Acc: {te_acc:.2f}%")

    save_model(model, MODEL_PATH)
    print("\nGenerating plots...")
    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)
    plot_predictions(model, test_loader)
    plot_confusion(model, test_loader)
    print("Done!")
