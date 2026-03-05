"""
single_machine.py — Baseline single-machine training for the MNIST
                    LinearRegressionModel defined in model.py.

Run:
    python single_machine.py
    python single_machine.py --epochs 20 --lr 0.05 --batch-size 128
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse

from model   import create_model, load_model, save_model, \
                    plot_metrics, plot_predictions, plot_confusion
from dataset import get_data_loaders

"""" 
what is epoch - An epoch is one complete pass through the entire training dataset. 
During an epoch, the model sees each training example once and updates its parameters based on the computed loss. 
"""
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def train_single(epochs=10, lr=0.01, batch_size=64, model_path="model.pth"):
    print("=" * 60)
    print("  SINGLE MACHINE TRAINING — LinearRegressionModel (MNIST)")
    print("=" * 60)
    print(f"  Epochs: {epochs}  |  LR: {lr}  |  Batch: {batch_size}\n")

    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    model     = load_model(model_path)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_losses, test_losses             = [], []
    train_accuracies, test_accuracies     = [], []
    epoch_times                           = []   # FIX: record per-epoch time for comparison plot

    global_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        te_loss, te_acc = eval_epoch(model, test_loader,  criterion)

        train_losses.append(tr_loss);       test_losses.append(te_loss)
        train_accuracies.append(tr_acc);    test_accuracies.append(te_acc)
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.2f}% | "
              f"Test Loss: {te_loss:.4f}  Acc: {te_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")

    total_time = time.time() - global_start

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"     Total time      : {total_time:.2f}s  ({total_time/60:.2f} min)")
    print(f"     Final Train Acc : {train_accuracies[-1]:.2f}%")
    print(f"     Final Test  Acc : {test_accuracies[-1]:.2f}%")
    print(f"     Final Test Loss : {test_losses[-1]:.4f}")
    print("=" * 60)

    save_model(model, model_path)

    # Save results as JSON so param_server.py can load them for the comparison chart
    results = {
        "total_time"        : total_time,
        "epoch_times"       : epoch_times,
        "train_losses"      : train_losses,
        "test_losses"       : test_losses,
        "train_accuracies"  : train_accuracies,
        "test_accuracies"   : test_accuracies,
    }
    with open("single_machine_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved -> single_machine_results.json")

    print("\nGenerating performance plots...")
    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)
    plot_predictions(model, test_loader)
    plot_confusion(model, test_loader)

    return {
        "total_time"        : total_time,
        "epoch_times"       : epoch_times,
        "train_losses"      : train_losses,
        "test_losses"       : test_losses,
        "train_accuracies"  : train_accuracies,
        "test_accuracies"   : test_accuracies,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-machine MNIST training baseline")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--lr",         type=float, default=0.01)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--model-path", type=str,   default="model.pth")
    args = parser.parse_args()

    train_single(
        epochs     = args.epochs,
        lr         = args.lr,
        batch_size = args.batch_size,
        model_path = args.model_path,
    )
