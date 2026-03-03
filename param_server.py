"""
param_server.py — Parameter server for distributed training of the MNIST
                  LinearRegressionModel defined in model.py.

Run (on server machine):
    python param_server.py
    python param_server.py --port 8080 --num-workers 3 --epochs 10 --lr 0.01

FIX: _collect_gradients() now logs exactly which worker IDs are missing on timeout,
     rather than printing a generic count. This makes debugging dropped workers easy.
FIX: epoch_times is now recorded and returned so plot_comparison() can use it.
FIX: Calls model.plot_comparison() at the end to generate the single vs distributed
     comparison chart (loads single_machine results from a JSON sidecar if present).
"""
import torch
import torch.nn as nn
import socket
import time
import threading
import pickle
import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

from model   import create_model, save_model, plot_comparison
from dataset import get_data_loaders


# ── Live-plot state ────────────────────────────────────────────────────────────
_loss_hist = deque(maxlen=100)
_acc_hist  = deque(maxlen=100)

def _save_live_plot(epoch):
    if len(_loss_hist) < 2:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(_loss_hist) + 1)
    ax1.plot(ep, list(_loss_hist), 'b-o', markersize=4)
    ax1.set_title('Training Loss (server-side)')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Cross-Entropy Loss')
    ax1.grid(True, alpha=0.4)

    ax2.plot(ep, list(_acc_hist), 'g-s', markersize=4)
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim(0, 100); ax2.grid(True, alpha=0.4)

    plt.suptitle(f'Distributed Training — Epoch {epoch}', fontweight='bold')
    plt.tight_layout()
    plt.savefig('distributed_training_live.png', dpi=100)
    plt.close(fig)


def _get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80)); return s.getsockname()[0]
    except Exception:               return '127.0.0.1'
    finally:                        s.close()


# ── Parameter Server ───────────────────────────────────────────────────────────
class ParameterServer:
    def __init__(self, port=8080, num_workers=3, epochs=10, lr=0.01):
        self.port        = port
        self.num_workers = num_workers
        self.epochs      = epochs
        self.lr          = lr
        self.model       = create_model()
        self.criterion   = nn.CrossEntropyLoss()
        self.workers     = []
        self.gradients   = {}
        self.lock        = threading.Lock()

    # ── Low-level I/O ──────────────────────────────────────────────────────────
    @staticmethod
    def _send(sock, data: bytes):
        sock.sendall(len(data).to_bytes(4, 'big'))
        sock.sendall(data)

    @staticmethod
    def _recv(sock) -> bytes:
        raw  = sock.recv(4)
        if not raw: return b''
        size = int.from_bytes(raw, 'big')
        buf  = bytearray()
        while len(buf) < size:
            chunk = sock.recv(min(65536, size - len(buf)))
            if not chunk: break
            buf.extend(chunk)
        return bytes(buf)

    # ── Worker listener thread ─────────────────────────────────────────────────
    def _handle_worker(self, sock, worker_id):
        try:
            while True:
                data = self._recv(sock)
                if not data: break
                grad = pickle.loads(data)
                with self.lock:
                    self.gradients[worker_id] = grad
        except Exception as e:
            print(f"  Warning: Worker {worker_id} thread error: {e}")
        finally:
            sock.close()

    # ── Gradient helpers ───────────────────────────────────────────────────────
    def _broadcast_weights(self):
        payload = pickle.dumps(self.model.state_dict())
        for sock in self.workers:
            try: self._send(sock, payload)
            except Exception as e: print(f"  Warning: Broadcast error: {e}")

    def _collect_gradients(self, timeout=120):
        self.gradients.clear()
        expected = set(range(1, self.num_workers + 1))
        deadline = time.time() + timeout
        while len(self.gradients) < self.num_workers:
            time.sleep(0.05)
            if time.time() > deadline:
                # FIX: report exactly which workers are missing, not just a count
                missing = expected - set(self.gradients.keys())
                print(f"  Warning: Timeout — missing gradients from workers: {sorted(missing)}")
                break
        received = [self.gradients[i] for i in range(1, self.num_workers + 1)
                    if i in self.gradients]
        print(f"    Gradients from {len(received)}/{self.num_workers} workers")
        return received

    def _average_and_update(self, gradients):
        if not gradients: return
        avg = {k: sum(g[k] for g in gradients) / len(gradients) for k in gradients[0]}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in avg:
                    param -= self.lr * avg[name]

    # ── Server-side evaluation ─────────────────────────────────────────────────
    def _evaluate(self):
        _, test_loader = get_data_loaders(batch_size=128)
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                out = self.model(images)
                total_loss += self.criterion(out, labels).item()
                _, pred = torch.max(out, 1)
                correct += (pred == labels).sum().item()
                total   += labels.size(0)
        return total_loss / len(test_loader), 100.0 * correct / total

    # ── Main ───────────────────────────────────────────────────────────────────
    def start(self):
        print("=" * 55)
        print("  PARAMETER SERVER  —  LinearRegressionModel (MNIST)")
        print("=" * 55)
        print(f"  IP        : {_get_ip()}")
        print(f"  Port      : {self.port}")
        print(f"  Workers   : {self.num_workers}")
        print(f"  Epochs    : {self.epochs}  |  LR: {self.lr}")
        print("=" * 55)

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(('0.0.0.0', self.port))
        srv.listen(self.num_workers)
        print(f"\nWaiting for {self.num_workers} workers...\n")

        for i in range(self.num_workers):
            conn, addr = srv.accept()
            self.workers.append(conn)
            print(f"  Worker {i+1} connected from {addr}")
            threading.Thread(target=self._handle_worker,
                             args=(conn, i + 1), daemon=True).start()

        print("\n" + "=" * 55)
        print("  ALL WORKERS CONNECTED — TRAINING STARTS")
        print("=" * 55 + "\n")

        global_start    = time.time()
        test_losses_log = []
        test_acc_log    = []
        epoch_times_log = []   # FIX: record per-epoch timing for comparison chart

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            print(f"\nEPOCH {epoch}/{self.epochs}")
            print("-" * 45)

            print("  Broadcasting weights...")
            self._broadcast_weights()

            print("  Collecting gradients...")
            grads = self._collect_gradients()

            print("  Averaging & updating model...")
            self._average_and_update(grads)

            print("  Evaluating on test set...")
            te_loss, te_acc = self._evaluate()
            epoch_time = time.time() - t0

            test_losses_log.append(te_loss)
            test_acc_log.append(te_acc)
            epoch_times_log.append(epoch_time)
            _loss_hist.append(te_loss)
            _acc_hist.append(te_acc)

            print(f"  Test Loss: {te_loss:.4f}  |  "
                  f"Test Acc: {te_acc:.2f}%  |  Time: {epoch_time:.2f}s")
            _save_live_plot(epoch)

        total_time = time.time() - global_start

        print("\n" + "=" * 55)
        print("  DISTRIBUTED TRAINING COMPLETE")
        print(f"     Total time     : {total_time:.2f}s  ({total_time/60:.2f} min)")
        print(f"     Final Test Acc : {test_acc_log[-1]:.2f}%")
        print(f"     Final Test Loss: {test_losses_log[-1]:.4f}")
        print("=" * 55)

        save_model(self.model, "model_distributed.pth")

        # Save distributed results as JSON sidecar for plot_comparison
        dist_results = {
            "test_losses"      : test_losses_log,
            "test_accuracies"  : test_acc_log,
            "epoch_times"      : epoch_times_log,
            "total_time"       : total_time,
            "num_workers"      : self.num_workers,
        }
        with open("distributed_results.json", "w") as f:
            json.dump(dist_results, f, indent=2)
        print("  Results saved -> distributed_results.json")

        # Generate comparison plot if single_machine_results.json exists
        try:
            with open("single_machine_results.json") as f:
                single_results = json.load(f)
            plot_comparison(single_results, dist_results,
                            save_path="comparison_single_vs_distributed.png")
            print("  Comparison plot -> comparison_single_vs_distributed.png")
        except FileNotFoundError:
            print("  (No single_machine_results.json found — skipping comparison plot)")
            print("  Tip: run single_machine.py first, then param_server.py for the full comparison.")

        self._plot_summary(test_losses_log, test_acc_log, epoch_times_log)

        for sock in self.workers:
            try: sock.close()
            except: pass
        srv.close()

    def _plot_summary(self, losses, accuracies, epoch_times):
        epochs = range(1, len(losses) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Distributed Training Summary — LinearRegressionModel (MNIST)",
                     fontsize=13, fontweight='bold')

        axes[0].plot(epochs, losses, 'b-o', linewidth=2, markersize=5)
        axes[0].set_title("Test Loss per Epoch")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
        axes[0].grid(True, alpha=0.4)

        axes[1].plot(epochs, accuracies, 'g-s', linewidth=2, markersize=5)
        axes[1].set_title("Test Accuracy per Epoch")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_ylim(0, 100); axes[1].grid(True, alpha=0.4)

        axes[2].bar(list(epochs), epoch_times, color='#2196F3', edgecolor='black')
        axes[2].set_title("Time per Epoch")
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Seconds")
        axes[2].grid(axis='y', alpha=0.4)

        plt.tight_layout()
        plt.savefig("distributed_training_summary.png", dpi=150)
        plt.close(fig)
        print("  Summary plot -> distributed_training_summary.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter server for distributed MNIST training")
    parser.add_argument('--port',        type=int,   default=8080)
    parser.add_argument('--num-workers', type=int,   default=3)
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--lr',          type=float, default=0.01)
    args = parser.parse_args()

    ps = ParameterServer(port=args.port, num_workers=args.num_workers,
                         epochs=args.epochs, lr=args.lr)
    ps.start()
