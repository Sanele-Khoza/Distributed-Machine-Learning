"""
worker.py — Distributed worker for the MNIST LinearRegressionModel.

Run on each worker machine:
    python worker.py --server 192.168.1.10 --id 1 --num-workers 3
    python worker.py --server 192.168.1.10 --id 2 --num-workers 3
    python worker.py --server 192.168.1.10 --id 3 --num-workers 3

FIX: Removed the unused SGD optimizer from compute_gradients().
     Workers compute gradients only; the parameter server applies the update.
     Previously the optimizer was created and zero_grad()/backward() were called
     but step() was never called — which looked like a bug. Now grad zeroing is
     done explicitly on parameters, making the intent clear.
"""
import torch
import torch.nn as nn
import socket
import pickle
import time
import argparse

from model   import create_model
from dataset import get_worker_loaders



class Worker:
    """
    Each worker:
      1. Connects to the parameter server.
      2. Per epoch: receives global weights -> trains on local shard
                    -> sends gradients back to server.
      NOTE: Workers do NOT call optimizer.step(). The parameter server
            averages all worker gradients and applies the weight update.
    """
    def __init__(self, server_ip: str, port: int,
                 worker_id: int, num_workers: int,
                 lr: float = 0.01, epochs: int = 10,
                 batch_size: int = 64):
        self.server_ip   = server_ip
        self.port        = port
        self.worker_id   = worker_id
        self.num_workers = num_workers
        self.lr          = lr
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.model       = create_model()
        self.criterion   = nn.CrossEntropyLoss()
        self.sock        = None

    # ── Connection ─────────────────────────────────────────────────────────────
    def connect(self, max_retries: int = 15, delay: float = 2.0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(90)
        for attempt in range(1, max_retries + 1):
            try:
                self.sock.connect((self.server_ip, self.port))
                print(f"[Worker {self.worker_id}] Connected to "
                      f"{self.server_ip}:{self.port}")
                return
            except (ConnectionRefusedError, OSError):
                print(f"[Worker {self.worker_id}] Waiting for server "
                      f"({attempt}/{max_retries})...")
                time.sleep(delay)
        raise RuntimeError(
            f"[Worker {self.worker_id}] Could not connect after {max_retries} attempts")

    # ── Low-level I/O ──────────────────────────────────────────────────────────
    def _send(self, data: bytes):
        self.sock.sendall(len(data).to_bytes(4, 'big'))
        self.sock.sendall(data)

    def _recv(self) -> bytes:
        raw  = self.sock.recv(4)
        size = int.from_bytes(raw, 'big')
        buf  = bytearray()
        while len(buf) < size:
            chunk = self.sock.recv(min(65536, size - len(buf)))
            if not chunk: break
            buf.extend(chunk)
        return bytes(buf)

    # ── Per-epoch steps ────────────────────────────────────────────────────────
    def receive_weights(self):
        weights = pickle.loads(self._recv())
        self.model.load_state_dict(weights)

    def compute_gradients(self):
        """
        Train one epoch on the local shard and return accumulated gradients.

        FIX: No optimizer is used here. Gradients are accumulated via backward()
        and then sent to the server. The server applies the SGD update centrally.
        Previously an SGD optimizer was instantiated but step() was never called,
        making the code misleadingly look like a bug.
        """
        train_loader, _ = get_worker_loaders(
            worker_id   = self.worker_id,
            num_workers = self.num_workers,
            batch_size  = self.batch_size,
        )
        total_loss, correct, total = 0.0, 0, 0

        # Zero any existing gradients before accumulation
        for param in self.model.parameters():
            param.grad = None

        self.model.train()
        for images, labels in train_loader:
            outputs = self.model(images)
            loss    = self.criterion(outputs, labels)
            loss.backward()    # accumulate gradients; server will apply step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

        gradients = {
            name: param.grad.clone()
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return gradients, avg_loss, accuracy

    def send_gradients(self, gradients):
        self._send(pickle.dumps(gradients))

    # ── Main training loop ─────────────────────────────────────────────────────
    def train(self):
        self.connect()
        print(f"\n[Worker {self.worker_id}] Shard seed = {42 + self.worker_id}  "
              f"|  Epochs = {self.epochs}\n")

        for epoch in range(1, self.epochs + 1):
            print(f"[Worker {self.worker_id}] -- Epoch {epoch}/{self.epochs}")

            print(f"  Receiving global weights...")
            self.receive_weights()

            print(f"  Training on local shard...")
            t0 = time.time()
            grads, loss, acc = self.compute_gradients()
            elapsed = time.time() - t0
            print(f"  {elapsed:.2f}s  |  Loss: {loss:.4f}  |  Acc: {acc:.2f}%")

            print(f"  Sending gradients to server...")
            self.send_gradients(grads)
            print(f"  Epoch {epoch} complete\n")

        print(f"[Worker {self.worker_id}] All epochs done. Closing connection.")
        self.sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Distributed worker — MNIST LinearRegressionModel")
    parser.add_argument('--server',      required=True,        help='Parameter server IP')
    parser.add_argument('--port',        type=int,   default=8080)
    parser.add_argument('--id',          type=int,   required=True, help='Worker ID (1, 2, 3, ...)')
    parser.add_argument('--num-workers', type=int,   default=3)
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--lr',          type=float, default=0.01)
    parser.add_argument('--batch-size',  type=int,   default=64)
    args = parser.parse_args()

    w = Worker(
        server_ip   = args.server,
        port        = args.port,
        worker_id   = args.id,
        num_workers = args.num_workers,
        lr          = args.lr,
        epochs      = args.epochs,
        batch_size  = args.batch_size,
    )
    w.train()
