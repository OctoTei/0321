#!/usr/bin/env python3
# fedml_trainer.py - FedTGP Trainer based on FedML Framework

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys
import os

# Add FedML to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from fedml.core.alg_frame.client_trainer import ClientTrainer
from fedtgp_loss import FedTGPClientLoss
from data_loader import FedTGPDataset, get_dataset_info

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not installed. Install with: pip install wandb")


class FedTGPClientTrainer(ClientTrainer):
    """FedTGP Client Trainer based on FedML Framework"""

    def __init__(self, model, args):
        super().__init__(model, args)

        if hasattr(args, 'device'):
            self.device = args.device
        else:
            device_config = getattr(args, 'using_gpu', 'cpu')
            if torch.cuda.is_available() and device_config != 'cpu':
                if isinstance(device_config, bool):
                    self.device = torch.device('cuda:0' if device_config else 'cpu')
                else:
                    self.device = torch.device(device_config)
            else:
                self.device = torch.device('cpu')

        self.model = model.to(self.device)
        self.args = args

        self.lamda = getattr(args, 'lamda', 1.0)
        self.global_protos = {}
        self.local_protos = {}
        self.fedtgp_loss = FedTGPClientLoss(lamda=self.lamda)

        self.train_loss = 0.0
        self.train_acc = 0.0
        self.current_round = 0

        self.use_wandb = getattr(args, 'enable_wandb', False) and WANDB_AVAILABLE
        if self.use_wandb:
            client_id = getattr(args, 'client_id', 0)
            feature_extractor = getattr(args, 'feature_extractor', 'unknown')
            classifier = getattr(args, 'classifier', 'unknown')
            wandb.init(
                project=getattr(args, 'wandb_project', 'fedtgp'),
                name=f"client_{client_id}_{feature_extractor}",
                config={
                    'client_id': client_id,
                    'feature_extractor': feature_extractor,
                    'classifier': classifier,
                    'learning_rate': getattr(args, 'learning_rate', 0.01),
                    'batch_size': getattr(args, 'batch_size', 64),
                    'epochs': getattr(args, 'epochs', 5),
                    'lamda': self.lamda,
                    'feature_dim': getattr(args, 'feature_dim', 512),
                },
                reinit=True,
            )
            logging.info(f"Wandb initialized for client {client_id}")

        logging.info(f"FedTGP trainer initialized with lamda={self.lamda}")

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
        self.model.to(self.device)

    def set_global_protos(self, global_protos):
        if global_protos:
            self.global_protos = {k: v.to(self.device) for k, v in global_protos.items()}
            logging.info(f"FedTGP Client: Updated global prototypes for {len(global_protos)} classes")
        else:
            self.global_protos = {}
            logging.info("FedTGP Client: Cleared global prototypes (first round)")

    def update_global_prototypes(self, global_protos):
        self.set_global_protos(global_protos)

    def train_and_extract_prototypes(self):
        self.current_round += 1

        print("=" * 60)
        print(f"Client Training - Round {self.current_round}")
        print(f"{'=' * 60}")

        logging.info(f"FedTGP Client: Round {self.current_round}, Using LOCAL model weights")
        logging.info(f"FedTGP Client: Model device: {self.device}, Lambda: {self.lamda}")

        for handler in logging.root.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()

        self.model.to(self.device)
        self.model.train()

        if hasattr(self, 'train_data_local'):
            train_loader = self.train_data_local
        else:
            logging.warning("No local training data set")
            return {}

        optimizer = optim.SGD(
            self.model.parameters(),
            lr=getattr(self.args, 'learning_rate', 0.01),
            momentum=0.9,
            weight_decay=getattr(self.args, 'weight_decay', 1e-4),
        )

        epochs = getattr(self.args, 'epochs', 5)
        total_loss = 0.0
        total_ce_loss = 0.0
        total_proto_loss = 0.0
        total_batches = 0
        total_correct = 0
        total_samples = 0
        skipped_small_batches = 0

        actual_batch_size = getattr(train_loader, 'batch_size', getattr(self.args, 'batch_size', 64))
        print(f"Training for {epochs} epochs...")
        print(f"Lambda (prototype alignment weight): {self.lamda}")
        print(f"Learning rate: {getattr(self.args, 'learning_rate', 0.01)}")
        print(f"Batch size (actual loader): {actual_batch_size}")
        print(f"{'-' * 60}")

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_proto_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            num_batches = 0

            for batch_idx, (data, labels) in enumerate(train_loader):
                # 关键修改：保留最后一个小 batch，但训练时跳过 size<2 的 batch，避免 BatchNorm 报错。
                if len(labels) < 2:
                    skipped_small_batches += 1
                    logging.warning(
                        f"Skipping small training batch with size={len(labels)} "
                        f"at epoch {epoch + 1}, batch {batch_idx}"
                    )
                    continue

                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                features = self.model.base(data)
                if features.dim() > 2:
                    features = features.view(features.size(0), -1)

                logits = self.model.head(features)
                loss, ce_loss, proto_loss = self.fedtgp_loss(
                    logits, labels, features, self.global_protos
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    logging.error(f"NaN/Inf detected in client training at epoch {epoch + 1}, batch {batch_idx}")
                    logging.error(f"  CE loss: {ce_loss.item()}")
                    logging.error(f"  Proto loss: {proto_loss.item()}")
                    logging.error(f"  Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
                    logging.error(f"  Features range: [{features.min().item():.4f}, {features.max().item():.4f}]")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                pred = logits.argmax(dim=1)
                correct = pred.eq(labels).sum().item()
                epoch_correct += correct
                epoch_samples += len(labels)

                epoch_loss += loss.item()
                epoch_ce_loss += ce_loss.item()
                epoch_proto_loss += proto_loss.item()
                num_batches += 1

            total_loss += epoch_loss
            total_ce_loss += epoch_ce_loss
            total_proto_loss += epoch_proto_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
            total_batches += num_batches

            avg_loss = epoch_loss / max(num_batches, 1)
            avg_ce = epoch_ce_loss / max(num_batches, 1)
            avg_proto = epoch_proto_loss / max(num_batches, 1)
            avg_acc = epoch_correct / max(epoch_samples, 1)

            print(f"Epoch [{epoch + 1}/{epochs}]:")
            print(f"  Total Loss:     {avg_loss:.6f}")
            print(f"  ├─ CE Loss:     {avg_ce:.6f}")
            print(f"  └─ Proto Loss:  {avg_proto:.6f}")
            print(f"  Accuracy:       {avg_acc * 100:.2f}% ({epoch_correct}/{epoch_samples})")

            logging.info(
                f"Client Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, "
                f"CE: {avg_ce:.6f}, Proto: {avg_proto:.6f}, Acc: {avg_acc:.4f}"
            )

            if self.use_wandb:
                wandb.log({
                    'round': self.current_round,
                    'epoch': epoch + 1,
                    'train/loss': avg_loss,
                    'train/ce_loss': avg_ce,
                    'train/proto_loss': avg_proto,
                    'train/accuracy': avg_acc,
                })

        print(f"{'-' * 60}")
        print("Extracting local prototypes...")

        if hasattr(self, 'train_data_local'):
            original_dataset = self.train_data_local.dataset
            original_data = original_dataset.data
            original_labels = original_dataset.labels

            dataset_name = getattr(self.args, 'dataset', 'cifar10')
            dataset_info = get_dataset_info(dataset_name)
            no_augment_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(dataset_info['mean'], dataset_info['std'])
            ])

            proto_dataset = FedTGPDataset(
                original_data,
                original_labels,
                transform=no_augment_transform,
            )

            proto_batch_size = getattr(self.train_data_local, 'batch_size', getattr(self.args, 'batch_size', 10))
            proto_loader = DataLoader(
                proto_dataset,
                batch_size=proto_batch_size,
                shuffle=False,
                drop_last=False,
            )

            logging.info(
                f"Created prototype extraction loader without data augmentation "
                f"({len(proto_dataset)} samples, batch_size={proto_batch_size})"
            )
            local_protos = self._collect_local_prototypes(proto_loader)
        else:
            logging.error("No training data available for prototype extraction")
            local_protos = {}

        print(f"✓ Extracted {len(local_protos)} local prototypes")
        logging.info(f"Extracted {len(local_protos)} local prototypes")
        if skipped_small_batches > 0:
            logging.info(f"Skipped {skipped_small_batches} small training batches with size < 2")

        self.train_loss = total_loss / max(total_batches, 1)
        self.train_acc = total_correct / max(total_samples, 1)

        print(f"{'-' * 60}")
        print("Evaluating on test set...")
        test_loss, test_acc, test_correct, test_total = self._evaluate_on_test_set()
        print("✓ Test evaluation completed")

        logging.info(f"Test results: loss={test_loss:.6f}, acc={test_acc:.4f} ({test_correct}/{test_total})")

        print(f"{'-' * 60}")
        print(f"Round {self.current_round} Summary:")
        print(f"  Train Loss:     {self.train_loss:.6f}")
        print(f"  Train Acc:      {self.train_acc * 100:.2f}%")
        print(f"  Test Loss:      {test_loss:.6f}")
        print(f"  Test Acc:       {test_acc * 100:.2f}% ({test_correct}/{test_total})")
        print(f"  Prototypes:     {len(local_protos)} classes")
        print(f"{'=' * 60}\n")

        for handler in logging.root.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()

        if self.use_wandb:
            wandb.log({
                'round': self.current_round,
                'train/round_loss': self.train_loss,
                'train/round_accuracy': self.train_acc,
                'test/loss': test_loss,
                'test/accuracy': test_acc,
                'train/num_prototypes': len(local_protos),
            })

        return local_protos

    def _evaluate_on_test_set(self):
        self.model.eval()

        test_loss = 0.0
        test_correct = 0
        test_total = 0
        num_batches = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, labels in self.test_data_local:
                data, labels = data.to(self.device), labels.to(self.device)
                features = self.model.base(data)
                if features.dim() > 2:
                    features = features.view(features.size(0), -1)
                logits = self.model.head(features)
                loss = criterion(logits, labels)
                test_loss += loss.item()

                pred = logits.argmax(dim=1)
                correct = pred.eq(labels).sum().item()
                test_correct += correct
                test_total += len(labels)
                num_batches += 1

        avg_test_loss = test_loss / max(num_batches, 1)
        test_acc = test_correct / max(test_total, 1)
        self.model.train()
        return avg_test_loss, test_acc, test_correct, test_total

    def set_client_data(self, client_idx, train_data_local, test_data_local, local_data_num):
        self.client_idx = client_idx
        self.train_data_local = train_data_local
        self.test_data_local = test_data_local
        self.local_sample_number = local_data_num

        batch_size = getattr(self.train_data_local, 'batch_size', getattr(self.args, 'batch_size', 10))
        if local_data_num < 2:
            logging.warning(f"⚠️ Client {client_idx} has only {local_data_num} samples, training should be skipped")
        elif local_data_num < batch_size:
            logging.warning(
                f"⚠️ Client {client_idx} has only {local_data_num} samples, less than batch_size={batch_size}; "
                f"loader batch_size has been reduced dynamically"
            )
        elif local_data_num % batch_size == 1:
            logging.warning(
                f"⚠️ Client {client_idx} will have one final batch of size 1 with batch_size={batch_size}; "
                f"that singleton batch will be skipped during training"
            )
        elif local_data_num < batch_size * 2:
            logging.warning(
                f"⚠️ Client {client_idx} has only {local_data_num} samples, which gives very few batches for batch_size={batch_size}"
            )

        logging.info(f"Client {client_idx} data set: {local_data_num} samples")

    def get_local_protos(self):
        return self.local_protos

    def train(self, train_data, device, args):
        logging.warning("⚠️ FedTGP: train() method called but FedTGP uses train_and_extract_prototypes()")
        local_protos = self.train_and_extract_prototypes()
        return self.local_sample_number, self.get_model_params(), local_protos

    def _collect_local_prototypes(self, train_loader):
        self.model.eval()
        prototypes = {}
        class_counts = {}

        with torch.no_grad():
            for data, labels in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                rep = self.model.base(data)
                if rep.dim() > 2:
                    rep = rep.view(rep.size(0), -1)

                for i, label in enumerate(labels):
                    y_c = label.item()
                    if y_c not in prototypes:
                        prototypes[y_c] = torch.zeros(rep.size(1)).to(rep.device)
                        class_counts[y_c] = 0
                    prototypes[y_c] += rep[i]
                    class_counts[y_c] += 1

        for class_id in prototypes:
            prototypes[class_id] = (prototypes[class_id] / class_counts[class_id]).cpu()

        expected_dim = getattr(self.args, 'feature_dim', 512)
        try:
            from fedtgp_loss import validate_prototype_dimensions
            validate_prototype_dimensions(
                prototypes,
                expected_dim,
                context=f"Client {getattr(self.args, 'client_id', 'unknown')}"
            )
            logging.info(f"✓ Validated {len(prototypes)} prototypes, each with dimension {expected_dim}")
        except ValueError as e:
            logging.error(f"Prototype validation failed: {e}")
            raise

        self.local_protos = prototypes
        return prototypes

    def test(self, test_data, device, args):
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, labels in test_data:
                data, labels = data.to(self.device), labels.to(self.device)
                rep = self.model.base(data)
                if rep.dim() > 2:
                    rep = rep.view(rep.size(0), -1)
                output = self.model.head(rep)
                loss = criterion(output, labels)
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += len(labels)

        test_loss = total_loss / len(test_data) if len(test_data) > 0 else 0
        test_acc = correct / total if total > 0 else 0
        logging.info(f"Test results: loss={test_loss:.4f}, acc={test_acc:.4f}")
        return correct, total

    def get_train_metrics(self):
        return self.train_loss, self.train_acc
