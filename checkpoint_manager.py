#!/usr/bin/env python3
"""
FedTGP Checkpoint Manager
用于保存和恢复训练状态，支持断电恢复
"""

import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from datetime import datetime


class CheckpointManager:
    """管理 FedTGP 训练的 checkpoint"""

    def __init__(self, checkpoint_dir="./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        logging.info(f"Checkpoint directory: {checkpoint_dir}")

    def _extract_round_num(self, filename):
        """
        从文件名中提取 round 编号
        支持:
        - server_round_5.pt
        - client_2_round_5.pt
        - server_xxx_state_round_5.pt
        """
        match = re.search(r"round_(\d+)\.pt$", filename)
        if not match:
            raise ValueError(f"Invalid checkpoint filename format: {filename}")
        return int(match.group(1))

    def _list_checkpoint_files(self, mode, entity_id=None):
        """列出 server/client 的 checkpoint 文件"""
        if mode == "server":
            return [
                f for f in os.listdir(self.checkpoint_dir)
                if f.startswith("server_round_") and f.endswith(".pt")
            ]
        elif mode == "client":
            return [
                f for f in os.listdir(self.checkpoint_dir)
                if f.startswith(f"client_{entity_id}_round_") and f.endswith(".pt")
            ]
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def save_server_checkpoint(self, round_num, aggregator, global_prototypes):
        """保存服务器 checkpoint"""
        checkpoint = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "global_prototypes": {
                k: v.detach().cpu().numpy().tolist() for k, v in global_prototypes.items()
            },
            "tgp_state": aggregator.TGP.state_dict(),
            "tgp_optimizer_state": (
                aggregator.tgp_optimizer.state_dict()
                if getattr(aggregator, "tgp_optimizer", None) is not None
                else None
            ),
            "trainable_protos": aggregator.trainable_protos.detach().cpu().numpy().tolist(),
            "adaptive_margin": (
                aggregator.adaptive_margin.item()
                if getattr(aggregator, "adaptive_margin", None) is not None
                else None
            ),
        }

        checkpoint_path = os.path.join(self.checkpoint_dir, f"server_round_{round_num}.pt")
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"✓ Server checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_server_checkpoint(self, aggregator):
        """加载最新的服务器 checkpoint"""
        checkpoint_files = self._list_checkpoint_files("server")

        if not checkpoint_files:
            logging.info("No server checkpoint found, starting from round 0")
            return None, 0

        latest_checkpoint = sorted(checkpoint_files, key=self._extract_round_num)[-1]
        checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=aggregator.device)
            round_num = checkpoint["round"]

            # 恢复 TGP 网络状态
            aggregator.TGP.load_state_dict(checkpoint["tgp_state"])

            # 恢复可训练原型
            aggregator.trainable_protos = nn.Parameter(
                torch.tensor(
                    checkpoint["trainable_protos"],
                    dtype=torch.float32,
                    device=aggregator.device,
                )
            )

            # 重新创建优化器，保证 trainable_protos 在 optimizer 参数组里
            aggregator.tgp_optimizer = optim.SGD(
                list(aggregator.TGP.parameters()) + [aggregator.trainable_protos],
                lr=aggregator.tgp_lr
            )

            # 恢复优化器状态（当前默认 SGD 无 momentum，即使没有状态也可安全继续）
            optimizer_state = checkpoint.get("tgp_optimizer_state", None)
            if optimizer_state is not None:
                try:
                    aggregator.tgp_optimizer.load_state_dict(optimizer_state)
                    logging.info("✓ TGP optimizer state restored")
                except Exception as opt_e:
                    logging.warning(f"Failed to restore TGP optimizer state, recreated optimizer only: {opt_e}")

            # 恢复 adaptive margin
            if checkpoint.get("adaptive_margin", None) is not None:
                aggregator.adaptive_margin = torch.tensor(
                    checkpoint["adaptive_margin"],
                    dtype=torch.float32,
                    device=aggregator.device
                )
            else:
                aggregator.adaptive_margin = None

            # 恢复全局原型
            global_prototypes = {
                int(k): torch.tensor(v, dtype=torch.float32, device=aggregator.device)
                for k, v in checkpoint["global_prototypes"].items()
            }

            logging.info(f"✓ Server checkpoint loaded from round {round_num}: {checkpoint_path}")
            logging.info("✓ Optimizer recreated with updated trainable_protos")

            # 注意：这里返回 round_num，不是 round_num + 1
            # 因为 server 主循环是 for round_num in range(start_round, max_rounds)
            # 保存的是“已完成的轮数”，所以下一轮应从同名数字继续
            return global_prototypes, round_num

        except Exception as e:
            logging.warning(
                f"Server checkpoint incompatible or corrupted, start from scratch. "
                f"File: {checkpoint_path}, Error: {e}"
            )
            return None, 0

    def save_client_checkpoint(self, client_id, round_num, model, local_protos):
        """保存客户端 checkpoint"""
        checkpoint = {
            "client_id": client_id,
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "model_state": model.state_dict(),
            "local_protos": {
                k: v.detach().cpu().numpy().tolist() for k, v in local_protos.items()
            },
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"client_{client_id}_round_{round_num}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"✓ Client {client_id} checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_client_checkpoint(self, client_id, model, device):
        """加载最新的客户端 checkpoint"""
        checkpoint_files = self._list_checkpoint_files("client", entity_id=client_id)

        if not checkpoint_files:
            logging.info(f"No client {client_id} checkpoint found, starting from round 0")
            return None, 0

        latest_checkpoint = sorted(checkpoint_files, key=self._extract_round_num)[-1]
        checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            round_num = checkpoint["round"]

            # 严格恢复模型状态；若结构不兼容则走 except
            model.load_state_dict(checkpoint["model_state"])
            model.to(device)

            # 恢复本地原型
            local_protos = {
                int(k): torch.tensor(v, dtype=torch.float32, device=device)
                for k, v in checkpoint["local_protos"].items()
            }

            logging.info(f"✓ Client {client_id} checkpoint loaded from round {round_num}: {checkpoint_path}")

            # 这里同样返回 round_num，保持语义一致
            return local_protos, round_num

        except Exception as e:
            logging.warning(
                f"Client {client_id} checkpoint incompatible or corrupted, start from scratch. "
                f"File: {checkpoint_path}, Error: {e}"
            )
            return None, 0

    def save_training_state(self, mode, entity_id, round_num, state_dict):
        """保存通用训练状态"""
        checkpoint = {
            "mode": mode,   # 'server' or 'client'
            "entity_id": entity_id,
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "state": state_dict,
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{mode}_{entity_id}_state_round_{round_num}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"✓ Training state saved: {checkpoint_path}")
        return checkpoint_path

    def get_latest_round(self, mode, entity_id=None):
        """获取最新 checkpoint 对应的轮数"""
        checkpoint_files = self._list_checkpoint_files(mode, entity_id)

        if not checkpoint_files:
            return 0

        latest = sorted(checkpoint_files, key=self._extract_round_num)[-1]
        return self._extract_round_num(latest)

    def cleanup_old_checkpoints(self, mode, entity_id=None, keep_last_n=3):
        """清理旧的 checkpoint，只保留最近的 N 个"""
        checkpoint_files = self._list_checkpoint_files(mode, entity_id)

        if len(checkpoint_files) <= keep_last_n:
            return

        sorted_files = sorted(checkpoint_files, key=self._extract_round_num)
        files_to_delete = sorted_files[:-keep_last_n]

        for f in files_to_delete:
            file_path = os.path.join(self.checkpoint_dir, f)
            try:
                os.remove(file_path)
                logging.info(f"Deleted old checkpoint: {f}")
            except OSError as e:
                logging.warning(f"Failed to delete old checkpoint {f}: {e}")