#!/usr/bin/env python3
# fedml_aggregator.py - FedTGP Aggregator based on FedML Framework

"""
FedTGP服务器聚合器

FedTGP通信逻辑：
1. 服务器发送全局原型给所有客户端（不发送模型参数）
2. 接收所有客户端的本地原型（不接收模型参数）
3. 使用TGP网络训练：
   - 输入：客户端上传的原型
   - 架构：可训练向量 -> FC -> ReLU -> FC
   - 输出：精炼后的全局原型
4. 生成新的全局原型用于下一轮



核心方法：
- aggregate_prototypes(): FedTGP专用聚合方法
- update_TGP(): 训练TGP网络
- generate_global_protos(): 生成全局原型
"""

import copy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import sys
import os

# Add FedML to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from fedml.core.alg_frame.server_aggregator import ServerAggregator
from model_heterogeneous import FedTGPModel
from fedtgp_loss import FedTGPServerLoss

# Wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not installed. Install with: pip install wandb")


class FedTGPServerAggregator(ServerAggregator):
    """FedTGP Server Aggregator based on FedML Framework"""
    
    def __init__(self, model, args):
        super().__init__(model, args)
        # Handle device configuration properly
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
        
        # FedTGP specific parameters
        self.global_protos = {}
        self.client_protos_dict = {}
        self.client_weights = {}
        self.uploaded_protos = {}
        self.flag_client_model_uploaded_dict = {}
        
        # Initialize client flags
        for idx in range(args.client_num_in_total):
            self.flag_client_model_uploaded_dict[idx] = False
        
        # TGP (Trainable Global Prototypes) parameters
        self.num_classes = getattr(args, 'num_classes', 10)
        self.feature_dim = getattr(args, 'feature_dim', 512)
        self.tgp_lr = getattr(args, 'tgp_lr', 0.01)
        self.tgp_epochs = getattr(args, 'tgp_epochs', 5)
        self.tau = getattr(args, 'tau', 1.0)
        
        # Initialize TGP generator - 按照论文Figure 3的架构
        # 输入: 可训练的原型向量 {P̂^c}_{c=1}^C
        # 架构: FC -> ReLU -> FC
        # 输出: 精炼后的全局原型
        
        # 可训练的原型向量（每个类别一个）- 使用Xavier初始化
        self.trainable_protos = nn.Parameter(
            torch.randn(self.num_classes, self.feature_dim) * 0.01
        )
        
        # TGP处理网络 θ_F: FC -> ReLU -> FC - 使用Xavier初始化
        self.TGP = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        ).to(self.device)
        
        # Xavier初始化TGP网络权重
        for module in self.TGP.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        self.tgp_optimizer = optim.SGD(
            list(self.TGP.parameters()) + [self.trainable_protos],
            lr=self.tgp_lr
        )
        
        # 使用FedTGP服务器端损失函数
        self.fedtgp_server_loss = FedTGPServerLoss(tau=self.tau)
        
        # Gap tracking for TGP training
        self.min_gap = None
        self.max_gap = None
        self.adaptive_margin = None
        self.current_round = 0
        
        # Wandb setup
        self.use_wandb = getattr(args, 'enable_wandb', False) and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=getattr(args, 'wandb_project', 'fedtgp'),
                name='server',
                config={
                    'num_classes': self.num_classes,
                    'feature_dim': self.feature_dim,
                    'tgp_lr': self.tgp_lr,
                    'tgp_epochs': self.tgp_epochs,
                    'tau': self.tau,
                    'num_clients': args.client_num_in_total,
                    'comm_rounds': getattr(args, 'comm_round', 100),
                },
                reinit=True
            )
            logging.info("Wandb initialized for server")
        
        logging.info(f"FedTGP aggregator initialized with {self.num_classes} classes, "
                    f"feature_dim={self.feature_dim}, tau={self.tau}")
    
    def get_model_params(self):
        """Get model parameters (required by FedML)"""
        return self.model.cpu().state_dict()
    
    def set_model_params(self, model_parameters):
        """Set model parameters (required by FedML)"""
        self.model.load_state_dict(model_parameters)
        self.model.to(self.device)
    
    def add_local_trained_result(self, client_idx, model_params, sample_num, local_protos=None):
        """
        Add client training results (FedML框架兼容性方法)
        
        ⚠️ 注意：FedTGP不使用这个方法
        FedTGP使用aggregate_prototypes()直接处理原型
        这个方法保留是为了FedML框架兼容性
        
        参数model_params在FedTGP中被忽略，只使用local_protos
        """
        # Store client prototypes (main focus for FedTGP)
        if local_protos is not None:
            # Validate prototype dimensions before storing
            try:
                from fedtgp_loss import validate_prototype_dimensions
                validate_prototype_dimensions(
                    local_protos,
                    self.feature_dim,
                    context=f"Server receiving from Client {client_idx}"
                )
            except ValueError as e:
                logging.error(f"Prototype validation failed for client {client_idx}: {e}")
                raise
            
            self.client_protos_dict[client_idx] = local_protos
            self.client_weights[client_idx] = sample_num
            self.flag_client_model_uploaded_dict[client_idx] = True
            
            # Store uploaded prototypes for TGP training
            for class_id, proto in local_protos.items():
                if class_id not in self.uploaded_protos:
                    self.uploaded_protos[class_id] = []
                
                # Ensure proto is 1D
                proto_tensor = proto.to(self.device)
                if proto_tensor.dim() > 1:
                    proto_tensor = proto_tensor.flatten()
                
                self.uploaded_protos[class_id].append(proto_tensor)
            
            logging.info(f"✓ FedTGP: Added prototypes from client {client_idx}: {len(local_protos)} classes (model_params ignored)")
    
    def check_whether_all_receive(self):
        """Check if all clients have sent their updates"""
        received_clients = 0
        for idx in range(self.args.client_num_in_total):
            if self.flag_client_model_uploaded_dict.get(idx, False):
                received_clients += 1
        
        logging.info(f"TGP Aggregator: {received_clients}/{self.args.client_num_in_total} clients have sent data")
        
        if received_clients == self.args.client_num_in_total:
            # Reset flags for next round
            for idx in range(self.args.client_num_in_total):
                self.flag_client_model_uploaded_dict[idx] = False
            return True
        return False
    
    def update_TGP(self):
        """Update TGP generator using uploaded prototypes with FedTGP server loss"""
        if not self.uploaded_protos:
            logging.warning("No uploaded prototypes for TGP training")
            return 0.0
        
        print(f"\n{'='*60}")
        print(f"Server TGP Training - Round {self.current_round}")
        print(f"{'='*60}")
        print(f"Uploaded prototypes: {sum(len(protos) for protos in self.uploaded_protos.values())} total")
        print(f"Classes covered: {len(self.uploaded_protos)}")
        print(f"TGP epochs: {self.tgp_epochs}")
        print(f"TGP learning rate: {self.tgp_lr}")
        print(f"Tau (margin threshold): {self.tau}")
        print(f"Architecture: Trainable Vectors -> FC -> ReLU -> FC")
        print(f"{'-'*60}")
        
        self.TGP.train()
        total_loss = 0.0
        total_contrastive_loss = 0.0
        
        for epoch in range(self.tgp_epochs):
            epoch_loss = 0.0
            epoch_contrastive = 0.0
            
            # 生成当前的全局原型（通过TGP网络处理可训练向量）
            current_global_protos = {}
            for class_idx in range(self.num_classes):
                # 获取该类别的可训练向量
                trainable_vec = self.trainable_protos[class_idx]
                # 通过TGP网络处理
                generated_proto = self.TGP(trainable_vec)
                current_global_protos[class_idx] = generated_proto
            
            # 计算对比损失（使用FedTGP服务器端损失函数 - 公式7）
            contrastive_loss, adaptive_margin = self.fedtgp_server_loss(
                self.client_protos_dict, current_global_protos
            )
            
            # 总损失 = 对比损失（论文中服务器端只有对比损失）
            loss = contrastive_loss
            
            # 检查NaN
            if torch.isnan(loss) or torch.isinf(loss):
                logging.error(f"NaN/Inf detected in TGP loss at epoch {epoch+1}")
                logging.error(f"  Contrastive loss: {contrastive_loss.item()}")
                logging.error(f"  Adaptive margin: {adaptive_margin.item()}")
                # 跳过这个epoch
                continue
            
            self.tgp_optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(list(self.TGP.parameters()) + [self.trainable_protos], max_norm=1.0)
            
            self.tgp_optimizer.step()
            
            epoch_loss += loss.item()
            epoch_contrastive += contrastive_loss.item()
            
            total_loss += epoch_loss
            total_contrastive_loss += epoch_contrastive
            
            # 输出每个epoch的信息
            print(f"TGP Epoch [{epoch+1}/{self.tgp_epochs}]:")
            print(f"  Total Loss:        {epoch_loss:.6f}")
            print(f"  └─ Contrastive:    {epoch_contrastive:.6f}")
            print(f"  Adaptive Margin:   {adaptive_margin:.6f}")
        
        avg_loss = total_loss / self.tgp_epochs
        avg_contrastive = total_contrastive_loss / self.tgp_epochs
        
        # 存储自适应margin
        self.adaptive_margin = adaptive_margin
        
        # Calculate gaps for monitoring
        self._calculate_gaps()
        
        # 输出TGP训练总结
        print(f"{'-'*60}")
        print(f"TGP Training Summary:")
        print(f"  Average Total Loss:  {avg_loss:.6f}")
        print(f"  Average Contrastive: {avg_contrastive:.6f}")
        print(f"  Adaptive Margin δ:   {adaptive_margin:.6f}")
        if self.min_gap is not None and self.max_gap is not None:
            print(f"  Min Gap:             {self.min_gap:.6f}")
            print(f"  Max Gap:             {self.max_gap:.6f}")
        print(f"{'='*60}\n")
        
        logging.info(f"TGP training completed - Total: {avg_loss:.6f}, "
                    f"Contrastive: {avg_contrastive:.6f}, "
                    f"Adaptive Margin: {adaptive_margin:.6f}")
        
        # 强制刷新日志
        for handler in logging.root.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
        
        return avg_loss
    
    def _calculate_gaps(self):
        """Calculate min and max gaps between uploaded and generated prototypes"""
        if not self.uploaded_protos:
            return
        
        self.TGP.eval()
        gaps = []
        
        with torch.no_grad():
            for class_id, proto_list in self.uploaded_protos.items():
                if len(proto_list) == 0:
                    continue
                
                # Generate prototype for this class using TGP
                trainable_vec = self.trainable_protos[class_id]
                generated_proto = self.TGP(trainable_vec)
                
                # Calculate gaps with uploaded prototypes
                for uploaded_proto in proto_list:
                    gap = torch.norm(generated_proto - uploaded_proto).item()
                    gaps.append(gap)
        
        if gaps:
            self.min_gap = torch.tensor(min(gaps))
            self.max_gap = torch.tensor(max(gaps))
        else:
            self.min_gap = torch.tensor(0.0)
            self.max_gap = torch.tensor(0.0)
    
    def generate_global_protos(self):
        """Generate global prototypes using trained TGP (按照论文Figure 3)"""
        self.TGP.eval()
        global_protos = {}
        
        with torch.no_grad():
            # Generate prototypes for all classes
            for class_idx in range(self.num_classes):
                # 获取可训练向量
                trainable_vec = self.trainable_protos[class_idx]
                # 通过TGP网络处理: FC -> ReLU -> FC
                generated_proto = self.TGP(trainable_vec)
                global_protos[class_idx] = generated_proto.cpu()
        
        self.global_protos = global_protos
        return global_protos
    
    def aggregate(self, raw_client_model_or_grad_list):
        """
        Main aggregation method (required by FedML framework compatibility)
        
        ⚠️ 注意：FedTGP不使用这个方法进行实际聚合
        FedTGP只传输原型，不传输模型参数
        实际聚合使用aggregate_prototypes()方法
        
        这个方法保留是为了FedML框架兼容性
        """
        self.current_round += 1
        logging.info(f"⚠️ FedTGP Server: aggregate() called but FedTGP only uses prototypes")
        logging.info(f"⚠️ Use aggregate_prototypes() for FedTGP aggregation")
        
        # 返回当前模型参数（仅用于框架兼容性，实际不使用）
        return self.get_model_params()
    
    def aggregate_prototypes(self, client_prototypes):
        """
        FedTGP专用聚合方法：使用TGP生成器创建全局原型
        
        这是FedTGP的核心聚合逻辑：
        1. 接收客户端原型（不接收模型参数）
        2. 使用TGP网络训练
        3. 生成新的全局原型
        4. 返回全局原型（不返回模型参数）
        """
        self.current_round += 1
        logging.info(f"✓ FedTGP Server: Starting prototype aggregation for round {self.current_round}")
        logging.info(f"✓ FedTGP: Only prototypes are transmitted, NO model parameters")
        
        # Clear previous data
        self.uploaded_protos = {}
        self.client_protos_dict = {}  # ✅ 清空client_protos_dict
        
        # Store client prototypes for TGP training
        for client_id, local_protos in client_prototypes.items():
            if local_protos:
                # ✅ 存储到client_protos_dict（用于对比损失计算）
                self.client_protos_dict[client_id] = {}
                
                for class_id, proto in local_protos.items():
                    # Ensure proto is 1D
                    proto_tensor = proto.to(self.device)
                    if proto_tensor.dim() > 1:
                        proto_tensor = proto_tensor.flatten()
                    
                    # Validate dimension
                    if proto_tensor.size(0) != self.feature_dim:
                        logging.error(f"Client {client_id} prototype dimension mismatch for class {class_id}: "
                                    f"expected {self.feature_dim}, got {proto_tensor.size(0)}")
                        raise ValueError(f"Prototype dimension mismatch: {proto_tensor.size(0)} != {self.feature_dim}")
                    
                    # 存储到uploaded_protos（按类别组织，用于MSE损失）
                    if class_id not in self.uploaded_protos:
                        self.uploaded_protos[class_id] = []
                    self.uploaded_protos[class_id].append(proto_tensor)
                    
                    # ✅ 存储到client_protos_dict（按客户端组织，用于对比损失）
                    self.client_protos_dict[client_id][class_id] = proto_tensor
        
        if not self.uploaded_protos:
            logging.warning("No client prototypes received for TGP training")
            return {}
        
        logging.info(f"Validated prototypes from {len(client_prototypes)} clients, feature_dim={self.feature_dim}")
        logging.info(f"✓ client_protos_dict has {len(self.client_protos_dict)} clients for contrastive loss")
        
        # Train TGP using uploaded prototypes
        tgp_loss = self.update_TGP()
        logging.info(f"✓ TGP training completed with loss: {tgp_loss:.4f}")
        
        # Generate global prototypes using trained TGP
        global_prototypes = self.generate_global_protos()
        logging.info(f"✓ Generated {len(global_prototypes)} global prototypes using TGP")
        
        # Wandb logging
        if self.use_wandb:
            metrics = {
                'round': self.current_round,
                'server/tgp_loss': tgp_loss,
                'server/num_global_protos': len(global_prototypes),
                'server/num_clients': len(client_prototypes),
            }
            
            if self.adaptive_margin is not None:
                metrics['server/adaptive_margin'] = self.adaptive_margin.item()
            if self.min_gap is not None:
                metrics['server/min_gap'] = self.min_gap.item()
            if self.max_gap is not None:
                metrics['server/max_gap'] = self.max_gap.item()
            
            wandb.log(metrics)
        
        return global_prototypes
    
    def get_global_prototypes(self):
        """Get global prototypes for distribution to clients"""
        return self.global_protos
    
    def get_tgp_metrics(self):
        """Get TGP training metrics"""
        return {
            'min_gap': self.min_gap.item() if self.min_gap is not None else 0.0,
            'max_gap': self.max_gap.item() if self.max_gap is not None else 0.0,
            'adaptive_margin': self.adaptive_margin.item() if self.adaptive_margin is not None else 0.0,
            'num_uploaded_protos': sum(len(protos) for protos in self.uploaded_protos.values())
        }
    
    def test(self, test_data, device, args):
        """Test the global model (required by FedML)"""
        # Note: In pure prototype aggregation, server doesn't perform testing
        # This is kept for FedML framework compatibility
        
        self.model.to(self.device)
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, labels in test_data:
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Forward pass
                rep = self.model.base(data)
                output = self.model.head(rep)
                
                # Calculate loss and accuracy
                loss = criterion(output, labels)
                total_loss += loss.item()
                
                pred = output.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += len(labels)
        
        test_loss = total_loss / len(test_data) if len(test_data) > 0 else 0
        test_acc = correct / total if total > 0 else 0
        
        logging.info(f"Server test results: loss={test_loss:.4f}, acc={test_acc:.4f}")
        
        return test_acc, test_loss
    
    def test_all(self, train_data_local_dict, test_data_local_dict, device, args):
        """Test on all clients' data (FedML compatibility)"""
        # In pure prototype aggregation, server doesn't test on client data
        # Return True for compatibility
        return True
    
    def clear_client_data(self):
        """Clear client data for next round"""
        self.client_protos_dict.clear()
        self.client_weights.clear()
        self.uploaded_protos.clear()
        logging.info("Cleared client data for next round")
    
    def get_aggregation_summary(self):
        """Get summary of aggregation results"""
        tgp_metrics = self.get_tgp_metrics()
        return {
            'clients_participated': len(self.client_protos_dict),
            'global_prototypes_count': len(self.global_protos),
            'total_samples': sum(self.client_weights.values()) if self.client_weights else 0,
            'min_gap': tgp_metrics['min_gap'],
            'max_gap': tgp_metrics['max_gap'],
            'adaptive_margin': tgp_metrics['adaptive_margin'],
            'num_uploaded_protos': tgp_metrics['num_uploaded_protos']
        }
