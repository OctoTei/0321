
import torch
import torch.nn as nn
import torch.nn.functional as F


class FedTGPClientLoss(nn.Module):
    """
    FedTGP客户端损失函数 (公式11)
    L_i = E_{(x,y)~D_i} l(h_i(f_i(x; θ_i); w_i), y) + λE_{c~C_i} φ(P_i^c, P̂^c)
    
    包含两部分:
    1. 分类损失: 标准交叉熵损失
    2. 原型对齐损失: 本地原型与全局原型的距离
    """
    
    def __init__(self, lamda=1.0):
        super().__init__()
        self.lamda = lamda
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, logits, labels, features, global_protos):
        """
        Args:
            logits: 模型输出 [batch_size, num_classes]
            labels: 真实标签 [batch_size]
            features: 特征表示 [batch_size, feature_dim]
            global_protos: 全局原型字典 {class_id: prototype_tensor}
        
        Returns:
            total_loss: 总损失
            ce_loss: 分类损失
            proto_loss: 原型对齐损失
        """
        # 分类损失
        ce_loss = self.ce_loss(logits, labels)
        
        # 检查CE loss是否为NaN
        if torch.isnan(ce_loss) or torch.isinf(ce_loss):
            ce_loss = torch.tensor(0.0).to(logits.device)
        
        # 原型对齐损失
        proto_loss = torch.tensor(0.0).to(logits.device)
        aligned_samples = 0
        
        if global_protos:
            for i, label in enumerate(labels):
                class_id = label.item()
                if class_id in global_protos:
                    global_proto = global_protos[class_id].to(logits.device)
                    sample_feature = features[i]
                    
                    # 确保维度匹配
                    if global_proto.dim() > 1:
                        global_proto = global_proto.flatten()
                    if sample_feature.dim() > 1:
                        sample_feature = sample_feature.flatten()
                    
                    # 检查是否包含NaN或Inf
                    if torch.isnan(global_proto).any() or torch.isinf(global_proto).any():
                        continue
                    if torch.isnan(sample_feature).any() or torch.isinf(sample_feature).any():
                        continue
                    
                    # 计算MSE损失
                    if global_proto.size(0) == sample_feature.size(0):
                        sample_proto_loss = self.mse_loss(sample_feature, global_proto)
                        
                        # 检查是否为NaN
                        if not torch.isnan(sample_proto_loss) and not torch.isinf(sample_proto_loss):
                            proto_loss += sample_proto_loss
                            aligned_samples += 1
            
            # 平均原型损失
            if aligned_samples > 0:
                proto_loss = proto_loss / aligned_samples
        
        # 总损失
        total_loss = ce_loss + self.lamda * proto_loss
        
        # 最终检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = ce_loss  # 如果总损失有问题，只使用CE损失
        
        return total_loss, ce_loss, proto_loss


class FedTGPServerLoss(nn.Module):
    """
    FedTGP服务器端损失函数 (公式10)
    L_P^c = Σ_{i∈I^t} -log [e^{-(φ(P_i^c, P̂^c) + δ(t))} / (e^{-(φ(P_i^c, P̂^c) + δ(t))} + Σ_{c'} e^{-φ(P_i^c, P̂^{c'})})]
    
    其中 δ(t) 是自适应margin (公式9):
    δ(t) = min(max_{c∈[C],c'∈[C],c≠c'} φ(Q_t^c, Q_t^{c'}), τ)
    """
    
    def __init__(self, tau=100.0, temperature=1.0):
        """
        Args:
            tau: margin阈值（论文使用100）
            temperature: 温度参数（论文公式10中为1.0，不进行缩放）
        """
        super().__init__()
        self.tau = tau  # margin阈值
        self.temperature = temperature  # 温度参数，论文中为1.0
    
    def compute_adaptive_margin(self, client_protos):
        """
        计算自适应margin δ(t) (公式9) - 优化版
        δ(t) = min(max_{c∈[C],c'∈[C],c≠c'} φ(Q_t^c, Q_t^{c'}), τ)
        
        Args:
            client_protos: 客户端原型字典 {client_id: {class_id: prototype}}
        
        Returns:
            delta: 自适应margin值
        """
        if not client_protos or len(client_protos) == 0:
            return torch.tensor(0.0)
        
        # 计算每个类别的平均原型 Q_t^c（类中心）
        class_centers = {}
        for client_id, protos in client_protos.items():
            for class_id, proto in protos.items():
                if class_id not in class_centers:
                    class_centers[class_id] = []
                # 确保原型是1D向量
                proto_flat = proto.flatten()
                class_centers[class_id].append(proto_flat)
        
        # 计算每个类别的平均原型（类中心）
        avg_class_protos = {}
        for class_id, proto_list in class_centers.items():
            if len(proto_list) > 0:
                # 堆叠所有客户端该类别的原型并求平均
                stacked_protos = torch.stack(proto_list)
                avg_class_protos[class_id] = stacked_protos.mean(dim=0)
        
        # 至少需要2个类别才能计算类间距离
        if len(avg_class_protos) < 2:
            return torch.tensor(0.0)
        
        # 计算所有类别对之间的最大距离
        max_distance = 0.0
        class_ids = list(avg_class_protos.keys())
        
        for i, c1 in enumerate(class_ids):
            for c2 in class_ids[i+1:]:
                proto1 = avg_class_protos[c1]
                proto2 = avg_class_protos[c2]
                
                # 计算欧氏距离 φ(Q_t^c, Q_t^{c'})
                distance = torch.norm(proto1 - proto2).item()
                max_distance = max(max_distance, distance)
        
        # 应用阈值 τ：δ(t) = min(max_distance, τ)
        delta = min(max_distance, self.tau)
        
        return torch.tensor(delta)
    
    def forward(self, client_protos, global_protos):
        """
        计算服务器端原型对比损失 (公式10)
        
        公式10:
        L_P^c = Σ_{i∈I^t} -log [e^{-(φ(P_i^c, P̂^c) + δ(t))} / 
                                 (e^{-(φ(P_i^c, P̂^c) + δ(t))} + Σ_{c'≠c} e^{-φ(P_i^c, P̂^{c'})})]
        
        使用LogSumExp技巧避免数值下溢:
        loss = -log(e^{-a} / (e^{-a} + Σ e^{-b}))
             = -log(e^{-a}) + log(e^{-a} + Σ e^{-b})
             = a - log(e^{-a} + Σ e^{-b})
             = a - logsumexp([-a, -b1, -b2, ...])
        
        其中: a = φ(P_i^c, P̂^c) + δ(t), b = φ(P_i^c, P̂^{c'})
        
        Args:
            client_protos: 客户端原型 {client_id: {class_id: prototype}}
            global_protos: 全局原型 {class_id: prototype}
        
        Returns:
            loss: 原型对比损失
            delta: 自适应margin
        """
        if not client_protos or not global_protos:
            return torch.tensor(0.0), torch.tensor(0.0)
        
        # 计算自适应margin δ(t)
        delta = self.compute_adaptive_margin(client_protos)
        
        # 获取device
        device = next(iter(global_protos.values())).device
        delta = delta.to(device)
        total_loss = torch.tensor(0.0).to(device)
        total_samples = 0
        
        # 对每个客户端的每个类别计算对比损失
        for client_id, local_protos in client_protos.items():
            for class_id, local_proto in local_protos.items():
                if class_id not in global_protos:
                    continue
                
                # 确保原型维度一致
                local_proto = local_proto.flatten().to(device)
                
                # 检查原型是否包含NaN或Inf
                if torch.isnan(local_proto).any() or torch.isinf(local_proto).any():
                    continue
                
                # 计算正样本距离（带margin）
                global_proto_pos = global_protos[class_id].flatten().to(device)
                if torch.isnan(global_proto_pos).any() or torch.isinf(global_proto_pos).any():
                    continue
                
                dist_pos = torch.norm(local_proto - global_proto_pos)
                if torch.isnan(dist_pos) or torch.isinf(dist_pos):
                    continue
                
                # 正样本项：φ(P_i^c, P̂^c) + δ(t)
                pos_term = dist_pos + delta
                
                # 收集分母中的所有项（用于LogSumExp）
                # 分母 = e^{-(φ+δ)} + Σ_{c'≠c} e^{-φ}
                # 转换为log空间: log_terms = [-(φ+δ), -φ_1, -φ_2, ...]
                log_terms = [-(pos_term)]  # 正样本项（带margin）
                
                # 遍历所有负样本类别
                for c_prime, global_proto_neg in global_protos.items():
                    if c_prime == class_id:  # 跳过正样本类别
                        continue
                    
                    global_proto_neg = global_proto_neg.flatten().to(device)
                    
                    # 检查原型
                    if torch.isnan(global_proto_neg).any() or torch.isinf(global_proto_neg).any():
                        continue
                    
                    # 计算负样本距离（不带margin）
                    dist_neg = torch.norm(local_proto - global_proto_neg)
                    
                    # 检查距离
                    if torch.isnan(dist_neg) or torch.isinf(dist_neg):
                        continue
                    
                    # 负样本项：-φ(P_i^c, P̂^{c'})
                    log_terms.append(-dist_neg)
                
                # 至少需要1个正样本 + 1个负样本
                if len(log_terms) < 2:
                    continue
                
                # 转换为tensor
                log_terms = torch.stack(log_terms)
                
                # 使用LogSumExp计算对比损失（数值稳定）
                # loss = -log(e^{log_terms[0]} / sum(e^{log_terms}))
                #      = -log_terms[0] + log(sum(e^{log_terms}))
                #      = -log_terms[0] + logsumexp(log_terms)
                # 其中 log_terms[0] = -(φ+δ)
                # 所以 loss = (φ+δ) + logsumexp([-(φ+δ), -φ_1, -φ_2, ...])
                log_sum_exp = torch.logsumexp(log_terms, dim=0)
                loss = -log_terms[0] + log_sum_exp
                # 等价于: loss = pos_term + logsumexp(log_terms)
                
                # 检查loss是否为NaN或Inf
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                total_loss += loss
                total_samples += 1
        
        # 平均损失
        if total_samples > 0:
            total_loss = total_loss / total_samples
        else:
            total_loss = torch.tensor(0.0).to(device)
        
        return total_loss, delta


def compute_prototype_distance(proto1, proto2):
    """计算两个原型之间的欧氏距离"""
    if proto1.dim() > 1:
        proto1 = proto1.flatten()
    if proto2.dim() > 1:
        proto2 = proto2.flatten()
    
    return torch.norm(proto1 - proto2)


def validate_prototype_dimensions(prototypes, expected_dim, context=""):
    """
    验证原型维度一致性
    
    Args:
        prototypes: 原型字典 {class_id: prototype_tensor}
        expected_dim: 期望的特征维度
        context: 上下文信息，用于错误提示
    
    Returns:
        bool: 验证是否通过
    
    Raises:
        ValueError: 如果维度不匹配
    """
    for class_id, proto in prototypes.items():
        proto_flat = proto.flatten() if proto.dim() > 1 else proto
        actual_dim = proto_flat.size(0)
        
        if actual_dim != expected_dim:
            raise ValueError(
                f"{context}: Prototype dimension mismatch for class {class_id}: "
                f"expected {expected_dim}, got {actual_dim}"
            )
    
    return True
