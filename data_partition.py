import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import json
import logging
import argparse
import random

def partition_and_save_dataset(dataset_name, data_dir, num_clients=4, sample_ratio=0.25, partition_alpha=0.1, train_ratio=0.75):
    """
    预先划分数据集并保存为独立的.npz文件
    
    流程：
    1. 加载原始数据集并合并
    2. 全局采样：每个类别保留 25% 样本
    3. Non-IID 划分：使用 Dirichlet 分布分配给 4 个客户端
    4. 客户端内部划分：对每个客户端的数据，按类别比例划分为 75% 训练和 25% 测试
    5. 保存为 .npz
    """
    # 设定随机种子以确保可复现
    seed = 42
    _set_seed(seed)

    # 创建保存目录
    save_dir = os.path.join('dataset', dataset_name.lower())
    train_dir = os.path.join(save_dir, 'train')
    test_dir = os.path.join(save_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    logging.info("="*60)
    logging.info(f"FedTGP 数据集预划分程序")
    logging.info(f"数据集: {dataset_name} | 客户端数: {num_clients} | 种子: {seed}")
    logging.info(f"采样比例: {sample_ratio*100}% | Alpha: {partition_alpha}")
    logging.info(f"本地划分: {train_ratio*100}% 训练 / {(1-train_ratio)*100}% 测试")
    logging.info("="*60)
    
    # 1. 加载原始数据集
    all_data, all_labels, num_classes = _load_raw_dataset(dataset_name, data_dir)
    logging.info(f"原始数据总量: {len(all_labels)}")
    
    # 2. 全局按类别采样 (保留 25%)
    sampled_data, sampled_labels = _sample_by_class(all_data, all_labels, sample_ratio, num_classes)
    logging.info(f"采样后数据总量: {len(sampled_labels)}")
    
    # 3. 使用 Dirichlet 分布划分到各个客户端
    # 确保每个客户端至少有 batch_size * 2 个样本（保证训练稳定）
    min_samples = max(10, int(len(sampled_labels) / num_clients * 0.1))  # 至少10个样本或总样本的10%/客户端数
    client_indices = _partition_dirichlet(sampled_labels, num_clients, partition_alpha, num_classes, min_samples_per_client=min_samples)
    
    config_info = {
        'dataset': dataset_name,
        'num_clients': num_clients,
        'num_classes': num_classes,
        'sample_ratio': sample_ratio,
        'partition_alpha': partition_alpha,
        'train_ratio': train_ratio,
        'clients': {}
    }
    
    # 4. 为每个客户端处理数据并保存
    for client_idx in range(num_clients):
        indices = client_indices[client_idx]
        if len(indices) == 0:
            logging.warning(f"Client {client_idx} 分配到的数据为空！")
            continue
            
        c_data = sampled_data[indices]
        c_labels = sampled_labels[indices]
        
        # 内部按类别分层划分训练集和测试集
        train_indices_local = []
        test_indices_local = []
        
        unique_classes = np.unique(c_labels)
        for cls in unique_classes:
            cls_mask = np.where(c_labels == cls)[0]
            np.random.shuffle(cls_mask)
            
            split_point = int(len(cls_mask) * train_ratio)
            train_indices_local.extend(cls_mask[:split_point])
            test_indices_local.extend(cls_mask[split_point:])
        
        # 最终打乱
        np.random.shuffle(train_indices_local)
        np.random.shuffle(test_indices_local)
        
        # 提取并保存
        c_train_data, c_train_labels = c_data[train_indices_local], c_labels[train_indices_local]
        c_test_data, c_test_labels = c_data[test_indices_local], c_labels[test_indices_local]
        
        np.savez_compressed(os.path.join(train_dir, f'{client_idx}.npz'), data=c_train_data, labels=c_train_labels)
        np.savez_compressed(os.path.join(test_dir, f'{client_idx}.npz'), data=c_test_data, labels=c_test_labels)
        
        # 统计分布
        train_dist = np.bincount(c_train_labels, minlength=num_classes).tolist()
        config_info['clients'][f'client_{client_idx}'] = {
            'train_samples': len(c_train_labels),
            'test_samples': len(c_test_labels),
            'train_class_distribution': train_dist
        }
        
        logging.info(f"Client {client_idx}: 训练集={len(c_train_labels)}, 测试集={len(c_test_labels)}")
        logging.info(f"  训练集分布: {train_dist}")
    
    # 保存配置
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config_info, f, indent=2)
    
    return save_dir

def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def _load_raw_dataset(dataset_name, data_dir):
    """加载并合并训练/测试集，返回 (data, labels, num_classes)"""
    dataset_name = dataset_name.lower()
    transform = transforms.ToTensor()
    
    if dataset_name == 'cifar10':
        train_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError(f"暂时不支持数据集: {dataset_name}")
    
    # 合并
    all_data = np.concatenate([train_ds.data, test_ds.data], axis=0)
    all_labels = np.array(train_ds.targets + test_ds.targets)
    
    # 调整维度为 [N, C, H, W]
    all_data = all_data.transpose(0, 3, 1, 2)
    return all_data, all_labels, num_classes

def _sample_by_class(all_data, all_labels, sample_ratio, num_classes):
    """每个类别采样指定比例"""
    sampled_indices = []
    for c in range(num_classes):
        idx_c = np.where(all_labels == c)[0]
        np.random.shuffle(idx_c)
        keep_num = int(len(idx_c) * sample_ratio)
        sampled_indices.extend(idx_c[:keep_num])
    
    # 采样后再进行一次整体打乱
    np.random.shuffle(sampled_indices)
    return all_data[sampled_indices], all_labels[sampled_indices]

def _partition_dirichlet(labels, num_clients, alpha, num_classes, min_samples_per_client=10):
    """
    使用 Dirichlet 分布生成索引映射，确保每个客户端至少有 min_samples_per_client 个样本
    
    Args:
        labels: 样本标签
        num_clients: 客户端数量
        alpha: Dirichlet分布参数
        num_classes: 类别数量
        min_samples_per_client: 每个客户端最少样本数
    
    Returns:
        client_indices: 每个客户端的样本索引列表
    """
    max_retries = 50  # 增加重试次数
    
    for retry in range(max_retries):
        client_indices = [[] for _ in range(num_clients)]
        
        for c in range(num_classes):
            idx_c = np.where(labels == c)[0]
            np.random.shuffle(idx_c)
            
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            
            split_idxs = np.split(idx_c, proportions)
            for i, idxs in enumerate(split_idxs):
                client_indices[i].extend(idxs.tolist())
        
        # 检查是否所有客户端都有足够的样本
        min_client_samples = min(len(indices) for indices in client_indices)
        
        if min_client_samples >= min_samples_per_client:
            logging.info(f"✓ Dirichlet分配成功 (尝试 {retry+1}/{max_retries})")
            logging.info(f"  最少样本数: {min_client_samples}, 最多样本数: {max(len(indices) for indices in client_indices)}")
            for i, indices in enumerate(client_indices):
                logging.info(f"  Client {i}: {len(indices)} samples")
            return client_indices
        else:
            if retry < max_retries - 1:
                logging.debug(f"  尝试 {retry+1}/{max_retries}: 最少客户端样本数={min_client_samples} < {min_samples_per_client}，重新分配")
    
    # 如果多次尝试都失败，抛出异常并给出建议
    logging.error(f"❌ Dirichlet分配{max_retries}次都未能确保每个客户端有至少{min_samples_per_client}个样本")
    logging.error(f"建议：")
    logging.error(f"  1. 增加 sample_ratio (当前: {len(labels) / 60000:.2%})")
    logging.error(f"  2. 减少 num_clients (当前: {num_clients})")
    logging.error(f"  3. 增加 alpha 值使分布更均匀 (当前: {alpha})")
    raise ValueError(f"无法通过Dirichlet分布为{num_clients}个客户端分配数据，每个客户端至少需要{min_samples_per_client}个样本")


def main():
    # 这里直接定义参数，不依赖外部配置文件
    DATASET = "cifar10"
    DATA_DIR = "./data"
    NUM_CLIENTS = 8
    SAMPLE_RATIO = 0.25
    ALPHA = 0.1
    TRAIN_RATIO = 0.75

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        save_dir = partition_and_save_dataset(
            dataset_name=DATASET,
            data_dir=DATA_DIR,
            num_clients=NUM_CLIENTS,
            sample_ratio=SAMPLE_RATIO,
            partition_alpha=ALPHA,
            train_ratio=TRAIN_RATIO
        )
        print(f"\n{'='*60}")
        print(f"✓ 数据集划分完成！")
        print(f"✓ 保存位置: {save_dir}")
        print(f"{'='*60}")
    except Exception as e:
        logging.error(f"划分过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()