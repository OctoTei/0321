#!/usr/bin/env python3
# data_loader.py - FedTGP-style data loader

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import json
import logging


class FedTGPDataset(Dataset):
    """FedTGP数据集类 - 从.npz文件加载"""

    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        if isinstance(label, np.ndarray):
            label = int(label)

        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        if self.transform:
            if len(img.shape) == 3 and (img.shape[0] == 3 or img.shape[0] == 1):
                img = np.transpose(img, (1, 2, 0))

            if len(img.shape) == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)

            from PIL import Image
            img = Image.fromarray(img)
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).float()
            if img.max() > 1.0:
                img = img / 255.0

        return img, label


def get_dataset_info(dataset_name):
    """获取数据集信息"""
    dataset_name = dataset_name.lower()

    dataset_configs = {
        'mnist': {
            'num_classes': 10,
            'input_channels': 1,
            'input_size': 28,
            'mean': (0.1307,),
            'std': (0.3081,)
        },
        'fashionmnist': {
            'num_classes': 10,
            'input_channels': 1,
            'input_size': 28,
            'mean': (0.2860,),
            'std': (0.3530,)
        },
        'cifar10': {
            'num_classes': 10,
            'input_channels': 3,
            'input_size': 32,
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2023, 0.1994, 0.2010)
        },
        'cifar100': {
            'num_classes': 100,
            'input_channels': 3,
            'input_size': 32,
            'mean': (0.5071, 0.4867, 0.4408),
            'std': (0.2675, 0.2565, 0.2761)
        }
    }

    for key, config in dataset_configs.items():
        if key in dataset_name:
            logging.info(f"Dataset '{dataset_name}' matched to config '{key}'")
            return config

    logging.warning(f"Dataset '{dataset_name}' not recognized, using default config")
    return {
        'num_classes': 10,
        'input_channels': 3,
        'input_size': 32,
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5)
    }


def _get_train_transform(dataset_name, dataset_info):
    """获取训练数据的transform"""
    if 'cifar' in dataset_name.lower():
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(dataset_info['mean'], dataset_info['std'])
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dataset_info['mean'], dataset_info['std'])
    ])


def _get_test_transform(dataset_name, dataset_info):
    """获取测试数据的transform"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dataset_info['mean'], dataset_info['std'])
    ])


def read_client_data(dataset_name, client_idx, is_train=True):
    """读取客户端数据（FedTGP风格）"""
    if is_train:
        data_dir = os.path.join('dataset', dataset_name.lower(), 'train')
    else:
        data_dir = os.path.join('dataset', dataset_name.lower(), 'test')

    file_path = os.path.join(data_dir, f'{client_idx}.npz')

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Client data file not found: {file_path}\n"
            f"Please run data_partition.py first to generate partitioned data."
        )

    data_file = np.load(file_path)
    data = data_file['data']
    labels = data_file['labels']

    if len(labels) == 0:
        logging.warning(f"Client {client_idx} has empty {'train' if is_train else 'test'} dataset")

    dataset_info = get_dataset_info(dataset_name)
    transform = _get_train_transform(dataset_name, dataset_info) if is_train else _get_test_transform(dataset_name, dataset_info)
    dataset = FedTGPDataset(data, labels, transform)

    logging.info(f"Loaded client {client_idx} {'train' if is_train else 'test'} data: {len(dataset)} samples")
    return dataset


def load_data(config):
    """FedTGP风格的数据加载函数"""
    dataset_name = config['data_args']['dataset']
    batch_size = config['train_args']['batch_size']
    num_clients = config['train_args']['client_num_in_total']

    dataset_dir = os.path.join('dataset', dataset_name.lower())
    config_file = os.path.join(dataset_dir, 'config.json')

    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"Dataset not partitioned yet: {config_file}\n"
            f"Please run: python data_partition.py"
        )

    with open(config_file, 'r') as f:
        dataset_config = json.load(f)

    num_classes = dataset_config['num_classes']

    logging.info(f"Loading FedTGP-style data for {dataset_name}")
    logging.info(f"Number of clients: {num_clients}")
    logging.info(f"Number of classes: {num_classes}")

    train_data_local_dict = {}
    test_data_local_dict = {}
    train_data_local_num_dict = {}

    total_train_samples = 0
    total_test_samples = 0

    for client_idx in range(num_clients):
        client_key = f'client_{client_idx}'
        if client_key not in dataset_config['clients']:
            logging.warning(f"⚠️ Client {client_idx} not found in dataset config (likely has 0 samples)")
            logging.warning(f"   Skipping client {client_idx} data loading")
            continue

        train_dataset = read_client_data(dataset_name, client_idx, is_train=True)

        if len(train_dataset) == 0:
            logging.warning(f"⚠️ Client {client_idx} has empty training dataset, skipping")
            continue
        if len(train_dataset) == 1:
            logging.warning(f"⚠️ Client {client_idx} has only 1 sample (BatchNorm requires >= 2), skipping")
            continue

        effective_batch_size = min(batch_size, len(train_dataset))
        effective_batch_size = max(2, effective_batch_size)

        # 关键修改：drop_last=False，把最后一个小批次保留下来；训练时由 trainer 手动跳过 size<2 的 batch。
        should_drop_last = False

        train_loader = DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            drop_last=should_drop_last,
        )
        train_data_local_dict[client_idx] = train_loader
        train_data_local_num_dict[client_idx] = len(train_dataset)
        total_train_samples += len(train_dataset)

        test_dataset = read_client_data(dataset_name, client_idx, is_train=False)
        test_data_local_dict[client_idx] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        total_test_samples += len(test_dataset)

        client_info = dataset_config['clients'][client_key]
        logging.info(f"Client {client_idx}:")
        logging.info(
            f"  Train: {client_info['train_samples']} samples "
            f"(batch_size={effective_batch_size}, drop_last={should_drop_last})"
        )
        logging.info(f"  Test: {client_info['test_samples']} samples")
        logging.info(f"  Train class dist: {client_info['train_class_distribution']}")

    if len(train_data_local_dict) == 0:
        logging.error("No valid clients with data found!")
        raise ValueError("No clients have data available for training")

    first_valid_client = min(train_data_local_dict.keys())
    train_data_global = train_data_local_dict[first_valid_client]
    test_data_global = test_data_local_dict[first_valid_client]

    logging.info(f"Using client {first_valid_client} as global data representative")
    logging.info(f"Available client partitions: {sorted(train_data_local_dict.keys())}")
    logging.info(f"Total training samples: {total_train_samples}")
    logging.info(f"Total test samples: {total_test_samples}")

    return (
        total_train_samples,
        total_test_samples,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        num_classes,
    )
