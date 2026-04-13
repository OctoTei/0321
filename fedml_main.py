#!/usr/bin/env python3
# fedml_main.py - FedTGP Main Script using Socket communication

import argparse
import logging
import sys
import os
import yaml
import torch
from datetime import datetime

# Import our implementations
from fedml_trainer import FedTGPClientTrainer
from fedml_aggregator import FedTGPServerAggregator
from model_heterogeneous import create_heterogeneous_model
from data_loader import load_data, get_dataset_info
from socket_communication import FedTGPSocketServer, FedTGPSocketClient


class FedTGPArgs:
    """Arguments class for FedTGP"""

    def __init__(self, config, client_id=None):
        # Training arguments
        self.federated_optimizer = config['train_args']['federated_optimizer']
        self.client_num_in_total = config['train_args']['client_num_in_total']
        self.client_num_per_round = config['train_args']['client_num_per_round']
        self.comm_round = config['train_args']['comm_round']
        self.epochs = config['train_args']['epochs']
        self.batch_size = config['train_args']['batch_size']
        self.learning_rate = config['train_args']['learning_rate']
        self.weight_decay = config['train_args']['weight_decay']
        self.lamda = config['train_args']['lamda']
        self.tgp_lr = config['train_args'].get('tgp_lr', 0.01)
        self.tgp_epochs = config['train_args'].get('tgp_epochs', 5)
        self.tau = config['train_args'].get('tau', 1.0)

        # Model arguments
        self.model = config['model_args']['model']
        self.feature_dim = config['model_args']['feature_dim']
        self.pretrained = config['model_args'].get('pretrained', False)

        # 自动获取数据集信息
        dataset_name = config['data_args']['dataset']
        dataset_info = get_dataset_info(dataset_name)
        self.num_classes = dataset_info['num_classes']
        self.in_channels = dataset_info['input_channels']
        self.input_size = dataset_info['input_size']

        # Device arguments
        device_config = config['device_args']['using_gpu']
        if torch.cuda.is_available() and device_config:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # Data arguments
        self.dataset = config['data_args']['dataset']
        self.data_cache_dir = config['data_args']['data_cache_dir']
        self.partition_method = config['data_args'].get('partition_method', 'iid')
        self.partition_alpha = config['data_args'].get('partition_alpha', 0.5)

        # Communication arguments
        self.server_ip = config['comm_args']['server_ip']
        self.server_port = config['comm_args']['server_port']
        self.initial_connect_timeout = config['comm_args'].get('initial_connect_timeout', 300)
        self.accept_poll_interval = config['comm_args'].get('accept_poll_interval', 5)
        self.client_handshake_timeout = config['comm_args'].get('client_handshake_timeout', 15)

        # Wandb arguments
        self.enable_wandb = config.get('wandb_args', {}).get('enable_wandb', False)
        self.wandb_project = config.get('wandb_args', {}).get('wandb_project', 'fedtgp')
        self.wandb_name = config.get('wandb_args', {}).get('wandb_name', 'fedtgp_run')
        self.wandb_key = config.get('wandb_args', {}).get('wandb_key', '')

        # Client-specific configuration
        self.client_id = client_id
        if client_id is not None and 'client_configs' in config:
            client_key = f'client_{client_id}'
            if client_key in config['client_configs']:
                client_config = config['client_configs'][client_key]
                self.feature_extractor = client_config['feature_extractor']
                self.classifier = client_config['classifier']
            else:
                self.feature_extractor = self.model
                self.classifier = 'classifier1'
        else:
            self.feature_extractor = self.model
            self.classifier = 'classifier1'


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config, args):
    """初始化日志"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = config['logging_args']['log_file_dir']
    os.makedirs(log_dir, exist_ok=True)

    if args.server_mode:
        log_file = f"{log_dir}/server_{timestamp}.log"
    elif args.client_mode:
        log_file = f"{log_dir}/client_{args.client_id}_{timestamp}.log"
    else:
        log_file = f"{log_dir}/fedtgp_{timestamp}.log"

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)

    logging.info("=" * 60)
    logging.info(f"FedTGP starting with config: {args.config}")
    logging.info(f"Log file: {log_file}")
    logging.info(f"Absolute path: {os.path.abspath(log_file)}")
    logging.info("=" * 60)
    file_handler.flush()

    if os.path.exists(log_file):
        file_size = os.path.getsize(log_file)
        logging.info(f"✓ Log file created successfully, size: {file_size} bytes")
        file_handler.flush()
    else:
        print(f"⚠️ WARNING: Log file not created: {log_file}")
        print(f"⚠️ Please check directory permissions: {log_dir}")

    return log_file


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='FedTGP Distributed Training')
    parser.add_argument('--config', type=str, default='config/fedtgp_distributed.yaml', help='Config file path')
    parser.add_argument('--server_mode', action='store_true', help='Run as server')
    parser.add_argument('--client_mode', action='store_true', help='Run as client')
    parser.add_argument('--client_id', type=int, default=1, help='Client ID (only for client mode)')

    args = parser.parse_args()

    if args.server_mode == args.client_mode:
        print("Error: Must specify exactly one of --server_mode or --client_mode")
        sys.exit(1)

    config = load_config(args.config)

    if args.client_mode:
        fedml_args = FedTGPArgs(config, client_id=args.client_id)
    else:
        fedml_args = FedTGPArgs(config, client_id=None)

    setup_logging(config, args)

    logging.info("Validating configuration...")
    logging.info(f"Model feature_dim: {config['model_args']['feature_dim']}")
    logging.info(f"Model num_classes: {fedml_args.num_classes}")
    logging.info(f"Server IP: {fedml_args.server_ip}")
    logging.info(f"Server Port: {fedml_args.server_port}")
    logging.info(f"Total clients: {fedml_args.client_num_in_total}")
    logging.info(f"Communication rounds: {fedml_args.comm_round}")
    logging.info(f"Initial connect timeout: {fedml_args.initial_connect_timeout}s")
    logging.info(f"Accept poll interval: {fedml_args.accept_poll_interval}s")

    # Load data using FedTGP-style loader
    train_data_num, test_data_num, train_data_global, test_data_global, \
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
    class_num = load_data(config)

    if class_num != fedml_args.num_classes:
        logging.warning(f"Dataset has {class_num} classes but config specifies {fedml_args.num_classes}")
        logging.warning(f"Using dataset's class number: {class_num}")
        fedml_args.num_classes = class_num

    # Create model (异构模型)
    if args.server_mode:
        model, extractor_type, classifier_type = create_heterogeneous_model(config, client_id=None)
        logging.info(f"Server model created: {extractor_type} + {classifier_type}")
    else:
        model, extractor_type, classifier_type = create_heterogeneous_model(config, client_id=args.client_id)
        logging.info(f"Client {args.client_id} model: {extractor_type} + {classifier_type}")

    logging.info(f"Model feature_dim: {fedml_args.feature_dim}, num_classes: {class_num}")

    if args.server_mode:
        print("=" * 60)
        print("FedTGP Server Starting")
        print("=" * 60)
        print(f"Server IP: {fedml_args.server_ip}")
        print(f"Server Port: {fedml_args.server_port}")
        print(f"Waiting for {fedml_args.client_num_in_total} clients")
        print(f"Communication rounds: {fedml_args.comm_round}")
        print(f"Initial connect timeout: {fedml_args.initial_connect_timeout}s")
        print("=" * 60)

        try:
            aggregator = FedTGPServerAggregator(model, fedml_args)
            server = FedTGPSocketServer(
                fedml_args,
                aggregator,
                fedml_args.server_port,
                fedml_args.client_num_in_total,
            )
            ok = server.run()
            print("=" * 60)
            if ok:
                print("FedTGP Server finished successfully")
            else:
                print("FedTGP Server terminated with errors")
            print("=" * 60)
            if not ok:
                sys.exit(1)
        except Exception as e:
            logging.error(f"Server error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    else:
        print("=" * 60)
        print(f"FedTGP Client {args.client_id} Starting")
        print("=" * 60)
        print(f"Server: {fedml_args.server_ip}:{fedml_args.server_port}")
        print(f"Communication rounds: {fedml_args.comm_round}")
        print("=" * 60)

        # data partition uses 0-based client index; network client_id uses 1-based id
        client_idx = args.client_id - 1
        available_client_indices = sorted(train_data_local_dict.keys())
        logging.info(f"Available data partitions: {available_client_indices}")

        if client_idx not in train_data_local_dict:
            raise ValueError(
                f"Client {args.client_id} maps to data partition {client_idx}, but that partition is unavailable. "
                f"Available partitions: {available_client_indices}. "
                f"This usually means the client has 0 or 1 usable samples after partitioning."
            )

        local_data_num = train_data_local_num_dict[client_idx]
        train_data_local = train_data_local_dict[client_idx]
        test_data_local = test_data_local_dict[client_idx]

        print(f"Client {args.client_id} uses data partition {client_idx}")
        print(f"Client {args.client_id} has {local_data_num} training samples")

        try:
            trainer = FedTGPClientTrainer(model, fedml_args)
            trainer.set_client_data(client_idx, train_data_local, test_data_local, local_data_num)

            client = FedTGPSocketClient(
                fedml_args,
                trainer,
                fedml_args.server_ip,
                fedml_args.server_port,
                args.client_id,
            )
            ok = client.run()
            print("=" * 60)
            if ok:
                print("FedTGP Client finished successfully")
            else:
                print("FedTGP Client terminated with errors")
            print("=" * 60)
            if not ok:
                sys.exit(1)
        except Exception as e:
            logging.error(f"Client error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
