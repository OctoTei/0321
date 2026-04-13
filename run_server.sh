#!/bin/bash
# FedTGP Server启动脚本
# 使用方法: ./run_server.sh

CONFIG_FILE="config/server_config.yaml"

# 创建logs目录
mkdir -p logs

echo "============================================================"
echo "FedTGP Server Starting"
echo "============================================================"
echo "Config file: $CONFIG_FILE"
echo "Logs will be saved to: logs/"
echo "All parameters will be read from config file"
echo "============================================================"

python fedml_main.py \
    --config $CONFIG_FILE \
    --server_mode

echo ""
echo "============================================================"
echo "FedTGP Server stopped"
echo "============================================================"
