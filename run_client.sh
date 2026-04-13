#!/bin/bash
# FedTGP Client启动脚本
# 使用方法: ./run_client.sh <client_id> [config_file]
# 例如: ./run_client.sh 1
# 或者: ./run_client.sh 1 config/client1_config.yaml

if [ $# -eq 0 ]; then
    echo "Error: Client ID is required"
    echo "Usage: ./run_client.sh <client_id> [config_file]"
    echo "Example: ./run_client.sh 1"
    echo "Example: ./run_client.sh 1 config/client1_config.yaml"
    echo ""
    echo "Available clients:"
    echo "  1 - CNN4 + Classifier1 (config/client1_config.yaml)"
    echo "  2 - MobileNetV2 + Classifier2 (config/client2_config.yaml)"
    echo "  3 - ResNet18 + Classifier3 (config/client3_config.yaml)"
    echo "  4 - ResNet34 + Classifier4 (config/client4_config.yaml)"
    exit 1
fi

CLIENT_ID=$1

# 如果提供了配置文件路径，使用它；否则根据client_id自动选择
if [ $# -ge 2 ]; then
    CONFIG_FILE=$2
else
    CONFIG_FILE="config/client${CLIENT_ID}_config.yaml"
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# 创建logs目录
mkdir -p logs

echo "============================================================"
echo "FedTGP Client $CLIENT_ID Starting"
echo "============================================================"
echo "Config file: $CONFIG_FILE"
echo "Client ID: $CLIENT_ID"
echo "Logs will be saved to: logs/"
echo "============================================================"

python fedml_main.py \
    --config $CONFIG_FILE \
    --client_mode \
    --client_id $CLIENT_ID

echo ""
echo "============================================================"
echo "FedTGP Client $CLIENT_ID stopped"
echo "============================================================"
