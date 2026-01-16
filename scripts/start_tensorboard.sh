#!/bin/bash
# 启动TensorBoard服务

LOG_DIR=${1:-/root/workplace/exp/TwoTST/logs}
PORT=${2:-6006}

echo "Starting TensorBoard..."
echo "Log directory: $LOG_DIR"
echo "Port: $PORT"
echo ""
echo "Access TensorBoard at: http://localhost:$PORT"
echo "Press Ctrl+C to stop"
echo ""

tensorboard --logdir=$LOG_DIR --port=$PORT --host=0.0.0.0
