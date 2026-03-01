# TensorBoard使用指南

## 启动TensorBoard

### 方法1: 使用启动脚本（推荐）

```bash
bash scripts/utils/start_tensorboard.sh
```

默认配置：
- 日志目录: `logs/`
- 端口: `6006`

### 方法2: 自定义参数

```bash
bash scripts/utils/start_tensorboard.sh <log_dir> <port>

# 示例
bash scripts/utils/start_tensorboard.sh logs 6007
```

### 方法3: 直接启动

```bash
tensorboard --logdir=logs --port=6006 --host=0.0.0.0
```

## 访问TensorBoard

启动后，在浏览器中访问：
- 本地: `http://localhost:6006`
- 远程服务器: `http://<服务器IP>:6006`

## 查看训练曲线

TensorBoard会自动记录：
- **Loss/train**: 训练损失（每个epoch）
- **Loss/val**: 验证损失（每个epoch）
- **Learning_Rate**: 学习率变化

## 日志目录结构

```
logs/
├── tst1/          # TST1预训练日志
│   └── events.out.tfevents.*
└── tst2/          # TST2预训练日志
    └── events.out.tfevents.*
```

## 停止TensorBoard

按 `Ctrl+C` 停止服务
