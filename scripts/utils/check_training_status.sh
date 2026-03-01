#!/bin/bash
# 检查训练进度（数据盘：/root/autodl-tmp/TwoTST）
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/TwoTST}"
echo "=== TwoTST 训练状态 (data: $DATA_ROOT) ==="
echo ""
echo "预训练 (无滑窗):"
ls -la "$DATA_ROOT/checkpoints/tst1/"*.pt 2>/dev/null || echo "  未完成"
ls -la "$DATA_ROOT/checkpoints/tst2/"*.pt 2>/dev/null || echo "  未完成"
echo ""
echo "预训练 (滑窗):"
ls -la "$DATA_ROOT/checkpoints_sw/tst1/"*.pt 2>/dev/null || echo "  未完成"
ls -la "$DATA_ROOT/checkpoints_sw/tst2/"*.pt 2>/dev/null || echo "  未完成"
echo ""
echo "微调结果:"
ls "$DATA_ROOT/checkpoints/finetune/" 2>/dev/null | head -10
ls "$DATA_ROOT/checkpoints_sw/finetune/" 2>/dev/null | head -10
echo ""
echo "最新日志 (tail -50):"
tail -50 "$DATA_ROOT/results/pretrain.log" 2>/dev/null || echo "  日志未生成"
