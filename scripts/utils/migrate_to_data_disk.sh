#!/bin/bash
# 将系统盘 (workplace) 的 checkpoints/logs/results/data 迁移到数据盘 (autodl-tmp)
# 系统盘仅保留 code 和 configs

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="/root/autodl-tmp/TwoTST"

if [ ! -d "$DATA_ROOT" ]; then
    echo "Data root $DATA_ROOT does not exist. Create it or run on AutoDL."
    exit 1
fi

echo "Migrating from $PROJECT_ROOT to $DATA_ROOT"

# 1. 合并 checkpoints
for d in checkpoints/finetune checkpoints/tst1 checkpoints/tst2; do
    src="$PROJECT_ROOT/$d"
    dst="$DATA_ROOT/$d"
    if [ -d "$src" ]; then
        mkdir -p "$dst"
        echo "Copying $src -> $dst"
        cp -rn "$src"/* "$dst"/ 2>/dev/null || true
    fi
done
if [ -d "$PROJECT_ROOT/checkpoints" ]; then
    mkdir -p "$DATA_ROOT/checkpoints"
    cp -rn "$PROJECT_ROOT/checkpoints"/* "$DATA_ROOT/checkpoints"/ 2>/dev/null || true
fi

# 2. 合并 logs
if [ -d "$PROJECT_ROOT/logs" ]; then
    mkdir -p "$DATA_ROOT/logs"
    cp -rn "$PROJECT_ROOT/logs"/* "$DATA_ROOT/logs"/ 2>/dev/null || true
fi

# 3. 合并 results
if [ -d "$PROJECT_ROOT/results" ]; then
    mkdir -p "$DATA_ROOT/results"
    cp -rn "$PROJECT_ROOT/results"/* "$DATA_ROOT/results"/ 2>/dev/null || true
fi

# 4. 迁移 data
if [ -d "$PROJECT_ROOT/data" ]; then
    mkdir -p "$DATA_ROOT/data"
    echo "Copying data (may take a while)..."
    cp -rn "$PROJECT_ROOT/data"/* "$DATA_ROOT/data"/ 2>/dev/null || true
fi

echo "Migration done. Removing from system disk..."

# 5. 删除系统盘上的大目录
rm -rf "$PROJECT_ROOT/checkpoints"
rm -rf "$PROJECT_ROOT/logs"
rm -rf "$PROJECT_ROOT/results"
rm -rf "$PROJECT_ROOT/data"

echo "Done. System disk now has: code + configs only."
