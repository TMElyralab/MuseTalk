#!/bin/bash

# 源目录
SOURCE_DIR="/mnt/f/Models_Download/MuseTalk"

# 目标目录（项目中的 models 目录）
TARGET_DIR="/home/smy/LM_projects/agentscope/MuseTalk/models"

echo "========================================="
echo "创建 MuseTalk 模型软链接"
echo "========================================="
echo "源目录: $SOURCE_DIR"
echo "目标目录: $TARGET_DIR"
echo "========================================="

# 检查源目录是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 源目录不存在: $SOURCE_DIR"
    exit 1
fi

# 创建目标目录（如果不存在）
if [ ! -d "$TARGET_DIR" ]; then
    echo "创建目标目录: $TARGET_DIR"
    mkdir -p "$TARGET_DIR"
fi

# 遍历源目录中的所有文件夹
count=0
for folder in "$SOURCE_DIR"/*/; do
    # 检查是否是有效的目录
    if [ -d "$folder" ]; then
        # 获取文件夹名称
        folder_name=$(basename "$folder")
        
        # 跳过隐藏文件夹（以 . 开头）
        if [[ "$folder_name" == .* ]]; then
            echo "跳过隐藏文件夹: $folder_name"
            continue
        fi
        
        # 检查目标链接是否已存在
        if [ -e "$TARGET_DIR/$folder_name" ] || [ -L "$TARGET_DIR/$folder_name" ]; then
            echo "警告: 目标已存在，跳过: $folder_name"
            continue
        fi
        
        # 创建绝对路径
        source_absolute_path=$(cd "$folder" && pwd)
        
        # 创建软链接
        ln -s "$source_absolute_path" "$TARGET_DIR/$folder_name"
        if [ $? -eq 0 ]; then
            echo "✓ 创建软链接: $folder_name -> $source_absolute_path"
            count=$((count + 1))
        else
            echo "✗ 创建软链接失败: $folder_name"
        fi
    fi
done

echo "========================================="
echo "完成！共创建了 $count 个软链接"
echo "========================================="

# 显示创建的软链接
echo ""
echo "创建的软链接列表:"
ls -la "$TARGET_DIR"
