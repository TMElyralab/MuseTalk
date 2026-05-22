import os

def reset_mmdet_init():
    # 明确指定 site-packages 路径
    target_file = "/root/miniconda3/envs/musetalk/lib/python3.10/site-packages/mmdet/__init__.py"
    
    if not os.path.exists(target_file):
        print(f"Error: 找不到文件 {target_file}")
        return

    # 定义极其精简且无害的 __init__.py 内容
    # 移除了所有版本检查逻辑
    minimal_content = """__version__ = '3.3.0'

def digit_version(v_str):
    return [int(x) for x in v_str.split('.') if x.isdigit()]
"""

    try:
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(minimal_content)
        print(f"✅ 成功重写: {target_file}")
        print("🚀 已彻底移除 MMCV 版本检查。")
    except Exception as e:
        print(f"❌ 重写失败: {e}")

if __name__ == "__main__":
    reset_mmdet_init()
