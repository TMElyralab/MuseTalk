# MuseTalk 环境兼容性分析报告 (Gemini CLI) - V10 (终极方案: 原生适配 Blackwell)

## 1. 基础环境现状
- **操作系统**: Ubuntu 24.04.3 LTS (noble)
- **Python**: 3.10 (Conda 虚拟环境: musetalk)
- **CUDA**: 12.8 (Driver 570.211.01)
- **GPU**: NVIDIA GeForce RTX 5070 (12GB 显存)
- **FFmpeg**: 6.1.1 (系统已安装，状态良好)
- **系统库**: `libsndfile1` 已预装

## 2. 核心部署结论

- **硬件支持**: **完全支持**。12GB 显存满足 MuseTalk 推理需求，RTX 50 系列架构性能卓越。
- **软件策略**: **执行终极原生方案 (方案 B)**。
    - 采用原生支持 RTX 50 系列 Blackwell 架构 (`sm_120`) 的 `PyTorch 2.7.0 (cu128)`。由于硬件架构太新，旧版 PyTorch 2.4 的 PTX 无法向前兼容。
    - 为了适配 PyTorch 2.7.0，项目源码中的 8 处兼容性报错（`torch.load` 等）已被自动修改完毕。
    - 采用**源码硬核编译安装 `mmcv`** 路线，突破无预编译包的限制。
- **风险点控制**:
    - **NumPy 必须 < 2.0.0**：NumPy 2.x 会导致 OpenCV/Librosa 崩溃。
    - **Huggingface_hub == 0.36.2**：必须满足 transformers 4.39.2 (<1.0) 约束。
    - **Gradio == 5.24.0**：确保 UI 稳定性并兼容旧版 Pillow。
    - **Pillow 必须 < 10.0.0**：Pillow 10+ 移除了 `Image.ANTIALIAS`，会导致 `moviepy 1.0.3` 运行时直接崩溃。

## 3. 标准部署步骤 (用户态执行)

### A. 环境准备 (建议彻底重置)
为避免之前的失败安装残留幽灵依赖，强烈建议**先删后建**：
```bash
conda deactivate
conda remove -n musetalk --all -y
conda create -n musetalk python=3.10 -y
conda activate musetalk
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### B. 安装原生架构 PyTorch
```bash
# 安装原生满血支持 Blackwell (sm_120) 的最新引擎
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```
*(注意：MMCV 的安装已移至后续源码编译环节)*

### C. 预装关键冲突依赖 (重要顺序)
```bash
# 1. 预装构建工具并锁死 NumPy (Python 3.12 必需)
pip install --upgrade pip setuptools wheel "numpy<2.0.0" 

# 2. 解决 chumpy 编译问题 (避免 PEP 517 构建隔离报错)
pip install chumpy --no-build-isolation

# 3. 强行锁定三方黄金版本 (解决 transformers/gradio 冲突)
pip uninstall hf-gradio -y
pip install "huggingface_hub==0.36.2" "gradio==5.24.0" "gradio-client==1.8.0" --force-reinstall

# 4. 再次检查并加固 NumPy (防止被上面的安装带跑)
pip install "numpy<2.0.0"
```

### D. 安装 MMLab 核心组件 (使用原生 pip 源码硬核编译)
由于我们使用了 PyTorch 2.7.0 + CUDA 12.8，官方**绝对没有**可用的预编译包，必须进行硬核的源码编译：
```bash
pip install -U openmim ninja Cython
# 关键步骤：Ubuntu 24.04 默认 GCC 13 编译 CUDA 极易失败，必须使用 -allow-unsupported-compiler 放宽限制。
# 同时必须禁用编译隔离（--no-build-isolation），否则找不到构建依赖 pkg_resources！
NVCC_APPEND_FLAGS="-allow-unsupported-compiler" pip install "mmcv>=2.1.0" --no-build-isolation --no-cache-dir
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.3.0"
```

### E. 安装项目剩余依赖
```bash
pip install -r requirements-rtx5070.txt
```

## 4. 故障排查与修复记录 (Troubleshooting)

| 问题 | 原因 | 修复方式 |
|------|------|----------|
| `ModuleNotFoundError: No module named 'mmdet'` | `mmpose` 的核心依赖缺失 | 执行 `mim install "mmdet>=3.1.0"` |
| `ImportError: cannot import name 'FaceAlignment'` | 全局包冲突 | `pip uninstall face_detection -y`；代码已改为相对导入 |
| `ImportError: huggingface-hub...` | transformers 版本冲突 | 强行执行 `pip install "huggingface_hub==0.36.2" --force-reinstall` |
| `AttributeError: module 'numpy' has no attribute...` | 使用了 NumPy 2.x | 强行降级 `pip install "numpy<2.0.0"` |
| `chumpy` 安装失败 | 隔离构建模式问题 (setup.py import pip) | 使用 `--no-build-isolation` 且需提前装好 setuptools |
| `AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'` | Pillow 10+ 移除了该属性，与 moviepy 1.0.3 冲突 | 强制锁定 `pip install "pillow<10.0.0"` |

## 5. 验证与运行
1. **核心验证**: 运行 `python3 -c "import mmcv; from mmcv.ops import nms; print('验证成功: OK')"`
2. **下载权重**: `bash download_weights.sh`
3. **推理测试**: `bash inference.sh v1.5 normal`

---
*更新日期: 2026年5月8日 (V10 - 回到方案B，原生支持 RTX 5070 并完成源码改造)*
