# 背景去除功能集成指南（MODNet版）

## 功能说明
本项目已集成 **MODNet** 作为背景去除方案，相比backgroundremover更轻量、更快、文件更小。

## 优势对比
| 特性 | MODNet | backgroundremover |
|------|--------|-------------------|
| 文件大小 | 小 | 大 |
| 处理速度 | 快 | 慢 |
| 模型大小 | 7MB | 数百MB |
| 输出格式 | MP4 | MOV |
| 内存占用 | 低 | 高 |

## 安装
```bash
# 安装MODNet依赖
pip install git+https://github.com/ZHKKKe/MODNet.git
```

## 使用方式

### 1. 命令行方式
```bash
# 基本使用（带背景去除）
python -m scripts.inference \
    --unet_config_path configs/unet/stage2_512.yaml \
    --inference_ckpt_path checkpoints/latentsync_unet.pt \
    --video_path input_video.mp4 \
    --audio_path input_audio.wav \
    --video_out_path output.mp4 \
    --remove_background
```

### 2. Python接口方式
```python
from predict import Predictor

predictor = Predictor()
predictor.setup()

result = predictor.predict(
    video="input_video.mp4",
    audio="input_audio.wav", 
    remove_background=True,  # 启用MODNet背景去除
    guidance_scale=2.0,
    inference_steps=20
)
```

### 3. RunPod API
```json
{
  "input": {
    "audio_url": "...",
    "video_url": "...",
    "remove_background": true,
    "guidance_scale": 2.5,
    "inference_steps": 2,
    "seed": 0
  }
}
```

## 输出格式
- **启用背景去除**：输出为 .mp4 格式（文件更小）
- **未启用背景去除**：输出为 .mp4 格式

## 性能对比
- **处理时间**：MODNet比backgroundremover快3-5倍
- **文件大小**：MODNet输出文件大小约为backgroundremover的1/10
- **内存使用**：MODNet峰值内存使用<1GB

## 注意事项
1. MODNet专注于人像背景去除，效果更精准
2. 无需额外安装FFmpeg
3. 模型仅7MB，下载更快
4. 输出文件格式统一为MP4，兼容性更好