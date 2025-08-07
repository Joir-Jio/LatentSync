#!/usr/bin/env python3
"""
MODNet背景去除测试脚本
"""

import os
import sys
import tempfile
import subprocess

def test_modnet():
    """测试MODNet背景去除功能"""
    
    # 检查测试文件是否存在
    test_video = "assets/demo1_video.mp4"
    test_audio = "assets/demo1_audio.wav"
    
    if not os.path.exists(test_video) or not os.path.exists(test_audio):
        print("测试文件不存在，请确保demo文件在assets/目录下")
        return False
    
    print("🧪 开始MODNet背景去除测试...")
    
    try:
        # 确保MODNet模型已下载
        model_path = "checkpoints/modnet/modnet_photographic_portrait_matting.ckpt"
        if not os.path.exists(model_path):
            print("📥 下载MODNet模型...")
            os.makedirs("checkpoints/modnet", exist_ok=True)
            subprocess.run([
                "wget", "-O", model_path,
                "https://github.com/ZHKKKe/MODNet/releases/download/v1.0/modnet_photographic_portrait_matting.ckpt"
            ], check=True)
        
        # 运行带背景去除的对嘴型处理
        output_path = "/tmp/modnet_test_output.mp4"
        
        print("🎬 开始对嘴型处理 + MODNet背景去除...")
        cmd = [
            sys.executable, "-m", "scripts.inference",
            "--unet_config_path", "configs/unet/stage2_512.yaml",
            "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
            "--video_path", test_video,
            "--audio_path", test_audio,
            "--video_out_path", output_path,
            "--inference_steps", "5",  # 测试用少量步骤
            "--remove_background"
        ]
        
        subprocess.run(cmd, check=True)
        
        # 检查输出文件
        output_with_bg = output_path.replace('.mp4', '_no_bg.mp4')
        if os.path.exists(output_with_bg):
            # 获取文件大小
            file_size = os.path.getsize(output_with_bg) / (1024 * 1024)  # MB
            print(f"✅ MODNet测试成功！")
            print(f"📁 输出文件：{output_with_bg}")
            print(f"📊 文件大小：{file_size:.2f} MB")
            return True
        else:
            print("❌ 输出文件未生成")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 测试失败：{e}")
        return False
    except Exception as e:
        print(f"❌ 测试异常：{e}")
        return False

if __name__ == "__main__":
    test_modnet()