#!/usr/bin/env python3
"""
测试背景去除功能的脚本
"""

import subprocess
import os
import sys

def test_background_removal():
    """测试背景去除功能"""
    
    # 检查backgroundremover是否已安装
    try:
        result = subprocess.run(["backgroundremover", "--help"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("backgroundremover未正确安装，请先安装：")
            print("pip install backgroundremover")
            return False
    except FileNotFoundError:
        print("backgroundremover未找到，请先安装：")
        print("pip install backgroundremover")
        return False
    
    # 测试用输入文件（使用项目自带的demo文件）
    test_video = "assets/demo1_video.mp4"
    test_audio = "assets/demo1_audio.wav"
    
    if not os.path.exists(test_video) or not os.path.exists(test_audio):
        print("测试文件不存在，请确保demo文件在assets/目录下")
        return False
    
    # 运行带背景去除的对嘴型处理
    print("开始对嘴型处理并去除背景...")
    
    cmd = [
        sys.executable, "-m", "scripts.inference",
        "--unet_config_path", "configs/unet/stage2_512.yaml",
        "--inference_ckpt_path", "checkpoints/latentsync_unet.pt",
        "--video_path", test_video,
        "--audio_path", test_audio,
        "--video_out_path", "/tmp/test_output.mp4",
        "--inference_steps", "10",  # 测试用少量步骤
        "--remove_background"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("背景去除测试完成！")
        print("输出文件：/tmp/test_output_no_bg.mov")
        return True
    except subprocess.CalledProcessError as e:
        print(f"测试失败：{e}")
        return False

if __name__ == "__main__":
    test_background_removal()