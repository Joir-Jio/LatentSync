#!/usr/bin/env python3
"""
测试背景移除功能
"""
import os
import sys
import cv2
import numpy as np

# 将当前目录添加到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from modnet_bg_removal import MODNetBackgroundRemover

def test_background_removal():
    """测试背景移除功能"""
    print("[TEST] 开始测试背景移除功能...")
    
    try:
        # 创建背景移除器
        print("[TEST] 创建MODNet背景移除器...")
        remover = MODNetBackgroundRemover(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # 检查模型是否存在
        model_path = "/app/checkpoints/modnet/modnet_photographic_portrait_matting.ckpt"
        if not os.path.exists(model_path):
            print(f"[TEST] 模型文件不存在: {model_path}")
            return False
        
        print(f"[TEST] 模型文件存在: {model_path}")
        
        # 创建测试图像（简单的彩色图像）
        test_image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        test_image[100:400, 100:400] = [255, 0, 0]  # 红色方块
        
        print("[TEST] 创建测试图像...")
        
        # 测试背景移除
        print("[TEST] 运行背景移除...")
        result = remover.remove_background_from_frame(test_image)
        
        print(f"[TEST] 背景移除完成，结果形状: {result.shape}")
        
        # 保存测试图像
        output_path = "/tmp/test_bg_removal.png"
        cv2.imwrite(output_path, result)
        print(f"[TEST] 结果保存到: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"[TEST] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import torch
    success = test_background_removal()
    if success:
        print("[TEST] 背景移除功能测试通过!")
    else:
        print("[TEST] 背景移除功能测试失败!")