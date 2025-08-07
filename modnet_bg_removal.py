"""
MODNet背景去除集成 - 轻量级替代backgroundremover
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
import os
import tempfile
from PIL import Image
import subprocess
import warnings

# MODNet模型路径 - 人像抠图专用
MODNET_MODEL_URL = "https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/models/modnet_photographic_portrait_matting.ckpt"
MODNET_MODEL_PATH = "/app/checkpoints/modnet/modnet_photographic_portrait_matting.ckpt"

class MODNetBackgroundRemover:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.download_model()
        self.load_model()
    
    def download_model(self):
        """下载MODNet模型 - 使用已知可用的模型"""
        if not os.path.exists(MODNET_MODEL_PATH):
            os.makedirs(os.path.dirname(MODNET_MODEL_PATH), exist_ok=True)
            print(f"[MODNET] Checking for MODNet model...")
            
            # 检查Dockerfile是否已经下载了模型
            if os.path.exists(MODNET_MODEL_PATH) and os.path.getsize(MODNET_MODEL_PATH) > 1000:
                print(f"[MODNET] Model already exists: {MODNET_MODEL_PATH}")
                return
            
            print(f"[MODNET] Model not found or too small, attempting download...")
            
            # 使用多个备用URL
            urls = [
                "https://github.com/ZHKKKe/MODNet/releases/download/v1.0/modnet_webcam_portrait_matting.ckpt",
                "https://huggingface.co/spaces/akhaliq/SMODNet/resolve/main/modnet_photographic_portrait_matting.ckpt"
            ]
            
            for url in urls:
                try:
                    print(f"[MODNET] Trying: {url}")
                    subprocess.run(["curl", "-L", "-f", "--retry", "3", "--max-time", "300", 
                                   url, "-o", MODNET_MODEL_PATH], check=True)
                    
                    # 检查文件大小
                    if os.path.getsize(MODNET_MODEL_PATH) > 1000000:  # 至少1MB
                        print(f"[MODNET] Model downloaded successfully from {url}")
                        return
                    else:
                        print(f"[MODNET] Downloaded file too small, trying next source")
                        
                except Exception as e:
                    print(f"[MODNET] Failed to download from {url}: {e}")
                    continue
            
            print(f"[MODNET] WARNING: Could not download MODNet model from any source")
            print(f"[MODNET] Background removal will be disabled")
    
    def load_model(self):
        """加载MODNet模型"""
        try:
            # 使用本地MODNet源码
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            modnet_path = os.path.join(current_dir, 'modnet_src')
            if os.path.exists(modnet_path):
                sys.path.insert(0, modnet_path)
            
            # 直接从modnet_src导入MODNet
            from modnet_src.models.modnet import MODNet
            self.model = MODNet(backbone_pretrained=False)
            
            # 加载检查点并处理键名不匹配问题
            checkpoint = torch.load(MODNET_MODEL_PATH, map_location=self.device)
            
            # 处理检查点格式
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 移除'module.'前缀（如果存在）
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('module.', '')
                new_state_dict[new_key] = value
            
            # 加载处理后的权重
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            print("MODNet model loaded successfully")
        except ImportError as e:
            print(f"MODNet import error: {e}")
            raise
    
    def remove_background_from_frame(self, frame):
        """对单帧图像去除背景"""
        print(f"[MODNET FRAME] Processing frame of size: {frame.shape}")
        
        # 转换图像
        h, w = frame.shape[:2]
        
        try:
            # 预处理
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            
            print(f"[MODNET FRAME] Input tensor shape: {img_tensor.shape}")
            
            # 推理
            with torch.no_grad():
                _, _, matte = self.model(img_tensor, inference=True)
            
            print(f"[MODNET FRAME] Matte shape after model: {matte.shape}")
            
            # 后处理
            matte = F.interpolate(matte, size=(h, w), mode='bilinear', align_corners=False)
            matte = matte[0][0].cpu().numpy()
            
            print(f"[MODNET FRAME] Final matte shape: {matte.shape}, min: {matte.min():.3f}, max: {matte.max():.3f}")
            
            # 创建透明背景图像
            rgba = np.dstack([frame, (matte * 255).astype(np.uint8)])
            
            print(f"[MODNET FRAME] Output RGBA shape: {rgba.shape}")
            return rgba
            
        except Exception as e:
            print(f"[MODNET FRAME] ERROR in remove_background_from_frame: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def process_video(self, input_path, output_path, background_color=(255, 255, 255)):
        """处理视频文件，将背景替换为白色"""
        print(f"[MODNET] Starting background removal process...")
        print(f"[MODNET] Input video: {input_path}")
        print(f"[MODNET] Output path: {output_path}")
        
        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            print(f"[MODNET] ERROR: Input file {input_path} does not exist!")
            return None
            
        # 读取视频
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"[MODNET] ERROR: Cannot open video file {input_path}")
            return None
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"[MODNET] Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # 创建输出视频 - 使用H.264编码以获得更好的浏览器兼容性
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264编码
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
        
        # 如果H.264失败，回退到更通用的编码
        if not out.isOpened():
            print("[MODNET] H.264 failed, trying mp4v...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
        
        if not out.isOpened():
            print(f"[MODNET] ERROR: Cannot create output video file {output_path}")
            return None
        
        frame_count = 0
        
        print(f"[MODNET] Starting frame processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 处理每一帧
            try:
                rgba_frame = self.remove_background_from_frame(frame)
                
                # 创建白色背景
                white_background = np.ones_like(frame) * 255
                
                # 提取alpha通道作为mask
                alpha = rgba_frame[:, :, 3] / 255.0
                alpha = alpha[:, :, np.newaxis]
                
                # 将前景（人物）与白色背景合成
                foreground = rgba_frame[:, :, :3]
                result_frame = (foreground * alpha + white_background * (1 - alpha)).astype(np.uint8)
                
                out.write(result_frame)
                frame_count += 1
                
                if frame_count % 10 == 0:
                    print(f"[MODNET] Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
                    
            except Exception as e:
                print(f"[MODNET] ERROR processing frame {frame_count}: {e}")
                continue
        
        cap.release()
        out.release()
        
        print(f"[MODNET] COMPLETED: Processed {frame_count} frames")
        print(f"[MODNET] Output saved to: {output_path}")
        
        # 使用FFmpeg重新编码确保浏览器兼容性
        temp_output = output_path.replace('.mp4', '_temp.mp4')
        os.rename(output_path, temp_output)
        
        try:
            print("[MODNET] Re-encoding with FFmpeg for browser compatibility...")
            subprocess.run([
                'ffmpeg', '-i', temp_output, '-c:v', 'libx264', 
                '-preset', 'medium', '-crf', '23', '-movflags', '+faststart',
                '-y', output_path
            ], check=True)
            os.remove(temp_output)
            print("[MODNET] FFmpeg re-encoding completed")
        except subprocess.CalledProcessError as e:
            print(f"[MODNET] FFmpeg re-encoding failed: {e}, using original")
            os.rename(temp_output, output_path)
        except FileNotFoundError:
            print("[MODNET] FFmpeg not found, using original video")
            os.rename(temp_output, output_path)
        
        return output_path
    
    def process_video_with_transparency(self, input_path, output_path):
        """处理视频并保留透明度（输出为mov格式）"""
        print(f"Processing video with transparency: {input_path}")
        
        # 使用FFmpeg处理透明度
        temp_dir = tempfile.mkdtemp()
        output_mov = output_path.replace('.mp4', '.mov')
        
        # 逐帧处理并保存为PNG序列
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        frame_idx = 0
        png_files = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgba_frame = self.remove_background_from_frame(frame)
            png_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(png_path, rgba_frame)
            png_files.append(png_path)
            frame_idx += 1
        
        cap.release()
        
        # 使用FFmpeg合并PNG序列为MOV
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(temp_dir, "frame_%06d.png"),
            "-c:v", "qtrle",  # QuickTime Animation codec for transparency
            "-pix_fmt", "argb",
            output_mov
        ]
        
        subprocess.run(cmd, check=True)
        
        # 清理临时文件
        for png_file in png_files:
            os.remove(png_file)
        os.rmdir(temp_dir)
        
        return output_mov

# 全局实例
_modnet_remover = None

def get_modnet_remover():
    """获取全局MODNet移除器实例"""
    global _modnet_remover
    if _modnet_remover is None:
        _modnet_remover = MODNetBackgroundRemover()
    return _modnet_remover

def remove_background_modnet(input_path, output_path, transparent=False):
    """简化接口：使用MODNet去除背景"""
    remover = get_modnet_remover()
    
    if transparent:
        return remover.process_video_with_transparency(input_path, output_path)
    else:
        return remover.process_video(input_path, output_path)