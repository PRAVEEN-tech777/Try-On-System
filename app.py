import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import math
import torch
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor, 
    AutoModelForDepthEstimation,
    SegformerImageProcessor, 
    AutoModelForSemanticSegmentation
)
from rembg import remove
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import logging
import onnxruntime as ort
import os
import requests

# ==========================================
# CONFIGURATION
# ==========================================

@dataclass
class TryOnConfig:
    """Configurable parameters for try-on pipeline"""
    # Sizing (relative to pupillary distance)
    necklace_width_ratio: float = 3.3
    earring_height_ratio: float = 0.75
    nosepin_width_ratio: float = 0.15
    
    # Positioning offsets
    necklace_y_offset: float = 2.2  # Below chin
    earring_y_offset: float = 0.1   # Below ear lobe
    
    # Blending parameters
    color_match_strength: float = 0.35
    edge_feather_radius: int = 3
    shadow_intensity: float = 0.2
    reflection_intensity: float = 0.3
    
    # Depth warping
    necklace_curve_amplitude: float = 0.25
    
    # Quality settings
    min_face_detection_confidence: float = 0.5
    use_gpu: bool = True

# ==========================================
# 1. MATTING ENGINE (Auto-Downloading)
# ==========================================

class MattingEngine:
    """
    Uses MODNet for strand-level hair matting.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None
        self.enabled = False
        self.input_name = None

        # 1. Check if YOUR specific file exists
        if not os.path.exists(self.model_path):
            logging.error(f"❌ Could not find model at: {self.model_path}")
            logging.warning("⚠️ Hair matting will be DISABLED.")
            return

        # 2. Load ONNX Session
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.enabled = True
            logging.info(f"✅ MODNet loaded from: {self.model_path}")
        except Exception as e:
            logging.warning(f"⚠️ Failed to load Matting Engine: {e}")
            self.enabled = False

    # (Keep the get_hair_matte function the same as before)
    def get_hair_matte(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        if not self.enabled or self.session is None:
            return None
        try:
            h, w = img_rgb.shape[:2]
            img_resized = cv2.resize(img_rgb, (512, 512))
            img_trans = (img_resized.astype('float32') / 127.5) - 1.0
            img_trans = np.transpose(img_trans, (2, 0, 1))
            img_trans = img_trans[np.newaxis, ...]
            matte = self.session.run(None, {self.input_name: img_trans})[0]
            matte = matte[0][0]
            matte = cv2.resize(matte, (w, h))
            return matte
        except Exception as e:
            logging.error(f"Matting inference error: {e}")
            return None
    """
    Uses MODNet for strand-level hair matting.
    Tries multiple mirrors to auto-download 'modnet.onnx'.
    """
    def __init__(self, model_path="modnet.onnx"):
        self.model_path = model_path
        self.session = None
        self.enabled = False
        self.input_name = None

        # 1. Check if file exists; if not, download
        if not os.path.exists(self.model_path) or os.path.getsize(self.model_path) < 1000:
            logging.info(f"⬇️ Matting model missing. Attempting download...")
            success = self._download_model_robust()
            if not success:
                logging.warning("⚠️ All download mirrors failed. Hair matting will be DISABLED.")
                return

        # 2. Load ONNX Session
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.enabled = True
            logging.info("✅ MODNet Matting Engine Loaded Successfully")
        except Exception as e:
            logging.warning(f"⚠️ Failed to initialize Matting Engine: {e}")
            self.enabled = False

    def _download_model_robust(self) -> bool:
        """Tries multiple mirrors to find the model"""
        
        # List of public mirrors for MODNet ONNX
        mirrors = [
            "https://huggingface.co/kirp/modnet/resolve/main/modnet_photographic_portrait_matting.onnx",
            "https://github.com/R3AP3/MODNet-ONNX/releases/download/v1.0/modnet.onnx",
            "https://huggingface.co/brekkie/modnet/resolve/main/modnet_photographic_portrait_matting.onnx"
        ]
        
        headers = {'User-Agent': 'Mozilla/5.0'} # Pretend to be a browser
        
        for url in mirrors:
            try:
                logging.info(f"Trying mirror: {url}...")
                with st.spinner(f"Downloading Matting Model from Mirror {mirrors.index(url)+1}..."):
                    response = requests.get(url, stream=True, headers=headers, timeout=30)
                    if response.status_code == 200:
                        with open(self.model_path, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        logging.info("✅ Download complete.")
                        return True
                    else:
                        logging.warning(f"Mirror returned status {response.status_code}")
            except Exception as e:
                logging.warning(f"Mirror failed: {e}")
                
        return False

    def get_hair_matte(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        if not self.enabled or self.session is None:
            return None
            
        try:
            h, w = img_rgb.shape[:2]
            # Preprocess
            img_resized = cv2.resize(img_rgb, (512, 512))
            img_trans = (img_resized.astype('float32') / 127.5) - 1.0
            img_trans = np.transpose(img_trans, (2, 0, 1))
            img_trans = img_trans[np.newaxis, ...]

            # Inference
            matte = self.session.run(None, {self.input_name: img_trans})[0]
            matte = matte[0][0]
            
            # Resize result
            matte = cv2.resize(matte, (w, h))
            return matte
        except Exception as e:
            logging.error(f"Matting inference error: {e}")
            return None
# ==========================================
# 2. MODEL MANAGER
# ==========================================

class ModelManager:
    """Centralized model loading and caching"""
    
    @staticmethod
    @st.cache_resource
    def load_all_models(config: TryOnConfig):
        device = "cuda" if (torch.cuda.is_available() and config.use_gpu) else "cpu"
        logging.info(f"🚀 Loading AI Models on {device}...")
        
        models = {}
        
        try:
            # 1. Depth Estimation
            models['depth_processor'] = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
            models['depth_model'] = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf").to(device)
            
            # 2. Semantic Segmentation
            models['seg_processor'] = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
            models['seg_model'] = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(device)
            
            # 3. MediaPipe
            models['face_mesh'] = mp.solutions.face_mesh
            models['pose'] = mp.solutions.pose
            
            # 4. Matting Engine (UPDATED WITH YOUR PATH)
            # We use r"..." to handle Windows backslashes correctly
            local_model_path = r"C:\Users\praveenraja\Downloads\chatgpt\src\app_pro_tryon_v2.py\models\modnet.onnx"
            
            models['matting_engine'] = MattingEngine(model_path=local_model_path)
            
            models['device'] = device
            logging.info("✅ All models loaded successfully")
            
        except Exception as e:
            logging.error(f"❌ Model loading failed: {e}")
            st.error(f"Model loading failed: {e}")
            raise
        
        return models
# ==========================================
# 3. UTILITIES & ENGINES
# ==========================================

class ImagePreprocessor:
    @staticmethod
    def remove_background(img_array: np.ndarray) -> np.ndarray:
        try:
            if img_array.shape[2] == 3:
                img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            else:
                img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGBA))
            output = remove(img_pil)
            return np.array(output)
        except Exception:
            if img_array.shape[2] == 3:
                return cv2.cvtColor(img_array, cv2.COLOR_BGR2BGRA)
            return img_array
    
    @staticmethod
    def ensure_rgba(img: np.ndarray) -> np.ndarray:
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img
    
    @staticmethod
    def apply_edge_feather(img_rgba: np.ndarray, radius: int = 3) -> np.ndarray:
        if img_rgba.shape[2] != 4: return img_rgba
        alpha = img_rgba[:, :, 3]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
        alpha_eroded = cv2.erode(alpha, kernel, iterations=1)
        alpha_soft = cv2.GaussianBlur(alpha_eroded, (radius*2+1, radius*2+1), 0)
        result = img_rgba.copy()
        result[:, :, 3] = alpha_soft
        return result

class DepthGeometryEngine:
    def __init__(self, models: Dict):
        self.depth_processor = models['depth_processor']
        self.depth_model = models['depth_model']
        self.device = models['device']
    
    def compute_depth_map(self, img_pil: Image.Image) -> np.ndarray:
        try:
            inputs = self.depth_processor(images=img_pil, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
            prediction = F.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=img_pil.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            depth = prediction.squeeze().cpu().numpy()
            return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        except Exception:
            return np.ones(img_pil.size[::-1], dtype=np.float32)
    
    def cylindrical_warp(self, jewel_img, depth_map, center_x, center_y, width, height, curve_amplitude=0.25):
        try:
            h_map, w_map = depth_map.shape
            if width <= 0 or height <= 0: return cv2.resize(jewel_img, (max(1, width), max(1, height)))
            
            x_start = max(0, center_x - width//2)
            x_end = min(w_map, center_x + width//2)
            y_sample = np.clip(center_y, 0, h_map - 1)
            
            if x_end <= x_start: return cv2.resize(jewel_img, (width, height))
            
            depth_slice = depth_map[y_sample, x_start:x_end]
            curve_profile = cv2.GaussianBlur((1.0 - depth_slice).reshape(1, -1), (25, 1), 0).flatten()
            
            jewel_resized = cv2.resize(jewel_img, (width, height), interpolation=cv2.INTER_AREA)
            
            if len(curve_profile) != width:
                curve_profile = cv2.resize(curve_profile.reshape(1, -1), (width, 1)).flatten()
            
            map_x = np.zeros((height, width), dtype=np.float32)
            map_y = np.zeros((height, width), dtype=np.float32)
            grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
            vertical_displacement = height * curve_amplitude
            
            for i in range(width):
                shift = curve_profile[i] * vertical_displacement
                map_y[:, i] = grid_y[:, i] - shift
                map_x[:, i] = grid_x[:, i]
            
            return cv2.remap(jewel_resized, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
        except Exception:
            return cv2.resize(jewel_img, (width, height))

class OcclusionAnalyzer:
    def __init__(self, models: Dict):
        self.seg_processor = models['seg_processor']
        self.seg_model = models['seg_model']
        self.device = models['device']
    
    def get_segmentation_masks(self, img_pil: Image.Image) -> Dict[str, np.ndarray]:
        try:
            inputs = self.seg_processor(images=img_pil, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.seg_model(**inputs)
            logits = outputs.logits.cpu()
            upsampled = F.interpolate(logits, size=img_pil.size[::-1], mode="bilinear", align_corners=False)
            seg_map = upsampled.argmax(dim=1)[0].numpy()
            return {
                'hair': (seg_map == 2).astype(np.uint8) * 255,
                'clothes': (seg_map == 4).astype(np.uint8) * 255,
                'skin': ((seg_map == 11) | (seg_map == 14) | (seg_map == 15)).astype(np.uint8) * 255
            }
        except Exception:
            h, w = img_pil.size[::-1]
            return {'hair': np.zeros((h, w)), 'clothes': np.zeros((h, w))}

@dataclass
class AnatomyMetrics:
    method: str
    pd_px: float
    center_x: int
    center_y: int
    roll_angle: float
    landmarks: any

class AnatomicalDetector:
    def __init__(self, models: Dict, config: TryOnConfig):
        self.mp_face = models['face_mesh']
        self.mp_pose = models['pose']
        self.config = config
    
    def detect_anatomy(self, img_rgb: np.ndarray) -> Optional[AnatomyMetrics]:
        h, w = img_rgb.shape[:2]
        
        with self.mp_face.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1, min_detection_confidence=self.config.min_face_detection_confidence) as face_mesh:
            result = face_mesh.process(img_rgb)
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                l_eye = np.array([lm[468].x * w, lm[468].y * h])
                r_eye = np.array([lm[473].x * w, lm[473].y * h])
                pd_px = np.linalg.norm(l_eye - r_eye)
                chin = lm[152]
                cx, cy = int(chin.x * w), int(chin.y * h)
                roll = math.degrees(math.atan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))
                return AnatomyMetrics(method='FaceMesh', pd_px=pd_px, center_x=cx, center_y=cy, roll_angle=roll, landmarks=lm)
        
        with self.mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=self.config.min_face_detection_confidence) as pose:
            result = pose.process(img_rgb)
            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                l_eye = np.array([lm[2].x * w, lm[2].y * h])
                r_eye = np.array([lm[5].x * w, lm[5].y * h])
                pd_px = np.linalg.norm(l_eye - r_eye)
                l_sh = np.array([lm[11].x * w, lm[11].y * h])
                r_sh = np.array([lm[12].x * w, lm[12].y * h])
                cx, cy = int((l_sh[0] + r_sh[0]) / 2), int((l_sh[1] + r_sh[1]) / 2)
                roll = math.degrees(math.atan2(r_eye[1] - l_eye[1], r_eye[0] - l_eye[0]))
                return AnatomyMetrics(method='Pose', pd_px=pd_px, center_x=cx, center_y=cy, roll_angle=roll, landmarks=lm)
        return None

# ==========================================
# 4. COMPOSITOR (Physics & Realism)
# ==========================================

class AdvancedCompositor:
    @staticmethod
    def match_colors_lab(foreground, background, strength=0.3):
        if background.size == 0 or foreground.size == 0: return foreground
        try:
            fg_lab = cv2.cvtColor(foreground, cv2.COLOR_BGR2LAB).astype(np.float32)
            bg_lab = cv2.cvtColor(background, cv2.COLOR_BGR2LAB).astype(np.float32)
            fg_lab[:,:,0] = np.clip(fg_lab[:,:,0] + (np.mean(bg_lab[:,:,0]) - np.mean(fg_lab[:,:,0])) * strength, 0, 255)
            return cv2.cvtColor(fg_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        except Exception: return foreground

    @staticmethod
    def generate_soft_shadow(jewelry_mask, offset_x=2, offset_y=3, blur_radius=5, intensity=0.15):
        h, w = jewelry_mask.shape[:2]
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        shadow = cv2.warpAffine(jewelry_mask, M, (w, h))
        shadow = cv2.GaussianBlur(shadow, (blur_radius*2+1, blur_radius*2+1), 0)
        return (shadow / 255.0) * intensity

    @staticmethod
    def apply_skin_pressure(bg_img: np.ndarray, jewel_mask: np.ndarray) -> np.ndarray:
        """Warps skin inward where jewelry sits"""
        h, w = bg_img.shape[:2]
        pressure_map = cv2.GaussianBlur(jewel_mask, (15, 15), 0).astype(np.float32) / 255.0
        displacement_y = pressure_map * 4.0
        
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_y = map_y.astype(np.float32) + displacement_y
        
        deformed_bg = cv2.remap(bg_img, map_x.astype(np.float32), map_y, interpolation=cv2.INTER_LINEAR)
        
        blend_mask = cv2.dilate(jewel_mask, np.ones((5,5)), iterations=2)
        blend_mask = cv2.GaussianBlur(blend_mask, (21, 21), 0).astype(np.float32) / 255.0
        blend_mask = np.dstack([blend_mask]*3)
        
        return ((deformed_bg * blend_mask) + (bg_img * (1.0 - blend_mask))).astype(np.uint8)

    @staticmethod
    def apply_fake_reflections(jewel_rgb: np.ndarray, bg_img: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """Adds environmental reflection to jewelry"""
        env_map = cv2.GaussianBlur(bg_img, (51, 51), 0)
        
        if env_map.shape[:2] != jewel_rgb.shape[:2]:
            env_map = cv2.resize(env_map, (jewel_rgb.shape[1], jewel_rgb.shape[0]))
            
        norm_jewel = jewel_rgb.astype(np.float32) / 255.0
        norm_env = env_map.astype(np.float32) / 255.0
        
        reflection = (1.0 - norm_jewel) * (norm_jewel * norm_env) + norm_jewel * (1.0 - (1.0 - norm_jewel) * (1.0 - norm_env))
        final = (reflection * 255).astype(np.uint8)
        return cv2.addWeighted(jewel_rgb, 1.0 - intensity, final, intensity, 0)

    @staticmethod
    def safe_overlay(background, foreground, x, y, config, occlusion_mask=None):
        try:
            h_bg, w_bg = background.shape[:2]
            h_fg, w_fg = foreground.shape[:2]
            y1_bg, y2_bg = max(0, y), min(h_bg, y + h_fg)
            x1_bg, x2_bg = max(0, x), min(w_bg, x + w_fg)
            y1_fg, y2_fg = max(0, -y), max(0, -y) + (y2_bg - y1_bg)
            x1_fg, x2_fg = max(0, -x), max(0, -x) + (x2_bg - x1_bg)
            
            if (y2_fg <= y1_fg) or (x2_fg <= x1_fg): return background
            
            fg_crop = foreground[y1_fg:y2_fg, x1_fg:x2_fg]
            bg_crop = background[y1_bg:y2_bg, x1_bg:x2_bg]
            
            if fg_crop.shape[2] == 4:
                alpha = fg_crop[:, :, 3] / 255.0
                fg_rgb = fg_crop[:, :, :3]
                
                fg_rgb_matched = AdvancedCompositor.match_colors_lab(fg_rgb, bg_crop, config.color_match_strength)
                shadow = AdvancedCompositor.generate_soft_shadow((alpha * 255).astype(np.uint8), intensity=config.shadow_intensity)
                
                bg_with_shadow = bg_crop.copy().astype(np.float32)
                for c in range(3): bg_with_shadow[:, :, c] *= (1.0 - shadow)
                
                alpha_3ch = cv2.merge([alpha, alpha, alpha])
                composite = (fg_rgb_matched * alpha_3ch) + (bg_with_shadow * (1.0 - alpha_3ch))
                
                if occlusion_mask is not None:
                    occ_crop = occlusion_mask[y1_bg:y2_bg, x1_bg:x2_bg]
                    if occ_crop.dtype == np.uint8:
                        occ_alpha = (occ_crop / 255.0)[:, :, np.newaxis]
                    else:
                        occ_alpha = occ_crop[:, :, np.newaxis]
                        
                    composite = (bg_crop * occ_alpha) + (composite * (1.0 - occ_alpha))
                
                background[y1_bg:y2_bg, x1_bg:x2_bg] = composite.astype(np.uint8)
            else:
                background[y1_bg:y2_bg, x1_bg:x2_bg] = fg_crop[:,:,:3]
            return background
        except Exception as e:
            return background

# ==========================================
# 5. MAIN ENHANCED ENGINE
# ==========================================

class EnhancedTryOnEngine:
    def __init__(self, models: Dict, config: TryOnConfig):
        self.config = config
        self.preprocessor = ImagePreprocessor()
        self.depth_engine = DepthGeometryEngine(models)
        self.occlusion = OcclusionAnalyzer(models)
        self.anatomy = AnatomicalDetector(models, config)
        self.compositor = AdvancedCompositor()
        self.matting = models.get('matting_engine', None)

    def process_necklace(self, model_img, jewel_img, metrics, depth_map):
        h, w = model_img.shape[:2]
        pd_px = metrics.pd_px
        necklace_width = int(pd_px * self.config.necklace_width_ratio)
        ar = jewel_img.shape[0] / jewel_img.shape[1]
        necklace_height = int(necklace_width * ar)
        
        center_y = int(metrics.center_y + (pd_px * self.config.necklace_y_offset)) if metrics.method == 'FaceMesh' else int(metrics.center_y + (pd_px * 0.5))
        center_x = metrics.center_x
        
        # 1. Geometry Warp
        warped_necklace = self.depth_engine.cylindrical_warp(
            jewel_img, depth_map, center_x, center_y, necklace_width, necklace_height, self.config.necklace_curve_amplitude
        )
        
        # 2. Reflections
        y_top = max(0, center_y - necklace_height // 2)
        y_bot = min(h, center_y + necklace_height // 2)
        x_left = max(0, center_x - necklace_width // 2)
        x_right = min(w, center_x + necklace_width // 2)
        
        bg_crop = model_img[y_top:y_bot, x_left:x_right]
        if bg_crop.size > 0 and warped_necklace.shape[2] == 4:
            rgb = warped_necklace[:,:,:3]
            alpha = warped_necklace[:,:,3]
            reflected_rgb = self.compositor.apply_fake_reflections(rgb, bg_crop, intensity=self.config.reflection_intensity)
            warped_necklace = cv2.merge([reflected_rgb, alpha])

        # 3. Skin Pressure
        full_mask = np.zeros((h, w), dtype=np.uint8)
        jewel_mask = warped_necklace[:,:,3]
        y1, y2 = max(0, center_y - necklace_height//2), min(h, center_y + necklace_height//2)
        x1, x2 = max(0, center_x - necklace_width//2), min(w, center_x + necklace_width//2)
        
        h_paste, w_paste = (y2-y1), (x2-x1)
        if h_paste > 0 and w_paste > 0:
            jy1 = max(0, -(center_y - necklace_height//2))
            jx1 = max(0, -(center_x - necklace_width//2))
            mask_crop = jewel_mask[jy1:jy1+h_paste, jx1:jx1+w_paste]
            full_mask[y1:y2, x1:x2] = mask_crop
            model_img = self.compositor.apply_skin_pressure(model_img, full_mask)

        # 4. Occlusion & Overlay
        img_pil = Image.fromarray(cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB))
        masks = self.occlusion.get_segmentation_masks(img_pil)
        
        result = self.compositor.safe_overlay(
            model_img.copy(),
            warped_necklace,
            center_x - necklace_width // 2,
            center_y - necklace_height // 2,
            self.config,
            occlusion_mask=masks['clothes']
        )
        return result

    def process_earrings(self, model_img, jewel_img, metrics):
        pd_px = metrics.pd_px
        target_height = int(pd_px * self.config.earring_height_ratio)
        ar = jewel_img.shape[1] / jewel_img.shape[0]
        target_width = int(target_height * ar)
        earring_resized = cv2.resize(jewel_img, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        positions = []
        if metrics.method == 'FaceMesh':
            lm = metrics.landmarks
            positions = [(int(lm[177].x * model_img.shape[1]), int(lm[177].y * model_img.shape[0])),
                         (int(lm[401].x * model_img.shape[1]), int(lm[401].y * model_img.shape[0]))]
        else:
             lm = metrics.landmarks
             positions = [(int(lm[7].x * model_img.shape[1]), int(lm[7].y * model_img.shape[0])),
                          (int(lm[8].x * model_img.shape[1]), int(lm[8].y * model_img.shape[0]))]

        # 1. Try MATTING
        hair_mask = None
        if self.matting and self.matting.enabled:
            hair_mask = self.matting.get_hair_matte(cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB))
            if hair_mask is not None:
                hair_mask = (hair_mask * 255).astype(np.uint8)
        
        if hair_mask is None:
            img_pil = Image.fromarray(cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB))
            masks = self.occlusion.get_segmentation_masks(img_pil)
            hair_mask = masks['hair']

        result = model_img.copy()
        for (lobe_x, lobe_y) in positions:
            x_pos = lobe_x - (target_width // 2)
            y_pos = lobe_y + int(pd_px * self.config.earring_y_offset)
            
            result = self.compositor.safe_overlay(
                result,
                earring_resized,
                x_pos,
                y_pos,
                self.config,
                occlusion_mask=hair_mask
            )
        return result
    
    def process_nosepin(self, model_img, jewel_img, metrics):
        if metrics.method == 'Pose': return model_img
        
        pd_px = metrics.pd_px
        pin_width = int(pd_px * self.config.nosepin_width_ratio)
        ar = jewel_img.shape[0] / jewel_img.shape[1]
        pin_height = int(pin_width * ar)
        pin_resized = cv2.resize(jewel_img, (pin_width, pin_height), interpolation=cv2.INTER_AREA)
        
        lm = metrics.landmarks
        nose_x = int(lm[48].x * model_img.shape[1])
        nose_y = int(lm[48].y * model_img.shape[0])
        
        return self.compositor.safe_overlay(
            model_img.copy(),
            pin_resized,
            nose_x - pin_width // 2,
            nose_y - pin_height // 2,
            self.config
        )

    def process(self, model_img, jewelry_dict):
        try:
            img_rgb = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            metrics = self.anatomy.detect_anatomy(img_rgb)
            if metrics is None: return model_img, "❌ Error: No person detected"
            
            result = model_img.copy()
            processed_items = []
            
            depth_map = None
            if 'Necklace' in jewelry_dict:
                depth_map = self.depth_engine.compute_depth_map(img_pil)

            if 'Necklace' in jewelry_dict:
                jewel = self.preprocessor.apply_edge_feather(self.preprocessor.ensure_rgba(self.preprocessor.remove_background(jewelry_dict['Necklace'])))
                result = self.process_necklace(result, jewel, metrics, depth_map)
                processed_items.append('Necklace')
            
            if 'Earrings' in jewelry_dict:
                jewel = self.preprocessor.apply_edge_feather(self.preprocessor.ensure_rgba(self.preprocessor.remove_background(jewelry_dict['Earrings'])))
                result = self.process_earrings(result, jewel, metrics)
                processed_items.append('Earrings')
            
            if 'Nose Pin' in jewelry_dict:
                jewel = self.preprocessor.apply_edge_feather(self.preprocessor.ensure_rgba(self.preprocessor.remove_background(jewelry_dict['Nose Pin'])))
                result = self.process_nosepin(result, jewel, metrics)
                processed_items.append('Nose Pin')

            return result, f"✅ Success: {', '.join(processed_items)}"
        except Exception as e:
            return model_img, f"❌ Error: {str(e)}"

# ==========================================
# 6. UI (MODIFIED WITH SIDE-BY-SIDE COMPARISON)
# ==========================================

def main():
    st.set_page_config(page_title="Enhanced AR Jewelry Try-On", layout="wide")
    st.title("💎 Professional Virtual Jewelry Try-On")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        config = TryOnConfig()
        
        config.necklace_width_ratio = st.slider("Necklace Width", 2.0, 4.0, 3.3, 0.1)
        config.earring_height_ratio = st.slider("Earring Size", 0.5, 1.2, 0.75, 0.05)
        st.divider()
        config.color_match_strength = st.slider("Color Matching", 0.0, 1.0, 0.35, 0.05)
        config.reflection_intensity = st.slider("Reflection", 0.0, 1.0, 0.3, 0.05)
        config.shadow_intensity = st.slider("Shadows", 0.0, 0.5, 0.2, 0.05)
    
    # Load models
    with st.spinner("🚀 Loading AI Models..."):
        models = ModelManager.load_all_models(config)
        engine = EnhancedTryOnEngine(models, config)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("📤 Upload")
        model_file = st.file_uploader("Model Photo", type=['jpg', 'jpeg', 'png'])
        jewelry_type = st.selectbox("Jewelry Type", ["Necklace", "Earrings", "Nose Pin", "Multiple Items"])
        
        jewelry_files = {}
        if jewelry_type == "Multiple Items":
            for item in ["Necklace", "Earrings", "Nose Pin"]:
                f = st.file_uploader(f"{item}", type=['png', 'jpg', 'jpeg'], key=f"jewel_{item}")
                if f: jewelry_files[item] = f
        else:
            f = st.file_uploader(f"{jewelry_type}", type=['png', 'jpg', 'jpeg'])
            if f: jewelry_files[jewelry_type] = f
    
    with col2:
        st.subheader("🎭 Results")
        if model_file and jewelry_files:
            model_img = cv2.cvtColor(np.array(Image.open(model_file).convert('RGB')), cv2.COLOR_RGB2BGR)
            jewelry_dict = {}
            for item_type, file in jewelry_files.items():
                img = Image.open(file).convert('RGBA')
                jewelry_dict[item_type] = np.array(img)
            
            # Process
            with st.spinner("🔮 Processing with AI..."):
                result, status = engine.process(model_img, jewelry_dict)
            
            # Display original and result side by side
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.image(
                    cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB),
                    use_column_width=True,
                    caption="📸 Original Photo"
                )
            
            with result_col2:
                st.image(
                    cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                    use_column_width=True,
                    caption="✨ Virtual Try-On Result"
                )
            
            if "Error" in status:
                st.error(status)
            else:
                st.success(status)
            
            # Download button
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_rgb)
            
            from io import BytesIO
            buf = BytesIO()
            result_pil.save(buf, format='PNG')
            
            st.download_button(
                label="⬇️ Download Result",
                data=buf.getvalue(),
                file_name="tryon_result.png",
                mime="image/png"
            )
        
        elif model_file:
            st.info("👆 Please upload jewelry asset(s)")
        else:
            st.info("👆 Please upload a model photo to begin")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    main()
