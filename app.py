"""
Enhanced Virtual Jewelry Try-On System
Combines best features from both codebases with photorealistic compositing
"""

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
    color_match_strength: float = 0.25
    edge_feather_radius: int = 3
    shadow_intensity: float = 0.15
    
    # Depth warping
    necklace_curve_amplitude: float = 0.25
    
    # Quality settings
    min_face_detection_confidence: float = 0.5
    use_gpu: bool = True

# ==========================================
# 1. MODEL MANAGER
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
            # 1. Depth Estimation (3D structure)
            models['depth_processor'] = AutoImageProcessor.from_pretrained(
                "LiheYoung/depth-anything-small-hf"
            )
            models['depth_model'] = AutoModelForDepthEstimation.from_pretrained(
                "LiheYoung/depth-anything-small-hf"
            ).to(device)
            
            # 2. Semantic Segmentation (occlusion)
            models['seg_processor'] = SegformerImageProcessor.from_pretrained(
                "mattmdjaga/segformer_b2_clothes"
            )
            models['seg_model'] = AutoModelForSemanticSegmentation.from_pretrained(
                "mattmdjaga/segformer_b2_clothes"
            ).to(device)
            
            # 3. Anatomical Landmarks
            models['face_mesh'] = mp.solutions.face_mesh
            models['pose'] = mp.solutions.pose
            
            models['device'] = device
            logging.info("✅ All models loaded successfully")
            
        except Exception as e:
            logging.error(f"❌ Model loading failed: {e}")
            raise
        
        return models

# ==========================================
# 2. IMAGE PREPROCESSING
# ==========================================

class ImagePreprocessor:
    """Handles jewelry asset preparation"""
    
    @staticmethod
    def remove_background(img_array: np.ndarray) -> np.ndarray:
        """
        Removes background using AI (U-2-Net via rembg)
        Handles both RGB and RGBA inputs
        """
        try:
            if img_array.shape[2] == 3:
                img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
            else:
                img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGRA2RGBA))
            
            output = remove(img_pil)
            return np.array(output)
        
        except Exception as e:
            logging.warning(f"Background removal failed: {e}, using original")
            # Ensure RGBA
            if img_array.shape[2] == 3:
                return cv2.cvtColor(img_array, cv2.COLOR_BGR2BGRA)
            return img_array
    
    @staticmethod
    def ensure_rgba(img: np.ndarray) -> np.ndarray:
        """Converts image to RGBA if needed"""
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img
    
    @staticmethod
    def apply_edge_feather(img_rgba: np.ndarray, radius: int = 3) -> np.ndarray:
        """
        Softens edges of alpha channel for natural blending
        """
        if img_rgba.shape[2] != 4:
            return img_rgba
        
        alpha = img_rgba[:, :, 3]
        
        # Erode slightly then blur for soft edge
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
        alpha_eroded = cv2.erode(alpha, kernel, iterations=1)
        alpha_soft = cv2.GaussianBlur(alpha_eroded, (radius*2+1, radius*2+1), 0)
        
        result = img_rgba.copy()
        result[:, :, 3] = alpha_soft
        return result

# ==========================================
# 3. DEPTH & GEOMETRY ENGINE
# ==========================================

class DepthGeometryEngine:
    """Handles 3D depth mapping and geometric warping"""
    
    def __init__(self, models: Dict):
        self.depth_processor = models['depth_processor']
        self.depth_model = models['depth_model']
        self.device = models['device']
    
    def compute_depth_map(self, img_pil: Image.Image) -> np.ndarray:
        """
        Computes normalized depth map (0=far, 1=close)
        """
        try:
            inputs = self.depth_processor(
                images=img_pil, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
            
            prediction = F.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=img_pil.size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            
            depth = prediction.squeeze().cpu().numpy()
            depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            
            return depth_norm
        
        except Exception as e:
            logging.warning(f"Depth estimation failed: {e}")
            # Return flat depth map
            return np.ones(img_pil.size[::-1], dtype=np.float32)
    
    def cylindrical_warp(
        self, 
        jewel_img: np.ndarray,
        depth_map: np.ndarray,
        center_x: int,
        center_y: int,
        width: int,
        height: int,
        curve_amplitude: float = 0.25
    ) -> np.ndarray:
        """
        Warps jewelry to follow 3D neck/body contour
        Creates realistic curvature effect
        """
        try:
            h_map, w_map = depth_map.shape
            
            if width <= 0 or height <= 0:
                return cv2.resize(jewel_img, (max(1, width), max(1, height)))
            
            # Sample horizontal slice of depth at necklace position
            x_start = max(0, center_x - width//2)
            x_end = min(w_map, center_x + width//2)
            y_sample = np.clip(center_y, 0, h_map - 1)
            
            if x_end <= x_start:
                return cv2.resize(jewel_img, (width, height))
            
            depth_slice = depth_map[y_sample, x_start:x_end]
            
            # Create curvature profile (inverted depth = displacement)
            curve_profile = 1.0 - depth_slice
            curve_profile = cv2.GaussianBlur(
                curve_profile.reshape(1, -1), 
                (25, 1), 
                0
            ).flatten()
            
            # Resize jewelry to target dimensions
            jewel_resized = cv2.resize(
                jewel_img, 
                (width, height), 
                interpolation=cv2.INTER_AREA
            )
            
            # Resize curve profile to match jewelry width
            if len(curve_profile) != width:
                curve_profile = cv2.resize(
                    curve_profile.reshape(1, -1), 
                    (width, 1)
                ).flatten()
            
            # Create displacement maps
            map_x = np.zeros((height, width), dtype=np.float32)
            map_y = np.zeros((height, width), dtype=np.float32)
            grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
            
            # Apply vertical displacement based on curve
            vertical_displacement = height * curve_amplitude
            
            for i in range(width):
                shift = curve_profile[i] * vertical_displacement
                map_y[:, i] = grid_y[:, i] - shift
                map_x[:, i] = grid_x[:, i]
            
            # Remap with smooth interpolation
            warped = cv2.remap(
                jewel_resized,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
            
            return warped
        
        except Exception as e:
            logging.warning(f"Warping failed: {e}")
            return cv2.resize(jewel_img, (width, height))

# ==========================================
# 4. OCCLUSION ANALYZER
# ==========================================

class OcclusionAnalyzer:
    """Detects hair, clothes, and skin for realistic layering"""
    
    # SegFormer B2 Clothes label mapping
    LABEL_MAP = {
        0: 'Background',
        1: 'Hat',
        2: 'Hair',
        3: 'Sunglasses',
        4: 'Upper-clothes',
        5: 'Skirt',
        6: 'Pants',
        7: 'Dress',
        8: 'Belt',
        9: 'Left-shoe',
        10: 'Right-shoe',
        11: 'Face',
        12: 'Left-leg',
        13: 'Right-leg',
        14: 'Left-arm',
        15: 'Right-arm',
        16: 'Bag',
        17: 'Scarf'
    }
    
    def __init__(self, models: Dict):
        self.seg_processor = models['seg_processor']
        self.seg_model = models['seg_model']
        self.device = models['device']
    
    def get_segmentation_masks(self, img_pil: Image.Image) -> Dict[str, np.ndarray]:
        """
        Returns dictionary of binary masks for different body parts/clothes
        """
        try:
            inputs = self.seg_processor(
                images=img_pil, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.seg_model(**inputs)
            
            logits = outputs.logits.cpu()
            upsampled = F.interpolate(
                logits,
                size=img_pil.size[::-1],
                mode="bilinear",
                align_corners=False
            )
            
            seg_map = upsampled.argmax(dim=1)[0].numpy()
            
            masks = {
                'hair': (seg_map == 2).astype(np.uint8) * 255,
                'clothes': (seg_map == 4).astype(np.uint8) * 255,
                'face': (seg_map == 11).astype(np.uint8) * 255,
                'skin': ((seg_map == 11) | (seg_map == 14) | (seg_map == 15)).astype(np.uint8) * 255
            }
            
            return masks
        
        except Exception as e:
            logging.warning(f"Segmentation failed: {e}")
            # Return empty masks
            h, w = img_pil.size[::-1]
            return {
                'hair': np.zeros((h, w), dtype=np.uint8),
                'clothes': np.zeros((h, w), dtype=np.uint8),
                'face': np.zeros((h, w), dtype=np.uint8),
                'skin': np.zeros((h, w), dtype=np.uint8)
            }

# ==========================================
# 5. ANATOMICAL DETECTOR
# ==========================================

@dataclass
class AnatomyMetrics:
    """Stores detected anatomical measurements"""
    method: str  # 'FaceMesh' or 'Pose'
    pd_px: float  # Pupillary distance in pixels
    center_x: int
    center_y: int
    roll_angle: float
    landmarks: any  # Raw landmark object for detailed access

class AnatomicalDetector:
    """Detects facial/body landmarks with smart fallback"""
    
    def __init__(self, models: Dict, config: TryOnConfig):
        self.mp_face = models['face_mesh']
        self.mp_pose = models['pose']
        self.config = config
    
    def detect_anatomy(self, img_rgb: np.ndarray) -> Optional[AnatomyMetrics]:
        """
        Attempts FaceMesh first, falls back to Pose if needed
        Returns None if both fail
        """
        h, w = img_rgb.shape[:2]
        
        # Strategy 1: Face Mesh (high precision for close-ups)
        with self.mp_face.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=self.config.min_face_detection_confidence
        ) as face_mesh:
            
            result = face_mesh.process(img_rgb)
            
            if result.multi_face_landmarks:
                lm = result.multi_face_landmarks[0].landmark
                
                # Iris centers (468=left, 473=right)
                l_eye = np.array([lm[468].x * w, lm[468].y * h])
                r_eye = np.array([lm[473].x * w, lm[473].y * h])
                pd_px = np.linalg.norm(l_eye - r_eye)
                
                # Chin (152)
                chin = lm[152]
                cx, cy = int(chin.x * w), int(chin.y * h)
                
                # Roll angle
                dy = r_eye[1] - l_eye[1]
                dx = r_eye[0] - l_eye[0]
                roll = math.degrees(math.atan2(dy, dx))
                
                return AnatomyMetrics(
                    method='FaceMesh',
                    pd_px=pd_px,
                    center_x=cx,
                    center_y=cy,
                    roll_angle=roll,
                    landmarks=lm
                )
        
        # Strategy 2: Pose Estimation (fallback for full body shots)
        logging.info("⚠️ FaceMesh failed, trying Pose Estimation...")
        
        with self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=self.config.min_face_detection_confidence
        ) as pose:
            
            result = pose.process(img_rgb)
            
            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                
                # Eyes (2=left, 5=right)
                l_eye = np.array([lm[2].x * w, lm[2].y * h])
                r_eye = np.array([lm[5].x * w, lm[5].y * h])
                pd_px = np.linalg.norm(l_eye - r_eye)
                
                # Roll angle
                dy = r_eye[1] - l_eye[1]
                dx = r_eye[0] - l_eye[0]
                roll = math.degrees(math.atan2(dy, dx))
                
                # Center point (clavicle area for necklaces)
                l_shoulder = np.array([lm[11].x * w, lm[11].y * h])
                r_shoulder = np.array([lm[12].x * w, lm[12].y * h])
                
                cx = int((l_shoulder[0] + r_shoulder[0]) / 2)
                cy = int((l_shoulder[1] + r_shoulder[1]) / 2)
                
                return AnatomyMetrics(
                    method='Pose',
                    pd_px=pd_px,
                    center_x=cx,
                    center_y=cy,
                    roll_angle=roll,
                    landmarks=lm
                )
        
        logging.error("❌ Both FaceMesh and Pose detection failed")
        return None

# ==========================================
# 6. ADVANCED COMPOSITOR
# ==========================================

class AdvancedCompositor:
    """Photorealistic image blending with color matching and occlusion"""
    
    @staticmethod
    def match_colors_lab(
        foreground: np.ndarray,
        background: np.ndarray,
        strength: float = 0.3
    ) -> np.ndarray:
        """
        Matches foreground colors to background lighting using LAB color space
        More perceptually accurate than RGB
        """
        if background.size == 0 or foreground.size == 0:
            return foreground
        
        try:
            # Convert to LAB
            fg_lab = cv2.cvtColor(foreground, cv2.COLOR_BGR2LAB).astype(np.float32)
            bg_lab = cv2.cvtColor(background, cv2.COLOR_BGR2LAB).astype(np.float32)
            
            # Match lightness (L channel)
            fg_mean_l = np.mean(fg_lab[:, :, 0])
            bg_mean_l = np.mean(bg_lab[:, :, 0])
            diff_l = bg_mean_l - fg_mean_l
            
            # Apply adjustment
            fg_lab[:, :, 0] = np.clip(
                fg_lab[:, :, 0] + (diff_l * strength),
                0,
                255
            )
            
            # Convert back
            result = cv2.cvtColor(fg_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            return result
        
        except Exception as e:
            logging.warning(f"Color matching failed: {e}")
            return foreground
    
    @staticmethod
    def generate_soft_shadow(
        jewelry_mask: np.ndarray,
        offset_x: int = 2,
        offset_y: int = 3,
        blur_radius: int = 5,
        intensity: float = 0.15
    ) -> np.ndarray:
        """
        Generates realistic soft shadow for jewelry
        """
        h, w = jewelry_mask.shape[:2]
        shadow = np.zeros((h, w), dtype=np.float32)
        
        # Shift mask to create shadow
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        shadow = cv2.warpAffine(jewelry_mask, M, (w, h))
        
        # Blur and reduce intensity
        shadow = cv2.GaussianBlur(shadow, (blur_radius*2+1, blur_radius*2+1), 0)
        shadow = (shadow / 255.0) * intensity
        
        return shadow
    
    @staticmethod
    def safe_overlay(
        background: np.ndarray,
        foreground: np.ndarray,
        x: int,
        y: int,
        config: TryOnConfig,
        occlusion_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Advanced overlay with:
        - Bounds checking
        - Color matching
        - Shadow generation
        - Occlusion handling
        """
        try:
            h_bg, w_bg = background.shape[:2]
            h_fg, w_fg = foreground.shape[:2]
            
            # Calculate valid regions
            y1_bg, y2_bg = max(0, y), min(h_bg, y + h_fg)
            x1_bg, x2_bg = max(0, x), min(w_bg, x + w_fg)
            
            y1_fg = max(0, -y)
            x1_fg = max(0, -x)
            y2_fg = y1_fg + (y2_bg - y1_bg)
            x2_fg = x1_fg + (x2_bg - x1_bg)
            
            # Check if completely offscreen
            if (y2_fg <= y1_fg) or (x2_fg <= x1_fg) or (y2_bg <= y1_bg) or (x2_bg <= x1_bg):
                return background
            
            # Extract regions
            fg_crop = foreground[y1_fg:y2_fg, x1_fg:x2_fg]
            bg_crop = background[y1_bg:y2_bg, x1_bg:x2_bg]
            
            if fg_crop.size == 0 or bg_crop.size == 0:
                return background
            
            # Handle RGBA foreground
            if fg_crop.shape[2] == 4:
                alpha = fg_crop[:, :, 3] / 255.0
                fg_rgb = fg_crop[:, :, :3]
                
                # Color matching
                fg_rgb_matched = AdvancedCompositor.match_colors_lab(
                    fg_rgb,
                    bg_crop,
                    config.color_match_strength
                )
                
                # Generate shadow
                shadow = AdvancedCompositor.generate_soft_shadow(
                    (alpha * 255).astype(np.uint8),
                    offset_x=2,
                    offset_y=3,
                    blur_radius=config.edge_feather_radius,
                    intensity=config.shadow_intensity
                )
                
                # Apply shadow to background
                bg_with_shadow = bg_crop.copy().astype(np.float32)
                for c in range(3):
                    bg_with_shadow[:, :, c] = bg_with_shadow[:, :, c] * (1.0 - shadow)
                
                # Alpha blend
                alpha_3ch = cv2.merge([alpha, alpha, alpha])
                composite = (fg_rgb_matched * alpha_3ch) + (bg_with_shadow * (1.0 - alpha_3ch))
                
                # Apply occlusion if provided
                if occlusion_mask is not None:
                    occ_crop = occlusion_mask[y1_bg:y2_bg, x1_bg:x2_bg]
                    if occ_crop.shape[:2] == composite.shape[:2]:
                        occ_alpha = (occ_crop / 255.0)[:, :, np.newaxis]
                        composite = (bg_crop * occ_alpha) + (composite * (1.0 - occ_alpha))
                
                background[y1_bg:y2_bg, x1_bg:x2_bg] = composite.astype(np.uint8)
            else:
                background[y1_bg:y2_bg, x1_bg:x2_bg] = fg_crop[:, :, :3]
            
            return background
        
        except Exception as e:
            logging.error(f"Overlay failed: {e}")
            return background

# ==========================================
# 7. MAIN PIPELINE
# ==========================================

class EnhancedTryOnEngine:
    """Complete virtual try-on pipeline"""
    
    def __init__(self, models: Dict, config: TryOnConfig):
        self.config = config
        self.preprocessor = ImagePreprocessor()
        self.depth_engine = DepthGeometryEngine(models)
        self.occlusion = OcclusionAnalyzer(models)
        self.anatomy = AnatomicalDetector(models, config)
        self.compositor = AdvancedCompositor()
    
    def process_necklace(
        self,
        model_img: np.ndarray,
        jewel_img: np.ndarray,
        metrics: AnatomyMetrics,
        depth_map: np.ndarray
    ) -> np.ndarray:
        """Processes and places necklace with 3D warping"""
        h, w = model_img.shape[:2]
        pd_px = metrics.pd_px
        
        # Calculate dimensions
        necklace_width = int(pd_px * self.config.necklace_width_ratio)
        ar = jewel_img.shape[0] / jewel_img.shape[1]
        necklace_height = int(necklace_width * ar)
        
        # Position
        if metrics.method == 'FaceMesh':
            center_y = int(metrics.center_y + (pd_px * self.config.necklace_y_offset))
        else:
            center_y = int(metrics.center_y + (pd_px * 0.5))
        
        center_x = metrics.center_x
        
        # Apply 3D warp
        warped_necklace = self.depth_engine.cylindrical_warp(
            jewel_img,
            depth_map,
            center_x,
            center_y,
            necklace_width,
            necklace_height,
            self.config.necklace_curve_amplitude
        )
        
        # Get occlusion masks
        img_pil = Image.fromarray(cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB))
        masks = self.occlusion.get_segmentation_masks(img_pil)
        
        # Overlay with occlusion
        result = self.compositor.safe_overlay(
            model_img.copy(),
            warped_necklace,
            center_x - necklace_width // 2,
            center_y - necklace_height // 2,
            self.config,
            occlusion_mask=masks['clothes']
        )
        
        return result
    
    def process_earrings(
        self,
        model_img: np.ndarray,
        jewel_img: np.ndarray,
        metrics: AnatomyMetrics
    ) -> np.ndarray:
        """Processes and places earrings with hair occlusion"""
        h, w = model_img.shape[:2]
        pd_px = metrics.pd_px
        
        # Calculate dimensions
        target_height = int(pd_px * self.config.earring_height_ratio)
        ar = jewel_img.shape[1] / jewel_img.shape[0]
        target_width = int(target_height * ar)
        
        # Resize
        earring_resized = cv2.resize(
            jewel_img,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA
        )
        
        # Get positions
        positions = []
        
        if metrics.method == 'FaceMesh':
            lm = metrics.landmarks
            # Left lobe (177), Right lobe (401)
            positions.append((int(lm[177].x * w), int(lm[177].y * h)))
            positions.append((int(lm[401].x * w), int(lm[401].y * h)))
        else:
            lm = metrics.landmarks
            # Left ear (7), Right ear (8)
            positions.append((int(lm[7].x * w), int(lm[7].y * h)))
            positions.append((int(lm[8].x * w), int(lm[8].y * h)))
        
        # Get hair mask
        img_pil = Image.fromarray(cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB))
        masks = self.occlusion.get_segmentation_masks(img_pil)
        
        result = model_img.copy()
        
        for (lobe_x, lobe_y) in positions:
            # Position: top-center of earring at lobe
            x_pos = lobe_x - (target_width // 2)
            y_pos = lobe_y + int(pd_px * self.config.earring_y_offset)
            
            result = self.compositor.safe_overlay(
                result,
                earring_resized,
                x_pos,
                y_pos,
                self.config,
                occlusion_mask=masks['hair']
            )
        
        return result
    
    def process_nosepin(
        self,
        model_img: np.ndarray,
        jewel_img: np.ndarray,
        metrics: AnatomyMetrics
    ) -> np.ndarray:
        """Processes and places nose pin"""
        if metrics.method == 'Pose':
            # Pose landmarks not accurate enough for nose pin
            logging.warning("Nose pin requires FaceMesh (close-up image)")
            return model_img
        
        h, w = model_img.shape[:2]
        pd_px = metrics.pd_px
        
        # Calculate dimensions
        pin_width = int(pd_px * self.config.nosepin_width_ratio)
        ar = jewel_img.shape[0] / jewel_img.shape[1]
        pin_height = int(pin_width * ar)
        
        # Resize
        pin_resized = cv2.resize(
            jewel_img,
            (pin_width, pin_height),
            interpolation=cv2.INTER_AREA
        )
        
        # Position (left nostril - landmark 48)
        lm = metrics.landmarks
        nose_x = int(lm[48].x * w)
        nose_y = int(lm[48].y * h)
        
        result = self.compositor.safe_overlay(
            model_img.copy(),
            pin_resized,
            nose_x - pin_width // 2,
            nose_y - pin_height // 2,
            self.config
        )
        
        return result
    
    def process(
        self,
        model_img: np.ndarray,
        jewelry_dict: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, str]:
        """
        Main processing pipeline
        
        Args:
            model_img: BGR image of person
            jewelry_dict: {item_type: RGBA jewelry image}
        
        Returns:
            (result_img, status_message)
        """
        try:
            # Convert to RGB for processing
            img_rgb = cv2.cvtColor(model_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Detect anatomy
            metrics = self.anatomy.detect_anatomy(img_rgb)
            
            if metrics is None:
                return model_img, "❌ Error: No person detected in image"
            
            logging.info(f"✅ Detection: {metrics.method}, PD={metrics.pd_px:.1f}px")
            
            result = model_img.copy()
            processed_items = []
            
            # Compute depth map once (for necklaces)
            depth_map = None
            if 'Necklace' in jewelry_dict and jewelry_dict['Necklace'] is not None:
                depth_map = self.depth_engine.compute_depth_map(img_pil)
            
            # Process each jewelry type
            if 'Necklace' in jewelry_dict and jewelry_dict['Necklace'] is not None:
                jewel = jewelry_dict['Necklace']
                jewel_clean = self.preprocessor.remove_background(jewel)
                jewel_clean = self.preprocessor.ensure_rgba(jewel_clean)
                jewel_clean = self.preprocessor.apply_edge_feather(
                    jewel_clean,
                    self.config.edge_feather_radius
                )
                
                result = self.process_necklace(result, jewel_clean, metrics, depth_map)
                processed_items.append('Necklace')
            
            if 'Earrings' in jewelry_dict and jewelry_dict['Earrings'] is not None:
                jewel = jewelry_dict['Earrings']
                jewel_clean = self.preprocessor.remove_background(jewel)
                jewel_clean = self.preprocessor.ensure_rgba(jewel_clean)
                jewel_clean = self.preprocessor.apply_edge_feather(
                    jewel_clean,
                    self.config.edge_feather_radius
                )
                
                result = self.process_earrings(result, jewel_clean, metrics)
                processed_items.append('Earrings')
            
            if 'Nose Pin' in jewelry_dict and jewelry_dict['Nose Pin'] is not None:
                jewel = jewelry_dict['Nose Pin']
                jewel_clean = self.preprocessor.remove_background(jewel)
                jewel_clean = self.preprocessor.ensure_rgba(jewel_clean)
                jewel_clean = self.preprocessor.apply_edge_feather(
                    jewel_clean,
                    self.config.edge_feather_radius
                )
                
                result = self.process_nosepin(result, jewel_clean, metrics)
                processed_items.append('Nose Pin')
            
            status = f"✅ Success ({metrics.method}): {', '.join(processed_items)}"
            return result, status
        
        except Exception as e:
            logging.error(f"Pipeline error: {e}", exc_info=True)
            return model_img, f"❌ Processing Error: {str(e)}"

# ==========================================
# 8. STREAMLIT UI
# ==========================================

def main():
    st.set_page_config(
        page_title="Enhanced AR Jewelry Try-On",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("💎 Enhanced Virtual Jewelry Try-On")
    st.markdown("""
    **Features:**
    - 🎨 Photorealistic color matching
    - 🌊 3D depth-aware warping
    - 🎭 Occlusion-aware compositing
    - 🔄 Smart fallback (close-up ↔ full body)
    - ✨ Soft edge blending & shadows
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        config = TryOnConfig()
        
        with st.expander("📏 Sizing", expanded=False):
            config.necklace_width_ratio = st.slider(
                "Necklace Width",
                2.0, 4.0, 3.3, 0.1
            )
            config.earring_height_ratio = st.slider(
                "Earring Size",
                0.5, 1.2, 0.75, 0.05
            )
        
        with st.expander("🎨 Blending", expanded=False):
            config.color_match_strength = st.slider(
                "Color Matching",
                0.0, 1.0, 0.25, 0.05
            )
            config.edge_feather_radius = st.slider(
                "Edge Softness",
                1, 10, 3, 1
            )
            config.shadow_intensity = st.slider(
                "Shadow Strength",
                0.0, 0.5, 0.15, 0.05
            )
    
    # Load models
    with st.spinner("🚀 Loading AI Models..."):
        models = ModelManager.load_all_models(config)
        engine = EnhancedTryOnEngine(models, config)
    
    # Main UI
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📤 Upload")
        
        model_file = st.file_uploader(
            "Model Photo",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a photo of a person (close-up or full body)"
        )
        
        jewelry_type = st.selectbox(
            "Jewelry Type",
            ["Necklace", "Earrings", "Nose Pin", "Multiple Items"]
        )
        
        jewelry_files = {}
        
        if jewelry_type == "Multiple Items":
            st.markdown("**Upload multiple jewelry items:**")
            for item in ["Necklace", "Earrings", "Nose Pin"]:
                f = st.file_uploader(
                    f"{item} (optional)",
                    type=['png', 'jpg', 'jpeg'],
                    key=f"jewel_{item}"
                )
                if f:
                    jewelry_files[item] = f
        else:
            f = st.file_uploader(
                f"{jewelry_type} Asset",
                type=['png', 'jpg', 'jpeg']
            )
            if f:
                jewelry_files[jewelry_type] = f
    
    with col2:
        st.subheader("🎭 Result")
        
        if model_file and jewelry_files:
            # Load images
            model_img = cv2.cvtColor(
                np.array(Image.open(model_file).convert('RGB')),
                cv2.COLOR_RGB2BGR
            )
            
            jewelry_dict = {}
            for item_type, file in jewelry_files.items():
                img = Image.open(file).convert('RGBA')
                jewelry_dict[item_type] = np.array(img)
            
            # Process
            with st.spinner("🔮 Processing with AI..."):
                result, status = engine.process(model_img, jewelry_dict)
            
            # Display
            st.image(
                cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption="Virtual Try-On Result"
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    main()