from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import torch
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    SegformerImageProcessor = None  # type: ignore
    SegformerForSemanticSegmentation = None  # type: ignore
    _TRANSFORMERS_AVAILABLE = False


logger = logging.getLogger(__name__)


class SegformerPredictor:
    """Wrapper for Hugging Face Segformer fine-tuned on Cityscapes.
    
    Maps Cityscapes classes to Forza AI terrain scores:
    - road (0) -> asphalt
    - vegetation (8) -> grass
    - terrain (9) -> dirt
    """

    def __init__(self, model_id: str = "nvidia/segformer-b0-finetuned-cityscapes-512-1024") -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library is required for neural vision. "
                "Install with: pip install transformers"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the image processor and model
        self.processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        
        # Pre-allocate mappings
        # Cityscapes classes: 0: road, 1: sidewalk, ..., 8: vegetation, 9: terrain
        self.road_id = 0
        self.vegetation_id = 8
        self.terrain_id = 9

    def predict_surface_scores(self, rgb: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
        """Run segmentation and return ratios for road, dirt, and grass.
        
        Expects rgb to be a numpy array of shape (H, W, 3) in [0, 1] float or [0, 255] uint8.
        """
        if rgb.size == 0:
            return {"road_score": 0.0, "dirt_score": 0.0, "grass_score": 0.0, "offroad_score": 0.0}

        # Ensure image is in [0, 255] uint8 for the processor
        if rgb.dtype == np.float32 or rgb.dtype == np.float64:
            if rgb.max(initial=0.0) <= 1.0:
                img_uint8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                img_uint8 = rgb.astype(np.uint8)
        else:
            img_uint8 = rgb.astype(np.uint8)

        inputs = self.processor(images=img_uint8, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape (1, num_classes, H/4, W/4)

        # Upsample logits to original image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=img_uint8.shape[:2],
            mode="bilinear",
            align_corners=False,
        )

        # Get the predicted class for each pixel
        predicted_mask = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

        if mask is not None:
            # Ensure mask matches shape and is boolean
            if mask.shape != predicted_mask.shape:
                import cv2
                mask_uint8 = mask.astype(np.uint8) * 255
                mask_uint8 = cv2.resize(mask_uint8, (predicted_mask.shape[1], predicted_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                valid = mask_uint8 > 0
            else:
                valid = mask.astype(bool)
            
            if not np.any(valid):
                return {"road_score": 0.0, "dirt_score": 0.0, "grass_score": 0.0, "offroad_score": 0.0}
                
            predicted_mask = predicted_mask[valid]
            total_pixels = float(predicted_mask.size)
        else:
            total_pixels = float(predicted_mask.size)
            
        if total_pixels == 0:
             return {"road_score": 0.0, "dirt_score": 0.0, "grass_score": 0.0, "offroad_score": 0.0}

        road_ratio = float(np.sum(predicted_mask == self.road_id)) / total_pixels
        grass_ratio = float(np.sum(predicted_mask == self.vegetation_id)) / total_pixels
        dirt_ratio = float(np.sum(predicted_mask == self.terrain_id)) / total_pixels

        return {
            "road_score": road_ratio,
            "grass_score": grass_ratio,
            "dirt_score": dirt_ratio,
            "offroad_score": min(1.0, grass_ratio + dirt_ratio)
        }
