from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    from PIL import Image
    _TORCHVISION_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = type("_NNStub", (), {"Module": object})()  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore
    models = None  # type: ignore
    transforms = None  # type: ignore
    Image = None  # type: ignore
    _TORCHVISION_AVAILABLE = False


logger = logging.getLogger(__name__)

CLASSES = ["road", "dirt", "mixed", "crash", "menu"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}


class VisionLabelsDataset(Dataset):
    def __init__(self, labels_path: str | Path, transform: Any = None):
        self.labels_path = Path(labels_path)
        self.transform = transform
        self.samples: list[dict[str, Any]] = []

        if not self.labels_path.exists():
            return

        with self.labels_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    label = str(record.get("label", "")).strip().lower()
                    if label in CLASS_TO_IDX:
                        # Use image_path if available (full screen), else fallback to roi_path
                        image_path = record.get("image_path") or record.get("roi_path")
                        if image_path and Path(image_path).exists():
                            self.samples.append({
                                "image_path": str(image_path),
                                "label_idx": CLASS_TO_IDX[label]
                            })
                except Exception as e:
                    logger.warning(f"Failed to parse label line: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, sample["label_idx"]


class CustomVisionClassifier:
    def __init__(self, model_path: str | Path | None = None):
        if not _TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for CustomVisionClassifier")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path) if model_path else None
        
        # Load MobileNetV3 small
        self.model = models.mobilenet_v3_small(weights=None)
        # Replace head for 5 classes
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_features, len(CLASSES))
        
        self.is_loaded = False

        if self.model_path and self.model_path.exists():
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.is_loaded = True
            except Exception as e:
                logger.error(f"Failed to load custom vision classifier: {e}")

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, rgb: np.ndarray) -> dict[str, float]:
        """Run classification on a numpy RGB image. Return probabilities."""
        if not self.is_loaded or rgb.size == 0:
            return {f"vision_custom_{cls}": 0.0 for cls in CLASSES}
            
        try:
            # rgb is expected to be [0, 1] float or [0, 255] uint8
            if rgb.dtype == np.float32 or rgb.dtype == np.float64:
                img_uint8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
            else:
                img_uint8 = rgb.astype(np.uint8)
                
            image = Image.fromarray(img_uint8)
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                
            return {f"vision_custom_{cls}": float(prob) for cls, prob in zip(CLASSES, probs)}
        except Exception as e:
            logger.error(f"Prediction error in custom classifier: {e}")
            return {f"vision_custom_{cls}": 0.0 for cls in CLASSES}


def train_classifier(labels_path: str | Path, output_path: str | Path, epochs: int = 15, batch_size: int = 16) -> None:
    if not _TORCHVISION_AVAILABLE:
        print("torchvision is required for training.")
        return

    labels_path = Path(labels_path)
    output_path = Path(output_path)

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = VisionLabelsDataset(labels_path, transform=transform_train)
    if len(dataset) == 0:
        print(f"No valid training samples found in {labels_path}.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} with {len(dataset)} samples.")

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, len(CLASSES))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved trained model to {output_path}")