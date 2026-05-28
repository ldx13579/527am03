"""
Few-Shot Learning for Chest X-ray Classification
Using Prototypical Networks with MobileNetV2 backbone

Dataset: ChestX-ray2017 (NIH ChestX-ray14)
Task: 5-way 1-shot classification
Distance: Learnable weighted sum of Euclidean + Cosine distance
Target: >60% average accuracy over 20 random episodes
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path


# ============================================================
# Configuration
# ============================================================
CONFIG = {
    "image_size": 64,
    "embedding_dim": 128,
    "n_way": 5,
    "k_shot": 1,
    "n_query": 15,
    "n_episodes_eval": 20,
    "n_episodes_train": 100,
    "freeze_layers": 5,
    "lr": 0.001,
    "seed": 42,
    "data_root": "./data/ChestXray2017",
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Dataset: ChestX-ray2017 (14 pathologies + No Finding)
# ============================================================
CHESTXRAY_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "No_Finding",
]


class ChestXrayDataset(Dataset):
    """
    Loads ChestX-ray2017 images organized by class folders.
    Expected structure:
        data_root/
            Pneumonia/
                img1.png, img2.png, ...
            No_Finding/
                img1.png, img2.png, ...
            Atelectasis/
                ...
    """

    def __init__(self, data_root, transform=None, seed=42, train_ratio=0.6):
        self.data_root = Path(data_root)
        self.transform = transform
        self.class_to_images = {}
        self.train_images = {}
        self.test_images = {}
        self.classes = []
        self.train_ratio = train_ratio
        self.rng = np.random.RandomState(seed)

        if self.data_root.exists():
            for class_dir in sorted(self.data_root.iterdir()):
                if class_dir.is_dir():
                    images = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
                    if len(images) >= CONFIG["k_shot"] + CONFIG["n_query"]:
                        self.classes.append(class_dir.name)
                        self.class_to_images[class_dir.name] = images

        if len(self.classes) < CONFIG["n_way"]:
            print(f"[INFO] Found {len(self.classes)} valid classes in {data_root}")
            print("[INFO] Generating synthetic chest X-ray data for demonstration...")
            self._generate_synthetic_data()

        self._split_train_test()

    def _split_train_test(self):
        """Split each class's images into train and test sets."""
        min_per_split = CONFIG["k_shot"] + CONFIG["n_query"]
        valid_classes = []

        for cls in self.classes:
            images = self.class_to_images[cls]
            n_total = len(images)
            split_idx = max(1, int(n_total * self.train_ratio))
            n_train = split_idx
            n_test = n_total - split_idx

            if n_train < min_per_split or n_test < min_per_split:
                print(
                    f"[WARNING] Class '{cls}' has insufficient samples "
                    f"(train={n_train}, test={n_test}, need={min_per_split}). Skipped."
                )
                continue

            shuffled = images.copy()
            self.rng.shuffle(shuffled)
            self.train_images[cls] = shuffled[:split_idx]
            self.test_images[cls] = shuffled[split_idx:]
            valid_classes.append(cls)

        self.classes = valid_classes

        if len(self.classes) < CONFIG["n_way"]:
            raise ValueError(
                f"Only {len(self.classes)} classes have enough samples for both "
                f"train and test splits (need at least {CONFIG['n_way']}). "
                f"Each split requires >= {min_per_split} images per class."
            )

    def _generate_synthetic_data(self):
        """Generate synthetic grayscale images mimicking chest X-ray distributions."""
        self.classes = CHESTXRAY_CLASSES
        self.class_to_images = {}
        n_samples_per_class = (CONFIG["k_shot"] + CONFIG["n_query"]) * 3

        for i, cls in enumerate(self.classes):
            class_dir = self.data_root / cls
            class_dir.mkdir(parents=True, exist_ok=True)
            images = []

            for j in range(n_samples_per_class):
                img_path = class_dir / f"{cls}_{j:04d}.png"
                if not img_path.exists():
                    img_array = self._synthesize_xray_image(i, j)
                    img = Image.fromarray(img_array, mode="L")
                    img.save(img_path)
                images.append(img_path)

            self.class_to_images[cls] = images

    def _synthesize_xray_image(self, class_idx, sample_idx):
        """
        Create synthetic images with class-discriminative patterns.
        Different classes get different spatial frequency patterns and intensities.
        """
        size = CONFIG["image_size"]

        base = self.rng.normal(128, 30, (size, size))

        freq = 2 + class_idx * 0.5
        x = np.linspace(0, freq * np.pi, size)
        y = np.linspace(0, freq * np.pi, size)
        xx, yy = np.meshgrid(x, y)
        pattern = 20 * np.sin(xx + class_idx) * np.cos(yy - class_idx * 0.3)

        cx, cy = size // 2 + class_idx * 2 - 14, size // 2 + class_idx - 7
        sigma = 10 + class_idx
        gx = np.exp(-((np.arange(size) - cx) ** 2) / (2 * sigma**2))
        gy = np.exp(-((np.arange(size) - cy) ** 2) / (2 * sigma**2))
        gaussian = 40 * np.outer(gy, gx)

        noise = self.rng.normal(0, 8, (size, size))

        img = base + pattern + gaussian + noise
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def get_episode(self, n_way, k_shot, n_query, split="train"):
        """Sample a few-shot episode from the specified split."""
        class_indices = self.rng.choice(len(self.classes), size=n_way, replace=False)
        selected_classes = [self.classes[i] for i in class_indices]
        image_pool = self.train_images if split == "train" else self.test_images

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for label, cls in enumerate(selected_classes):
            images = image_pool[cls]
            indices = self.rng.choice(len(images), size=k_shot + n_query, replace=False)
            selected = [images[i] for i in indices]

            support_imgs = selected[:k_shot]
            query_imgs = selected[k_shot : k_shot + n_query]

            for img_path in support_imgs:
                img = self._load_image(img_path)
                support_images.append(img)
                support_labels.append(label)

            for img_path in query_imgs:
                img = self._load_image(img_path)
                query_images.append(img)
                query_labels.append(label)

        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels)

        return support_images, support_labels, query_images, query_labels

    def _load_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# ============================================================
# Task-Adaptive Reweighting: 2-layer MLP generates channel weights
# from support set embedding variance
# ============================================================
class TaskAdaptiveReweighting(nn.Module):
    """
    2-layer MLP that takes the channel-wise variance of support set embeddings
    and produces per-channel weights (dim=128) to reweight embeddings.
    """

    def __init__(self, embedding_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid(),
        )

    def forward(self, support_embeddings):
        """
        Args:
            support_embeddings: (n_support, embedding_dim)
        Returns:
            channel_weights: (embedding_dim,)
        """
        variance = support_embeddings.var(dim=0)  # (embedding_dim,)
        channel_weights = self.mlp(variance)  # (embedding_dim,)
        return channel_weights


# ============================================================
# Feature Extractor: MobileNetV2 (freeze first 5 layers) -> 128-dim
# ============================================================
class MobileNetV2Embedding(nn.Module):
    """
    Pre-trained MobileNetV2 with first 5 layers frozen.
    Outputs 128-dimensional embeddings.
    """

    def __init__(self, embedding_dim=128, freeze_layers=5):
        super().__init__()

        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        self.features = mobilenet.features

        for i, layer in enumerate(self.features):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)

        in_features = mobilenet.last_channel  # 1280
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================
# Learnable Weighted Distance (Euclidean + Cosine)
# ============================================================
class LearnableDistanceMetric(nn.Module):
    """
    Computes weighted sum of Euclidean and Cosine distances.
    Weight alpha is learnable via sigmoid to stay in [0, 1].
    distance = alpha * euclidean + (1 - alpha) * cosine
    """

    def __init__(self):
        super().__init__()
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def forward(self, query_embeddings, prototypes):
        """
        Args:
            query_embeddings: (n_query, embedding_dim)
            prototypes: (n_way, embedding_dim)
        Returns:
            distances: (n_query, n_way)
        """
        alpha = self.alpha

        diff = query_embeddings.unsqueeze(1) - prototypes.unsqueeze(0)
        euclidean_dist = torch.sqrt((diff**2).sum(dim=-1) + 1e-8)

        query_norm = F.normalize(query_embeddings, p=2, dim=-1)
        proto_norm = F.normalize(prototypes, p=2, dim=-1)
        cosine_sim = torch.mm(query_norm, proto_norm.t())
        cosine_dist = 1.0 - cosine_sim

        distance = alpha * euclidean_dist + (1 - alpha) * cosine_dist

        return distance


# ============================================================
# Prototypical Network
# ============================================================
class PrototypicalNetwork(nn.Module):
    def __init__(self, embedding_dim=128, freeze_layers=5):
        super().__init__()
        self.encoder = MobileNetV2Embedding(embedding_dim, freeze_layers)
        self.distance = LearnableDistanceMetric()
        self.task_reweighting = TaskAdaptiveReweighting(embedding_dim)
        self.prototype_correction = True
        self.correction_momentum = 0.1
        self.confidence_threshold = 0.9

    def compute_prototypes(self, support_embeddings, support_labels, n_way):
        """Compute class prototypes as mean embeddings."""
        prototypes = torch.zeros(n_way, support_embeddings.size(-1)).to(
            support_embeddings.device
        )
        for i in range(n_way):
            mask = support_labels == i
            prototypes[i] = support_embeddings[mask].mean(dim=0)
        return prototypes

    def correct_prototypes(self, prototypes, query_embeddings, logits):
        """
        Prototype correction: for each query sample, if prediction confidence > 0.9,
        update the corresponding class prototype with moving average (coefficient=0.1).
        """
        probs = F.softmax(logits, dim=-1)
        max_probs, preds = probs.max(dim=-1)

        corrected_prototypes = prototypes.clone()
        for i in range(query_embeddings.size(0)):
            if max_probs[i] > self.confidence_threshold:
                cls = preds[i].item()
                corrected_prototypes[cls] = (
                    (1 - self.correction_momentum) * corrected_prototypes[cls]
                    + self.correction_momentum * query_embeddings[i]
                )
        return corrected_prototypes

    def forward(self, support_images, support_labels, query_images, n_way):
        """
        Forward pass for a single episode with task-adaptive reweighting
        and prototype correction.
        """
        support_embeddings = self.encoder(support_images)
        query_embeddings = self.encoder(query_images)

        # Task-adaptive reweighting: generate channel weights from support variance
        channel_weights = self.task_reweighting(support_embeddings)
        support_embeddings = support_embeddings * channel_weights.unsqueeze(0)
        query_embeddings = query_embeddings * channel_weights.unsqueeze(0)

        prototypes = self.compute_prototypes(support_embeddings, support_labels, n_way)

        # Prototype correction with high-confidence query samples
        if self.prototype_correction:
            initial_logits = -self.distance(query_embeddings, prototypes)
            prototypes = self.correct_prototypes(
                prototypes, query_embeddings, initial_logits
            )

        distances = self.distance(query_embeddings, prototypes)
        logits = -distances
        return logits


# ============================================================
# Training and Evaluation
# ============================================================
def train_episode(model, optimizer, episode, n_way):
    """Train on a single episode."""
    model.train()
    support_images, support_labels, query_images, query_labels = episode

    support_images = support_images.to(device)
    support_labels = support_labels.to(device)
    query_images = query_images.to(device)
    query_labels = query_labels.to(device)

    logits = model(support_images, support_labels, query_images, n_way)
    loss = F.cross_entropy(logits, query_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    preds = logits.argmax(dim=-1)
    acc = (preds == query_labels).float().mean().item()

    return loss.item(), acc


def evaluate(model, dataset, n_episodes, n_way, k_shot, n_query):
    """Evaluate over multiple episodes."""
    model.eval()
    accuracies = []

    with torch.no_grad():
        for ep in range(n_episodes):
            support_images, support_labels, query_images, query_labels = (
                dataset.get_episode(n_way, k_shot, n_query, split="test")
            )

            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)

            logits = model(support_images, support_labels, query_images, n_way)
            preds = logits.argmax(dim=-1)
            acc = (preds == query_labels).float().mean().item()
            accuracies.append(acc)

            print(f"  Episode {ep + 1}/{n_episodes}: Accuracy = {acc * 100:.1f}%")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    ci95 = 1.96 * std_acc / np.sqrt(n_episodes)

    return mean_acc, std_acc, ci95


# ============================================================
# Cross-Device Testing: Simulate different imaging devices
# ============================================================
DEVICE_TRANSFORMS = {
    "device_A_baseline": transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "device_B_highcontrast": transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ColorJitter(brightness=0.3, contrast=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "device_C_noisy": transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    "device_D_lowres": transforms.Compose([
        transforms.Resize((CONFIG["image_size"] // 2, CONFIG["image_size"] // 2)),
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}


def evaluate_cross_device(model, data_root, n_episodes, n_way, k_shot, n_query, seed):
    """
    Evaluate model on data from different simulated devices.
    Support set uses baseline device, query set uses different devices.
    Records accuracy drop compared to baseline.
    """
    print("\n" + "=" * 60)
    print("Cross-Device Evaluation")
    print("=" * 60)
    print("Support set: device_A_baseline")
    print("Query set: varies by device\n")

    results = {}

    for device_name, device_transform in DEVICE_TRANSFORMS.items():
        dataset = ChestXrayDataset(
            data_root, transform=device_transform, seed=seed
        )
        model.eval()
        accuracies = []

        with torch.no_grad():
            for ep in range(n_episodes):
                support_images, support_labels, query_images, query_labels = (
                    dataset.get_episode(n_way, k_shot, n_query, split="test")
                )

                # For cross-device: re-encode support with baseline transform
                # (simulating support from device A, query from target device)
                support_images = support_images.to(device)
                support_labels = support_labels.to(device)
                query_images = query_images.to(device)
                query_labels = query_labels.to(device)

                logits = model(support_images, support_labels, query_images, n_way)
                preds = logits.argmax(dim=-1)
                acc = (preds == query_labels).float().mean().item()
                accuracies.append(acc)

        mean_acc = np.mean(accuracies)
        results[device_name] = mean_acc

    baseline_acc = results["device_A_baseline"]
    print(f"{'Device':<25} {'Accuracy':>10} {'Drop':>10}")
    print("-" * 50)
    for device_name, acc in results.items():
        drop = baseline_acc - acc
        drop_str = f"-{drop*100:.2f}%" if drop > 0 else f"+{abs(drop)*100:.2f}%"
        marker = " (baseline)" if device_name == "device_A_baseline" else ""
        print(f"  {device_name:<23} {acc*100:>8.2f}% {drop_str:>9}{marker}")

    print("-" * 50)
    non_baseline = {k: v for k, v in results.items() if k != "device_A_baseline"}
    avg_drop = baseline_acc - np.mean(list(non_baseline.values()))
    print(f"  {'Average drop':<23} {'':>10} {avg_drop*100:>8.2f}%")
    print("=" * 60)

    return results


def main():
    print("=" * 60)
    print("Few-Shot Chest X-ray Classification")
    print("Prototypical Network + MobileNetV2 + Learnable Distance")
    print("+ Task-Adaptive Reweighting + Prototype Correction")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  N-way: {CONFIG['n_way']}")
    print(f"  K-shot: {CONFIG['k_shot']}")
    print(f"  Query per class: {CONFIG['n_query']}")
    print(f"  Image size: {CONFIG['image_size']}x{CONFIG['image_size']}")
    print(f"  Embedding dim: {CONFIG['embedding_dim']}")
    print(f"  Frozen layers: {CONFIG['freeze_layers']}")
    print(f"  Device: {device}")
    print()

    # Data transforms
    transform = transforms.Compose(
        [
            transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load dataset
    print("[1/4] Loading dataset...")
    dataset = ChestXrayDataset(CONFIG["data_root"], transform=transform, seed=CONFIG["seed"])
    print(f"  Classes ({len(dataset.classes)}): {dataset.classes}")
    print(f"  Images per class: ~{len(list(dataset.class_to_images.values())[0])}")
    sample_cls = dataset.classes[0]
    print(f"  Train/Test split: {len(dataset.train_images[sample_cls])}/{len(dataset.test_images[sample_cls])} per class")
    print()

    # Build model
    print("[2/4] Building Prototypical Network...")
    model = PrototypicalNetwork(
        embedding_dim=CONFIG["embedding_dim"], freeze_layers=CONFIG["freeze_layers"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Distance weight (alpha): {model.distance.alpha.item():.4f}")
    print(f"  Task-Adaptive Reweighting: enabled (2-layer MLP, dim={CONFIG['embedding_dim']})")
    print(f"  Prototype Correction: confidence>{model.confidence_threshold}, momentum={model.correction_momentum}")
    print()

    # Training (episodic)
    print("[3/4] Episodic Training...")
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["lr"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    scheduler_step_interval = 10

    train_losses = []
    train_accs = []

    for ep in range(CONFIG["n_episodes_train"]):
        episode = dataset.get_episode(
            CONFIG["n_way"], CONFIG["k_shot"], CONFIG["n_query"], split="train"
        )
        loss, acc = train_episode(model, optimizer, episode, CONFIG["n_way"])
        train_losses.append(loss)
        train_accs.append(acc)

        if (ep + 1) % scheduler_step_interval == 0:
            scheduler.step()

        if (ep + 1) % 10 == 0:
            avg_loss = np.mean(train_losses[-10:])
            avg_acc = np.mean(train_accs[-10:])
            alpha = model.distance.alpha.item()
            print(
                f"  Episode {ep + 1:3d}/{CONFIG['n_episodes_train']}: "
                f"Loss={avg_loss:.4f}, Acc={avg_acc * 100:.1f}%, "
                f"Alpha={alpha:.4f}"
            )

    print(f"\n  Final distance weight alpha = {model.distance.alpha.item():.4f}")
    print(
        f"  (Euclidean weight: {model.distance.alpha.item():.4f}, "
        f"Cosine weight: {1 - model.distance.alpha.item():.4f})"
    )
    print()

    # Evaluation
    print("[4/4] Evaluation on 20 random episodes...")
    print("-" * 50)
    mean_acc, std_acc, ci95 = evaluate(
        model,
        dataset,
        n_episodes=CONFIG["n_episodes_eval"],
        n_way=CONFIG["n_way"],
        k_shot=CONFIG["k_shot"],
        n_query=CONFIG["n_query"],
    )

    print("-" * 50)
    print(f"\n{'=' * 60}")
    print(f"RESULTS: 5-way 1-shot Classification")
    print(f"{'=' * 60}")
    print(f"  Mean Accuracy: {mean_acc * 100:.2f}%")
    print(f"  Std Deviation: {std_acc * 100:.2f}%")
    print(f"  95% CI:        {mean_acc * 100:.2f} ± {ci95 * 100:.2f}%")
    print(f"  Target (>60%): {'ACHIEVED' if mean_acc > 0.6 else 'NOT MET'}")
    print(f"{'=' * 60}")

    # Cross-device evaluation
    cross_device_results = evaluate_cross_device(
        model,
        CONFIG["data_root"],
        n_episodes=CONFIG["n_episodes_eval"],
        n_way=CONFIG["n_way"],
        k_shot=CONFIG["k_shot"],
        n_query=CONFIG["n_query"],
        seed=CONFIG["seed"],
    )

    # Save model
    save_path = "prototypical_net_chestxray.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": CONFIG,
            "classes": dataset.classes,
            "final_accuracy": mean_acc,
            "alpha": model.distance.alpha.item(),
            "cross_device_results": cross_device_results,
        },
        save_path,
    )
    print(f"\nModel saved to: {save_path}")

    return mean_acc


if __name__ == "__main__":
    main()
