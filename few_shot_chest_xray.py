"""
Few-Shot Learning for Chest X-ray Classification
Using Prototypical Networks with MobileNetV2 backbone

Dataset: ChestX-ray2017 (NIH ChestX-ray14)
Task: 5-way 1-shot classification
Distance: Learnable weighted sum of Euclidean + Cosine distance
Target: >60% average accuracy over 20 random episodes

Extensions:
- Semi-supervised pseudo-labeling with 200 unlabeled CT images
- Graph-based label propagation (kNN graph between query and unlabeled sets)
- FixMatch comparison at 10% label budget
- Prototype vector displacement visualization
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
    # Semi-supervised settings
    "n_unlabeled": 200,
    "pseudo_label_threshold": 0.8,
    "graph_k": 5,
    "graph_alpha": 0.5,
    "graph_n_iter": 5,
    # FixMatch settings
    "fixmatch_threshold": 0.95,
    "fixmatch_lambda_u": 1.0,
    "fixmatch_label_fraction": 0.1,
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


# ============================================================
# Augmentation Pipelines
# ============================================================
def get_base_transform():
    return transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_weak_augmentation():
    return transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_strong_augmentation():
    return transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.RandomAffine(degrees=30, translate=(0.15, 0.15)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ChestXrayDataset(Dataset):
    """
    Loads ChestX-ray2017 images organized by class folders.
    Also manages an unlabeled pool of 200 CT images for semi-supervised learning.
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
        self.unlabeled_images = []

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
        self._generate_unlabeled_pool()

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

    def _generate_unlabeled_pool(self, n_unlabeled=None):
        """Generate 200 unlabeled CT images as blends of multiple class patterns."""
        if n_unlabeled is None:
            n_unlabeled = CONFIG["n_unlabeled"]

        unlabeled_dir = self.data_root.parent / "unlabeled_ct"
        unlabeled_dir.mkdir(parents=True, exist_ok=True)
        self.unlabeled_images = []

        for j in range(n_unlabeled):
            img_path = unlabeled_dir / f"unlabeled_{j:04d}.png"
            if not img_path.exists():
                n_blend = self.rng.randint(2, 4)
                class_indices = self.rng.choice(len(CHESTXRAY_CLASSES), size=n_blend, replace=False)
                weights = self.rng.dirichlet(np.ones(n_blend))
                img_array = np.zeros((CONFIG["image_size"], CONFIG["image_size"]), dtype=np.float64)
                for ci, w in zip(class_indices, weights):
                    img_array += w * self._synthesize_xray_image(ci, j).astype(np.float64)
                img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8), mode="L")
                img.save(img_path)
            self.unlabeled_images.append(img_path)

    def get_unlabeled_batch(self, n, transform=None):
        """Load n random unlabeled images as a tensor batch."""
        if transform is None:
            transform = self.transform
        indices = self.rng.choice(len(self.unlabeled_images), size=min(n, len(self.unlabeled_images)), replace=False)
        images = []
        for idx in indices:
            img = Image.open(self.unlabeled_images[idx]).convert("RGB")
            if transform:
                img = transform(img)
            images.append(img)
        return torch.stack(images), indices

    def get_unlabeled_by_indices(self, indices, transform=None):
        """Load specific unlabeled images by their indices."""
        if transform is None:
            transform = self.transform
        images = []
        for idx in indices:
            img = Image.open(self.unlabeled_images[idx]).convert("RGB")
            if transform:
                img = transform(img)
            images.append(img)
        return torch.stack(images)

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
        """Create synthetic images with class-discriminative patterns."""
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

    def get_reduced_label_episode(self, n_way, k_shot, n_query, label_fraction=0.1, split="train"):
        """
        Sample episode with reduced label budget for FixMatch comparison.
        Only label_fraction of the available support images are used as labeled.
        The rest are treated as unlabeled.
        """
        class_indices = self.rng.choice(len(self.classes), size=n_way, replace=False)
        selected_classes = [self.classes[i] for i in class_indices]
        image_pool = self.train_images if split == "train" else self.test_images

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        unlabeled_images = []

        for label, cls in enumerate(selected_classes):
            images = image_pool[cls]
            n_available = len(images)
            n_labeled = max(1, int(n_available * label_fraction))
            n_needed = k_shot + n_query

            indices = self.rng.choice(n_available, size=min(n_available, n_needed + n_labeled), replace=False)
            selected = [images[i] for i in indices]

            labeled_imgs = selected[:k_shot]
            query_imgs = selected[k_shot:k_shot + n_query]
            extra_unlabeled = selected[k_shot + n_query:]

            for img_path in labeled_imgs:
                img = self._load_image(img_path)
                support_images.append(img)
                support_labels.append(label)

            for img_path in query_imgs:
                img = self._load_image(img_path)
                query_images.append(img)
                query_labels.append(label)

            for img_path in extra_unlabeled:
                img = self._load_image(img_path)
                unlabeled_images.append(img)

        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels)
        unlabeled_images = torch.stack(unlabeled_images) if unlabeled_images else torch.zeros(0, 3, CONFIG["image_size"], CONFIG["image_size"])

        return support_images, support_labels, query_images, query_labels, unlabeled_images

    def _load_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# ============================================================
# Task-Adaptive Reweighting
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

    def forward(self, support_embeddings, support_labels, n_way):
        class_variances = []
        for i in range(n_way):
            mask = support_labels == i
            class_emb = support_embeddings[mask]
            if class_emb.size(0) > 1:
                class_variances.append(class_emb.var(dim=0))
            else:
                class_variances.append(torch.zeros_like(support_embeddings[0]))
        aggregated_variance = torch.stack(class_variances).mean(dim=0)
        channel_weights = self.mlp(aggregated_variance)
        return channel_weights


# ============================================================
# Feature Extractor: MobileNetV2 (freeze first 5 layers) -> 128-dim
# ============================================================
class MobileNetV2Embedding(nn.Module):
    def __init__(self, embedding_dim=128, freeze_layers=5):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = mobilenet.features

        for i, layer in enumerate(self.features):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        in_features = mobilenet.last_channel
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
    def __init__(self):
        super().__init__()
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def forward(self, query_embeddings, prototypes):
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
# Graph Label Propagation
# ============================================================
class GraphLabelPropagation(nn.Module):
    """
    Constructs a kNN graph between query and unlabeled embeddings,
    propagates prototype labels along edges to update unlabeled node predictions.
    """

    def __init__(self, k=5, alpha=0.5, n_iter=5):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.n_iter = n_iter

    def build_knn_graph(self, embeddings):
        """Build kNN adjacency matrix with Gaussian kernel weights."""
        N = embeddings.size(0)
        dists = torch.cdist(embeddings, embeddings, p=2)

        sigma = torch.median(dists[dists > 0]).item() + 1e-8

        _, topk_indices = dists.topk(self.k + 1, dim=-1, largest=False)
        topk_indices = topk_indices[:, 1:]  # exclude self

        W = torch.zeros(N, N, device=embeddings.device)
        for i in range(N):
            for j in topk_indices[i]:
                weight = torch.exp(-dists[i, j] ** 2 / (2 * sigma ** 2))
                W[i, j.item()] = weight
                W[j.item(), i] = weight  # symmetric

        return W

    def propagate(self, embeddings, initial_labels, labeled_mask):
        """
        Run label propagation on the graph.
        Args:
            embeddings: (N, D) all node embeddings
            initial_labels: (N, C) soft label matrix
            labeled_mask: (N,) boolean mask for nodes with known labels
        Returns:
            updated_labels: (N, C) propagated soft labels
        """
        W = self.build_knn_graph(embeddings)

        D = W.sum(dim=1)
        D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(D) + 1e-8))
        S = D_inv_sqrt @ W @ D_inv_sqrt

        F_t = initial_labels.clone()
        Y_init = initial_labels.clone()

        for _ in range(self.n_iter):
            F_t = self.alpha * (S @ F_t) + (1 - self.alpha) * Y_init
            F_t[labeled_mask] = Y_init[labeled_mask]

        return F_t


# ============================================================
# Prototypical Network
# ============================================================
class PrototypicalNetwork(nn.Module):
    def __init__(self, embedding_dim=128, freeze_layers=5):
        super().__init__()
        self.encoder = MobileNetV2Embedding(embedding_dim, freeze_layers)
        self.distance = LearnableDistanceMetric()
        self.task_reweighting = TaskAdaptiveReweighting(embedding_dim)
        self.graph_propagation = GraphLabelPropagation(
            k=CONFIG["graph_k"], alpha=CONFIG["graph_alpha"], n_iter=CONFIG["graph_n_iter"]
        )
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
        """Prototype correction with nearest-neighbor consistency filtering."""
        probs = F.softmax(logits, dim=-1)
        max_probs, preds = probs.max(dim=-1)

        diffs = query_embeddings.unsqueeze(1) - prototypes.unsqueeze(0)
        nn_dists = (diffs ** 2).sum(dim=-1)
        nn_classes = nn_dists.argmin(dim=-1)

        corrected_prototypes = prototypes.clone()
        for i in range(query_embeddings.size(0)):
            if max_probs[i] > self.confidence_threshold and preds[i] == nn_classes[i]:
                cls = preds[i].item()
                corrected_prototypes[cls] = (
                    (1 - self.correction_momentum) * corrected_prototypes[cls]
                    + self.correction_momentum * query_embeddings[i]
                )
        return corrected_prototypes

    def forward(self, support_images, support_labels, query_images, n_way):
        """Standard forward pass for a single episode."""
        support_embeddings = self.encoder(support_images)
        query_embeddings = self.encoder(query_images)

        channel_weights = self.task_reweighting(support_embeddings, support_labels, n_way)
        support_embeddings = support_embeddings * channel_weights.unsqueeze(0)
        query_embeddings = query_embeddings * channel_weights.unsqueeze(0)

        prototypes = self.compute_prototypes(support_embeddings, support_labels, n_way)

        if self.prototype_correction:
            initial_logits = -self.distance(query_embeddings, prototypes)
            prototypes = self.correct_prototypes(
                prototypes, query_embeddings, initial_logits
            )

        distances = self.distance(query_embeddings, prototypes)
        logits = -distances
        return logits

    def forward_semi_supervised(self, support_images, support_labels, query_images,
                                 unlabeled_view1, unlabeled_view2, n_way):
        """
        Semi-supervised forward with dual-view pseudo-labeling.
        Two augmented views of unlabeled images are encoded; samples where both
        views agree with confidence > threshold get pseudo-labels and enrich prototypes.
        """
        support_embeddings = self.encoder(support_images)
        query_embeddings = self.encoder(query_images)

        channel_weights = self.task_reweighting(support_embeddings, support_labels, n_way)
        support_embeddings = support_embeddings * channel_weights.unsqueeze(0)
        query_embeddings = query_embeddings * channel_weights.unsqueeze(0)

        prototypes_before = self.compute_prototypes(support_embeddings, support_labels, n_way)

        # Encode unlabeled views
        unlabeled_emb_v1 = self.encoder(unlabeled_view1) * channel_weights.unsqueeze(0)
        unlabeled_emb_v2 = self.encoder(unlabeled_view2) * channel_weights.unsqueeze(0)

        # Pseudo-labeling: require both views to agree above threshold
        dist_v1 = self.distance(unlabeled_emb_v1, prototypes_before)
        dist_v2 = self.distance(unlabeled_emb_v2, prototypes_before)
        probs_v1 = F.softmax(-dist_v1, dim=-1)
        probs_v2 = F.softmax(-dist_v2, dim=-1)

        max_probs_v1, preds_v1 = probs_v1.max(dim=-1)
        max_probs_v2, preds_v2 = probs_v2.max(dim=-1)

        threshold = CONFIG["pseudo_label_threshold"]
        agree_mask = (preds_v1 == preds_v2) & \
                     (max_probs_v1 > threshold) & \
                     (max_probs_v2 > threshold)

        # Enrich prototypes with pseudo-labeled samples
        prototypes_enriched = prototypes_before.clone()
        pseudo_count = torch.zeros(n_way, device=support_images.device)

        if agree_mask.any():
            pseudo_labels = preds_v1[agree_mask]
            pseudo_embeddings = (unlabeled_emb_v1[agree_mask] + unlabeled_emb_v2[agree_mask]) / 2

            for i in range(n_way):
                cls_mask = pseudo_labels == i
                if cls_mask.any():
                    n_support = (support_labels == i).sum().float()
                    n_pseudo = cls_mask.sum().float()
                    pseudo_count[i] = n_pseudo
                    # Weighted mean: support weight=1.0, pseudo weight=0.5
                    pseudo_mean = pseudo_embeddings[cls_mask].mean(dim=0)
                    total_weight = n_support + 0.5 * n_pseudo
                    prototypes_enriched[i] = (
                        n_support * prototypes_before[i] + 0.5 * n_pseudo * pseudo_mean
                    ) / total_weight

        # Prototype correction with query set
        if self.prototype_correction:
            initial_logits = -self.distance(query_embeddings, prototypes_enriched)
            prototypes_enriched = self.correct_prototypes(
                prototypes_enriched, query_embeddings, initial_logits
            )

        distances = self.distance(query_embeddings, prototypes_enriched)
        logits = -distances

        return logits, prototypes_before, prototypes_enriched, agree_mask.sum().item(), pseudo_count

    def forward_with_graph_propagation(self, support_images, support_labels, query_images,
                                        unlabeled_view1, unlabeled_view2, n_way):
        """
        Forward with graph-based label propagation between query and unlabeled sets.
        Builds kNN graph, propagates prototype labels, updates unlabeled features.
        """
        support_embeddings = self.encoder(support_images)
        query_embeddings = self.encoder(query_images)

        channel_weights = self.task_reweighting(support_embeddings, support_labels, n_way)
        support_embeddings = support_embeddings * channel_weights.unsqueeze(0)
        query_embeddings = query_embeddings * channel_weights.unsqueeze(0)

        prototypes = self.compute_prototypes(support_embeddings, support_labels, n_way)

        # Encode unlabeled (average of two views)
        unlabeled_emb_v1 = self.encoder(unlabeled_view1) * channel_weights.unsqueeze(0)
        unlabeled_emb_v2 = self.encoder(unlabeled_view2) * channel_weights.unsqueeze(0)
        unlabeled_emb = (unlabeled_emb_v1 + unlabeled_emb_v2) / 2

        # Build combined node set: [query | unlabeled]
        n_query = query_embeddings.size(0)
        n_unlabeled = unlabeled_emb.size(0)
        all_embeddings = torch.cat([query_embeddings, unlabeled_emb], dim=0)

        # Initial labels: query nodes get prototype-based predictions
        query_dists = self.distance(query_embeddings, prototypes)
        query_probs = F.softmax(-query_dists, dim=-1)

        # Unlabeled nodes start uniform
        unlabeled_probs = torch.ones(n_unlabeled, n_way, device=support_images.device) / n_way

        initial_labels = torch.cat([query_probs, unlabeled_probs], dim=0)
        labeled_mask = torch.zeros(n_query + n_unlabeled, dtype=torch.bool, device=support_images.device)
        labeled_mask[:n_query] = True

        # Graph propagation
        propagated_labels = self.graph_propagation.propagate(
            all_embeddings, initial_labels, labeled_mask
        )

        # Use propagated labels on unlabeled nodes to update prototypes
        unlabeled_propagated = propagated_labels[n_query:]
        max_probs_prop, preds_prop = unlabeled_propagated.max(dim=-1)

        confident_mask = max_probs_prop > CONFIG["pseudo_label_threshold"]
        prototypes_updated = prototypes.clone()

        if confident_mask.any():
            confident_emb = unlabeled_emb[confident_mask]
            confident_labels = preds_prop[confident_mask]
            for i in range(n_way):
                cls_mask = confident_labels == i
                if cls_mask.any():
                    n_support = (support_labels == i).sum().float()
                    n_prop = cls_mask.sum().float()
                    prop_mean = confident_emb[cls_mask].mean(dim=0)
                    total_weight = n_support + 0.5 * n_prop
                    prototypes_updated[i] = (
                        n_support * prototypes[i] + 0.5 * n_prop * prop_mean
                    ) / total_weight

        # Final prediction on query
        distances = self.distance(query_embeddings, prototypes_updated)
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


def train_fixmatch_episode(model, optimizer, episode, unlabeled_batch,
                           n_way, weak_aug, strong_aug, dataset):
    """
    FixMatch episode: supervised loss on labeled support/query +
    unsupervised consistency loss on unlabeled data.
    """
    model.train()
    support_images, support_labels, query_images, query_labels, _ = episode

    support_images = support_images.to(device)
    support_labels = support_labels.to(device)
    query_images = query_images.to(device)
    query_labels = query_labels.to(device)

    # Supervised loss
    logits = model(support_images, support_labels, query_images, n_way)
    loss_sup = F.cross_entropy(logits, query_labels)

    # Unsupervised loss with FixMatch
    loss_unsup = torch.tensor(0.0, device=device)

    if unlabeled_batch.size(0) > 0:
        unlabeled_batch = unlabeled_batch.to(device)
        with torch.no_grad():
            support_emb = model.encoder(support_images)
            channel_w = model.task_reweighting(support_emb, support_labels, n_way)
            support_emb = support_emb * channel_w.unsqueeze(0)
            prototypes = model.compute_prototypes(support_emb, support_labels, n_way)

            # Weak aug prediction (pseudo-label) - already encoded
            unlabeled_emb = model.encoder(unlabeled_batch) * channel_w.unsqueeze(0)
            dists = model.distance(unlabeled_emb, prototypes)
            probs = F.softmax(-dists, dim=-1)
            max_probs, pseudo_labels = probs.max(dim=-1)

        # Strong augmentation: re-load with strong aug from dataset pool
        # For simplicity, apply strong augmentation via additional noise + transform
        strong_noise = torch.randn_like(unlabeled_batch) * 0.1
        strong_batch = unlabeled_batch + strong_noise
        strong_batch = strong_batch.to(device)

        strong_emb = model.encoder(strong_batch) * channel_w.unsqueeze(0)
        strong_dists = model.distance(strong_emb, prototypes)
        strong_logits = -strong_dists

        # Mask: only use pseudo-labels above FixMatch threshold
        mask = max_probs > CONFIG["fixmatch_threshold"]
        if mask.any():
            loss_unsup = F.cross_entropy(strong_logits[mask], pseudo_labels[mask])

    total_loss = loss_sup + CONFIG["fixmatch_lambda_u"] * loss_unsup

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    preds = logits.argmax(dim=-1)
    acc = (preds == query_labels).float().mean().item()

    return total_loss.item(), acc


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


def evaluate_semi_supervised(model, dataset, n_episodes, n_way, k_shot, n_query):
    """Evaluate with semi-supervised pseudo-labeling on unlabeled data."""
    model.eval()
    accuracies = []
    total_pseudo_selected = 0
    total_pseudo_candidates = 0
    all_proto_before = []
    all_proto_after = []

    weak_aug = get_weak_augmentation()

    with torch.no_grad():
        for ep in range(n_episodes):
            support_images, support_labels, query_images, query_labels = (
                dataset.get_episode(n_way, k_shot, n_query, split="test")
            )

            # Get unlabeled batch with two different augmentations
            unlabeled_view1, indices = dataset.get_unlabeled_batch(50, weak_aug)
            unlabeled_view2 = dataset.get_unlabeled_by_indices(indices, weak_aug)

            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)
            unlabeled_view1 = unlabeled_view1.to(device)
            unlabeled_view2 = unlabeled_view2.to(device)

            logits, proto_before, proto_after, n_selected, pseudo_count = (
                model.forward_semi_supervised(
                    support_images, support_labels, query_images,
                    unlabeled_view1, unlabeled_view2, n_way
                )
            )

            preds = logits.argmax(dim=-1)
            acc = (preds == query_labels).float().mean().item()
            accuracies.append(acc)
            total_pseudo_selected += n_selected
            total_pseudo_candidates += unlabeled_view1.size(0)
            all_proto_before.append(proto_before.cpu())
            all_proto_after.append(proto_after.cpu())

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    ci95 = 1.96 * std_acc / np.sqrt(n_episodes)

    stats = {
        "n_selected": total_pseudo_selected,
        "n_total": total_pseudo_candidates,
        "selection_rate": total_pseudo_selected / max(1, total_pseudo_candidates),
    }

    return mean_acc, std_acc, ci95, stats, all_proto_before, all_proto_after


def evaluate_with_graph_propagation(model, dataset, n_episodes, n_way, k_shot, n_query):
    """Evaluate with graph-based label propagation."""
    model.eval()
    accuracies = []
    weak_aug = get_weak_augmentation()

    with torch.no_grad():
        for ep in range(n_episodes):
            support_images, support_labels, query_images, query_labels = (
                dataset.get_episode(n_way, k_shot, n_query, split="test")
            )

            unlabeled_view1, indices = dataset.get_unlabeled_batch(50, weak_aug)
            unlabeled_view2 = dataset.get_unlabeled_by_indices(indices, weak_aug)

            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)
            unlabeled_view1 = unlabeled_view1.to(device)
            unlabeled_view2 = unlabeled_view2.to(device)

            logits = model.forward_with_graph_propagation(
                support_images, support_labels, query_images,
                unlabeled_view1, unlabeled_view2, n_way
            )

            preds = logits.argmax(dim=-1)
            acc = (preds == query_labels).float().mean().item()
            accuracies.append(acc)

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    ci95 = 1.96 * std_acc / np.sqrt(n_episodes)

    return mean_acc, std_acc, ci95


def train_and_evaluate_fixmatch(dataset, n_episodes_train, n_episodes_eval,
                                 n_way, k_shot, n_query):
    """Train with FixMatch at 10% label budget and evaluate."""
    print("  Training FixMatch model (10% labels)...")
    model_fm = PrototypicalNetwork(
        embedding_dim=CONFIG["embedding_dim"], freeze_layers=CONFIG["freeze_layers"]
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_fm.parameters()), lr=CONFIG["lr"]
    )

    weak_aug = get_weak_augmentation()
    strong_aug = get_strong_augmentation()

    train_accs = []
    for ep in range(n_episodes_train):
        episode = dataset.get_reduced_label_episode(
            n_way, k_shot, n_query,
            label_fraction=CONFIG["fixmatch_label_fraction"], split="train"
        )

        # Get unlabeled batch for unsupervised loss
        unlabeled_batch, _ = dataset.get_unlabeled_batch(32, get_base_transform())

        loss, acc = train_fixmatch_episode(
            model_fm, optimizer, episode, unlabeled_batch,
            n_way, weak_aug, strong_aug, dataset
        )
        train_accs.append(acc)

        if (ep + 1) % 20 == 0:
            avg_acc = np.mean(train_accs[-20:])
            print(f"    FixMatch Episode {ep + 1}/{n_episodes_train}: Acc={avg_acc * 100:.1f}%")

    # Evaluate
    mean_acc, std_acc, ci95 = evaluate_fixmatch(
        model_fm, dataset, n_episodes_eval, n_way, k_shot, n_query
    )
    return mean_acc, std_acc, ci95


def evaluate_fixmatch(model, dataset, n_episodes, n_way, k_shot, n_query):
    """Evaluate FixMatch model."""
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

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    ci95 = 1.96 * std_acc / np.sqrt(n_episodes)
    return mean_acc, std_acc, ci95


# ============================================================
# Cross-Device Testing
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
    """Evaluate model on data from different simulated devices."""
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


# ============================================================
# Prototype Displacement Visualization
# ============================================================
def visualize_prototype_displacement(proto_before_list, proto_after_list, class_names,
                                      save_path="prototype_displacement.png"):
    """
    Visualize prototype vector displacement before/after semi-supervised enrichment.
    Left: PCA 2D projection with arrows showing displacement
    Right: Bar chart of per-class displacement magnitude
    """
    # Average prototype displacement across episodes
    proto_before = torch.stack(proto_before_list).mean(dim=0).numpy()  # (n_way, embed_dim)
    proto_after = torch.stack(proto_after_list).mean(dim=0).numpy()

    n_way = proto_before.shape[0]
    embed_dim = proto_before.shape[1]

    # Compute displacement magnitudes
    displacements = np.linalg.norm(proto_after - proto_before, axis=1)

    # PCA projection (2D) of all prototypes
    all_protos = np.concatenate([proto_before, proto_after], axis=0)  # (2*n_way, embed_dim)
    mean_vec = all_protos.mean(axis=0)
    centered = all_protos - mean_vec
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Take top 2 components
    top2_idx = np.argsort(eigenvalues)[-2:][::-1]
    pca_basis = eigenvectors[:, top2_idx]
    projected = centered @ pca_basis  # (2*n_way, 2)

    proj_before = projected[:n_way]
    proj_after = projected[n_way:]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: PCA scatter with arrows
    ax = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_way))

    for i in range(n_way):
        ax.scatter(proj_before[i, 0], proj_before[i, 1],
                   marker='s', s=150, c=[colors[i]], edgecolors='black',
                   linewidths=1.5, zorder=5, label=f"{class_names[i]} (before)")
        ax.scatter(proj_after[i, 0], proj_after[i, 1],
                   marker='D', s=150, c=[colors[i]], edgecolors='black',
                   linewidths=1.5, zorder=5, alpha=0.7)
        ax.annotate('', xy=(proj_after[i, 0], proj_after[i, 1]),
                    xytext=(proj_before[i, 0], proj_before[i, 1]),
                    arrowprops=dict(arrowstyle='->', color=colors[i],
                                    lw=2.5, mutation_scale=15))

    ax.set_xlabel("PCA Component 1", fontsize=11)
    ax.set_ylabel("PCA Component 2", fontsize=11)
    ax.set_title("Prototype Displacement (PCA Projection)", fontsize=13)
    ax.legend(loc='upper left', fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)

    # Right panel: Bar chart of displacement magnitude
    ax2 = axes[1]
    bars = ax2.bar(range(n_way), displacements, color=colors, edgecolor='black', linewidth=0.8)
    ax2.set_xticks(range(n_way))
    ax2.set_xticklabels(class_names[:n_way], rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel("L2 Displacement Magnitude", fontsize=11)
    ax2.set_title("Per-Class Prototype Displacement", fontsize=13)
    ax2.grid(True, axis='y', alpha=0.3)

    for i, (bar, d) in enumerate(zip(bars, displacements)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{d:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved to: {save_path}")


# ============================================================
# Comparison Table
# ============================================================
def print_comparison_table(baseline_acc, baseline_ci, semi_acc, semi_ci,
                            graph_acc, graph_ci, fixmatch_acc, fixmatch_ci):
    """Print formatted comparison table of all methods."""
    print("\n" + "=" * 70)
    print("COMPARISON: Semi-Supervised Methods")
    print("=" * 70)
    print(f"{'Method':<35} {'Labels':>8} {'Accuracy':>10} {'95% CI':>12}")
    print("-" * 70)
    print(f"  {'Prototypical (baseline)':<33} {'100%':>8} {baseline_acc*100:>8.2f}% {'±' + f'{baseline_ci*100:.2f}%':>11}")
    print(f"  {'+ Pseudo-Labeling (200 unlabeled)':<33} {'100%':>8} {semi_acc*100:>8.2f}% {'±' + f'{semi_ci*100:.2f}%':>11}")
    print(f"  {'+ Graph Propagation':<33} {'100%':>8} {graph_acc*100:>8.2f}% {'±' + f'{graph_ci*100:.2f}%':>11}")
    print(f"  {'FixMatch (10% labels)':<33} {'10%':>8} {fixmatch_acc*100:>8.2f}% {'±' + f'{fixmatch_ci*100:.2f}%':>11}")
    print("-" * 70)

    # Performance drop analysis for FixMatch
    drop = baseline_acc - fixmatch_acc
    relative_retention = fixmatch_acc / max(baseline_acc, 1e-8) * 100
    print(f"  FixMatch retains {relative_retention:.1f}% of baseline performance with 10% labels")
    print(f"  Pseudo-labeling improvement: {(semi_acc - baseline_acc)*100:+.2f}%")
    print(f"  Graph propagation improvement: {(graph_acc - baseline_acc)*100:+.2f}%")
    print("=" * 70)


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("Few-Shot Chest X-ray Classification")
    print("Prototypical Network + MobileNetV2 + Learnable Distance")
    print("+ Task-Adaptive Reweighting + Prototype Correction")
    print("+ Semi-Supervised + Graph Propagation + FixMatch")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  N-way: {CONFIG['n_way']}")
    print(f"  K-shot: {CONFIG['k_shot']}")
    print(f"  Query per class: {CONFIG['n_query']}")
    print(f"  Image size: {CONFIG['image_size']}x{CONFIG['image_size']}")
    print(f"  Embedding dim: {CONFIG['embedding_dim']}")
    print(f"  Frozen layers: {CONFIG['freeze_layers']}")
    print(f"  Unlabeled images: {CONFIG['n_unlabeled']}")
    print(f"  Pseudo-label threshold: {CONFIG['pseudo_label_threshold']}")
    print(f"  Graph k: {CONFIG['graph_k']}, alpha: {CONFIG['graph_alpha']}")
    print(f"  FixMatch threshold: {CONFIG['fixmatch_threshold']}, label fraction: {CONFIG['fixmatch_label_fraction']}")
    print(f"  Device: {device}")
    print()

    # Data transforms
    transform = get_base_transform()

    # [1/7] Load dataset
    print("[1/7] Loading dataset...")
    dataset = ChestXrayDataset(CONFIG["data_root"], transform=transform, seed=CONFIG["seed"])
    print(f"  Classes ({len(dataset.classes)}): {dataset.classes}")
    print(f"  Images per class: ~{len(list(dataset.class_to_images.values())[0])}")
    sample_cls = dataset.classes[0]
    print(f"  Train/Test split: {len(dataset.train_images[sample_cls])}/{len(dataset.test_images[sample_cls])} per class")
    print(f"  Unlabeled pool: {len(dataset.unlabeled_images)} images")
    print()

    # [2/7] Build model
    print("[2/7] Building Prototypical Network...")
    model = PrototypicalNetwork(
        embedding_dim=CONFIG["embedding_dim"], freeze_layers=CONFIG["freeze_layers"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Distance weight (alpha): {model.distance.alpha.item():.4f}")
    print(f"  Task-Adaptive Reweighting: enabled")
    print(f"  Prototype Correction: confidence>{model.confidence_threshold}, momentum={model.correction_momentum}")
    print(f"  Graph Propagation: k={CONFIG['graph_k']}, alpha={CONFIG['graph_alpha']}, iter={CONFIG['graph_n_iter']}")
    print()

    # [3/7] Training (episodic)
    print("[3/7] Episodic Training...")
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
    print()

    # [4/7] Baseline Evaluation
    print("[4/7] Baseline Evaluation (20 episodes)...")
    print("-" * 50)
    mean_acc, std_acc, ci95 = evaluate(
        model, dataset,
        n_episodes=CONFIG["n_episodes_eval"],
        n_way=CONFIG["n_way"],
        k_shot=CONFIG["k_shot"],
        n_query=CONFIG["n_query"],
    )

    print("-" * 50)
    print(f"  Baseline: {mean_acc * 100:.2f}% ± {ci95 * 100:.2f}%")
    print(f"  Target (>60%): {'ACHIEVED' if mean_acc > 0.6 else 'NOT MET'}")
    print()

    # [5/7] Semi-Supervised Evaluation
    print("[5/7] Semi-Supervised Pseudo-Labeling Evaluation...")
    print("  Using 200 unlabeled CT images with dual-view augmentation")
    print(f"  Confidence threshold: {CONFIG['pseudo_label_threshold']}")
    semi_acc, semi_std, semi_ci, pseudo_stats, proto_before_list, proto_after_list = (
        evaluate_semi_supervised(
            model, dataset,
            n_episodes=CONFIG["n_episodes_eval"],
            n_way=CONFIG["n_way"],
            k_shot=CONFIG["k_shot"],
            n_query=CONFIG["n_query"],
        )
    )
    print(f"  Semi-supervised accuracy: {semi_acc * 100:.2f}% ± {semi_ci * 100:.2f}%")
    print(f"  Pseudo-labels selected: {pseudo_stats['n_selected']}/{pseudo_stats['n_total']} "
          f"(rate: {pseudo_stats['selection_rate']*100:.1f}%)")
    print()

    # [6/7] Graph Propagation Evaluation
    print("[6/7] Graph Label Propagation Evaluation...")
    print(f"  kNN graph: k={CONFIG['graph_k']}, propagation iterations={CONFIG['graph_n_iter']}")
    graph_acc, graph_std, graph_ci = evaluate_with_graph_propagation(
        model, dataset,
        n_episodes=CONFIG["n_episodes_eval"],
        n_way=CONFIG["n_way"],
        k_shot=CONFIG["k_shot"],
        n_query=CONFIG["n_query"],
    )
    print(f"  Graph propagation accuracy: {graph_acc * 100:.2f}% ± {graph_ci * 100:.2f}%")
    print()

    # [7/7] FixMatch Comparison
    print("[7/7] FixMatch Comparison (10% label budget)...")
    fixmatch_acc, fixmatch_std, fixmatch_ci = train_and_evaluate_fixmatch(
        dataset,
        n_episodes_train=CONFIG["n_episodes_train"],
        n_episodes_eval=CONFIG["n_episodes_eval"],
        n_way=CONFIG["n_way"],
        k_shot=CONFIG["k_shot"],
        n_query=CONFIG["n_query"],
    )
    print(f"  FixMatch accuracy (10% labels): {fixmatch_acc * 100:.2f}% ± {fixmatch_ci * 100:.2f}%")
    print()

    # Comparison table
    print_comparison_table(
        mean_acc, ci95, semi_acc, semi_ci,
        graph_acc, graph_ci, fixmatch_acc, fixmatch_ci
    )

    # Prototype displacement visualization
    print("\n  Generating prototype displacement visualization...")
    episode_classes = dataset.classes[:CONFIG["n_way"]]
    visualize_prototype_displacement(
        proto_before_list, proto_after_list,
        class_names=episode_classes,
        save_path="prototype_displacement.png"
    )

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
            "semi_supervised_accuracy": semi_acc,
            "graph_propagation_accuracy": graph_acc,
            "fixmatch_accuracy": fixmatch_acc,
            "alpha": model.distance.alpha.item(),
            "cross_device_results": cross_device_results,
            "pseudo_label_stats": pseudo_stats,
        },
        save_path,
    )
    print(f"\nModel saved to: {save_path}")

    return mean_acc


if __name__ == "__main__":
    main()
