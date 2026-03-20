from __future__ import annotations
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
import random
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


@dataclass
class DatasetReport:
    """Container for dataset statistics."""
    dataset_path: str
    num_images: int
    num_labels: int
    num_classes: int
    class_names: list[str]
    object_count_per_class: dict
    image_count_per_class: dict


class DatasetExplorer:
    """
    Explore YOLO datasets and generate statistics and visualizations.

    Features
    --------
    • Counts images, labels and classes  
    • Computes class distribution  
    • Generates colorful class distribution plots  
    • Optionally shows random annotated image samples  

    Expected Dataset Structure
    --------------------------
    dataset/
        images/
        labels/
        data.yaml

    Parameters
    ----------
    dataset_paths : list[str | Path]
        Paths of YOLO datasets to analyze.

    plot : bool
        If True, plot class distribution charts.

    plot_samples : bool
        If True, show 5 random annotated images.

    sample_count : int
        Number of sample images to visualize.
    """

    def __init__(
        self,
        dataset_paths,
        plot=True,
        plot_samples=False,
        sample_count=3
    ):
        self.dataset_paths = [Path(p) for p in dataset_paths]
        self.plot = plot
        self.plot_samples = plot_samples
        self.sample_count = sample_count
        self.image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def _load_classes(self, dataset_path):
        """Load class names from data.yaml."""
        with open(dataset_path / "data.yaml") as f:
            data = yaml.safe_load(f)

        names = data["names"]

        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]

        return names

    def _collect_paths(self, dataset_path):
        """Collect image and label paths."""
        img_dir = dataset_path / "images"
        lbl_dir = dataset_path / "labels"

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in self.image_exts]
        labels = list(lbl_dir.glob("*.txt"))

        return images, labels

    def _parse_label(self, label_path):
        """Parse YOLO label file."""
        ids = []

        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    ids.append(int(parts[0]))

        return ids

    def _build_report(self, dataset_path):

        class_names = self._load_classes(dataset_path)
        images, labels = self._collect_paths(dataset_path)

        obj_counter = Counter()
        img_counter = Counter()

        image_map = {p.stem: p for p in images}

        for lbl in labels:

            ids = self._parse_label(lbl)

            if not ids:
                continue

            obj_counter.update(ids)

            for i in set(ids):
                img_counter[i] += 1

        return DatasetReport(
            dataset_path=str(dataset_path),
            num_images=len(images),
            num_labels=len(labels),
            num_classes=len(class_names),
            class_names=class_names,
            object_count_per_class={
                class_names[i]: obj_counter.get(i, 0)
                for i in range(len(class_names))
            },
            image_count_per_class={
                class_names[i]: img_counter.get(i, 0)
                for i in range(len(class_names))
            }
        )

    def _plot_distribution(self, report):

        names = report.class_names
        objs = [report.object_count_per_class[n] for n in names]
        imgs = [report.image_count_per_class[n] for n in names]

        x = range(len(names))

        plt.figure(figsize=(10, 5))

        plt.bar(x, objs, color="tab:blue", label="objects")
        plt.bar(x, imgs, color="tab:orange", alpha=0.7, label="images")

        plt.xticks(x, names, rotation=30)
        plt.title(f"Class distribution: {Path(report.dataset_path).name}")
        plt.ylabel("count")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _plot_samples(self, dataset_path, class_names):

        img_dir = dataset_path / "images"
        lbl_dir = dataset_path / "labels"

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in self.image_exts]

        samples = random.sample(images, min(self.sample_count, len(images)))

        fig, axes = plt.subplots(1, len(samples), figsize=(15, 4))

        if len(samples) == 1:
            axes = [axes]

        for ax, img_path in zip(axes, samples):

            img = Image.open(img_path)
            w, h = img.size

            ax.imshow(img)

            lbl = lbl_dir / f"{img_path.stem}.txt"

            if lbl.exists():

                with open(lbl) as f:
                    for line in f:

                        cls, x, y, bw, bh = map(float, line.split())

                        cls = int(cls)

                        x *= w
                        y *= h
                        bw *= w
                        bh *= h

                        xmin = x - bw/2
                        ymin = y - bh/2

                        rect = patches.Rectangle(
                            (xmin, ymin),
                            bw,
                            bh,
                            linewidth=2,
                            edgecolor="lime",
                            facecolor="none"
                        )

                        ax.add_patch(rect)

                        ax.text(
                            xmin,
                            ymin,
                            class_names[cls],
                            color="white",
                            fontsize=8,
                            bbox=dict(facecolor="green", alpha=0.6)
                        )

            ax.axis("off")

        plt.suptitle("Random annotated samples", fontsize=14)
        plt.show()

    def run(self):
        """Run exploration for all datasets."""

        reports = []

        for ds in self.dataset_paths:

            report = self._build_report(ds)

            reports.append(report)

            print("\n==============================")
            print("DATASET:", report.dataset_path)
            print("==============================")

            print("images:", report.num_images)
            print("labels:", report.num_labels)
            print("classes:", report.class_names)

            print("\nobjects per class:")
            for k, v in report.object_count_per_class.items():
                print(f"  {k}: {v}")

            if self.plot:
                self._plot_distribution(report)

            if self.plot_samples:
                self._plot_samples(ds, report.class_names)

        return reports