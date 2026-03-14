from pathlib import Path
import shutil
import random
import yaml
import argparse
import json

class DatasetMerger:
    """
    Utility for merging multiple YOLO-format datasets into a single dataset
    with optional class remapping and dataset splitting.

    The input datasets must follow the standard YOLO structure:

        dataset/
        ├── images/
        ├── labels/
        └── data.yaml

    This tool allows:
    - merging several datasets
    - mapping original classes to new classes
    - optional train/val/test splitting
    - automatic generation of a new data.yaml

    Example
    -------
    >>> class_mapping = {
    ...     "tracked_armored": ["Battle-Tank", "M2A4-Slammer", "varsuk"],
    ...     "wheeled_vehicle": ["Logistics-Vehicle", "marid", "zamak"],
    ...     "support_system": ["Anti-air Defence", "Mobile-Radar"]
    ... }
    ...
    >>> merger = YoloDatasetMerger(
    ...     dataset_paths=["dataset1", "dataset2"],
    ...     output_dir="combined_dataset",
    ...     class_mapping=class_mapping,
    ...     split=True,
    ...     split_ratio=(0.8, 0.1, 0.1)
    ... )
    >>> merger.run()
    """

    def __init__(
        self,
        dataset_paths,
        output_dir,
        class_mapping,
        split=False,
        split_ratio=(0.8, 0.1, 0.1),
        seed=42,
    ):
        """
        Parameters
        ----------
        dataset_paths : list[str | Path]
            Paths to input YOLO datasets.

        output_dir : str | Path
            Destination directory for the merged dataset.

        class_mapping : dict
            Mapping from new class names to original class names.

            Example:
            {
                "tracked_armored": ["Battle-Tank", "M2A4-Slammer", "varsuk"],
                "wheeled_vehicle": ["Logistics-Vehicle", "marid", "zamak"]
            }

        split : bool, default=False
            If True, the merged dataset will be split into
            train / val / test subsets.

        split_ratio : tuple(float,float,float)
            Fraction of samples assigned to train, val, test.

        seed : int
            Random seed used for dataset shuffling.
        """

        self.dataset_paths = [Path(p) for p in dataset_paths]
        self.output_dir = Path(output_dir)
        self.class_mapping = class_mapping
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed

        self.image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        self.target_classes = list(class_mapping.keys())
        self.target_class_to_id = {
            name: i for i, name in enumerate(self.target_classes)
        }

        self._validate_inputs()

    def _validate_inputs(self):
        """Validate dataset paths and split configuration."""

        if len(self.split_ratio) != 3:
            raise ValueError("split_ratio must contain (train,val,test)")

        if abs(sum(self.split_ratio) - 1.0) > 1e-6:
            raise ValueError("split_ratio must sum to 1")

        for ds in self.dataset_paths:
            if not ds.exists():
                raise FileNotFoundError(f"Dataset not found: {ds}")

            if not (ds / "images").exists():
                raise FileNotFoundError(f"Missing images folder in {ds}")

            if not (ds / "labels").exists():
                raise FileNotFoundError(f"Missing labels folder in {ds}")

    def _load_dataset_classes(self, dataset_path):
        """Load class names from a dataset's data.yaml file."""

        yaml_path = dataset_path / "data.yaml"

        if not yaml_path.exists():
            raise FileNotFoundError(f"Missing data.yaml in {dataset_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        names = data["names"]

        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys(), key=int)]

        return names

    def _build_reverse_mapping(self):
        """
        Convert class_mapping into a reverse lookup table.

        Returns
        -------
        dict
            {original_class_name : new_class_id}
        """

        reverse = {}

        for target_class, source_classes in self.class_mapping.items():
            target_id = self.target_class_to_id[target_class]

            for src in source_classes:
                reverse[src] = target_id

        return reverse

    def _prepare_output_dirs(self):
        """Create output directory structure."""

        if self.split:
            for split in ["train", "val", "test"]:
                (self.output_dir / split / "images").mkdir(
                    parents=True, exist_ok=True
                )
                (self.output_dir / split / "labels").mkdir(
                    parents=True, exist_ok=True
                )
        else:
            (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels").mkdir(parents=True, exist_ok=True)

    def _collect_samples(self):
        """
        Read all images and labels and apply class mapping.

        Returns
        -------
        list[dict]
            List of merged samples.
        """

        reverse = self._build_reverse_mapping()
        samples = []

        for ds in self.dataset_paths:
            class_names = self._load_dataset_classes(ds)

            for img_path in (ds / "images").iterdir():

                if img_path.suffix.lower() not in self.image_exts:
                    continue

                label_path = ds / "labels" / f"{img_path.stem}.txt"

                if not label_path.exists():
                    continue

                new_lines = []

                with open(label_path) as f:
                    for line in f:

                        parts = line.strip().split()

                        if len(parts) < 5:
                            continue

                        old_id = int(parts[0])

                        if old_id >= len(class_names):
                            continue

                        old_name = class_names[old_id]

                        if old_name not in reverse:
                            continue

                        new_id = reverse[old_name]

                        x, y, w, h = parts[1:5]

                        new_lines.append(
                            f"{new_id} {x} {y} {w} {h}"
                        )

                if new_lines:
                    samples.append({
                        "dataset": ds.name,
                        "image": img_path,
                        "labels": new_lines
                    })

        return samples

    def _split_samples(self, samples):
        """Split dataset into train / val / test."""

        random.seed(self.seed)
        random.shuffle(samples)

        n = len(samples)

        n_train = int(n * self.split_ratio[0])
        n_val = int(n * self.split_ratio[1])

        return {
            "train": samples[:n_train],
            "val": samples[n_train:n_train+n_val],
            "test": samples[n_train+n_val:]
        }

    def _write_samples(self, splits):
        """Write images and labels to disk."""

        for split_name, samples in splits.items():

            for sample in samples:

                new_img = f"{sample['dataset']}_{sample['image'].name}"
                new_lbl = f"{Path(new_img).stem}.txt"

                img_out = self.output_dir / split_name / "images" / new_img
                lbl_out = self.output_dir / split_name / "labels" / new_lbl

                shutil.copy(sample["image"], img_out)

                with open(lbl_out, "w") as f:
                    f.write("\n".join(sample["labels"]))

    def _write_data_yaml(self):
        """Generate YOLO data.yaml for the merged dataset."""

        data = {
            "nc": len(self.target_classes),
            "names": self.target_classes,
        }

        if self.split:
            data["train"] = "./train/images"
            data["val"] = "./val/images"
            data["test"] = "./test/images"
        else:
            data["train"] = "./images"
            data["val"] = "./images"

        with open(self.output_dir / "data.yaml", "w") as f:
            yaml.dump(data, f, sort_keys=False)

    def run(self):
        """
        Execute dataset merge.

        Steps
        -----
        1. Validate input datasets
        2. Collect and remap annotations
        3. Optionally split dataset
        4. Write merged dataset
        5. Generate data.yaml
        """

        self._prepare_output_dirs()

        samples = self._collect_samples()

        if self.split:
            splits = self._split_samples(samples)
        else:
            splits = {"all": samples}

        if self.split:
            self._write_samples(splits)
        else:
            self._write_samples({"train": samples})

        self._write_data_yaml()

        print("Dataset merge complete.")
        print(f"Output: {self.output_dir}")
        print(f"Classes: {self.target_classes}")
        print(f"Total images: {len(samples)}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of dataset paths"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output dataset folder"
    )

    parser.add_argument(
        "--mapping",
        required=True,
        help="JSON file with class mapping"
    )

    parser.add_argument(
        "--split",
        action="store_true"
    )

    parser.add_argument(
        "--ratio",
        nargs=3,
        type=float,
        default=[0.8,0.1,0.1]
    )

    args = parser.parse_args()

    with open(args.mapping) as f:
        mapping = json.load(f)

    merger = DatasetMerger(
        dataset_paths=args.datasets,
        output_dir=args.output,
        class_mapping=mapping,
        split=args.split,
        split_ratio=tuple(args.ratio)
    )

    merger.run()