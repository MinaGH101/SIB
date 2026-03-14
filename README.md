Merging Multiple YOLO Datasets

This repository includes a utility for combining multiple YOLO-format datasets into a single dataset while optionally remapping classes and splitting the dataset into train/validation/test sets.

This is useful when datasets use different class definitions but represent similar object types.

Input Dataset Format

Each dataset must follow the standard YOLO structure:

dataset/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│
├── labels/
│   ├── img1.txt
│   ├── img2.txt
│
└── data.yaml

Each label file must contain annotations in YOLO format:

class_id x_center y_center width height

The data.yaml file must contain the class names:

nc: 5
names: ['class1','class2','class3','class4','class5']
Class Mapping

Since different datasets may use different class names, the merger allows mapping them into a new unified class taxonomy.

Example mapping file:

{
  "tracked_armored": ["Battle-Tank", "M2A4-Slammer", "varsuk"],
  "wheeled_vehicle": ["Logistics-Vehicle", "marid", "zamak"],
  "support_system": ["Anti-air Defence", "Mobile-Radar"]
}

This means all objects labeled Battle-Tank, M2A4-Slammer, or varsuk will become the new class tracked_armored.

Running the Dataset Merger

Run the tool from the terminal:

python src/merge_yolo_datasets.py \
--datasets data/dataset1 data/dataset2 \
--output data/combined_dataset \
--mapping configs/class_mapping.json \
--split \
--ratio 0.8 0.1 0.1
Arguments

--datasets
List of input dataset paths.

--output
Destination folder where the merged dataset will be created.

--mapping
JSON file describing the class mapping.

--split
Enable dataset splitting into train/validation/test.

--ratio
Split ratio for train/validation/test.

Example:

--ratio 0.8 0.1 0.1

means

80% train
10% validation
10% test
Output Dataset Structure

After merging, the output dataset will look like:

combined_dataset/
├── train/
│   ├── images/
│   └── labels/
│
├── val/
│   ├── images/
│   └── labels/
│
├── test/
│   ├── images/
│   └── labels/
│
└── data.yaml

The generated data.yaml file will automatically include the new classes.

Example:

nc: 3
names: ['tracked_armored','wheeled_vehicle','support_system']

train: ./train/images
val: ./val/images
test: ./test/images
Notes

Images from different datasets are renamed automatically to avoid filename conflicts.

Only classes specified in the mapping file will be included.

Images without mapped objects will be skipped.