from pathlib import Path
import torch
from anomalib.data import Folder
from anomalib.models import Padim
from anomalib.engine import Engine
from anomalib.utils.post_processing import superimpose_anomaly_map
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from anomalib.models import (
    Dfm, EfficientAd, Padim, Patchcore,
    ReverseDistillation, Stfpm, Supersimplenet, Uflow,
    Cflow, Csflow, Draem, Fastflow, Ganomaly, 
    Dfkde, WinClip, Cfa, Dsr
)

# Step 1: Define the path to your dataset
dataset_root = "./datasets"

# Step 2: Configure the datamodule using the Folder class
datamodule = Folder(
    name="custom_dataset",  # Provide a name for your dataset
    root=dataset_root,
    normal_dir="train/good",
    abnormal_dir="test"
)


# Step 1: Define the mapping of algorithm names to their respective model classes and checkpoint paths
model_info = {
    "CFA": {
        "class": Cfa,
        "checkpoint": Path("results/CFA/v1/Cfa/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "CFlow": {
        "class": Cflow,
        "checkpoint": Path("results/CFlow/v1/Cflow/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "CS-Flow": {
        "class": Csflow,
        "checkpoint": Path("results/CS-Flow/v1/Csflow/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "DFKDE": {
        "class": Dfkde,
        "checkpoint": Path("results/DFKDE/v1/Dfkde/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "DFM": {
        "class": Dfm,
        "checkpoint": Path("results/DFM/v1/Dfm/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "DRAEM": {
        "class": Draem,
        "checkpoint": Path("results/DRAEM/v1/Draem/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "DSR": {
        "class": Dsr,
        "checkpoint": Path("results/DSR/v1/Dsr/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "EfficientAD": {
        "class": EfficientAd,
        "checkpoint": Path("results/EfficientAD/v1/EfficientAd/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "GANomaly": {
        "class": Ganomaly,
        "checkpoint": Path("results/GANomaly/v1/Ganomaly/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "PaDiM": {
        "class": Padim,
        "checkpoint": Path("results/PaDiM/v1/Padim/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "PatchCore": {
        "class": Patchcore,
        "checkpoint": Path("results/PatchCore/v1/Patchcore/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "ReverseDistillation": {
        "class": ReverseDistillation,
        "checkpoint": Path("results/ReverseDistillation/v1/ReverseDistillation/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "STFPM": {
        "class": Stfpm,
        "checkpoint": Path("results/STFPM/v1/Stfpm/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
    "U-Flow": {
        "class": Uflow,
        "checkpoint": Path("results/U-Flow/v1/Uflow/custom_dataset/latest/weights/lightning/model.ckpt"),
    },
}

# Step 2: Select the algorithm
selected_algorithm = "GANomaly"  # Change this to select a different algorithm


# Step 3: Load the model architecture and weights
if selected_algorithm in model_info:
    model_class = model_info[selected_algorithm]["class"]
    checkpoint_path = model_info[selected_algorithm]["checkpoint"]
    if checkpoint_path.exists():
        model = model_class.load_from_checkpoint(checkpoint_path)
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
else:
    raise ValueError(f"Algorithm {selected_algorithm} is not recognized.")



# Step 4: Set up the engine
engine = Engine(accelerator="auto", devices=1, logger=False)

# Step 5: Perform inference
# engine.test(datamodule=datamodule, model=model)



# Step 6: Predict on a single image
dataset_root = Path("./datasets")
data_path = dataset_root / "test" / "000_fold.png"
predictions = engine.predict(model=model, data_path=data_path)
prediction = predictions[0]  # Get the first and only prediction
# print("Prediction:", prediction)


# Step 7: Visualize the results
image_path = prediction.image_path[0]
image_size = prediction.image.shape[-2:]
image = np.array(Image.open(image_path).resize(image_size))



anomaly_map = prediction.anomaly_map[0].cpu().numpy().squeeze()
plt.imshow(anomaly_map)
plt.title("Anomaly Map")
plt.show()

heat_map = superimpose_anomaly_map(anomaly_map=anomaly_map, image=image, normalize=True)
plt.imshow(heat_map)
plt.title("Heat Map")
plt.show()

pred_score = prediction.pred_score[0].item()
pred_label = prediction.pred_label[0].item()
print(f"Prediction Score: {pred_score}, Prediction Label: {pred_label}")

pred_mask = prediction.pred_mask[0].squeeze().cpu().numpy()
plt.imshow(pred_mask)
plt.title("Predicted Mask")
plt.show()
