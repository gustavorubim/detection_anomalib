import os
import torch
import warnings

warnings.simplefilter('ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message=r".*openvino\.runtime.*deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning,
                        message=r".*Importing from timm\.models\.layers is deprecated.*")

from anomalib.data import Folder
from anomalib.models import (
    Dfm, EfficientAd, Padim, Patchcore,
    ReverseDistillation, Stfpm, Supersimplenet, Uflow,
    Cflow, Csflow, Draem, Fastflow, Ganomaly, 
    Dfkde, WinClip, Cfa, Dsr
)
from anomalib.engine import Engine
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from anomalib.data.dataclasses.torch.image import ImageBatch
from anomalib.visualization import visualize_anomaly_map, ImageVisualizer
from anomalib.loggers import AnomalibTensorBoardLogger

# Set matrix multiplication precision for CUDA devices with Tensor Cores
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

MODELS = {
    "DFM": Dfm, # 0
    "EfficientAD": EfficientAd, # 1
    "PaDiM": Padim, # 2
    "PatchCore": Patchcore, # 3
    "ReverseDistillation": ReverseDistillation, # 4
    "STFPM": Stfpm, # 5
    "U-Flow": Uflow, # 6
    "CFlow": Cflow, # 7
    "CS-Flow": Csflow, # 8
    "GANomaly": Ganomaly, # 9
    "DFKDE": Dfkde, # 10
    "CFA": Cfa, # 11
    "DSR": Dsr,  # 12
    "DRAEM": Draem, # 13

    # "FastFlow": Fastflow,
    # "SuperSimpleNet": Supersimplenet,
    # "WinCLIP": WinClip,
}

SINGLE_EPOCH_MODELS = ["PaDiM", "PatchCore", "WinCLIP"]
SMALL_BATCH_MODELS = ["EfficientAD", "CFlow", "CS-Flow"]

# Custom datamodule that sets persistent_workers=True and uses ImageBatch.collate.
class CustomFolder(Folder):
    def train_dataloader(self) -> DataLoader:
        if not hasattr(self, "train_data"):
            self.setup(stage="fit")
        return DataLoader(
            dataset=self.train_data,  # Folder creates train_data
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=ImageBatch.collate,  # Use the custom collate function
        )

    def val_dataloader(self) -> DataLoader:
        if not hasattr(self, "val_data"):
            self.setup(stage="fit")
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=ImageBatch.collate,
        )
    
    def test_dataloader(self) -> DataLoader:
        if not hasattr(self, "test_data"):
            self.setup(stage="test")
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=ImageBatch.collate,
        )

def train_model(model_name, model_class, dataset_path):
    print(f"\n--- Training {model_name} ---")
    max_epochs = 1 if model_name in SINGLE_EPOCH_MODELS else 100
    train_batch_size = 1 if model_name in SMALL_BATCH_MODELS else 32
    eval_batch_size = 1 if model_name in SMALL_BATCH_MODELS else 32
    print(f"Using {max_epochs} epochs and batch size {train_batch_size}")

    tensorboard_logger = AnomalibTensorBoardLogger("logs")

    callbacks = [
            ModelCheckpoint(
                mode="max",
                monitor="pixel_AUROC",
            ),
            EarlyStopping(
                monitor="pixel_AUROC",
                mode="max",
                patience=3,
            ),
        ]
    engine = Engine(
        callbacks=callbacks,
        max_epochs=max_epochs,
        default_root_dir=os.path.join("results", model_name, "v1"),
        logger=tensorboard_logger
    )
    
    datamodule = CustomFolder(
        name="custom_dataset",
        root=dataset_path,
        normal_dir="train/good",
        abnormal_dir="test",
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=0,
    )
    
    
    model = model_class()
    engine.fit(datamodule=datamodule, model=model )

def main():
    dataset_path = "./datasets"  # Set your dataset path here
    for model_name, model_class in MODELS.items():
        train_model(model_name, model_class, dataset_path)

if __name__ == "__main__":
    main()
