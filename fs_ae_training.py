import os, random, copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from sklearn.neighbors import KernelDensity

from torch.utils.tensorboard import SummaryWriter

# ---------------------- Parameters ----------------------
SIZE = 128            # image size for training (if using full 1024, consider adjustments)
BATCH_SIZE = 244
NUM_EPOCHS = 100      # maximum epochs; early stopping may halt earlier
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 10   # epochs without improvement before stopping
SCHEDULER_PATIENCE = 3         # epochs with no improvement for lr scheduler
SCHEDULER_FACTOR = 0.5         # factor by which to reduce learning rate
LOG_DIR = "runs/autoencoder_experiment"
RESULTS_DIR = "torch_results"  # folder to save result images

# Global reconstruction error threshold (for algorithm prediction)
global_recon_threshold = 0.005
# Pixel-level threshold for binary anomaly mask (adjust based on your data)
pixel_threshold = 0.05

# ---------------------- Prepare Results Directory ----------------------
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# ---------------------- Transforms ----------------------
transform = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor()
])

# ---------------------- Datasets ----------------------
class SimpleImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root)
                            if fname.lower().endswith(('.png','.jpg','.jpeg'))]
    def __len__(self):
         return len(self.image_paths)
    def __getitem__(self, idx):
         img = Image.open(self.image_paths[idx]).convert('RGB')
         if self.transform:
              img = self.transform(img)
         return img, 0

train_path = os.path.join('datasets', 'train')
test_path = os.path.join('datasets', 'test')
good_for_test_path = os.path.join('datasets', 'good_for_test')

# Training uses ImageFolder (expects subfolder, e.g. "good")
train_dataset = ImageFolder(root=train_path, transform=transform)
# Custom datasets for test and good_for_test which have images directly
test_dataset = SimpleImageDataset(root=test_path, transform=transform)
good_test_dataset = SimpleImageDataset(root=good_for_test_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
good_test_loader = DataLoader(good_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------- Model Definition ----------------------
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),    # (64,128,128)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # (64,64,64)
            nn.Conv2d(64, 32, kernel_size=3, padding=1),    # (32,64,64)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # (32,32,32)
            nn.Conv2d(32, 16, kernel_size=3, padding=1),    # (16,32,32)
            nn.ReLU(),
            nn.MaxPool2d(2)                                # (16,16,16)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),    # (16,16,16)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),     # (16,32,32)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),    # (32,32,32)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),     # (32,64,64)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),    # (64,64,64)
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),     # (64,128,128)
            nn.Conv2d(64, 3, kernel_size=3, padding=1),     # (3,128,128)
            nn.Sigmoid()
        )
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon

model = ConvAutoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=SCHEDULER_FACTOR,
                                                 patience=SCHEDULER_PATIENCE, verbose=True)

# ---------------------- TensorBoard Setup ----------------------
writer = SummaryWriter(LOG_DIR)

# ---------------------- Early Stopping Setup ----------------------
best_val_loss = float('inf')
patience_counter = 0
best_model_wts = copy.deepcopy(model.state_dict())

# ---------------------- Training Loop ----------------------
train_losses = []
val_losses = []

for epoch in range(1, NUM_EPOCHS+1):
    model.train()
    running_loss = 0.0
    for imgs, _ in train_loader:
        imgs = imgs.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    
    # Evaluate on good_for_test as validation
    model.eval()
    running_val = 0.0
    with torch.no_grad():
        for imgs, _ in good_test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            running_val += loss.item() * imgs.size(0)
    epoch_val = running_val / len(good_test_loader.dataset)
    val_losses.append(epoch_val)
    
    # Log to TensorBoard
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Loss/validation', epoch_val, epoch)
    writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
    
    print(f"Epoch [{epoch}/{NUM_EPOCHS}] Train Loss: {epoch_loss:.6f} Val Loss: {epoch_val:.6f}")
    
    # Scheduler update
    scheduler.step(epoch_val)
    
    # Early stopping check
    if epoch_val < best_val_loss:
        best_val_loss = epoch_val
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered")
            break

model.load_state_dict(best_model_wts)

# ---------------------- Save Training Plots ----------------------
plt.figure()
plt.plot(range(1, len(train_losses)+1), train_losses, 'y', label='Training loss')
plt.plot(range(1, len(val_losses)+1), val_losses, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "training_loss.png"))
plt.close()

# ---------------------- Visualization of Reconstructions ----------------------
def save_reconstruction_images(loader, filename_prefix, num_images=5):
    model.eval()
    imgs, _ = next(iter(loader))
    imgs = imgs.to(DEVICE)
    with torch.no_grad():
        recons = model(imgs)
    imgs_np = imgs.cpu().numpy().transpose(0,2,3,1)
    recons_np = recons.cpu().numpy().transpose(0,2,3,1)
    # Save individual results later from a random subset
    indices = random.sample(range(imgs_np.shape[0]), min(num_images, imgs_np.shape[0]))
    for idx in indices:
        # Original image
        orig = imgs_np[idx]
        # Reconstruction
        recon = recons_np[idx]
        # Compute anomaly heatmap as absolute difference (mean over channels)
        heatmap = np.mean(np.abs(orig - recon), axis=2)
        # Normalize heatmap to [0,1] for visualization
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max()-heatmap.min() + 1e-8)
        # Predicted binary mask from the heatmap (50% threshold: adjust pixel_threshold as necessary)
        mask = (heatmap_norm > pixel_threshold).astype(np.float32)
        # Overlay the heatmap on the original image.
        # First, get a colored heatmap using a colormap (e.g., 'jet')
        cmap = plt.get_cmap('jet')
        heatmap_color = cmap(heatmap_norm)[:,:,:3]  # drop alpha channel
        overlay = orig * 0.6 + heatmap_color * 0.4
        # Overall image-level anomaly decision (using mean recon error)
        image_error = np.mean(np.abs(orig - recon))
        prediction = "Anomaly" if image_error > global_recon_threshold else "Not Anomaly"
        
        # Create a figure with 4 subplots (horizontal grid)
        fig, axes = plt.subplots(1, 4, figsize=(20,5))
        axes[0].imshow(orig)
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        im1 = axes[1].imshow(heatmap_norm, cmap='jet')
        axes[1].set_title("Anomaly Heatmap")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1])
        
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")
        
        axes[3].imshow(mask, cmap='gray')
        axes[3].set_title("Predicted Mask")
        axes[3].axis("off")
        
        fig.suptitle(f"Algorithm Prediction: {prediction}", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        # Save the figure in the results directory
        save_path = os.path.join(RESULTS_DIR, f"{filename_prefix}_{idx}.png")
        fig.savefig(save_path)
        plt.close(fig)

# Use the custom function to save results from 5 random defective test images
save_reconstruction_images(test_loader, filename_prefix="defective")

writer.close()
