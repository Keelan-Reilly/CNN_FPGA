import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import argparse
import os

# Simple CNN: Conv -> ReLU -> MaxPool -> Dense
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1 input channel (MNIST grayscale), 8 output feature maps, 3x3 kernel, padding=1 to keep 28x28
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        # 2x2 max pooling reduces spatial size to 14x14
        self.pool = nn.MaxPool2d(2)
        # Fully connected layer: flattened 8*14*14 features to 10 classes
        self.fc1 = nn.Linear(8 * 14 * 14, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))     # convolution + ReLU
        x = self.pool(x)              # downsample
        x = x.view(x.size(0), -1)     # flatten
        x = self.fc1(x)               # dense logits
        return x

# Evaluation helper: compute accuracy over a DataLoader
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # no grads for eval
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total

# Training loop with early stopping at target accuracy
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # prefer GPU if available

    # Data transforms: convert to tensor in [0,1]
    transform = transforms.Compose([transforms.ToTensor()])

    # Load MNIST datasets (will download if missing)
    trainset = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    testset = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

    # Instantiate model, optimizer, loss
    model = SmallCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training epochs
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate on test set
        acc = evaluate(model, testloader, device)
        print(f"Epoch {epoch}: train_loss={total_loss/len(trainloader):.4f}, test_acc={acc:.2f}%")

        # Early stop if reached target accuracy
        if acc >= args.target_acc:
            print("Reached target accuracy; stopping early.")
            break

    # Save model and a sample batch for downstream verification
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "small_cnn.pth"))

    sample_data, sample_label = next(iter(testloader))
    # Save first 16 test samples (images + labels)
    torch.save({'images': sample_data[:16], 'labels': sample_label[:16]}, os.path.join(args.out_dir, "sample_batch.pt"))
    print(f"Model and sample batch saved to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train small CNN on MNIST")
    parser.add_argument('--data-dir', type=str, default='./data', help="MNIST download / storage directory")
    parser.add_argument('--out-dir', type=str, default='./output', help="Where to save trained model and samples")
    parser.add_argument('--epochs', type=int, default=10, help="Maximum training epochs")
    parser.add_argument('--batch-size', type=int, default=128, help="Training batch size")
    parser.add_argument('--target-acc', type=float, default=90.0, help="Early stop threshold on test accuracy (percent)")
    args = parser.parse_args()
    train(args)