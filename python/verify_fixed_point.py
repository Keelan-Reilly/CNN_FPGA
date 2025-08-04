import torch
import os
import argparse
from quantise import SmallCNN  # reuse the same model definition
from pathlib import Path

def load_sample_images(sample_path):
    """Load saved sample batch (images + labels)."""
    d = torch.load(sample_path)
    return d['images'], d['labels']

def evaluate_float(model, images, labels):
    """Evaluate the float (original) model on the sample images and report accuracy."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(images.shape[0]):
            img = images[i].unsqueeze(0)  # add batch dim
            out = model(img)
            pred = out.argmax(dim=1).item()
            if pred == labels[i].item():
                correct += 1
    acc = 100.0 * correct / images.shape[0]
    print(f"Float model accuracy on sample batch: {acc:.2f}%")

def main(args):
    # Load sample batch and model
    images, labels = load_sample_images(args.sample)
    model = SmallCNN()
    state_path = os.path.join(args.model_dir, "small_cnn.pth")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Model weights not found at {state_path}")
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)

    # Evaluate float model
    evaluate_float(model, images, labels)

    # Placeholder for future: integrate quantised forward pass simulation and comparison with Verilog output

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify float-model accuracy on saved sample batch")
    parser.add_argument('--sample', type=str, default='./output/sample_batch.pt', help="Path to saved sample batch")
    parser.add_argument('--model-dir', type=str, default='./output', help="Directory containing saved model")
    args = parser.parse_args()
    main(args)