import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = PROJECT_ROOT / "datasets" / "classifiers" / "demographic_train.csv"
VAL_CSV = PROJECT_ROOT / "datasets" / "classifiers" / "demographic_val.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


TASK_CONFIGS = {
    "gender": {
        "label_column": "gender_label",
        "name_column": "gender_name",
        "num_classes": 2,
        "class_names": ["male", "female"],
        "output_model": MODELS_DIR / "gender_classifier.pt",
        "output_report": MODELS_DIR / "gender_classifier_report.txt",
    },
    "age_group": {
        "label_column": "age_group_label",
        "name_column": "age_group_name",
        "num_classes": 3,
        "class_names": ["child", "young", "old"],
        "output_model": MODELS_DIR / "age_group_classifier.pt",
        "output_report": MODELS_DIR / "age_group_classifier_report.txt",
    },
}


class DemographicDataset(Dataset):
    def __init__(self, csv_path, label_column, transform=None):
        self.df = pd.read_csv(csv_path)
        self.label_column = label_column
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = row["image_path"]
        label = int(row[self.label_column])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def build_model(num_classes):
    try:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        print("Using pretrained MobileNetV3-Small weights.")
    except Exception as e:
        print("Pretrained weights could not be loaded. Using random initialization.")
        print(f"Reason: {e}")
        model = mobilenet_v3_small(weights=None)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy, all_labels, all_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["gender", "age_group"],
        required=True,
        help="Which classifier to train."
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    config = TASK_CONFIGS[args.task]

    print("==============================")
    print(f"Training task: {args.task}")
    print("==============================")
    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Val CSV:   {VAL_CSV}")
    print(f"Output:    {config['output_model']}")
    print()

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    print("Train distribution:")
    print(train_df[config["name_column"]].value_counts())
    print()

    print("Val distribution:")
    print(val_df[config["name_column"]].value_counts())
    print()

    device = get_device()
    print(f"Device: {device}")
    print()

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = DemographicDataset(
        TRAIN_CSV,
        label_column=config["label_column"],
        transform=train_transform
    )

    val_dataset = DemographicDataset(
        VAL_CSV,
        label_column=config["label_column"],
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = build_model(config["num_classes"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    best_val_acc = 0.0
    best_state = None

    history_lines = []

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc, val_labels, val_predictions = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        line = (
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}"
        )

        print(line)
        print()

        history_lines.append(line)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_acc, val_labels, val_predictions = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    report = classification_report(
        val_labels,
        val_predictions,
        target_names=config["class_names"]
    )

    cm = confusion_matrix(val_labels, val_predictions)
    acc = accuracy_score(val_labels, val_predictions)

    print("==============================")
    print("Final Evaluation")
    print("==============================")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Final val accuracy: {acc:.4f}")
    print()
    print("Confusion matrix:")
    print(cm)
    print()
    print("Classification report:")
    print(report)

    checkpoint = {
        "task": args.task,
        "model_name": "mobilenet_v3_small",
        "num_classes": config["num_classes"],
        "class_names": config["class_names"],
        "state_dict": model.state_dict(),
        "image_size": 224,
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
    }

    torch.save(checkpoint, config["output_model"])

    with open(config["output_report"], "w", encoding="utf-8") as f:
        f.write(f"Task: {args.task}\n")
        f.write("Model: MobileNetV3-Small\n")
        f.write(f"Best val accuracy: {best_val_acc:.4f}\n")
        f.write(f"Final val accuracy: {acc:.4f}\n\n")

        f.write("Training history:\n")
        for line in history_lines:
            f.write(line + "\n")

        f.write("\nConfusion matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification report:\n")
        f.write(report)

    print()
    print("==============================")
    print("Saved")
    print("==============================")
    print(f"Model saved to: {config['output_model']}")
    print(f"Report saved to: {config['output_report']}")


if __name__ == "__main__":
    main()
