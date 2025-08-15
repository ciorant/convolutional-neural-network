import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Optional, Union

class Convolutional(nn.Module):
    def __init__(self, input_channels: int, hidden_units: int, num_classes: int, image_size: Tuple[int, int]):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.image_size = image_size

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Dynamically determine flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *image_size)  # e.g., (1, 1, 28, 28)
            dummy_out = self.block2(self.block1(dummy))
            flattened_size = dummy_out.numel()  # total number of elements
        # --------------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, num_classes),
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = None

    def forward(self, x) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

    def setup_training(self,optimizer: torch.optim, learning_rate: float = 0.01, weight_decay: float = 0):
        self.optimizer = optimizer(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.to(self.device)

    def calculate_accuracy(self, y_pred_logits : torch.Tensor, y_true : torch.Tensor) -> float:
        y_pred = torch.argmax(y_pred_logits, dim=1)
        return (y_pred == y_true).float().mean().cpu().item()

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float,float]:
        self.train()
        epoch_loss, epoch_acc = 0.0, 0.0

        for batch_id, (X,y) in enumerate(dataloader):
            X,y = X.to(self.device), y.to(self.device)
            y_pred = self(X)
            loss = self.loss_fn(y_pred,y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += self.calculate_accuracy(y_pred, y)

        return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

    def validate(self,dataloader: DataLoader) -> Tuple[float,float]:
        self.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.inference_mode():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                y_pred = self(X)
                loss = self.loss_fn(y_pred,y)
                val_loss += loss.item()
                val_acc += self.calculate_accuracy(y_pred, y)

            return val_loss / len(dataloader), val_acc / len(dataloader)

    def fit(self, train_dataloader: DataLoader, verbose: bool = False,
            val_dataloader: DataLoader = None, epochs: int = 10,
            val_every: int = 1, early_stopping: bool = False,
            patience: int = 3) -> Dict[str,List[float]]:

        if self.optimizer is None:
            raise ValueError("Call setup_training() first")

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        best_val_acc = -math.inf
        no_improve_count = 0

        for epoch in range(epochs):
            #Train:
            train_loss, train_acc = self.train_epoch(train_dataloader)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)

            #Validate
            val_loss, val_acc = (None, None)
            if val_dataloader is not None and epoch % val_every == 0:
                    val_loss, val_acc = self.validate(val_dataloader)
                    history["val_loss"].append(val_loss)
                    history["val_accuracy"].append(val_acc)

                    if early_stopping and val_acc is not None:
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                            if no_improve_count >= patience:
                                if verbose:
                                    print(f"Early stopping at epoch {epoch+1}: no improvement for {patience} epochs.")
                                break
            else:
                history["val_accuracy"].append(None)
                history["val_loss"].append(None)

            if verbose:
                log_str = f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                if val_loss is not None and val_acc is not None:
                    log_str += f" - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                print(log_str)
        return history


    def predict(self, dataloader: DataLoader) -> Tuple[torch.Tensor,torch.Tensor]:
        self.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X, y in dataloader:
                X,y = X.to(self.device), y.to(self.device)
                logits = self(X)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds)
                all_labels.append(y)

        return torch.cat(all_preds), torch.cat(all_labels)

    def evaluate(self, dataloader: DataLoader) -> Dict[str,float]:
        val_loss, val_acc = self.validate(dataloader)
        return {'loss': val_loss, 'accuracy': val_acc}

    def save_model(self,filepath: str):
        torch.save({
            'model_state_dict' : self.state_dict(),
            'model_config' : {
                'input_channels' : self.input_channels,
                'num_classes' : self.num_classes,
                'hidden_units' : self.hidden_units,
                'image_size' : self.image_size
                }
            }, filepath)


    @classmethod
    def load_model(cls, filepath: str, device: str = None) -> torch.nn.Module:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(filepath, map_location = device)
        config = checkpoint['model_config']

        model = cls(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model

