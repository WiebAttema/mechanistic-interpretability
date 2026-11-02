"""Linear and MLP probes on pooled activations, plus the metrics every script reports.

Probes keep their best epoch on validation UAR (unweighted average recall, i.e. macro
recall). On imbalanced AIBO an all-neutral epoch would win on accuracy but scores only
1/C on UAR.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPProbe(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


PROBES = {"linear": LinearProbe, "mlp": MLPProbe}


def zscore_stats(X):
    """Per-dimension mean and std, to standardize every split with train statistics."""
    return X.mean(dim=0), X.std(dim=0).clamp_min(1e-6)


def balanced_class_weights(y, num_classes):
    weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=y.numpy())
    return torch.tensor(weights, dtype=torch.float)


@torch.no_grad()
def predict(model, X, device):
    model.eval()
    return model(X.to(device)).argmax(dim=-1).cpu().numpy()


def uar(true, preds):
    return recall_score(true, preds, average="macro")


def train_probe(model, X_train, y_train, X_val, y_val, class_weights, device,
                epochs=300, lr=1e-3, batch_size=256):
    """AdamW on class-weighted cross-entropy. Returns the model at its best
    validation-UAR epoch, and that UAR."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loader = DataLoader(TensorDataset(X_train.to(device), y_train.to(device)),
                        batch_size=batch_size, shuffle=True)

    best_uar, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            criterion(model(X_batch), y_batch).backward()
            optimizer.step()
        val_uar = uar(y_val.numpy(), predict(model, X_val, device))
        if val_uar > best_uar:
            best_uar = val_uar
            best_state = {key: value.clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    return model, best_uar


def score_fields(class_names):
    """CSV columns written by test_scores."""
    return ["test_acc", "test_wf1", "test_uar"] + [f"test_recall_{name}" for name in class_names]


def test_scores(true, preds, class_names):
    """Accuracy, weighted F1, UAR and per-class recall on the test split."""
    recalls = recall_score(true, preds, average=None, labels=np.arange(len(class_names)), zero_division=0)
    scores = {"test_acc": accuracy_score(true, preds),
              "test_wf1": f1_score(true, preds, average="weighted"),
              "test_uar": uar(true, preds)}
    scores.update({f"test_recall_{name}": float(r) for name, r in zip(class_names, recalls)})
    return scores
