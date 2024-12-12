import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def load_data(batch_size, features, labels, is_train=True):
    features = torch.tensor(features.values.tolist())
    labels = torch.tensor(labels.values.tolist())
    features = (features - features.mean(0)) / features.std(0)
    dataset = TensorDataset(features, labels)
    data_iter = DataLoader(dataset, batch_size, shuffle=is_train)
    if is_train:    
        W, b = torch.rand((6, 1), requires_grad=True), torch.rand(1, requires_grad=True)
        return data_iter, W, b
    else:
        return features, labels
    
class HingeLoss(nn.Module):
    def __init__(self, margin=1.0, lambda_param=1e-3):
        super().__init__()
        self.margin = margin
        self.lambda_param = lambda_param

    def forward(self, outputs, labels):
        batch_size, num_classes = outputs.size()
        correct_class_scores = outputs[torch.arange(batch_size), labels].unsqueeze(1)
        margins = torch.clamp(outputs - correct_class_scores + self.margin, 0)
        margins[range(batch_size), labels] = 0
        hinge_loss = margins.sum() / batch_size
        return hinge_loss
        
def predict(net, X):
    with torch.no_grad():
        outputs = net(X)
        _, predicted = torch.max(outputs, 1)
    return predicted

def compute_accuracy(net, X, y):
    y_hat = predict(net, X)
    return (y_hat == y).float().mean().item()

def train_svm(num_epochs, net, lr, weight_decay, train_iter, val_iter): 
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss()
    train_loss, train_acc, val_loss, val_acc = [], [], [], []
    with tqdm(total=num_epochs) as pbar:
        for _ in range(num_epochs):
            train_l, train_a = 0, 0
            for X, y in train_iter:
                optimizer.zero_grad()
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                train_l += l.item()
                train_a += compute_accuracy(net, X, y)
            train_loss.append(train_l / len(train_iter))
            train_acc.append(train_a / len(train_iter))
            val_l, val_a = 0, 0
            with torch.no_grad():
                for X, y in val_iter:
                    y_hat = net(X)
                    l = loss(y_hat, y)
                    val_l += l.item()
                    val_a += compute_accuracy(net, X, y)
                val_loss.append(val_l / len(val_iter))
                val_acc.append(val_a / len(val_iter))
            pbar.set_description(f"train_acc: {train_acc[-1]:.4f}, val_acc: {val_acc[-1]:.4f}")
            pbar.update()
    return train_loss, val_loss, train_acc, val_acc, net