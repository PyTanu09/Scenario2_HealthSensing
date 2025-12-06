# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import warnings
warnings.filterwarnings('ignore')

class WindowDataset(Dataset):
    def __init__(self, df, label_map=None):
        self.df = df.reset_index(drop=True)
        # if no label_map provided, build one from this dataframe's labels
        if label_map is None:
            unique = list(self.df['label'].astype(str).unique())
            self.label_map = {lab: i for i, lab in enumerate(sorted(unique))}
        else:
            self.label_map = label_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Convert string representations to numpy arrays if needed
        try:
            if isinstance(row.nasal, str):
                nasal = np.fromstring(row.nasal.strip('[]'), sep=' ', dtype=np.float32)
            else:
                nasal = np.asarray(row.nasal, dtype=np.float32)
            
            if isinstance(row.thor, str):
                thor = np.fromstring(row.thor.strip('[]'), sep=' ', dtype=np.float32)
            else:
                thor = np.asarray(row.thor, dtype=np.float32)
        except (ValueError, AttributeError):
            # If parsing fails, create dummy arrays
            nasal = np.zeros(960, dtype=np.float32)
            thor = np.zeros(960, dtype=np.float32)
        
        x = np.stack([nasal, thor], axis=0)  # shape (2, timesteps)
        y = self.label_map.get(str(row.label), self.label_map.get(row.label, 0))
        return torch.from_numpy(x).float(), torch.tensor(int(y), dtype=torch.long)

# Simple 1D CNN
class Simple1DCNN(nn.Module):
    def __init__(self, in_ch=2, n_classes=3, seq_len=30*32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        # compute flattened size dynamically using seq_len
        reduced_len = seq_len // 4  # two poolings of 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * reduced_len, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x shape: (batch, channels, timesteps)
        return self.classifier(self.conv(x))

# Conv -> LSTM model (Conv for feature extraction, LSTM for temporal modeling)
class ConvLSTM(nn.Module):
    def __init__(self, in_ch=2, n_classes=3, seq_len=30*32, conv_channels=64, lstm_hidden=128, n_layers=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, conv_channels, kernel_size=5, padding=2), nn.ReLU(),
        )
        # after conv, create sequence along time with features = conv_channels
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=lstm_hidden, num_layers=n_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: (batch, ch, timesteps)
        conv_out = self.conv(x)  # (batch, conv_channels, timesteps)
        conv_out = conv_out.permute(0, 2, 1)  # (batch, timesteps, features)
        lstm_out, _ = self.lstm(conv_out)  # (batch, timesteps, hidden)
        # use last timestep
        last = lstm_out[:, -1, :]
        return self.fc(last)

def run_lopo(csv_path, model_type='cnn', epochs=10, batch_size=64, device=None):
    df = pd.read_csv(csv_path)
    
    if 'participant' not in df.columns:
        raise ValueError("CSV must include 'participant' column")
    
    # ensure label and participant are strings for stable mapping
    df['participant'] = df['participant'].astype(str)
    df['label'] = df['label'].astype(str)
    
    # global label mapping so all folds use same integer mapping
    unique_labels = sorted(df['label'].unique())
    label_map = {lab: i for i, lab in enumerate(unique_labels)}
    n_classes = len(label_map)
    
    logo = LeaveOneGroupOut()
    participants = df['participant'].values
    metrics_per_fold = []
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    print(f"Running LOPO with {n_classes} classes: {unique_labels}")
    print(f"Using device: {device}")
    
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(df, groups=participants)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        test_participant = test_df['participant'].iloc[0]
        
        train_ds = WindowDataset(train_df, label_map=label_map)
        test_ds = WindowDataset(test_df, label_map=label_map)
        tr_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        te_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # determine seq_len from sample
        sample = train_ds[0][0]  # (2, timesteps)
        seq_len = sample.shape[1]
        
        if model_type == 'cnn':
            model = Simple1DCNN(seq_len=seq_len, n_classes=n_classes)
        elif model_type == 'convlstm':
            model = ConvLSTM(seq_len=seq_len, n_classes=n_classes)
        else:
            raise NotImplementedError("model_type must be 'cnn' or 'convlstm'")

        model = model.to(device)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        for ep in range(epochs):
            model.train()
            epoch_loss = 0.0
            for x, y in tr_dl:
                x = x.float().to(device)
                y = y.to(device)
                out = model(x)
                loss = loss_fn(out, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()

        # evaluation
        y_true = []
        y_pred = []
        model.eval()
        with torch.no_grad():
            for x, y in te_dl:
                x = x.float().to(device)
                out = model(x)
                preds = torch.argmax(out, dim=1).cpu().numpy()
                y_pred.extend(preds.tolist())
                y_true.extend(y.numpy().tolist())
        
        acc = accuracy_score(y_true, y_pred)
        labels = list(range(n_classes))
        prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        metrics_per_fold.append({
            'participant_test': test_participant,
            'acc': acc,
            'precision': prec,
            'recall': rec,
            'confusion': cm
        })
        print(f"Fold {fold_idx+1} (test participant: {test_participant}): acc={acc:.4f}")
    
    # aggregate metrics
    precisions = np.stack([m['precision'] for m in metrics_per_fold])
    recalls = np.stack([m['recall'] for m in metrics_per_fold])
    print(f"\nPrecision mean per class: {precisions.mean(axis=0)}, std: {precisions.std(axis=0)}")
    print(f"Recall mean per class: {recalls.mean(axis=0)}, std: {recalls.std(axis=0)}")
    
    # save metrics
    import json
    def nd_to_list(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    
    with open(f'metrics_lopo_{model_type}.json', 'w') as f:
        json.dump(metrics_per_fold, f, default=nd_to_list)
    print(f"Saved metrics to metrics_lopo_{model_type}.json\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', required=True, help='path to breathing_dataset.csv')
    parser.add_argument('--model', choices=['cnn', 'convlstm'], default='cnn')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()
    
    run_lopo(args.dataset, model_type=args.model, epochs=args.epochs, batch_size=args.batch)
