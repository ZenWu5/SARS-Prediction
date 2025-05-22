import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
import argparse
import logging
import os
import time
import math
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='preprocess/ESM/output/sampled_output_esm_embeddings.h5')
parser.add_argument('--max_seq_len', type=int, default=None)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--num_heads', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--k_folds', type=int, default=5)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--output_dir', type=str, default='ModelResults/TransformerResults_h5')
args = parser.parse_args()

# 配置plt字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
logger = logging.getLogger("transformer_training")

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# Transformer 回归模型
class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        if mask is not None:
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x), None

# 加载数据
class H5ProteinDataset(Dataset):
    def __init__(self, h5_path, max_seq_len=None, normalize=True):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, 'r') as f:
            self.total = int(f.attrs['total_samples'])
            self.emb_dim = int(f.attrs['embedding_dim'])
            self.targets = f['mean_log10Ka'][:].astype(np.float32)
            self.seq_lengths = np.array([f['embeddings'][f'emb_{i}'].shape[0] for i in range(self.total)])
        self.max_seq_len = max_seq_len or int(self.seq_lengths.max())
        self._file = None
        self.normalize = normalize
        self.feature_mean = None
        self.feature_std = None
        if normalize:
            self._compute_norm()

    def _compute_norm(self):
        indices = np.random.choice(self.total, min(1000, self.total), replace=False)
        feats = []
        with h5py.File(self.h5_path, 'r') as f:
            for i in indices:
                feats.append(f['embeddings'][f'emb_{i}'][:].astype(np.float32))
        feats = np.vstack(feats)
        self.feature_mean = feats.mean(axis=0)
        self.feature_std = feats.std(axis=0) + 1e-8

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
        emb = self._file['embeddings'][f'emb_{idx}'][:].astype(np.float32)
        if self.normalize:
            emb = (emb - self.feature_mean) / self.feature_std
        L = emb.shape[0]
        if L > self.max_seq_len:
            proc = emb[:self.max_seq_len]
            mask = np.ones(self.max_seq_len, dtype=bool)
        else:
            pad = self.max_seq_len - L
            proc = np.vstack([emb, np.zeros((pad, self.emb_dim), dtype=np.float32)])
            mask = np.concatenate([np.ones(L, dtype=bool), np.zeros(pad, dtype=bool)])
        return (
            torch.from_numpy(proc),
            torch.from_numpy(mask),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

# 模型训练函数
def train_model(train_loader, val_loader, model, optimizer, scheduler, criterion, params):
    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(params['epochs']):
        model.train()
        losses = []
        for x, mask, y in train_loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            pred, _ = model(x, mask)
            loss = criterion(pred.squeeze(), y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, mask, y in val_loader:
                x, mask = x.to(device), mask.to(device)
                pred, _ = model(x, mask)
                val_preds.append(pred.squeeze().cpu().numpy())
                val_targets.append(y.numpy())
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_loss = mean_squared_error(val_targets, val_preds)
        history.append(val_loss)
        scheduler.step()
        logger.info(f"Epoch {epoch+1}, TrainLoss={np.mean(losses):.4f}, ValMSE={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= params['patience']:
            break
    model.load_state_dict(best_state)
    return model, best_loss, history

# 主流程
def main():
    set_seed(args.seed)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(args.output_dir, timestamp)
    os.makedirs(outdir, exist_ok=True)
    dataset = H5ProteinDataset(args.data_path, args.max_seq_len, normalize=True)
    indices = np.random.permutation(len(dataset))
    n_test = int(0.15 * len(dataset))
    test_set = Subset(dataset, indices[:n_test])
    trainval_set = Subset(dataset, indices[n_test:])
    model_params = {
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'dropout': args.dropout
    }
    training_params = {
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience
    }

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    best_model = None
    best_score = float('inf')
    all_hist = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(trainval_set), 1):
        logger.info(f"Fold {fold}/{args.k_folds}")
        train_loader = DataLoader(Subset(trainval_set, train_idx), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(trainval_set, val_idx), batch_size=args.batch_size)
        input_dim = dataset.emb_dim
        model = TransformerRegressor(input_dim, **model_params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        def cosine_schedule(epoch):
            warmup = 5
            if epoch < warmup:
                return (epoch + 1) / warmup
            factor = (epoch - warmup) / (args.epochs - warmup)
            return 1e-3 + (1 - 1e-3) * 0.5 * (1 + math.cos(math.pi * factor))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_schedule)
        criterion = nn.MSELoss()
        model, val_loss, hist = train_model(train_loader, val_loader, model, optimizer, scheduler, criterion, training_params)
        if val_loss < best_score:
            best_score = val_loss
            best_model = model
        all_hist.append(hist)

    # 保存模型
    model_path = os.path.join(outdir, 'best_transformer.pt')
    torch.save(best_model.state_dict(), model_path)
    logger.info(f"最佳模型保存至: {model_path}")

    # 评估测试集
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    preds, targets = [], []
    best_model.eval()
    with torch.no_grad():
        for x, mask, y in test_loader:
            x, mask = x.to(device), mask.to(device)
            pred, _ = best_model(x, mask)
            preds.append(pred.squeeze().cpu().numpy())
            targets.append(y.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    pcc, _ = pearsonr(targets, preds)
    logger.info(f"测试集结果: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, Pearson={pcc:.4f}")

if __name__ == "__main__":
    main()
