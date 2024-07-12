import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.data import Field, TabularDataset, BucketIterator
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 数据预处理
def preprocess_data():
    # 定义Field
    text_field = Field(tokenize='spacy', tokenizer_language='en_core_web_sm', lower=True, include_lengths=True)
    label_field = Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)

    # 加载数据
    fields = [('text', text_field), ('label', label_field)]
    train_data, valid_data, test_data = TabularDataset.splits(
        path='./data',
        train='train.csv',
        validation='valid.csv',
        test='test.csv',
        format='csv',
        fields=fields,
        skip_header=True
    )

    # 构建词汇表
    text_field.build_vocab(train_data, max_size=10000, min_freq=2)

    # 创建迭代器
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=64,
        sort_within_batch=True,
        sort_key=lambda x: len(x.text),
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )

    return train_iterator, valid_iterator, test_iterator, text_field, label_field

# Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, emb_dim, n_heads, n_layers, pf_dim, dropout, max_length=100):
        super().__init__()
        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        self.layers = nn.ModuleList([TransformerLayer(emb_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.fc_out = nn.Linear(emb_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([emb_dim])).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, src, src_len):
        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(src.device)
        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)
        embedded = self.dropout((tok_embedded + pos_embedded) * self.scale)

        for layer in self.layers:
            embedded = layer(embedded)

        output = self.fc_out(embedded[:, 0, :])
        return output

class TransformerLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        _src, _ = self.self_attention(src, src, src)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x

# 训练和评估
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        optimizer.zero_grad()
        src, src_len = batch.text
        predictions = model(src, src_len).squeeze(1)
        loss = criterion(predictions, batch.label.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in iterator:
            src, src_len = batch.text
            predictions = model(src, src_len).squeeze(1)
            loss = criterion(predictions, batch.label.float())
            epoch_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch.label.cpu().numpy())
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_preds)
    return epoch_loss / len(iterator), mse, rmse, mae

# 主函数
def preprocess_data_main():
    train_iterator, valid_iterator, test_iterator, text_field, label_field = preprocess_data()
    INPUT_DIM = len(text_field.vocab)
    EMB_DIM = 256
    N_HEADS = 8
    N_LAYERS = 3
    PF_DIM = 512
    DROPOUT = 0.1
    model = TransformerModel(INPUT_DIM, EMB_DIM, N_HEADS, N_LAYERS, PF_DIM, DROPOUT)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    N_EPOCHS = 10
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_mse, valid_rmse, valid_mae = evaluate(model, valid_iterator, criterion)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best-model.pt')
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.3f}, Valid Loss: {valid_loss:.3f}, Valid MSE: {valid_mse:.3f}, Valid RMSE: {valid_rmse:.3f}, Valid MAE: {valid_mae:.3f}')

    model.load_state_dict(torch.load('best-model.pt'))
    test_loss, test_mse, test_rmse, test_mae = evaluate(model, test_iterator, criterion)
    print(f'Test Loss: {test_loss:.3f}, Test MSE: {test_mse:.3f}, Test RMSE: {test_rmse:.3f}, Test MAE: {test_mae:.3f}')

if __name__ == '__main__':
    preprocess_data_main()