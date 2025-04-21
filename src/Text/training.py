import data_processing as dp
from dataset import TextDataset
from torch.utils.data import DataLoader

df = dp.dataset_path

train_df, test_df = dp.load_data(df)

train_enc, test_enc = dp.tokenize_data(train_df, test_df)

train_dataset = TextDataset(train_enc, train_df['label'].tolist())
test_dataset = TextDataset(test_enc, test_df['label'].tolist())

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)
