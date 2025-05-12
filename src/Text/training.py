import data_processing as dp
from dataset import TextDataset
from torch.utils.data import DataLoader
import torch
from transformers import BertForSequenceClassification
from torch.optim import AdamW

df = dp.dataset_path

train_df, test_df = dp.load_data(df)

train_enc, test_enc = dp.tokenize_data(train_df, test_df)

train_dataset = TextDataset(train_enc, train_df['label'].tolist())
test_dataset = TextDataset(test_enc, test_df['label'].tolist())

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
model.to(device)


optimizer = AdamW(model.parameters(), lr=2e-5)

epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} complete! Avg loss: {avg_loss:.4f}")

model.save_pretrained("saved_model/")
tokenizer = dp.get_tokenizer()
tokenizer.save_pretrained("saved_model/")

