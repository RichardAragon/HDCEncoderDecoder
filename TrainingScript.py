import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

class SimpleDataset(Dataset):
    """A simple dataset class for token sequences."""
    def __init__(self, tokenizer, texts):
        self.tokenizer = tokenizer
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx], return_tensors="pt").input_ids
        return tokens.squeeze()

# Define the dataset
tokenizer = LlamaTokenizer.from_pretrained('huggingface/llama-1b')
texts = ["Hello world!", "This is a test.", "How are you?", "Good morning!"]
dataset = SimpleDataset(tokenizer, texts)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Instantiate the model
model = HDC_LLaMA(vocab_size=tokenizer.vocab_size)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
def train_model(model, dataloader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for tokens in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(tokens)
            loss = criterion(outputs, tokens)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader)}')

train_model(model, dataloader, optimizer, criterion)
