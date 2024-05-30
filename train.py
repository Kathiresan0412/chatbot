import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

# Load the intents file
with open('intents.json', 'r') as f:
    intents_data = json.load(f)

# Initialize lists to hold all words, tags, and pattern-tag pairs
vocabulary = []
labels = []
pattern_tag_pairs = []

# Process each intent in the intents file
for intent in intents_data['intents']:
    tag = intent['tag']
    # Add the tag to the labels list
    labels.append(tag)
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        word_tokens = tokenize(pattern)
        # Add the tokens to the vocabulary list
        vocabulary.extend(word_tokens)
        # Add the pattern and tag as a pair to the pattern_tag_pairs list
        pattern_tag_pairs.append((word_tokens, tag))

# Define words to ignore
ignore_words = ['?', '.', '!']
# Stem and lowercase each word, ignoring certain symbols
vocabulary = [stem(word) for word in vocabulary if word not in ignore_words]
# Remove duplicates and sort the vocabulary and labels
vocabulary = sorted(set(vocabulary))
labels = sorted(set(labels))

print(len(pattern_tag_pairs), "patterns")
print(len(labels), "tags:", labels)
print(len(vocabulary), "unique stemmed words:", vocabulary)

# Prepare training data
X_train = []
y_train = []
for (pattern_sentence, tag) in pattern_tag_pairs:
    # Create a bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, vocabulary)
    X_train.append(bag)
    # Find the index of the tag and add it to y_train
    label_index = labels.index(tag)
    y_train.append(label_index)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyperparameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_layer_size = 8
output_size = len(labels)
print(input_size, output_size)

# Custom Dataset class
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Create dataset and data loader
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model, loss function, and optimizer
model = NeuralNet(input_size, hidden_layer_size, output_size).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for (batch_words, batch_labels) in train_loader:
        batch_words = batch_words.to(device)
        batch_labels = batch_labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        predictions = model(batch_words)
        # Compute the loss
        loss = loss_function(predictions, batch_labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Final loss: {loss.item():.4f}')

# Save the trained model
model_data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_layer_size": hidden_layer_size,
    "output_size": output_size,
    "vocabulary": vocabulary,
    "labels": labels
}

FILE_NAME = "data.pth"
torch.save(model_data, FILE_NAME)

print(f'Training complete. File saved to {FILE_NAME}')
