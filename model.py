import numpy as np
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import RGATConv
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import Linear
from text_embedding import get_embedding, get_saved_embedding
from data_spliting import training_set, validation_set, test_set 
from torch_geometric.data import Data
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def role_to_number(role): # PM , ME , UI , ID
    if(role == "PM"):
        return [1,0,0,0]
    if(role == "ME"):
        return [0,1,0,0]
    if(role == "UI"):
        return [0,0,1,0]
    if(role == "ID"):
        return [0,0,0,1]
    
def relation_to_number(relation):
      
    relation_dict = {
        'Acknowledgement': 0,
        'Alternation': 1,
        'Background': 2,
        'Clarification_question': 3,
        'Comment': 4,
        'Conditional': 5,
        'Continuation': 6,
        'Contrast': 7,
        'Correction': 8,
        'Elaboration': 9,
        'Explanation': 10,
        'Narration': 11,
        'Parallel': 12,
        'Q-Elab': 13,
        'Question-answer_pair': 14,
        'Result': 15
    }
    return relation_dict.get(relation, -1)

def data_for_graph(path, dataset, X, labeled : bool = True):
    # Adding Edge and type of edge
    roles, edges, edge_types, edge_attrs  = [], [], [], []
    shift = 0
    for transcription_id in dataset :
        # Adding 1-hot coding Roles
        with open(path / f"{transcription_id}.json", 'r') as file :
            transcription = json.load(file)
        for utterance in transcription :
            role = role_to_number(utterance["speaker"])
            roles.append(role)
        # Adding edges and their type
        with open(path / f"{transcription_id}.txt", 'r') as file :
            for line in file :
                source, edge_relation, target = line.strip().split()
                source = int(source) + shift
                target = int(target) + shift
                type_of_edge = relation_to_number(edge_relation)
                # backward edge 
                edges.append((target, source))
                edge_types.append(type_of_edge)

                edges.append((source, target))
                edge_types.append(type_of_edge + 16)
            shift += len(transcription)
        
    # Adding self loop
    if (shift != len(X)) :
        print('wtf')
    for k in range(len(X)) :
        edges.append((k,k))
        edge_types.append(32)
        
    roles = np.array(roles)
    X_data = np.column_stack([roles, X])
    X_data = np.column_stack([np.ones((X_data.shape[0], 1)), X_data])
    X_data = torch.tensor(X_data, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_types = torch.tensor(edge_types, dtype=torch.long)
    edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)
    if labeled :
        labels = []
        if labeled :
            with open("training_labels.json", "r") as file:
                training_labels = json.load(file)
            for transcription_id in dataset :
                labels += training_labels[transcription_id]
            labels = torch.tensor(labels, dtype=torch.long)
            return Data(x=X_data.to(device), edge_index=edge_index.to(device), edge_type = edge_types.to(device), y=labels.to(device),)
    return Data(x=X_data.to(device), edge_index=edge_index.to(device), edge_type = edge_types.to(device))

path_to_training = Path('training')
path_to_validation = Path('training')
path_to_test = Path('test')

X_train = get_embedding(path_to_training, training_set, method='mean_pooling', transformer = 'deberta')
X_valid =  get_embedding(path_to_validation, validation_set, method='mean_pooling', transformer = 'deberta')
# X_train, X_valid = get_saved_embedding("deberta")

# Adding the edges using data_for_graph function defined above
train_data = data_for_graph(path_to_training, training_set, X_train)
valid_data = data_for_graph(path_to_validation, validation_set, X_valid)
train_data = train_data.to(device)
valid_data = valid_data.to(device)


# Define the TextGCN model class
class TextGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_relations):
        super(TextGCN, self).__init__()
        # Linear layer for output
        self.lin2 = Linear(hidden_channels, 1)
        # First RGAT convolution layer
        self.conv1  = RGATConv(num_features, hidden_channels, num_relations=num_relations)
        # Second RGAT convolution layer
        self.conv2 = RGATConv(hidden_channels, hidden_channels, num_relations=num_relations)

    # Forward pass of the model
    def forward(self, data):
        # Unpack data
        x, edge_index, edge_type, edge_attr = data.x, data.edge_index, data.edge_type, data.edge_attr
        # Apply first RGAT convolution and ReLU activation
        x = F.relu(self.conv1(x, edge_index, edge_type, edge_attr=edge_attr))
        # Apply dropout
        x = F.dropout(x, training=self.training)
        # Apply second RGAT convolution and ReLU activation
        x = F.relu(self.conv2(x, edge_index, edge_type, edge_attr=edge_attr))
        # Apply final linear layer
        x = self.lin2(x)
        return x
        
# Define the GNN text classifier
class GNNTextClassifier:
    # Initialization of the classifier
    def __init__(self, num_features, num_channels, num_classes, num_relations, alpha, learning_rate):
        self.model = TextGCN(num_features, num_channels,  num_relations)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        # Define the loss function with class weighting
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - alpha) / alpha), reduction='mean')

    # Function to load a pre-trained model
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        
    # Training step
    def train(self, data):
        self.model.train()
        pred = self.model(data)
        loss = self.criterion(pred, data.y.float().unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # Prediction function
    def predict(self, data, threshold=0.5):
        self.model.eval()
        pred = self.model(data)
        y_pred = np.array((pred.cpu() > threshold).int()).flatten()
        return y_pred

    # Evaluation function to compute F1 score
    def evaluate(self, data, threshold=0.5): 
        self.model.eval()
        y_true = data.y.cpu()
        y_pred = self.predict(data, threshold)
        f1 = f1_score(y_true, y_pred)
        return f1
    
NUM_CLASSES = 2 
NUM_RELATION = 33 # 16 * 2 + 1

# Function to train the GNN model
def train_model(data, test, hidden_channels = 20, max_epoch = 100, learning_rate = 0.01, alpha = 0.2):
    f1_best, epoch_best = 0, 0  # Best F1 score and corresponding epoch
    f1_test, f1_train, losses = [], [], []  # Lists to store F1 scores and losses

    # Define dimensions and learning parameters
    NUM_FEATURES = data.x.shape[1]
    NUM_CHANNELS = hidden_channels
    ALPHA = alpha
    LR = learning_rate
    
    # Initialize the classifier and move to GPU if available
    classifier = GNNTextClassifier(NUM_FEATURES, NUM_CHANNELS, NUM_CLASSES, NUM_RELATION, ALPHA, LR)
    classifier.model.to(device)
    
    # Move the data to GPU
    data.to(device)
    test.to(device)
    
    # Training loop
    for epoch in tqdm(range(max_epoch), desc="Training"):
        loss = classifier.train(data)
        eval_f1 = classifier.evaluate(test) 
        train_f1 = classifier.evaluate(data) 
        
        # Record loss and F1 scores
        losses.append(loss)
        f1_test.append(eval_f1)
        f1_train.append(train_f1)
        
        # Update best model if current epoch gives better F1 score
        if eval_f1 > f1_best:
            f1_best, epoch_best = eval_f1, epoch
            torch.save(classifier.model.state_dict(), 'best_model.pth')

        # Print best F1 score every 25 epochs
        if epoch % 25 == 0:
            print(f1_best)
    
    # Load the best model
    classifier.load_model('best_model.pth')
    return f1_train, f1_test, losses, epoch_best, f1_best, classifier

# Training the model
f1_train, f1_test, losses, epoch_best, f1_best, classifier = train_model(train_data, valid_data, hidden_channels=30, max_epoch=400, learning_rate=0.01, alpha=0.18)
print('the best f1-score on the validation set', f1_best)

# Plotting the F1 scores and loss
plt.figure(figsize=(12, 5))

# F1 score plot
plt.subplot(1, 2, 1)
plt.plot(np.arange(len(f1_test)), f1_test, label='Validation Set')
plt.plot(np.arange(len(f1_train)), f1_train, label='Training Set')
plt.title(f'Learning Rate = {0.01}')
plt.scatter(epoch_best, f1_best, color='red', marker='x', label='Best Epoch')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(np.arange(len(losses)), losses, label='Training Loss', color='orange')
plt.title(f'Learning Rate = {0.01}')
plt.scatter(epoch_best, losses[epoch_best], color='red', marker='x', label='Best Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



## PREDICT FOR TESTING
test_labels = {}
embedding_test = {}

for transcription_id in test_set:
    with open(path_to_test / f"{transcription_id}.json", "r") as file:
        transcription = json.load(file)
    X_test = get_embedding(path_to_test, [transcription_id], transformer = 'deberta')
    embedding_test[transcription_id] = X_test
    data_test = data_for_graph(path_to_test, [transcription_id], X_test, labeled = False)
    y_pred = classifier.predict(data_test)
    test_labels[transcription_id] = y_pred.tolist()
torch.save(X_test , 'our_submission.pth')
with open("our_submission", "w") as file:
    json.dump(test_labels, file, indent=4)