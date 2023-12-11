"""
    This code does the embedding,
    you have the choice to eather :
    - Bert model (Version all-MiniLM-L6-v2) (1*384)
    - Large DeBERTa Model with mean (1*1024)
    """
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer
from transformers import DebertaTokenizer, DebertaModel
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')


def my_embedding(path, dataset, transformer = 'bert', method='mean_pooling') -> np.array:
    """Using Deberta Model for sentence embedding"""
    X, embedding_length = [], []  # Will contain  the embedding of each uttera
    progress, size = 0, len(dataset)

    if transformer == 'bert' :
        # Load the BERT model
        bert = SentenceTransformer('all-MiniLM-L6-v2')
        # Move the model to the GPU
        bert =  bert.to(device)
        
        for transcription_id in dataset:
            progress += 1
            print(f"progress : {(progress * 100) / size} %")
            with open(path / f"{transcription_id}.json", "r") as file:
                transcription = json.load(file)
            for utterance in transcription:
                text = utterance["text"]
                X.append(bert.encode(text, show_progress_bar=False))
                
    else :  ## transformer == 'deberta' 
        
        # Load the DeBERTa tokenizer and model
        model_name = 'microsoft/deberta-large'
        tokenizer = DebertaTokenizer.from_pretrained(model_name)
        model = DebertaModel.from_pretrained(model_name)
        # Move the model to the GPU
        model = model.to(device)
        
        for transcription_id in dataset:
            progress += 1
            print(f"progress : {(progress * 100) / size} %")
            with open(path / f"{transcription_id}.json", "r") as file:
                transcription = json.load(file)
            for utterance in transcription:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

                # Move the inputs to the GPU
                inputs = {key: value.to(device) for key, value in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

                last_hidden_states = outputs.last_hidden_state[0, : , :]
                
                if method == 'concat' :
                    # Word embedding
                    embedding_length.append(last_hidden_states.shape[0])
                    for k in range(last_hidden_states.shape[0]) :
                        word_embedding = last_hidden_states[k, :]
                        X.append(word_embedding.cpu().numpy())
                        
                # Sentence Embedding       
                elif method == 'max_pooling':
                    max_pooled = torch.max(last_hidden_states, dim=0).values
                    X.append(max_pooled.cpu().numpy())  
                elif method == 'mean_pooling':
                    mean_pooled = torch.mean(last_hidden_states, dim=0)
                    X.append(mean_pooled.cpu().numpy())  
                else:  # method = 'min_pooling'
                    min_pooled = torch.min(last_hidden_states, dim=0).values
                    X.append(min_pooled.cpu().numpy())  
   
    return np.array(X), embedding_length

def get_embedding(path, dataset, transformer = 'bert', method = 'mean_pooling') :
    # We already did the embedding and save it
    
    X, _ = my_embedding(path, dataset, transformer, method)
    
    return X

def get_saved_embedding( transformer = "bert") :
    if transformer == "bert" :
        embedding_data = np.load('train_valid_embedding_bert.npz')
        X_train = embedding_data['array1']
        X_valid = embedding_data['array2']
    else : 
        embedding_data = np.load('train_valid_embedding_deberta.npz')
        X_train = embedding_data['array1']
        X_valid = embedding_data['array2']
    return X_train, X_valid