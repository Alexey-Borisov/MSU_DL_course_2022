import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
import torch

from tqdm.notebook import tqdm 
from collections import defaultdict
    

def train(model, train_dataloader, n_epochs, loss_function, 
          optimizer, device, save_path, save_name, val=False, val_dataloader=None, silent=False):
    
    model.to(device)
    
    history = defaultdict(list, { 'train_mean_loss': [], 'val_mean_loss': [], 'train_mean_score': [], 'val_mean_score': [], 'epoch_num': [],
                'train_epoch_time': []})
    
    for epoch_num in tqdm(range(n_epochs)):
        
        model.train()
        losses = []
        scores = []
        epoch_time = 0.0
        
        tmp_dict = defaultdict(list)
        for i, (images, labels) in enumerate(train_dataloader):
            
            images, labels = images.to(device), labels.to(device)
            
            model_begin = time.time()
            
            optimizer.zero_grad(set_to_none=True)
            model_outputs = model.forward(images)
            loss = loss_function(model_outputs, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            for i, p in enumerate(model.parameters()):
                if p.requires_grad:
                    if p.grad is not None:
                        tmp_dict[i].append(p.grad.flatten().detach().cpu())
    
            model_end = time.time()
            epoch_time += model_end - model_begin
            
            score = (model_outputs.argmax(axis=1) == labels).float().mean()
            losses.append(loss.detach().item() * images.shape[0])
            scores.append(score.detach().item() * images.shape[0])      
        
        for i in tmp_dict.keys():
            epoch_grads = torch.hstack(tmp_dict[i])
            history[i].append(epoch_grads)
            
        scores = np.array(scores)
        losses = np.array(losses)
        objects_count = train_dataloader.dataset.data.shape[0]
        history['train_mean_loss'].append(losses.sum() / objects_count)
        history['train_mean_score'].append(scores.sum() / objects_count)
        history['epoch_num'].append(epoch_num + 1)
        history['train_epoch_time'].append(epoch_time)
    
        if val is True and val_dataloader is not None:
            mean_loss, mean_score = evaluate(model, val_dataloader, loss_function, device)
            history['val_mean_loss'].append(mean_loss)
            history['val_mean_score'].append(mean_score)
        
        if silent is False:
            print(f"Epoch {epoch_num+1}:\ntrain loss - {history['train_mean_loss'][-1]:.3}, train score - {history['train_mean_score'][-1]:.3}\ntest loss - {mean_loss:.3}, test score - {mean_score:.3}")
        torch.save(model.state_dict(), save_path + save_name + '_model.torch')
        with open(save_path + save_name + '_history.dict', 'wb') as history_file:
            pickle.dump(history, history_file)
    return history

def evaluate(model, val_dataloader, loss_function, device):
    
    model.to(device)
    
    losses = []
    scores = []
    for i, (images, labels) in enumerate(val_dataloader):
        images, labels = images.to(device), labels.to(device)
        model_outputs = model.forward(images)
        loss = loss_function(model_outputs, labels)
        score = (model_outputs.argmax(axis=1) == labels).float().mean()
        losses.append(loss.detach().item() * images.shape[0])
        scores.append(score.detach().item() * images.shape[0])
    objects_count = val_dataloader.dataset.data.shape[0]
    mean_score = np.array(scores).sum() / objects_count
    mean_loss = np.array(losses).sum() / objects_count
    return mean_loss, mean_score


def get_info(filename):
    with open(filename, "rb") as file:
        history = pickle.load(file)
    layers_count = len(list(filter(lambda x: (type(x) is int and x % 4 == 0), history.keys()))) - 1
    grads = []
    for i in range(layers_count):
        grads.append(history[4 * i])
    return grads, history['train_mean_loss'][-1], history['val_mean_loss'][-1], history['train_mean_score'][-1], history['val_mean_score'][-1]

def init_conv_layers(model, init):
    for layer in model.nn_blocks:
        if hasattr(layer, 'conv'):
            
            if init == "zeros":
                torch.nn.init.zeros_(layer.conv.weight)
            elif init == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(layer.conv.weight)
            elif init == "kaiming_normal":
                torch.nn.init.kaiming_normal_(layer.conv.weight)
            elif init == "xavier_uniform":
                torch.nn.init.xavier_uniform_(layer.conv.weight)
            elif init == "xavier_normal":
                torch.nn.init.xavier_normal_(layer.conv.weight)
            else:
                raise AttributeError("Incorrect init type")