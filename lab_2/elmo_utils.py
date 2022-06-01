import torch
import numpy as np
from torch.nn.functional import softmax
from data_utils import tokenize
from tqdm.notebook import tqdm

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


class ELMo(torch.nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab, pretrained_embeddings, tuning, num_layers, dropout):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab = vocab
        self.output_dim = len(self.vocab)
        self.pretrained_embeddings = pretrained_embeddings
        self.tuning = tuning
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.noncontext_embeddings = torch.nn.Embedding(num_embeddings=len(self.vocab), embedding_dim=self.embedding_dim, padding_idx=0)
        if self.tuning is False:
            self.noncontext_embeddings.weight.requires_grad = False
        if self.pretrained_embeddings is not None:
            with torch.no_grad():
                for token_idx, token in enumerate(self.vocab.get_itos()):
                    if token in self.pretrained_embeddings:
                        self.noncontext_embeddings.weight[token_idx] = torch.FloatTensor(self.pretrained_embeddings.get_vector(token).copy())
                    else:
                        self.noncontext_embeddings.weight[token_idx] = 0.0
        self.rnn_forward = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                      batch_first=False, dropout=self.dropout, bidirectional=False)
        self.rnn_backward = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                      batch_first=False, dropout=self.dropout, bidirectional=False)
        
        self.output_forward = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim)
        self.output_backward = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.output_dim)
        
        
    def forward(self, tokens, reversed_tokens):
        
        # forward Language Model
        embs = self.noncontext_embeddings(tokens)
        output_forward, (h_state, C_state) = self.rnn_forward.forward(embs)
        result_forward = self.output_forward.forward(output_forward)
        
        # backward Language Model
        reversed_embs = self.noncontext_embeddings(reversed_tokens)
        output_backward, (h_state, C_state) = self.rnn_backward.forward(reversed_embs)
        result_backward = self.output_backward.forward(output_backward)
        
        return result_forward, result_backward

    
def get_embeddings(model, tokens, reversed_tokens, tokens_lens):
        
    # forward Language Model
    noncontext_embs = model.noncontext_embeddings(tokens)
    embs_forward, (h_state, C_state) = model.rnn_forward.forward(noncontext_embs)
    
    # backward Language Model
    noncontext_reversed_embs = model.noncontext_embeddings(reversed_tokens)
    embs_backward, (h_state, C_state) = model.rnn_backward.forward(noncontext_reversed_embs)
    
    for text_idx in range(tokens.shape[1]):
        cur_len = tokens_lens[text_idx] + 2
        embs_backward[0:cur_len, text_idx, :] = torch.flip(embs_backward[0:cur_len, text_idx, :], dims=[0])
    
    context_embs = torch.concat((embs_forward, embs_backward), dim=-1)
    return context_embs


def raw_text_embeddings(model, text, vocab, device):
    tokenized_text = tokenize(text)
    tokens_lens = torch.tensor(len(tokenized_text))
    tokens = torch.tensor(vocab.lookup_indices(["<sos>"] + tokenized_text + ["<eos>"]), dtype=int)
    reversed_tokens = torch.flip(tokens, dims=[0])
    
    tokens, reversed_tokens, tokens_lens = tokens.reshape(-1, 1), reversed_tokens.reshape(-1, 1), tokens_lens.reshape(1)
    tokens, reversed_tokens = tokens.to(device), reversed_tokens.to(device)
    embeddings = get_embeddings(model, tokens, reversed_tokens, tokens_lens)[:, 0, :]
    return embeddings

def get_nearest_texts(model, chosen_embedding, all_embeddings, count=10):
    
    dists = torch.Tensor(size=(0,))
    texts = torch.Tensor(size=(0,))
    positions = torch.Tensor(size=(0,))
    for text_idx in tqdm(range(len(all_embeddings))):
        cur_embeddings = all_embeddings[text_idx]
        cur_dists = ((cur_embeddings - chosen_embedding)**2).sum(dim=1)
        dists = torch.hstack((dists, cur_dists))
        texts = torch.hstack((texts, torch.full(size=(cur_embeddings.shape[0],), fill_value=text_idx)))
        positions = torch.hstack((positions, torch.arange(cur_embeddings.shape[0])))
    args = torch.argsort(dists)
    return texts[args[:count]].long(), positions[args[:count]].long()


def token_info(token, vocab, dataset, all_embeddings):
    token_idx = vocab.lookup_indices([token])[0]
    texts = torch.Tensor(size=(0,))
    positions = torch.Tensor(size=(0,))
    all_token_embeddings = torch.Tensor(size=(0,all_embeddings[0].shape[1]))
    for text_idx in tqdm(range(len(dataset))):
        cur_tokens = dataset[text_idx]['tokens']
        if token_idx in cur_tokens:
            cur_positions = torch.argwhere(cur_tokens == token_idx).ravel()
            cur_embeddings = all_embeddings[text_idx][cur_positions].cpu()
            all_token_embeddings = torch.concat((all_token_embeddings, cur_embeddings))
            texts = torch.hstack((texts, torch.full(size=(cur_positions.shape[0],), fill_value=text_idx)))
            positions = torch.hstack((positions, cur_positions))
    return texts.long(), positions.long(), all_token_embeddings

def show_token_info(all_token_embeddings, n_clusters=2, title='', ax=None):
    kmeans = KMeans(n_clusters=2)
    y_pred = kmeans.fit_predict(all_token_embeddings.detach().numpy())
    tsne = TSNE()
    res = tsne.fit_transform(all_token_embeddings.detach().numpy())
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
    sns.set_style('darkgrid')
    ax.set_title(title, pad=15, fontsize=20)
    sns.scatterplot(x=res[:, 0], y=res[:, 1], hue=y_pred, palette=sns.color_palette('tab10')[:2], ax=ax)
    return y_pred

class LMCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, outputs, tokens, tokens_lens):
        """
        :param torch.tensor outputs: Output from RNNLM.forward. Shape: [T, B, V]
        :param torch.tensor tokens: Batch of tokens. Shape: [T, B]
        :param torch.tensor tokens_lens: Length of each sequence in batch
        :return torch.tensor: CrossEntropyLoss between corresponding logits and tokens
        """
        # Use torch.nn.utils.rnn.pack_padded_sequence().data to remove padding and flatten logits and tokens
        # Do not forget specify enforce_sorted=False and correct value of batch_first 
        packed_outputs = torch.nn.utils.rnn.pack_padded_sequence(outputs, tokens_lens+1, batch_first=False, enforce_sorted=False)[0]
        packed_tokens = torch.nn.utils.rnn.pack_padded_sequence(tokens[1:], tokens_lens+1, batch_first=False, enforce_sorted=False)[0]
        
        # Use super().forward(..., ...) to compute CrossEntropyLoss
        return super().forward(packed_outputs, packed_tokens)
    

class LMAccuracy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, outputs, tokens, tokens_lens):
        """
        :param torch.tensor outputs: Output from RNNLM.forward. Shape: [T, B, V]
        :param torch.tensor tokens: Batch of tokens. Shape: [T, B]
        :param torch.tensor tokens_lens: Length of each sequence in batch
        :return torch.tensor: Accuracy for given logits and tokens
        """
        # Use torch.nn.utils.rnn.pack_padded_sequence().data to remove padding and flatten logits and tokens
        # Do not forget specify enforce_sorted=False and correct value of batch_first 
        packed_outputs = torch.nn.utils.rnn.pack_padded_sequence(outputs, tokens_lens+1, batch_first=False, enforce_sorted=False)[0]
        packed_tokens = torch.nn.utils.rnn.pack_padded_sequence(tokens[1:], tokens_lens+1, batch_first=False, enforce_sorted=False)[0]
        
        predicted_tokens = packed_outputs.argmax(axis=-1)
        return (predicted_tokens == packed_tokens).float().mean()

    
def train_epoch_lm(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for idx, data in enumerate(dataloader):
        # 1. Take data from batch
        # 2. Perform forward pass
        # 3. Evaluate loss
        # 4. Make optimizer step
        
        tokens, reversed_tokens, tokens_lens = data['tokens'], data['reversed_tokens'], data['tokens_lens']
        tokens, reversed_tokens = tokens.to(device), reversed_tokens.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs_forward, outputs_backward = model.forward(tokens, reversed_tokens)
        loss_forward = loss_fn(outputs_forward, tokens, tokens_lens)
        loss_backward = loss_fn(outputs_backward, reversed_tokens, tokens_lens)
        loss = loss_forward + loss_backward
        loss.backward()
        optimizer.step()
        
def evaluate_lm(dataloader, model, loss_fn, device):
    model.eval()
    
    total_tokens = 0
    total_loss_forward = 0.0
    total_loss_backward = 0.0
    total_accuracy_forward = 0.0
    total_accuracy_backward = 0.0
    
    accuracy_fn = LMAccuracy()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            # 1. Take data from batch
            # 2. Perform forward pass
            # 3. Evaluate loss
            # 4. Evaluate accuracy
            
            tokens, reversed_tokens, tokens_lens = data['tokens'],  data['reversed_tokens'], data['tokens_lens']
            tokens, reversed_tokens = tokens.to(device), reversed_tokens.to(device)
            outputs_forward, outputs_backward = model.forward(tokens, reversed_tokens)
            
            total_loss_forward += loss_fn(outputs_forward, tokens, tokens_lens) * (tokens_lens+1).sum()
            total_loss_backward += loss_fn(outputs_backward, reversed_tokens, tokens_lens) * (tokens_lens+1).sum()
            
            total_accuracy_forward += accuracy_fn(outputs_forward, tokens, tokens_lens) * (tokens_lens+1).sum()
            total_accuracy_backward += accuracy_fn(outputs_backward, reversed_tokens, tokens_lens) * (tokens_lens+1).sum()
            
            total_tokens += (tokens_lens+1).sum()
            
    return total_loss_forward / total_tokens, total_loss_backward / total_tokens, total_accuracy_forward / total_tokens, total_accuracy_backward / total_tokens

def train_lm(train_loader, val_loader, model, loss_fn, optimizer, device, num_epochs):
    
    # init history
    history = {
        'val_losses_forward': [],
        'val_losses_backward': [],
        'train_losses_forward': [],
        'train_losses_backward': [],
        'val_accuracies_forward': [],
        'val_accuracies_backward': [],
        'train_accuracies_forward': [],
        'train_accuracies_backward': []
    }
    
    for epoch in range(num_epochs):
        
        # train one epoch
        train_epoch_lm(train_loader, model, loss_fn, optimizer, device)
        
        # collect losses and accuracies for train data
        train_loss_forward, train_loss_backward, train_acc_forward, train_acc_backward = evaluate_lm(train_loader, model, loss_fn, device)
        history['train_losses_forward'].append(train_loss_forward)
        history['train_losses_backward'].append(train_loss_backward)
        history['train_accuracies_forward'].append(train_acc_forward)
        history['train_accuracies_backward'].append(train_acc_backward)
        
        # collect losses and accuracies for val data
        val_loss_forward, val_loss_backward, val_acc_forward, val_acc_backward = evaluate_lm(val_loader, model, loss_fn, device)
        history['val_losses_forward'].append(val_loss_forward)
        history['val_losses_backward'].append(val_loss_backward)
        history['val_accuracies_forward'].append(val_acc_forward)
        history['val_accuracies_backward'].append(val_acc_backward)
        
        # print results for current epoch (forward and backward)
        print(
            'FORWARD: Epoch: {0:d}/{1:d}. Loss (Train/Val): {2:.3f}/{3:.3f}. Accuracy (Train/Val): {4:.3f}/{5:.3f}'.format(
                epoch + 1, num_epochs, history['train_losses_forward'][-1], history['val_losses_forward'][-1], 
                history['train_accuracies_forward'][-1], history['val_accuracies_forward'][-1]
            )
        )
        print(
            'BACKWARD: Epoch: {0:d}/{1:d}. Loss (Train/Val): {2:.3f}/{3:.3f}. Accuracy (Train/Val): {4:.3f}/{5:.3f}'.format(
                epoch + 1, num_epochs, history['train_losses_backward'][-1], history['val_losses_backward'][-1], 
                history['train_accuracies_backward'][-1], history['val_accuracies_backward'][-1]
            )
        )
    return history


def decode(model, start_tokens, start_tokens_lens, max_generated_len=20, top_k=None, mode='forward'):
    """
    :param RNNLM model: Model
    :param torch.tensor start_tokens: Batch of seed tokens. Shape: [T, B]
    :param torch.tensor start_tokens_lens: Length of each sequence in batch. Shape: [B]
    :return Tuple[torch.tensor, torch.tensor]. Newly predicted tokens and length of generated part. Shape [T*, B], [B]
    """
    # Get embedding for start_tokens
    # YOUR CODE HERE
    embedding = model.noncontext_embeddings(start_tokens)
 
    # Pass embedding through rnn and collect hidden states and cell states for each time moment
    all_h, all_c = [], []
    h_cur, c_cur = None, None
    if mode == 'forward':
        h = embedding.new_zeros([model.rnn_forward.num_layers, start_tokens.shape[1], model.hidden_dim])
        c = embedding.new_zeros([model.rnn_forward.num_layers, start_tokens.shape[1], model.hidden_dim])
    elif mode == 'backward':
        h = embedding.new_zeros([model.rnn_backward.num_layers, start_tokens.shape[1], model.hidden_dim])
        c = embedding.new_zeros([model.rnn_backward.num_layers, start_tokens.shape[1], model.hidden_dim])
    for time_step in range(start_tokens.shape[0]):
        
        if mode == 'forward':
            if h_cur is None:
                output, (h_cur, c_cur) = model.rnn_forward.forward(embedding[time_step].unsqueeze(0))
            else:
                output, (h_cur, c_cur) = model.rnn_forward.forward(embedding[time_step].unsqueeze(0), (h_cur, c_cur))
        
        elif mode == 'backward':
            if h_cur is None:
                output, (h_cur, c_cur) = model.rnn_backward.forward(embedding[time_step].unsqueeze(0))
            else:
                output, (h_cur, c_cur) = model.rnn_backward.forward(embedding[time_step].unsqueeze(0), (h_cur, c_cur))
                
        all_h.append(h_cur)
        all_c.append(c_cur)

    all_h = torch.stack(all_h, dim=1)
    all_c = torch.stack(all_c, dim=1)
    # Take final hidden state and cell state for each start sequence in batch
    # We will use them as h_0, c_0 for generation new tokens
    h = all_h[:, start_tokens_lens - 1, torch.arange(start_tokens_lens.shape[0])]
    c = all_c[:, start_tokens_lens - 1, torch.arange(start_tokens_lens.shape[0])]
    # return h, c
    # List of predicted tokens for each time step
    predicted_tokens = []
    # Length of generated part for each object in the batch
    decoded_lens = torch.zeros_like(start_tokens_lens, dtype=torch.long)
    # Boolean mask where we store if the sequence has already generated
    # i.e. `<eos>` was selected on any step
    is_finished_decoding = torch.zeros_like(start_tokens_lens, dtype=torch.bool)
    
    # Stop when all sequences in the batch are finished
    while not torch.all(is_finished_decoding) and torch.max(decoded_lens) < max_generated_len:
        # Evaluate next token distribution using hidden state h.
        # Note. Over first dimension h has hidden states for each layer of LSTM.
        #     We must use hidden state from the last layer
   
        if mode == 'forward':
            logits = model.output_forward.forward(h[-1])
        elif mode == 'backward':
            logits = model.output_backward.forward(h[-1])
            
        if top_k is not None:
            # Top-k sampling. Use only top-k most probable logits to sample next token
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            # Mask non top-k logits
            logits[indices_to_remove] = -1e10
            # Sample next_token. 
            # YOUR CODE HERE
            probs = softmax(logits[logits > 0].reshape(-1, top_k), dim=-1)
            token_idx = torch.multinomial(probs, 1)
            real_indices = torch.argwhere((logits > 0))[:, 1].reshape(-1, top_k)
            next_token = real_indices[np.arange(real_indices.shape[0]), token_idx.flatten()]
        else:
            # Select most probable token
            # YOUR CODE HERE
            next_token = logits.argmax(axis=-1)
            
        predicted_tokens.append(next_token)
        
        decoded_lens += (~is_finished_decoding)
        is_finished_decoding |= (next_token == torch.tensor(model.vocab.lookup_indices(['<eos>'])[0]))

        # Evaluate embedding for next token
        embs = model.noncontext_embeddings(next_token).unsqueeze(0)

        # Update hidden and cell states
        if mode == 'forward':
            output, (h, c) = model.rnn_forward.forward(embs, (h, c))
        elif mode == 'backward':
            output, (h, c) = model.rnn_backward.forward(embs, (h, c))

    return torch.stack(predicted_tokens), decoded_lens