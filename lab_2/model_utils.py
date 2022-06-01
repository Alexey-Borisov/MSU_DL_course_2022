import torch


class RNNClassifier(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_size, vocab, pretrained_vectors, tuning, elmo, num_layers, dropout, concat, get_emb_f):
        super().__init__()
    
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.vocab = vocab
        self.pretrained_vectors = pretrained_vectors
        self.tuning = tuning
        self.elmo = elmo
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.concat = concat
        self.get_embeddings = get_emb_f
        
        # Create a simple lookup table that stores embeddings of a fixed dictionary and size.
        #    Use torch.nn.Embedding. Do not forget specify padding_idx!
        
        if True: # self.elmo is None:
            self.word_embeddings = torch.nn.Embedding(len(self.vocab), 25, padding_idx=0)
            if self.tuning is False:
                self.word_embeddings.weight.requires_grad = False
            if pretrained_vectors is not None:
                with torch.no_grad():
                    for token_idx, token in enumerate(self.vocab.vocab.itos_):
                        if token in pretrained_vectors:
                            self.word_embeddings.weight[token_idx] = torch.FloatTensor(pretrained_vectors.get_vector(token).copy())
                        else:
                            self.word_embeddings.weight[token_idx] = 0
        
        self.rnn_layer = torch.nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
                                      batch_first=False, dropout=self.dropout, bidirectional=True)
        
        # Create linear layer for classification
        self.output_layer = torch.nn.Linear(in_features=2*self.hidden_dim, out_features=self.output_size)
    
    def forward(self, tokens, reversed_tokens, tokens_lens):
        """
        :param torch.tensor(dtype=torch.long) tokens: Batch of texts represented with tokens.
        :param torch.tensor(dtype=torch.long) tokens_lens: Number of non-padding tokens for each object in batch.
        :return torch.tensor(dtype=torch.long): Vector representation for each sequence in batch
        """
        # tokens - (tokens_length, batch_size)
        # Evaluate embeddings
        if self.elmo is None:
            embs = self.word_embeddings(tokens)
        else:
            embs = self.get_embeddings(self.elmo, tokens, reversed_tokens, tokens_lens)
            if self.concat:
                noncontext_embs = self.word_embeddings(tokens)
                embs = torch.concat((embs, noncontext_embs), dim=-1)
            
        # embs - (tokens_length, batch_size, embedding_dim)
        
        # Make forward pass through recurrent network
        output, (h_state, C_state) = self.rnn_layer.forward(embs)
        # output - (tokens_length, batch_size, hidden_dim)
        # h_state - (1, batch_size, hidden_dim)
        # C_state - (1, batch_size, hidden_dim)
        
        # Pass output from rnn to linear layer 
        # Note: each object in batch has its own length 
        #     so we must take rnn hidden state after the last token for each text in batch
        correct_hidden = output[tokens_lens-1, torch.arange(output.size(1))]
        # correct_hidden - (batch_size, hidden_dim)
        
        result = self.output_layer.forward(correct_hidden.squeeze())
        # result - (batch_size, output_size)

        return result


def train_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    for idx, data in enumerate(dataloader):
        
        tokens, reversed_tokens, targets, tokens_lens = data['tokens'], data['reversed_tokens'], data['labels'], data['tokens_lens']
        tokens, reversed_tokens, targets = tokens.to(device), reversed_tokens.to(device), targets.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        model_outputs = model.forward(tokens, reversed_tokens, tokens_lens)
        loss = loss_fn(model_outputs, targets)
        loss.backward()
        optimizer.step()
        
    
def evaluate(dataloader, model, loss_fn, device):
    model.eval()
    
    total_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            
            tokens, reversed_tokens, targets, tokens_lens = data['tokens'], data['reversed_tokens'], data['labels'], data['tokens_lens']
            tokens, reversed_tokens, targets = tokens.to(device), reversed_tokens.to(device), targets.to(device)
            
            model_outputs = model.forward(tokens, reversed_tokens, tokens_lens)
            
            total_loss += loss_fn(model_outputs, targets) * targets.size(0)
            total_accuracy += (model_outputs.argmax(axis=1) == targets).sum()
        
    return total_loss / len(dataloader.dataset), total_accuracy / len(dataloader.dataset)
    

def train(
    train_loader, val_loader, model, loss_fn, optimizer, device, num_epochs
):
    # init history
    history = {
        'val_losses': [],
        'train_losses': [],
        'val_accuracies': [],
        'train_accuracies': [],
    }
    for epoch in range(num_epochs):
        
        # train one epoch
        train_epoch(train_loader, model, loss_fn, optimizer, device)
        
        # collect losses and accuracies for train data
        train_loss, train_acc = evaluate(train_loader, model, loss_fn, device)
        history['train_accuracies'].append(train_acc)
        history['train_losses'].append(train_loss)
        
        # collect losses and accuracies for val data
        val_loss, val_acc = evaluate(val_loader, model, loss_fn, device)
        history['val_accuracies'].append(val_acc)
        history['val_losses'].append(val_loss)
        
        # print results for current epoch
        print(
            'Epoch: {0:d}/{1:d}. Loss (Train/Val): {2:.3f}/{3:.3f}. Accuracy (Train/Val): {4:.3f}/{5:.3f}'.format(
                epoch + 1, num_epochs, history['train_losses'][-1], history['val_losses'][-1], history['train_accuracies'][-1], history['val_accuracies'][-1]
            )
        )
    return history


def get_results(dataloader, model, device):
    model.eval()
    
    results = torch.Tensor(size=(0,2)).to(device)
    true_labels = torch.Tensor(size=(0,)).to(device)
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            
            tokens, reversed_tokens, targets, tokens_lens = data['tokens'], data['reversed_tokens'], data['labels'], data['tokens_lens']
            tokens, reversed_tokens, targets = tokens.to(device), reversed_tokens.to(device), targets.to(device)
            
            model_outputs = model.forward(tokens, reversed_tokens, tokens_lens)
            
            results = torch.vstack((results, model_outputs))
            true_labels = torch.hstack((true_labels, targets))
        
    return results.cpu(), true_labels.cpu()