import nltk
import torch
import torchtext

from torch.utils.data import Dataset, DataLoader
from collections import Counter
from nltk.tokenize import word_tokenize
from tqdm.notebook import tqdm

nltk.download('punkt', quiet=True)


def tokenize(text):
    return word_tokenize(text.lower())

def get_vocab(data, tokenize_function=tokenize, vocab_size=10_000, sos_eos=False):
    
    counter = Counter()
    for text in tqdm(data.comment_text):
        tokenized_text = tokenize(text)
        counter.update(tokenized_text)
    
    top_counter = dict(Counter(counter).most_common(vocab_size))
    if sos_eos is True:
        specials = ['<pad>', '<unk>', '<sos>', '<eos>']
    else:
        specials = ['<pad>', '<unk>']
    vocab = torchtext.vocab.vocab(top_counter, specials=specials, special_first=True)
    vocab.set_default_index(vocab["<unk>"])
    
    return vocab


class CommentsDataset(Dataset):
    def __init__(self, dataframe, vocab, max_len, tokenize_function=tokenize, pad_sos=False, pad_eos=False):
        """
        :param str dataframe: DataFrame with columns comment_text and toxic
        :param torchtext.vocab.Vocab vocab: dictionary with lookup_indices method
        :param int max_len: Maximum length of tokenized text
        :param bool pad_sos: If True pad sequence at the beginning with <sos> 
        :param bool pad_eos: If True pad sequence at the end with <eos>         
        """
        super().__init__()
        
        self.pad_sos = pad_sos
        if self.pad_sos:
            self.sos_id = vocab.lookup_indices(['<sos>'])[0]
        self.pad_eos = pad_eos
        if self.pad_eos:
            self.eos_id = vocab.lookup_indices(['<eos>'])[0]
        
        self.vocab = vocab
        self.max_len = max_len
        self.dataframe = dataframe
        self.texts = []
        self.tokens = []
        self.labels = []
        # Read each file in data_path, tokenize it, get tokens ids, its rating and store
        for index, row in tqdm(self.dataframe.iterrows(), total=self.dataframe.shape[0]):
            text = row.comment_text
            label = row.toxic
            tokens = tokenize(text)
            vocab_tokens = list(filter(self.vocab.__contains__, tokens))
            vocab_tokens = vocab_tokens[:self.max_len]
            
            self.texts.append(text)
            self.tokens.append(self.vocab.lookup_indices(vocab_tokens))
            self.labels.append(label)
            
        
    def __getitem__(self, idx):
        """
        :param int idx: index of object in dataset
        :return dict: Dictionary with all useful object data 
            {
                'text' str: unprocessed text,
                'label' torch.tensor(dtype=torch.long): is text toxic (1 for toxic otherwise 0)
                'tokens' torch.tensor(dtype=torch.long): tensor of tokens ids for the text
                'tokens_len' torch.tensor(dtype=torch.long): number of tokens
            }
        """
        # YOUR CODE HERE
        # Do not forget to add padding if needed!
        
        cur_tokens = self.tokens[idx].copy()
        
        if self.pad_sos:
            cur_tokens.insert(0, self.sos_id)
            
        if self.pad_sos:
            cur_tokens.append(self.eos_id)
        
        result_dict = {
            'text': self.texts[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'tokens': torch.tensor(cur_tokens, dtype=torch.long),
            'reversed_tokens': torch.tensor(cur_tokens[::-1], dtype=torch.long),
            'tokens_len': torch.tensor(len(self.tokens[idx]), dtype=torch.long)
        }
        
        return result_dict
    
    def __len__(self):
        """
        :return int: number of objects in dataset 
        """
        # YOUR CODE HERE
        return len(self.texts)


def collate_fn(batch, padding_value, batch_first=False):
    """
    :param List[Dict] batch: List of objects from dataset
    :param int padding_value: Value that will be used to pad tokens
    :param bool batch_first: If True resulting tensor with tokens must have shape [B, T] otherwise [T, B]
    :return dict: Dictionary with all data collated
        {
            'labels' torch.tensor(dtype=torch.long): toxicity of the text for each object in batch
            'texts' List[str]: All texts in one list
            'tokens' torch.tensor(dtype=torch.long): tensor of tokens ids padded with @padding_value
            'tokens_lens' torch.tensor(dtype=torch.long): number of tokens for each object in batch
        }
    """
    result_dict = {
        'labels': torch.stack([obj['label'] for obj in batch]),
        'texts': [obj['text'] for obj in batch],
        'tokens': torch.nn.utils.rnn.pad_sequence([obj['tokens'] for obj in batch],
                                batch_first=batch_first, padding_value=padding_value),
        'reversed_tokens': torch.nn.utils.rnn.pad_sequence([obj['reversed_tokens'] for obj in batch],
                                batch_first=batch_first, padding_value=padding_value),
        'tokens_lens': torch.stack([obj['tokens_len'] for obj in batch])
    }
    
    return result_dict