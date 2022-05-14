from torch import nn
import torch

class CustomCNNBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, batchnorm, activation, dropout):
        super(CustomCNNBlock, self).__init__()
        self.bn_flag = batchnorm
        self.dropout_p = dropout
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.act = activation()
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.do = nn.Dropout(p=self.dropout_p)
        
    def forward(self, x):
        
        x = self.conv(x)
        if self.bn_flag is True:
            x = self.bn(x)
        x = self.act(x)
        x = self.do(x)
        return x
    
    
class CustomCNN(nn.Module):
    
    def __init__(self, depth, width, skip, dropout, batchnorm, activation=nn.ReLU, block_series_count=5):
        super(CustomCNN, self).__init__()
        depth -= 1 # start block 
        if block_series_count > depth:
            block_series_count = depth
        if block_series_count == 0:
              self.block_in_series = [0, 0, 0, 0, 0]
        elif depth % block_series_count == 0:
            self.blocks_in_series = [depth // block_series_count for i in range(block_series_count)]
        else:
            self.blocks_in_series = [(depth // block_series_count + 1) for i in range(depth % block_series_count)]
            self.blocks_in_series += [(depth // block_series_count) for i in range(block_series_count - depth % block_series_count)]
        
        pool_layer = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.nn_blocks = nn.ModuleList()
        self.skip_blocks = []
        cur_channels = 3
        start_block = CustomCNNBlock(in_channels=cur_channels, out_channels=round(width*cur_channels), batchnorm=batchnorm,
                                           activation=activation, dropout=dropout)
        self.nn_blocks.append(start_block)
        for block_series_idx in range(block_series_count):
            for block_idx in range(self.blocks_in_series[block_series_idx] - 1):
                
                cur_block = CustomCNNBlock(in_channels=round(width*cur_channels), out_channels=round(width*cur_channels), batchnorm=batchnorm,
                                           activation=activation, dropout=dropout)
                self.nn_blocks.append(cur_block)
                self.skip_blocks.append(len(self.nn_blocks) - 1)
                
            cur_block = CustomCNNBlock(in_channels=round(width*cur_channels), out_channels=round(2*width*cur_channels), batchnorm=batchnorm,
                                        activation=activation, dropout=dropout)
            self.nn_blocks.append(cur_block)
            self.nn_blocks.append(pool_layer)
            cur_channels *= 2
        
        if skip is False:
            self.skip_blocks = []
        fc_input_size = round(width*cur_channels) * 2**(5 - block_series_count) * 2**(5 - block_series_count)
        self.fc = nn.Linear(fc_input_size, 10)
        self.logsoftmax = nn.LogSoftmax(dim=1)
            
    def forward(self, x):
        
        for block_idx, block in enumerate(self.nn_blocks):
            if block_idx in self.skip_blocks:
                x = block.forward(x) + x
            else:
                x = block.forward(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        output = self.logsoftmax(x)
        return output