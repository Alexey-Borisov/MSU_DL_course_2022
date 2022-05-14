import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


def show_samples(train_dataset):
    
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    axes = axes.ravel()
    
    for ax in axes:
        elem_idx = np.random.randint(0, train_dataset.data.shape[0])
        img, number = train_dataset[elem_idx]
        ax.set_xlabel(number, fontsize=35, labelpad=15)
        ax.imshow(img.permute(1, 2, 0) )
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_xticks([])
    plt.show()
    

def show_layer_grads(layer_grads, epoch_idx, color=sns.color_palette("tab10")[0]):
    
    if len(layer_grads) <= 4:
        fig, axes = plt.subplots(nrows=1, ncols=len(layer_grads), figsize=(22, 5))
        if len(layer_grads) == 1:
            axes = [axes]
        for i, layer_grad in enumerate(layer_grads):
            show_hist(layer_grad[epoch_idx], ax=axes[i], title=f'Слой {i+1}, Эпоха {epoch_idx+1}', color=color)
    else:
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(22, 5))
        idx_1 = round(len(layer_grads)/3)
        idx_2 = round(2*len(layer_grads)/3)
        show_hist(layer_grads[0][epoch_idx], ax=axes[0], title=f'Слой 1, Эпоха {epoch_idx+1}', color=color)
        show_hist(layer_grads[idx_1][epoch_idx], ax=axes[1], title=f'Слой {idx_1 + 1}, Эпоха {epoch_idx+1}', color=color)
        show_hist(layer_grads[idx_2][epoch_idx], ax=axes[2], title=f'Слой {idx_2 + 1}, Эпоха {epoch_idx+1}', color=color)
        show_hist(layer_grads[-1][epoch_idx], ax=axes[3], title=f'Слой {len(layer_grads)}, Эпоха {epoch_idx+1}', color=color)
        plt.tight_layout()
        
        
def show_hist(x, bins=80, title='', color=sns.color_palette("tab10")[0], ax=None):
    sns.set_style('darkgrid')
    sns.set_style('darkgrid')
    limit = 100_000
    if x.shape[0] > limit:
        perm = torch.randperm(x.shape[0])
        idx = perm[:limit]
        x = x[idx]
    coef = 1.5
    x_left = torch.quantile(x, 0.01)
    if x_left >= 0: 
        x_left /= coef
    else:
        x_left *= coef
    x_right = torch.quantile(x, 0.99)
    if x_right >= 0: 
        x_right *= coef
    else:
        x_right /= coef
    x = x[(x > x_left) & (x < x_right)]
    if ax is None:
        plt.title(title, fontsize=20, pad=15)
        plt.xlabel('Величина градиента', fontsize=15)
        plt.ylabel('Количество объектов', fontsize=15)
        plt.xlim(x_left, x_right)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
    else:
        ax.set_title(title, fontsize=20, pad=15)
        ax.set_xlabel('Величина градиента', fontsize=15)
        ax.set_ylabel('Количество объектов', fontsize=15)
        ax.set_xlim(x_left, x_right)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
    sns.histplot(x, bins=bins, ax=ax, kde=True, color=color)   