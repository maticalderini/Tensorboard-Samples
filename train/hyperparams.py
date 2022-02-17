#%%
from pathlib import Path

from itertools import product
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

from models.networks import GarmentClassifier
from datasets.mnist import FashionDataset, FashionLoader

from config import default_configs as configs

#%% General parameters
opt = configs['hyper']

data_root = opt['data_root']
board_dir = Path(opt['board_dir'])
save_dir = Path(opt['checkpoint_save_dir'])
board_dir.mkdir(parents=True, exist_ok=True)
save_dir.mkdir(parents=True, exist_ok=True)

download = opt['download_mnist']
shuffle = opt['shuffle']
num_workers = opt['workers']

save_count = 1 / opt['save_ratio']
board_count = 1 / opt['board_ratio']
display_count = 1 / opt['display_ratio']

#%% HyperParams
lrs = opt['lr']
momentums = opt['momentum']
batch_sizes = opt['batch_size']
epochs = opt['epochs']

#%%
N_combos = len(lrs) * len(momentums) * len(batch_sizes)
for combo, (lr, momentum, batch_size) in enumerate(product(lrs, momentums, batch_sizes)):
    message = f'Combination {combo} / {N_combos}'
    message += f' --- Batch size: {batch_size}'
    message += f' --- Momentum: {momentum}'
    message += f' --- Learning rate: {lr}'
    print(message)

    #%% Training data
    train_set = FashionDataset(data_root, download=download, train=True)
    train_loader = FashionLoader(train_set,
                                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    #%% Validation data
    val_set = FashionDataset(data_root, download=download, train=False)
    val_loader = FashionLoader(val_set,
                            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    #%% Model
    model = GarmentClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                        lr=lr, momentum=momentum)

    #%% Loop
    # Basic loop with no early stopping
    comment = f' batch_size = {batch_size} lr = {lr} momentum = {momentum}'
    writer = SummaryWriter(board_dir, comment=comment)

    print('Started training')
    best_val, best_train = np.inf, np.inf
    for epoch in range(epochs):
        # Validation Loop
        ## eval and no_grad are not redundant. First is to set all modules to eval mode, second to stop gradient calculation (should make it faster)
        model.eval()
        val_loss, accuracy = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()*inputs.size(0)
                
                preds = torch.argmax(outputs, 1)
                accuracy += torch.sum(preds == labels) / len(preds)
                
        val_loss /= len(val_loader.sampler)
        accuracy /= len(val_loader.sampler)
        best_val = min(best_val, val_loss)
        
        # Training loop
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            # Forward
            outputs = model(inputs)
            
            # Backward
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Running loss if using average already (default), multiply by inputs.size(0), not batch_size (in case last batch not full)
            train_loss += loss.item()*inputs.size(0)
        
        train_loss /= len(train_loader.sampler)
        best_train = min(best_train, train_loss)
        
        # Logging
        if int(epoch % display_count) == 0:
            message = f"Epoch: {epoch}"
            message += f" --- train loss: {train_loss :.6f} --- validation loss: {val_loss :.6f}"
            message += f" --- accuracy {accuracy :.3f}"
            print(message)
            
        if int(epoch % board_count) == 0:
            writer.add_scalar('Accuracy', accuracy, epoch)
            writer.add_scalars('Loss', 
                            {'Train': train_loss,
                            'Validation': val_loss},
                            epoch)
            writer.flush()

    writer.add_hparams({"lr": lr, "bsize": batch_size, "momentum": momentum},
                       {"Final Validation Loss": val_loss,
                        "Final Train Loss": train_loss,
                        "Best Validation Loss": best_val,
                        "Best Training Loss": best_train})
#%%
writer.add_graph(model, inputs)
writer.close()
print('Done')
# %%
