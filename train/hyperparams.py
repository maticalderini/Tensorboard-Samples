#%%
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

from models.networks import GarmentClassifier
from datasets.mnist import FashionDataset, FashionLoader
from utils.utils import save_checkpoints

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
lr = opt['lr']
momentum = opt['momentum']
batch_size = opt['batch_size']
epochs = opt['epochs']

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
                      lr=opt['lr'], momentum=opt['momentum'])

#%% Loop
# Basic loop with no early stopping
writer = SummaryWriter(board_dir)

val_best = torch.inf
print('Started training')
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
        
        if val_best > val_loss:
            message = f'Epoch {epoch}'
            message += f' --- Model validation improved: {val_best:.6f} --> {val_loss:.6f}'
            message += ' --- Saving model'
            print(message)            
            save_checkpoints(model, save_dir / f'base_model_best.pth')
            val_best = val_loss
    
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
    # Logging
    if int(epoch % display_count) == 0:
        message = f"Epoch: {epoch}"
        message += f" --- train loss: {train_loss :.6f} --- validation loss: {val_loss :.6f}"
        message += f" --- accuracy {accuracy :.3f}"
        print(message)
    
    if int(epoch % save_count) == 0:
        print(f"> Epoch {epoch} - Saving checkpoint")
        save_checkpoints(model, save_dir / f'base_model_{epoch}.pth')
        
    if int(epoch % board_count) == 0:
        writer.add_scalar('Accuracy', accuracy, epoch)
        writer.add_scalars('Loss', 
                          {'Train': train_loss,
                           'Validation': val_loss},
                          epoch)
        writer.add_images('Sample training image', inputs, epoch)
        writer.flush()
        
save_checkpoints(model, save_dir / f'base_model_final.pth')
#%%
writer.add_graph(model, inputs)
writer.close()
print('Done')
# %%
