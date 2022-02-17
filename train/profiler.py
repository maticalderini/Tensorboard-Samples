#%%
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

from models.networks import GarmentClassifier
from datasets.mnist import FashionDataset, FashionLoader

from config import default_configs as configs

#%% General parameters
opt = configs['profiler']

data_root = opt['data_root']
board_dir = Path(opt['board_dir'])
board_dir.mkdir(parents=True, exist_ok=True)

download = opt['download_mnist']
shuffle = opt['shuffle']
num_workers = opt['workers']

#%% HyperParams
lr = opt['lr']
momentum = opt['momentum']
batch_size = opt['batch_size']
epochs = opt['epochs']

#%% profiler
wait = opt['wait'] # Number of steps to skip
warmup = opt['warmup'] # Number of steps until warmup (tracing but discard to reduce overhead)
active = opt['active'] # Number of steps to record
repeat = opt['repeat'] # Number of times to repeat cycle

shapes = opt['shapes']
memory = opt['memory']
stack = opt['stack']

#%% Training data
train_set = FashionDataset(data_root, download=download, train=True)
train_loader = FashionLoader(train_set,
                             batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
#%% Model
model = GarmentClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=lr, momentum=momentum)

#%% Loop
schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
board_handler = torch.profiler.tensorboard_trace_handler(board_dir)

with torch.profiler.profile(schedule=schedule, on_trace_ready=board_handler,
                            record_shapes=shapes, profile_memory=memory, with_stack=stack) as prof:
    
    for step, (inputs, labels) in enumerate(train_loader):
        if step >= (wait + warmup + active) * repeat:
            break
        # Forward
        outputs = model(inputs)
        
        # Backward
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        prof.step()
print('Done')
# %%
