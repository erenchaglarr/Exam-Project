import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
## Settings 
data_path = ""  ## Where is your data
class_path = "" ## Where are the different things you would like to know 
labels_path = "" ## Where are the different things labels. (What are they)


batch_size = 256



## Network

Net = nn.Sequential(
    
    nn.Conv2d(1,32,3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    
    
)
