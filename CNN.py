#%% Import libraries

import torch
import torchvision
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset
import os
from glob import glob
import matplotlib.pyplot as plt

#%% indstillinger og filveje

# dataset med billeder af objecter

#train data
data_path_imagenette = "/Users/gustavbellaiche/Desktop/Alle_KID_dokumenter/1.Semester/Introduktion_til_KID/Eksamen_ITIS/ITIS_projekt/imagenette2-160/train"
class_path_imagenette = ["n01440764", "n02102040", "n02979186" , "n03000684" , "n03028079" , "n03394916" , "n03417042" , "n03425413" , "n03445777" , "n03888257"]
class_labels_imagenette = ["fish", "dog", "stereo" , "chainsaw" , "church" ,"trumpet" , "truck" , "gas" , "golfball" , "parachute" ]
                   #Train   #963     #955   #993      #858         #941       #956       #961      #931     #951         #960

#val data
data_path_imagenette_val =   "/Users/gustavbellaiche/Desktop/Alle_KID_dokumenter/1.Semester/Introduktion_til_KID/Eksamen_ITIS/ITIS_projekt/imagenette2-160/val"
class_path_imagenette_val =   ["n01440764", "n02102040", "n02979186" , "n03000684" , "n03028079" , "n03394916" , "n03417042" , "n03425413" , "n03445777" , "n03888257"]
class_labels_imagenette_val = ["fish", "dog", "stereo" , "chainsaw" , "church" ,"trumpet" , "truck" , "gas" , "golfball" , "parachute" ]
                        #Val    #387   #395    #357       #386         #409      #394        #389      #419    #399         #390


# dataset MNIST

#train data
data_path_mnist = "/Users/gustavbellaiche/Desktop/Alle_KID_dokumenter/1.Semester/Introduktion_til_KID/Eksamen_ITIS/ITIS_projekt/MNIST Dataset JPG format/train"
class_path_mnist = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
class_labels_mnist = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# val data
data_path_mnist_val = "/Users/gustavbellaiche/Desktop/Alle_KID_dokumenter/1.Semester/Introduktion_til_KID/Eksamen_ITIS/ITIS_projekt/MNIST Dataset JPG format/val"
class_path_mnist_val = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
class_labels_mnist_val = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


# settings
num_classes = 10
batch_size = 256
num_epochs = 500
learning_rate = 0.0001
weight_decay = 0.001
#%% Data preprocessing function

# for transormering af billeder data
def preprocess(image):
    # Convert to color if black-and-white
    if image.shape[0] == 1:
        image = image.repeat(3,1,1)
    # Convert to floating point numbers between 0 and 1
    image = image.float()/255
    # Crop to 160x160
    image = torchvision.transforms.functional.crop(image, 0, 0, 140, 140)
    # Resize to 32x32
    image = torchvision.transforms.functional.resize(image, [28, 28], antialias=True)

    return image


#%% Load data

# imagenette
# Empty lists to store images and labels
images_imagenette = []
labels_imagenette = []

# vores val data
images_imagenette_val = []
labels_imagenette_val = []

# Add each image to list
for i, label in enumerate(class_labels_imagenette):
    # Get all JPEG files in directory
    filenames = glob(os.path.join(data_path_imagenette, class_path_imagenette[i], '*.JPEG')) # søger om der er JPEG i vores dictionary
    for file in filenames:
        # Put image on list
        image = torchvision.io.read_image(file)
        image = preprocess(image)
        images_imagenette.append(image)
        # Put label on list
        labels_imagenette.append(i)

print("labels til træning er lavet (imagenette)")
# Add each image to list
for i, label in enumerate(class_labels_imagenette_val):
    # Get all JPEG files in directory
    filenames = glob(os.path.join(data_path_imagenette_val, class_path_imagenette_val[i], '*.JPEG')) # søger om der er JPEG i vores dictionary
    for file in filenames:
        # Put image on list
        image = torchvision.io.read_image(file)
        image = preprocess(image)
        images_imagenette_val.append(image)
        # Put label on list
        labels_imagenette_val.append(i)

print("labels til val er lavet (imagenette)")

# #MNIST
# # lister for træning
images_mnist = []
labels_mnist = []

# vores val data
images_mnist_val = []
labels_mnist_val = []

# til trænings data
for i, label in enumerate(class_labels_mnist):
    # Get all JPEG files in directory
    filenames = glob(os.path.join(data_path_mnist, class_path_mnist[i], '*.jpg')) # søger om der er JPEG i vores dictionary
    for file in filenames:
        # Put image on list
        image = torchvision.io.read_image(file)
        image = preprocess(image)
        images_mnist.append(image)
        # Put label on list
        labels_mnist.append(i)

print("labels til træning er nu lavet (MNIST)")

# til validation data
for i, label in enumerate(class_labels_mnist_val):
    # Get all JPEG files in directory
    filenames = glob(os.path.join(data_path_mnist_val, class_path_mnist_val[i], '*.jpg')) # søger om der er JPEG i vores dictionary
    for file in filenames:
        # Put image on list
        image = torchvision.io.read_image(file)
        image = preprocess(image)
        images_mnist_val.append(image)
        # Put label on list
        labels_mnist_val.append(i)

print("liste med træningsdata" , len(images_mnist))
print("liste med valdata" , len(images_mnist_val))

print("labels til val er nu lavet (MNIST)")

# laver vores imaginette tensors
images_tensor_imagenette = torch.stack(images_imagenette).float()
labels_tensor_imagenette = torch.tensor(labels_imagenette)

images_tensor_imagenette_val = torch.stack(images_imagenette_val).float()
labels_tensor_imagenette_val = torch.tensor(labels_imagenette_val)

# laver vores MNIST tensors
images_tensor_mnist = torch.stack(images_mnist).float()
labels_tensor_mnist = torch.tensor(labels_mnist)

images_tensor_mnist_val = torch.stack(images_mnist_val).float()
labels_tensor_mnist_val = torch.tensor(labels_mnist_val)

#%% Device
# Run on GPU if available
# device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cuda"

# print("device sat til cuda")

# For mac (M-chip):

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS backend is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("MPS backend is not available. Using CPU.")
#%% Create dataloader

# imagenette

#train
train_imagenette_data = TensorDataset(images_tensor_imagenette, labels_tensor_imagenette)
train_imagenette_loader = DataLoader(train_imagenette_data, batch_size = batch_size, shuffle=True)
#val
val_imagenette_data = TensorDataset(images_tensor_imagenette_val , labels_tensor_imagenette_val)
val_imagenette_loader = DataLoader(val_imagenette_data , batch_size = len(labels_imagenette))
print("imagenette tensor færdigt")

# MNIST

#train
train_mnist_data = TensorDataset(images_tensor_mnist , labels_tensor_mnist)
train_mnist_loader = DataLoader(train_mnist_data , batch_size = batch_size , shuffle = True)
#val
val_mnist_data  =TensorDataset(images_tensor_mnist_val , labels_tensor_mnist_val)
val_mnist_loader = DataLoader(val_mnist_data , batch_size = len(labels_mnist))

print("MNIST tensor færdigt")
#%% Neural network

# # imaginette_train # med dropout
net_imaginette = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, kernel_size=3),   # 16 x 26 x 26
    torch.nn.BatchNorm2d(16),
    torch.nn.ReLU(),
    torch.nn.Conv2d(16, 16, kernel_size=3),   # 16 x 24 x 24
    torch.nn.BatchNorm2d(16),
    torch.nn.ReLU(),
    torch.nn.Conv2d(16, 32, kernel_size=3),  # 32 x 22 x 22
    torch.nn.BatchNorm2d(32),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),                      # 32 x 11 x 11
    torch.nn.Conv2d(32, 32, kernel_size=3), # 32 x 9 x 9
    torch.nn.BatchNorm2d(32),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),                      # 32 x 4 x 4
    torch.nn.Flatten(),
    torch.nn.Dropout(0.5),
    torch.nn.Dropout(0.25),
    torch.nn.Linear(32 * 4 * 4, num_classes),
).to(device)

# #MNIST train # med dropout
net_mnist = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, kernel_size=3),   # 16 x 26 x 26
    torch.nn.BatchNorm2d(16),
    torch.nn.ReLU(),
    torch.nn.Conv2d(16, 16, kernel_size=3),   # 16 x 24 x 24
    torch.nn.BatchNorm2d(16),
    torch.nn.ReLU(),
    torch.nn.Conv2d(16, 32, kernel_size=3),  # 32 x 22 x 22
    torch.nn.BatchNorm2d(32),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),                      # 32 x 11 x 11
    torch.nn.Conv2d(32, 32, kernel_size=3), # 32 x 9 x 9
    torch.nn.BatchNorm2d(32),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2),                      # 32 x 4 x 4
    torch.nn.Flatten(),
    torch.nn.Dropout(0.5),
    torch.nn.Dropout(0.25),
    torch.nn.Linear(32 * 4 * 4, num_classes),
).to(device)


# %% Load trained network from file
# net.load_state_dict(torch.load('net.pt'))    # hovsa, den skal du lige passe på med at bruge

#%% Loss and optimizer
loss_function = torch.nn.CrossEntropyLoss() #sandsynligheds lossfunction. #softmax indbygget i crossentrypy

optimizer_imagenette = torch.optim.Adam(net_imaginette.parameters(), lr = learning_rate, weight_decay=weight_decay)   # det er en anden

optimizer_mnist = torch.optim.Adam(net_mnist.parameters(), lr = learning_rate, weight_decay=weight_decay)

#%% Metrics
accuracy_imaginette_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes).to(device)
accuracy_imaginette_metric_val = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes).to(device)

accuracy_mnist_metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes).to(device)
accuracy_mnist_metric_val = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes).to(device)

print("vi begynder træning")
#%% Train

# lister til plot

#imagenette
acc_train_imagenette = []
acc_val_imagenette = []
#MNIST
acc_train_mnist = []
acc_val_mnist = []

epoch_værdier = range(1 , num_epochs+1)


# træner på imagenette data
for epoch in range(num_epochs):
    net_imaginette.train()
    net_mnist.train()
    accuracy_imaginette_metric.reset()
    accuracy_imaginette_metric_val.reset()
    accuracy_mnist_metric.reset()
    accuracy_mnist_metric_val.reset()

    # træner model på imaginette
    for x, y in train_imagenette_loader:

        # Put data on GPU 
        x = x.to(device)
        y = y.to(device)

        # Compute loss and take gradient step
        out = net_imaginette(x)

        loss = loss_function(out, y)

        optimizer_imagenette.zero_grad()
        loss.backward()
        optimizer_imagenette.step()        

        # Update accuracy metric
        accuracy_imaginette_metric.update(out, y)

    # validere imagenette
    net_imaginette.eval()
    for x_val , y_val in val_imagenette_loader:

        x_val = x_val.to(device)
        y_val = y_val.to(device)

        out_val = net_imaginette(x_val)

        accuracy_imaginette_metric_val.update(out_val , y_val)

    for x, y in train_mnist_loader:

        x = x.to(device)
        y = y.to(device)

        out = net_mnist(x)

        loss = loss_function(out , y)

        optimizer_mnist.zero_grad()
        loss.backward()
        optimizer_mnist.step()

        accuracy_mnist_metric.update(out, y)
        
    net_mnist.eval()
    for x_val , y_val in val_mnist_loader:

        x_val = x_val.to(device)
        y_val = y_val.to(device)

        out_val = net_mnist(x_val)

        accuracy_mnist_metric_val.update(out_val , y_val)

    # Print accuracy for epoch            
    acc_imagenette_train = accuracy_imaginette_metric.compute()
    acc_imaginatte_val =  accuracy_imaginette_metric_val.compute()
    #torch.Tensor.floattorch.Tensor.float
    acc_val_imagenette.append(float(acc_imaginatte_val))
    acc_train_imagenette.append(float(acc_imagenette_train))

    print(f'epoch: {epoch + 1} Training accuracy vs val data = {acc_imagenette_train} | {acc_imaginatte_val} imagenette' )

    acc_mnist_train = accuracy_mnist_metric.compute()
    acc_mnist_val =  accuracy_mnist_metric_val.compute()
    #torch.Tensor.floattorch.Tensor.float
    acc_train_mnist.append(float(acc_mnist_train))
    acc_val_mnist.append(float(acc_mnist_val))

    print(f'epoch: {epoch + 1} Training accuracy vs val data = {acc_mnist_train} | {acc_mnist_val} MNIST')
    print("------")


#%% Save the trained model
torch.save(net_imaginette.state_dict(), 'net_im_no_pad.pt')
torch.save(net_mnist.state_dict(), 'net__mnist_no_pad.pt')

# Beregn gennemsnitlig valideringsaccuracy for de sidste 20 epochs
imagenette_val_avg = sum(acc_val_imagenette[-20:]) / min(len(acc_val_imagenette), 20)
mnist_val_avg = sum(acc_val_mnist[-20:]) / min(len(acc_val_mnist), 20)
overall_avg = (imagenette_val_avg + mnist_val_avg) / 2

# Dynamisk generér information om det neurale netværk fra variablerne
batch_norm = "Ja" if "BatchNorm2d" in str(net_imaginette) else "Nej"
padding = "Ja" if "padding=" in str(net_imaginette) else "Nej"
dropout_layers = [float(layer.p) for layer in net_imaginette if isinstance(layer, torch.nn.Dropout)]
num_conv_layers = sum(1 for layer in net_imaginette if isinstance(layer, torch.nn.Conv2d))
num_maxpool_layers = sum(1 for layer in net_imaginette if isinstance(layer, torch.nn.MaxPool2d))
fc_layers = [layer.in_features for layer in net_imaginette if isinstance(layer, torch.nn.Linear)]
fc_layers.append([layer.out_features for layer in net_imaginette if isinstance(layer, torch.nn.Linear)][-1])

# Lav teksten til boksen
info_text = (
    f"Learning rate: {learning_rate}\n"
    f"Weight Decay: {weight_decay}\n"
    f"Dropout Layers: {len(dropout_layers)} ({', '.join(map(str, dropout_layers))})\n"
    f"Gns. valideringsaccuracy (Imagenette - sidste 20 epochs): {imagenette_val_avg:.2f}\n"
    f"Gns. valideringsaccuracy (MNIST - sidste 20 epochs): {mnist_val_avg:.2f}\n"
    f"Gns. accuracy (sidste 20 epochs): {overall_avg:.2f}"
)

# Plot med boksen
plt.figure(figsize=(12, 6))

# Plot Imagenette accuracy
plt.plot(epoch_værdier, acc_train_imagenette, label="Imagenette Training Accuracy", linestyle="--")
plt.plot(epoch_værdier, acc_val_imagenette, label="Imagenette Validation Accuracy", linestyle="-")

# Plot MNIST accuracy
plt.plot(epoch_værdier, acc_train_mnist, label="MNIST Training Accuracy", linestyle="--")
plt.plot(epoch_værdier, acc_val_mnist, label="MNIST Validation Accuracy", linestyle="-")

# Sæt y-aksen til at være mellem 0 og 1
plt.ylim(0, 1)

# Tilføj labels, titel og en legend
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy for Imagenette and MNIST")
plt.legend()

# Tilføj gitter
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Tilføj gitter med mindre intervaller
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

import matplotlib.ticker as ticker

# Indstil major og minor ticks for y-aksen
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.2))  # Hovedstreger ved 0.2, 0.4, osv.
plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(0.1))  # Mindre streger ved 0.1, 0.3, osv.

# Placér boksen i nederste venstre hjørne og gør den mindre
plt.text(
    x=0.05, y=0.05,  # Juster koordinater for at placere den nederst til venstre
    s=info_text,
    fontsize=8,  # Reducer skrifttypestørrelsen for at gøre boksen mindre
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="black")
)

# Vis plottet
plt.show()


