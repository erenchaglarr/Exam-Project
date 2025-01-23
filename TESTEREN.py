import torch
import torchvision
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

#val data
data_path_imagenette_val =    "/Users/gustavbellaiche/Desktop/Alle_KID_dokumenter/1.Semester/Introduktion_til_KID/Eksamen_ITIS/ITIS_projekt/imagenette2-160/val"
class_path_imagenette_val =   ["n01440764", "n02102040", "n02979186" , "n03000684" , "n03028079" , "n03394916" , "n03417042" , "n03425413" , "n03445777" , "n03888257"]
class_labels_imagenette_val = ["fish", "dog", "stereo" , "chainsaw" , "church" ,"trumpet" , "truck" , "gas" , "golfball" , "parachute" ]
                        #Val    #387   #395    #357       #386         #409      #394        #389      #419    #399         #390

# dataset MNIST

# val data
data_path_mnist_val = "/Users/gustavbellaiche/Desktop/Alle_KID_dokumenter/1.Semester/Introduktion_til_KID/Eksamen_ITIS/ITIS_projekt/MNIST Dataset JPG format/val"
class_path_mnist_val = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
class_labels_mnist_val = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

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

# vores val data
images_imagenette_val = []
labels_imagenette_val = []

images_mnist_val = []
labels_mnist_val = []

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

images_tensor_imagenette_val = torch.stack(images_imagenette_val).float()
labels_tensor_imagenette_val = torch.tensor(labels_imagenette_val)

images_tensor_mnist_val = torch.stack(images_mnist_val).float()
labels_tensor_mnist_val = torch.tensor(labels_mnist_val)

# For mac (M-chip):

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS backend is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("MPS backend is not available. Using CPU.")

val_imagenette_data = TensorDataset(images_tensor_imagenette_val , labels_tensor_imagenette_val)
val_imagenette_loader = DataLoader(val_imagenette_data , batch_size = 1)

val_mnist_data  =TensorDataset(images_tensor_mnist_val , labels_tensor_mnist_val)
val_mnist_loader = DataLoader(val_mnist_data , batch_size = 1)

#%% Neural network
print("Her er den så")
print(len(labels_mnist_val))
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
    torch.nn.Linear(32 * 4 * 4, 10),
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
    torch.nn.Linear(32 * 4 * 4, 10),
).to(device)


net_imaginette.load_state_dict(torch.load('net_im_no_pad.pt'))
net_mnist.load_state_dict(torch.load('net__mnist_no_pad.pt'))

total_acc_imagenette = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
total_acc_mnist = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)

accuracy_imaginette_metric_fish = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
accuracy_imaginette_metric_dog = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
accuracy_imaginette_metric_stereo = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
accuracy_imaginette_metric_chainsaw = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
accuracy_imaginette_metric_church = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
accuracy_imaginette_metric_trumpet = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
accuracy_imaginette_metric_truck = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
accuracy_imaginette_metric_gas = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
accuracy_imaginette_metric_golfball = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
accuracy_imaginette_metric_parachute = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)

accuracy_mnist_metric_0 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
accuracy_mnist_metric_1 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
accuracy_mnist_metric_2 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
accuracy_mnist_metric_3 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
accuracy_mnist_metric_4 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
accuracy_mnist_metric_5 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
accuracy_mnist_metric_6 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
accuracy_mnist_metric_7 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
accuracy_mnist_metric_8 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)
accuracy_mnist_metric_9 = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10).to(device)

# validere imagenette
net_imaginette.eval()
net_mnist.eval()

#definere tomme lister til plots
y_imagenette = []
y_mnist = []

# kom så
for x_val , y_val in val_imagenette_loader:

    x_val = x_val.to(device)
    y_val = y_val.to(device)

    out_val = net_imaginette(x_val)

    total_acc_imagenette.update(out_val , y_val)
    
    if y_val == 0:
        accuracy_imaginette_metric_fish.update(out_val, y_val)
    elif y_val == 1:
        accuracy_imaginette_metric_dog.update(out_val, y_val)
    elif y_val == 2:
        accuracy_imaginette_metric_stereo.update(out_val, y_val)
    elif y_val == 3:
        accuracy_imaginette_metric_chainsaw.update(out_val, y_val)
    elif y_val == 4:
        accuracy_imaginette_metric_church.update(out_val, y_val)
    elif y_val == 5:
        accuracy_imaginette_metric_trumpet.update(out_val, y_val)
    elif y_val == 6:
        accuracy_imaginette_metric_truck.update(out_val, y_val)
    elif y_val == 7:
        accuracy_imaginette_metric_gas.update(out_val, y_val)
    elif y_val == 8:
        accuracy_imaginette_metric_golfball.update(out_val, y_val)
    else:
        accuracy_imaginette_metric_parachute.update(out_val, y_val)

# udregner samlet accuracy for imagenette
acc_fish = accuracy_imaginette_metric_fish.compute()
acc_dog = accuracy_imaginette_metric_dog.compute()
acc_stereo = accuracy_imaginette_metric_stereo.compute()
acc_chainsaw = accuracy_imaginette_metric_chainsaw.compute()
acc_church = accuracy_imaginette_metric_church.compute()
acc_trumpet = accuracy_imaginette_metric_trumpet.compute()
acc_truck = accuracy_imaginette_metric_truck.compute()
acc_gas = accuracy_imaginette_metric_gas.compute()
acc_golfball = accuracy_imaginette_metric_golfball.compute()
acc_parachute = accuracy_imaginette_metric_parachute.compute()

acc_total_imagenette = total_acc_imagenette.compute().item()   # total acc imagenette

# tilføjer til listen
y_imagenette.append(acc_fish.item())
y_imagenette.append(acc_dog.item())
y_imagenette.append(acc_stereo.item())
y_imagenette.append(acc_chainsaw.item())
y_imagenette.append(acc_church.item())
y_imagenette.append(acc_trumpet.item())
y_imagenette.append(acc_truck.item()) 
y_imagenette.append(acc_gas.item())
y_imagenette.append(acc_golfball.item())
y_imagenette.append(acc_parachute.item())

# printer
print("model accuracy for imagenette")
print(f"Fish Accuracy:      {acc_fish.item():.2f}")
print(f"Dog Accuracy:       {acc_dog.item():.2f}")
print(f"Stereo Accuracy:    {acc_stereo.item():.2f}")
print(f"Chainsaw Accuracy:  {acc_chainsaw.item():.2f}")
print(f"Church Accuracy:    {acc_church.item():.2f}")
print(f"Trumpet Accuracy:   {acc_trumpet.item():.2f}")
print(f"Truck Accuracy:     {acc_truck.item():.2f}")
print(f"Gas Accuracy:       {acc_gas.item():.2f}")
print(f"Golfball Accuracy:  {acc_golfball.item():.2f}")
print(f"Parachute Accuracy: {acc_parachute.item():.2f}")

# tester alt data for mnist 
for x_val , y_val in val_mnist_loader:

    x_val = x_val.to(device)
    y_val = y_val.to(device)

    out_val = net_mnist(x_val)

    total_acc_mnist.update(out_val , y_val)

    if y_val == 0:
        accuracy_mnist_metric_0.update(out_val , y_val)
    elif y_val == 1:
        accuracy_mnist_metric_1.update(out_val , y_val)
    elif y_val == 2:
        accuracy_mnist_metric_2.update(out_val , y_val)
    elif y_val == 3:
        accuracy_mnist_metric_3.update(out_val , y_val)
    elif y_val == 4:
        accuracy_mnist_metric_4.update(out_val , y_val)
    elif y_val == 5:
        accuracy_mnist_metric_5.update(out_val , y_val)
    elif y_val == 6:
        accuracy_mnist_metric_6.update(out_val , y_val)
    elif y_val == 7:
        accuracy_mnist_metric_7.update(out_val , y_val)
    elif y_val == 8:
        accuracy_mnist_metric_8.update(out_val , y_val)
    else:
        accuracy_mnist_metric_9.update(out_val , y_val)

# udregner samlet acc
acc_0 = accuracy_mnist_metric_0.compute()
acc_1 = accuracy_mnist_metric_1.compute()
acc_2 = accuracy_mnist_metric_2.compute()
acc_3 = accuracy_mnist_metric_3.compute()
acc_4 = accuracy_mnist_metric_4.compute()
acc_5 = accuracy_mnist_metric_5.compute()
acc_6 = accuracy_mnist_metric_6.compute()
acc_7 = accuracy_mnist_metric_7.compute()
acc_8 = accuracy_mnist_metric_8.compute()
acc_9 = accuracy_mnist_metric_9.compute()

acc_total_mnist = total_acc_mnist.compute().item() # total acc mnist

#tilføjer til tom liste
y_mnist.append(acc_0.item())
y_mnist.append(acc_1.item())
y_mnist.append(acc_2.item())
y_mnist.append(acc_3.item())
y_mnist.append(acc_4.item())
y_mnist.append(acc_5.item())
y_mnist.append(acc_6.item())
y_mnist.append(acc_7.item())
y_mnist.append(acc_8.item())
y_mnist.append(acc_9.item())

#printer
print("model accuracy for mnist:")
print(f"0 Accuracy: {acc_0.item():.2f}")
print(f"1 Accuracy: {acc_1.item():.2f}")
print(f"2 Accuracy: {acc_2.item():.2f}")
print(f"3 Accuracy: {acc_3.item():.2f}")
print(f"4 Accuracy: {acc_4.item():.2f}")
print(f"5 Accuracy: {acc_5.item():.2f}")
print(f"6 Accuracy: {acc_6.item():.2f}")
print(f"7 Accuracy: {acc_7.item():.2f}")
print(f"8 Accuracy: {acc_8.item():.2f}")
print(f"9 Accuracy: {acc_9.item():.2f}")

print(y_imagenette)
print(y_mnist)

# udregner konfidens intervaller
upper_imagenette = np.array(y_imagenette) + 1.96*np.sqrt((np.array(y_imagenette)*(1 - np.array(y_imagenette)))/len(labels_imagenette_val))
lower_imagenette = np.array(y_imagenette) - 1.96*np.sqrt((np.array(y_imagenette)*(1 - np.array(y_imagenette)))/len(labels_imagenette_val))

upper_mnist = np.array(y_mnist) + 1.96*np.sqrt((np.array(y_mnist)*(1 - np.array(y_mnist)))/len(labels_mnist_val))
lower_mnist = np.array(y_mnist) - 1.96*np.sqrt((np.array(y_mnist)*(1 - np.array(y_mnist)))/len(labels_mnist_val))


print("----------")
print(upper_imagenette , lower_imagenette , upper_mnist , lower_mnist)
print("-------------")
print(acc_total_imagenette , acc_total_mnist)

print("den er her__________")
print(len(class_labels_imagenette_val))


imagenette_error = [
    [y - l for y, l in zip(y_imagenette, lower_imagenette)],  # Lower error
    [u - y for u, y in zip(upper_imagenette, y_imagenette)],  # Upper error
]
mnist_error = [
    [y - l for y, l in zip(y_mnist, lower_mnist)],  # Lower error
    [u - y for u, y in zip(upper_mnist, y_mnist)],  # Upper error
]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

acc_total_imagenette


# Plot for Imagenette
ax[0].bar(class_labels_imagenette_val, y_imagenette, color='orange' , yerr = imagenette_error ,  capsize = 3)
ax[0].set_xlabel("Imagenette Classes")
ax[0].set_ylabel("Accuracy")
ax[0].set_title("Imagenette Class-wise Accuracy")
ax[0].set_ylim(0 , 1)
ax[0].grid(True , axis = "y")
ax[0].tick_params(axis='x', rotation=45)

# Plot for MNIST
ax[1].bar(class_labels_mnist_val, y_mnist, color='red' ,yerr = mnist_error ,  capsize = 3,  alpha=0.8)
ax[1].set_xlabel("MNIST Digits")
ax[1].set_ylabel("Accuracy")
ax[1].set_ylim(0 , 1)
ax[1].grid(True , axis = "y")
ax[1].set_title("MNIST Digit-wise Accuracy")

# Adjust layout
plt.tight_layout()

# Save the figure as a single image
plt.savefig("combined_accuracy_plots.png")

# Show the combined plots
plt.show()
