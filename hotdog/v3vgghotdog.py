import glob
import os

import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm.notebook import tqdm

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Hotdog_NotHotdog(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path='/dtu/datasets1/02516/hotdog_nothotdog'):
        """Initialization"""
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path + '/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')

    def __len__(self):
        """Returns the total number of samples"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Generates one sample of data"""
        image_path = self.image_paths[idx]

        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y


# Define the size and rotation degrees
size = 64
rotation_degrees = 90  # Adjust as needed

# Modify the train_transform to include random rotations
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=(-rotation_degrees, rotation_degrees)),
    transforms.Resize((size, size)),
    transforms.ToTensor(),
])

# Keep the test_transform as it is
test_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
])

# Rest of your code remains unchanged
batch_size = 64
trainset = Hotdog_NotHotdog(train=True, transform=train_transform)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
testset = Hotdog_NotHotdog(train=False, transform=test_transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)

images, labels = next(iter(train_loader))
plt.figure(figsize=(20, 10))

for i in range(21):
    plt.subplot(5, 7, i + 1)
    plt.imshow(np.swapaxes(np.swapaxes(images[i].numpy(), 0, 2), 0, 1))
    plt.title(['hotdog', 'not hotdog'][labels[i].item()])
    plt.axis('off')


#################################################################################################################

# Instantiate the VGG11 model with or without pretrained weights
model = models.vgg11(pretrained=False)

# Modify the classifier for your specific task
# Example: Replace the final fully connected layer for binary classification
model.classifier[6] = nn.Sequential(
    nn.Linear(4096, 1024),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),  # Add a dropout layer with dropout probability of 0.5
    nn.Linear(1024, 2)  # Change 2 to the number of classes in your dataset
)

# Set device and define loss function and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


#################################################################################################################

def visualize_images(images):
    global i, image, true_label, predicted_label
    plt.figure(figsize=(20, 10))
    for i in range(min(21, len(images))):
        plt.subplot(5, 7, i + 1)

        image = images[i].cpu().numpy()
        if image.shape[0] == 1:
            image = np.squeeze(image, axis=0)
            plt.imshow(image, cmap='gray')
        else:
            image = np.transpose(image, (1, 2, 0))
            plt.imshow(image)

        true_label = ['hotdog', 'not hotdog'][misclassified_labels[i].item()]
        predicted_label = ['hotdog', 'not hotdog'][misclassified_predictions[i].item()]
        plt.title(f'True: {true_label}\nPredicted: {predicted_label}', fontsize=8)
        plt.axis('off')
    plt.show()


#################################################################################################################


# Initialize the weights of the linear layers for the model without pretained weights

for layer in model.classifier.children():
    if isinstance(layer, nn.Linear):
        init.xavier_uniform_(layer.weight)

for layer in model.classifier.children():
    if isinstance(layer, nn.Conv2d):
        init.xavier_uniform_(layer.weight)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=4, gamma=0.1)  # Adjust step_size and gamma as needed

num_epochs = 10


for epoch in tqdm(range(num_epochs), unit='epoch'):
    # For each epoch
    model.train()
    train_correct = 0

    for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)

        # Zero the gradients computed for each weight
        optimizer.zero_grad()

        # Forward pass your image through the network
        output = model(data)

        # Compute the loss
        loss = criterion(output, target)

        # Backward pass through the network
        loss.backward()

        # Update the weights
        optimizer.step()

        # Compute how many were correctly classified
        predicted = output.argmax(1)
        train_correct += (target == predicted).sum().cpu().item()

    scheduler.step()  # Update the learning rate at the end of each epoch
    # Compute the test accuracy

    model.eval()
    test_correct = 0
    correct_indices = []  # List to store indices of correct predictions
    misclassified_images = []
    misclassified_labels = []
    misclassified_predictions = []
    correct_images = []  # List to store correctly predicted images
    correct_labels = []
    correct_predictions = []

    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)

        with torch.no_grad():
            output = model(data)

        predicted = output.argmax(1)
        target = target.to(device)

        correct_mask = target == predicted
        correct_indices.extend([i for i, value in enumerate(correct_mask) if value])

        misclassified_mask = target != predicted
        misclassified_images.extend(data[misclassified_mask])
        misclassified_labels.extend(target[misclassified_mask])
        misclassified_predictions.extend(predicted[misclassified_mask])

        correct_images.extend(data[correct_mask])
        correct_labels.extend(target[correct_mask])
        correct_predictions.extend(predicted[correct_mask])

        test_correct += correct_mask.sum().item()

    # Visualization of misclassified images
    visualize_images(misclassified_images)

    # Visualization of correctly predicted images
    visualize_images(correct_images)

    train_acc = train_correct / len(trainset)
    test_acc = test_correct / len(testset)
    print("Accuracy train: {train:.1f}%\t test: {test:.1f}%".format(test=100 * test_acc, train=100 * train_acc))
