import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import TeamMateDataset
from torchvision.models import mobilenet_v3_small
from torchvision import transforms

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create the datasets and dataloaders
    trainset = TeamMateDataset(n_images=50, train=True)
    testset = TeamMateDataset(n_images=10, train=False)

    # Apply random horizontal flip and random rotation to the training dataset
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),  # Random rotation, max 10 degrees
        transforms.ToTensor(),
    ])

    # Modify the dataset to apply the transform
    trainset.transform = train_transform
    testset.transform = transforms.ToTensor()  # No augmentation on testset

    # Increase the batch size to 16
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # Create the model and optimizer
    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)

    # Increase the learning rate to 0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define both loss functions
    criterion_ce = torch.nn.CrossEntropyLoss()  # CrossEntropy loss
    criterion_nll = torch.nn.NLLLoss()          # NLL loss

    # Tracking best models for both loss types
    best_train_loss_ce = 1e9
    best_train_loss_nll = 1e9

    # Loss lists for tracking
    train_losses_ce = []
    test_losses_ce = []
    train_losses_nll = []
    test_losses_nll = []

    # Epoch Loop
    for epoch in range(1, 100):

        # Start timer
        t = time.time_ns()

        # Train the model for CrossEntropy loss (CE)
        model.train()
        train_loss_ce = 0
        for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):
            # Apply transformations during training (handled by dataset transform)
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute the CrossEntropy loss
            loss_ce = criterion_ce(outputs, labels)

            # Backward pass and optimize (we use CE loss for backpropagation)
            loss_ce.backward()
            optimizer.step()

            # Accumulate the losses
            train_loss_ce += loss_ce.item()

        # Train the model for Negative Log-Likelihood loss (NLL)
        model.train()
        train_loss_nll = 0
        for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):
            # Apply transformations during training (handled by dataset transform)
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute the NLL loss (log softmax first)
            log_outputs = torch.log_softmax(outputs, dim=1)
            loss_nll = criterion_nll(log_outputs, labels)

            # Backward pass and optimize (we use NLL loss for backpropagation)
            loss_nll.backward()
            optimizer.step()

            # Accumulate the losses
            train_loss_nll += loss_nll.item()

        # Test the model for CrossEntropy loss (CE)
        model.eval()
        test_loss_ce = 0
        correct_ce = 0
        total_ce = 0
        for images, labels in tqdm(testloader, total=len(testloader), leave=False):
            # Move the data to the device (CPU or GPU)
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute the CrossEntropy loss
            loss_ce = criterion_ce(outputs, labels)

            # Accumulate the losses
            test_loss_ce += loss_ce.item()

            # Get the predicted class from the maximum value in the output-list of class scores
            _, predicted = torch.max(outputs, 1)
            total_ce += labels.size(0)

            # Accumulate the number of correct classifications
            correct_ce += (predicted == labels).sum().item()

        # Test the model for Negative Log-Likelihood loss (NLL)
        model.eval()
        test_loss_nll = 0
        correct_nll = 0
        total_nll = 0
        for images, labels in tqdm(testloader, total=len(testloader), leave=False):
            # Move the data to the device (CPU or GPU)
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute the NLL loss (log softmax first)
            log_outputs = torch.log_softmax(outputs, dim=1)
            loss_nll = criterion_nll(log_outputs, labels)

            # Accumulate the losses
            test_loss_nll += loss_nll.item()

            # Get the predicted class from the maximum value in the output-list of class scores
            _, predicted = torch.max(outputs, 1)
            total_nll += labels.size(0)

            # Accumulate the number of correct classifications
            correct_nll += (predicted == labels).sum().item()

        # Print the epoch statistics
        print(f'Epoch: {epoch}, '
              f'Train Loss (CE): {train_loss_ce / len(trainloader):.4f}, Train Loss (NLL): {train_loss_nll / len(trainloader):.4f}, '
              f'Test Loss (CE): {test_loss_ce / len(testloader):.4f}, Test Loss (NLL): {test_loss_nll / len(testloader):.4f}, '
              f'Test Accuracy (CE): {correct_ce / total_ce:.4f}, Test Accuracy (NLL): {correct_nll / total_nll:.4f}, '
              f'Time: {(time.time_ns() - t) / 1e9:.2f}s')

        # Update loss lists for both loss functions
        train_losses_ce.append(train_loss_ce / len(trainloader))
        test_losses_ce.append(test_loss_ce / len(testloader))
        train_losses_nll.append(train_loss_nll / len(trainloader))
        test_losses_nll.append(test_loss_nll / len(testloader))

        # Save the best model for CrossEntropy loss (CE)
        if train_loss_ce < best_train_loss_ce:
            best_train_loss_ce = train_loss_ce
            torch.save(model.state_dict(), 'lab8/best_model_ce.pth')

        # Save the best model for NLL loss
        if train_loss_nll < best_train_loss_nll:
            best_train_loss_nll = train_loss_nll
            torch.save(model.state_dict(), 'lab8/best_model_nll.pth')

        # Save the model for each epoch (based on CE loss)
        torch.save(model.state_dict(), 'lab8/current_model.pth')

        # Create the loss plot
        plt.plot(train_losses_ce, label='Train Loss (CE)')
        plt.plot(test_losses_ce, label='Test Loss (CE)')
        plt.plot(train_losses_nll, label='Train Loss (NLL)')
        plt.plot(test_losses_nll, label='Test Loss (NLL)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('lab8/task2_loss_plot.png')
        plt.close()  # Close the plot to avoid memory issues over epochs
