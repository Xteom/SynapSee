import torch
from sklearn.model_selection import train_test_split


def stratified_split(dataset,
                      stratify_by,
                      test_size=0.2, 
                      random_state=420, 
                      print_distributions=False):
    """
    Splits a dataset into training and validation subsets while preserving the class distribution.

    Args:
        dataset (Dataset): The dataset to split.
        stratify_by (str): The name of the attribute to stratify by.
        test_size (float): The proportion of the dataset to include in the validation set.
        random_state (int): The random seed to use for reproducible results.
        print_distributions (bool): Whether to print the class distributions in each subset.

    Returns:
        (Dataset, Dataset): A tuple containing the training and validation subsets.
    """
    # Get the indices corresponding to each class
    labels = [item[stratify_by] for item in dataset.data]

    # Split the dataset into training and validation sets while preserving the class distribution
    train_idx, val_idx = train_test_split(
        range(len(dataset)), 
        test_size=test_size, 
        stratify=labels,
        random_state=random_state)
    
    # Create the subsets
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)

    if print_distributions:
        def count_classes(dataset):
            """ Count the occurrences of each class in the dataset. """
            class_counts = {}
            for item in dataset:
                label = item['class']
                if label in class_counts:
                    class_counts[label] += 1
                else:
                    class_counts[label] = 1
            return class_counts

        # Count classes in each subset
        train_class_counts = count_classes(train_subset)
        val_class_counts = count_classes(val_subset)

        # print counts
        print("Training set class counts:", train_class_counts)
        print("Validation set class counts:", val_class_counts)

        # print percentages
        print("Training set class percentages:")
        for label, count in train_class_counts.items():
            print(f"{label}: {count / len(train_subset):.1%}")
        print("Validation set class percentages:")
        for label, count in val_class_counts.items():
            print(f"{label}: {count / len(val_subset):.1%}")

    return train_subset, val_subset


# training loop function
def train_model(model,
                  train_loader, 
                  val_loader,
                  criterion,
                  optimizer,
                  device,
                  num_epochs=50,
                  patience=-1,
                  verbose=False):
    """
    Trains a model and returns the training and validation losses and accuracies.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): A DataLoader for the training set.
        val_loader (DataLoader): A DataLoader for the validation set.
        criterion (torch.nn.Module): The loss function to use.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        device (str): The device to run the training on.
        num_epochs (int): The number of epochs to train for.
        patience (int): The number of epochs to wait for the validation loss to improve before early stopping.
                default: -1 (no early stopping) (all negative values are treated as -1)
        verbose (int): Whether to print the training progress every `verbose` epochs.

    Returns:
        (list, list, list): A tuple of lists containing the training losses, validation losses, and validation accuracies.
    """
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0 # total loss for the training set
        correct_train = 0 # total number of correct predictions in the training set
        total_train = 0 # total number of samples in the training set

        for batch in train_loader:
            eeg_signals = batch['eeg_signal'].to(device)
            targets = batch['class'].to(device)

            # Forward pass
            outputs = model(eeg_signals)
            loss = criterion(outputs, targets)

            # Backward and optimize
            # make gradients zero
            optimizer.zero_grad()
            # backpropagate
            loss.backward()
            # update parameters
            optimizer.step()

            total_train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train


        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in val_loader:
                eeg_signals = batch['eeg_signal'].to(device)
                targets = batch['class'].to(device)

                outputs = model(eeg_signals)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val  += (predicted == targets).sum().item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % verbose == 0 or epoch == num_epochs - 1):
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

        if epochs_no_improve == patience:
            print("Early stopping triggered")
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            break
        
    return train_losses, train_accuracies, val_losses, val_accuracies

