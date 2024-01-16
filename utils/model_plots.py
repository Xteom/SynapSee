import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import torch


def plot_loss_accuracy(train_losses, train_accuracies, val_losses, val_accuracies, grey_background=True):
    """
    Plots the training and validation losses and accuracies.

    Args:
        train_losses (list): A list of training losses.
        val_losses (list): A list of validation losses.
        val_accuracies (list): A list of validation accuracies.
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plotting training and validation losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')

    # Plotting training and validation accuracies
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')

    if grey_background:
        # Set the background color of the figure
        fig.set_facecolor('grey')
        # Set the background color of the plot
        ax1.set_facecolor('grey')
        ax2.set_facecolor('grey')
        # set the label color
        ax1.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1, facecolor=(0.70, 0.70, 0.70))
        ax2.legend(fancybox=True, framealpha=1, shadow=False, borderpad=1, facecolor=(0.70, 0.70, 0.70))
    else:
        ax1.legend()
        ax2.legend()

    plt.show()

# plot confusion matrix and accuracy
def plot_metrics(true_labels, pred_labels, class_names, grey_background=True, size=(6, 6)):
    """
    Plots the confusion matrix and the accuracy for each class.

    :param true_labels: List or array of true labels
    :param pred_labels: List or array of predicted labels
    :param class_names: List of class names
    :param grey_background: Boolean, if True, set the background color to grey
    :param size: Tuple, the figure size
    """
    # Calculate the confusion matrix and class accuracies
    cm = confusion_matrix(true_labels, pred_labels)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=size)

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    if grey_background:
        # https://colors-picker.com/hex-color-picker/
        blue = "#2196f3"  
        grey = "#192c4d"  
        custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_blue_grey", [blue, grey])
        # custom_cmap = sns.cubehelix_palette(start=2.8, rot=0.1, dark=0.5, light=0.8, 
        #                                     reverse=False, as_cmap=True)
        disp.plot(cmap=custom_cmap, ax=axes[0], xticks_rotation='vertical', colorbar=False)
    else:
        disp.plot(cmap='Blues', ax=axes[0], xticks_rotation='vertical', colorbar=False)

    axes[0].set_title('Confusion Matrix')

    # Plot accuracy for each class
    axes[1].bar(class_names, class_accuracies)
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy for Each Class')
    axes[1].set_ylim([0, 1])

    # Set background color if grey_background is True
    if grey_background:
        fig.set_facecolor('grey')
        for ax in axes:
            ax.set_facecolor('grey')

    plt.tight_layout()
    plt.show()



def evaluate_class_metrics(model, 
                           data_loader, 
                           device, 
                           class_names, 
                           print_metrics=True, 
                           grey_background=True, 
                           size=(12, 6),
                           channels_to_include=None):
    """
    Evaluates the model on the given DataLoader and calculates the accuracy for each class.
    Also prints total number of observations, count per class, correct counts per class, and overall accuracy.

    :param model: The trained model
    :param data_loader: DataLoader for the dataset to evaluate
    :param device: The device to run the model on ('cuda' or 'cpu')
    :param class_names: List of class names
    :param print_metrics: Boolean, if True, print the metrics
    :param grey_background: Boolean, if True, set the background color to grey in the plots
    :param size: Tuple, the size of the plot
    :param channels_to_include: List or array of channel indices to include in the evaluation. If None, all channels are included.
                                the order of the channels is ['Fp1', 'Fp2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2', 'F7', 'F8', 'F3', 'F4', 'T7', 'T8', 'P3', 'P4']
    :return: None
    """
    model.eval()  # Set the model to evaluation mode
    true_labels = []
    pred_labels = []
    class_counts = [0] * len(class_names)
    correct_class_counts = [0] * len(class_names)
    correct_predictions = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['eeg_signal'].to(device)
            labels = batch['class'].to(device)

            # Select specified channels if channels_to_include is not None
            if channels_to_include is not None:
                inputs = inputs[:, :, channels_to_include]

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            labels_cpu = labels.cpu().numpy()
            predicted_cpu = predicted.cpu().numpy()
            true_labels.extend(labels_cpu)
            pred_labels.extend(predicted_cpu)

            for label, prediction in zip(labels_cpu, predicted_cpu):
                class_counts[label] += 1
                if label == prediction:
                    correct_class_counts[label] += 1
                    correct_predictions += 1

    total_observations = len(true_labels)
    overall_accuracy = correct_predictions / total_observations

    if print_metrics:    
        print(f"Total correct predictions: {correct_predictions}/{total_observations}")
        print(f"Overall accuracy: {overall_accuracy:.4f}")
        for i, class_name in enumerate(class_names):
            print(f"Correct predictions for {class_name}: {correct_class_counts[i]}/{class_counts[i]}; Accuracy: {correct_class_counts[i] / class_counts[i]:.4f}")

    # Calculate and plot accuracies
    plot_metrics(true_labels, pred_labels, class_names, 
                 grey_background=grey_background, size=size)