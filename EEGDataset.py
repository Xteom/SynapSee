import json
import torch
from torch.utils.data import Dataset

SEGMENT_LENGTH = 123

class EEGDataset(Dataset):
    """
    A dataset class for EEG signals from various subjects.

    Attributes:
        class_mapping (dict): A mapping from class labels to integers.
        data (list): A list to store EEG data and metadata.

    Args:
        json_file (str): Path to the JSON file containing the dataset.
        subjects_to_include (list, optional): List of subject names to include in the dataset.
                                              If None, includes all subjects in the dataset.
    """

    # Define a mapping from class labels to integers as a class attribute
    class_mapping = {'cat': 0, 'dog': 1, 'rabbit': 2, 'control': 3}

    def __init__(self, json_file, subjects_to_include=None):
        with open(json_file, 'r') as file:
            data = json.load(file)

        self.data = []
        for subject in data['subjects']:
            # Include only the subjects specified in the subjects_to_include list
            if subjects_to_include is not None and subject['name'] not in subjects_to_include:
                continue

            for image in subject['view_images']:
                image_class = image['class']

                # Exclude samples with class 'no_stimuli'
                if image_class == 'no_stimuli':
                    continue

                # Map the class label to an integer
                class_int = self.class_mapping.get(image_class, -1)  # Default to -1 for unknown classes

                eeg_signal = image['EEG_signal']
                # Segment the EEG signal into predefined lengths
                for i in range(0, len(eeg_signal), SEGMENT_LENGTH):
                    segment = eeg_signal[i:i + SEGMENT_LENGTH]
                    if len(segment) == SEGMENT_LENGTH:
                        self.data.append({
                            "subject": subject['name'],
                            "EEG_signal": segment,
                            "class": class_int,
                            "age": subject['age'],
                            "sex": subject['sex'],
                            "has_cat": subject['has_cat'],
                            "has_dog": subject['has_dog'],
                            "has_rabbit": subject['has_rabbit']
                        })

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        used internally in PyTorch DataLoader to determine the size of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single item from the dataset.
        Args:
            idx (int): The index of the item to retrieve.
        Returns:
            dict: A dictionary containing the EEG signal and metadata.
        """
        item = self.data[idx]
        eeg_signal = torch.tensor(item['EEG_signal'], dtype=torch.float)
        return {
            "eeg_signal": eeg_signal,
            "class": item['class'],
            "subject": item['subject'],
            "age": item['age'],
            "sex": item['sex'],
            "has_cat": item['has_cat'],
            "has_dog": item['has_dog'],
            "has_rabbit": item['has_rabbit']
        }
