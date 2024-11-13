import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import seaborn as sns
import matplotlib.pyplot as plt

# Function to read data from a TXT file, skipping bad lines
def read_data(file_path, delimiter=' '):
    print(f"Reading data from: {file_path}")
    try:
        # Read the file, skipping bad lines
        df = pd.read_csv(file_path, delimiter=delimiter, header=None, on_bad_lines='skip', engine='python')
        print(f"Successfully read file with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error reading the file {file_path}: {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of failure

# Function to save output files to a specified directory
def save_output_files(directory, y_test, y_pred, cf_matrix, accuracy, class_report, metrics):
    np.savetxt(os.path.join(directory, 'actualValues.txt'), y_test, fmt='%d')
    np.savetxt(os.path.join(directory, 'predictedValues.txt'), y_pred, fmt='%d')
    with open(os.path.join(directory, 'mlp_performance_results.txt'), 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cf_matrix) + "\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(class_report + "\n")
        f.write(f"Precision (PPV): {metrics['precision']:.4f}\n")
        f.write(f"Recall (TPR): {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
        f.write(f"Specificity (TNR): {metrics['specificity']:.4f}\n")
        f.write(f"True Positive (TP): {metrics['TP']}\n")
        f.write(f"True Negative (TN): {metrics['TN']}\n")
        f.write(f"False Positive (FP): {metrics['FP']}\n")
        f.write(f"False Negative (FN): {metrics['FN']}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")

    print(f"\nModel performance metrics have been saved successfully to '{directory}'.")

# Function to calculate additional metrics
def calculate_metrics(cf_matrix):
    TN = cf_matrix[0][0]  # True Negative
    FP = cf_matrix[0][1]  # False Positive
    FN = cf_matrix[1][0]  # False Negative
    TP = cf_matrix[1][1]  # True Positive

    # Calculate Specificity (TNR), Sensitivity (TPR/Recall), and Precision (PPV)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # TPR / Sensitivity
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # PPV

    return {
        'specificity': specificity,
        'recall': recall,
        'precision': precision,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'TP': TP
    }

# Function to display the confusion matrix as a heatmap
def plot_confusion_matrix(cf_matrix):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='Blues', cbar=True)
    plt.title('Occupied or Non-occupied Seat Detection Confusion Matrix')
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()

# Initialize Tkinter for GUI file selection
root = Tk()
root.withdraw()  # Close the root window to prevent an extra Tkinter window from showing
print("Select Non-occupied (Empty) Data File")
non_occupied_chair_path = askopenfilename(title="Select Non-occupied (Empty) Data File", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
print(f"Non-occupied file selected: {non_occupied_chair_path}")
print("Select Occupied (Human) Data File")
occupied_chair_path = askopenfilename(title="Select Occupied (Human) Data File", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
print(f"Occupied file selected: {occupied_chair_path}")
if non_occupied_chair_path and occupied_chair_path:
    non_occupied_df = read_data(non_occupied_chair_path, delimiter=' ')
    occupied_df = read_data(occupied_chair_path, delimiter=' ')
    if non_occupied_df.empty or occupied_df.empty:
        print("Error reading one or both of the data files. Exiting.")
        exit()
    X = pd.concat([non_occupied_df, occupied_df], ignore_index=True)
    y = [1] * len(non_occupied_df) + [2] * len(occupied_df)  # Label 1 for non-occupied (empty), 2 for occupied (human)
    X = X.values
    y = np.array(y)
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the MLP classifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-3,
                        hidden_layer_sizes=(3,), random_state=42, max_iter=30000)
    print("Training the MLP classifier...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Compute the confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cf_matrix)
    plot_confusion_matrix(cf_matrix)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    class_report = classification_report(y_test, y_pred, target_names=['Non-occupied', 'Occupied'])
    print("\nClassification Report:")
    print(class_report)
    metrics = calculate_metrics(cf_matrix)
    f1 = f1_score(y_test, y_pred, average='micro')
    metrics['f1_score'] = f1
    print(f"\nDetailed Metrics:")
    print(f"True Positive (TP): {metrics['TP']}")
    print(f"True Negative (TN): {metrics['TN']}")
    print(f"False Positive (FP): {metrics['FP']}")
    print(f"False Negative (FN): {metrics['FN']}")
    print(f"Precision (PPV): {metrics['precision']:.4f}")
    print(f"Recall (TPR): {metrics['recall']:.4f}")
    print(f"Specificity (TNR): {metrics['specificity']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # Ask the user to select a directory to save the output files
    print("Select a directory to save the output files")
    output_directory = askdirectory(title="Select Directory to Save Output Files")

    if output_directory:
        save_output_files(output_directory, y_test, y_pred, cf_matrix, accuracy, class_report, metrics)
    else:
        print("No directory selected. Files were not saved.")
else:
    print("No file selected for non-occupied or occupied data. Exiting the program.")