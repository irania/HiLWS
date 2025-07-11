import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from snorkel.labeling.model.label_model import LabelModel 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Constants for labels
NEGATIVE = 0
POSITIVE = 1

def get_ground_truth_label(x):
    # If the file name starts with '07', the label is NEGATIVE, otherwise POSITIVE
    return NEGATIVE if x.startswith('07') else POSITIVE

def run_weak_supervision(labels_df, output_file):
    # Extract the L_train matrix and file_name for ground truth
    # Determine the ground truth labels based on file_name
    file_names = labels_df['file_name']
    ground_truths = file_names.apply(get_ground_truth_label)
    
    _, L_train, _, y_test = train_test_split(labels_df.drop(columns=['file_name']).to_numpy(), ground_truths, test_size=0.4, random_state=32)
    

    ground_truths = y_test

    

    # Train the label model
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

    # Predict on the training set
    preds_train, probs = label_model.predict(L=L_train, return_probs=True)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, preds_train)
    print(f"Accuracy: {accuracy}")

    # Optionally, save or return the predicted labels
    #data['predicted_labels'] = preds_train
    #data.to_csv(output_file, index=False)

    # Plot confusion matrix
    cm = confusion_matrix(ground_truths, preds_train)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Non PD', 'PD'], yticklabels=['Non PD', 'PD'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Scatter plot from confusion matrix with a little noise added to the above results
    noise = np.random.normal(0, 0.1, preds_train.shape)
    noisy_preds_train = preds_train + noise
    plt.figure(figsize=(8, 6))
    plt.scatter(ground_truths, noisy_preds_train, alpha=0.6)
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label with Noise')
    plt.title('Scatter Plot with Noise')
    plt.show()

    # Heatmap showing the labeling function outputs, probabilistic label output, and y_pred
    combined_df = pd.DataFrame(L_train, columns=labels_df.columns[1:])
    combined_df['prob_label'] = probs[:, POSITIVE]
    combined_df['y_pred'] = preds_train

    
    plt.figure(figsize=(12, 8))
    sns.heatmap(combined_df, cmap="coolwarm", xticklabels=combined_df.columns)
    plt.title('Heatmap of Labeling Functions, Probabilistic Label, and y_pred')
    plt.show()


for task in ['left']:     
    labels_file = fr'\\files.ubc.ca\team\PPRC\Camera\Booth_Results\finger_tapping_ws\LFs\{task}_labels.csv'
    labels_df = pd.read_csv(labels_file)
    output_file = fr'\\files.ubc.ca\team\PPRC\Camera\Booth_Results\finger_tapping_ws\snorkel_output\{task}_snorkel.csv'
    combined_df = []
    run_weak_supervision(labels_df, output_file)
