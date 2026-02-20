# ==========================================================
# CIFAR-10 Classification Pipeline
# Baseline -> HOG -> PCA -> Transfer Learning -> Fine-Tuning
# ==========================================================

import numpy as np
import pickle
import time
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage.feature import hog

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader

# ==========================================================
# 1) DATA LOADING
# ==========================================================

def unpickle(file):
    """Load a CIFAR batch file"""
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def load_cifar(filenames):
    """Load multiple CIFAR batch files"""
    images_list = []
    labels_list = []

    for file_name in filenames:
        batch = unpickle(file_name)
        images = batch[b'data']
        labels = batch[b'labels']

        # Reshape to (N, 3, 32, 32)
        images = images.reshape(-1, 3, 32, 32)

        # Convert to (N, 32, 32, 3)
        images = images.transpose(0, 2, 3, 1)

        images_list.append(images)
        labels_list += labels

    return np.vstack(images_list), labels_list


print("Loading training set...")
training_files = [f"D:/python/feature engineering bookcamp/ch06/data_batch_{i}" for i in range(1, 6)]
training_images, int_training_labels = load_cifar(training_files)

print("Loading testing set...")
testing_files = [r'D:\python\feature engineering bookcamp\ch06\test_batch']
testing_images, int_testing_labels = load_cifar(testing_files)

print("Loading label names...")
label_names = unpickle(r'D:\python\feature engineering bookcamp\ch06\batches.meta')[b'label_names']

training_labels = [str(label_names[_]) for _ in int_training_labels]
testing_labels = [str(label_names[_]) for _ in int_testing_labels]

training_images = training_images.astype(np.float32)
testing_images = testing_images.astype(np.float32)

# ==========================================================
# 2) VISUALIZATION
# ==========================================================

plt.imshow(training_images[0].astype(np.uint8))
plt.title(training_labels[0])
plt.show()

# ==========================================================
# 3) GRID SEARCH HELPER
# ==========================================================

def advanced_grid_search(x_train, y_train, x_test, y_test,
                         ml_pipeline, params,
                         cv=3, include_probas=False,
                         is_regression=False):
    """
    Perform GridSearchCV and print classification report
    """
    model_grid_search = GridSearchCV(
        ml_pipeline,
        param_grid=params,
        cv=cv,
        error_score=-1
    )

    start_time = time.time()
    model_grid_search.fit(x_train, y_train)

    best_model = model_grid_search.best_estimator_
    y_preds = best_model.predict(x_test)

    if is_regression:
        rmse = np.sqrt(mean_squared_error(y_test, y_preds))
        print(f'RMSE: {rmse:.5f}')
    else:
        print(classification_report(y_test, y_preds))

    print(f'Best params: {model_grid_search.best_params_}')
    print(f"Overall took {(time.time() - start_time):.2f} seconds")

    if include_probas:
        y_probas = best_model.predict_proba(x_test).max(axis=1)
        return best_model, y_preds, y_probas

    return best_model, y_preds


# ==========================================================
# 4) BASELINE: Average Pixel + Logistic Regression
# ==========================================================

avg_training_images = training_images.mean(axis=3).reshape(50000, -1)
avg_testing_images = testing_images.mean(axis=3).reshape(10000, -1)

clf = LogisticRegression(max_iter=200, solver='saga')

ml_pipeline = Pipeline([
    ('classifier', clf)
])

params = {
    'classifier__C': [0.1, 1, 10]
}

print("Average Pixel Value + LogReg")
advanced_grid_search(
    avg_training_images, training_labels,
    avg_testing_images, testing_labels,
    ml_pipeline, params
)

# ==========================================================
# 5) HOG FEATURE EXTRACTION
# ==========================================================

def calculate_hogs(images):
    """Compute HOG descriptors for all images"""
    hog_descriptors = []

    for image in tqdm(images):
        descriptor = hog(
            image,
            orientations=8,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            channel_axis=-1,
            transform_sqrt=True,
            block_norm='L2-Hys',
            visualize=False
        )
        hog_descriptors.append(descriptor)

    return np.array(hog_descriptors)


hog_training = calculate_hogs(training_images)
hog_testing = calculate_hogs(testing_images)

print("HOG + LogReg")
advanced_grid_search(
    hog_training, training_labels,
    hog_testing, testing_labels,
    ml_pipeline, params
)

# ==========================================================
# 6) HOG + PCA
# ==========================================================

pca = PCA(n_components=600)
hog_training_pca = pca.fit_transform(hog_training)
hog_testing_pca = pca.transform(hog_testing)

print("HOG + PCA + LogReg")
advanced_grid_search(
    hog_training_pca, training_labels,
    hog_testing_pca, testing_labels,
    ml_pipeline, params
)

# ==========================================================
# 7) VGG11 FEATURE EXTRACTION (TRANSFER LEARNING)
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg_model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
vgg_model.eval()
vgg_model.to(device)

# Normalize images using ImageNet statistics
normalized_training_images = ((training_images / 255.0)
                              - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

normalized_testing_images = ((testing_images / 255.0)
                             - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

train_tensor = torch.tensor(
    normalized_training_images.transpose(0, 3, 1, 2)
).float()

train_labels_tensor = torch.tensor(
    int_training_labels
).long()

test_tensor = torch.tensor(
    normalized_testing_images.transpose(0, 3, 1, 2)
).float()

test_labels_tensor = torch.tensor(
    int_testing_labels
).long()

train_loader = DataLoader(
    TensorDataset(train_tensor, train_labels_tensor),
    batch_size=256,
    shuffle=False
)

test_loader = DataLoader(
    TensorDataset(test_tensor, test_labels_tensor),
    batch_size=256,
    shuffle=False
)


def extract_vgg_features(model, dataloader):
    """Extract convolutional features from VGG"""
    features = []
    labels = []

    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = images.to(device)
            outputs = model.features(images)
            outputs = torch.flatten(outputs, 1)
            features.append(outputs.cpu().numpy())
            labels += targets.numpy().tolist()

    return np.vstack(features), labels


vgg_train_features, vgg_train_labels = extract_vgg_features(vgg_model, train_loader)
vgg_test_features, vgg_test_labels = extract_vgg_features(vgg_model, test_loader)

print("VGG11 (ImageNet) + LogReg")
advanced_grid_search(
    vgg_train_features, vgg_train_labels,
    vgg_test_features, vgg_test_labels,
    ml_pipeline, params
)

# ==========================================================
# 8) FINE-TUNING VGG11
# ==========================================================

fine_tuned_vgg_model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)

# Replace final classification layer for CIFAR-10
fine_tuned_vgg_model.classifier[-1] = nn.Linear(4096, 10)

fine_tuned_vgg_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    fine_tuned_vgg_model.parameters(),
    lr=0.01,
    momentum=0.9
)

n_epochs = 15
valid_loss_min = np.inf

for epoch in range(1, n_epochs + 1):

    fine_tuned_vgg_model.train()
    running_loss = 0
    correct = 0
    total = 0

    print(f"Epoch {epoch}")

    for images, targets in train_loader:

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = fine_tuned_vgg_model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == targets).item()
        total += targets.size(0)

    train_acc = 100 * correct / total
    print(f"Train Loss: {running_loss:.4f}, Train Acc: {train_acc:.2f}%")

    # Validation
    fine_tuned_vgg_model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = fine_tuned_vgg_model(images)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == targets).item()
            val_total += targets.size(0)

    val_acc = 100 * val_correct / val_total
    print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%")

    if val_loss < valid_loss_min:
        valid_loss_min = val_loss
        torch.save(fine_tuned_vgg_model.state_dict(), "vgg_cifar10.pt")
        print("Saving improved model...")

# ==========================================================
# 9) FINE-TUNED FEATURE EXTRACTION + LOGREG
# ==========================================================

cifar_fine_tuned_vgg_model = models.vgg11(
    weights=models.VGG11_Weights.IMAGENET1K_V1
)

cifar_fine_tuned_vgg_model.classifier[-1] = nn.Linear(4096, 10)

cifar_fine_tuned_vgg_model.load_state_dict(
    torch.load("vgg_cifar10.pt", map_location=device)
)

cifar_fine_tuned_vgg_model.eval()
cifar_fine_tuned_vgg_model.to(device)

cifar_train_features, cifar_train_labels = extract_vgg_features(
    cifar_fine_tuned_vgg_model, train_loader
)

cifar_test_features, cifar_test_labels = extract_vgg_features(
    cifar_fine_tuned_vgg_model, test_loader
)

print("Fine-Tuned VGG11 + LogReg")
advanced_grid_search(
    cifar_train_features, cifar_train_labels,
    cifar_test_features, cifar_test_labels,
    ml_pipeline, params
)

