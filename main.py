import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Function to compute GLCM for a patch
def compute_glcm_patch(patch, distances, angles):
    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    glcm = greycomatrix(patch_gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    return glcm


# Function to extract features from a precomputed GLCM
def extract_features_from_glcm(glcm):
    # Compute texture features using GLCM
    contrast = greycoprops(glcm, 'contrast').ravel()
    dissimilarity = greycoprops(glcm, 'dissimilarity').ravel()
    homogeneity = greycoprops(glcm, 'homogeneity').ravel()
    energy = greycoprops(glcm, 'energy').ravel()
    correlation = greycoprops(glcm, 'correlation').ravel()

    # Concatenate all features into a single feature vector
    features = np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation])
    return features


# Function to split an image into patches
def split_image(image, size):
    height, width = image.shape[:2]
    slices = []
    for i in range(0, height - size + 1, size):
        for j in range(0, width - size + 1, size):
            slice_img = image[i:i + size, j:j + size]
            slices.append((slice_img, (i, j)))
    return slices


# Function to prepare data for training
def prepare_data(image_list, mask_list, slice_size, distances, angles):
    X = []
    y = []
    for image, mask in zip(image_list, mask_list):
        slices = split_image(image, slice_size)
        masks = split_image(mask, slice_size)
        glcm_cache = {}

        for i, (patch, (y_pos, x_pos)) in enumerate(slices):
            # Compute GLCM if not already computed for this patch
            if (y_pos, x_pos) not in glcm_cache:
                glcm = compute_glcm_patch(patch, distances, angles)
                glcm_cache[(y_pos, x_pos)] = glcm
            else:
                glcm = glcm_cache[(y_pos, x_pos)]

            features = extract_features_from_glcm(glcm)
            label = masks[i][0][slice_size // 2, slice_size // 2]  # Decision for the center pixel
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


# Load and preprocess the images and masks
img_index = 7
new_width = int(3504 * 0.25)
new_height = int(2336 * 0.25)

# Read, resize, and normalize the mask images
image_mask_list = [cv2.resize(cv2.imread(filename, 0), (new_width, new_height)) // 255
                   for filename in sorted(glob.glob('mask/*.tif'))]

# Read, resize the original images, and apply the corresponding mask
image_list = [cv2.resize(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB), (new_width, new_height)) *
              np.dstack([mask, mask, mask])
              for filename, mask in zip(sorted(glob.glob('images/*.jpg')), image_mask_list)]

# Read and resize the labeled images
image_labeled_list = [cv2.resize(cv2.imread(filename, 1), (new_width, new_height))
                      for filename in sorted(glob.glob('manual1/*.tif'))]

# Define distances and angles for GLCM
distances = [1]
angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

# Prepare the data
slice_size = 5  # Size of the patches
X, y = prepare_data(image_list, image_labeled_list, slice_size, distances, angles)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier (Random Forest Classifier in this case)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Test the classifier
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Predict and display the mask for a given image
def predict_image(image, clf, slice_size, distances, angles):
    height, width = image.shape[:2]
    mask_pred = np.zeros((height, width), dtype=np.uint8)

    slices = split_image(image, slice_size)
    glcm_cache = {}

    for patch, (y_pos, x_pos) in slices:
        # Compute GLCM if not already computed for this patch
        if (y_pos, x_pos) not in glcm_cache:
            glcm = compute_glcm_patch(patch, distances, angles)
            glcm_cache[(y_pos, x_pos)] = glcm
        else:
            glcm = glcm_cache[(y_pos, x_pos)]

        features = extract_features_from_glcm(glcm)
        pred_label = clf.predict([features])[0]
        mask_pred[y_pos + slice_size // 2, x_pos + slice_size // 2] = pred_label * 255

    return mask_pred


# Example usage
test_image = image_list[0]  # Using the first image in the list as an example
predicted_mask = predict_image(test_image, clf, slice_size, distances, angles)

# Display the predicted mask
plt.imshow(predicted_mask, cmap='gray')
plt.title("Predicted Vessels Mask")
plt.axis('off')
plt.show()
