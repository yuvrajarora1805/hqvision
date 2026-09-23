import os
import glob
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pennylane as qml

# --- 1. CONFIGURATION ---
DATA_DIR = "data"
PATCH_SIZE = 4
NUM_FEATURES = 4
NUM_QUBITS = 4
MAX_CANDIDATES_PER_IMAGE = 5

# --- 2. DATA LOADING & PREPROCESSING ---
def parse_xml_and_extract_patches(xml_file, img_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except:
        return [], []
    
    # Check calcification type
    calc_elem = root.find(".//calcifications")
    if calc_elem is None or calc_elem.text is None:
        return [], []
    
    calc_type = calc_elem.text.strip().lower()
    if calc_type == "microcalcifications":
        label = 1
    elif calc_type == "non":
        label = 0
    else:
        return [], [] # Skip 'macro' or others for now
        
    import json
    # Get bounding box from polygon inside <svg>
    svgs = root.findall(".//svg")
    x_coords = []
    y_coords = []
    for svg_elem in svgs:
        if svg_elem.text:
            try:
                data = json.loads(svg_elem.text)
                for region in data:
                    for p in region.get("points", []):
                        x_coords.append(int(p["x"]))
                        y_coords.append(int(p["y"]))
            except:
                pass
            
    if not x_coords or not y_coords:
        return [], []
        
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Load Image
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [], []
        
    # Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(img)
    
    # Top-Hat Filter to find bright spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    tophat = cv2.morphologyEx(enhanced_img, cv2.MORPH_TOPHAT, kernel)
    
    # Focus only on the nodule bounding box
    nodule_mask = np.zeros_like(img)
    cv2.fillPoly(nodule_mask, [np.array(list(zip(x_coords, y_coords)))], 255)
    
    masked_tophat = cv2.bitwise_and(tophat, tophat, mask=nodule_mask)
    
    # Find local maxima (brightest spots)
    _, binary = cv2.threshold(masked_tophat, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    patches = []
    labels = []
    
    # Sort contours by area or peak intensity, just take first few for simplicity
    for cnt in contours[:MAX_CANDIDATES_PER_IMAGE]:
        # Get center
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = cnt[0][0]
            
        # Crop 4x4 patch at full resolution
        half_size = PATCH_SIZE // 2
        # Ensure bounds
        if cy - half_size >= 0 and cy + half_size < img.shape[0] and cx - half_size >= 0 and cx + half_size < img.shape[1]:
            # Use original image or enhanced image for the patch?
            # Using enhanced image to retain contrast.
            patch = enhanced_img[cy-half_size:cy+half_size, cx-half_size:cx+half_size]
            patches.append(patch.flatten())
            labels.append(label)
            
    return patches, labels

print("Loading and processing dataset...")
xml_files = glob.glob(os.path.join(DATA_DIR, "*.xml"))
X_raw = []
y = []

for xml_file in xml_files:
    base_name = os.path.basename(xml_file).replace(".xml", "")
    # Find matching images (could be multiple like 106_1.jpg, 106_2.jpg)
    img_files = glob.glob(os.path.join(DATA_DIR, f"{base_name}_*.jpg"))
    for img_file in img_files:
        patches, labels = parse_xml_and_extract_patches(xml_file, img_file)
        X_raw.extend(patches)
        y.extend(labels)

X_raw = np.array(X_raw)
y = np.array(y)

print(f"Extracted {len(X_raw)} total patches. Class distribution: {np.bincount(y) if len(y) > 0 else '0'}")

if len(X_raw) == 0:
    print("No valid patches extracted. Check the XML formatting and images.")
    exit()

# --- 3. DIMENSIONALITY REDUCTION (PCA) ---
print("Applying PCA to compress 16 pixels to 4 features...")
pca = PCA(n_components=NUM_FEATURES)
X_compressed = pca.fit_transform(X_raw)

# Normalize for angle encoding (scale between 0 and pi)
X_min = np.min(X_compressed, axis=0)
X_max = np.max(X_compressed, axis=0)
X_norm = np.pi * (X_compressed - X_min) / (X_max - X_min + 1e-8)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {len(X_train)} samples. Test set: {len(X_test)} samples.")


# --- 4. QUANTUM KERNEL DEFINITION ---
print("Setting up PennyLane Quantum Kernel...")
dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev, interface="autograd")
def kernel_circuit(x1, x2):
    # Amplitude/Angle Encoding for x1
    qml.AngleEmbedding(x1, wires=range(NUM_QUBITS))
    # Non-linear entangling map
    qml.StronglyEntanglingLayers(weights=np.zeros((1, NUM_QUBITS, 3)), wires=range(NUM_QUBITS))
    
    qml.adjoint(qml.StronglyEntanglingLayers)(weights=np.zeros((1, NUM_QUBITS, 3)), wires=range(NUM_QUBITS))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(NUM_QUBITS))
    
    return qml.probs(wires=range(NUM_QUBITS))

def q_kernel(x1, x2):
    # The probability of measuring |0000> is the kernel value
    return kernel_circuit(x1, x2)[0]

def q_kernel_matrix(A, B):
    """Computes the quantum kernel matrix between datasets A and B."""
    matrix = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            matrix[i, j] = q_kernel(a, b)
    return matrix

# Note: Quantum kernel computation can be slow. 
# We'll use a subset if there are too many samples.
MAX_TRAIN_SAMPLES = 100
MAX_TEST_SAMPLES = 30

if len(X_train) > MAX_TRAIN_SAMPLES:
    print(f"Subsampling training set to {MAX_TRAIN_SAMPLES} for faster execution.")
    X_train = X_train[:MAX_TRAIN_SAMPLES]
    y_train = y_train[:MAX_TRAIN_SAMPLES]
    
if len(X_test) > MAX_TEST_SAMPLES:
    X_test = X_test[:MAX_TEST_SAMPLES]
    y_test = y_test[:MAX_TEST_SAMPLES]

print("Computing Quantum Kernel Matrix for Training...")
K_train = q_kernel_matrix(X_train, X_train)

# --- 5. TRAINING AND EVALUATION ---
print("Training Quantum Support Vector Classifier (QSVC)...")
svc = SVC(kernel="precomputed")
svc.fit(K_train, y_train)

print("Computing Quantum Kernel Matrix for Testing...")
K_test = q_kernel_matrix(X_test, X_train)

print("Evaluating...")
y_pred = svc.predict(K_test)

print("\n--- RESULTS ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
