# Glaucoma Detection Using Deep Learning, XAI, and VLM

This repository contains Python code files developed in Google Colab for processing, analyzing, and interpreting glaucoma-related image datasets. Below is a summary of each file's purpose and key functionality.

## 01_Glaucoma_Datasets_Unification.py
**Purpose**: Unifies multiple glaucoma image datasets into a single dataset with a consistent structure and metadata.

**Key Functionality**:
- Mounts Google Drive and sets up paths for dataset processing.
- Extracts five zip files (DRISHTI, RIMONE, ACRIMA, REFUGE, EyePACS) containing glaucoma and normal retinal images.
- Creates a unified dataset directory (unified_dataset/images) and copies images from each dataset, renaming them with a dataset-specific prefix.
- Assigns binary labels (1 for Glaucoma, 0 for Normal) based on dataset folder structures.
- Generates a labels.csv file with metadata (image path, label, dataset name) for all images.
- Zips the unified dataset and moves it to Google Drive.

**Output**:
- A unified_dataset.zip file containing all images and a labels.csv file, stored in /content/drive/MyDrive/GlaucomaProject.

## 02_Glaucoma_Classification_Using_CNNs and 03_Glaucoma_Classification_Using_Transformers.py
**Purpose**: Trains and evaluates CNN and Transformer models for glaucoma classification using the unified dataset.

**Key Functionality**:
- Loads the unified dataset (labels.csv) and splits it into training (80%) and validation (20%) sets with stratification.
- Applies image preprocessing (CLAHE, resizing, augmentation) using albumentations.
- Defines a custom GlaucomaDataset class and creates DataLoader for training and validation.
- Implements five models: ResNet50, EfficientNet-B0, DenseNet121, ViT-Base, and Swin Transformer, with pretrained weights.
- Trains models with AdamW optimizer, cross-entropy loss, and ReduceLROnPlateau scheduler, using early stopping (patience=5).
- Evaluates models on accuracy, precision, recall, F1-score, and AUC, saving the best model based on validation F1-score.
- Visualizes training metrics (loss, accuracy, etc.) and normalized confusion matrices.
- Generates Grad-CAM (for CNNs) and Attention Rollout (for Transformers) visualizations for interpretability on random validation images.
- Saves final metrics to CSV and plots to PDF.

**Output**:
- Trained model weights (<model_name>_best.pth).
- Metrics plots (<model_name>_metrics.pdf).
- Final metrics CSV (<model_name>_final_metrics.csv).
- Visualizations of Grad-CAM/Attention Rollout for selected models.

## 04_Glaucoma_Explanability_Using_CNNs.py
**Purpose**: Generates Grad-CAM visualizations for CNN models to interpret glaucoma classification results.

**Key Functionality**:
- Loads the unified dataset (labels.csv) and applies preprocessing (CLAHE, resizing, normalization) using albumentations.
- Defines a GlaucomaDataset class for loading images and labels.
- Implements Grad-CAM for three CNN models (ResNet50, EfficientNet-B0, DenseNet121) to highlight regions influencing predictions.
- Loads pretrained model weights from specified paths (<model_name>_best.pth).
- Generates Grad-CAM visualizations for:
  - A randomly selected image from the dataset.
  - Ten specific images (g0001.jpg, g0002.jpg, g0003.jpg, V0344.jpg, V0347.jpg, n0001.jpg, n0002.jpg, n0003.jpg, n0204.jpg, n0205.jpg).
- Displays predictions, ground truth, and confidence scores alongside Grad-CAM heatmaps overlaid on original images.

**Output**:
- Grad-CAM visualizations for specified images, showing model attention areas for each CNN model.

## 05_Glaucoma_Explanability_Using_Transformers.py
**Purpose**: Generates Attention Rollout visualizations for Transformer models to interpret glaucoma classification results.

**Key Functionality**:
- Loads the unified dataset (labels.csv) and applies preprocessing (resizing to 224x224, normalization) using albumentations.
- Defines a GlaucomaDataset class for loading images and labels.
- Implements Attention Rollout for ViT-Base and Swin Transformer models to visualize attention patterns.
- Loads pretrained model weights (vit_base_patch16_224_best.pth, swin_base_patch4_window7_224_best.pth).
- Patches attention modules to capture attention matrices during forward passes.
- Generates Attention Rollout visualizations for ten specific images (g0001.jpg, g0002.jpg, g0003.jpg, V0344.jpg, V0347.jpg, n0001.jpg, n0002.jpg, n0003.jpg, V0399.jpg, V0400.jpg for ViT; similar set for Swin, with n0204.jpg, n0205.jpg instead of V0399.jpg, V0400.jpg).
- Displays predictions, ground truth (1 for glaucoma, 0 for normal), and attention heatmaps overlaid on original images.
- Handles Swin Transformer’s masked attention with custom forward hooks and robust normalization.

**Output**:
- Attention Rollout visualizations for specified images, showing attention areas for ViT-Base and Swin Transformer models.

## 06_Glaucoma_Ensemble_GradCAM.py
**Purpose**: Performs ensemble prediction and generates Grad-CAM visualizations for CNN models on a user-uploaded fundus image.

**Key Functionality**:
- Installs pytorch-grad-cam for Grad-CAM visualizations.
- Loads five pretrained models (ResNet50, EfficientNet-B0, DenseNet121, ViT-Base, Swin Transformer) from specified paths.
- Defines a custom PatchedDenseNet class to fix DenseNet121 compatibility with Grad-CAM.
- Accepts a user-uploaded fundus image and preprocesses it for two resolutions (512x512 for CNNs, 224x224 for Transformers).
- Performs ensemble prediction by averaging probabilities from all models, selecting the class (Normal or Glaucoma) with the highest average probability.
- Generates Grad-CAM visualizations for CNN models (ResNet50, EfficientNet-B0, DenseNet121) to highlight regions influencing predictions.
- Combines the original image and Grad-CAM overlays into a single image with labels for each model.
- Saves individual and combined visualizations, along with zero-shot, one-shot, and few-shot prompts for further analysis (e.g., with LLaVA).
- Stores metadata (image ID, prediction, probabilities) and zips all outputs.

**Output**:
- Combined image (combined_gradcam.jpg) showing original image and Grad-CAM overlays for ResNet50, EfficientNet-B0, and DenseNet121.
- Individual Grad-CAM images (<img_id>_<model_name>_cam.jpg).
- Prompt files (prompt_zero_shot.txt, prompt_one_shot.txt, prompt_few_shot.txt) for clinical interpretation.
- Metadata file (metadata.json) and prediction summary (ensemble_prediction.txt).
- Zipped outputs (llava_inputs.zip).

## 07_Glaucoma_Independent_Evaluation.py
**Purpose**: Evaluates the performance of individual models and an ensemble on an independent test dataset (drishti_test.zip) for glaucoma classification.

**Key Functionality**:
- Unzips the DRISHTI test dataset and creates a DataFrame with image paths and labels (Normal: 0, Glaucoma: 1).
- Loads five pretrained models (ResNet50, EfficientNet-B0, DenseNet121, ViT-Base, Swin Transformer) with appropriate transforms (512x512 for CNNs, 224x224 for Transformers).
- Defines a custom PatchedDenseNet class to ensure compatibility with DenseNet121.
- Evaluates each model on the test dataset, computing accuracy, precision, recall, F1-score, and AUC.
- Performs ensemble prediction by averaging probabilities across all models.
- Visualizes results with:
  - Confusion matrices for each model and the ensemble.
  - ROC curves for each model and the ensemble.
  - Bar plot comparing model accuracies.
- Generates a summary table of performance metrics (accuracy, precision, recall, F1-score, AUC) sorted by accuracy.

**Output**:
- Confusion matrix plots for each model and the ensemble.
- ROC curve plot comparing all models and the ensemble.
- Bar plot of model accuracies.
- Summary table (results_df) with performance metrics for all models and the ensemble.

## 08_Glaucoma_Explanability_Gemini.py
**Purpose**: Generates a simulated ophthalmologist report using Gemini AI to interpret Grad-CAM visualizations for glaucoma classification on a user-uploaded fundus image.

**Key Functionality**:
- Installs and configures google-generativeai for Gemini AI integration.
- Loads five pretrained models (ResNet50, EfficientNet-B0, DenseNet121, ViT-Base, Swin Transformer) and defines a PatchedDenseNet class for DenseNet121 compatibility.
- Accepts a user-uploaded fundus image, applies CLAHE preprocessing, and transforms it for two resolutions (512x512 for CNNs, 224x224 for Transformers).
- Performs ensemble prediction by averaging probabilities across all models to determine the class (Normal or Glaucoma) and confidence score.
- Generates a Grad-CAM visualization for DenseNet121 to highlight regions influencing the prediction.
- Combines the original image and Grad-CAM overlay into a single image (merged_densenet121.jpg).
- Uses Gemini AI (gemini-1.5-flash) to analyze the Grad-CAM overlay, focusing on glaucomatous signs (e.g., cup-to-disc ratio, rim thinning, optic disc hemorrhages).
- Generates a simulated ophthalmologist report with image ID, model prediction, confidence, Gemini's explanation, and final verdict (Likely Glaucomatous or Non-Glaucomatous).
- Cleans Gemini's output to remove markdown and excessive newlines for readability.

**Output**:
- Combined image showing the original fundus image and DenseNet121 Grad-CAM overlay.
- Simulated ophthalmologist report with clinical interpretation and disclaimer, printed to the console.
