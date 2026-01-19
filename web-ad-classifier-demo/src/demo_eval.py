# src/demo_eval.py
import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

#from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from tensorflow.keras.preprocessing.image import load_img  # image loading for examples

SEED = 42
#tf.random.set_seed(SEED)


if hasattr(tf, "random") and hasattr(tf.random, "set_seed"):
    tf.random.set_seed(SEED)
elif hasattr(tf, "set_random_seed"):
    tf.set_random_seed(SEED)
np.random.seed(SEED)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16


def load_demo_dataset(demo_data_dir: str):
    """
    Modern replacement for ImageDataGenerator.flow_from_directory.
    Returns:
      ds: tf.data.Dataset yielding (images, labels)
      class_names: list of class folder names in index order
      file_paths: list of file paths in dataset order (matches shuffle=False)
    """
    # Build dataset (order is deterministic when shuffle=False)
    ds = tf.keras.utils.image_dataset_from_directory(
        demo_data_dir,
        labels="inferred",
        label_mode="binary",          # yields float32 labels shaped (batch, 1)
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    class_names = list(ds.class_names)

    # Build file list in the SAME order TF uses when shuffle=False:
    # alphabetical class folders, and within each, alphabetical file names
    file_paths = []
    for cls in class_names:
        cls_dir = os.path.join(demo_data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")):
                file_paths.append(os.path.join(cls, fname))  # relative path like flow_from_directory

    # Normalize pixels to [0, 1] (replacement for rescale=1./255)
    rescale = tf.keras.layers.Rescaling(1.0 / 255)
    ds = ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds, class_names, file_paths


def evaluate_model(model_path: str, demo_data_dir: str, show_examples: bool = True):
    # 1) Load data
    demo_ds, class_names, filenames = load_demo_dataset(demo_data_dir)
    class_indices = {name: i for i, name in enumerate(class_names)}
    print("Class indices:", class_indices)

    # 2) Load model
    model = tf.keras.models.load_model(model_path)
    model.summary()

    # 3) Evaluate on demo set
    loss, acc = model.evaluate(demo_ds, verbose=0)
    print(f"Demo set accuracy: {acc:.4f}")
    print(f"Demo set loss:     {loss:.4f}")

    # 4) Predictions & metrics
    # Collect true labels from dataset
    y_true = np.concatenate([y.numpy().ravel() for _, y in demo_ds]).astype(int)

    # Predict probabilities
    y_prob = model.predict(demo_ds, verbose=0).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:\n", cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
        print(f"ROC-AUC (demo set): {roc_auc:.4f}")
    except Exception as e:
        print("ROC-AUC could not be computed:", e)

    # 5) Build results dataframe
    idx_to_class = {i: name for name, i in class_indices.items()}
    results_df = pd.DataFrame({
        "filename": filenames,
        "true_label": [idx_to_class[i] for i in y_true],
        "pred_label": [idx_to_class[i] for i in y_pred],
        "prob_Ad": y_prob
    })

    print("\nSample predictions:")
    print(results_df.head(10))

    # 6) Optionally plot a few images with predictions
    if show_examples and len(filenames) > 0:
        from keras.utils import load_img  # works in modern Keras/TensorFlow stacks
        #from tensorflow.keras.utils import load_img

        n_show = min(8, len(filenames))
        plt.figure(figsize=(14, 6))

        for i in range(n_show):
            img_path = os.path.join(demo_data_dir, filenames[i])
            img = load_img(img_path, target_size=IMG_SIZE)
            plt.subplot(2, math.ceil(n_show / 2), i + 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(
                f"T: {results_df.loc[i, 'true_label']}\n"
                f"P: {results_df.loc[i, 'pred_label']} ({results_df.loc[i, 'prob_Ad']:.2f})"
            )

        plt.tight_layout()
        plt.show()

    return results_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo evaluation of Web Ad vs Non-Ad classifier."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/Pretrained_MobileNet_ad_class_with_Aug.h5",
        help="Path to the pretrained Keras model."
    )
    parser.add_argument(
        "--demo-data-dir",
        type=str,
        default="data/demo",
        help="Path to directory with demo images (subfolders per class)."
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting of sample predictions."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        model_path=args.model_path,
        demo_data_dir=args.demo_data_dir,
        show_examples=not args.no_plot
    )
