# eval_cam_keras.py
from __future__ import print_function
import tensorflow as tf
from pathlib import Path
import csv
import numpy as np
from keras.optimizers import Adam
import keras,time
from keras import models, layers
from sklearn.metrics import confusion_matrix, classification_report

from data_concepts import get_keras_generators
from model_backbones import build_backbone
from train_factor_regularized import (
    cam_from_fmap,
    soft_topk_mask_fmap,
    region_pool,
    upsample_mask,
    grayscale_blend,
    blur_blend,
    tileshuffle_blend,
    make_axis_loss,
)



def load_val(img_size=224, batch_size=4):
    """
    Keras equivalent of the Torch load_val():
    returns val_gen and class_names.
    Uses data/validation/<class> structure via get_keras_generators.
    """
    # get_keras_generators returns: train_gen, val_gen, num_classes, class_names
    train_gen, val_gen, num_classes, class_names = get_keras_generators(
        root="data",
        val_data="validation",
        img_size=img_size,
        batch_size=batch_size,
    )
    # We only need val_gen + class_names for evaluation
    return val_gen, class_names


def build_backbone_from_class_model(class_model):
    """
    Given a trained classifier model:
        input -> ... -> GAP -> Dense(num_classes)
    reconstruct a backbone model:
        input -> last conv feature map
    by taking the GAP layer's input tensor.
    Assumes the second-to-last layer is GlobalAveragePooling2D.
    """
    gap_layer = class_model.layers[-2]
    fmap_tensor = gap_layer.input  # input to GAP is the conv feature map
    backbone_model = models.Model(class_model.input, fmap_tensor)
    return backbone_model


def build_axis_model(feat_dim, weights_path):
    """
    Rebuild axis gate head with the same architecture as in training
    and load saved weights.
    """
    feat_in = keras.Input(shape=(feat_dim,), name="feat_in")
    axis_out = layers.Dense(3, activation="softmax", name="axis_gate")(feat_in)
    axis_model = models.Model(feat_in, axis_out, name="axis_gate_head")
    axis_model.load_weights(str(weights_path))
    return axis_model


def evaluate_cam_keras(model_path="artifacts_keras/conceptcnn_full_keras_final.h5",
                       out_dir="artifacts_keras",
                       img_size=224,
                       batch_size=4,
                       axis_weights_path="artifacts_keras/axis_gate_keras_final_weights.h5",
                       keep_ratio=0.20):
    """
    Keras version of your Torch evaluate_cam.

    - Loads validation data from data/validation.
    - Loads classifier model (full model .h5 or weights .h5).
    - Computes confusion matrix + classification report.
    - Additionally:
        * Rebuilds backbone and axis gate
        * Computes factor distributions p (from ablations) and q (from axis gate)
        * Reports mean KL(p || q) and mean p/q for shape/color/texture.
    - Saves CSVs (same style as original Torch code) for classification metrics.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- data ---
    val_gen, class_names = load_val(img_size=img_size, batch_size=batch_size)
    num_classes = len(class_names)
    n_samples = val_gen.samples
    y_true = val_gen.classes.copy()  # numpy array of ints

    # --- model: load classifier and backbone ---
    try:
        print("[eval_cam_keras] Trying to load full model from:", model_path)
        class_model = keras.models.load_model(model_path, compile=False)
        print("[eval_cam_keras] Loaded full Keras model.")
        backbone_model = build_backbone_from_class_model(class_model)
    except Exception as e:
        print("[eval_cam_keras] Could not load full model, trying as weights only.")
        print("  Reason:", e)
        # Rebuild architecture (must match training configuration)
        class_model, backbone_model = build_backbone(
            backbone_name="efficientnetb0",   # must match train_factor_regularized_keras
            num_classes=num_classes,
            input_shape=(img_size, img_size, 3),
        )
        class_model.load_weights(model_path)
        print("[eval_cam_keras] Loaded architecture + weights.")

    model = class_model  # for clarity

    # --- axis gate: rebuild and load weights (if available) ---
    axis_weights_path = Path(axis_weights_path)
    compute_factors = axis_weights_path.is_file()
    axis_model = None
    if compute_factors:
        feat_dim = backbone_model.output_shape[-1]
        print("[eval_cam_keras] Loading axis gate weights from:", axis_weights_path)
        axis_model = build_axis_model(feat_dim, axis_weights_path)
    else:
        print("[eval_cam_keras] Axis gate weights not found at:", axis_weights_path)
        print("[eval_cam_keras] Factor evaluation (q, KL) will be skipped.")

    # --- predictions & (optionally) factor evaluation ---
    y_pred = []
    steps = int(np.ceil(n_samples / float(batch_size)))

    # Factor statistics
    sum_ps = sum_pc = sum_pt = 0.0   # from p (ablations)
    sum_qs = sum_qc = sum_qt = 0.0   # from q (axis gate)
    sum_kl = 0.0
    n_axes = 0
    eps = 1e-8



    #----------------------------------------------------------------------------
    lr=1e-3
    keep_ratio=0.20
    ablate_fraction=0.5
    margin=0.0
    lambda_hinge=0.5
    lambda_kl=0.5
    lambda_ent=0.01

    # build axis gate head on region features
    feat_dim = backbone_model.output_shape[-1]   # channels of last conv layer
    feat_in = keras.Input(shape=(feat_dim,), name="feat_in")
    axis_out = layers.Dense(3, activation="softmax", name="axis_gate")(feat_in)
    axis_model = models.Model(feat_in, axis_out, name="axis_gate_head")

    axis_model.compile(
        optimizer=Adam(lr=lr),
        loss=make_axis_loss(lambda_kl=lambda_kl, lambda_ent=lambda_ent),
    )

    from keras.models import load_model
    from flops_counter import count_flops


    #----------------------------------------------------------------------------

    # IMPORTANT: val_gen.shuffle == False, so val_gen.classes order
    # matches the batches we iterate here.
    for _ in range(steps):
        x_batch, y_batch = next(val_gen)
        B = x_batch.shape[0]
        if B == 0:
            continue

        # --- classification predictions ---
        start = time.time()
        probs = model.predict_on_batch(x_batch)
        preds = probs.argmax(axis=1)
        end = time.time()
        print('pred=', end-start)

        input_shape = (224, 224, 3)
        flops = count_flops(model, input_shape)

        print("FLOPs:", flops)
        print("GFLOPs:", flops / 1e9)

        y_pred.extend(preds.astype("int32").tolist())

        # --- factor evaluation (if axis_model available) ---
        if  1 == 1:#compute_factors and axis_model is not None:
            # CAM + region mask
            Fmap = backbone_model.predict_on_batch(x_batch)       # [B,H',W',C]
            dense_layer = model.layers[-1]

            start = time.time()
            W, b = dense_layer.get_weights()                      # W:[C,num_classes]
            M = cam_from_fmap(Fmap, W, y_batch)                   # [B,H',W']
            R_fmap = soft_topk_mask_fmap(M, keep_ratio=keep_ratio)  # [B,H',W']

            # Ablations to get p
            R_img = upsample_mask(R_fmap, img_size)               # [B,H,W]

            # Region features + axis gate: q
            f_reg = region_pool(Fmap, R_fmap)                     # [B,C]
            q = axis_model.predict_on_batch(f_reg)                # [B,3]

            end = time.time()
            print('pred2=', end-start)


            # Accumulate statistics
            sum_qs += q[:, 0].sum()
            sum_qc += q[:, 1].sum()
            sum_qt += q[:, 2].sum()
    

    y_pred = np.array(y_pred[:n_samples], dtype="int32")
    y_true = y_true[:n_samples]

    # --- classification metrics ---
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    total = len(y_true)
    correct = int((y_true == y_pred).sum())
    acc = float(correct) / float(total) if total > 0 else 0.0

    rep = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
    )

    # --- write confusion matrix CSV ---
    cm_path = out / "cam_confusion_matrix_keras.csv"
    with open(str(cm_path), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([""] + class_names)
        for i, row in enumerate(cm):
            w.writerow([class_names[i]] + row.tolist())

    # --- write classification report CSV ---
    rep_path = out / "cam_classification_report_keras.csv"
    with open(str(rep_path), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])

        # per-class rows
        for k in class_names:
            if k in rep:
                d = rep[k]
                w.writerow([
                    k,
                    d.get("precision", 0.0),
                    d.get("recall", 0.0),
                    d.get("f1-score", 0.0),
                    d.get("support", 0),
                ])

        # accuracy row (manually computed)
        w.writerow(["accuracy", "", "", acc, int(cm.sum())])

        # macro avg + weighted avg if present
        for k in ["macro avg", "weighted avg"]:
            if k in rep:
                d = rep[k]
                w.writerow([
                    k,
                    d.get("precision", 0.0),
                    d.get("recall", 0.0),
                    d.get("f1-score", 0.0),
                    d.get("support", 0),
                ])

    print("[CAM-Keras] VAL accuracy: {:.4f}".format(acc))
    print("[CAM-Keras] wrote -> {} and {}".format(cm_path, rep_path))

    # --- factor / axis gate metrics (if available) ---
    if compute_factors and axis_model is not None and n_axes > 0:
        n_axes_safe = float(max(1, n_axes))
        p_shape_mean = sum_ps / n_axes_safe
        p_color_mean = sum_pc / n_axes_safe
        p_texture_mean = sum_pt / n_axes_safe
        q_shape_mean = sum_qs / n_axes_safe
        q_color_mean = sum_qc / n_axes_safe
        q_texture_mean = sum_qt / n_axes_safe
        kl_mean = sum_kl / n_axes_safe

        print("---- Factor / Axis gate stats ----")
        print("p (ablations) mean: shape {:.3f}, color {:.3f}, texture {:.3f}".format(
            p_shape_mean, p_color_mean, p_texture_mean
        ))
        print("q (axis gate) mean: shape {:.3f}, color {:.3f}, texture {:.3f}".format(
            q_shape_mean, q_color_mean, q_texture_mean
        ))
        print("KL(p || q) mean:    {:.4f}".format(kl_mean))


if __name__ == "__main__":
    evaluate_cam_keras()
