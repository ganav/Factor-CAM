# train_factor_regularized_keras.py
# Keras implementation of factor-regularized training:
# - classifier trained with CE
# - axis gate trained with KL + entropy
# - hinge, KL, entropy all computed and logged (hinge not backpropagated to classifier)

from __future__ import print_function
import time
import math
import csv
from pathlib import Path
import numpy as np
from keras import backend as K
from keras.optimizers import Adam
from data_concepts import get_keras_generators
from model_backbones import build_backbone
import keras,sys
from keras import layers, models


# ImageNet normalization (must match data_concepts_keras)
MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
STD = np.array([0.229, 0.224, 0.225], dtype="float32")


def soft_topk_mask_fmap(M, keep_ratio=0.20):
    """
    M: [N,H',W'] in [0,1]
    returns mask [N,H',W'] with 1 on top keep_ratio values, 0 elsewhere.
    (Keras/numpy analogue of _soft_topk_mask in Torch version.) 
    """
    N, H, W = M.shape
    flat = M.reshape(N, -1)
    n = flat.shape[1]
    k = max(1, int(round(n * keep_ratio)))
    idx = n - k
    thr = np.partition(flat, idx, axis=1)[:, idx:idx + 1]  # [N,1]
    mask = (flat >= thr).astype("float32")
    return mask.reshape(N, H, W)


def cam_from_fmap(Fmap, W, y):
    """
    Fmap: [N,H',W',C]
    W:    [C,num_classes] (Dense kernel)
    y:    [N] int class ids
    Returns normalized CAM M: [N,H',W'] in [0,1]
    (Analogue of CAMConceptCNN.cam_from_F.) 
    """
    N, H, Wf, C = Fmap.shape
    M = np.zeros((N, H, Wf), dtype="float32")
    for i in range(N):
        cls = int(y[i])
        w = W[:, cls]  # [C]
        M_i = np.tensordot(Fmap[i], w, axes=([2], [0]))  # [H',W']
        M_i = np.maximum(M_i, 0.0)
        mn = M_i.min()
        mx = M_i.max()
        M_i = (M_i - mn) / (mx - mn + 1e-6)
        M[i] = M_i
    return M


def region_pool(Fmap, R_fmap):
    """
    Fmap:   [N,H',W',C]
    R_fmap: [N,H',W'] in {0,1}
    Returns: f_reg [N,C] = weighted average of Fmap over region.
    (Analogue of model.region_pool in Torch.) 
    """
    R = R_fmap[..., None]  # [N,H',W',1]
    num = (Fmap * R).sum(axis=(1, 2))       # [N,C]
    den = R.sum(axis=(1, 2)) + 1e-6         # [N,1]
    return num / den


def upsample_mask(R_fmap, img_size):
    """
    R_fmap: [N,H',W'] -> nearest-neighbor upsample to [N,img_size,img_size]
    """
    N, Hf, Wf = R_fmap.shape
    scale_h = img_size // Hf
    scale_w = img_size // Wf
    R = np.repeat(np.repeat(R_fmap, scale_h, axis=1), scale_w, axis=2)
    return R[:, :img_size, :img_size]


def grayscale_blend(x, R_img):
    """
    x: [N,H,W,3] normalized by MEAN/STD.
    R_img: [N,H,W] mask in {0,1}
    Replace region with grayscale (remove color).
    (Analogue of _grayscale_blend in Torch.) 
    """
    rgb = x * STD[None, None, None, :] + MEAN[None, None, None, :]
    rgb = np.clip(rgb, 0.0, 1.0)
    w = np.array([0.2989, 0.5870, 0.1140], dtype="float32")
    gray = (rgb * w[None, None, None, :]).sum(axis=3, keepdims=True)
    gray3 = np.repeat(gray, 3, axis=3)
    R = R_img[..., None]
    out_rgb = rgb * (1.0 - R) + gray3 * R
    out = (out_rgb - MEAN[None, None, None, :]) / STD[None, None, None, :]
    return out


def box_blur_channel(ch, k):
    pad = k // 2
    padded = np.pad(ch, ((pad, pad), (pad, pad)), mode="reflect")
    H, W = ch.shape
    out = np.zeros_like(ch)
    for i in range(H):
        for j in range(W):
            patch = padded[i:i + k, j:j + k]
            out[i, j] = patch.mean()
    return out


def blur_blend(x, R_img, k=11):
    """
    x: [N,H,W,3] normalized
    R_img: [N,H,W] mask
    Strong blur inside region (remove texture).
    (Analogue of _blur_blend.) 
    """
    rgb = x * STD[None, None, None, :] + MEAN[None, None, None, :]
    rgb = np.clip(rgb, 0.0, 1.0)
    N, H, W, C = rgb.shape
    blurred = np.zeros_like(rgb)
    for n in range(N):
        for c in range(C):
            blurred[n, :, :, c] = box_blur_channel(rgb[n, :, :, c], k)
    R = R_img[..., None]
    out_rgb = rgb * (1.0 - R) + blurred * R
    out = (out_rgb - MEAN[None, None, None, :]) / STD[None, None, None, :]
    return out


def tileshuffle_blend(x, R_img, patch=16):
    """
    x: [N,H,W,3] normalized
    R_img: [N,H,W] mask
    Break shape by shuffling non-overlapping tiles in region.
    (Analogue of _tileshuffle_blend.) 
    """
    N, H, W, C = x.shape
    x_shuf = np.zeros_like(x)
    ph = max(1, min(patch, H))
    pw = max(1, min(patch, W))
    gh = H // ph
    gw = W // pw
    for n in range(N):
        img = x[n]
        tiles = []
        for i in range(gh):
            for j in range(gw):
                tiles.append(img[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw, :])
        tiles = np.array(tiles)
        idx = np.arange(len(tiles))
        np.random.shuffle(idx)
        tiles_shuf = tiles[idx]
        out = np.zeros_like(img)
        t = 0
        for i in range(gh):
            for j in range(gw):
                out[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw, :] = tiles_shuf[t]
                t += 1
        x_shuf[n] = out
    R = R_img[..., None]
    out = x * (1.0 - R) + x_shuf * R
    return out


def make_axis_loss(lambda_kl, lambda_ent):
    """
    Axis loss combining lambda_kl * KL(p || q) + lambda_ent * sum q log q.
    y_true: p (factor distribution); y_pred: q (axis gate probs).
    """
    def axis_loss(y_true, y_pred):
        eps = K.epsilon()
        y_true_c = K.clip(y_true, eps, 1.0)
        y_pred_c = K.clip(y_pred, eps, 1.0)
        kl = K.sum(y_true_c * (K.log(y_true_c) - K.log(y_pred_c)), axis=-1)
        ent = K.sum(y_pred_c * K.log(y_pred_c), axis=-1)  # q log q (negative entropy)
        return lambda_kl * kl + lambda_ent * ent
    return axis_loss

def train_factor_regularized_keras(
    backbone_name = 'none',
    epochs=30,
    img_size=224,
    batch_size=4,
    lr=1e-3,
    keep_ratio=0.20,
    ablate_fraction=0.5,
    margin=0.0,
    lambda_hinge=0.5,
    lambda_kl=0.5,
    lambda_ent=0.01,
    save_dir="artifacts_keras",
):
    """
    Keras analogue of Torch train_factor_regularized: 
    - Classifier trained with CE.
    - Axis gate trained with KL + entropy.
    - Hinge(Lc,Lt,Ls vs L0), KL, entropy all computed and logged.
    NOTE: hinge term is not backpropagated into classifier;
          doing that exactly in TF1+Keras would require a low-level graph.
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_gen, val_gen, num_classes, class_names = get_keras_generators(val_data = "validation_for_train", img_size=img_size, batch_size=batch_size)

    class_model, backbone_model = build_backbone(
        backbone_name=backbone_name,
        num_classes=num_classes,
        input_shape=(img_size, img_size, 3),
    )


    # build axis gate head on region features
    feat_dim = backbone_model.output_shape[-1]   # channels of last conv layer
    feat_in = keras.Input(shape=(feat_dim,), name="feat_in")
    axis_out = layers.Dense(3, activation="softmax", name="axis_gate")(feat_in)
    axis_model = models.Model(feat_in, axis_out, name="axis_gate_head")

    # compile models
    class_model.compile(
        optimizer=Adam(lr=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    axis_model.compile(
        optimizer=Adam(lr=lr),
        loss=make_axis_loss(lambda_kl=lambda_kl, lambda_ent=lambda_ent),
    )

    steps_per_epoch = int(math.ceil(train_gen.samples / float(batch_size)))
    val_steps = int(math.ceil(val_gen.samples / float(batch_size)))

    # CSV logs
    ftr = open(str(save_dir / "train_factor_keras.csv"), "w", newline="", encoding="utf-8-sig")
    fvl = open(str(save_dir / "val_factor_keras.csv"), "w", newline="", encoding="utf-8-sig")
    trw = csv.writer(ftr)
    vlw = csv.writer(fvl)
    trw.writerow([
        "epoch",
        "loss_cls",
        "acc",
        "hinge_mean",
        "kl_mean",
        "ent_mean",
        "total_loss",
        "p_shape_mean",
        "p_color_mean",
        "p_texture_mean",
    ])
    vlw.writerow([
        "epoch",
        "acc",
        "p_shape_mean",
        "p_color_mean",
        "p_texture_mean",
    ])

    start_time = time.time()
    for ep in range(1, epochs + 1):
        # -------- TRAIN --------
        sum_loss_cls = 0.0
        sum_acc = 0.0
        sum_ps = sum_pc = sum_pt = 0.0
        sum_hinge = 0.0
        sum_kl = 0.0
        sum_ent = 0.0
        n_seen = 0
        n_axes = 0  # number of samples used for ablations/factors

        for step in range(steps_per_epoch):
            x_batch, y_batch = next(train_gen)
            B = x_batch.shape[0]
            if B == 0:
                continue

            # (1) classification update (CE)
            out = class_model.train_on_batch(x_batch, y_batch)
            if isinstance(out, (list, tuple)):
                loss_cls = float(out[0])
                acc = float(out[1]) if len(out) > 1 else 0.0
            else:
                loss_cls = float(out)
                acc = 0.0

            # (2) factor distribution via ablations (numpy, no grad)
            Fmap = backbone_model.predict_on_batch(x_batch)  # [B,H',W',C]
            dense_layer = class_model.layers[-1]
            W, b = dense_layer.get_weights()  # W:[C,num_classes]
            M = cam_from_fmap(Fmap, W, y_batch)  # [B,H',W']
            R_fmap = soft_topk_mask_fmap(M, keep_ratio=keep_ratio)  # [B,H',W']

            # choose subset to ablate
            if ablate_fraction < 1.0:
                sel_k = max(1, int(round(B * ablate_fraction)))
                idx_sel = np.random.permutation(B)[:sel_k]
            else:
                idx_sel = np.arange(B)

            x_sel = x_batch[idx_sel]
            y_sel = y_batch[idx_sel].astype("int32")
            R_sel_fmap = R_fmap[idx_sel]

            R_sel_img = upsample_mask(R_sel_fmap, img_size)  # [k,H,W]

            x_nc = grayscale_blend(x_sel, R_sel_img)
            x_nt = blur_blend(x_sel, R_sel_img, k=11)
            x_ns = tileshuffle_blend(x_sel, R_sel_img, patch=16)

            probs0 = class_model.predict_on_batch(x_sel)
            probsc = class_model.predict_on_batch(x_nc)
            probst = class_model.predict_on_batch(x_nt)
            probss = class_model.predict_on_batch(x_ns)

            idx2 = np.arange(len(y_sel))
            L0 = np.log(probs0[idx2, y_sel] + 1e-8)
            Lc = np.log(probsc[idx2, y_sel] + 1e-8)
            Lt = np.log(probst[idx2, y_sel] + 1e-8)
            Ls = np.log(probss[idx2, y_sel] + 1e-8)

            d_color = np.maximum(0.0, L0 - Lc)
            d_texture = np.maximum(0.0, L0 - Lt)
            d_shape = np.maximum(0.0, L0 - Ls)

            # hinge (sum over three ablations, same as Torch version)
            hinge_vec = (
                np.maximum(0.0, Lc - L0 + margin) +
                np.maximum(0.0, Lt - L0 + margin) +
                np.maximum(0.0, Ls - L0 + margin)
            )

            S = d_color + d_texture + d_shape + 1e-6 # d_s + d_c + d_t + ϵ / eq 18
            p = np.stack([d_shape / S, d_color / S, d_texture / S], axis=1)  # [k,3] / eq 18

            # region features for axis gate
            F_sel = Fmap[idx_sel]  # [k,H',W',C]
            f_reg = region_pool(F_sel, R_sel_fmap)  # [k,C]

            # axis update (KL + entropy)
            axis_model.train_on_batch(f_reg, p)
            q = axis_model.predict_on_batch(f_reg)

            #print(backbone_model.summary())
            #print("*************/////////////")
            #print(axis_model.summary())
            #sys.exit()

            # KL and entropy (for logging, raw values, no lambdas)
            eps = 1e-8
            kl_vec = np.sum(p * (np.log(p + eps) - np.log(q + eps)), axis=1)
            ent_vec = np.sum(q * np.log(q + eps), axis=1)

            # accumulators
            sum_loss_cls += loss_cls * B
            sum_acc += acc * B
            sum_ps += p[:, 0].sum()
            sum_pc += p[:, 1].sum()
            sum_pt += p[:, 2].sum()
            sum_hinge += hinge_vec.sum()
            sum_kl += kl_vec.sum()
            sum_ent += ent_vec.sum()
            n_seen += B
            n_axes += len(y_sel)

        # per-sample means
        n_seen_safe = float(max(1, n_seen))
        n_axes_safe = float(max(1, n_axes))

        loss_cls_mean = sum_loss_cls / n_seen_safe
        acc_mean = sum_acc / n_seen_safe
        hinge_mean = sum_hinge / n_axes_safe
        kl_mean = sum_kl / n_axes_safe
        ent_mean = sum_ent / n_axes_safe
        p_shape_mean = sum_ps / n_axes_safe
        p_color_mean = sum_pc / n_axes_safe
        p_texture_mean = sum_pt / n_axes_safe

        total_loss = (
            loss_cls_mean +
            lambda_hinge * hinge_mean +
            lambda_kl * kl_mean +
            lambda_ent * ent_mean
        )

        trw.writerow([
            ep,
            loss_cls_mean,
            acc_mean,
            hinge_mean,
            kl_mean,
            ent_mean,
            total_loss,
            p_shape_mean,
            p_color_mean,
            p_texture_mean,
        ])
        ftr.flush()

        # -------- VAL (no extra training, just metrics like Torch) --------
        sum_acc_v = 0.0
        sum_ps_v = sum_pc_v = sum_pt_v = 0.0
        n_seen_v = 0

        for step in range(val_steps):
            x_batch, y_batch = next(val_gen)
            B = x_batch.shape[0]
            if B == 0:
                continue

            probs = class_model.predict_on_batch(x_batch)
            preds = probs.argmax(axis=1)
            acc_v = float((preds == y_batch.astype("int32")).sum()) / float(B)

            Fmap = backbone_model.predict_on_batch(x_batch)
            dense_layer = class_model.layers[-1]
            W, b = dense_layer.get_weights()
            M = cam_from_fmap(Fmap, W, y_batch)
            R_fmap = soft_topk_mask_fmap(M, keep_ratio=keep_ratio)

            R_img = upsample_mask(R_fmap, img_size)
            x_nc = grayscale_blend(x_batch, R_img)
            x_nt = blur_blend(x_batch, R_img, k=11)
            x_ns = tileshuffle_blend(x_batch, R_img, patch=16)

            probs0 = probs
            probsc = class_model.predict_on_batch(x_nc)
            probst = class_model.predict_on_batch(x_nt)
            probss = class_model.predict_on_batch(x_ns)

            idx2 = np.arange(B)
            y_int = y_batch.astype("int32")
            L0 = np.log(probs0[idx2, y_int] + 1e-8)
            Lc = np.log(probsc[idx2, y_int] + 1e-8)
            Lt = np.log(probst[idx2, y_int] + 1e-8)
            Ls = np.log(probss[idx2, y_int] + 1e-8)

            d_color = np.maximum(0.0, L0 - Lc)
            d_texture = np.maximum(0.0, L0 - Lt)
            d_shape = np.maximum(0.0, L0 - Ls)
            S = d_color + d_texture + d_shape + 1e-6
            p = np.stack([d_shape / S, d_color / S, d_texture / S], axis=1)

            sum_acc_v += acc_v * B
            sum_ps_v += p[:, 0].sum()
            sum_pc_v += p[:, 1].sum()
            sum_pt_v += p[:, 2].sum()
            n_seen_v += B

        n_seen_v_safe = float(max(1, n_seen_v))
        acc_v_mean = sum_acc_v / n_seen_v_safe
        p_shape_mean_v = sum_ps_v / n_seen_v_safe
        p_color_mean_v = sum_pc_v / n_seen_v_safe
        p_texture_mean_v = sum_pt_v / n_seen_v_safe

        vlw.writerow([
            ep,
            acc_v_mean,
            p_shape_mean_v,
            p_color_mean_v,
            p_texture_mean_v,
        ])
        fvl.flush()


        print(
            "ep {}/{} | train acc {:.3f} | val acc {:.3f}".format(
                ep,
                epochs,
                acc_mean,
                acc_v_mean,
            )
        )

        # ---- save checkpoint every 10 epochs ----
        if ep % 10 == 0:
            ckpt_path = save_dir / ("conceptcnn_full_keras_ep%03d.h5" % ep)
            class_model.save(str(ckpt_path))
            print("[train_factor_keras] checkpoint MODEL saved (epoch {}) -> {}".format(ep, ckpt_path))

            axis_weights_path = save_dir / ("axis_gate_keras_final_weights%03d.h5" % ep)
            axis_model.save_weights(str(axis_weights_path))

        # ---- time / ETA ----
        elapsed = time.time() - start_time
        epochs_done = ep
        total_epochs = epochs
        avg_per_epoch = elapsed / float(epochs_done)
        total_estimated = avg_per_epoch * float(total_epochs)
        remaining = total_estimated - elapsed

        def format_seconds(sec):
            sec = int(sec)
            m, s = divmod(sec, 60)
            h, m = divmod(m, 60)
            return "%02dh:%02dm:%02ds" % (h, m, s)

        print(
            "Time: elapsed {} | ETA to finish all epochs {}".format(
                format_seconds(elapsed),
                format_seconds(remaining),
            )
        )

    ftr.close()
    fvl.close()

    final_path = save_dir / backbone_name + "_conceptcnn_full_keras_final.h5"
    class_model.save(str(final_path))
    print("[train_factor_keras] final MODEL saved ->", final_path)
