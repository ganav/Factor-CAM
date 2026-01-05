import cv2
import numpy as np
import tensorflow as tf

# ---------- CONFIG ----------
model_path = "conceptcnn_full_keras_final.h5"
img_path = "89.png"
last_conv_name = "b6_0_project_conv"   # CHANGE THIS
output_path = "cam_result.jpg"
img_size = 224
# ----------------------------

# Load model
model = tf.keras.models.load_model(model_path)

# Load image
img = cv2.imread(img_path)
orig = img.copy()
img = cv2.resize(img, (img_size, img_size))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

# Get last conv layer output
conv_layer = model.get_layer(last_conv_name)
conv_model = tf.keras.Model(model.inputs, conv_layer.output)

# Get classifier weights (Dense layer after GAP)
dense = model.layers[-1]
weights = dense.get_weights()[0]   # shape: (channels, num_classes)

# Forward pass
img = tf.convert_to_tensor(img)

conv_out = conv_model(img)[0]      # (H, W, C)
preds = model(img)
class_idx = tf.keras.backend.eval(tf.argmax(preds[0]))


# Compute CAM
cam = np.zeros(conv_out.shape[:2], dtype=np.float32)
for i in range(weights.shape[0]):
    cam += weights[i, class_idx] * conv_out[:, :, i]

# Normalize CAM
cam = tf.maximum(cam, 0)
cam = cam / tf.reduce_max(cam)

cam = tf.keras.backend.eval(cam)   # convert to NumPy ONCE


# Resize CAM to original image size
cam = cv2.resize(cam, (orig.shape[1], orig.shape[0]))

# Apply heatmap
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

# Save result
cv2.imwrite(output_path, overlay)
