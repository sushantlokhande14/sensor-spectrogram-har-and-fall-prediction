# Raw Sensor Signal → Spectrogram Image

> How we converted 6-channel inertial data into 224×224 RGB images for ResNet50 and VGG16

---

## Why this matters

ResNet50 and VGG16 expect a **224×224×3 image** as input — they were built for photographs. Our input is a 3-second window of raw accelerometer and gyroscope readings: a `(300, 6)` array of floating-point numbers. Bridging this gap required a custom signal-to-image conversion designed to preserve the motion signatures that distinguish falls from everyday activities.

---

## Pipeline

```
Raw 6-channel window (300 samples × 6 channels @ 100 Hz)
        │
        ├──  acc_x, acc_y, acc_z  (3 channels)
        └──  gyro_x, gyro_y, gyro_z  (3 channels)
        │
        ▼
[Step 1]  STFT per channel
          window = 64 samples, overlap = 48 samples (75%)
          output: 2D time-frequency map per channel
        │
        ▼
[Step 2]  Decibel compression
          X_dB = 20 × log10(|X| + ε),  ε = 1e-8
          clip floor at −40 dB
        │
        ▼
[Step 3]  Per-channel normalization
          rescale to [0, 255] using 2nd–98th percentile
          (robust to outliers; preserves local contrast)
        │
        ▼
[Step 4]  Stack into composite RGB image
          Top half  → accelerometer (R = acc_x, G = acc_y, B = acc_z)
          Bottom half → gyroscope   (R = gyro_x, G = gyro_y, B = gyro_z)
        │
        ▼
[Step 5]  Resize to 224 × 224 pixels
          fed directly to ResNet50 / VGG16 backbone
```

---

## Step-by-step breakdown

### Step 1 — Short-Time Fourier Transform (STFT)

Each of the 6 sensor channels is transformed independently using STFT.

| Parameter | Value | Reason |
|---|---|---|
| Window length | 64 samples | ~0.64 s at 100 Hz; long enough to resolve low-frequency motion |
| Overlap | 48 samples (75%) | High time resolution to catch the brief fall impact |
| Output per channel | 2D array: time × frequency | One time-frequency map per sensor axis |

The STFT slides a short analysis window across the signal and reports the energy at each frequency within that window. The result is a spectrogram: time on the x-axis, frequency on the y-axis, energy as brightness.

```python
from scipy.signal import stft

_, _, Z = stft(signal_channel, fs=100, nperseg=64, noverlap=48)
magnitude = np.abs(Z)
```

### Step 2 — Decibel compression

Raw STFT magnitudes span several orders of magnitude. Direct mapping to pixel values produces near-black images where almost all structure is invisible.

```python
eps = 1e-8
mag_db = 20 * np.log10(magnitude + eps)
mag_db = np.clip(mag_db, -40, None)   # clip noise floor
```

The `−40 dB` floor corresponds to sensor noise. Clipping it sharpens contrast on the actual signal energy.

### Step 3 — Per-channel normalization

Rather than global min/max (which is dominated by outliers), each spectrogram is rescaled using its **2nd–98th percentile**, making contrast consistent across windows regardless of overall signal amplitude.

```python
lo = np.percentile(mag_db, 2)
hi = np.percentile(mag_db, 98)
normalized = np.clip((mag_db - lo) / (hi - lo + 1e-8), 0, 1) * 255
```

### Step 4 — Composite image layout

The 6 normalized spectrograms are packed into a single 3-channel image:

```
┌─────────────────────────────────┐
│  acc_x (R)  acc_y (G)  acc_z (B)│  ← top half: accelerometer
├─────────────────────────────────┤
│ gyro_x (R) gyro_y (G) gyro_z (B)│  ← bottom half: gyroscope
└─────────────────────────────────┘
         Resized to 224 × 224
```

Accelerometer and gyroscope occupy separate halves so the network can learn sensor-specific patterns without channel mixing.

### Step 5 — Resize and feed

The stacked image is resized to `224 × 224` using bilinear interpolation — the exact format expected by both ResNet50 and VGG16 pretrained on ImageNet.

---

## What falls and ADLs look like as spectrograms

| Signal type | Raw signal signature | Spectrogram signature |
|---|---|---|
| **Fall** | Quiet baseline → sharp broadband spike → silence | Bright vertical stripe at impact time |
| **Walking** | Periodic footstep rhythm (~2 Hz) | Repeating vertical bands at regular intervals |
| **Jogging** | Higher-frequency periodic rhythm (~3–4 Hz) | Denser repeating bands, higher frequency row active |
| **Sitting still** | Very low amplitude, near-flat | Dark, uniform, low-energy region |

---

## Why ImageNet pretraining transfers

ResNet50 and VGG16 were trained to recognize edges, textures, and periodic patterns in natural photographs. These **same low-level visual features** are exactly what fall spectrograms contain:

| Visual primitive (ImageNet) | Spectrogram equivalent |
|---|---|
| Edge (sharp intensity boundary) | Fall impact — bright vertical stripe appearing suddenly |
| Periodic texture (repeating pattern) | Walking/jogging — regular vertical band repetition |
| Flat region (uniform low intensity) | Sitting still — dark, featureless spectrogram |

The domain shift is real — spectrograms are not photographs — but the underlying visual vocabulary overlaps enough for transfer to work, particularly for the binary fall detection task.

> **Why VGG16 outperforms ResNet50 here:** Spectrograms are low-resolution and visually simple compared to natural photographs. ResNet50's far greater depth offers no advantage when the texture is this constrained — it overfits on the harder fall-type task. VGG16's simpler architecture is better matched to the signal complexity.

---

## Results

| Model | Binary (fall/ADL) | Fall4 (4-class) | Multi13 (13-class) |
|---|---|---|---|
| ResNet50 | 0.9791 | 0.6505 | 0.7804 |
| VGG16 | **0.9895** | **0.7059** | **0.8261** |
| CNN from scratch | 0.9935 | 0.8062 | 0.9111 |

Spectrogram transfer outperforms UCI HAR transfer on the harder tasks, but the from-scratch CNN trained directly on raw multichannel windows remains the strongest model overall.

---

## Implementation

The full conversion is in [`notebooks/05_transfer_spectrogram_resnet_vgg.ipynb`](notebooks/05_transfer_spectrogram_resnet_vgg.ipynb).

```python
def window_to_spectrogram_image(X_window, fs=100, nperseg=64,
                                noverlap=48, img_size=224, db_floor=-40):
    """
    X_window : (T, 6) raw sensor window — acc_xyz + gyro_xyz
    Returns  : (img_size, img_size, 3) float32 image in [0, 255]
    Layout   : top half = acc channels (RGB = x,y,z)
               bottom half = gyro channels (RGB = x,y,z)
    """
    spectra = []
    for ch in range(6):
        _, _, Z = stft(X_window[:, ch], fs=fs,
                       nperseg=nperseg, noverlap=noverlap)
        mag = np.abs(Z)
        mag_db = 20 * np.log10(mag + 1e-8)
        mag_db = np.clip(mag_db, db_floor, None)
        lo = np.percentile(mag_db, 2)
        hi = np.percentile(mag_db, 98)
        norm = np.clip((mag_db - lo) / (hi - lo + 1e-8), 0, 1) * 255
        spectra.append(norm.astype(np.float32))

    h = img_size // 2
    acc_img = np.stack(
        [cv2.resize(spectra[i], (img_size, h)) for i in range(3)], axis=-1)
    gyro_img = np.stack(
        [cv2.resize(spectra[i], (img_size, h)) for i in range(3, 6)], axis=-1)

    composite = np.concatenate([acc_img, gyro_img], axis=0)
    return composite.astype(np.float32)
```
