# Image dataset layout for CNN training

To train the fashion CNN, organise images by category — one subfolder per
class, image files inside. The folder names become the class labels.

```
dataset/images/
    ankara_fusion/
        owambe-001.jpg
        owambe-002.jpg
        ...
    streetwear/
        campus-drip-001.jpg
        ...
    casual/
    alte/
    corporate_chic/
```

Guidance:

- **Aim for 80–200 images per class** at minimum. Below ~40/class the model
  will overfit even with augmentation.
- **JPEG or PNG**, any resolution (training resizes to 224x224).
- **Keep classes balanced** — if one class has 500 images and another has 30,
  the model will just predict the majority class.
- **Collect with consent / from public profiles.** Document your data
  collection method in your thesis methodology chapter.

After populating, train with:

```
python train_image_model.py
```

This writes `models/cnn_model.h5`, `models/cnn_labels.json`, and a
`models/cnn_report.txt` with accuracy, precision/recall, and a confusion
matrix you can paste into your thesis.
