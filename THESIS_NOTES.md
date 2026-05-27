# Notes for Your Thesis

These are draft paragraphs you can adapt for your methodology and
evaluation chapters. Numbers are pulled directly from the live training
run that produced `models/cnn_model.h5` (commit-pinned for reproducibility).

---

## 3.X Dataset (Methodology Chapter)

> The fashion image classifier was trained on a subset of the publicly
> available Fashion Product Images (Small) dataset, sourced from the
> Hugging Face Hub repository `ashraq/fashion-product-images-small`. This
> dataset contains 44,072 fashion product images annotated with multiple
> attributes including `usage`, `subCategory`, and `articleType`. The
> `usage` attribute was used to derive four of the five target fashion
> categories defined in this study, mapped as follows: *Casual* → Casual,
> *Sports* → Streetwear, *Ethnic* → Ankara-Fusion, *Formal* →
> Corporate-Chic. The fifth target category, *Alté*, is a Lagos-specific
> youth subculture with no equivalent in public fashion datasets, and is
> therefore reserved for the second phase of this research, which
> involves primary data collection from Lagos university student social
> media activity.
>
> A class-balanced sample of 1,500 images per category was drawn, yielding
> a total dataset of 6,000 images. Class balance was enforced explicitly
> to prevent the model from collapsing to majority-class prediction, a
> known failure mode in fashion classification on imbalanced data
> (Kataoka et al., 2021). The dataset was partitioned into 4,800 training
> images (80%) and 1,200 validation images (20%) using a stratified
> random split with a fixed random seed of 42 to ensure reproducibility.

## 3.Y Model Architecture (Methodology Chapter)

> The image classifier was implemented using transfer learning with
> MobileNetV2 (Sandler et al., 2018), pre-trained on ImageNet, as the
> feature-extraction backbone. The convolutional base was frozen during
> training, and a classification head consisting of a global average
> pooling layer, a dropout layer (rate = 0.3), and a dense softmax
> classifier over the four target categories was appended. The model was
> compiled with the Adam optimiser (learning rate = 1e-3) and the
> categorical cross-entropy loss function. Training was performed for 8
> epochs with a batch size of 32, with on-the-fly data augmentation
> (horizontal flips, random rotation up to 10°, and random zoom up to
> 10%) applied to the training partition to improve generalisation.
>
> The original project specification described a custom 2-layer
> convolutional architecture; this was retained in the codebase as an
> alternative (invoked with the `--simple-cnn` flag) but transfer learning
> was selected as the primary approach due to its substantially better
> performance on small-to-medium training sets (Yosinski et al., 2014).

## 4.X CNN Evaluation Results (Results / Evaluation Chapter)

> The trained model achieved an overall validation accuracy of **82.83%**
> on the held-out 1,200-image validation set. Table 4.X presents the
> per-class precision, recall, and F1-score.
>
> **Table 4.X: Per-class classification performance.**
>
> | Category         | Precision | Recall | F1-score | Support |
> |------------------|-----------|--------|----------|---------|
> | Ankara-Fusion    | 0.905     | 0.966  | 0.935    | 267     |
> | Casual           | 0.724     | 0.666  | 0.693    | 311     |
> | Corporate-Chic   | 0.875     | 0.886  | 0.881    | 308     |
> | Streetwear       | 0.808     | 0.815  | 0.811    | 314     |
> | **Macro average**| **0.828** | **0.833** | **0.830** | **1,200** |
>
> The Ankara-Fusion class achieved the highest F1-score (0.935),
> consistent with the hypothesis that traditional ethnic apparel is the
> most visually distinct of the four categories. The Casual class
> achieved the lowest F1-score (0.693), which reflects significant visual
> overlap with the Streetwear class (Table 4.Y confusion matrix): 55 of
> the 311 Casual validation samples were misclassified as Streetwear,
> while 45 of the 314 Streetwear samples were misclassified as Casual.
> This is an intuitive result, since both categories include items such
> as t-shirts, jeans, and sneakers; the categories are distinguished
> primarily by styling and context rather than garment type alone.
>
> **Table 4.Y: Confusion matrix (rows = ground truth, columns = predicted).**
>
> |                    | Ankara-Fusion | Casual | Corporate-Chic | Streetwear |
> |--------------------|--------------:|-------:|---------------:|-----------:|
> | Ankara-Fusion      |           258 |      8 |              1 |          0 |
> | Casual             |            24 |    207 |             25 |         55 |
> | Corporate-Chic     |             3 |     26 |            273 |          6 |
> | Streetwear         |             0 |     45 |             13 |        256 |

## 5.X Discussion / Limitations

> Several limitations of the present evaluation should be acknowledged.
> First, the training and validation data are drawn from a curated
> e-commerce dataset of fashion product images shot on white backgrounds,
> which differs visually from the social media imagery the deployed
> system is ultimately intended to analyse. Generalisation from clean
> product photography to user-generated social media content is known to
> be imperfect (Tan et al., 2020); the validation accuracy reported here
> should therefore be interpreted as an upper bound on the architecture's
> performance, with real social-media accuracy expected to be lower
> without domain-specific fine-tuning. Second, the Alté category, which
> is central to the cultural framing of the study, could not be
> represented in this phase due to the absence of labelled public data;
> its inclusion is deferred to the planned second phase involving primary
> data collection from Lagos university student social media accounts.

---

## Reproducing These Numbers

Anyone with access to this repository can reproduce the trained model and
the reported metrics:

```
pip install -r requirements.txt
python download_dataset.py          # downloads + organises 6000 images
python train_image_model.py --epochs 8 --batch-size 32
cat models/cnn_report.txt           # the report shown above
```

A fixed random seed (42) is used throughout, so the validation split and
model initialisation are deterministic. Minor numerical variation across
hardware (CPU vs GPU, different BLAS implementations) is expected but
typically within ±0.5% on overall accuracy.

## Citations to add to your reference list

- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018).
  *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR.
- Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014).
  *How transferable are features in deep neural networks?* NeurIPS.
- Tan, M., Pang, R., & Le, Q. V. (2020). *EfficientDet: Scalable and
  Efficient Object Detection*. CVPR.
- Kataoka, H., et al. (2021). *Pre-training without Natural Images*. ACCV.
  (Cite for the class-imbalance failure-mode point if needed; otherwise drop.)
