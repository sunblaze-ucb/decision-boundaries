This is the code we used in our paper [Decision Boundary Analysis of Adversarial Examples](https://openreview.net/forum?id=BkpiPMbA-).
They are not especially cleaned.

* **gxr3.py**: Measure the decision boundary distances and adjacent classes. We are also publishing the randomly chosen directions we used in the Releases page of this repo.
* **eval_cg.py**: Evaluate images under Cao & Gong's region classification or under point classification.
* **optens_attack.py** Perform the OPTMARGIN attack (MNIST and CIFAR-10 datasets; **attack_v2.py** in ImageNet).
* **classify.py** and **classify_test.py**: Train a classifier on decision boundary distances and adjacent class data and test the classifier.
* **save_orig.py**: Extract images and labels from dataset into Numpy files used in some scripts.
* **save_correctness.py**: Create a compact representation of classification correctness used in some scripts.

For ImageNet, the code expects a copy of [`research/slim/nets` from the TensorFlow models repository](https://github.com/tensorflow/models/tree/master/research/slim/nets) in `imagenet/nets`.
