# Wheatlo
Wheatlo is an object detector for wheat heads. The image above should give you a clear picture of what it does. It is based on YOLOv3. I created this implementation for Kaggle's Global Wheat Detection competition (https://www.kaggle.com/c/global-wheat-detection/overview), and used its dataset for training and validation. You can download the dataset from the competition's webpage.

## Key Differences from YOLOv3
Wheatlo was created by modifying YOLOv3. If you are not familiar with YOLOv3, I would recommend Ayoosh Kathuria's tutorial on the subject (https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/). It is a nicely written tutorial acompanied by a well-documented implementation of the YOLOv3 object detector.

### Feature Extractor
Wheatlo's feature extractor has the same architecture as Darknet-53 (the feature extractor of YOLOv3), except the final layer has only 2 outputs. It is a binary classifier that tells us if an image has wheat heads in it or not.

#### Train the Feature Extractor
To create a training dataset, I mixed *the wheat head images from the competition's training set* with *COCO 2017 validation set*. The reason I chose COCO was I wanted an image set that
1. does not have images of wheat heads (I assumed COCO does not have any images of wheat heads, even if it does, the number would be small and the resulting error rate would be insignificant)
2. has a wide variety of features.

Besides, Darknet-53 was trained on COCO and it has already learnt the features in the images of COCO. Fine-tuning it with the mixed set I created is basically asking it to learn the features of one more object, the wheat head, on top of what it already knew (or at least that was the effect I presumed it would have).

To train the feature extractor, I initialized it with the weights of Darknet-53. The initial learning rate was 0.001. It is reduced by a factor of 10 every 10 epoches until it reaches 1e-6. I trained  it for 150 epoches. In retrospect, I probably did not have to train it for so many epoches as I was essentially just fine-tuning Darknet-53 for a simpler task. We can see that during the last tens of epoches, the validation error and accuracy were just jumping back and forth in a small neighbourhood.

You can find the code for training the feature extractor in train_extractor.ipynb.

### Detector
Wheatlo has almost identical architecture as YOLOv3. The differences are all in the detection layers and are detailed below:

<ol>
<li>YOLOv3 has 3 detection layers with each one supposedly can handle a particular resolution better than the others. Wheatlo has only 1 detection layer. It is partly due to laziness in implementation and partly because, as I glanced through the training images, there was not that much varations in the sizes of the wheat heads and all the images are of the same resolution, 1024 x 1024.
</li>

<li>YOLOv3 has 9 anchor boxes, and each detection layer uses three of them. They are of the dimensions:

<p align="center">[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]</p>
In wheatlo, there is only 1 detection layer but it uses 6 anchor boxes. They are of the dimensions:

<p align="center">[21,24],  [28,28],  [36,31], [49,34],  [12,23], [76,40]</p>

I chose these dimensions by inspecting the ground truth bounding boxes of the training images. The added options reflect the typical dimensions of the wheat heads in the images.
</li>
<li>In YOLOv3, the detection layer's feature map has a depth of 85 x 3, which corresponds to the 85 attributes of each of the 3 potential bounding boxes. The 85 attributes include the 4 bounding box coordinates, an objectness score, and the class probablities of the 80 objects that YOLOv3 is capable of detecting.

<img src="/images/yolov3_feature_map.jpg" alt="YOLOv3 Feature maps attributes" class="center">

In Wheatlo, since there is only one class, I took away all the class probablities attributies. The detection layer's feature map therefore has a depth of 5 x 6, which corresponds to the 5 attributes of each of the 6 potential bounding boxes.

<img src="/images/wheatlo_feature_map.jpg" alt="Wheatlo Feature maps attributes" class="center">
</li>
</ol>

#### Train the  detector
