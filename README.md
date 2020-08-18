# Wheatlo
Wheatlo is an object detector for wheat heads. The image above should give you a clear picture of what it does. It is based on YOLOv3. I created this implementation for Kaggle's Global Wheat Detection competition (https://www.kaggle.com/c/global-wheat-detection/overview), and used its dataset for training and validation. You can download the dataset from the competition's webpage.

## Key Differences from YOLOv3
Wheatlo was created by modifying YOLOv3. If you are not familiar with YOLOv3, I would recommend Ayoosh Kathuria's tutorial on the subject (https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/). It is a nicely written tutorial acompanied by a well-explained implementation of the YOLOv3 object detector.

### Feature Extractor
Wheatlo's feature extractor has the same architecture as Darknet-53 (the feature extractor of YOLOv3), except the final layer has only 2 outputs. It is a binary classifier that tells us if an image has wheat heads in it or not. 

#### Train the Feature Extractor
To create a training dataset, I mixed *the wheat head images from the competition's training set* with *COCO 2017 validation set*. The reason I chose COCO was I wanted an image set that 
1. does not have images of wheat heads (I assumed COCO does not have any images of wheat heads, even if it does, the number would be small and the resulting error rate would be insignificant)
2. has a wide variety of features.

Besides, Darknet-53 was trained on COCO and it has already learnt the features in the images of COCO. Fine-tuning it with the mixed set I created is basically asking it to learn the features of one more object, the wheat head, on top of what it already knew (or at least that was the effect I presumed it would have).

To train the feature extractor, I initialized it with the weights of Darknet-53. The initial learning rate was 0.001. It is reduced by a factor of 10 every 10 epoches until it reaches 1e-6. I trained  it for 150 epoches. In retrospect, I probably did not have to train it for so many epoches as I was essentially just fine-tuning Darknet-53 for a simpler task.

You can find the code for training the feature extractor in train_extractor.ipynb.

### Detector
