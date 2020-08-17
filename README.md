# Wheatlo
Wheatlo is an object detector for wheat head. The image above should give you a clear picture of what it does. It is based on YOLOv3. I created this project for Kaggle's Global Wheat Detection (https://www.kaggle.com/c/global-wheat-detection/overview) competition, and used its dataset for training and validation. You can download the dataset from the competition's webpate.

## Key Differences from YOLOv3
Wheatlo was created by modifying YOLOv3. If you are not familiar with YOLOv3, I would recommend Ayoosh Kathuria's tutorial on the subject (https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/).

### Feature Extractor
Wheatlo's feature extractor has the same structure as Darknet-53, except the final layer has only 2 outputs. It is a binary classifier that tells us if an image has wheat heads in it or not.


### Detector
