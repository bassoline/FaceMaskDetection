# FaceMaskDetection
Face Mask Detection using YoloV3 and YoloV4

### Papers 
[YoloV3](https://arxiv.org/abs/1506.02640) 

[YoloV4](https://arxiv.org/abs/2004.10934) 

### About 
All training was done using Google Colab's GPUs, details about what GPU was used are available in the log files

### Getting Started 
It is best to follow along with AlexeyAB's repo found [here](https://github.com/AlexeyAB/darknet). 

To train from scratch you will need to download the weights for either [YoloV3](https://pjreddie.com/media/files/darknet53.conv.74) or [YoloV4](https://drive.google.com/open?id=1JKF-bdIklxOOVy-2Cr5qdvjgGpmGfcbp). 

To use the weights from my traing set you can find the YoloV3 weights [here](https://drive.google.com/file/d/1-MU044HqsE97e779T1INn3nZ8C71CRkZ/view?usp=sharing) and YoloV4 weights [here](https://drive.google.com/file/d/1Y50u8HiL-veZlvNg1ynhX2RnQc96wd0R/view?usp=sharing). 

You will also need to download the test data, a zip of it can be found [here](https://www.dropbox.com/s/6gewe947ake1g95/kaggle_and_no-mask_dataset.zip?dl=1) 

Once you have everything downloaded - you can just update the file paths I already have in the folders yolov(3/4). For example the yolov3-face_mask-setup.data file, it references everything from darknet, so make sure the file references are based on that.  

Then you need to make sure all the file paths are also correct when running or testing the darknet command line script. 

I.e: this was what I used based on my file locations, they will need to be updated depending on your set up (note colab also timed out on me so you can see that I was reusing the last.weights instead of the downloaded weights for yoloV4)

```
!./darknet detector train ../FaceMask_Code/yolov4/yolov4-face-mask-setup.data ../FaceMask_Code/yolov4/yolov4-face-mask-train.cfg ../FaceMask_Code/yolov4/backup/yolov4-face-mask-train_last.weights -dont_show -map 2> ../FaceMask_Code/yolov4/train_log.txt
```

### Config files 

The following descriptions are based on the OpenCV Course 2 material you can get course information [here](https://opencv.org/courses/#:~:text=AI%20Courses%20by%20OpenCV.%201%20Atanas%20Rogachev.%20Senior,4%20Leo%20Quiroa.%205%20Ikbel%20Boulabiar.%20More%20items)

#### Batch 
"When the batch size is set to 64, it means 64 images are used in one iteration to update the parameters of the neural network."
#### Subdivisions
"Even though you may want to use a batch size of 64 for training your neural network, you may not have a GPU with enough memory to use a batch size of 64. Fortunately, Darknet allows you to specify a variable called subdivisions that lets you process a fraction of the batch size at one time on your GPU."
#### Momentum
"Momentum is used to penalize large weight changes between iterations." 
#### Decay 
"One of the ways to mitigate this(over fitting) problem is to penalize large value for weights. The parameter decay controls this penalty term."
#### Learning Rate/Steps/burn_in/Scale
"The parameter learning rate controls how aggressively we should learn based on the current batch of data. Typically this is a number between 0.01 and 0.0001."

At the beginning of the training process, we are starting with very generic information so the learning rate should be high. As the neural network sees more data, the weights should change less aggressively hence, the learning rate should decrease over time. In the configuration file, this decrease in learning rate is accomplished by first specifying that our learning rate decreasing policy is steps. The learning rate will start from 0.001 and remain constant for 1000 iterations (burn_in), and then it will multiply by scales to get the new learning rate.

Even though the learning rate should be high in the beginning and low later on - it has been found that the training speed tends to increase if we have a lower learning rate for a short period of time at the very beginning. This is controlled by the burn_in parameter. 

#### Data Augmentation (angle/saturation/exposure/hue)
Quality data can be hard to come by and labeling the data is time consuming so we can use Darknet's built in data augmentation to change the image slightly in order to make our predictions more robust. 

#### YoloV3 Config
For YoloV3 the config parameters were set to

```
batch=16
subdivisions=8
width=608
height=608
channels=3
momentum=0.9
decay=0.0005
angle=10
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches=10000
policy=steps
steps=7000,8000,9000
scales=.1,.1,.1
```

The batch size was set to 16 by default from the OpenCV Course - I left it as is. Possibly changing it to 64 may have helped with accuracy. Width and height were both increased to 608 to improve the accuracy. Also updating the angle to 10 also helped improve the accuracy. I had to set the max_batch to 10000 as I was not getting great results with YoloV3 and it needed extra time to train - also that is why I manually set the steps to such high numbers to slow down the training and hopefully improve the accuracy.

#### YoloV4 Config
For YoloV4 I changed the training config parameters

```
batch=64
subdivisions=64
width=608
height=608
```

The GPU from Google Colab was crashing with any subdivision less than 64 so I had no choice on this one. 
When training YoloV3 and reading AlexeyAB's documentation, it seemed that larger images would lead to more accuracy, and 608 pixels worked out well for me. 

For YoloV4 I also updated the following

```
burn_in=1000
max_batches = 6000
policy=steps
steps=5400
scales=.1
```

The burn in of 1000 was the default and it worked well, then I just followed the 6000 minimum max_batches rule and the 90% rule for the steps that were laid out in the documentation. 

The only other things that were changed were for the class size and filters that are dependent on the class sizes

```
[convolutional]
size=1
stride=1
pad=1
filters=21 # <- here 
activation=linear


[yolo]
mask = 6,7,8
anchors = 12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401
classes=2 # <- AND here 
``` 

### Results 
With YoloV3 I was able to get 84% accuracy. 

With YoloV4 I was able to get about 89% accuracy - though I was lazy and stopped training after only around 1300 batches for YoloV4 as the results were good enough for this project. If I had training longer it is possible it may have been a few percent higher. 

You can check the .png files for these percentages. 
