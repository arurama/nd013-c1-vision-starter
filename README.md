# Object Detection in an Urban Environment

## Motivation 

Object dectection in a image is a important operation for self driving car. We need to train neutral network with different type of images.In this project we are going to train the model so that it can detect the cars, humans & cyclist.
## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.

### Data

The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/waymo
    - training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (60 files)
    - val: contain the val data (37)
    - test - contains 3 files to test your model and create inference videos
```

The `training_and_validation` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.

You will split this `training_and_validation` data into `train`, and `val` sets by completing and executing the `create_splits.py` file.
```
python create_splits.py --source  /home/workspace/data/waymo/training_and_validation --destination /home/workspace/data/waymo
```
A good rule of thumb is to use something around an 60:30:10 for training:validation:testing split.

### Experiments
The experiments folder will be organized as follow:
```
experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - modification with augmentation
    - label_map.pbtxt
    ...
```

## Prerequisites

### Local Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

### Download and process the data

**Note:** ”If you are using the classroom workspace, we have already completed the steps in the section for you. You can find the downloaded and processed files within the `/home/workspace/data/preprocessed_data/` directory. Check this out then proceed to the **Exploratory Data Analysis** part.

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

You are downloading 100 files (unless you changed the `size` parameter) so be patient! Once the script is done, you can look inside your `data_dir` folder to see if the files have been downloaded and processed correctly.

### Workspace

In the workspace, every library and package should already be installed in your environment. You will NOT need to make use of `gcloud` to download the images.

## Instructions

### Exploratory Data Analysis

[//]: # (Image References)

[imagerandom]: ./images/EDA_img/Random.png "RandomImage"
[imagecount]: ./images/EDA_img/count.png "countImage"
[imagefreq]: ./images/EDA_img/frequencyperset.png "freqImage"
[imageloss]: ./images/Tensor_img/Loss_improve.png "tensorImage"

[imageAugOri]: ./images/Aug_img/ori.png "AugImage"
[imageAugBri]: ./images/Aug_img/brightness_change.png "AugBriImage"
[imageAugCon]: ./images/Aug_img/contras_change.png "AugConImage"
[imageAugFlipHor]: ./images/Aug_img/flip_hor.png "AugFlHorImage"
[imageAugFlipVer]: ./images/Aug_img/flip_ver.png "AugFlverImage"
[imageAugGray]: ./images/Aug_img/gray_img.png "AugGrayImage"

[imageTrainVal]: ./images/Tensor_img/train_val.png "trainImage"

You should use the data already present in `/home/workspace/data/waymo` directory to explore the dataset! 
Here are the some highlights of the exploratory data analysis & deatils analysis is done in Exploratory Data Analysis notebook.
Following are points:
1) Random images 

  ![alt text][imagerandom]

2) There are count of vehicle 3494322 & pedestrian 1044709 & Cyclist 27093

  ![alt text][imagecount]
  
3) Frequency of different classes in dataset

 ![alt text][imagefreq]



### Create the training - validation splits
In the class, we talked about cross-validation and the importance of creating meaningful training and validation splits. For this project, you will have to create your own training and validation sets using the files located in `/home/workspace/data/waymo`. The `split` function in the `create_splits.py` file does the following:
* create three subfolders: `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`
* split the tf records files between these three folders by symbolically linking the files from `/home/workspace/data/waymo/` to `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`

Use the following command to run the script once your function is implemented:
```
python create_splits.py --source  /home/workspace/data/waymo/training_and_validation --destination /home/workspace/data/waymo
```

### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.


####  Reference experiment

For this experiment below model is used :
(ssc_resnet50)
http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz

After training & validate following total loss plot :
 ![alt text][imageTrainVal]

There is lots of noisel total loss signal  training is avg 3.5 & Validation 3.87

### Improve the performances

With the change in the augumentation let see the improvement in model
following are the changes made in config files.
a) random_horizontal_flip b) random_vertical_flip c) random_crop_image
d)random_jpeg_quality e) random_distort_color f) random_adjust_saturation
g) random_adjust_contrast h) random_adjust_brightness i) random_pixel_value_scale
```
data_augmentation_options {
random_horizontal_flip {
      keypoint_flip_permutation: 1
      keypoint_flip_permutation: 0
      keypoint_flip_permutation: 2
      keypoint_flip_permutation: 3
      keypoint_flip_permutation: 5
      keypoint_flip_permutation: 4
      probability: 0.5
    }
  }
  data_augmentation_options {
    random_vertical_flip {
      keypoint_flip_permutation: 1
      keypoint_flip_permutation: 0
      keypoint_flip_permutation: 2
      keypoint_flip_permutation: 3
      keypoint_flip_permutation: 5
      keypoint_flip_permutation: 4
      probability: 0.5
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  data_augmentation_options {
    random_jpeg_quality {
      random_coef: 0.6
      min_jpeg_quality: 50
      max_jpeg_quality: 95
    }
}
 data_augmentation_options {
    random_distort_color {
      color_ordering: 1
    }
}
data_augmentation_options {
    random_adjust_saturation {
      min_delta: 0.75
      max_delta: 1.15
    }
}
data_augmentation_options {
    random_adjust_contrast {
      min_delta: 0.7
      max_delta: 1.1
    }
}
data_augmentation_options {
    random_adjust_brightness {
      max_delta: 0.2
    }
}
data_augmentation_options {
    random_rgb_to_gray {
      probability: 0.8
    }
}
data_augmentation_options {
    random_pixel_value_scale {
      minval: 0.7
      maxval: 1.1
    }
}
```
some examples of augmentation:

Orignal image :

![alt text][imageAugOri]

Coverting to gray 

![alt text][imageAugGray]

Change of Brightness

![alt text][imageAugBri]

Change of Contrast

![alt text][imageAugCon]

Horizontal flip of image

![alt text][imageAugFlipHor] 

Vertical flip of image

![alt text][imageAugFlipVer]

After arumentation changes, we can see the improvement in the model training. There is less noise & overall loss is less.

![alt text][imageloss]
**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path data/waymo/test/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif
```

