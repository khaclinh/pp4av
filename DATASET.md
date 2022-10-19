---
annotations_creators:
- expert-generated
language_creators:
- found
language:
- en
license:
- cc-by-nc-nd-4.0
multilinguality:
- monolingual
size_categories:
- 1K<n<10K
source_datasets:
- extended
task_categories:
- object-detection
task_ids:
- face-detection
- license-plate-detection
pretty_name: PP4AV
---

# Dataset Card for PP4AV

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Languages](#languages)
- [Dataset Creation](#dataset-creation)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Dataset folder](#folder)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://github.com/khaclinh/pp4av
- **Repository:** https://github.com/khaclinh/pp4av
- **Paper:** [PP4AV: A benchmarking Dataset for Privacy-preserving Autonomous Driving]
- **Point of Contact:** linhtk.dhbk@gmail.com

### Dataset Summary

PP4AV is the first public dataset with faces and license plates annotated with driving scenarios. P4AV provides 3,447 annotated driving images for both faces and license plates. For normal camera data, dataset sampled images from the existing videos in which cameras were mounted in moving vehicles, running around the European cities. The images in PP4AV were sampled from 6 European cities at various times of day, including nighttime. This dataset use the fisheye images from the WoodScape dataset to select 244 images from the front, rear, left, and right cameras for fisheye camera data. PP4AV dataset can be used as a benchmark suite (evaluating dataset) for data anonymization models in autonomous driving.

### Languages

English


## Dataset Creation

### Source Data

#### Initial Data Collection and Normalization

The objective of PP4AV is to build a benchmark dataset that can be used to evaluate face and license plate detection models for autonomous driving. For normal camera data, we sampled images from the existing videos in which cameras were mounted in moving vehicles, running around the European cities. We focus on sampling data in urban areas rather than highways in order to provide sufficient samples of license plates and pedestrians. The images in PP4AV were sampled from **6** European cities at various times of day, including nighttime. The source data from 6 cities in European was described as follow:
  - `Paris`: This subset contains **1450** images of the car driving down a Parisian street during the day. The video frame rate is 30 frames per second. The video is longer than one hour. We cut a shorter video for sampling and annotation. The original video can be found at the following URL:
    URL: [paris_youtube_video](https://www.youtube.com/watch?v=nqWtGWymV6c)  
  - `Netherland day time`: This subset consists of **388** images of Hague, Amsterdam city in day time. The image of this subset are sampled from the bellow original video:  
    URL: [netherland_youtube_video](https://www.youtube.com/watch?v=Xuo4uCZxNrE)  
    The frame rate of the video is 30 frames per second. We cut a shorter video for sampling and annotation. The original video was longer than a half hour.
  - `Netherland night time`: This subset consists of **824** images of Hague, Amsterdam city in night time sampled by the following original video:   
    URL: [netherland_youtube_video](https://www.youtube.com/watch?v=eAy9eHsynhM)  
    The frame rate of the video is 30 frames per second. We cut a shorter video for sampling and annotation. The original video was longer than a half hour.
  - `Switzerland`: This subset consists of **372** images of Switzerland sampled by the following video:   
    URL: [switzerland_youtube_video](https://www.youtube.com/watch?v=0iw5IP94m0Q)  
    The frame rate of the video is 30 frames per second. We cut a shorter video for sampling and annotation. The original video was longer than one hour.
  - `Zurich`: This subset consists of **50** images of Zurich city provided by the Cityscapes training set in package [leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3)
  - `Stuttgart`: This subset consists of **69** images of Stuttgart city provided by the Cityscapes training set in package [leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3)
  - `Strasbourg`: This subset consists of **50** images of Strasbourg city provided by the Cityscapes training set in package [leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3)

We use the fisheye images from the WoodScape dataset to select **244** images from the front, rear, left, and right cameras for fisheye camera data. 
The source of fisheye data for sampling is located at WoodScape's [Fisheye images](https://woodscape.valeo.com/download).

In total, **3,447** images were selected and annotated in PP4AV.


### Annotations

#### Annotation process

Annotators annotate facial and license plate objects in images. For facial objects, bounding boxes are defined by all detectable human faces from the forehead to the chin to the ears. Faces were labelled with diverse sizes, skin tones, and faces partially obscured by a transparent material, such as a car windshield. For license plate objects,  bounding boxes consists of all recognizable license plates with high variability, such as different sizes, countries, vehicle types (motorcycle, automobile, bus, truck), and occlusions by other vehicles. License plates were annotated for vehicles involved in moving traffic. To ensure the quality of annotation, there are two-step process for annotation. In the first phase, two teams of annotators will independently annotate identical image sets. After their annotation output is complete, a merging method based on the IoU scores between the two bounding boxes of the two annotations will be applied. Pairs of annotations with IoU scores above a threshold will be merged and saved as a single annotation. Annotated pairs with IoU scores below a threshold will be considered conflicting. In the second phase, two teams of reviewers will inspect the conflicting pairs of annotations for revision before a second merging method similar to the first is applied. The results of these two phases will be combined to form the final annotation. All work is conducted on the CVAT tool https://github.com/openvinotoolkit/cvat.

#### Who are the annotators?

Vantix Data Science team

### Dataset Folder
The `data` folder contains below files:
- `images.zip`: contains all preprocessed images of PP4AV dataset. In this `zip` file, there are bellow folder included:  
  `fisheye`: folder contains 244 fisheye images in `.png` file type  
  `zurich`: folder contains 244 fisheye images in `.png` file type  
  `strasbourg`: folder contains 244 fisheye images in `.png` file type  
  `stuttgart`: folder contains 244 fisheye images in `.png` file type  
  `switzerland`: folder contains 244 fisheye images in `.png` file type  
  `netherlands_day`: folder contains 244 fisheye images in `.png` file type  
  `netherlands_night`: folder contains 244 fisheye images in `.png` file type  
  `paris`: folder contains 244 fisheye images in `.png` file type  

- `annotations.zip`: contains annotation data corresponding to `images.zip` data. In this file, there are bellow folder included:  
    `fisheye`: folder contains 244 annotation `.txt` file type for fisheye image following `yolo v1.1` format.     
    `zurich`: folder contains 50 file `.txt` annotation following `yolo v1.1` format, which corresponding to 50 images file of `zurich` subset.  
    `strasbourg`: folder contains 50 file `.txt` annotation following `yolo v1.1` format, which corresponding to 50 images file of `strasbourg` subset.  
    `stuttgart`: folder contains 69 file `.txt` annotation following `yolo v1.1` format, which corresponding to 69 images file of `stuttgart` subset.  
    `switzerland`: folder contains 372 file `.txt` annotation following `yolo v1.1` format, which corresponding to 372 images file of `switzerland` subset.  
    `netherlands_day`: folder contains 388 file `.txt` annotation following `yolo v1.1` format, which corresponding to 388 images file of `netherlands_day` subset.  
    `netherlands_night`: folder contains 824 file `.txt` annotation following `yolo v1.1` format, which corresponding to 824 images file of `netherlands_night` subset.  
    `paris`: folder contains 1450 file `.txt` annotation following `yolo v1.1` format, which corresponding to 1450 images file of `paris` subset.  
-   `soiling_annotations.zip`: contain raw annotation data without filtering. The folder structure stored in this file is similar to format of `annotations.zip`.


### Personal and Sensitive Information

[More Information Needed]

## Dataset Structure

### Data Instances

A data point comprises an image and its face and license plate annotations.

```
{
  'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1920x1080 at 0x19FA12186D8>, 'objects': {
    'bbox': [
      [0 0.230078 0.317081 0.239062 0.331367],
      [1 0.5017185 0.0306425 0.5185935 0.0410975],
      [1 0.695078 0.0710145 0.7109375 0.0863355],
      [1 0.4089065 0.31646 0.414375 0.32764],
      [0 0.1843745 0.403416 0.201093 0.414182],
      [0 0.7132 0.3393474 0.717922 0.3514285]
    ]
  }
}
```

### Data Fields

- `image`: A `PIL.Image.Image` object containing the image. Note that when accessing the image column: `dataset[0]["image"]` the image file is automatically decoded. Decoding of a large number of image files might take a significant amount of time. Thus it is important to first query the sample index before the `"image"` column, *i.e.* `dataset[0]["image"]` should **always** be preferred over `dataset["image"][0]`
- `objects`: a dictionary of face and license plate bounding boxes present on the image
  - `bbox`: the bounding box of each face and license plate (in the [yolo](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#yolo) format). Basically, each row in annotation `.txt` file for each image `.png` file consists of data in format: `<object-class> <x_center> <y_center> <width> <height>`:
    - `object-class`: integer number of object from 0 to 1, where 0 indicate face object, and 1 indicate licese plate object
    - `x_center`: normalized x-axis coordinate of the center of the bounding box.  
      `x_center = <absolute_x_center> / <image_width>`
    - `y_center`: normalized y-axis coordinate of the center of the bounding box.  
      `y_center = <absolute_y_center> / <image_height>`
    - `width`: normalized width of the bounding box.  
      `width = <absolute_width> / <image_width>`
    - `height`: normalized wheightdth of the bounding box.  
      `height = <absolute_height> / <image_height>`
    - Example lines in YOLO v1.1 format `.txt' annotation file:  
      `1 0.716797 0.395833 0.216406 0.147222  
      0 0.687109 0.379167 0.255469 0.158333  
      1 0.420312 0.395833 0.140625 0.166667
      `


## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

Linh Trinh

### Licensing Information

[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/).

### Citation Information

```
@article{PP4AV2022,
  title = {PP4AV: A benchmarking Dataset for Privacy-preserving Autonomous Driving},
  author = {Linh Trinh, Phuong Pham, Hoang Trinh, Nguyen Bach, Dung Nguyen, Giang Nguyen, Huy Nguyen},
  booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year = {2023}
}
```

### Contributions

Thanks to [@khaclinh](https://github.com/khaclinh) for adding this dataset.


