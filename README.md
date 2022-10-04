
# PP4AV: A benchmarking Dataset for Privacy-preserving Autonomous Driving

## Quick start
- [Dataset Description and Download](https://huggingface.co/datasets/khaclinh/pp4av)
- [Model for face and license plate detection](models/README.md)
- [Evaluation](evaluations/README.md)

## News
**2022.10**: PP4AV dataset is available in huggingface


**2022.7**: PP4AV v1.0 is released with images, face and license plate bounding box annotations.
. 

## Dataset
### Data Summary
PP4AV is the first public dataset with faces and license plates annotated with driving scenarios. P4AV provides 3,447 annotated driving images for both faces and license plates. For normal camera data, we sampled images from the existing videos in which cameras were mounted in moving vehicles, running around the European cities. The images in PP4AV were sampled from 6 European cities at various times of day, including nighttime. We use the fisheye images from the WoodScape dataset to select 244 images from the front, rear, left, and right cameras for fisheye camera data. PP4AV dataset can be used as a benchmark suite (evaluating dataset) for data anonymization models in autonomous driving.

### Dataset description
The detail of dataset **collection, structure, annotation, format** are described in [Hugging Face PP4AV dataset](https://huggingface.co/datasets/khaclinh/pp4av).

### Download
- PP4AV images [[Google Drive](https://drive.google.com/file/d/1eJDei81PTpVFRNjzPSYkaDiQZoS5S7xm/view?usp=sharing)] [[Hugging Face](https://huggingface.co/datasets/khaclinh/pp4av/blob/main/data/images.zip)]
- PP4AV annotations [[Google Drive](https://drive.google.com/file/d/1njVbQp-CMrn0_Em778NLCvd2DNrSTSUX/view?usp=sharing)] [[Hugging Face](https://huggingface.co/datasets/khaclinh/pp4av/blob/main/data/soiling_annotations.zip)]

### Dataset Manipulation

### Dataset License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This PP4AV dataset is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.


## Baseline model

## Evaluation

Now that you have the dataset and are familiar with its structure, you are ready to train or test [3D-RetinaNet](https://github.com/gurkirt/3D-ReintaNet), which contains a dataloader class and evaluation scripts required for all the tasks in ROAD dataset. 

You can find the **evaluation** functions in [3D-RetinaNet/modules/evaluation.py](https://github.com/gurkirt/3D-RetinaNet/blob/master/modules/evaluation.py).


## Citation 
If you think this work is useful for you, please cite 

    @article{PP4AV2022,
      title = {PP4AV: A benchmarking Dataset for Privacy-preserving Autonomous Driving},
      author = {Linh Trinh, Phuong Pham, Hoang Trinh, Nguyen Bach, Dung Nguyen, Giang Nguyen, Huy Nguyen},
      booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
      year = {2023}
    }

## License

## Contact
If you have any problems about PP4AV, please contact Linh Trinh at linhtk.dhbk@gmail.com. 
