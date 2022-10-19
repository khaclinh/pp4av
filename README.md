
# PP4AV: A benchmarking Dataset for Privacy-preserving Autonomous Driving

## Quick start
- [Dataset Description and Download](https://huggingface.co/datasets/khaclinh/pp4av)
- [Model for face and license plate detection](models/README.md)
- [Evaluation](evaluations/README.md)

## News
**2022.10**: 
- PP4AV dataset is available in [pp4av huggingface](https://huggingface.co/datasets/khaclinh/pp4av)
- Pretrained baseline model and demo is available in [self-driving-anonymization huggingface](https://huggingface.co/spaces/khaclinh/self-driving-anonymization)

**2022.7**: PP4AV v1.0 is released with images, face and license plate bounding box annotations.


## Prerequisites


The code of baseline model and auxiliary scripts is built with following libraries:

- Python >= 3.6, \<3.9
- Pillow = 8.4.0 (see [here](https://github.com/mit-han-lab/bevfusion/issues/63))
- loguru==0.5.3
- matplotlib==3.3.4
- numpy==1.19.5
- opencv_python==4.5.3.56
- PyYAML==6.0
- recommonmark==0.7.1
- setuptools==58.0.4
- Sphinx==3.2.1
- sphinx_rtd_theme==0.5.0
- tabulate==0.8.7
- thop==0.0.31.post2005241907
- tqdm==4.31.1
- tensorboard==2.3.0

## Dataset
### Data Summary
PP4AV is the first public dataset with faces and license plates annotated with driving scenarios. P4AV provides 3,447 annotated driving images for both faces and license plates. For normal camera data, we sampled images from the existing videos in which cameras were mounted in moving vehicles, running around the European cities. The images in PP4AV were sampled from 6 European cities at various times of day, including nighttime. We use the fisheye images from the WoodScape dataset to select 244 images from the front, rear, left, and right cameras for fisheye camera data. PP4AV dataset can be used as a benchmark suite (evaluating dataset) for data anonymization models in autonomous driving.

### Dataset description
The detail of dataset **collection, structure, annotation, format** are described in [Hugging Face PP4AV dataset](https://huggingface.co/datasets/khaclinh/pp4av).
You also can check the description of PP4AV dataset in [this document](DATASET.md).

### Download
- PP4AV images [[Google Drive](https://drive.google.com/file/d/1eJDei81PTpVFRNjzPSYkaDiQZoS5S7xm/view?usp=sharing)] [[Hugging Face](https://huggingface.co/datasets/khaclinh/pp4av/blob/main/data/images.zip)]
- PP4AV annotations [[Google Drive](https://drive.google.com/file/d/1njVbQp-CMrn0_Em778NLCvd2DNrSTSUX/view?usp=sharing)] [[Hugging Face](https://huggingface.co/datasets/khaclinh/pp4av/blob/main/data/soiling_annotations.zip)]

### Dataset Manipulation
We profile the utility scripts for manipulating the PP4A dataset. Please check [this document](tools/data_processing/README.md) for detail of guidance.

### Dataset License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This PP4AV dataset is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.


## Baseline model and performance
- The pretrained model and the demo of Self Driving Anonymization model is available in Hugging Face:  
[PP4AV Self Driving Anonymization](https://huggingface.co/spaces/khaclinh/self-driving-anonymization)

- Environment installation
```shell
conda create --name pp4av-env python=3.8
conda activate pp4av-env
pip3 install -U pip && pip3 install -r requirements.txt
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -v -e .  # or  python3 setup.py develop
```

- Model performance reports
    <table>
    <thead>
      <tr>
        <th rowspan="2"></th>
        <th rowspan="2">Method</th>
        <th colspan="2">Normal images</th>
        <th colspan="2">Fisheye images</th>
      </tr>
      <tr>
        <th>AP_50</th>
        <th>AR_50</th>
        <th>AP_50</th>
        <th>AR_50</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td rowspan="6">Face</td>
        <td>UAI Anonymizer</td>
        <td>42.64%</td>
        <td>83.70%</td>
        <td>43.98%</td>
        <td>53.33%</td>
      </tr>
      <tr>
        <td>AWS API</td>
        <td>63.69%</td>
        <td>73.33%</td>
        <td>40.72%</td>
        <td>46.67%</td>
      </tr>
      <tr>
        <td>Google API</td>
        <td>7.97%</td>
        <td>8.99%</td>
        <td>7.64%</td>
        <td>8.89%</td>
      </tr>
      <tr>
        <td>RetinaFace</td>
        <td>62.71%</td>
        <td>88.28%</td>
        <td>43.82%</td>
        <td>62.96%</td>
      </tr>
      <tr>
        <td>Yolo5Face</td>
        <td>69.31%</td>
        <td><b>93.96%</b></td>
        <td><b>69.59%</b></td>
        <td><b>82.96%</b></td>
      </tr>
      <tr>
        <td>PP4AV</td>
        <td><b>76.22%</b></td>
        <td>92.52%</td>
        <td>59.20%</td>
        <td>63.92%</td>
      </tr>
      <tr>
        <td rowspan="4">License plate</td>
        <td>ALPR</td>
        <td>38.79%</td>
        <td>41.68%</td>
        <td>17.26%</td>
        <td>31.21%</td>
      </tr>
      <tr>
        <td>Nvidia LPDnet</td>
        <td>57.41%</td>
        <td>58.44%</td>
        <td>24.90%</td>
        <td>26.24%</td>
      </tr>
      <tr>
        <td>UAI Anonymizer</td>
        <td>84.89%</td>
        <td>85.61%</td>
        <td>44.14%</td>
        <td>53.90%</td>
      </tr>
      <tr>
        <td>PP4AV</td>
        <td><b>88.12%</b></td>
        <td><b>91.88%</b></td>
        <td><b>49.53%</b></td>
        <td><b>58.17%</b></td>
      </tr>
    </tbody>
    </table>


## Evaluation
- We provide scripts to evaluate models in both WIDER FACE and standard evaluative methods for object detection.
You can follow up the [WIDER FACE evaludation document](evaluations/widerface_evals/README.md) for plotting the PR-curve.

- We also provide the evaluation script, which includes performance metrics such as Average Recall, Average Precision, and qualitative analysis. This evaluation's document and coding script are available at [this document and coding] (evaluations/baseline_evals/README.md).

- The script for comprehensive analysis is available in [this section](tools/analyze_datasets_models/README.md).

## Citation 
If you think this work is useful for you, please cite 

    @article{PP4AV2022,
      title = {PP4AV: A benchmarking Dataset for Privacy-preserving Autonomous Driving},
      author = {Linh Trinh, Phuong Pham, Hoang Trinh, Nguyen Bach, Dung Nguyen, Giang Nguyen, Huy Nguyen},
      booktitle = {IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
      year = {2023}
    }

## FAQ
TBD

## Acknowledgements
The baseline model is based on [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX). The data annotation tool was done with [CVAT too](https://github.com/opencv/cvat). 

## Contact
If you have any problems about PP4AV, please contact Linh Trinh at linhtk.dhbk@gmail.com. 
