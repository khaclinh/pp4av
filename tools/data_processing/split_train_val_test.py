import os
import splitfolders
from glob import glob 
import shutil

def get_label_from_image_name(data_image: str, data_label: str, label_folder_save: str):
    """get annotaion file follow image file name

    Args:
        data_image (_str_): _folder path data image_
        data_label (_str_): _folder path data label_
        label_folder_save (_str_): _folder path save label_

    """
    image_folder = glob(f'{data_image}/*')
    for folder in image_folder:
        folder_name = os.path.join(folder.split('/')[-2], folder.split('/')[-1])
        os.makedirs(os.path.join(label_folder_save, folder_name), exist_ok=True)
    images = glob(f'{data_image}/*/*')
    labels = glob(f'{data_label}/*/*')
    for image in images:
        img = os.path.join(image.split("/")[-2], image.split("/")[-1][:-4])
        for label in labels:
            lab = os.path.join(label.split("/")[-2], label.split("/")[-1][:-4])
            if img == lab:
                folder_save = os.path.join(label_folder_save, image.split("/")[-3])
                shutil.copy(label, os.path.join(folder_save, label.split("/")[-2]))
    return {'message':'Copy Done!'}

class DataCurate:
    """
        Data curate train, val, test follow specific ratio
    """
    def __init__(self, 
                 image_data: str, 
                 label_data: str, 
                 image_folder_save: str,
                 label_folder_save: str, 
                 train_rate: float, 
                 val_rate: float, 
                 test_rate: float):
        self.image_data = image_data
        self.label_data = label_data
        self.image_folder_save = image_folder_save
        self.label_folder_save = label_folder_save
        self.train_rate = train_rate
        self.val_rate = val_rate
        self.test_rate = test_rate

    def split_train_val_test(self):
        """split train, val, test data sample follow ratio

        Args:
            image_data (_str_): _path raw data folder_
            label_data (_str_): _path raw label folder_
            image_folder_save (_str_): _path data folder splited_
            label_folder_save (_str_): _path label folder splited_
            train_rate (_float_): _ratio train dataset split_
            val_rate (_float_): _ratio val dataset split_
            test_rate (_float_): _ratio test dataset split_
        """
        splitfolders.ratio(self.image_data, 
                           output= self.image_folder_save, 
                           seed=1337,
                           ratio=(train_rate, val_rate, test_rate)) 
        train = self.image_folder_save + "/train"
        val = self.image_folder_save + "/val"
        test = self.image_folder_save + "/test"
        return train, val, test

    def get_label(self):
        train, val, test = self.split_train_val_test()
        get_label_from_image_name(train, self.label_data, self.label_folder_save)
        get_label_from_image_name(val, self.label_data, self.label_folder_save)
        get_label_from_image_name(test, self.label_data, self.label_folder_save)
        

if __name__=='__main__':
    image_data = "/data/"
    label_data = "/annotation//label/"
    image_folder_save = "/images"
    label_folder_save = "/labels"
    train_rate, val_rate, test_rate = 0.7, 0.2, 0.1
    data_curate = DataCurate(image_data, 
                             label_data, 
                             image_folder_save, 
                             label_folder_save, 
                             train_rate, 
                             val_rate, 
                             test_rate)
    data_curate.get_label()
    print("Done!")