import os
import cv2
from glob import glob


class ReadLabelDataset():
    """
    Convert ground-truth label to Yolo format
    """

    def __init__(self, img_path, img_ext, lb_path, lb_ext=None, is_save_yolo=False, yolo_save_path=''):
        self.img_path = img_path
        # self.img_ext = img_ext
        self.list_image = self.list_file(img_path, img_ext)
        self.list_label = self.list_file(lb_path, lb_ext)
        # self.lb_path = lb_path
        # self.lb_ext = lb_ext
        self.labels = self.read_label()
        self.save_yolo = self.save_yolo_format(yolo_save_path, is_save_yolo)

    def list_file(self, file_path, file_ext):
        """List all image paths from input path and input ext"""
        if type(file_path) == list:
            return file_path
        elif os.path.isfile(file_path):
            return [file_path]

        assert os.path.isdir(file_path) is True
        # print(f"{file_path.rstrip('/')}/**/*{file_ext}")
        list_path = glob(f"{file_path.rstrip('/')}/**/*{file_ext}", recursive=True)

        print(f'There are {len(list_path)} file(s) in {file_path}')

        return list_path

    def read_label(self):
        """Read and convert all labels of dataset to dict
        {
            image_path: [
                [class_id, x_tl, y_tl, w, h, **],
                ...
            ],
            ...
        }
        """
        return {}

    def save_yolo_format(self, save_path, is_save):
        if not is_save:
            return None
        for img_name, list_box in self.labels.items():
            height, weight = cv2.imread(os.path.join(self.img_path, img_name)).shape[:2]
            label_file = []
            for box in list_box:
                x, y, w, h = box[1:5]
                cenx = (x + 1/2 * w) / weight
                ceny = (y + 1/2 * h) / height
                w_yl = w / weight
                h_yl = h / height
                label_file.append(' '.join([str(x) for x in [box[0], cenx, ceny, w_yl, h_yl]]))

            os.makedirs(os.path.join(save_path, img_name).rsplit('/', 1)[0], exist_ok=True)
            with open(os.path.join(save_path, img_name).rsplit('.', 1)[0] + '.txt', 'w') as wf:
                wf.write('\n'.join(label_file))


    def visualize(self, list_color):
        """
        Visualize labeled sequence images by opencv
        Args:
            img_path: path of a image
            list_box: list box corresponding the image
            list_color: list of color RGB want to plot, index box in list is class id
        Return:
            Pop up new screen to watch
        """

        for img_name, list_box in self.labels.items():
            image = cv2.imread(os.path.join(self.img_path, img_name))
            # print(image.shape)
            for box in list_box:
                cv2.rectangle(
                    image,                 # image
                    tuple(box[1:3]),        # top-left
                    tuple([box[1] + box[3], box[2] + box[4]]), # bottom-right
                    list_color[int(box[5])] # color
            )
            # cv2.putText(
            #     image,
            #     str(box['extra'].get('box_id', -1)),
            #     (rm_negative(box['hbox'][0]), rm_negative(box['hbox'][1] - 10)),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #     (0, 255, 0) if box['tag'] == 'person' and box['hbox'] != box['fbox'] and box['hbox'] != box[
            #         'vbox'] else (0, 0, 255),
            #     1)
            cv2.imshow(img_name, image)
            cv2.waitKey(0)


class WiderFace(ReadLabelDataset):
    def __init__(self, img_path, img_ext, lb_path, lb_ext, is_save_yolo=False, yolo_save_path=''):
        super().__init__(img_path, img_ext, lb_path, lb_ext, is_save_yolo, yolo_save_path)


    def read_label(self):
        labels = {}
        with open(self.list_label[0]) as f:
            label_data = f.read().strip().split('\n')
        idx = 0
        while idx < len(label_data):
#             print(label_data[idx : idx+5])
            labels[label_data[idx]] = []
            num_boxes = max(int(label_data[idx + 1]), 1)
            if label_data[idx + 1] == '0': print(label_data[idx : idx + num_boxes + 3])
            for row in label_data[idx + 2: idx + 2 + num_boxes]:
                labels[label_data[idx]].append([0] + list(map(lambda x: int(x), row.strip().split(' '))))
            idx += (num_boxes + 2)
        return labels


# class CrowdHuman(ReadLabelDataset):
#     def __init__(self, img_path, img_ext, lb_path, lb_ext):
#         super().__init__(img_path, img_ext, lb_path, lb_ext)
#         self.list_label = self.list_file(lb_path, None)
#
#     def read_label(self):
#         labels = {}
#         with open(self.list_label[0]) as f:
#             label_data = f.read().strip().split('\n')
#
#         for lb in label_data:
#             lb = json.loads(lb)
#             # if img_name == lb['ID']:
#             #     img_lb = lb['gtboxes']
#             labels[lb['ID']] = []
#                 for box in lb['gtboxes']:
#                     if box['tag'] == 'person' and box['hbox'] != box['fbox'] and box['hbox']:
#                         labels[lb['ID']].append(
#                             (rm_negative(box['hbox'][0]), rm_negative(box['hbox'][1])),
#                             (box['hbox'][0] + box['hbox'][2], box['hbox'][1] + box['hbox'][3]),
#                         )
    #
    # def abc():
    #
    #     path = '/home/hiengl/Downloads/annotation_train.odgt'
    #     with open(path) as f:
    #         # data = json.load(f)
    #         data = f.read()
    #         data1 = data.split('\n')
    #     # print(data1[0])
    #
    #     img_list = glob.glob('*.jpg')
    #     # print(img_list)
    #     for img in img_list:
    #         img_name = (img.rsplit('/', 1)[-1]).rsplit('.', 1)[0]
    #         print('img_name', img_name)
    #         img_lb = {}
    #         for lb in data1:
    #             lb = json.loads(lb)
    #             # print(type(lb))
    #             if img_name == lb['ID']:
    #                 img_lb = lb['gtboxes']
    #                 break
    #         # print(img_lb)
    #
    #         image = cv2.imread(img)
    #         print(image.shape)
    #         for box in img_lb:
    #             print(box)
    #             # fbox - red
    #             # cv2.rectangle(image, tuple(box['fbox'][:2]), tuple(box['fbox'][2:]), (255, 0, 0))
    #             # hbox - green
    #             cv2.rectangle(
    #                 image,
    #                 (rm_negative(box['hbox'][0]), rm_negative(box['hbox'][1])),
    #                 (box['hbox'][0] + box['hbox'][2], box['hbox'][1] + box['hbox'][3]),
    #                 (0, 255, 0) if box['tag'] == 'person' and box['hbox'] != box['fbox'] and box['hbox'] != box[
    #                     'vbox'] else (0, 0, 255)
    #             )
    #             cv2.putText(
    #                 image,
    #                 str(box['extra'].get('box_id', -1)),
    #                 (rm_negative(box['hbox'][0]), rm_negative(box['hbox'][1] - 10)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                 (0, 255, 0) if box['tag'] == 'person' and box['hbox'] != box['fbox'] and box['hbox'] != box[
    #                     'vbox'] else (0, 0, 255),
    #                 1)
    #
    #         if image.shape[0] > 1000:
    #             image = cv2.resize(image, (round(image.shape[1] * 0.6), round(image.shape[0] * 0.6)),
    #                                interpolation=cv2.INTER_AREA)
    #         cv2.imshow(img_name, image)
    #         cv2.waitKey(0)


if __name__ == "__main__":
    set_type = 'train' # train or val, test not have label grond truth
    img_path = f'/OpenDataset/wider_face/WIDER_{set_type}/images'
    img_ext = '.jpg'
    lb_path = f'/OpenDataset/wider_face/wider_face_split/wider_face_{set_type}_bbx_gt.txt'
    yolo_save_path = f'//OpenDataset/wider_face/WIDER_{set_type}/annotations'

    a = WiderFace(img_path, img_ext, lb_path, None, True, yolo_save_path)
    # a.read_label()
    # a.visualize([(255, 0, 0), (0, 255, 0), (0, 0, 255)])