import glob
import os
import cv2

def filter_box_size(img_path, lb_path, save_path):
    """
    Filter box following the rule:
      - with box of face: 
        + width and hight of box >= 13 px
        + area of box >= 256
        + rate of width and hight of box is r: 0.5 < r < 2 # r = width/height 
      - with box of plate:
        + width and hight of box >= 8 px
        + area of box >= 225
    """
    print(img_path)
    img_list = sorted(glob.glob(img_path + '/*png'))

    lb_list = sorted(glob.glob(lb_path + '/*txt'))
    os.makedirs(save_path, exist_ok=True)

    for imp, lbp in zip(img_list, lb_list):
        if os.path.basename(imp)[:-3] != os.path.basename(lbp)[:-3]:
            print(imp, lbp)
            continue

        height, width = cv2.imread(imp).shape[:2]

        with open(lbp) as f:
            data = f.read().strip().split('\n')

        out = []
        box_visual = []
        for row_str in data:
            row = row_str.split(' ')
            if len(row) < 5:
                continue
            wbox, hbox = float(row[3]) * width, float(row[4]) * height
            xbox, ybox = float(row[1]) * width - 0.5 * wbox, float(row[2]) * height - 0.5 * hbox

            if row[0] == '0':
#                 out.append(row_str)
                if wbox >= 13 and hbox >= 13:
                    if wbox * hbox >= 256:
                        if wbox / hbox >= 0.5 and wbox / hbox <= 2:
                            out.append(row_str)
                            box_visual.append([xbox, ybox, xbox + wbox, ybox + hbox, int(row[0])])
                            
            elif row[0] == '1':
                if wbox >= 8 and hbox >= 8:
                    if wbox * hbox >= 225:
                        out.append(row_str)
                        box_visual.append([xbox, ybox, xbox + wbox, ybox + hbox, int(row[0])])
        # plot_image(imp, save_path.replace('filter_face256_plate225', 'plot_label'),
        #            box_visual)

        with open(os.path.join(save_path, os.path.basename(lbp)), 'w') as fw:
            fw.write('\n'.join(out))


def filter_box_size_dynamic(img_path, lb_path, save_path, hei_threshold):

    
    img_list = sorted(glob.glob(img_path + '/*png'))

    lb_list = sorted(glob.glob(lb_path + '/*txt'))
    os.makedirs(save_path, exist_ok=True)

    for imp, lbp in zip(img_list, lb_list):
     
        if os.path.basename(imp)[:-3] != os.path.basename(lbp)[:-3]:
            continue

        height, width = cv2.imread(imp).shape[:2]

        with open(lbp) as f:
            data = f.read().strip().split('\n')

        out = []
        box_visual = []
        for row_str in data:
            row = row_str.split(' ')
            if len(row) < 5:
                continue
            wbox, hbox = float(row[3]) * width, float(row[4]) * height
            xbox, ybox = float(row[1]) * width - 0.5 * wbox, float(row[2]) * height - 0.5 * hbox
            
            if row[0] == '0':                
                if wbox >= hei_threshold:
                    out.append(row_str)
                            
            elif row[0] == '1':
                if hbox >= hei_threshold:
                    out.append(row_str)
                    

        with open(os.path.join(save_path, os.path.basename(lbp)), 'w') as fw:
            fw.write('\n'.join(out))

def plot_image(img_path, path_to_save, boxes, color=[(0, 0, 255), (0, 255, 0)], text=['face', 'plate']):
    img = cv2.imread(img_path)
    # print(color[0], text[0])
    for box in boxes:
        img = cv2.rectangle(img, [round(box[0]), round(box[1])], [round(box[2]), round(box[3])], color[box[-1]], 2)
        img = cv2.putText(img, text[box[-1]], (round(box[0]), round(box[1]) - 5), cv2.FONT_HERSHEY_PLAIN, 1, color[box[-1]])

    os.makedirs(path_to_save, exist_ok=True)
    cv2.imwrite(path_to_save + os.path.basename(img_path), img)

if __name__ == "__main__":
        save_path = ''
        lb_path = f''
        img_path = f''

        filter_box_size(img_path, lb_path, save_path)