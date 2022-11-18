from pathlib import Path
import json
import pandas as pd
import cv2
from tqdm import tqdm
import re
import warnings
import os
warnings.filterwarnings("ignore")

Path.mkdir(Path('./Training/Cropped'),parents=True, exist_ok=True)
Path.mkdir(Path('./Validation/Cropped'),parents=True, exist_ok=True)
train_origin_list = [path for path in tqdm(Path('./Training').rglob('*.json'))]
val_origin_list = [path for path in tqdm(Path('./Validation').rglob('*.json'))]
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result)

def crop_image(origin_list, is_train):
    cropped_path_list = []
    cropped_label_list = []
    for file_num, path in enumerate(tqdm(origin_list)):
        with open(path, 'r', encoding="utf-8-sig") as f:
            json_data = json.load(f)
            file_name = json_data['images'][0]['file_name']
            file_path = Path(path).parent / file_name
            temp_file_path = './temp.jpg'
            new_file_path = (Path('./Training/Cropped')) if (is_train)\
                else (Path('./Validation/Cropped'))
            try:
                os.symlink(str(file_path), str(temp_file_path))
            except FileExistsError:
                os.remove(str(temp_file_path))
                os.symlink(str(file_path), str(temp_file_path))
            try:
                img = cv2.imread(str(temp_file_path), cv2.IMREAD_COLOR)
                for count, annotation in enumerate(json_data['annotations']):
                    new_file_name = new_file_path/(str(file_num)+'_'+str(count) + file_path.suffix)
                    label = annotation['text']
                    if not re.search(r'[ 가-힣]', label):
                        continue
                    if len(label)<=0 or len(label) > 70:
                        continue
                    roi = annotation['bbox']
                    if (type(roi[0]) is not int) or (type(roi[1]) is not int) or\
                        (type(roi[2]) is not int) or (type(roi[3]) is not int) : continue
                    if (roi[0]< 0 or roi[1]<0 or roi[2]<0 or roi[3]<0): continue
                    # if not (Path(str(new_file_name)).exists()):
                    cropped_img = img[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]
                    cropped_img = automatic_brightness_and_contrast(cropped_img)
                    cv2.imwrite(str(new_file_name), cropped_img)
                    cropped_path_list.append(str(new_file_name))
                    cropped_label_list.append(annotation["text"])
            except:
                continue
            finally:
                os.remove(str(temp_file_path))
    return cropped_path_list, cropped_label_list

train_path_list, train_label_list = crop_image(train_origin_list, is_train = True)
val_path_list, val_label_list = crop_image(val_origin_list, is_train = False)
train_df = pd.DataFrame({'img_path': train_path_list, 'text': train_label_list})
train_df.to_csv('./train.txt', index=False, sep="\t", header=False, encoding="utf-8-sig")
val_df = pd.DataFrame({'img_path': val_path_list, 'text': val_label_list})
val_df.to_csv('./val.txt', index=False, sep="\t", header=False, encoding="utf-8-sig")
