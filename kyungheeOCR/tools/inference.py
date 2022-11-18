import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2
import re
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path

data_root = '../data'


from vedastr.runners import InferenceRunner
from vedastr.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('config', default='configs/small_satrn.py', help='Config file path')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file path')
    parser.add_argument('gpus', default='0', help='target gpus')
    args = parser.parse_args()

    return args

# Histogram Clipping
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

class Predictor:
    def __init__(self, runner):
        self.runner = runner

    def load_image(self, image_path):
        img = cv2.imread(str(Path(data_root)/image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = automatic_brightness_and_contrast(img)
        return img

    def prediction(self, img_path_list):
        preds = list()
        for image_path in tqdm(img_path_list):
            img = self.load_image(image_path)
            text, probs = self.runner(img)
            if (text[0] == ' ') : text[0] = 'xxx'
            preds.append(text[0])
            tqdm.write(str((image_path, text[0], (probs[0] < 0.1).item())))
        print('Done.')
        return preds


# Inference
def main():
    # Load Configuration Files
    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    deploy_cfg = cfg['deploy']
    common_cfg = cfg.get('common')
    deploy_cfg['gpu_id'] = args.gpus.replace(" ", "")

    # Load trained model
    runner = InferenceRunner(deploy_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)
    
    # Inference
    predictor = Predictor(runner)
    submit = pd.read_csv(str(Path(data_root)/'sample_submission.csv'))
    test_predicts = predictor.prediction(submit['img_path'].values)
    submit['text'] = test_predicts
    print(submit)

    # Make submission file
    submit.to_csv(str(Path(data_root)/'submit.csv'), index=False, encoding="utf-8-sig")


if __name__ == '__main__':
    main()
