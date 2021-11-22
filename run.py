import argparse
import pandas as pd
from glob import glob
from pathlib import Path

from .modules import Inferencer


def main(args):
    MODE = args.mode
    WORK_DIR = args.work_dir
    INPUT_PATH = WORK_DIR + '/input'
    OUTPUT_PATH = WORK_DIR + '/output'
    WEIGHTS_PATH = WORK_DIR + '/pretrained'

    MODEL = Inferencer()

    # Start Train
    if MODE == 'train':
        pass

    # Start Predict
    elif MODE == 'predict':
        image_lst = glob(INPUT_PATH + '/*jpg')
        columns=['filename', 'x_c', 'y_c', 'w', 'h', 'class_labelconfidence', 'image_width', 'image_height']
        submit_sample = pd.DataFrame(columns=columns)
        for img_path in image_lst:
            prediction = MODEL.predict(img_path)
            df = pd.DataFrame(data=np.zeros((len(prediction['class_label']), len(columns))), columns=columns)
            for key in prediction.keys():
              df.loc[:, key] = prediction[key]
            submit_sample = pd.concate([submit_sample, df])
        submit_sample.to_csv(OUTPUT_PATH + '/submission.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='DeersAPP')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'])
    parser.add_argument('--work_dir', type=str, required=True, help='App work dir')

    args = parser.parse_args()
    main(args)

