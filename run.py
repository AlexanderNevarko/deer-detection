import argparse
import pandas as pd
from glob import glob
from pathlib import Path

from module import AnswerNet

def get_deers_params(prediction):
    '''
    prediction: dictionary
            {'image_name': str,
            'bboxs': dictionary ({'x_c': list[int], 'y_c': list[int], 'w': list[int], 'h': list[int], 'class_labelconfidence': Interval[0, 1]]}),
              'classes': list[Union('reindeer', 'fawn')], 
              'image_width': int, 
              'image_height': int
              }
    '''
    columns=['filename', 'x_c', 'y_c', 'w', 'h', 'class_labelconfidence', 'image_width', 'image_height']
    df = pd.DataFrame(data=np.zeros((len(classes), len(columns))), columns=columns)
    df.loc[:, 'filename'] = prediction['image_name']
    df.loc[:, 'image_width'] = prediction['image_width']
    df.loc[:, 'image_height'] = prediction['image_height']
    df.loc[:, 'classes'] = prediction['classes']
    for key in bbox.keys():
        df.loc[:, key] = bbox[key]
    return df

def main(args):
    MODE = args.mode
    WORK_DIR = args.work_dir
    INPUT_PATH = WORK_DIR + '/input'
    OUTPUT_PATH = WORK_DIR + '/output'
    WEIGHTS_PATH = WORK_DIR + '/pretrained'

    MODEL = AnswerNet()

    # Start Train
    if MODE == 'train':
        pass

    # Start Predict
    elif MODE == 'predict':
        image_lst = glob(INPUT_PATH + '/*jpg')
        submit_sample = pd.DataFrame(columns=['filename', 'x_c', 'y_c', 'w', 'h', 'class_labelconfidence', 'image_width', 'image_height'])
        for img_path in image_lst:
            prediction = MODEL.predict(img_path)
            df = get_deers_params(prediction)
            submit_sample = pd.concate([submit_sample, df])
        submit_sample.to_csv(OUTPUT_PATH + '/submission.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='DeersAPP')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'])
    parser.add_argument('--work_dir', type=str, required=True, help='App work dir')

    args = parser.parse_args()
    main(args)

