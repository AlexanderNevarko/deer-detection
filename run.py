import argparse
import pandas as pd
from glob import glob
from pathlib import Path

def main(args):
    MODE = args.mode
    WORK_DIR = args.work_dir
    INPUT_PATH = WORK_DIR + '/input'
    OUTPUT_PATH = WORK_DIR + '/output'
    WEIGHTS_PATH = WORK_DIR + '/pretrained'

    # Start Train
    if MODE == 'train':
        pass

    # Start Predict
    elif MODE == 'predict':
        image_lst = glob(INPUT_PATH + '/*jpg')
        submit_sample = pd.read_csv(WORK_DIR + '/submission_sample.csv')
        submission = {}
        for img_path in image_lst:
            submission[Path(img_path).name] = submit_sample.values[0][1:]
        submission = pd.DataFrame.from_dict(submission, orient='columns').T
        submission.columns = submit_sample.columns[1:]
        submission['filename'] = submission.index
        submission = submission[submit_sample.columns].reset_index(drop=True)
        submission.to_csv(OUTPUT_PATH + '/submission.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='DeersAPP')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'])
    parser.add_argument('--work_dir', type=str, required=True, help='App work dir')

    args = parser.parse_args()
    main(args)

