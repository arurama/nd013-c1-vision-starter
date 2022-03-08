import argparse
import glob
import os

import shutil

from sklearn.model_selection import train_test_split
from utils import get_module_logger





def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    # TODO: Implement function
    files= glob.glob(source + '\\*.tfrecord')

    File_train, File_val_test = train_test_split( files, train_size=0.6, random_state=45)
    File_val, File_test = train_test_split( File_val_test, train_size=0.5, random_state=45)


    TrainFolder = os.path.join(destination, 'train')
    TestFolder = os.path.join(destination, 'test')
    ValFolder = os.path.join(destination, 'val')

    # make dir 
    if not os.path.exists(TrainFolder):
        os.makedirs(TrainFolder)
        
    if not os.path.exists(TestFolder):
        os.makedirs(TestFolder)

    if not os.path.exists(ValFolder):
        os.makedirs(ValFolder)
        

    for File in File_train :
        shutil.move( File, TrainFolder )
        
    for File in File_val :
        shutil.move(File, ValFolder )
        
    for File in File_test :
        shutil.move(File, TestFolder)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)