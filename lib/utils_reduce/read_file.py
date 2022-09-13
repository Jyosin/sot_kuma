''' Details
Author: Zhipeng Zhang (zpzhang1995@gmail.com)
Function: read files with [.yaml] [.txt]
Data: 2021.6.23
'''
import os
import yaml
import numpy as np

def load_yaml(path):
    '''
    load [.yaml] files
    '''
    file = open(path, 'r')
    yaml_obj = yaml.load(file.read(), Loader=yaml.FullLoader)
    return yaml_obj

# ----------------------- MOT ----------------------

def read_results(filename, data_type: str, is_gt=False, is_ignore=False):
    """
    read mot results
    :param filename:
    :param data_type:
    :param is_gt:
    :param is_ignore:
    :return:
    """
    if data_type in ('MOTChallenge', 'lab'):
        read_fun = read_mot_results
    else:
        raise ValueError('Unknown data type: {}'.format(data_type))

    return read_fun(filename, is_gt, is_ignore)


def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)

    return tlwhs, ids, scores


def check_file(file):
    # Search for file if not found
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        return files[0]  # return first file if multiple found