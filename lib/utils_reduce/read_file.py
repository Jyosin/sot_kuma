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
