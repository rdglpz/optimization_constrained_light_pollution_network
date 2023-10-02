
import shutil
import json

import os
import sys
import config.config as cfg

root = os.getcwd() + "/.."
sys.path.insert(0, root)


def readConfigFile(fn):
    """
    
    """
    prefix = cfg.experiment
    
    cfgr = json.load(open(prefix + fn))
    return cfgr

def createProject(fn):
    """
    
    """
    
    

    
    
    
    
    prefix = cfg.experiment
    
    file_path = prefix + fn
    print("Read File:", file_path)    
    cfgr = readConfigFile(file_path)
    
    path = prefix + cfgr["experiment_id"]
    print("Creating Directory in: " + path)
    os.makedirs(prefix + cfgr["experiment_id"],exist_ok=True)
    
    path = prefix + cfgr["experiment_id"] + cfgr["folder_output"]
    print("Creating Directory " + path)
    os.makedirs(path, exist_ok=True)
    
    path = prefix + cfgr["experiment_id"]+cfgr["folder_semivariances"]
    print("Creating Directory: " + path)
    os.makedirs(path, exist_ok = True)
    
    path = prefix + cfgr["experiment_id"] + fn
    print("Copy "+ fn + " -> " + path)
    shutil.copyfile(file_path, path)
    