import os, sys
sys.path.append(os.path.abspath(os.curdir))
import yaml

from utils.MasterLogger import _get_logger_
from utils.cuda_device import CudaDevice
    
class Opt():
    def __init__(self):
        #
        path_cwd=get_main_working_directory('MultiphaseBioImgGen')
        print(path_cwd)

        #
        self.base = {'PATH_BASE_DIR': path_cwd}

        # Utils
        self.logger = _get_logger_(path_base_dir=self.base['PATH_BASE_DIR'], verbose=False)
        self.pytorch_cuda = CudaDevice()
        self.logger.debug(f'Master_Logger started at {self.base["PATH_BASE_DIR"]}')

        # Models
        self.imagen = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/model/imagen.yaml'))
        self.elucidated_imagen = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/model/elucidated_imagen.yaml'))
        self.logger.debug(f'Model configs loaded')

        # Deploying
        self.conductor = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/conductor.yaml'))
        self.data = _get_config_(path=os.path.join(self.base['PATH_BASE_DIR'],'configs/data.yaml'))
        self.logger.debug('Data configs loaded')


def _get_config_(path :str):
    
    with open(path,'r') as file:
        config = yaml.safe_load(file)

    return config


def get_main_working_directory(name):
    
    path_base = os.getcwd()
    
    for i in range(len(path_base.split('/'))):

        if path_base.split('/')[-1] == name:
            break
        else:
            path_base = '/'.join(path_base.split('/')[0:-1])
    
    assert len(path_base) > 0, 'Could not find current directory'
    
    return path_base


def main():
    opt = Opt()
    opt.logger.info(opt.imagen)

if __name__ == '__main__':
    main()
