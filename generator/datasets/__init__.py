
from .robot_dataset import RobotMotionDataset

__all__ = ['RobotMotionDataset', 'get_dataset']

def get_dataset(opt, split='train', mode='train', accelerator=None):
    if opt.dataset_name in ['robot', 'robotv2', 'robotv2_hard', 'kungfu']:
        dataset = RobotMotionDataset(opt, split, mode, accelerator)
    else:
        raise KeyError('Dataset Does Not Exist: %s' % opt.dataset_name)
    
    if accelerator:
        accelerator.print('Completing loading %s dataset' % opt.dataset_name)
    else:
        print('Completing loading %s dataset' % opt.dataset_name)
    
    return dataset
