import torch
from os import makedirs
from os.path import join, exists



def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth.tar'):
    """
    save checkpoint
    """
    torch.save(states, join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'], join(output_dir, 'model_best.pth'))


def save_model(model, epoch, optimizer, model_name, cfg, isbest=False):
    """
    save model
    """
    if not exists(cfg.COMMON.CHECKPOINT_DIR):
        makedirs(cfg.COMMON.CHECKPOINT_DIR)

    save_checkpoint({
        'epoch': epoch + 1,
        'arch': model_name,
        'state_dict': model.module.state_dict(),
        'optimizer': optimizer.state_dict()
    }, isbest, cfg.COMMON.CHECKPOINT_DIR, 'checkpoint_e%d.pth' % (epoch + 1))