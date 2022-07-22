import math
import time
import torch
import pdb
import utils.log_helper_kuma as recorder
import utils.model_helper_kuma as loader
import torch.distributed as dist

def siamese_train(inputs):
    # parser inputs
    train_loader, model, optimizer, device = inputs['data_loader'], inputs['model'], inputs['optimizer'], inputs['device']
    cfg = inputs['config']
                                                              

    # recorder
    batch_time = recorder.AverageMeter()
    data_time = recorder.AverageMeter()
    losses = recorder.AverageMeter()
    cls_losses = recorder.AverageMeter()
    reg_losses = recorder.AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    model = model.to(device)

    for iter, batchinfo in enumerate(train_loader):
        data_time.update(time.time() - end)

        # SiamFC/SiamDW
        batch_keys = list(batchinfo.keys())
        template = batchinfo['template'].to(device)
        search = batchinfo['search'].to(device)
        cls_label = batchinfo['cls_label'].type(torch.FloatTensor).to(device)
        
        # Ocean
        reg_label = batchinfo['reg_label'].float().to(device) if 'reg_label' in batch_keys else None
        reg_weight = batchinfo['reg_weight'].float().to(device) if 'reg_weight' in batch_keys else None

        # OceanPlus
        template_mask = batchinfo['template_mask'].to(device) if 'template_mask' in batch_keys else None

        # AUtoMatch
        template_bbox = batchinfo['template_bbox'].to(device) if 'template_bbox' in batch_keys else None
        search_bbox = batchinfo['search_bbox'].to(device) if 'search_bbox' in batch_keys else None
        jitterBox = batchinfo['jitterBox'].float().to(device) if 'jitterBox' in batch_keys else None
        jitter_ious = batchinfo['jitter_ious'].float().to(device) if 'jitter_ious' in batch_keys else None

        model_inputs = {'template': template, 'search': search, 'cls_label': cls_label, 'reg_label': reg_label,
                        'reg_weight': reg_weight, 'template_bbox': template_bbox, 'search_bbox': search_bbox,
                        'template_mask': template_mask, 'jitterBox': jitterBox, 'jitter_ious': jitter_ious}
        import pdb
        pdb.set_trace()
        model_loss = model(model_inputs)
        cls_loss = torch.mean(model_loss['cls_loss'])
        # reg_loss = torch.mean(model_loss['reg_loss']) if 'reg_loss' in model_loss.keys() else None
        reg_loss = torch.mean(model_loss['reg_loss']) if 'reg_loss' in model_loss.keys() else None
        loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.REG_WEIGHT * reg_loss if reg_loss is not None else cls_loss
        loss = torch.mean(loss)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()

        if cfg.TRAIN.CLIP_GRAD:
            torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if loader.is_valid_number(loss.item()):
            optimizer.step()

        # record loss
        loss = loss.item()
        losses.update(loss, template.size(0))

        cls_loss = cls_loss.item()
        cls_losses.update(cls_loss, template.size(0))


        reg_loss = reg_loss.item() if reg_loss is not None else cls_loss
        reg_losses.update(reg_loss, template.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

    return model, _
