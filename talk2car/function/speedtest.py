import os
import pprint
import shutil

import json
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn.functional as F

from common.utils.load import smart_load_model_state_dict
from common.trainer import to_cuda
from common.utils.create_logger import create_logger
from talk2car.data.build import make_dataloader
from talk2car.modules import *

POSITIVE_THRESHOLD = 0.5


def cacluate_iou(pred_boxes, gt_boxes):
    x11, y11, x12, y12 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
    x21, y21, x22, y22 = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
    xA = np.maximum(x11, x21)
    yA = np.maximum(y11, y21)
    xB = np.minimum(x12, x22)
    yB = np.minimum(y12, y22)
    interArea = (xB - xA + 1).clip(0) * (yB - yA + 1).clip(0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + boxBArea - interArea)

    return iou


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def test_net(args, config):
    print('test net...')
    pprint.pprint(args)
    pprint.pprint(config)
    device_ids = [int(d) for d in config.GPUS.split(',')]
    # os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS
    config.DATASET.TEST_IMAGE_SET = args.split
    ckpt_path = args.ckpt
    save_path = args.result_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # shutil.copy2(ckpt_path,
    #              os.path.join(save_path, '{}_test_ckpt_{}.model'.format(config.MODEL_PREFIX, config.DATASET.TASK)))

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # get network
    model = eval(config.MODULE)(config)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        torch.cuda.set_device(device_ids[0])
        model = model.cuda()
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    smart_load_model_state_dict(model, checkpoint['state_dict'])

    # Check number of params
    print("number of params: ", count_parameters(model))

    # loader
    test_loader = make_dataloader(config, mode='test', distributed=False)
    test_dataset = test_loader.dataset
    test_database = test_dataset.database

    # test
    # ref_ids = []
    token = []
    pred_boxes = []
    model.eval()
    cur_id = 0
    result = {}
    import time
    t_final = 0
    print(len(test_loader))
    total_tally = 0
    for nbatch, batch in zip(trange(len(test_loader)), test_loader):
        # for nbatch, batch in tqdm(enumerate(test_loader)):
        # bs = test_loader.batch_sampler.batch_size if test_loader.batch_sampler is not None else test_loader.batch_size
        # ref_ids.extend([test_database[id]['ref_id'] for id in range(cur_id, min(cur_id + bs, len(test_database)))])
        image, boxes, im_info, exp_ids, index = batch
        batch = to_cuda((image, boxes, im_info, exp_ids))
        t1 = time.time()
        output = model(*batch)
        t_final += time.time() - t1
        total_tally += len(image)

        # pred_boxes.extend(output['pred_boxes'].detach().cpu().tolist())
        # cur_id += bs
        boxes = output['pred_boxes'].detach().cpu().tolist()
        for i_, box_ in enumerate(boxes):
            token = test_dataset.get_command_token(index[i_])
            if token in result.keys():
                print('Token already exists')
            box = [box_[0], box_[1] + 100, box_[2] - box_[0], box_[3] - box_[1]]
            result[token] = box

    # result = [{'ref_id': ref_id, 'box': box} for ref_id, box in zip(ref_ids, pred_boxes)]
    print("Average time : {}".format(t_final / total_tally))
    result_json_path = os.path.join(save_path, '{}_talk2car_{}.json'.format(
        config.MODEL_PREFIX if args.result_name is None else args.result_name, config.DATASET.TEST_IMAGE_SET))
    with open(result_json_path, 'w') as f:
        json.dump(result, f)
    print('result json saved to {}.'.format(result_json_path))

    return result_json_path
