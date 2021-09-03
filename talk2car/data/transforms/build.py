from . import transforms as T



def build_transforms(cfg, mode='train'):
    assert mode in ['train', 'test', 'val']
    min_size = cfg.SCALES[0]
    max_size = cfg.SCALES[1]
    assert min_size <= max_size

    if mode == 'train':
        flip_prob = cfg.TRAIN.FLIP_PROB
        affine_prob = cfg.TRAIN.AFFINE_PROB
    elif mode == 'test':
        flip_prob = cfg.TEST.FLIP_PROB
        affine_prob = 0
    else:
        flip_prob = cfg.VAL.FLIP_PROB
        affine_prob = 0

    to_bgr255 = True

    normalize_transform = T.Normalize(
        mean=cfg.NETWORK.PIXEL_MEANS, std=cfg.NETWORK.PIXEL_STDS, to_bgr255=to_bgr255
    )

    # transform = T.Compose(
    #     [
    #         T.Resize(min_size, max_size),
    #         T.RandomHorizontalFlip(flip_prob),
    #         T.ToTensor(),
    #         normalize_transform,
    #         T.FixPadding(min_size, max_size, pad=0)
    #     ]
    # )

    transform = T.Compose(
        [
            #T.Resize(min_size, max_size),
            T.RandomHorizontalFlip(flip_prob),
            T.RandomAffine(affine_prob, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            T.ToTensor(),
            normalize_transform,
        ]
    )

    return transform
