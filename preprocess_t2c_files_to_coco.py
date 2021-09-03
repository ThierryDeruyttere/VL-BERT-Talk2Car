import json
import os
import sys

if __name__ == "__main__":
    splits = ['train', 'val', 'test']
    box_ind = 0
    idx = 0

    categories = []
    cat2ind = {}
    with open(os.path.join('data', 'talk2car', 'annotations', 'talk2car_w_rpn_no_duplicates.json'), 'rb') as f:
        data = json.load(f)
        for split in splits:

            data_split = data[split]

            coco = {
                'images': [],
                'annotations': [],
                'categories': [],
            }

            ref = []
            proposal = []
            for k, v in data_split.items():
                # print(k)
                coco['images'].append(
                    {
                        'file_name': v['img'],
                        'id': idx,
                        'width': 1600,
                        'height': 900,
                    }
                )

                ref.append(
                    {
                        'sent_ids': idx,
                        'file_name': v['img'],
                        'ann_id': idx,
                        'ref_id': idx,
                        'image_id': idx,
                        'split': split,
                        'sentences': [{
                            'tokens': v['tokenized_command'],
                            'command_token': v['command_token'],
                            'raw': v['command'],
                            'sent_id': idx,
                            'sent': v['command'],
                        }],
                        'category_id': 1
                    }
                )
                for box_dict in v['centernet']:
                    full_name = box_dict['class']
                    cat_split = full_name.split('.')
                    cat_name = cat_split[-1]
                    try:
                        supercategory = cat_split[-2]
                    except:
                        supercategory = cat_split[-1]
                    if cat_name not in cat2ind.keys():
                        cat_ind = len(categories)
                        categories.append({
                            'supercategory': supercategory,
                            'name': cat_name,
                            'id': cat_ind,
                        })
                        cat2ind[cat_split[-1]] = cat_ind
                    proposal.append(
                        {
                            'box': box_dict['bbox'],
                            'h5_id': box_ind,
                            'det_id': box_ind,
                            'image_id': idx,
                            'score': box_dict['score'],
                            'category_id': cat2ind[cat_name],
                            'category_name': cat_name,
                        }
                    )
                    box_ind = box_ind + 1
                if split != 'test' and split != "c4av":
                    coco['annotations'].append(
                        {
                            'image_id': idx,
                            'category_id': 255,
                            'bbox': v['referred_object'],
                            'area': v['referred_object'][2] * v['referred_object'][3],
                            'iscrowd': 0,
                            'id': idx,
                            # 'command': v['command'],
                            # 'command_token': v['command_token'],
                            # 'tokenized_command': v['tokenized_command'],
                        }
                    )
                else:
                    coco['annotations'].append(
                        {
                            'image_id': idx,
                            'iscrowd': 0,
                            'id': idx,
                            'category_id': 255,
                            # 'command': v['command'],
                            # 'command_token': v['command_token'],
                            # 'tokenized_command': v['tokenized_command'],
                        }
                    )
                idx = idx + 1

            coco['categories'] = categories

            with open(os.path.join('data', 'talk2car', 'annotations', 'annotations_{}.json'.format(split)), 'w') as fw:
                json.dump(coco, fw)
            with open(os.path.join('data', 'talk2car', 'annotations', 'proposal_{}.json'.format(split)), 'w') as fw:
                json.dump(proposal, fw)
            with open(os.path.join('data', 'talk2car', 'annotations', 'ref_{}.json'.format(split)), 'w') as fw:
                json.dump(ref, fw)
        print('Done, Total {} proposals, {} images'.format(box_ind, idx))
