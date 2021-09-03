python talk2car/test.py \
  --split test \
  --cfg cfgs/talk2car/base_gt_boxes_4x16G.yaml \
  --ckpt /cw/liir/NoCsBack/testliir/thierry/PathProjection/VL-BERT-master/t2c/output/vl-bert/t2c/base_gt_boxes_4x16G/train_train/vl-bert_base_res101_t2c-best.model\
  --gpus 1 \
  --result-path result --result-name t2c_vlbert_test
