# Instructions for Talk2Car


### Environment
* Ubuntu 16.04, CUDA 9.0, GCC 4.9.4
* Python 3.6.x
    ```bash
    # We recommend you to use Anaconda/Miniconda to create a conda environment
    conda create -n vl-bert python=3.6 pip
    conda activate vl-bert
    ```
* PyTorch 1.0.0 or 1.1.0
    ```bash
    conda install pytorch=1.1.0 cudatoolkit=9.0 -c pytorch
    ```
* Apex (optional, for speed-up and fp16 training)
    ```bash
    git clone https://github.com/jackroos/apex
    cd ./apex
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./  
    ```
* Other requirements:
    ```bash
    pip install Cython
    pip install -r requirements.txt
    ```
* Compile
    ```bash
    ./scripts/init.sh
    ```

## Data

1. In the ``data``folder create a `talk2car` folder.
2. Create a `images` folder and `annotations` in the `talk2car` folder.
3. Download the images from [here](https://drive.google.com/file/d/1bhcdej7IFj5GqfvXGrHGPk2Knxe77pek/view).
4. In `annotations` put `talk2car_w_rpn_no_duplicates.json` which you can get from [here](https://github.com/talk2car/Talk2Car/blob/master/c4av_model/data/talk2car_w_rpn_no_duplicates.json).
5. Then run `preprocess_t2c_files_to_coco.py` at the root dir of VL-BERT-master

## Running

```
python scripts/launch.py \
  --nproc_per_node 1 \
   "talk2car/train_end2end.py" \
   --cfg "cfgs/talk2car/base_gt_boxes_4x16G.yaml" \
   --model-dir "t2c"
```

## Testing

```
python talk2car/test.py \
  --split test \
  --cfg cfgs/talk2car/base_gt_boxes_4x16G.yaml \
  --ckpt CHECKPOINT\
  --gpus 1 \
  --result-path result --result-name t2c_vlbert_test
```