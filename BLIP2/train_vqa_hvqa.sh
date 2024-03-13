log=blip2/vqa_hvqa.log 
echo ${log} 

CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.run --master_port=6091 --nproc_per_node=3 train.py \
            --cfg-path=lavis/projects/blip2/train/uvqa_vqa.yaml > ${log} 2>&1
            