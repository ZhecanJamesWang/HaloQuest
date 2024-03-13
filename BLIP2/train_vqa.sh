log=blip2/vqa_instruct.log 
echo ${log} 

CUDA_VISIBLE_DEVICES=4,5,6 python -m torch.distributed.run --master_port=6097 --nproc_per_node=3 train.py \
            --cfg-path=./lavis/projects/blip2/train/vqa.yaml > ${log} 2>&1
