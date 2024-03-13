log=blip2/hvqa.log 
echo ${log} 

CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.run --master_port=6099 --nproc_per_node=3 lavis/train.py \
            --cfg-path=lavis/projects/blip2/train/hvqa.yaml > ${log} 2>&1
