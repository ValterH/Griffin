accelerate launch --config_file hconfig.yaml rt_comparison.py \
    datasets/joint-v65 logs/rel-hm user-churn \
    --loadpath /lfs/local/0/valter/Griffin/checkpoints/commerce-2/FULL/best_checkpoint/model.safetensors \
    --seed 0 \
    --savepath results/hm --tasks rel-hm-user-churn --hop 2 --fanout 20 \
    --maxepoch 50 --patience 15 --eval_per_epoch 2 \
    --batchsize 256 --lr 3e-4 --wd 2e-4 --num_mp 4 \
    --use_rev True --use_gate False --fewshotfanout 3 --hiddim 512 