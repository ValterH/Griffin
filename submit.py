# %%
from roach.queue import Queue

# %%
date = "2025-07-30"
q = Queue(f"~/scratch/roach/queues/{date}-griffin")

# %%
################################################################################
# PRE-TRAIN
################################################################################

# %%

checkpoint_dict = {
    "rel-hm": "commerce-2"
}

all_pairs = [
    # clf
    # ("rel-amazon", "user-churn"),
    ("rel-hm", "user-churn"),
    # ("rel-stack", "user-badge"),
    # ("rel-amazon", "item-churn"),
    # ("rel-stack", "user-engagement"),
    # ("rel-avito", "user-visits"),
    # ("rel-avito", "user-clicks"),
    # ("rel-event", "user-ignore"),
    # ("rel-trial", "study-outcome"),
    # ("rel-f1", "driver-dnf"),
    # ("rel-event", "user-repeat"),
    # ("rel-f1", "driver-top3"),
    # reg
    ("rel-hm", "item-sales"),
    # ("rel-amazon", "user-ltv"),
    # ("rel-amazon", "item-ltv"),
    # ("rel-stack", "post-votes"),
    # ("rel-trial", "site-success"),
    # ("rel-trial", "study-adverse"),
    # ("rel-event", "user-attendance"),
    # ("rel-f1", "driver-position"),
    # ("rel-avito", "ad-ctr"),
]

# %%
for seed in [
    0,
    123,
    1234,
]:
    for dataset, task in all_pairs:
        for pretrain in [
            True, 
            False
        ]:
            cmd = rf"""
accelerate launch --config_file hconfig.yaml rt_comparison.py \
    datasets/joint-v65 logs/{dataset} {task} \
    --seed {seed} \
    --savepath results/{dataset}/{task} \
    --tasks {dataset}-{task} \
    --hop 2 \
    --fanout 20 \
    --maxepoch 50 \
    --patience 15 \
    --eval_per_epoch 50 \
    --batchsize 256 \
    --lr 3e-4 \
    --wd 2e-4 \
    --num_mp 4 \
    --use_rev True \
    --use_gate False \
    --fewshotfanout 3 \
    --hiddim 512 \
    --date {date} \
    --max_steps {2**13+1} \
    """
            if pretrain:
                ckpt_path = f"/lfs/local/0/valter/Griffin/checkpoints/single-sft/best_checkpoint/model.safetensors"
                cmd += f"--loadpath {ckpt_path}"
                chk = f"test -e {ckpt_path}"
            else:
                chk = "true"
            q.submit(cmd, chk)
            print(f"Submitted job to queue: {date}-griffin")
# %%
