#!/bin/bash

### Mini Dataset ###
MINI_OUTPUT_PREFIX='predictions/fungi_mini'
# baselines
python test_fungi.py --dataset mini --model efficientnet_b0 --checkpoint baselines_with_loss_mini/df2020_efficientnet_b0_ce_11-23-2021_13-25-42 --output $MINI_OUTPUT_PREFIX
python test_fungi.py --dataset mini --model vit_base_224 --checkpoint baselines_mini/df2020_vit_base_224_ce_11-04-2021_19-52-30 --output $MINI_OUTPUT_PREFIX

# convnets
python test_fungi.py --dataset mini --model efficientnet_b4 --checkpoint baselines_with_loss_mini/df2020_efficientnet_b4_ce_12-26-2021_15-10-20 --output $MINI_OUTPUT_PREFIX
python test_fungi.py --dataset mini --model efficientnet_b4_ns --checkpoint baselines_with_loss_mini/df2020_efficientnet_b4_ns_ce_12-26-2021_17-39-04 --output $MINI_OUTPUT_PREFIX
python test_fungi.py --dataset mini --model efficientnetv2_s --checkpoint baselines_with_loss_mini/df2020_efficientnetv2_s_ce_12-26-2021_20-07-20 --output $MINI_OUTPUT_PREFIX

# transformers
python test_fungi.py --dataset mini --model vit_base_384 --checkpoint baselines_mini/df2020_vit_base_384_ce_11-23-2021_22-36-27 --output $MINI_OUTPUT_PREFIX
python test_fungi.py --dataset mini --model deit_base_384 --checkpoint baselines_mini/df2020_deit_base_384_ce_11-24-2021_01-21-59 --output $MINI_OUTPUT_PREFIX
python test_fungi.py --dataset mini --model beit_base_384 --checkpoint baselines_mini/df2020_beit_base_384_ce_11-24-2021_04-07-33 --output $MINI_OUTPUT_PREFIX

python test_fungi.py --dataset mini --model vit_large_384 --checkpoint baselines_mini/df2020_vit_large_384_ce_11-26-2021_04-19-52 --output $MINI_OUTPUT_PREFIX


### Full Dataset ###
FULL_OUTPUT_PREFIX='predictions/fungi_full'
# baselines
python test_fungi.py --dataset full --model efficientnet_b0 --checkpoint baselines_with_loss/df2020_efficientnet_b0_ce_12-24-2021_13-17-47 --output $FULL_OUTPUT_PREFIX
python test_fungi.py --dataset full --model vit_base_224 --checkpoint baselines/df2020_vit_base_224_ce_11-29-2021_14-26-24 --output $FULL_OUTPUT_PREFIX

# convnets
python test_fungi.py --dataset full --model efficientnet_b4 --checkpoint baselines_with_loss/df2020_efficientnet_b4_ce_12-24-2021_13-17-14 --output $FULL_OUTPUT_PREFIX
python test_fungi.py --dataset full --model efficientnet_b4_ns --checkpoint baselines_with_loss/df2020_efficientnet_b4_ns_ce_12-25-2021_10-22-25 --output $FULL_OUTPUT_PREFIX
python test_fungi.py --dataset full --model efficientnetv2_s --checkpoint baselines_with_loss/df2020_efficientnetv2_s_ce_12-26-2021_06-59-07 --output $FULL_OUTPUT_PREFIX

# transformers
python test_fungi.py --dataset full --model vit_base_384 --checkpoint baselines/df2020_vit_base_384_ce_11-29-2021_15-53-09 --output $FULL_OUTPUT_PREFIX
python test_fungi.py --dataset full --model deit_base_384 --checkpoint baselines/df2020_deit_base_384_ce_11-30-2021_13-59-27 --output $FULL_OUTPUT_PREFIX
python test_fungi.py --dataset full --model beit_base_384 --checkpoint baselines/df2020_beit_base_384_ce_12-01-2021_09-35-05 --output $FULL_OUTPUT_PREFIX

python test_fungi.py --dataset full --model vit_large_384 --checkpoint baselines/df2020_vit_large_384_ce_12-01-2021_09-35-07 --output $FULL_OUTPUT_PREFIX
