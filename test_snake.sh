#!/bin/bash

### Mini Dataset ###
MINI_OUTPUT_PREFIX='predictions/snake_mini'
# baselines
python test_snake.py --model efficientnet_b0 --checkpoint baselines_mini/clef2021_efficientnet_b0_ce_11-10-2021_11-09-53 --output $MINI_OUTPUT_PREFIX
python test_snake.py --model vit_base_224 --checkpoint baselines_with_loss_mini/clef2021_vit_base_224_ce_11-25-2021_21-43-37 --output $MINI_OUTPUT_PREFIX

# convnets
python test_snake.py --model efficientnet_b4 --checkpoint baselines_mini/clef2021_efficientnet_b4_ce_11-23-2021_15-22-07 --output $MINI_OUTPUT_PREFIX
python test_snake.py --model efficientnet_b4_ns --checkpoint baselines_mini/clef2021_efficientnet_b4_ns_ce_11-28-2021_15-20-31 --output $MINI_OUTPUT_PREFIX
python test_snake.py --model efficientnetv2_s --checkpoint baselines_mini/clef2021_efficientnetv2_s_ce_11-23-2021_22-45-37 --output $MINI_OUTPUT_PREFIX

# transformers
python test_snake.py --model vit_base_384 --checkpoint baselines_with_loss_mini/clef2021_vit_base_384_ce_12-20-2021_18-12-36 --output $MINI_OUTPUT_PREFIX
python test_snake.py --model deit_base_384 --checkpoint baselines_with_loss_mini/clef2021_deit_base_384_ce_12-21-2021_01-03-57 --output $MINI_OUTPUT_PREFIX
python test_snake.py --model beit_base_384 --checkpoint baselines_with_loss_mini/clef2021_beit_base_384_ce_12-21-2021_07-57-06 --output $MINI_OUTPUT_PREFIX

python test_snake.py --model vit_large_384 --checkpoint baselines_with_loss_mini/clef2021_vit_large_384_ce_12-21-2021_15-41-06 --output $MINI_OUTPUT_PREFIX


### Full Dataset ###
FULL_OUTPUT_PREFIX='predictions/snake_full'
# baselines
python test_snake.py --model efficientnet_b0 --checkpoint baselines/clef2021_efficientnet_b0_ce_11-29-2021_16-52-38 --output $FULL_OUTPUT_PREFIX
python test_snake.py --model vit_base_224 --checkpoint baselines_with_loss/clef2021_vit_base_224_ce_12-22-2021_15-35-04 --output $FULL_OUTPUT_PREFIX

# convnets
python test_snake.py --model efficientnet_b4 --checkpoint baselines/clef2021_efficientnet_b4_ce_12-01-2021_21-42-26 --output $FULL_OUTPUT_PREFIX
python test_snake.py --model efficientnet_b4_ns --checkpoint baselines/clef2021_efficientnet_b4_ns_ce_12-01-2021_09-37-32 --output $FULL_OUTPUT_PREFIX
python test_snake.py --model efficientnetv2_s --checkpoint baselines/clef2021_efficientnetv2_s_ce_12-02-2021_14-00-43 --output $FULL_OUTPUT_PREFIX

# transformers
python test_snake.py --model vit_base_384 --checkpoint baselines_with_loss/clef2021_vit_base_384_ce_12-23-2021_05-56-12 --output $FULL_OUTPUT_PREFIX
python test_snake.py --model deit_base_384 --checkpoint baselines_with_loss/clef2021_deit_base_384_ce_12-22-2021_15-35-07 --output $FULL_OUTPUT_PREFIX
python test_snake.py --model beit_base_384 --checkpoint baselines_with_loss/clef2021_beit_base_384_ce_12-23-2021_22-51-08 --output $FULL_OUTPUT_PREFIX

python test_snake.py --model vit_large_384 --checkpoint baselines_with_loss/clef2021_vit_large_384_ce_12-24-2021_13-12-48 --output $FULL_OUTPUT_PREFIX
