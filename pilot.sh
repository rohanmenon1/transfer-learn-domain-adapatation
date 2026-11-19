#!/bin/bash
P=~/miniconda3/envs/tlda/bin/python
[ -f models/ft_dinov3_painting+real_pc50_s0.pt ] || ./run.sh logs/ft_dinov3_painting+real_pc50.log $P -u finetune.py --source painting real
[ -f models/soup_painting+real.pt ] || $P soup.py soup_painting+real ft_dinov3_painting_pc50_s0 ft_dinov3_real_pc50_s0
for t in clipart quickdraw; do
    for i in none soup_painting+real ft_dinov3_painting+real_pc50_s0; do
        ./run.sh logs/adapt_${i}_to-$t.log $P -u adapt.py --init $i --target $t
    done
done
