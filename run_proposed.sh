models=("resnet34" "resnet18" "vgg13" "swins" "mobile050" "mobile100" "pvt")
datasets=("isic2017" "isic2018")
splits=(0 1 2 3 4)

for m in "${models[@]}"; do
    for d in "${datasets[@]}"; do
        for s in "${splits[@]}"; do
            CUDA_VISIBLE_DEVICES=0 python train_proposed_V5_adj_aux04_implement01_c_multi_models_noProg.py \
                --m "$m" --d "$d" --s "$s" --inner_dim 512 --input_size 224 \
                --gloss type1c --rloss mae --alpha 1.0 --beta 1.0 \
                --gtype Generator_Wrapper3 --mix_rate 0.3 --mix_ech 20 \
                --mask_thd 0.3 --mix_type hard
        done
    done
done
