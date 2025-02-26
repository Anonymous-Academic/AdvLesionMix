models=("resnet34" "resnet18" "vgg13" "swins" "mobile050" "mobile100" "pvt")
datasets=("isic2017" "isic2018")
splits=(0 1 2 3 4)

for m in "${models[@]}"; do
    for d in "${datasets[@]}"; do
        for s in "${splits[@]}"; do
            CUDA_VISIBLE_DEVICES=1 python train_baseline.py --m "$m" --d "$d" --s "$s"
        done
    done
done
