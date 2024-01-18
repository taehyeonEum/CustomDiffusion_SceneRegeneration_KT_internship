## launch training script 
## (2 GPUs recommended, increase --max_train_steps to 1000 if 1 GPU)

accelerate launch src/diffusers_training.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --output_dir=./logs/chris_pratt_gog_background_xxx  \
          --concepts_list=./assets/concept_list_gog.json \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=1000 \
          --num_class_images=200 \
          --scale_lr --hflip  \
          --modifier_token "<new1>+<new2>" 

## sample 
python src/diffusers_sample.py \
    --delta_ckpt logs/chris_pratt_gog_background/delta.bin \
    --ckpt "CompVis/stable-diffusion-v1-4" \
    --from-file "prompts/gog_chris_pratt_background.txt" \
    --keyword "base_setting" \