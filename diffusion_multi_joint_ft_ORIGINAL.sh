## launch training script 
## (2 GPUs recommended, increase --max_train_steps to 1000 if 1 GPU)

accelerate launch src/diffusers_training.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --output_dir=./logs/cat_wooden_pot  \
          --concepts_list=./assets/concept_list.json \
          --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
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
    --delta_ckpt logs/cat_wooden_pot/delta.bin \
    --ckpt "CompVis/stable-diffusion-v1-4" \
    --prompt "<new1> cat sitting inside a <new2> wooden pot and looking up"