## launch training script (2 GPUs recommended, increase --max_train_steps to 500 if 1 GPU)
MODEL_NAME="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="./logs/jjanggu3"
DELTA_CKPT="./logs/jjanggu3/delta.bin"
FROM_FILE="./prompts/jjanggu.txt"
KEYWORD="base_setting"

accelerate launch src/diffusers_training.py \
          --pretrained_model_name_or_path=$MODEL_NAME  \
          --instance_data_dir=./data/jjanggu/im  \
          --class_data_dir=./real_reg/samples_boy_cartoon_character \
          --output_dir ${OUTPUT_DIR}  \
          --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
          --instance_prompt="cartoon of a <new1> boy character"  \
          --class_prompt="boy" \
          --resolution=512  \
          --train_batch_size=2  \
          --learning_rate=1e-5  \
          --lr_warmup_steps=0 \
          --max_train_steps=500 \
          --num_class_images=200 \
          --scale_lr --hflip  \
          --modifier_token "<new1>"

## sample 
python src/diffusers_sample.py --delta_ckpt ${DELTA_CKPT} --ckpt ${MODEL_NAME} --from-file ${FROM_FILE} --keyword ${KEYWORD}