MODEL_NAME="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="./logs/jjanggu_and_scene8_2"
CONCEPT_LIST="./assets/concept_list_jjanggu_scene.json"
DELTA_CKPT="logs/jjanggu_and_scene8_2/delta.bin"
FROM_FILE="prompts/jjanggu.txt"
KEYWORD="base_setting"


# accelerate launch src/diffusers_training.py \
#           --pretrained_model_name_or_path $MODEL_NAME  \
#           --output_dir ${OUTPUT_DIR}  \
#           --concepts_list ${CONCEPT_LIST} \
#           --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
#           --resolution=512  \
#           --train_batch_size=2  \
#           --learning_rate=1e-5  \
#           --lr_warmup_steps=0 \
#           --max_train_steps=1000 \
#           --num_class_images=200 \
#           --scale_lr --hflip  \
#           --modifier_token "<new1>+<new2>" 

## sample 
python src/diffusers_sample.py \
    --delta_ckpt ${DELTA_CKPT} \
    --ckpt ${MODEL_NAME} \
    --from-file ${FROM_FILE} \
    --keyword ${KEYWORD} \