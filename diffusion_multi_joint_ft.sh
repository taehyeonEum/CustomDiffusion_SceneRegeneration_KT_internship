# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/chris_pratt_gog_background1_2"
# CONCEPT_LIST="./assets/concept_list_gog.json"
# DELTA_CKPT="logs/chris_pratt_gog_background1_2/delta.bin"
# FROM_FILE="prompts/gog_chris_pratt_background3.txt"
# KEYWORD="base_setting"


# # accelerate launch src/diffusers_training.py \
# #           --pretrained_model_name_or_path $MODEL_NAME  \
# #           --output_dir ${OUTPUT_DIR}  \
# #           --concepts_list ${CONCEPT_LIST} \
# #           --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
# #           --resolution=512  \
# #           --train_batch_size=2  \
# #           --learning_rate=1e-5  \
# #           --lr_warmup_steps=0 \
# #           --max_train_steps=1000 \
# #           --num_class_images=200 \
# #           --scale_lr --hflip  \
# #           --modifier_token "<new1>+<new2>" 

# ## sample 
# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \

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


## launch training script 
## (2 GPUs recommended, increase --max_train_steps to 1000 if 1 GPU)
MODEL_NAME="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="./logs/jjanggu_and_scene9"
CONCEPT_LIST="./assets/concept_list_jjanggu_scene.json"
DELTA_CKPT="logs/jjanggu_and_scene9/delta.bin"
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
#           --max_train_steps=2000 \
#           --num_class_images=200 \
#           --scale_lr --hflip  \
#           --modifier_token "<new1>+<new2>" 

## sample 
python src/diffusers_sample.py \
    --delta_ckpt ${DELTA_CKPT} \
    --ckpt ${MODEL_NAME} \
    --from-file ${FROM_FILE} \
    --keyword ${KEYWORD} \


MODEL_NAME="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="./logs/jjanggu_and_scene10"
CONCEPT_LIST="./assets/concept_list_jjanggu_scene.json"
DELTA_CKPT="logs/jjanggu_and_scene10/delta.bin"
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
#           --max_train_steps=1500 \
#           --num_class_images=200 \
#           --scale_lr --hflip  \
#           --modifier_token "<new1>+<new2>" 

## sample 
python src/diffusers_sample.py \
    --delta_ckpt ${DELTA_CKPT} \
    --ckpt ${MODEL_NAME} \
    --from-file ${FROM_FILE} \
    --keyword ${KEYWORD} \


MODEL_NAME="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="./logs/jjanggu_and_scene11"
CONCEPT_LIST="./assets/concept_list_jjanggu_scene.json"
DELTA_CKPT="logs/jjanggu_and_scene11/delta.bin"
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
#           --max_train_steps=1250 \
#           --num_class_images=200 \
#           --scale_lr --hflip  \
#           --modifier_token "<new1>+<new2>" 

## sample 
python src/diffusers_sample.py \
    --delta_ckpt ${DELTA_CKPT} \
    --ckpt ${MODEL_NAME} \
    --from-file ${FROM_FILE} \
    --keyword ${KEYWORD} \



MODEL_NAME="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="./logs/jjanggu_and_scene12"
CONCEPT_LIST="./assets/concept_list_jjanggu_scene2.json"
DELTA_CKPT="logs/jjanggu_and_scene12/delta.bin"
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
#           --max_train_steps=1500 \
#           --num_class_images=200 \
#           --scale_lr --hflip  \
#           --modifier_token "<new1>+<new2>" 

## sample 
python src/diffusers_sample.py \
    --delta_ckpt ${DELTA_CKPT} \
    --ckpt ${MODEL_NAME} \
    --from-file ${FROM_FILE} \
    --keyword ${KEYWORD} \