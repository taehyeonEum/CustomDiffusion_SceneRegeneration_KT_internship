# # MODEL_NAME="CompVis/stable-diffusion-v1-4"
# # OUTPUT_DIR="./logs/chris_pratt_gog_background1_2"
# # CONCEPT_LIST="./assets/concept_list_gog.json"
# # DELTA_CKPT="logs/chris_pratt_gog_background1_2/delta.bin"
# # FROM_FILE="prompts/gog_chris_pratt_background3.txt"
# # KEYWORD="base_setting"


# # # accelerate launch src/diffusers_training.py \
# # #           --pretrained_model_name_or_path $MODEL_NAME  \
# # #           --output_dir ${OUTPUT_DIR}  \
# # #           --concepts_list ${CONCEPT_LIST} \
# # #           --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
# # #           --resolution=512  \
# # #           --train_batch_size=2  \
# # #           --learning_rate=1e-5  \
# # #           --lr_warmup_steps=0 \
# # #           --max_train_steps=1000 \
# # #           --num_class_images=200 \
# # #           --scale_lr --hflip  \
# # #           --modifier_token "<new1>+<new2>" 

# # ## sample 
# # python src/diffusers_sample.py \
# #     --delta_ckpt ${DELTA_CKPT} \
# #     --ckpt ${MODEL_NAME} \
# #     --from-file ${FROM_FILE} \
# #     --keyword ${KEYWORD} \

# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene8_2"
# CONCEPT_LIST="./assets/concept_list_jjanggu_scene.json"
# DELTA_CKPT="logs/jjanggu_and_scene8_2/delta.bin"
# FROM_FILE="prompts/jjanggu.txt"
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


# ## launch training script 
# ## (2 GPUs recommended, increase --max_train_steps to 1000 if 1 GPU)
# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene9"
# CONCEPT_LIST="./assets/concept_list_jjanggu_scene.json"
# DELTA_CKPT="logs/jjanggu_and_scene9/delta.bin"
# FROM_FILE="prompts/jjanggu.txt"
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
# #           --max_train_steps=2000 \
# #           --num_class_images=200 \
# #           --scale_lr --hflip  \
# #           --modifier_token "<new1>+<new2>" 

# ## sample 
# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \


# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene10"
# CONCEPT_LIST="./assets/concept_list_jjanggu_scene.json"
# DELTA_CKPT="logs/jjanggu_and_scene10/delta.bin"
# FROM_FILE="prompts/jjanggu.txt"
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
# #           --max_train_steps=1500 \
# #           --num_class_images=200 \
# #           --scale_lr --hflip  \
# #           --modifier_token "<new1>+<new2>" 

# ## sample 
# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \


# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene11"
# CONCEPT_LIST="./assets/concept_list_jjanggu_scene.json"
# DELTA_CKPT="logs/jjanggu_and_scene11/delta.bin"
# FROM_FILE="prompts/jjanggu.txt"
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
# #           --max_train_steps=1250 \
# #           --num_class_images=200 \
# #           --scale_lr --hflip  \
# #           --modifier_token "<new1>+<new2>" 

# ## sample 
# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \



# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene12"
# CONCEPT_LIST="./assets/concept_list_jjanggu_scene2.json"
# DELTA_CKPT="logs/jjanggu_and_scene12/delta.bin"
# FROM_FILE="prompts/jjanggu.txt"
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
# #           --max_train_steps=1500 \
# #           --num_class_images=200 \
# #           --scale_lr --hflip  \
# #           --modifier_token "<new1>+<new2>" 

# ## sample 
# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \


# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene13"
# CONCEPT_LIST="./assets/concept_list_jjanggu_scene13.json"
# DELTA_CKPT="logs/jjanggu_and_scene13/delta.bin"
# FROM_FILE="prompts/jjanggu.txt"
# KEYWORD="base_setting"        

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

# ## sample 
# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \


# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene13_1"
# CONCEPT_LIST="./assets/concept_list_jjanggu_scene13_1.json"
# DELTA_CKPT="logs/jjanggu_and_scene13_1/delta.bin"
# FROM_FILE="prompts/jjanggu.txt"
# KEYWORD="base_setting"

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

# ## sample 
# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \



# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene14"
# CONCEPT_LIST="./assets/concept_list_jjanggu_scene14.json"
# DELTA_CKPT="logs/jjanggu_and_scene14/delta.bin"
# FROM_FILE="prompts/jjanggu.txt"
# KEYWORD="base_setting"


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

# ## sample 
# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \


# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene15"
# CONCEPT_LIST="./assets/concept_list_jjanggu_scene15.json"
# DELTA_CKPT="logs/jjanggu_and_scene15/delta.bin"
# FROM_FILE="prompts/jjanggu.txt"
# KEYWORD="base_setting"


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

# ## sample 
# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \



# # ----------------------------------------- ex16

# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene16"
# CONCEPT_LIST="./assets/concept_list_jjanggu_scene16.json"

# # ##### fine-tuning #####
# # accelerate launch src/diffusers_training.py \
# #           --pretrained_model_name_or_path $MODEL_NAME  \
# #           --output_dir ${OUTPUT_DIR}  \
# #           --concepts_list ${CONCEPT_LIST} \
# #           --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
# #           --resolution=512  \
# #           --train_batch_size=2  \
# #           --learning_rate=1e-5  \
# #           --lr_warmup_steps=0 \
# #           --max_train_steps=2000 \
# #           --num_class_images=200 \
# #           --scale_lr --hflip  \
# #           --modifier_token "<new1>+<new2>" 


# ##### sample #####
# FROM_FILE="prompts/jjanggu2.txt"
# OUTPUT_DIR="./logs/jjanggu_and_scene16"
# DELTA_CKPT="logs/jjanggu_and_scene16/delta-500.bin"
# KEYWORD="scene16_500"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene16"
# DELTA_CKPT="logs/jjanggu_and_scene16/delta-1000.bin"
# KEYWORD="scene16_1000"
# FROM_FILE="prompts/jjanggu2.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene16"
# DELTA_CKPT="logs/jjanggu_and_scene16/delta-1500.bin"
# KEYWORD="scene16_1500"
# FROM_FILE="prompts/jjanggu2.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene16"
# DELTA_CKPT="logs/jjanggu_and_scene16/delta-2000.bin"
# FROM_FILE="prompts/jjanggu2.txt"
# KEYWORD="scene16_2000"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ----------------------------------------- ex9

# OUTPUT_DIR="./logs/jjanggu_and_scene9"
# CONCEPT_LIST="./assets/concept_list_jjanggu_scene16.json"

# ##### fine-tuning #####
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


# ##### sample #####
# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# FROM_FILE="prompts/jjanggu2.txt"
# OUTPUT_DIR="./logs/jjanggu_and_scene9"
# DELTA_CKPT="logs/jjanggu_and_scene9/delta-500.bin"
# KEYWORD="scene9_500"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene9"
# DELTA_CKPT="logs/jjanggu_and_scene9/delta-1000.bin"
# KEYWORD="scene9_1000"
# FROM_FILE="prompts/jjanggu2.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene9"
# DELTA_CKPT="logs/jjanggu_and_scene9/delta-1500.bin"
# KEYWORD="scene9_1500"
# FROM_FILE="prompts/jjanggu2.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene9"
# DELTA_CKPT="logs/jjanggu_and_scene9/delta-2000.bin"
# FROM_FILE="prompts/jjanggu2.txt"
# KEYWORD="scene9_2000"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \

# python src/run_concatenated_by_steps.py \
#     --file_path="logs/jjanggu_and_scene9" \
#     --output_path="logs/jjanggu_and_scene9" \
#     --keywords="scene9_500/scene9_1000/scene9_1500/scene9_2000" \

# ----------------------------------------- ex12

# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene12"
# CONCEPT_LIST="./assets/concept_list_jjanggu_scene16.json"

# ##### fine-tuning #####
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


# ##### sample #####
# FROM_FILE="prompts/jjanggu2.txt"
# OUTPUT_DIR="./logs/jjanggu_and_scene12"
# DELTA_CKPT="logs/jjanggu_and_scene12/delta-500.bin"
# KEYWORD="scene12_500"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene12"
# DELTA_CKPT="logs/jjanggu_and_scene12/delta-1000.bin"
# KEYWORD="scene12_1000"
# FROM_FILE="prompts/jjanggu2.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene12"
# DELTA_CKPT="logs/jjanggu_and_scene12/delta-1500.bin"
# KEYWORD="scene12_1500"
# FROM_FILE="prompts/jjanggu2.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \

# python src/run_concatenated_by_steps.py \
#     --file_path="logs/jjanggu_and_scene12" \
#     --output_path="logs/jjanggu_and_scene12" \
#     --keywords="scene12_500/scene12_1000/scene12_1500" \

##### sample #####
# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# FROM_FILE="prompts/jjanggu.txt"
# OUTPUT_DIR="./logs/jjanggu_and_scene9"
# DELTA_CKPT="logs/jjanggu_and_scene9/delta-500.bin"
# KEYWORD="scene9_500_prompt1"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene9"
# DELTA_CKPT="logs/jjanggu_and_scene9/delta-1000.bin"
# KEYWORD="scene9_1000_prompt1"
# FROM_FILE="prompts/jjanggu.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene9"
# DELTA_CKPT="logs/jjanggu_and_scene9/delta-1500.bin"
# KEYWORD="scene9_1500_prompt1"
# FROM_FILE="prompts/jjanggu.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene9"
# DELTA_CKPT="logs/jjanggu_and_scene9/delta-2000.bin"
# FROM_FILE="prompts/jjanggu.txt"
# KEYWORD="scene9_2000_prompt1"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \

# python src/run_concatenated_by_steps.py \
#     --file_path="logs/jjanggu_and_scene9" \
#     --output_path="logs/jjanggu_and_scene9" \
#     --keywords="scene9_500_prompt1/scene9_1000_prompt1/scene9_1500_prompt1/scene9_2000_prompt1" \


#### 13_1 to check im2 really good reference image_set by sampling via jjanggu2.txt .  
#### sample #####
# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# FROM_FILE="prompts/jjanggu2.txt"
# OUTPUT_DIR="./logs/jjanggu_and_scene13_1"
# DELTA_CKPT="logs/jjanggu_and_scene13_1/delta-500.bin"
# KEYWORD="scene13_1_500_prompt2"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene13_1"
# DELTA_CKPT="logs/jjanggu_and_scene13_1/delta-1000.bin"
# KEYWORD="scene13_1_1000_prompt2"
# FROM_FILE="prompts/jjanggu2.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene13_1"
# DELTA_CKPT="logs/jjanggu_and_scene13_1/delta-1500.bin"
# KEYWORD="scene13_1_1500_prompt2"
# FROM_FILE="prompts/jjanggu2.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \

# python src/run_concatenated_by_steps.py \
#     --file_path="logs/jjanggu_and_scene13_1" \
#     --output_path="logs/jjanggu_and_scene13_1" \
#     --keywords="scene13_1_500_prompt2/scene13_1_1000_prompt2/scene13_1_1500_prompt2" \


# #### 9_prompt2 to check 9's performance whem prompt set is jjanggu2.txt
# #### sample #####
# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# FROM_FILE="prompts/jjanggu2.txt"
# OUTPUT_DIR="./logs/jjanggu_and_scene9"
# DELTA_CKPT="logs/jjanggu_and_scene9/delta-500.bin"
# KEYWORD="scene9_500_prompt2"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene9"
# DELTA_CKPT="logs/jjanggu_and_scene9/delta-1000.bin"
# KEYWORD="scene9_1000_prompt2"
# FROM_FILE="prompts/jjanggu2.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene9"
# DELTA_CKPT="logs/jjanggu_and_scene9/delta-1500.bin"
# KEYWORD="scene9_1500_prompt2"
# FROM_FILE="prompts/jjanggu2.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \

# ##### sample #####
# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene9"
# DELTA_CKPT="logs/jjanggu_and_scene9/delta-2000.bin"
# KEYWORD="scene9_2000_prompt2"
# FROM_FILE="prompts/jjanggu2.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \

python src/run_concatenated_by_steps.py \
    --file_path="logs/jjanggu_and_scene9" \
    --output_path="logs/jjanggu_and_scene9" \
    --keywords="scene9_500_prompt2/scene9_1000_prompt2/scene9_1500_prompt2/scene9_2000_prompt2" \