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

# python src/run_concatenated_by_steps.py \
#     --file_path="logs/jjanggu_and_scene9" \
#     --output_path="logs/jjanggu_and_scene9" \
#     --keywords="scene9_500_prompt2/scene9_1000_prompt2/scene9_1500_prompt2/scene9_2000_prompt2" \



# # ----------------------------------------- ex17

# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene17"
# CONCEPT_LIST="./assets/concept_list_jjanggu_scene17.json"

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
# OUTPUT_DIR="./logs/jjanggu_and_scene17"
# DELTA_CKPT="logs/jjanggu_and_scene17/delta-500.bin"
# KEYWORD="scene17_500"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene17"
# DELTA_CKPT="logs/jjanggu_and_scene17/delta-1000.bin"
# KEYWORD="scene17_1000"
# FROM_FILE="prompts/jjanggu2.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene17"
# DELTA_CKPT="logs/jjanggu_and_scene17/delta-1500.bin"
# KEYWORD="scene17_1500"
# FROM_FILE="prompts/jjanggu2.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene17"
# DELTA_CKPT="logs/jjanggu_and_scene17/delta-2000.bin"
# FROM_FILE="prompts/jjanggu2.txt"
# KEYWORD="scene17_2000"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \

# python src/run_concatenated_by_steps.py \
#     --file_path="logs/jjanggu_and_scene17" \
#     --output_path="logs/jjanggu_and_scene17" \
#     --keywords="scene17_500/scene17_1000/scene17_1500/scene17_2000" \

# ---------------------------------- no ex_num experiment. 

# ##### sample #####
# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene17"
# DELTA_CKPT="logs/jjanggu_and_scene17/delta-2000.bin"
# FROM_FILE="prompts/jjanggu2.txt"
# KEYWORD="scene17_2000_2"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \
#     --batch_size 2


# ##### sample #####
# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene17"
# DELTA_CKPT="logs/jjanggu_and_scene17/delta-2000.bin"
# FROM_FILE="prompts/jjanggu2.txt"
# KEYWORD="scene17_2000_10"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \
#     --batch_size 10

# # ----------------------------------------- ex18

# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene18"
# CONCEPT_LIST="./assets/concept_list_jjanggu_scene18.json"

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
#           --max_train_steps=2500 \
#           --num_class_images=200 \
#           --scale_lr --hflip  \
#           --modifier_token "<new1>+<new2>+<new3>" 


# ##### sample #####
# FROM_FILE="prompts/jjanggu18.txt"
# OUTPUT_DIR="./logs/jjanggu_and_scene18"
# DELTA_CKPT="logs/jjanggu_and_scene18/delta-500.bin"
# KEYWORD="scene18_500"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene18"
# DELTA_CKPT="logs/jjanggu_and_scene18/delta-1000.bin"
# KEYWORD="scene18_1000"
# FROM_FILE="prompts/jjanggu18.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene18"
# DELTA_CKPT="logs/jjanggu_and_scene18/delta-1500.bin"
# KEYWORD="scene18_1500"
# FROM_FILE="prompts/jjanggu18.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene18"
# DELTA_CKPT="logs/jjanggu_and_scene18/delta-2000.bin"
# FROM_FILE="prompts/jjanggu18.txt"
# KEYWORD="scene18_2000"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \

# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene18"
# DELTA_CKPT="logs/jjanggu_and_scene18/delta-2500.bin"
# FROM_FILE="prompts/jjanggu18.txt"
# KEYWORD="scene18_2500"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \

# python src/run_concatenated_by_steps.py \
#     --file_path="logs/jjanggu_and_scene18" \
#     --output_path="logs/jjanggu_and_scene18" \
#     --keywords="scene18_500/scene18_1000/scene18_1500/scene18_2000/scene18_2500" \

# -----------------------------------------ex_17 sdxl

# # MODEL_NAME="CompVis/stable-diffusion-v1-4"
# MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# OUTPUT_DIR="./logs/jjanggu_and_scene17_sdxl_sdxl"
# CONCEPT_LIST="./assets/concept_list_jjanggu_scene17.json"

# ##### fine-tuning #####
# accelerate launch src/diffusers_training_sdxl.py \
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
# OUTPUT_DIR="./logs/jjanggu_and_scene17_sdxl"
# DELTA_CKPT="logs/jjanggu_and_scene17_sdxl/delta-500.bin"
# KEYWORD="xl_scene17_500"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene17_sdxl"
# DELTA_CKPT="logs/jjanggu_and_scene17_sdxl/delta-1000.bin"
# KEYWORD="xl_scene17_1000"
# FROM_FILE="prompts/jjanggu2.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene17_sdxl"
# DELTA_CKPT="logs/jjanggu_and_scene17_sdxl/delta-1500.bin"
# KEYWORD="xl_scene17_1500"
# FROM_FILE="prompts/jjanggu2.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene17_sdxl"
# DELTA_CKPT="logs/jjanggu_and_scene17_sdxl/delta-2000.bin"
# FROM_FILE="prompts/jjanggu2.txt"
# KEYWORD="xl_scene17_2000"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \

# python src/run_concatenated_by_steps.py \
#     --file_path="logs/jjanggu_and_scene17_sdxl" \
#     --output_path="logs/jjanggu_and_scene17_sdxl" \
#     --keywords="xl_scene17_500/xl_scene17_1000/xl_scene17_1500/xl_scene17_2000" \

# -------------------------------------------- not ex_num experiments

# ##### sample #####
# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene17"
# DELTA_CKPT="logs/jjanggu_and_scene17/delta-2000.bin"
# FROM_FILE="prompts/jjanggu2.txt"
# KEYWORD="scene17_2000_2"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \
#     --batch_size 2


# ##### sample #####
# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene17"
# DELTA_CKPT="logs/jjanggu_and_scene17/delta-2000.bin"
# FROM_FILE="prompts/jjanggu2.txt"
# KEYWORD="scene17_2000_10"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \
#     --batch_size 10


# # ----------------------------------------- ex19 not done yet..!!

# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene18"
# CONCEPT_LIST="./assets/concept_list_jjanggu_scene18.json"

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
#           --modifier_token "<new1>+<new2>+<new3>" 


# ##### sample #####
# FROM_FILE="prompts/jjanggu19.txt"
# OUTPUT_DIR="./logs/jjanggu_and_scene19"
# DELTA_CKPT="logs/jjanggu_and_scene19/delta-500.bin"
# KEYWORD="scene19_500"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene19"
# DELTA_CKPT="logs/jjanggu_and_scene19/delta-1000.bin"
# KEYWORD="scene19_1000"
# FROM_FILE="prompts/jjanggu19.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene19"
# DELTA_CKPT="logs/jjanggu_and_scene19/delta-1500.bin"
# KEYWORD="scene19_1500"
# FROM_FILE="prompts/jjanggu19.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene19"
# DELTA_CKPT="logs/jjanggu_and_scene19/delta-2000.bin"
# FROM_FILE="prompts/jjanggu19.txt"
# KEYWORD="scene19_2000"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \

# ##### sample #####
# OUTPUT_DIR="./logs/jjanggu_and_scene19"
# DELTA_CKPT="logs/jjanggu_and_scene19/delta-2500.bin"
# FROM_FILE="prompts/jjanggu19.txt"
# KEYWORD="scene19_2500"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \

# python src/run_concatenated_by_steps.py \
#     --file_path="logs/jjanggu_and_scene19" \
#     --output_path="logs/jjanggu_and_scene19" \
#     --keywords="scene19_500/scene19_1000/scene19_1500/scene19_2000/scene19_2500" \

# # ----------------------------------------- ex17_crossattn

# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene17_crossattn"
# CONCEPT_LIST="./assets/concept_list_jjanggu_scene17.json"
# FROM_FILE="prompts/jjanggu2.txt"

# ##### fine-tuning #####
# accelerate launch src/diffusers_training.py \
#           --pretrained_model_name_or_path $MODEL_NAME  \
#           --output_dir ${OUTPUT_DIR}  \
#           --concepts_list ${CONCEPT_LIST} \
#           --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
#           --resolution=512  \
#           --train_batch_size=6  \
#           --learning_rate=1e-5  \
#           --lr_warmup_steps=0 \
#           --max_train_steps=2000 \
#           --num_class_images=200 \
#           --scale_lr --hflip  \
#           --modifier_token "<new1>+<new2>" \
#           --freeze_model "crossattn" \


##### sample #####
# DELTA_CKPT="logs/jjanggu_and_scene17_crossattn/delta-500.bin"
# KEYWORD="scene17_500"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


##### sample #####
# DELTA_CKPT="logs/jjanggu_and_scene17_crossattn/delta-1000.bin"
# KEYWORD="scene17_1000"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# DELTA_CKPT="logs/jjanggu_and_scene17_crossattn/delta-1500.bin"
# KEYWORD="scene17_1500"


# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


##### sample #####
# DELTA_CKPT="logs/jjanggu_and_scene17_crossattn/delta-2000.bin"
# KEYWORD="scene17_2000"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \

# python src/run_concatenated_by_steps.py \
#     --file_path="logs/jjanggu_and_scene17_crossattn" \
#     --output_path="logs/jjanggu_and_scene17_crossattn" \
#     --keywords="scene17_500/scene17_1000/scene17_1500/scene17_2000" \
#     --image_name="concatenated_by_step_crossattn" \

# # ----------------------------------------- ex17_ with G earth

# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene17_batch_6"
# CONCEPT_LIST="./assets/concept_list_jjanggu_scene17.json"
# FROM_FILE="prompts/jjanggu2.txt"

# ##### fine-tuning #####
# # accelerate launch src/diffusers_training.py \
# #           --pretrained_model_name_or_path $MODEL_NAME  \
# #           --output_dir ${OUTPUT_DIR}  \
# #           --concepts_list ${CONCEPT_LIST} \
# #           --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
# #           --resolution=512  \
# #           --train_batch_size=6  \
# #           --learning_rate=1e-5  \
# #           --lr_warmup_steps=0 \
# #           --max_train_steps=2000 \
# #           --num_class_images=200 \
# #           --scale_lr --hflip  \
# #           --modifier_token "<new1>+<new2>" \

# #### sample #####
# DELTA_CKPT="logs/jjanggu_and_scene17_batch_6/delta-500.bin"
# KEYWORD="scene17_500"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# #### sample #####
# DELTA_CKPT="logs/jjanggu_and_scene17_batch_6/delta-1000.bin"
# KEYWORD="scene17_1000"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# DELTA_CKPT="logs/jjanggu_and_scene17_batch_6/delta-1500.bin"
# KEYWORD="scene17_1500"


# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# #### sample #####
# DELTA_CKPT="logs/jjanggu_and_scene17_batch_6/delta-2000.bin"
# KEYWORD="scene17_2000"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \

# python src/run_concatenated_by_steps.py \
#     --file_path ${OUTPUT_DIR} \
#     --output_path ${OUTPUT_DIR} \
#     --keywords="scene17_500/scene17_1000/scene17_1500/scene17_2000" \
#     --image_name="concatenated_by_step" \

# ------------------------------------- jjanggu_and_scene_19 train only style! 

# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# INSTANCE_DIR="./data/jjanggu_scene/im_f"
# REFERENCE_DIR="./real_reg/samples_cartoon"
# OUTPUT_DIR="./logs/jjanggu_and_scene19"
# FROM_FILE="prompts/jjanggu19.txt"

# accelerate launch src/diffusers_training.py \
#           --pretrained_model_name_or_path=$MODEL_NAME  \
#           --instance_data_dir ${INSTANCE_DIR}  \
#           --class_data_dir ${REFERENCE_DIR} \
#           --output_dir ${OUTPUT_DIR}  \
#           --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
#           --instance_prompt="image of a <new2> style"  \
#           --class_prompt="style" \
#           --resolution=512  \
#           --train_batch_size=6  \
#           --learning_rate=1e-5  \
#           --lr_warmup_steps=0 \
#           --max_train_steps=2000 \
#           --num_class_images=200 \
#           --scale_lr --hflip  \
#           --modifier_token "<new2>" \
#           --initializer_token "pll" \

# #### sample #####
# DELTA_CKPT="logs/jjanggu_and_scene19/delta-500.bin"
# KEYWORD="scene17_500"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# #### sample #####
# DELTA_CKPT="logs/jjanggu_and_scene19/delta-1000.bin"
# KEYWORD="scene17_1000"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# DELTA_CKPT="logs/jjanggu_and_scene19/delta-1500.bin"
# KEYWORD="scene17_1500"


# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# #### sample #####
# DELTA_CKPT="logs/jjanggu_and_scene19/delta-2000.bin"
# KEYWORD="scene17_2000"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \

# python src/run_concatenated_by_steps.py \
#     --file_path ${OUTPUT_DIR} \
#     --output_path ${OUTPUT_DIR} \
#     --keywords="scene19_500/scene19_1000/scene19_1500/scene19_2000" \
#     --image_name="concatenated_by_step" \

# ------------------------------------- jjanggu_and_scene18_G_earth train 3 object!

# MODEL_NAME="CompVis/stable-diffusion-v1-4"
# OUTPUT_DIR="./logs/jjanggu_and_scene18_G_earth"
# CONCEPT_LIST="./assets/concept_list_jjanggu_scene18.json"
# FROM_FILE="prompts/jjanggu18.txt"

# ##### fine-tuning #####
# accelerate launch src/diffusers_training.py \
#           --pretrained_model_name_or_path $MODEL_NAME  \
#           --output_dir ${OUTPUT_DIR}  \
#           --concepts_list ${CONCEPT_LIST} \
#           --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
#           --resolution=512  \
#           --train_batch_size=6  \
#           --learning_rate=1e-5  \
#           --lr_warmup_steps=0 \
#           --max_train_steps=2500 \
#           --num_class_images=200 \
#           --scale_lr --hflip  \
#           --modifier_token "<new1>+<new2>+<new3>" 


# ##### sample #####
# DELTA_CKPT="logs/jjanggu_and_scene18_G_earth/delta-500.bin"
# KEYWORD="scene18_500"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# DELTA_CKPT="logs/jjanggu_and_scene18_G_earth/delta-1000.bin"
# KEYWORD="scene18_1000"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# DELTA_CKPT="logs/jjanggu_and_scene18_G_earth/delta-1500.bin"
# KEYWORD="scene18_1500"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \


# ##### sample #####
# DELTA_CKPT="logs/jjanggu_and_scene18_G_earth/delta-2000.bin"
# KEYWORD="scene18_2000"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \

# ##### sample #####
# DELTA_CKPT="logs/jjanggu_and_scene18_G_earth/delta-2500.bin"
# KEYWORD="scene18_2500"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \

# python src/run_concatenated_by_steps.py \
#     --file_path="logs/jjanggu_and_scene18_G_earth" \
#     --output_path="logs/jjanggu_and_scene18_G_earth" \
#     --keywords="scene18_500/scene18_1000/scene18_1500/scene18_2000/scene18_2500" \

# ---------- to get no finetuned model

MODEL_NAME="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="./logs/SD_no_finetuned"
CONCEPT_LIST="./assets/concept_list_jjanggu_scene18.json"

##### fine-tuning #####
# accelerate launch src/diffusers_training.py \
#           --pretrained_model_name_or_path $MODEL_NAME  \
#           --output_dir ${OUTPUT_DIR}  \
#           --concepts_list ${CONCEPT_LIST} \
#           --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
#           --resolution=512  \
#           --train_batch_size=6  \
#           --learning_rate=1e-100  \
#           --lr_warmup_steps=0 \
#           --max_train_steps=2500 \
#           --num_class_images=200 \
#           --scale_lr --hflip  \
#           --modifier_token "<new1>+<new2>+<new3>" \
#           --save_steps 1 \

# #### sample #####
# DELTA_CKPT="logs/SD_no_finetuned/delta-1.bin"
# KEYWORD="scene17_1"
# FROM_FILE="prompts/jjanggu2.txt"

# python src/diffusers_sample.py \
#     --delta_ckpt ${DELTA_CKPT} \
#     --ckpt ${MODEL_NAME} \
#     --from-file ${FROM_FILE} \
#     --keyword ${KEYWORD} \
#     --output_dir ${OUTPUT_DIR} \

DELTA_CKPT="logs/SD_no_finetuned/delta-1.bin"
KEYWORD="scene17_1_plain"
FROM_FILE="prompts/jjanggu2_plain.txt"

python src/diffusers_sample.py \
    --delta_ckpt ${DELTA_CKPT} \
    --ckpt ${MODEL_NAME} \
    --from-file ${FROM_FILE} \
    --keyword ${KEYWORD} \
    --output_dir ${OUTPUT_DIR} \
