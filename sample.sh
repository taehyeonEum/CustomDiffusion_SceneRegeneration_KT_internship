MODEL_NAME="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="./logs/check_bin"
CONCEPT_LIST="./assets/concept_list_jjanggu_scene2.json"
DELTA_CKPT="logs/jjanggu_and_scene9/delta-1500.bin"
FROM_FILE="prompts/jjanggu.txt"
KEYWORD="scene9_1500"


## sample 
python src/diffusers_sample.py \
    --delta_ckpt ${DELTA_CKPT} \
    --ckpt ${MODEL_NAME} \
    --from-file ${FROM_FILE} \
    --keyword ${KEYWORD} \
    --output_dir ${OUTPUT_DIR} \
    




MODEL_NAME="CompVis/stable-diffusion-v1-4"
OUTPUT_DIR="./logs/check_bin"
CONCEPT_LIST="./assets/concept_list_jjanggu_scene2.json"
DELTA_CKPT="logs/jjanggu_and_scene10/delta.bin"
FROM_FILE="prompts/jjanggu.txt"
KEYWORD="scene10_1500"


## sample 
python src/diffusers_sample.py \
    --delta_ckpt ${DELTA_CKPT} \
    --ckpt ${MODEL_NAME} \
    --from-file ${FROM_FILE} \
    --keyword ${KEYWORD} \
    --output_dir ${OUTPUT_DIR} \