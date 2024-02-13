# example.
# python evaluate.py --sample_root {folder} --target_path {target-folder} --numgen {numgen}

# echo "----------------------------------------------------------------------"
# echo "-------------------------scene17_500----------------------------------"
# echo "----------------------------------------------------------------------"
# FOLDER="./logs/jjanggu_and_scene17_batch_6/scene17_500"
# TARGET_FOLDER="./data/jjanggu2/im_f"
# NUMGEN="50"
# OUTPKL="./eval/jjanggu_and_scene17_batch_6/evaluation_500.pkl"

# python evaluate.py --sample_root ${FOLDER} --target_path ${TARGET_FOLDER} --numgen ${NUMGEN} --outpkl ${OUTPKL}


# echo "----------------------------------------------------------------------"
# echo "-------------------------scene17_1000---------------------------------"
# echo "----------------------------------------------------------------------"
# FOLDER="./logs/jjanggu_and_scene17_batch_6/scene17_1000"
# TARGET_FOLDER="./data/jjanggu2/im_f"
# NUMGEN="50"
# OUTPKL="./eval/jjanggu_and_scene17_batch_6/evaluation_1000.pkl"

# python evaluate.py --sample_root ${FOLDER} --target_path ${TARGET_FOLDER} --numgen ${NUMGEN} --outpkl ${OUTPKL}


# echo "----------------------------------------------------------------------"
# echo "-------------------------scene17_1500---------------------------------"
# echo "----------------------------------------------------------------------"
# FOLDER="./logs/jjanggu_and_scene17_batch_6/scene17_1500"
# TARGET_FOLDER="./data/jjanggu2/im_f"
# NUMGEN="50"
# OUTPKL="./eval/jjanggu_and_scene17_batch_6/evaluation_1500.pkl"

# python evaluate.py --sample_root ${FOLDER} --target_path ${TARGET_FOLDER} --numgen ${NUMGEN} --outpkl ${OUTPKL}


# echo "----------------------------------------------------------------------"
# echo "-------------------------scene17_2000---------------------------------"
# echo "----------------------------------------------------------------------"
# FOLDER="./logs/jjanggu_and_scene17_batch_6/scene17_2000"
# TARGET_FOLDER="./data/jjanggu2/im_f"
# NUMGEN="50"
# OUTPKL="./eval/jjanggu_and_scene17_batch_6/evaluation.pkl"

# python evaluate.py --sample_root ${FOLDER} --target_path ${TARGET_FOLDER} --numgen ${NUMGEN} --outpkl ${OUTPKL}

# echo "----------------------------------------------------------------------"
# echo "-------------------------scene17_500----------------------------------"
# echo "----------------------------------------------------------------------"
# FOLDER="./logs/jjanggu_and_scene17_b6_LR5e-6/scene17_500"
# TARGET_FOLDER="./data/jjanggu2/im_f"
# NUMGEN="50"
# OUTPKL="./eval/jjanggu_and_scene17_b6_LR5e-6/evaluation_500.pkl"
# OUTCSV="./eval/jjanggu_and_scene17_b6_LR5e-6/evaluation_500.csv"

# python evaluate.py --sample_root ${FOLDER} --target_path ${TARGET_FOLDER} --numgen ${NUMGEN} --outpkl ${OUTPKL} --outcsv ${OUTCSV}


# echo "----------------------------------------------------------------------"
# echo "-------------------------scene17_1000---------------------------------"
# echo "----------------------------------------------------------------------"
# FOLDER="./logs/jjanggu_and_scene17_b6_LR5e-6/scene17_1000"
# TARGET_FOLDER="./data/jjanggu2/im_f"
# NUMGEN="50"
# OUTPKL="./eval/jjanggu_and_scene17_b6_LR5e-6/evaluation_1000.pkl"
# OUTCSV="./eval/jjanggu_and_scene17_b6_LR5e-6/evaluation_1000.csv"


# python evaluate.py --sample_root ${FOLDER} --target_path ${TARGET_FOLDER} --numgen ${NUMGEN} --outpkl ${OUTPKL} --outcsv ${OUTCSV} 


# echo "----------------------------------------------------------------------"
# echo "-------------------------scene17_1500---------------------------------"
# echo "----------------------------------------------------------------------"
# FOLDER="./logs/jjanggu_and_scene17_b6_LR5e-6/scene17_1500"
# TARGET_FOLDER="./data/jjanggu2/im_f"
# NUMGEN="50"
# OUTPKL="./eval/jjanggu_and_scene17_b6_LR5e-6/evaluation_1500.pkl"
# OUTCSV="./eval/jjanggu_and_scene17_b6_LR5e-6/evaluation_1500.csv"

# python evaluate.py --sample_root ${FOLDER} --target_path ${TARGET_FOLDER} --numgen ${NUMGEN} --outpkl ${OUTPKL} --outcsv ${OUTCSV}


# echo "----------------------------------------------------------------------"
# echo "-------------------------scene17_2000---------------------------------"
# echo "----------------------------------------------------------------------"
# FOLDER="./logs/jjanggu_and_scene17_b6_LR5e-6/scene17_2000"
# TARGET_FOLDER="./data/jjanggu2/im_f"
# NUMGEN="50"
# OUTPKL="./eval/jjanggu_and_scene17_b6_LR5e-6/evaluation.pkl"
# OUTCSV="./eval/jjanggu_and_scene17_b6_LR5e-6/evaluation_2000.csv"

# python evaluate.py --sample_root ${FOLDER} --target_path ${TARGET_FOLDER} --numgen ${NUMGEN} --outpkl ${OUTPKL} --outcsv ${OUTCSV}

# echo "----------------------------------------------------------------------"
# echo "-------------------------scene17_2000---------------------------------"
# echo "----------------------------------------------------------------------"
# FOLDER="./logs/jjanggu_and_scene17_b6_LR5e-6/scene17_2000"
# TARGET_FOLDER="./data/jjanggu2/im_f"
# NUMGEN="45"
# OUTPKL="./eval/jjanggu_and_scene17_b6_LR5e-6_f/evaluation.pkl"
# OUTCSV="./eval/jjanggu_and_scene17_b6_LR5e-6_f/evaluation_2000.csv"

# python evaluate.py --sample_root ${FOLDER} --target_path ${TARGET_FOLDER} --numgen ${NUMGEN} --outpkl ${OUTPKL} --outcsv ${OUTCSV}

echo "----------------------------------------------------------------------"
echo "-------------------------dreambooth_1000------------------------------"
echo "----------------------------------------------------------------------"
FOLDER="./baseline_result/dreambooth_result"
TARGET_FOLDER="./data/jjanggu2/im_f"
NUMGEN="45"
OUTPKL="./eval/dreambooth/evaluation.pkl"
OUTCSV="./eval/dreambooth/evaluation.csv"

python evaluate.py --sample_root ${FOLDER} --target_path ${TARGET_FOLDER} --numgen ${NUMGEN} --outpkl ${OUTPKL} --outcsv ${OUTCSV}

echo "----------------------------------------------------------------------"
echo "---------------------lora_dreambooth_1250-----------------------------"
echo "----------------------------------------------------------------------"
FOLDER="./baseline_result/lora_result"
TARGET_FOLDER="./data/jjanggu2/im_f"
NUMGEN="45"
OUTPKL="./eval/lora/evaluation.pkl"
OUTCSV="./eval/lora/evaluation.csv"

python evaluate.py --sample_root ${FOLDER} --target_path ${TARGET_FOLDER} --numgen ${NUMGEN} --outpkl ${OUTPKL} --outcsv ${OUTCSV}

