# example.
# python evaluate.py --sample_root {folder} --target_path {target-folder} --numgen {numgen}

echo "----------------------------------------------------------------------"
echo "-------------------------scene17_500----------------------------------"
echo "----------------------------------------------------------------------"
FOLDER="./logs/jjanggu_and_scene17_batch_6/scene17_500"
TARGET_FOLDER="./data/jjanggu2/im_f"
NUMGEN="50"
OUTPKL="./eval/jjanggu_and_scene17_batch_6/evaluation_500.pkl"

python evaluate.py --sample_root ${FOLDER} --target_path ${TARGET_FOLDER} --numgen ${NUMGEN} --outpkl ${OUTPKL}


echo "----------------------------------------------------------------------"
echo "-------------------------scene17_1000---------------------------------"
echo "----------------------------------------------------------------------"
FOLDER="./logs/jjanggu_and_scene17_batch_6/scene17_1000"
TARGET_FOLDER="./data/jjanggu2/im_f"
NUMGEN="50"
OUTPKL="./eval/jjanggu_and_scene17_batch_6/evaluation_1000.pkl"

python evaluate.py --sample_root ${FOLDER} --target_path ${TARGET_FOLDER} --numgen ${NUMGEN} --outpkl ${OUTPKL}


echo "----------------------------------------------------------------------"
echo "-------------------------scene17_1500---------------------------------"
echo "----------------------------------------------------------------------"
FOLDER="./logs/jjanggu_and_scene17_batch_6/scene17_1500"
TARGET_FOLDER="./data/jjanggu2/im_f"
NUMGEN="50"
OUTPKL="./eval/jjanggu_and_scene17_batch_6/evaluation_1500.pkl"

python evaluate.py --sample_root ${FOLDER} --target_path ${TARGET_FOLDER} --numgen ${NUMGEN} --outpkl ${OUTPKL}


echo "----------------------------------------------------------------------"
echo "-------------------------scene17_2000---------------------------------"
echo "----------------------------------------------------------------------"
FOLDER="./logs/jjanggu_and_scene17_batch_6/scene17_2000"
TARGET_FOLDER="./data/jjanggu2/im_f"
NUMGEN="50"
OUTPKL="./eval/jjanggu_and_scene17_batch_6/evaluation.pkl"

python evaluate.py --sample_root ${FOLDER} --target_path ${TARGET_FOLDER} --numgen ${NUMGEN} --outpkl ${OUTPKL}