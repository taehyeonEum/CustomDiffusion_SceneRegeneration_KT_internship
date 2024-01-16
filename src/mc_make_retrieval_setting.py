import os

target_name = "cat"

class_data_dir = f"real_reg/simple_{target_name}" 

num_class_images = 200

os.makedirs(f"outputs/{class_data_dir}/{target_name}", exist_ok=True)

f1 = open(f"outputs/{class_data_dir}/caption.txt", 'w')
f2 = open(f'outputs/{class_data_dir}/urls.txt', 'w')
f3 = open(f'outputs/{class_data_dir}/iamges.txt' 'w')

f1_con = []
f2_con = []
f3_con = []

for i in range(200):
    f1_con.append(f'a image of {target_name}')
    f2_con.append(f'www...')

