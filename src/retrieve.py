# Copyright 2022 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import argparse
import os
import tqdm
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
from clip_retrieval.clip_client import ClipClient

# target_name = cat
# class data dir = real_reg/samples_cat
# num_class_images = 200

def retrieve(target_name, outpath, num_class_images):
    num_images = 2*num_class_images
    # make num Images X2
    client = ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion_400m", num_images=num_images,  aesthetic_weight=0.1)

    if len(target_name.split()):
        target = '_'.join(target_name.split())
    else:
        target = target_name
    os.makedirs(f'{outpath}/{target}', exist_ok=True)

    # GENERATE real_reg/samples_cat/cat


    if len(list(Path(f'{outpath}/{target}').iterdir())) >= num_class_images:
        return
    
    # if reference image is more than num_class_image than return.

    while True:
        results = client.query(text=target_name)
        if len(results) >= num_class_images or num_images > 1e4:
            break
        else:
            num_images = int(1.5*num_images)
            client = ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion_400m", num_images=num_images,  aesthetic_weight=0.1)

    count = 0
    urls = []
    captions = []

    pbar = tqdm.tqdm(desc='downloading real regularization images', total=num_class_images)

    for each in results:
        name = f'{outpath}/{target}/{count}.jpg'
        success = True
        while True:
            try:
                img = requests.get(each['url'])
                success = True
                break
            except:
                success = False
                break
        # status code : 200 means it is a normal website!! fine website not number of image is 200 !! 
        if success and img.status_code == 200:
            try:
                _ = Image.open(BytesIO(img.content))
                with open(name, 'wb') as f:
                    f.write(img.content) # web page information like b'<!DOCTYPE html>\r\n<html>\r\n<body>\r\n\r\n<h1>This is a Test Page</h1>\r\n\r\n</body>\r\n</html>'
                urls.append(each['url'])
                captions.append(each['caption']) # laion DB's caption for images.. 
                count += 1
                pbar.update(1)
            except:
                pass
        if count > num_class_images:
            break

    with open(f'{outpath}/caption.txt', 'w') as f:
        for each in captions:
            f.write(each.strip() + '\n')

            # in real_reg/simple_cat/caption.txt ( each images caption is logged per lines )

    with open(f'{outpath}/urls.txt', 'w') as f:
        for each in urls:
            f.write(each.strip() + '\n')

            # in real_reg/simple_cat/urls.txt (each image's url is logged per lines)

    with open(f'{outpath}/images.txt', 'w') as f:
        for p in range(count):
            f.write(f'{outpath}/{target}/{p}.jpg' + '\n')

            # in real_reg/simple_cat/images.txt ( each image file name is logged per lines  for example "real_reg/simple_cat/1.jpg")


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--target_name', help='target string for query',
                        type=str)
    parser.add_argument('--outpath', help='path to save retrieved images', default='./',
                        type=str)
    parser.add_argument('--num_class_images', help='number of retrieved images', default=200,
                        type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    retrieve(args.target_name, args.outpath, args.num_class_images)
