## sample 
python src/diffusers_sample.py \
    --delta_ckpt logs/chris_pratt_gog_background/delta.bin \
    --ckpt "CompVis/stable-diffusion-v1-4" \
    --from-file "prompts/gog_chris_pratt_background.txt" \
    --keyword "base_setting" \