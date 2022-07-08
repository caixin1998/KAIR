# python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 main_train_psnr.py --opt soptions/swinir/ffhq_base.json  --dist True
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 main_train_psnr.py --opt soptions/swinir/ffhq_base_gan.json  --dist True
# python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 main_train_psnr.py --opt soptions/swinir/full_base.json  --dist True
python -m torch.distributed.launch --nproc_per_node=4 --master_port=1234 main_train_psnr.py --opt soptions/swinir/full_base_gan.json  --dist True
