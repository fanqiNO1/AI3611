# vae_linear
# nohup python main.py --model vae_linear --data_path ~/intel/VAE/data --device 1 --lmbda 1 --z_dim 1 > logs/vae_linear_z1_lambda1.log 2>&1 &
# nohup python main.py --model vae_linear --data_path ~/intel/VAE/data --device 1 --lmbda 1 --z_dim 2 > logs/vae_linear_z2_lambda1.log 2>&1 &
# nohup python main.py --model vae_linear --data_path ~/intel/VAE/data --device 1 --lmbda 1 --z_dim 4 > logs/vae_linear_z4_lambda1.log 2>&1 &
# nohup python main.py --model vae_linear --data_path ~/intel/VAE/data --device 1 --lmbda 5 --z_dim 1 > logs/vae_linear_z1_lambda5.log 2>&1 &
# nohup python main.py --model vae_linear --data_path ~/intel/VAE/data --device 1 --lmbda 5 --z_dim 2 > logs/vae_linear_z2_lambda5.log 2>&1 &
# nohup python main.py --model vae_linear --data_path ~/intel/VAE/data --device 1 --lmbda 5 --z_dim 4 > logs/vae_linear_z4_lambda5.log 2>&1 &
# nohup python main.py --model vae_linear --data_path ~/intel/VAE/data --device 1 --lmbda 0.2 --z_dim 1 > logs/vae_linear_z1_lambda0.2.log 2>&1 &
# nohup python main.py --model vae_linear --data_path ~/intel/VAE/data --device 1 --lmbda 0.2 --z_dim 2 > logs/vae_linear_z2_lambda0.2.log 2>&1 &
# nohup python main.py --model vae_linear --data_path ~/intel/VAE/data --device 1 --lmbda 0.2 --z_dim 4 > logs/vae_linear_z4_lambda0.2.log 2>&1 &
# nohup python main.py --model vae_linear --data_path ~/intel/VAE/data --device 1 --dynamic_lmbda --z_dim 1 > logs/vae_linear_z1_lambda_dynamic.log 2>&1 &
# nohup python main.py --model vae_linear --data_path ~/intel/VAE/data --device 1 --dynamic_lmbda --z_dim 2 > logs/vae_linear_z2_lambda_dynamic.log 2>&1 &
# nohup python main.py --model vae_linear --data_path ~/intel/VAE/data --device 1 --dynamic_lmbda --z_dim 4 > logs/vae_linear_z4_lambda_dynamic.log 2>&1 &
# vae_conv &
# nohup python main.py --model vae_conv --data_path ~/intel/VAE/data --device 1 --lmbda 1 --z_dim 1 > logs/vae_conv_z1_lambda1.log 2>&1 &
# nohup python main.py --model vae_conv --data_path ~/intel/VAE/data --device 1 --lmbda 1 --z_dim 2 > logs/vae_conv_z2_lambda1.log 2>&1 &
# nohup python main.py --model vae_conv --data_path ~/intel/VAE/data --device 1 --lmbda 1 --z_dim 4 > logs/vae_conv_z4_lambda1.log 2>&1 &
# nohup python main.py --model vae_conv --data_path ~/intel/VAE/data --device 1 --lmbda 5 --z_dim 1 > logs/vae_conv_z1_lambda5.log 2>&1 &
# nohup python main.py --model vae_conv --data_path ~/intel/VAE/data --device 2 --lmbda 5 --z_dim 2 > logs/vae_conv_z2_lambda5.log 2>&1 &
# nohup python main.py --model vae_conv --data_path ~/intel/VAE/data --device 2 --lmbda 5 --z_dim 4 > logs/vae_conv_z4_lambda5.log 2>&1 &
# nohup python main.py --model vae_conv --data_path ~/intel/VAE/data --device 2 --lmbda 0.2 --z_dim 1 > logs/vae_conv_z1_lambda0.2.log 2>&1 &
# nohup python main.py --model vae_conv --data_path ~/intel/VAE/data --device 2 --lmbda 0.2 --z_dim 2 > logs/vae_conv_z2_lambda0.2.log 2>&1 &
# nohup python main.py --model vae_conv --data_path ~/intel/VAE/data --device 3 --lmbda 0.2 --z_dim 4 > logs/vae_conv_z4_lambda0.2.log 2>&1 &
# nohup python main.py --model vae_conv --data_path ~/intel/VAE/data --device 3 --dynamic_lmbda --z_dim 1 > logs/vae_conv_z1_lambda_dynamic.log 2>&1 &
# nohup python main.py --model vae_conv --data_path ~/intel/VAE/data --device 3 --dynamic_lmbda --z_dim 2 > logs/vae_conv_z2_lambda_dynamic.log 2>&1 &
# nohup python main.py --model vae_conv --data_path ~/intel/VAE/data --device 3 --dynamic_lmbda --z_dim 4 > logs/vae_conv_z4_lambda_dynamic.log 2>&1 &
# vae_vit &
# nohup python main.py --model vae_vit --data_path ~/intel/VAE/data --device 1 --lmbda 1 --z_dim 1 > logs/vae_vit_z1_lambda1.log 2>&1 &
# nohup python main.py --model vae_vit --data_path ~/intel/VAE/data --device 1 --lmbda 1 --z_dim 2 > logs/vae_vit_z2_lambda1.log 2>&1 &
# nohup python main.py --model vae_vit --data_path ~/intel/VAE/data --device 1 --lmbda 1 --z_dim 4 > logs/vae_vit_z4_lambda1.log 2>&1 &
# nohup python main.py --model vae_vit --data_path ~/intel/VAE/data --device 1 --lmbda 5 --z_dim 1 > logs/vae_vit_z1_lambda5.log 2>&1 &
# nohup python main.py --model vae_vit --data_path ~/intel/VAE/data --device 2 --lmbda 5 --z_dim 2 > logs/vae_vit_z2_lambda5.log 2>&1 &
# nohup python main.py --model vae_vit --data_path ~/intel/VAE/data --device 2 --lmbda 5 --z_dim 4 > logs/vae_vit_z4_lambda5.log 2>&1 &
# nohup python main.py --model vae_vit --data_path ~/intel/VAE/data --device 3 --lmbda 0.2 --z_dim 1 > logs/vae_vit_z1_lambda0.2.log 2>&1 &
# nohup python main.py --model vae_vit --data_path ~/intel/VAE/data --device 2 --lmbda 0.2 --z_dim 2 > logs/vae_vit_z2_lambda0.2.log 2>&1 &
# nohup python main.py --model vae_vit --data_path ~/intel/VAE/data --device 3 --lmbda 0.2 --z_dim 4 > logs/vae_vit_z4_lambda0.2.log 2>&1 &
# nohup python main.py --model vae_vit --data_path ~/intel/VAE/data --device 3 --dynamic_lmbda --z_dim 1 > logs/vae_vit_z1_lambda_dynamic.log 2>&1 &
# nohup python main.py --model vae_vit --data_path ~/intel/VAE/data --device 3 --dynamic_lmbda --z_dim 2 > logs/vae_vit_z2_lambda_dynamic.log 2>&1 &
# nohup python main.py --model vae_vit --data_path ~/intel/VAE/data --device 3 --dynamic_lmbda --z_dim 4 > logs/vae_vit_z4_lambda_dynamic.log 2>&1 &