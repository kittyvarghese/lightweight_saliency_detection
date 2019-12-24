python image_test.py --model_dir /media/vplab/Kitty/models/mse_1520ep/1320epo_72600step.ckpt --dataset /media/vplab/Kitty/TestImages/img --cuda cuda --save_dir /media/vplab/Kitty/Results/mse;

python image_test.py --model_dir /media/vplab/Kitty/models/bce_2280ep/1880epo_103400step.ckpt --dataset /media/vplab/Kitty/TestImages/img --cuda cuda --save_dir /media/vplab/Kitty/Results/bce;

python image_test.py --model_dir /media/vplab/Kitty/models/huber_3112ep/3090epo_170000step.ckpt --dataset /media/vplab/Kitty/TestImages/img --cuda cuda --save_dir /media/vplab/Kitty/Results/huber;

python image_test.py --model_dir /media/vplab/Kitty/models/mse_bce_1770ep/1734epo_95400step.ckpt --dataset /media/vplab/Kitty/TestImages/img --cuda cuda --save_dir /media/vplab/Kitty/Results/bce-mse;

python image_test.py --model_dir /media/vplab/Kitty/models/huber_bce_1800ep/1600epo_88000step.ckpt --dataset /media/vplab/Kitty/TestImages/img --cuda cuda --save_dir /media/vplab/Kitty/Results/huber_bce;

python image_test.py --model_dir /media/vplab/Kitty/models/triplet_2130ep/2130epo_117200step.ckpt --dataset /media/vplab/Kitty/TestImages/img --cuda cuda --save_dir /media/vplab/Kitty/Results/triplet;

python image_test.py --model_dir /media/vplab/Kitty/models/huber_mse_2760ep/2680epo_147400step.ckpt --dataset /media/vplab/Kitty/TestImages/img --cuda cuda --save_dir /media/vplab/Kitty/Results/huber_mse;


