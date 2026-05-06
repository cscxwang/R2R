import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--train_mode', type=str, default='pretrain', choices=['pretrain', 'finetune'])
parser.add_argument('--epochs', type=int, default=None, help='maximum number of epochs to train the total model.')
parser.add_argument('--warmup_epochs', type=int, default=None, help='warmup epochs for pretraining.')
parser.add_argument('--batch_size', type=int, default=None, help='Batch size to use per GPU for training.')
parser.add_argument('--lr', type=float, default=None, help='learning rate of encoder.')
parser.add_argument('--de_type', nargs='+', default=['deblur'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--patch_size', type=int, default=None, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=None, help='number of workers.')

parser.add_argument('--data_file_dir', type=str, default='data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='data/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--deblur_dir', type=str, default='data/Train/Deblur/',
                    help='where training images of deblurring saves.')
parser.add_argument('--lowlight_dir', type=str, default='data/Train/Lowlight/',
                    help='where training images of lowlight saves.')
parser.add_argument('--output_path', type=str, default="output/1D/", help='output save path')
parser.add_argument("--ckpt_dir", type=str, default=None, help="Directory where the checkpoint is saved")
parser.add_argument("--prompt_dir", type=str, default=None, help="Directory where the prompt bank is saved")
parser.add_argument("--init_ckpt_dir", type=str, default="train_ckpt_1D", help="Directory of the pretrained checkpoints used to initialize finetuning")
parser.add_argument("--init_prompt_dir", type=str, default="save_prompts_1D", help="Directory of the pretrained prompts used to initialize finetuning")
parser.add_argument("--init_prompt_name", type=str, default="last", help="Checkpoint/prompt name used to initialize finetuning")
parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for training")

options = parser.parse_args()

mode_defaults = {
    "pretrain": {
        "epochs": 240,
        "warmup_epochs": 15,
        "batch_size": 22,
        "lr": 2e-4,
        "patch_size": 256,
        "num_workers": 16,
        "ckpt_dir": "train_ckpt_1D",
        "prompt_dir": "save_prompts_1D",
    },
    "finetune": {
        "epochs": 30,
        "warmup_epochs": 0,
        "batch_size": 32,
        "lr": 1e-6,
        "patch_size": 224,
        "num_workers": 32,
        "ckpt_dir": "train_ckpt_1D_finetune",
    },
}

for key, value in mode_defaults[options.train_mode].items():
    if getattr(options, key) is None:
        setattr(options, key, value)
