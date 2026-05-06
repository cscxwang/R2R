import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--train_mode', type=str, default='pretrain', choices=['pretrain', 'finetune'])
parser.add_argument('--epochs', type=int, default=None, help='maximum number of epochs to train the total model.')
parser.add_argument('--warmup_epochs', type=int, default=None, help='warmup epochs for pretraining.')
parser.add_argument('--batch_size', type=int, default=None, help='Batch size to use per GPU for training.')
parser.add_argument('--lr', type=float, default=None, help='learning rate of encoder.')
parser.add_argument('--patch_size', type=int, default=None, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=None, help='number of workers.')

parser.add_argument('--trainset', type=str, default='all')
parser.add_argument('--data_file_dir', type=str, default='data/Train/CDD11/')
parser.add_argument("--ckpt_dir", type=str, default=None, help="Directory where the checkpoint is saved")
parser.add_argument("--prompt_dir", type=str, default=None, help="Directory where the prompt bank is saved")
parser.add_argument("--init_ckpt_dir", type=str, default="train_ckpt_compos", help="Directory of the pretrained checkpoints used to initialize finetuning")
parser.add_argument("--init_prompt_dir", type=str, default="save_prompts_compos", help="Directory of the pretrained prompts used to initialize finetuning")
parser.add_argument("--init_prompt_name", type=str, default="last", help="Checkpoint/prompt name used to initialize finetuning")
parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for training")

options = parser.parse_args()

mode_defaults = {
    "pretrain": {
        "epochs": 240,
        "warmup_epochs": 15,
        "batch_size": 96,
        "lr": 2e-4,
        "patch_size": 128,
        "num_workers": 24,
        "ckpt_dir": "train_ckpt_compos",
        "prompt_dir": "save_prompts_compos",
    },
    "finetune": {
        "epochs": 30,
        "warmup_epochs": 0,
        "batch_size": 32,
        "lr": 1e-6,
        "patch_size": 224,
        "num_workers": 32,
        "ckpt_dir": "train_ckpt_compos_finetune",
    },
}

for key, value in mode_defaults[options.train_mode].items():
    if getattr(options, key) is None:
        setattr(options, key, value)
