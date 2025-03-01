from pathlib import Path
from typing import Optional, Tuple
import shutil

from main import pretrain, Options as TrainOptions
from linear_eval import linear_eval, Options as LinearEvalOptions
from dataparser import dataparser, from_args, Field
import time


@dataparser
class Options:
    "Train & Linear eval"

    data_folder: Path = Path("./data/")
    result_folder: Path = Path("./results/")

    method: str = "stsw" # Options: ssw, sw, hypersphere, simclr, ari_s3w, ri_s3w, s3w

    ntrees: int = 200
    nlines: int = 20
    delta: float = 2.0

    unif_w: float = 20.0
    align_w: float = 1.0
    feat_dim: int = 10
    
    batch_size: int = 512
    identifier: Optional[str] = "_runtime2"
    seed: int = 0
    epochs: int = 200
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 1e-3

    gpu: int = 0


def train_eval(opt: Options):
    identifier = (
        f"method_{opt.method}_epochs_{opt.epochs}_"
        f"feat_dim_{opt.feat_dim}_batch_size_{opt.batch_size}"
        f"_ntrees_{opt.ntrees}_nlines_{opt.nlines}_delta_{opt.delta}_unif_w_{opt.unif_w}_align_w_{opt.align_w}"
        f"_lr_{opt.lr}_momentum_{opt.momentum}_seed_{opt.seed}_weight_decay_{opt.weight_decay}"
        + (opt.identifier if opt.identifier is not None else "")
    )

    checkpoint_folder = opt.result_folder / identifier
    checkpoint_folder.mkdir(exist_ok=True, parents=True)

    # code_folder = checkpoint_folder / "code"
    # code_folder.mkdir(exist_ok=True)
    # for file in ["train_eval.py", "main.py", "linear_eval.py", "encoder.py", "sw_sphere.py"]:
    #     shutil.copy(file, code_folder / file, follow_symlinks=False)

    runtime_file = open(checkpoint_folder / "runtime.txt", "a")

    encoder_checkpoint = checkpoint_folder / "encoder.pth"

    train_opt = TrainOptions(
        data_folder=opt.data_folder,
        method=opt.method,
        feat_dim=opt.feat_dim,
        result_folder=opt.result_folder,
        align_w=opt.align_w,
        unif_w=opt.unif_w,
        batch_size=opt.batch_size,
        cosine_schedule=True,
        identifier=identifier,
        lr=opt.lr,
        momentum=opt.momentum,
        seed=opt.seed,
        epochs=opt.epochs,
        ntrees=opt.ntrees,
        nlines=opt.nlines,
        delta=opt.delta,
        weight_decay=opt.weight_decay,
        gpu=opt.gpu,
    )

    pretrain_start = time.time()
    pretrain(train_opt)
    pretrain_end = time.time()
    print(f"Pretraining time: {pretrain_end - pretrain_start}s", file=runtime_file)
    print(f"Time per epoch: {(pretrain_end - pretrain_start)/opt.epochs}s", file=runtime_file)

    eval_opt = LinearEvalOptions(
        encoder_checkpoint=encoder_checkpoint,
        data_folder=opt.data_folder,
        seed=opt.seed,
        feat_dim=opt.feat_dim,
        gpu = opt.gpu,
    )

    # We test with both layer_index options
    eval_opt.layer_index = -2
    linear_eval_start = time.time()
    linear_eval(eval_opt)
    linear_eval_end = time.time()
    print(f"Linear eval -2 time: {linear_eval_end - linear_eval_start}s", file=runtime_file)

    eval_opt.layer_index = -1
    linear_eval_start = time.time()
    linear_eval(eval_opt)
    linear_eval_end = time.time()
    print(f"Linear eval -1 time: {linear_eval_end - linear_eval_start}s", file=runtime_file)

    runtime_file.close()

def main():
    opt = from_args(Options)
    train_eval(opt)


if __name__ == "__main__":
    main()
