import pytorch_lightning as pl
from project.datamodules.dvs_datamodule import DVSDataModule
from project.ssl_module import SSLModule
import torch
from itertools import chain, combinations
import os
from matplotlib import pyplot as plt

from project.utils.eval_callback import OnlineFineTuner
import traceback
from datetime import datetime
from argparse import ArgumentParser


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 1000
learning_rate = 1e-2
timesteps = 12
batch_size = 128


def main(args):
    dataset = args.dataset
    edas = args.edas
    encoder1 = args.encoder1
    encoder2 = args.encoder2

    name = f"{dataset}_{encoder1}_{encoder2}"
    for eda in edas:
        name += f"_{eda}"

    if encoder1 == encoder2:
        checkpoint_callbacks = [
            pl.callbacks.ModelCheckpoint(
                monitor="online_val_acc",
                filename=name + "-{epoch:03d}-{online_val_acc:.4f}",
                save_top_k=1,
                mode="max",
            )
        ]
    else:
        checkpoint_callbacks = [
            pl.callbacks.ModelCheckpoint(
                monitor="online_val_acc_enc1",
                filename=name + "ENC1-{epoch:03d}-{online_val_acc_enc1:.4f}",
                save_top_k=1,
                mode="max",
            ),
            pl.callbacks.ModelCheckpoint(
                monitor="online_val_acc_enc2",
                filename=name + "ENC2-{epoch:03d}-{online_val_acc_enc2:.4f}",
                save_top_k=1,
                mode="max",
            ),
        ]

    datamodule = DVSDataModule(
        batch_size=batch_size,
        dataset=dataset,
        timesteps=timesteps,
        data_dir="data/",
        transforms_list=edas,
    )

    module = SSLModule(
        learning_rate=learning_rate,
        epochs=epochs,
        timesteps=timesteps,
        enc1=encoder1,
        enc2=encoder2,
    )

    if encoder1 == encoder2:
        online_finetuners = [
            OnlineFineTuner(
                encoder_output_dim=512,
                num_classes=datamodule.num_classes,
                enc=None,
            )
        ]
    else:
        online_finetuners = [
            OnlineFineTuner(
                encoder_output_dim=512, num_classes=datamodule.num_classes, enc="enc1"
            ),
            OnlineFineTuner(
                encoder_output_dim=512, num_classes=datamodule.num_classes, enc="enc2"
            ),
        ]
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=torch.cuda.device_count(),
        callbacks=[*online_finetuners, *checkpoint_callbacks],
        logger=pl.loggers.TensorBoardLogger("experiments", name=name),
        default_root_dir=f"experiments/{name}",
        precision=16,
    )

    try:
        trainer.fit(module, datamodule=datamodule)
    except:
        mess = traceback.format_exc()
        report = open("errors.txt", "a")
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        report.write(f"{dt_string} ===> {mess}\n=========\n\n")
        report.flush()
        report.close()
        return -1

    # write in score
    report = open(f"report.txt", "a")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    report.write(
        f"{dt_string} {dataset} {checkpoint_callbacks[0].best_model_score} {encoder1} {encoder2} {trans}\n"
    )
    report.flush()
    report.close()
    return checkpoint_callbacks[0].best_model_score


def get_args():
    parser = ArgumentParser(
        "Exploring Joint Embedding Architectures and Data Augmentations for Self-Supervised Representation Learning in Event-Based Vision"
    )
    parser.add_argument(
        "--encoder1", type=str, choices=["snn", "3dcnn", "cnn"], default="cnn"
    )
    parser.add_argument(
        "--encoder2", type=str, choices=["snn", "3dcnn", "cnn"], default="cnn"
    )
    parser.add_argument(
        "--edas",
        type=str,
        default="background_activity,flip_polarity,crop,event_copy_drop,geostatdyn",
        help="List of employed event data augmentations. They must be separated by commas. Example: 'transform1,transform2,...,transformN'.",
    )
    parser.add_argument(
        "--dataset",
        choices=["asl-dvs", "dvsgesture", "daily_action_dvs", "n-caltech101", "ncars"],
        default="dvsgesture",
    )
    args = parser.parse_args()

    allowed_transforms = (
        "background_activity",
        "flip_polarity",
        "crop",
        "event_drop",
        "cutout",
        "event_copy",
        "event_copy_drop",
        "geostatdyn",
        "static_translation",
        "static_rotation",
        "dynamic_translation",
        "dynamic_rotation",
    )

    edas = args.edas.split(",")
    for eda in edas:
        if eda not in allowed_transforms:
            raise ValueError(
                f"edas arguments must contain only transforms in the following list: {allowed_transforms}. Got: {eda}."
            )

    args.edas = edas

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
