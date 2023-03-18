from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, Subset
import tonic
import os
import copy

from project.datamodules.ncaltech101 import NCALTECH101
from project.datamodules.ncars import NCARS
from project.utils.eda_transforms import EdaDistribution
from project.datamodules.daily_action_dvs import DailyActionDVS


class DVSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        dataset: str,
        timesteps: int = 10,
        data_dir: str = "data/",
        transforms_list=None,
        subset_len=None,
        **kwargs
    ):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.dataset = dataset  # name of the dataset
        self.timesteps = timesteps
        self.subset_len = subset_len

        # create the directory if not exist
        os.makedirs(data_dir, exist_ok=True)

        # transform
        self.sensor_size, self.num_classes = self._get_dataset_info()
        
        self.train_transform = EdaDistribution(
            self.sensor_size,
            timesteps=timesteps,
            transforms_list=transforms_list,
            dataset=dataset,
            data_dir=data_dir,
        )

    def _get_dataset_info(self):
        if self.dataset == "dvsgesture":
            return tonic.datasets.DVSGesture.sensor_size, len(
                tonic.datasets.DVSGesture.classes
            )
        elif self.dataset == "n-caltech101":
            return None, 101  # variable sensor_size for NCaltech
        elif self.dataset == "asl-dvs":
            return tonic.datasets.ASLDVS.sensor_size, len(tonic.datasets.ASLDVS.classes)
        elif self.dataset == "ncars":
            return NCARS.sensor_size, len(NCARS.classes)
        elif self.dataset == "daily_action_dvs":
            return DailyActionDVS.sensor_size, len(DailyActionDVS.classes)

    def prepare_data(self) -> None:
        # downloads the dataset if it does not exist
        # NOTE: since we use the library named "Tonic", all the download process is handled, we just have to make an instanciation
        if self.dataset == "dvsgesture":
            tonic.datasets.DVSGesture(save_to=self.data_dir)
        elif self.dataset == "n-caltech101":
            NCALTECH101(save_to=self.data_dir)
        elif self.dataset == "asl-dvs":
            tonic.datasets.ASLDVS(save_to=self.data_dir)
        elif self.dataset == "ncars":
            NCARS(save_to=self.data_dir, download=True)
        elif self.dataset == "daily_action_dvs":
            DailyActionDVS(save_to=self.data_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset == "dvsgesture":
            self.train_set = tonic.datasets.DVSGesture(
                save_to=self.data_dir,
                transform=self.train_transform,
                target_transform=None,
                train=True,
            )
            self.val_set = tonic.datasets.DVSGesture(
                save_to=self.data_dir,
                transform=self.train_transform,
                target_transform=None,
                train=False,
            )

        elif self.dataset == "n-caltech101":
            dataset = NCALTECH101(save_to=self.data_dir, transform=self.train_transform)
            full_length = len(dataset)
            train_len = int(0.9 * full_length)
            val_len = full_length - train_len
            self.train_set, self.val_set = random_split(dataset, [train_len, val_len])

        elif self.dataset == "asl-dvs":
            dataset = tonic.datasets.ASLDVS(
                save_to=self.data_dir, transform=self.train_transform
            )
            full_length = len(dataset)
            train_len = int(0.8 * full_length)
            val_len = full_length - train_len
            self.train_set, self.val_set = random_split(dataset, [train_len, val_len])
        elif self.dataset == "ncars":
            self.train_set = NCARS(
                self.data_dir, train=True, transform=self.train_transform
            )
            self.val_set = NCARS(
                self.data_dir, train=False, transform=self.train_transform
            )
        elif self.dataset == "daily_action_dvs":
            dataset = DailyActionDVS(
                save_to=self.data_dir, transform=self.train_transform
            )
            full_length = len(dataset)
            train_len = int(0.9 * full_length)
            val_len = full_length - train_len
            self.train_set, self.val_set = random_split(dataset, [train_len, val_len])

        if self.subset_len is not None:
            print("CREATE SUBSET FOR SEMI-SUPERVISED!!!")
            print("looking for classewise len")
            classewise_len = {}
            for (inp, target) in self.train_set:
                target = str(target)
                if target in classewise_len:
                    classewise_len[target] += 1
                else:
                    classewise_len[target] = 1

            if self.subset_len == "10%":
                sublen = 0.1
            elif self.subset_len == "25%":
                sublen = 0.25
            else:
                raise ValueError('subset_len must be either "10%" or "25%"')

            print(classewise_len, len(self.train_set))

            curr_len = copy.deepcopy(classewise_len)
            for key in curr_len:
                curr_len[key] = round(curr_len[key] * sublen)

            indices = []
            print("looking for indices")
            indi = 0
            for inp, target in self.train_set:
                target = str(target)
                if curr_len[target] > 0:
                    indices.append(indi)
                    curr_len[target] -= 1

                indi += 1

            self.train_set = Subset(self.train_set, indices=indices)

            print("DONE")

        print(len(self.train_set), len(self.val_set))

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=9,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=3,
            shuffle=False,
        )  # self.num_workers

    def test_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
