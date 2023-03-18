import os
import numpy as np
from tonic.dataset import Dataset
from tonic.download_utils import download_and_extract_archive
import struct


class DailyActionDVS(Dataset):
    """DailyActionDVS action recognition dataset <https://github.com/CrystalMiaoshu/PAFBenchmark>. Events have (xypt) ordering.
    ::
        @inproceedings{tan2022multi,
            title={Multi-Grained Spatio-Temporal Features Perceived Network for Event-Based Lip-Reading},
            author={Tan, Ganchao and Wang, Yang and Han, Han and Cao, Yang and Wu, Feng and Zha, Zheng-Jun},
            booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
            pages={20094--20103},
            year={2022}
        }
        Implementation inspired from original script: https://github.com/tgc1997/event-based-lip-reading/blob/main/utils/dataset.py
    Parameters:
        save_to (string): Location to save files to on disk.
        train (bool): If True, uses training subset, otherwise testing subset.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    base_url = "https://nextcloud.univ-lille.fr/index.php/s/DyAXqJyqeRL4kyD/download"
    filename = "DailyAction-DVS.zip"
    base_folder = "DailyAction-DVS"
    file_md5 = "99adac8babb1f613a1e1c66232519348"

    sensor_size = (128, 128, 2)  # DVS 128
    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("p", bool), ("t", np.int64)])
    ordering = dtype.names

    classes = [
        "bend",
        "climb",
        "fall",
        "get up",
        "jump",
        "lie",
        "lift",
        "pick",
        "run",
        "sit",
        "stand",
        "walk",
    ]

    def __init__(self, save_to, transform=None, target_transform=None):
        super(DailyActionDVS, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )
        self.url = self.base_url
        # self.folder_name = os.path.join(
        #     self.base_folder, "train" if self.train else "test"
        # )

        if not self._check_exists():
            print(
                f"""
                WARNING: this dataset is available from Google Drive and must be downloaded manually.
                Please download and extract the zip file ( {self.url} ) and place it in {self.location_on_system}."""
            )
            download_and_extract_archive(self.base_url, self.location_on_system, filename=self.filename, md5=self.file_md5)
            exit()

        file_path = os.path.join(self.location_on_system, self.base_folder)

        for act_dir in os.listdir(file_path):
            label = self.classes.index(act_dir)

            for file in os.listdir(os.path.join(file_path, act_dir)):
                if file.endswith("npy"):
                    self.targets.append(label)
                    self.data.append(os.path.join(file_path, act_dir, file))

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        input_path = self.data[index]
        events = np.load(input_path)

        target = self.targets[index]
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.isdir(
            os.path.join(
                self.location_on_system, self.base_folder
            )  # check if directory exists
        ) and self._folder_contains_at_least_n_files_of_type(100, ".npy")


def getDVSeventsDavis(file, numEvents=1e10, startTime=0):
    """DESCRIPTION: This function reads a given aedat file and converts it into four lists indicating
                     timestamps, x-coordinates, y-coordinates and polarities of the event stream.

    Args:
        file: the path of the file to be read, including extension (str).
        numEvents: the maximum number of events allowed to be read (int, default value=1e10).
        startTime: the start event timestamp (in microseconds) where the conversion process begins (int, default value=0).
    Return:
        ts: list of timestamps in microseconds.
        x: list of x-coordinates in pixels.
        y: list of y-coordinates in pixels.`
        pol: list of polarities (0: on -> off, 1: off -> on).
    """
    # print("\ngetDVSeventsDavis function called \n")
    sizeX = 346
    sizeY = 260
    x0 = 0
    y0 = 0
    x1 = sizeX
    y1 = sizeY

    # print("Reading in at most", str(numEvents))

    triggerevent = int("400", 16)
    polmask = int("800", 16)
    xmask = int("003FF000", 16)
    ymask = int("7FC00000", 16)
    typemask = int("80000000", 16)
    typedvs = int("00", 16)
    xshift = 12
    yshift = 22
    polshift = 11
    x = []
    y = []
    ts = []
    pol = []
    numeventsread = 0

    length = 0
    aerdatafh = open(file, "rb")
    k = 0
    p = 0
    statinfo = os.stat(file)
    if length == 0:
        length = statinfo.st_size
    # print("file size", length)

    lt = aerdatafh.readline()
    while lt and str(lt)[2] == "#":
        p += len(lt)
        k += 1
        lt = aerdatafh.readline()
        continue

    aerdatafh.seek(p)
    tmp = aerdatafh.read(8)
    p += 8
    while p < length:
        ad, tm = struct.unpack_from(">II", tmp)
        ad = abs(ad)
        if tm >= startTime:
            if (ad & typemask) == typedvs:
                xo = sizeX - 1 - float((ad & xmask) >> xshift)
                yo = float((ad & ymask) >> yshift)
                polo = 1 - float((ad & polmask) >> polshift)
                if xo >= x0 and xo < x1 and yo >= y0 and yo < y1:
                    x.append(xo)
                    y.append(yo)
                    pol.append(polo)
                    ts.append(tm)
        aerdatafh.seek(p)
        tmp = aerdatafh.read(8)
        p += 8
        numeventsread += 1

    # print("Total number of events read =", numeventsread)
    # print("Total number of DVS events returned =", len(ts))

    return ts, x, y, pol
