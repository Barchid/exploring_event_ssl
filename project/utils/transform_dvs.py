from dataclasses import dataclass
from typing import Tuple
import torch
import numpy as np
from tonic.transforms import functional
from torchvision import transforms
import random


@dataclass(frozen=True)
class RandomFlipPolarity:
    """Flips polarity of individual events with p.
    Changes polarities 1 to -1 and polarities [-1, 0] to 1

    Parameters:
        p (float): probability of flipping individual event polarities
    """

    p: float = 0.5

    def __call__(self, events):
        events = events.copy()
        assert "p" in events.dtype.names
        # flips = np.ones(len(events))
        probs = np.random.rand(len(events))
        mask = probs < self.p
        events["p"][mask] = np.logical_not(events["p"][mask])
        return events


def get_sensor_size(events: np.ndarray):
    return events["x"].max() + 1, events["y"].max() + 1, 2  # H,W,2


@dataclass(frozen=True)
class BackgroundActivityNoise:
    severity: int
    sensor_size: Tuple[int, int, int] = None

    def __call__(self, events):
        c = [0.005, 0.01, 0.03, 0.10, 0.2][
            self.severity - 1
        ]  # percentage of events to add in noise
        if self.sensor_size is None:
            sensor_size = get_sensor_size(events)
        else:
            sensor_size = self.sensor_size
        n_noise_events = int(c * len(events))
        noise_events = np.zeros(n_noise_events, dtype=events.dtype)
        for channel in events.dtype.names:
            event_channel = events[channel]
            if channel == "x":
                low, high = 0, sensor_size[0]
            if channel == "y":
                low, high = 0, sensor_size[1]
            if channel == "p":
                low, high = 0, sensor_size[2]
            if channel == "t":
                low, high = events["t"].min(), events["t"].max()

            if channel == "p":
                noise_events[channel] = np.random.choice(
                    [True, False], size=n_noise_events
                )
            else:
                noise_events[channel] = np.random.uniform(
                    low=low, high=high, size=n_noise_events
                )
        events = np.concatenate((events, noise_events))
        new_events = events[np.argsort(events["t"])]
        # new_events['p'] = events['p']

        return new_events


def get_frame_representation(sensor_size, timesteps):
    return transforms.Compose(
        [
            CustomToFrame(
                timesteps=timesteps, sensor_size=sensor_size, event_count=2500
            ),
            BinarizeFrame(),
        ]
    )


@dataclass(frozen=True)
class CustomToFrame:
    timesteps: int
    sensor_size: Tuple[int, int, int] = None
    event_count: int = 2500

    def __call__(self, events):
        if self.sensor_size is None:
            sensor_size = get_sensor_size(events)
        else:
            sensor_size = self.sensor_size
        return functional.to_frame_numpy(
            events=events,
            sensor_size=sensor_size,
            time_window=None,
            event_count=None,
            n_time_bins=self.timesteps,
            n_event_bins=None,
            overlap=0.0,
            include_incomplete=False,
        )


@dataclass(frozen=True)
class BinarizeFrame:
    def __call__(self, x):
        x = (x > 0).astype(np.float32)
        x = torch.from_numpy(x)
        return x


@dataclass(frozen=True)
class EventCopy:
    num_paste: int = 1
    ratio: Tuple[float, float] = (0.2, 0.5)
    sensor_size: Tuple[int, int, int] = None

    def __call__(self, events):
        if self.sensor_size is None:
            sensor_size = get_sensor_size(events)
        else:
            sensor_size = self.sensor_size

        for _ in range(self.num_paste):
            paste = events.copy()

            bbx1, bby1, bbx2, bby2 = self._bbox(sensor_size[1], sensor_size[0])

            dx = random.randint(-bbx1, sensor_size[0] - bbx2 - 1)
            dy = random.randint(-bby1, sensor_size[1] - bby2 - 1)

            # print(dx, dy, bbx1, bby1, bbx2, bby2)
            # print('\ndx = ', dx, 'lx = ', (sensor_size[0] - bbx2))
            # print('\ndy = ', dy, 'ly = ', (sensor_size[1] - bby2))

            # filter image
            mask_events = (
                (events["x"] >= bbx1 + dx)
                & (events["y"] >= bby1 + dy)
                & (events["x"] <= bbx2 + dx)
                & (events["y"] <= bby2 + dy)
            )

            # delete events of bbox
            events = np.delete(events, mask_events)  # remove events
            mask_events = (
                (events["x"] >= bbx1)
                & (events["y"] >= bby1)
                & (events["x"] <= bbx2)
                & (events["y"] <= bby2)
            )
            paste = events[mask_events].copy()
            paste["x"] = paste["x"] + dx
            paste["y"] = paste["y"] + dy

            # add mix events in bbox
            events = np.concatenate((events, paste))
            new_events = events[np.argsort(events["t"])]
            # new_events['p'] = events['p']
            events = new_events

        return events

    def _bbox(self, H, W):
        ratio = np.random.randint(1, 6) / 20.0
        # ratio = random.uniform(self.ratio[0], self.ratio[1])

        cut_w = int(W * ratio)
        cut_h = int(H * ratio)

        # uniform
        bbx1 = random.randint(0, (W - cut_w))
        bby1 = random.randint(0, (H - cut_h))
        bbx2 = bbx1 + cut_w
        bby2 = bby1 + cut_h

        return bbx1, bby1, bbx2, bby2
