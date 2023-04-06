"""Data stores and loaders"""

import gc

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import cuda, Tensor


class LoadedDataset:
    """Iterator slicing through data that is fully loaded on the target device."""

    def __init__(
        self,
        file_path: str,
        batch_size: int,
        device: str = 'cuda',
        shuffle: bool = True,
        shuffle_on_cpu: bool = False
    ):
        self.data = torch.load(file_path).to(device)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.shuffle = shuffle
        self.shuffle_on_cpu = shuffle_on_cpu
        self.iter_ptr = self.len = len(self.data)

    def __len__(self) -> int:
        return self.len

    def __iter__(self) -> 'LoadedDataset':
        if self.shuffle:
            if self.shuffle_on_cpu:
                self.data = self.data.cpu()
                gc.collect()

                self.data = self.data[torch.randperm(self.len)].to(self.device)

            else:
                self.data = self.data[torch.randperm(self.len, device=self.device)]

        self.iter_ptr = -self.batch_size

        return self

    def __next__(self) -> Tensor:
        self.iter_ptr += self.batch_size

        if self.iter_ptr == self.len:
            raise StopIteration

        return self.data[self.iter_ptr:self.iter_ptr+self.batch_size]


class SideLoadingDataset:
    """
    Iterator sideloading batches of data from regular RAM to target CUDA device.

    NOTE: Data is not streamed from storage, so RAM itself must still be able
    to hold at least the equivalent of two datasets, as shuffling and pinning
    memory both make a copy and memory might also not be released instantly.

    Reference:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py#L265
    """

    def __init__(
        self,
        file_path: str,
        batch_size: int,
        device: str = 'cuda',
        shuffle: bool = True,
        pin_memory: bool = True
    ):
        assert device.startswith('cuda'), f'Dataset expected cuda, got {device}.'

        self.data = torch.load(file_path)
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.iter_ptr = self.len = len(self.data)

        if pin_memory:
            self.data = self.data.pin_memory()

        # Side stream
        self.stream = cuda.Stream()
        self.next_batch = None

    def __len__(self) -> int:
        return self.len

    def __iter__(self) -> 'SideLoadingDataset':
        if self.shuffle:
            self.data = self.data[torch.randperm(self.len)]

            if self.pin_memory:
                gc.collect()

                self.data = self.data.pin_memory()

        self.iter_ptr = -self.batch_size

        # Load on side stream
        with cuda.stream(self.stream):
            self.next_batch = self.data[-self.batch_size:].to(self.device, non_blocking=True)

        return self

    def __next__(self) -> Tensor:
        self.iter_ptr += self.batch_size

        if self.iter_ptr == self.len:
            self.next_batch = None

            raise StopIteration

        # Wait until loading in side stream completes
        cuda.current_stream().wait_stream(self.stream)

        batch = self.next_batch

        # Signal side stream not to reuse this memory while the main stream can work on it
        batch.record_stream(cuda.current_stream())

        # Load on side stream
        with cuda.stream(self.stream):
            self.next_batch = self.data[self.iter_ptr:self.iter_ptr+self.batch_size].to(self.device, non_blocking=True)

        return batch


class TensorDict(dict):
    """Wrapper around a dict with assumptions for key and value content."""

    VAL_TYPES = {
        'act': 'tuple[Tensor, ...]',
        'val': Tensor,
        'obs': 'tuple[Tensor, ...]',
        'mem': 'tuple[Tensor, ...]',
        'rew': Tensor,
        'ret': Tensor,
        'adv': Tensor,
        'rst': Tensor,
        'nrst': Tensor}

    def size(self) -> int:
        try:
            size = next(len(v[0] if isinstance(v, tuple) else v) for v in self.values())

        except StopIteration:
            raise RuntimeError('Tried to infer batch size from empty batch.')

        return size

    def device(self) -> torch.device:
        try:
            device = next((v[0] if isinstance(v, tuple) else v).device for v in self.values())

        except StopIteration:
            raise RuntimeError('Tried to infer device from empty batch.')

        return device

    def to_list(self) -> 'list[Tensor]':
        return [
            t
            for v in self.values()
            for t in (v if isinstance(v, tuple) else (v,))]

    def from_list(self, lst: 'list[Tensor]') -> 'TensorDict':
        lst = iter(lst)

        return TensorDict({
            k: (
                tuple([next(lst) for _ in range(len(v))])
                if isinstance(v, tuple)
                else next(lst))
            for k, v in self.items()})

    def slice(self, arg: 'int | slice | Ellipsis | ArrayLike') -> 'TensorDict':
        return TensorDict({
            k: (
                tuple([t[arg] for t in v])
                if isinstance(v, tuple)
                else v[arg])
            for k, v in self.items()})

    def cat_alike(self, batches: 'list[TensorDict]') -> 'TensorDict':
        return TensorDict({
            k: (
                tuple([torch.cat([b[k][i] for b in batches]) for i in range(len(v))])
                if isinstance(v, tuple)
                else torch.cat([b[k] for b in batches]))
            for k, v in self.items()})


class ExperienceBuffer:
    """
    Wrapper around a list with fixed length, storing state transitions
    (obs., act., rew., etc.) to form trajectories for eventual optimisation.

    NOTE: The implementation lacks checks to handle e.g. tensor content
    of different sizes or changes to the buffer mid-iteration.
    """

    batches: 'list[TensorDict | None]'
    bind_ptr: int

    def __init__(self, buffer_len: int, data: 'tuple[list[TensorDict | None], int]' = None):
        if buffer_len < 1:
            raise ValueError(f'Invalid buffer length: {buffer_len}')

        self.buffer_len = buffer_len
        self.iter_step = 1
        self.iter_ptr = buffer_len
        self.item_iter = False

        if data is None:
            self.clear()

        else:
            self.batches, self.bind_ptr = data

    def clear(self):
        self.batches = [None] * self.buffer_len
        self.bind_ptr = 0

    def is_full(self) -> bool:
        return self.bind_ptr == self.buffer_len

    def size(self) -> int:
        return self.buffer_len

    def batch_size(self) -> int:
        try:
            return self.batches[0].size()

        except AttributeError as err:
            raise RuntimeError('Tried to infer batch size from empty buffer.') from err

    def device(self) -> torch.device:
        try:
            return self.batches[0].device()

        except AttributeError as err:
            raise RuntimeError('Tried to infer device from empty buffer.') from err

    def append(self, val: TensorDict):
        if self.bind_ptr == self.buffer_len:
            raise IndexError('Appended to full buffer.')

        self.batches[self.bind_ptr] = val
        self.bind_ptr += 1

    def pop(self) -> TensorDict:
        if self.bind_ptr == 0:
            raise IndexError('Popped from empty buffer.')

        val = self.batches[self.bind_ptr]

        self.batches[self.bind_ptr] = None
        self.bind_ptr -= 1

        return val

    def __setitem__(self, *args):
        raise RuntimeError('Arbitrary assignment is prohibited. Use append & pop instead.')

    def __getitem__(self, arg: 'int | slice | tuple[int | slice, int | slice]') -> 'TensorDict | ExperienceBuffer':
        if isinstance(arg, int):
            return self.batches[arg]

        if isinstance(arg, slice):
            if arg.start is arg.stop is arg.step is None:
                return self

            batches = self.batches[arg]
            bind_ptr = min(self.bind_ptr, len(batches))

            return ExperienceBuffer(len(batches), data=(batches, bind_ptr))

        if isinstance(arg, tuple):
            buffer_arg, batch_arg = arg

            if isinstance(buffer_arg, int):
                batch = self[buffer_arg]

                return None if batch is None else batch.slice(batch_arg)

            buffer = self[buffer_arg]
            buffer.batches = [None if b is None else b.slice(batch_arg) for b in buffer.batches]

            return buffer

        else:
            raise TypeError(f'Invalid indexing argument: {arg}')

    def __len__(self) -> int:
        return self.bind_ptr

    def __iter__(self) -> 'ExperienceBuffer':
        if self.bind_ptr < self.buffer_len:
            raise RuntimeError('Need full buffer to iterate.')

        self.iter_ptr = -self.iter_step

        return self

    def __next__(self) -> 'list[TensorDict] | TensorDict':
        self.iter_ptr += self.iter_step

        if self.iter_ptr == self.buffer_len:
            self.iter_step = 1
            self.item_iter = False

            raise StopIteration

        # Return a single batch
        if self.item_iter:
            return self.batches[self.iter_ptr]

        # Return a sublist of batches
        return self.batches[self.iter_ptr:self.iter_ptr+self.iter_step]

    def iter_slices(self, iter_step: int):
        if iter_step < 1:
            raise ValueError(f'Invalid iter. step: {iter_step}')

        if self.buffer_len % iter_step:
            raise ValueError(f'Iter. step ({iter_step}) inconsistent with buffer length ({self.buffer_len}).')

        self.iter_step = iter_step

        return iter(self)

    def iter_items(self):
        self.item_iter = True

        return iter(self)

    def shuffle(
        self,
        rng: np.random.Generator,
        key: str = None,
        batch_wise: bool = False,
        alpha: float = 0.7,
        beta: float = 0.5,
        eps: float = 0.
    ) -> 'None | Tensor | list[Tensor]':

        if self.bind_ptr < self.buffer_len:
            raise RuntimeError('Need full buffer to shuffle.')

        if batch_wise:
            batch_size = self.batch_size()

            # Reorder samples per batch wrt. permutation (uniform probs.)
            if key is None:
                probs = None
                idcs = [rng.permutation(batch_size) for _ in self.batches]

            # Probs. wrt. key, e.g. advantages
            # NOTE: Using abs, assuming advantages
            else:
                try:
                    with torch.no_grad():
                        probs = [b[key].abs().cpu().numpy() ** alpha + eps for b in self.batches]
                        probs = [prob / prob.sum() for prob in probs]

                except KeyError as err:
                    raise RuntimeError('Tried to prio. shuffle with unlabelled samples.') from err

                # Sample index sequences
                idcs = [rng.choice(batch_size, batch_size, replace=False, p=p) for p in probs]

            # Reorder samples per batch
            self.batches = [b.slice(i) for b, i in zip(self.batches, idcs)]

            if probs is None:
                return

            # Weights to scale loss contributions
            weights = [(p[i] * batch_size) ** -beta for p, i in zip(probs, idcs)]
            weights = [
                torch.from_numpy(w / w.max()).to(self.batches[0][key].device, dtype=torch.float32)
                for w in weights]

            return weights

        # Reorder batches wrt. permutation (uniform probs.)
        if key is None:
            self.batches = [self.batches[i] for i in rng.permutation(self.buffer_len)]

            return

        # Probs. wrt. key, e.g. advantages
        # NOTE: Using abs, assuming advantages
        try:
            with torch.no_grad():
                probs = np.array([b[key].abs().mean().item() for b in self.batches]) ** alpha + eps
                probs /= probs.sum()

        except KeyError as err:
            raise RuntimeError('Tried to prio. shuffle with unlabelled samples.') from err

        # Sample index sequence
        idcs = rng.choice(self.buffer_len, self.buffer_len, replace=False, p=probs)

        # Reorder batches
        self.batches = [self.batches[i] for i in idcs]

        # Weights to scale loss contributions
        weights = (probs[idcs] * self.buffer_len) ** -beta
        weights = torch.from_numpy(weights / weights.max()).to(self.batches[0][key].device, dtype=torch.float32)

        return weights

    def stack(self, n_slices: int) -> 'ExperienceBuffer':
        """Increase batch size by stacking multiple buffer slices."""

        if n_slices < 2:
            raise ValueError(f'Invalid slice num.: {n_slices}')

        if self.buffer_len % n_slices or n_slices > self.buffer_len:
            raise ValueError(f'Slice num. ({n_slices}) inconsistent with buffer length ({self.buffer_len}).')

        if self.bind_ptr < self.buffer_len:
            raise RuntimeError('Need full buffer to stack.')

        slice_len = self.buffer_len // n_slices
        batch_ref = self.batches[0]

        batches = [batch_ref.cat_alike(self.batches[i::slice_len]) for i in range(slice_len)]

        return ExperienceBuffer(slice_len, data=(batches, slice_len))

    def unstack(self, n_slices: int) -> 'ExperienceBuffer':
        """Reduce batch size by splitting batches into multiple buffer slices."""

        if n_slices < 2:
            raise ValueError(f'Invalid slice num.: {n_slices}')

        if self.bind_ptr < self.buffer_len:
            raise RuntimeError('Need full buffer to unstack.')

        batch_size = self.batch_size()

        if batch_size % n_slices or n_slices > batch_size:
            raise ValueError(f'Slice num. ({n_slices}) inconsistent with batch size ({batch_size}).')

        buffer_len = self.buffer_len * n_slices
        new_size = batch_size // n_slices
        slices = [slice(i, i+new_size) for i in range(0, batch_size, new_size)]

        batches = [b.slice(s) for s in slices for b in self.batches]

        return ExperienceBuffer(buffer_len, data=(batches, buffer_len))

    def label(self, values: Tensor, gamma: float, lam: float) -> 'tuple[float, float]':
        """
        Compute advantages and returns via generalised advantage estimation (GAE)
        in a traceable way and add them to existing batches.
        """

        if not all(self.batches):
            raise RuntimeError('Need full buffer to label.')

        advantages = torch.zeros_like(self.batches[-1]['rew'])

        # NOTE: Final values (future returns) in an episode are assumed to be zeros
        # Earlier returns are based (bootstrapped) on the model's estimates
        for batch in reversed(self.batches):
            deltas = batch['rew'] + batch['nrst'] * gamma * values - batch['val']
            advantages = deltas + batch['nrst'] * gamma * lam * advantages
            values = batch['val']

            batch['adv'] = advantages
            batch['ret'] = advantages + values

        advantages = torch.stack([batch['adv'] for batch in self.batches], dim=1)

        # NOTE: Standardisation works best over the whole rollout (more samples, fewer outliers)
        # NOTE: Div. by scale is clipped to limit the noise of sparse rewards (max. 10x larger)
        adv_mean = advantages.mean()
        adv_std = advantages.std()

        advantages = (advantages - adv_mean) / torch.clip(adv_std, 0.1)

        for i, batch in enumerate(self.batches):
            batch['adv'] = advantages[:, i]

        return adv_mean.item(), adv_std.item()
