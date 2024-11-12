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
        data: 'Tensor | np.ndarray',
        batch_size: int,
        device: str = 'cuda',
        shuffle: bool = True,
        shuffle_on_cpu: bool = False
    ):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        self.data = data.to(device)

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

        self.iter_ptr = 0

        return self

    def __next__(self) -> Tensor:
        curr_iter_ptr = self.iter_ptr
        self.iter_ptr += self.batch_size

        if self.iter_ptr > self.len:
            raise StopIteration

        return self.data[curr_iter_ptr:self.iter_ptr]


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
        data: 'Tensor | np.ndarray',
        batch_size: int,
        device: str = 'cuda',
        shuffle: bool = True,
        pin_memory: bool = True
    ):
        assert device.startswith('cuda'), f'Dataset expected cuda, got {device}.'

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        self.data = data

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

        self.iter_ptr = 0

        # Load on side stream
        with cuda.stream(self.stream):
            self.next_batch = self.data[-self.batch_size:].to(self.device, non_blocking=True)

        return self

    def __next__(self) -> Tensor:
        curr_iter_ptr = self.iter_ptr
        self.iter_ptr += self.batch_size

        if self.iter_ptr > self.len:
            self.next_batch = None

            raise StopIteration

        # Wait until loading in side stream completes
        cuda.current_stream().wait_stream(self.stream)

        batch = self.next_batch

        # Signal side stream not to reuse this memory while the main stream can work on it
        batch.record_stream(cuda.current_stream())

        # Load on side stream
        with cuda.stream(self.stream):
            self.next_batch = self.data[curr_iter_ptr:self.iter_ptr].to(self.device, non_blocking=True)

        return batch


class TensorDict(dict):
    """Wrapper around a dict with assumptions for key and value content."""

    VAL_TYPES = {
        'act': 'tuple[Tensor, ...]',
        'args': 'tuple[Tensor, ...]',
        'val': 'Tensor | tuple[Tensor, ...]',
        'obs': 'tuple[Tensor, ...]',
        'mem': 'tuple[Tensor, ...]',
        'rwd': 'Tensor | tuple[Tensor, ...]',
        'ret': 'Tensor | tuple[Tensor, ...]',
        'adv': 'Tensor',
        'nrst': 'Tensor'}

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

    def clone(self) -> 'TensorDict':
        return TensorDict({
            k: (
                tuple([t.clone() for t in v])
                if isinstance(v, tuple)
                else v.clone())
            for k, v in self.items()})

    @staticmethod
    def _chunk(t: Tensor, i: int, n: int) -> Tensor:
        chunk_size = len(t) // n
        return t[chunk_size*i:chunk_size*(i+1)]

    def chunk(self, i: int, n: int) -> 'TensorDict':
        if n == 1:
            return self

        return TensorDict({
            k: (
                tuple([self._chunk(t, i, n) for t in v])
                if isinstance(v, tuple)
                else self._chunk(v, i, n))
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
    Wrapper around a list with capped length, storing state transitions
    (obs., act., rwd., etc.) to form trajectories for eventual optimisation.

    NOTE: The implementation lacks checks to handle e.g. tensor content
    of different sizes or changes to the buffer mid-iteration.
    """

    def __init__(self, buffer_len: int, data: 'tuple[list[TensorDict | None], int]' = None):
        if buffer_len < 1:
            raise ValueError(f'Invalid buffer length: {buffer_len}')

        self.buffer_len = buffer_len
        self.iter_step = 1
        self.iter_ptr = buffer_len
        self.item_iter = False

        if data is None:
            self.batches: 'list[TensorDict]' = []
            self.bind_ptr = 0

        else:
            self.batches, self.bind_ptr = data

    def clear(self, n_clear: int = None):
        if n_clear is None or n_clear >= self.buffer_len:
            self.batches.clear()

        elif n_clear == 0:
            return

        else:
            self.batches = self.batches[-self.buffer_len + n_clear:]

        self.bind_ptr = len(self.batches)

    def is_full(self) -> bool:
        return self.bind_ptr == self.buffer_len

    def load_ratio(self) -> float:
        return self.bind_ptr / self.buffer_len

    def size(self) -> int:
        return self.buffer_len

    def batch_size(self) -> int:
        try:
            return self.batches[0].size()

        except IndexError as err:
            raise RuntimeError('Tried to infer batch size from empty buffer.') from err

    def device(self) -> torch.device:
        try:
            return self.batches[0].device()

        except IndexError as err:
            raise RuntimeError('Tried to infer device from empty buffer.') from err

    def extend(self, buffer: 'ExperienceBuffer'):
        if self.buffer_len < (self.bind_ptr + len(buffer)):
            raise IndexError('Extending overfull buffer.')

        self.batches.extend(buffer.batches)
        self.bind_ptr += len(buffer)

    def append(self, val: TensorDict):
        if self.bind_ptr == self.buffer_len:
            raise IndexError('Appended to full buffer.')

        self.batches.append(val)
        self.bind_ptr += 1

    def pop(self) -> TensorDict:
        if self.bind_ptr == 0:
            raise IndexError('Popped from empty buffer.')

        val = self.batches.pop()
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

            return ExperienceBuffer(self.buffer_len, data=(batches, bind_ptr))

        if isinstance(arg, tuple):
            buffer_arg, batch_arg = arg

            if isinstance(buffer_arg, int):
                batch = self[buffer_arg]

                return batch.slice(batch_arg)

            buffer = self[buffer_arg]
            buffer.batches = [b.slice(batch_arg) for b in buffer.batches]

            return buffer

        else:
            raise TypeError(f'Invalid indexing argument: {arg}')

    def __len__(self) -> int:
        return self.bind_ptr

    def __iter__(self) -> 'ExperienceBuffer':
        self.iter_ptr = -self.iter_step

        return self

    def __next__(self) -> 'list[TensorDict] | TensorDict':
        self.iter_ptr += self.iter_step
        slice_end_ptr = self.iter_ptr + self.iter_step

        if slice_end_ptr > self.bind_ptr:
            self.iter_step = 1
            self.item_iter = False

            raise StopIteration

        # Return a single batch
        if self.item_iter:
            return self.batches[self.iter_ptr]

        # Return a sublist of batches
        return self.batches[self.iter_ptr:slice_end_ptr]

    def iter_slices(self, iter_step: int):
        if iter_step < 1:
            raise ValueError(f'Invalid iter. step: {iter_step}')

        if self.bind_ptr % iter_step:
            print(f'Warning: Iter. step ({iter_step}) inconsistent with buffer length ({self.bind_ptr}).')

        self.iter_step = iter_step

        return iter(self)

    def iter_items(self):
        self.item_iter = True

        return iter(self)

    def shuffle(
        self,
        rng: np.random.Generator,
        seq_length: int = 1,
        n_in_chunks: int = 1,
        n_out_chunks: int = 1
    ) -> 'ExperienceBuffer':

        if self.bind_ptr < seq_length:
            raise RuntimeError(f'Seq. length {seq_length} incompatible with {self.bind_ptr} batches in buffer.')

        # List of sequences of sliced batches
        seq_list = [
            [b.chunk(j, n_in_chunks) for b in self.batches[i:i+seq_length]]
            for i in range(0, self.bind_ptr - self.bind_ptr % seq_length, seq_length)
            for j in range(n_in_chunks)]

        # Reorder sequences uniformly
        rng.shuffle(seq_list)

        # Flatten list into buffer
        batches = [b for batches in seq_list for b in batches]

        buffer = ExperienceBuffer(len(batches), data=(batches, len(batches)))

        # Stack if necessary
        if n_out_chunks > 1:
            buffer = buffer.stack(n_out_chunks)

        return buffer

    def sample(
        self,
        rng: np.random.Generator,
        seq_length: int = 1,
        n_in_chunks: int = 1,
        n_out_chunks: int = 1,
        keep_structure: bool = False
    ) -> 'list[TensorDict]':

        if seq_length > self.bind_ptr:
            raise RuntimeError(f'Cannot sample seq. of length {seq_length} from {self.bind_ptr} batches in buffer.')

        if (seq_length * n_out_chunks) > (self.bind_ptr * n_in_chunks):
            raise RuntimeError(
                f'Cannot sample seq. of length {seq_length} and {n_out_chunks} chunks '
                f'from {self.bind_ptr} batches of {n_in_chunks} chunks.')

        # Choose chunks and starting batches uniformly
        batch_idcs = rng.choice(self.bind_ptr - seq_length + 1, n_out_chunks)

        if keep_structure:
            batch_idcs -= batch_idcs % seq_length

        # Flattened list of sequences of sliced batches
        if n_in_chunks == 1:
            batches = [
                b
                for i in batch_idcs
                for b in self.batches[i:i+seq_length]]

        else:
            chunk_idcs = rng.choice(n_in_chunks, n_out_chunks)

            batches = [
                b.chunk(j, n_in_chunks)
                for i, j in zip(batch_idcs, chunk_idcs)
                for b in self.batches[i:i+seq_length]]

        # Stack into a single sequence of batches
        if n_out_chunks == 1:
            return batches

        batch_ref = batches[0]

        return [batch_ref.cat_alike(batches[i::seq_length]) for i in range(seq_length)]

    def restack(self, n_to_unstack: int, rng: np.random.Generator = None) -> 'ExperienceBuffer':
        """Change batch size based on the argument's sign."""

        return self.unstack(n_to_unstack, rng) if n_to_unstack > 0 else self.stack(-n_to_unstack, rng)

    def stack(self, n_slices: int, rng: np.random.Generator = None) -> 'ExperienceBuffer':
        """Increase batch size by stacking multiple buffer slices."""

        if n_slices < 2:
            raise ValueError(f'Invalid slice num.: {n_slices}')

        if self.bind_ptr % n_slices or n_slices > self.bind_ptr:
            raise ValueError(f'Slice num. ({n_slices}) inconsistent with buffer length ({self.bind_ptr}).')

        # Optional shuffle
        if rng is not None:
            batches = [self.batches[i] for i in rng.permutation(self.buffer_len)]

        else:
            batches = self.batches

        slice_len = self.buffer_len // n_slices
        batch_ref = batches[0]

        batches = [batch_ref.cat_alike(batches[i::slice_len]) for i in range(slice_len)]

        return ExperienceBuffer(slice_len, data=(batches, slice_len))

    def unstack(self, n_slices: int, rng: np.random.Generator = None) -> 'ExperienceBuffer':
        """Reduce batch size by splitting batches into multiple buffer slices."""

        if n_slices < 2:
            raise ValueError(f'Invalid slice num.: {n_slices}')

        batch_size = self.batch_size()

        if batch_size % n_slices or n_slices > batch_size:
            raise ValueError(f'Slice num. ({n_slices}) inconsistent with batch size ({batch_size}).')

        # Optional batch-wise shuffle
        if rng is not None:
            idcs = rng.permutation(batch_size)
            batches = [b.slice(idcs) for b in self.batches]

        else:
            batches = self.batches

        buffer_len = self.buffer_len * n_slices
        new_size = batch_size // n_slices
        slices = [slice(i, i+new_size) for i in range(0, batch_size, new_size)]

        batches = [b.slice(s) for s in slices for b in batches]

        return ExperienceBuffer(buffer_len, data=(batches, buffer_len))

    def label(
        self,
        values: Tensor,
        gammas: 'float | Tensor',
        lambda_: float,
        n_actors_per_env: int = 1,
        bias_returns: bool = False,
        skip_std: bool = False
    ) -> 'tuple[Tensor, Tensor] | None':
        """
        Compute advantages via generalised advantage estimation (GAE),
        external advantage estimates (ACPPO), and bootstrapped returns
        in a traceable way and add them to existing batches.
        """

        multi_agent = n_actors_per_env != 1

        # Bootstrap returns
        # NOTE: Final values (future returns) in an episode are assumed to be zeros
        # Earlier returns are based (bootstrapped) on the model's estimates
        returns = values
        advantages = torch.zeros_like(values)

        for batch in reversed(self.batches):
            nrst_gammas = batch['nrst'] * gammas

            # GAE
            deltas = batch['rwd'] + nrst_gammas * values - batch['val']
            advantages = deltas + nrst_gammas * lambda_ * advantages

            values = batch['val']

            # Biased or discounted sum
            if bias_returns:
                returns = advantages + values

            else:
                returns = batch['rwd'] + nrst_gammas * returns

            # Value targets and advantages
            if multi_agent:
                batch['retj'] = returns[::n_actors_per_env, :1]
                batch['reti'] = returns[:, 1:]

                # Joint advantage targets
                batch['advt'] = advantages[::n_actors_per_env, :1]

                # Replace joint policy advantages with external advantages
                adv_ext = batch['nrst'] * batch['advx']

                batch['advp'] = adv_ext.sum(-1) + advantages[:, 1:].sum(-1)

            else:
                batch['ret'] = returns
                batch['advp'] = advantages.sum(-1)

        if skip_std:
            return

        # Only force expectation of zero mean for adv. target
        if multi_agent:
            adv_tar = torch.stack([batch['advt'] for batch in self.batches])
            adv_tar = adv_tar - adv_tar.mean()

            for i, batch in enumerate(self.batches):
                batch['advt'] = adv_tar[i]

        # Standardise variance of adv. as policy loss factor
        # NOTE: Standardisation works best over the whole rollout (more samples, fewer gaps)
        adv_pi = torch.stack([batch['advp'] for batch in self.batches])

        adv_mean = adv_pi.mean()
        adv_std = adv_pi.std()

        # NOTE: Div. by scale is clipped to limit the magnitude of sparse rewards (max. 100x larger)
        adv_pi = (adv_pi - adv_mean) / adv_std.clip(0.01)

        for i, batch in enumerate(self.batches):
            batch['advp'] = adv_pi[i]

        return adv_mean, adv_std
