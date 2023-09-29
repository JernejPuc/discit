"""Acceleration with CUDA graphs"""

from typing import Callable

from torch import cuda, Tensor


def capture_graph(
    fn: Callable,
    input_tensors: 'tuple[Tensor, ...]',
    warmup_tensor_list: 'list[tuple[Tensor, ...]]' = None,
    copy_idcs_in: 'tuple[int, ...]' = None,
    copy_idcs_out: 'tuple[int, ...]' = None,
    single_input: bool = False,
    mem_pool: 'tuple[int, int]' = None,
    device: 'str' = None
) -> 'tuple[Callable, dict[str, tuple[Tensor, ...] | cuda.CUDAGraph | Callable]]':

    # Warmup
    if warmup_tensor_list is None:
        warmup_tensor_list = [input_tensors]*3

    if warmup_tensor_list:
        s = cuda.Stream(device)
        s.wait_stream(cuda.current_stream(device))

        with cuda.stream(s):
            for warmup_tensors in warmup_tensor_list:
                fn(warmup_tensors) if single_input else fn(*warmup_tensors)

        cuda.current_stream(device).wait_stream(s)

    # Capture
    graph = cuda.CUDAGraph()

    if device is None:
        capture_stream = None

    else:
        capture_stream = cuda.Stream(device)

    with cuda.graph(graph, pool=mem_pool, stream=capture_stream):
        output_tensors = fn(input_tensors) if single_input else fn(*input_tensors)

    # Copy all by default, otherwise copy inputs at given indices
    if isinstance(input_tensors, Tensor):
        input_tensors = (input_tensors,)
        single_input = False

    if copy_idcs_in is None:
        copy_idcs_in = range(len(input_tensors))

    copy_idcs_in = sorted(set(copy_idcs_in))
    do_copy_in = bool(copy_idcs_in)

    # Similar for outputs
    if output_tensors is None:
        output_tensors = ()

    single_output = isinstance(output_tensors, Tensor)
    n_out = 1 if single_output else len(output_tensors)

    if copy_idcs_out is None:
        copy_idcs_out = range(n_out)

    copy_idcs_out = sorted(set(copy_idcs_out))
    do_copy_out = bool(copy_idcs_out)

    # Check for nested inputs
    if do_copy_in and not all(isinstance(i, Tensor) for i in input_tensors):
        raise TypeError('Cannot copy nested input of unknown depth. Try flattening.')

    # Construct
    def call(*new_inputs: 'tuple[Tensor, ...] | tuple[tuple[Tensor, ...]]') -> 'tuple[Tensor, ...]':
        if do_copy_in:
            if single_input:
                new_inputs = new_inputs[0]

            for i in copy_idcs_in:
                input_tensors[i].copy_(new_inputs[i])

        graph.replay()

        if do_copy_out:
            if single_output:
                return output_tensors.clone()

            else:
                new_outputs = list(output_tensors)

                for i in copy_idcs_out:
                    new_outputs[i] = new_outputs[i].clone()

                return new_outputs

        return output_tensors

    return call, {'call': call, 'graph': graph, 'in': input_tensors, 'out': output_tensors}
