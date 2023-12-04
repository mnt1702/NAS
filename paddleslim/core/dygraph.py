import paddle
import collections
import logging
import numpy as np
from paddle.fluid.framework import _dygraph_tracer, dygraph_only, _dygraph_guard
from paddle.fluid.dygraph.base import program_desc_tracing_guard
from paddle.nn.layer import Layer
from paddle.fluid.framework import Block, EagerParamBase, Program, Variable
from ..common import get_logger

__all__ = ["dygraph2program"]

_logger = get_logger(__name__, level=logging.INFO)


def _is_shape(values):
    if not isinstance(values, (list, tuple)):
        return False
    for v in values:
        if not isinstance(v, int):
            return False
    return True


def _is_shapes(values):
    if not isinstance(values, (list, tuple)):
        return False
    for v in values:
        if not _is_shape(v):
            return False
    return True


def _create_tensors(shapes, dtypes=None):
    if dtypes is not None:
        assert len(shapes) == len(
            dtypes
        ), "Length of shapes and dtypes must be same. But get len(shapes): {}; len(dtypes): {}; shapes: {}; dtypes: {}".format(
            len(shapes), len(dtypes), shapes, dtypes)
    else:
        dtypes = len(shapes) * ['float32']
    tensors = []
    for shape, dtype in zip(shapes, dtypes):
        data = np.ones(tuple(shape)).astype(dtype)
        tensors.append(paddle.to_tensor(data))
    return tensors


def extract_vars(inputs):
    """
    Extract a list of variables from inputs.
    Args:
        inputs(Variable | list<Object> | dict): 
    """
    vars = []
    if isinstance(inputs, Variable):
        vars = [inputs]
    elif isinstance(inputs, dict):
        for _key, _value in inputs.items():
            if isinstance(_value, Variable):
                vars.append(_value)
            else:
                _logger.warn(
                    f"Variable is excepted, but get an element with type({type(_value)}) from inputs whose type is dict. And the key of element is {_key}."
                )
    elif isinstance(inputs, (tuple, list)):
        for _value in inputs:
            vars.extend(extract_vars(_value))
    if len(vars) == 0:
        _logger.warn(f"Extract none variables from inputs.")
    return vars


def to_variables(inputs):
    """
    Find and rename variables. Find np.ndarray and convert it to variable.
    """
    if isinstance(inputs, Variable) or isinstance(inputs, np.ndarray):
        return paddle.fluid.dygraph.to_variable(inputs)
    elif isinstance(inputs, dict):
        ret = {}
        for _key in inputs:
            ret[_key] = to_variables(inputs[_key])
        return inputs
    elif isinstance(inputs, list):
        ret = []
        for _value in inputs:
            ret.append(to_variables(_value))
        return ret


@dygraph_only
def dygraph2program(layer,
                    inputs,
                    feed_prefix='feed_',
                    fetch_prefix='fetch_',
                    tmp_prefix='t_',
                    extract_inputs_fn=None,
                    extract_outputs_fn=None,
                    dtypes=None):
    assert isinstance(layer, Layer)

    extract_inputs_fn = extract_inputs_fn if extract_inputs_fn is not None else extract_vars
    extract_outputs_fn = extract_outputs_fn if extract_outputs_fn is not None else extract_vars
    tracer = _dygraph_tracer()._get_program_desc_tracer()

    with program_desc_tracing_guard(True):

        if _is_shape(inputs):
            shapes = [inputs]
            inputs = _create_tensors(shapes, dtypes=dtypes)
            input_var_list = inputs
        elif _is_shapes(inputs):
            inputs = _create_tensors(inputs, dtypes=dtypes)
            input_var_list = inputs
        else:
            inputs = to_variables(inputs)
            input_var_list = extract_inputs_fn(inputs)
        original_outputs = layer(*inputs)
        # 'original_outputs' may be dict, so we should convert it to list of varibles.
        # And should not create new varibles in 'extract_vars'.
        out_var_list = extract_outputs_fn(original_outputs)
        program_desc, feed_names, fetch_names, parameters = tracer.create_program_desc(
            input_var_list, out_var_list, fetch_prefix, tmp_prefix)
        tracer.reset()

    with _dygraph_guard(None):
        program = Program()
        program.desc = program_desc
        program.blocks = [Block(program, 0)]
        program._sync_with_cpp()
    return program






# import os
# import paddle
# import collections
# import logging
# import numpy as np
# from ..common import get_logger

# __all__ = ["dygraph2program"]

# _logger = get_logger(__name__, level=logging.INFO)


# class NameGenerator:
#     def __init__(self):
#         self.ids = collections.defaultdict(int)

#     def name(self, prefix):
#         assert isinstance(prefix, str)

#         name = "{}_{}".format(prefix, self.ids[prefix])
#         self.ids[prefix] += 1

#         return name


# NG = NameGenerator()


# def _is_shape(values):
#     if not isinstance(values, (list, tuple)):
#         return False
#     for v in values:
#         if not isinstance(v, int):
#             return False
#     return True


# def _is_shapes(values):
#     if not isinstance(values, (list, tuple)):
#         return False
#     for v in values:
#         if not _is_shape(v):
#             return False
#     return True


# def _create_tensors(shapes, dtypes=None, is_static=False):
#     if dtypes is not None:
#         assert len(shapes) == len(
#             dtypes
#         ), "Length of shapes and dtypes must be same. But get len(shapes): {}; len(dtypes): {}; shapes: {}; dtypes: {}".format(
#             len(shapes), len(dtypes), shapes, dtypes)
#     else:
#         dtypes = len(shapes) * ['float32']
#     tensors = []
#     for shape, dtype in zip(shapes, dtypes):
#         if is_static:
#             tensors.append(
#                 paddle.static.data(
#                     shape=shape, dtype=dtype, name=NG.name("feed")))
#         else:
#             data = np.ones(tuple(shape)).astype(dtype)
#             tensors.append(paddle.to_tensor(data))
#     return tensors


# def _to_var(x):
#     """
#     Convert Variable or np.array into Placeholder.
#     """
#     shape = x.shape
#     dtype = x.dtype
#     name = getattr(x, "name", None) or NG.name("feed")
#     return paddle.static.data(shape=shape, dtype=dtype, name=name)


# def to_variables(inputs):
#     """
#     Find and rename variables. Find np.ndarray and convert it to variable.
#     """
#     if isinstance(inputs,
#                   (paddle.static.Variable, paddle.Tensor)) or isinstance(
#                       inputs, np.ndarray):
#         return _to_var(inputs)
#     elif isinstance(inputs, dict):
#         ret = {}
#         for _key in inputs:
#             ret[_key] = to_variables(inputs[_key])
#         return ret
#     elif isinstance(inputs, list):
#         ret = []
#         for _value in inputs:
#             ret.append(to_variables(_value))
#         return ret


# def dygraph2program(layer, inputs, dtypes=None):
#     assert isinstance(layer, paddle.nn.Layer)
#     return _dy2prog(layer, inputs, dtypes)


# def _dy2prog(layer, inputs, dtypes=None):
#     """
#     Tracing program in Eager Mode.
#     """
#     paddle.enable_static()
#     program = paddle.static.Program()
#     # convert ParamBase into Parameter automatically by _switch_declarative_mode_guard_
#     with paddle.static.program_guard(
#             program), paddle.base.dygraph.base._to_static_mode_guard_(
#                 is_to_static=True
#             ), paddle.base.framework._stride_in_no_check_dy2st_diff():
#         if _is_shape(inputs):
#             shapes = [inputs]
#             inputs = _create_tensors(shapes, dtypes=dtypes)
#         elif _is_shapes(inputs):
#             inputs = _create_tensors(inputs, dtypes=dtypes)
#         else:
#             inputs = to_variables(inputs)
#         if isinstance(inputs, list):
#             outputs = layer(*inputs)
#         else:
#             outputs = layer(inputs)

#     paddle.disable_static()

#     return program
