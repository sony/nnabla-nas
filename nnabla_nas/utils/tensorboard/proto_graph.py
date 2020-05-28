# Copyright (c) 2020 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto


def tensor_shape_proto(shape):
    r"""Creates a shape opbject.

    Args:
        shape (tuple of int): A tuple of integers.

    Returns:
        TensorShapeProto: A Tesorshape.
    """
    return TensorShapeProto(dim=[TensorShapeProto.Dim(size=d) for d in shape])


def node_proto(name, op='UnSpecified', inputs=None, output_shapes=None,
               need_grad=None, info=None):
    """Converts a node to `proto`.

    Args:
        name (str): Name of the node.
        op (str, optional): Name of the operator. Defaults to 'UnSpecified'.
        inputs (list of str, optional): A list of inputs. Defaults to None.
        output_shapes (list, optional): A list of tuple of integers containing the output shapes. Defaults to None.

    Returns:
        proto: A node with `proto` format.
    """
    inputs = inputs or []
    attributes = dict()
    if output_shapes is not None:
        attributes['_output_shapes'] = AttrValue(
            list=AttrValue.ListValue(
                shape=[tensor_shape_proto(o) for o in output_shapes]
            )
        )

    if need_grad is not None:
        attributes['need_grad'] = AttrValue(b=need_grad)

    if info is not None:
        for k, v in info.items():
            if type(v) == bool:
                value = AttrValue(b=v)
            elif type(v) == int:
                value = AttrValue(i=v)
            elif type(v) == float:
                value = AttrValue(f=v)
            elif type(v) == str:
                value = AttrValue(s=v)
            elif type(v) == list:
                if len(v) == 0 or type(v[0]) == int:
                    value = AttrValue(list=AttrValue.ListValue(i=v))
                else:
                    value = AttrValue(list=AttrValue.ListValue(f=v))
            else:
                continue
            attributes[k] = value

    proto = NodeDef(
        name=name.encode(encoding='utf_8'), op=op,
        input=inputs, attr=attributes
    )

    return proto
