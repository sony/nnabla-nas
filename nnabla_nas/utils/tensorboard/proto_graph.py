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
               attributes=None, need_grad=None, info=None):
    """Converts a node to `proto`.

    Args:
        name (str): Name of the node.
        op (str, optional): Name of the operator. Defaults to 'UnSpecified'.
        inputs (list of str, optional): A list of inputs. Defaults to None.
        output_shapes (list, optional): A list of tuple of integers containing the output shapes. Defaults to None.
        attributes (str, optional): A description of attributes. Defaults to None.

    Returns:
        proto: A node with `proto` format.
    """
    inputs = inputs or []
    attr = {}
    if output_shapes:
        attr['_output_shapes'] = AttrValue(
            list=AttrValue.ListValue(
                shape=[tensor_shape_proto(o) for o in output_shapes]
            )
        )

    if attributes:
        attr['attr'] = AttrValue(s=attributes.encode(encoding='utf_8'))

    if need_grad:
        attr['need_grad'] = AttrValue(b=need_grad)

    info = info or {}
    for k, v in info.items():
        if type(v) == str:
            attr[k] = AttrValue(s=v)
        if type(v) == int:
            attr[k] = AttrValue(i=v)
        if type(v) == float:
            attr[k] = AttrValue(f=v)

    proto = NodeDef(
        name=name.encode(encoding='utf_8'),
        op=op,
        input=inputs,
        attr=attr
    )

    return proto
