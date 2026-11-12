# Ported from torch 2.15.0 (BSD-3-Clause)
#
# Custom autograd: subclass `Node` (= torch._tensor._NodeBase[Tensor]), fill
# `self.inputs: List[Tensor]` and override
# `apply(self, grad: Tensor) -> List[Optional[Tensor]]` (one entry per input)
# and `name()`. Then `attach(out, node)` when `needs_grad(inputs)`. Backward
# dispatches `apply` virtually and accumulates into leaf `.grad`.

from torch._tensor import Tensor, Node, no_grad, enable_grad, set_grad_enabled, is_grad_enabled, _needs, _attach, _sum_to, zeros_like

def needs_grad(inputs: List[Tensor]) -> bool:
    return _needs(inputs)

def attach(out: Tensor, node: Node) -> Tensor:
    return _attach(out, node)

def backward(tensors: Tensor, grad_tensors: Optional[Tensor] = None, retain_graph: bool = False, create_graph: bool = False):
    tensors.backward(grad_tensors, retain_graph, create_graph)

def grad(outputs: Tensor, inputs: List[Tensor], grad_outputs: Optional[Tensor] = None, retain_graph: bool = False, create_graph: bool = False, allow_unused: bool = False) -> List[Tensor]:
    saved = [x.grad for x in inputs]
    for x in inputs:
        x.grad = None
    outputs.backward(grad_outputs, retain_graph, create_graph)
    res = List[Tensor]()
    for i in range(len(inputs)):
        g = inputs[i].grad
        if g is None:
            res.append(zeros_like(inputs[i]))
        else:
            res.append(g)
        inputs[i].grad = saved[i]
    return res

def sum_to(g: Tensor, shape: List[int]) -> Tensor:
    return _sum_to(g, shape)
