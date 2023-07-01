from ctypes import ArgumentError


class Pipeline:
    """
    Basic "monad" like class for pipelining the node functions.
    """

    def __init__(self, nodes: list = []):
        self.nodes = nodes

    def link(self, other):
        if type(other) == Pipeline:
            return Pipeline(self.nodes + other.nodes)
        else:
            raise Exception("Cannot link a pipeline with a non-pipeline.")

    def bind(self, node_function):
        self.nodes.append(node_function)

    def eval(self, *inputs):
        input_proc = inputs

        # Evaluate each node in pipeline order
        for i, node_function in enumerate(self.nodes):
            try:
                input_proc = node_function(*input_proc)
            except (ArgumentError, TypeError) as err:
                raise Exception(f"Pipeline error at index {i-1}->{i}.\n{err}")

        return input_proc
