import sys
from typing import Tuple, Union

import torch
from cuda.core.experimental import Device, LaunchConfig, Program, ProgramOptions, launch

# Create a wrapper class that implements __cuda_stream__
class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream

    def __cuda_stream__(self):
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)  # Return format required by CUDA Python

class TorchCUDA:
    def __init__(self):
        self.dev = Device()
        self.dev.set_current()

        pt_stream = torch.cuda.current_stream()
        # Get PyTorch's current stream
        # print(f"PyTorch stream: {pt_stream}")
        self.stream = self.dev.create_stream(PyTorchStreamWrapper(pt_stream))

        self.modules = {}
    

    def compile(self, code: str, name_expressions):
        program_options = ProgramOptions(std="c++17", arch=f"sm_{self.dev.arch}")
        prog = Program(code, code_type="c++", options=program_options)
        mod = prog.compile(
            "cubin",
            logs=sys.stdout,
            name_expressions=name_expressions,
        )

        for exp in name_expressions:
            self.modules[exp] = mod
    

    def compile_file(self, code_file_path: str, name_expressions):
        code = None
        with open(code_file_path, 'r') as f:
            code = f.read()
        
        self.compile(code, name_expressions)

    def launch(self, kernel: str, config: LaunchConfig, *args):
        if kernel not in self.modules:
            raise Exception("Kernel not found!")

        ker = self.modules[kernel].get_kernel(kernel)
        launch(self.stream, config, ker, *args)