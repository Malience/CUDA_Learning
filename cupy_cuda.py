from typing import Tuple, Union
import cupy as cp
import torch
import sys

def _parse_tuple(tup: Union[Tuple, int]) -> Tuple:
    if type(tup) == int:
        return (tup, 1, 1)
    
    if type(tup) != tuple:
        raise Exception("Invalid datatype!")
    
    ln = len(tup)

    if ln == 3:
        return tup
    if ln > 3:
        return tup[:3]
    if ln == 2:
        return (tup[0], tup[1], 1)
    if ln == 1:
        return (tup[0], 1, 1)

class CupyCUDA:
    def __init__(self):
        # self.dev = cp.cuda.device

        stream_ptr = torch.cuda.current_stream().cuda_stream
        self.stream = cp.cuda.ExternalStream(stream_ptr)
        self.stream.use()

        self.module = None


    def compile(self, code: str, name_expressions: Tuple) -> None:
        # Oh hey, really wish cupy actually supported compiling with logs
        # You know like how cuda.core does it
        # This can only go horribly wrong
        self.module = cp.RawModule(code=code, 
                           options=('--std=c++17',), #TODO: Support arch for optimization
                           name_expressions=name_expressions)
        self.module.compile(log_stream=sys.stdout)
    

    def compile_file(self, code_file_path: str, name_expressions):
        code = None
        with open(code_file_path, 'r') as f:
            code = f.read()
        
        self.compile(code, name_expressions)


    def launch(self, kernel: str, grid: Union[Tuple, int], block: Union[Tuple, int], args: Tuple, shared_mem: int = 0):
        if not self.module:
            raise Exception("Module not compiled!")
        
        grid = _parse_tuple(grid)
        block = _parse_tuple(block)

        fun = self.module.get_function(kernel)
        fun(grid, block, args, shared_mem=shared_mem)
    

    def get_global(self, name: str):
        return self.module.get_global(name)


    def set_global(self, var: str, value: torch.Tensor):
        ptr = self.module.get_global(var)
        cparr = cp.array(value)
        mem = cp.ndarray(cparr.shape, cparr.dtype, ptr)
        mem[...] = cparr