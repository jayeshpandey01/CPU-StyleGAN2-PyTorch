try:
    from .cpu_ops import CPULeakyReLU as FusedLeakyReLU
    from .cpu_ops import cpu_leaky_relu as fused_leaky_relu
    from .cpu_ops import cpu_upfirdn2d as upfirdn2d
    print("Using CPU operations")
except ImportError:
    from .fused_act import FusedLeakyReLU, fused_leaky_relu
    from .upfirdn2d import upfirdn2d
    print("Using CUDA operations")
