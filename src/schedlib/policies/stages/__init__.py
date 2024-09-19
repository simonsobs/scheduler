from .build_op import BuildOp, BuildOpSimple
from .build_sched import BuildSched

def get_build_stage(name, **kwargs):
    return {
        'build_op': BuildOp,
        'build_op_simple': BuildOpSimple,
        'build_sched': BuildSched
    }[name](**kwargs)