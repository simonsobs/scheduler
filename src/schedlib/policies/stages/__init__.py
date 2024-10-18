from .build_op import BuildOp
from .build_sched import BuildSched

def get_build_stage(name, **kwargs):
    return {
        'build_op': BuildOp,
        'build_sched': BuildSched
    }[name](**kwargs)