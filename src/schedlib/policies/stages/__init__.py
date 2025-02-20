from .build_op import BuildOpSimple
from .build_sched import BuildSched

def get_build_stage(name, kwargs):
    return {
        'build_op': BuildOpSimple,
        'build_sched': BuildSched
    }[name](**kwargs)