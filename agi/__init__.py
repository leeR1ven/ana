# -*- coding: utf-8 -*-
"""关联神经元架构（ANA）—— v0 空框架。

    from agi import build_spec, build
    brain = build(build_spec("full"))
    brain.engine.set_input(brain.encoder.encode(obs))
    brain.engine.step()
    brain.forces()
"""
from .spec import REGIONS, Geometry, ScalePreset, Spec, build_spec
from .engine import Engine, EngineConfig
from .build import Brain, build
from .encoding import Encoder, Observation
from .loop import Cortex, Tick, Trace
from .plastic import PlasticConfig, PlasticRule

__all__ = [
    "REGIONS", "Spec", "Geometry", "ScalePreset", "build_spec",
    "Engine", "EngineConfig", "build", "Brain",
    "Encoder", "Observation", "Cortex", "Tick", "Trace",
    "PlasticRule", "PlasticConfig",
]
__version__ = "0.1.0"
