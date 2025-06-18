"""
Neural Network Name Generator Package

A neural network-based name generator that learns patterns from real names
and generates new, plausible-sounding names using character-level sequence prediction.
"""

__version__ = "1.0.0"
__author__ = "BigEnglishWords Team"

from .name_generator import NameGenerator
from .generate_names import NameGeneratorInference

__all__ = ["NameGenerator", "NameGeneratorInference"] 