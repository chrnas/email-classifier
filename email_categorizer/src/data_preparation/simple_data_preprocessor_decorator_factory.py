import pandas as pd
from .data_processor import (
    NoiseRemovalDecorator,
    TranslatorDecorator,
    DeDuplicationDecorator,
    UnicodeConversionDecorator,
    DataProcessorDecorator,
    DataProcessor
)


class SimpleDataPreProcessorDecoratorFactory:
    @staticmethod
    def create_data_preprocessor(processor: DataProcessor, feature: str) -> DataProcessorDecorator:

        if feature == "noise_removal":
            return NoiseRemovalDecorator(processor)
        elif feature == "translation":
            return TranslatorDecorator(processor)
        elif feature == "deduplication":
            return DeDuplicationDecorator(processor)
        elif feature == "unicode_conversion":
            return UnicodeConversionDecorator(processor)
        else:
            raise ValueError(f"Unknown feature type: {feature}")
