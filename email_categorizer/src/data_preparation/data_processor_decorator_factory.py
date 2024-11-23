from src.data_preparation.data_processor import (
    NoiseRemovalDecorator,
    TranslatorDecorator,
    UnicodeConversionDecorator,
    DataProcessorDecorator,
    DataProcessor
)


class DataProcessorDecoratorFactory:
    @staticmethod
    def create_data_preprocessor(processor: DataProcessor, feature: str) -> DataProcessorDecorator:
        """Create and return a decorated DataProcessor based on the specified feature."""
        if feature == "noise_removal":
            return NoiseRemovalDecorator(processor)
        elif feature == "translation":
            return TranslatorDecorator(processor)
        elif feature == "unicode_conversion":
            return UnicodeConversionDecorator(processor)
        else:
            raise ValueError(f"Unknown feature type: {feature}")
