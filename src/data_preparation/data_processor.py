from abc import ABC, abstractmethod
import pandas as pd
import stanza
from tqdm import tqdm
from stanza.pipeline.core import DownloadMethod
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from src.classifier_config_singleton import ClassifierConfigSingleton


class DataProcessorBase(ABC):
    """Base class for Data Processor"""

    @abstractmethod
    def process(self, df: pd.DataFrame):
        """Process the DataFrame and return the processed result."""
        ...

    def __str__(self):
        return "DataProcessor"


class DataProcessor(DataProcessorBase):
    """Concrete Data Processor"""

    def process(self, df: pd.DataFrame):
        """By default remove rows with empty or NaN values in column 'y'."""
        df = df.loc[(df["y"] != '') & (~df["y"].isna()),]
        return df

    def __str__(self):
        return super().__str__() + ": features: base"


class DataProcessorDecorator(DataProcessorBase):
    """Decorator base class"""

    config: ClassifierConfigSingleton

    def __init__(self, processor: DataProcessorBase):
        """Initialize the decorator with a processor instance."""
        self._processor = processor
        self.config = ClassifierConfigSingleton()

    def process(self, df: pd.DataFrame):
        """Delegate the processing to the wrapped processor."""
        return self._processor.process(df)

    def __str__(self):
        return str(self._processor)


class NoiseRemovalDecorator(DataProcessorDecorator):
    """Concrete decorator for noise removal"""

    def __str__(self):
        return f"{self._processor}, noise_removal"

    def process(self, df: pd.DataFrame):
        """Remove noise patterns from 'Ticket Summary' and 'Interaction content'."""

        df = self._processor.process(
            df)  # Get the current DataFrame from the previous processor

        # Define the noise patterns for 'Ticket Summary'
        noise = r"(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"
        df[self.config.ticket_summary] = df[self.config.ticket_summary].str.lower().replace(
            noise, " ", regex=True).replace(r'\s+', ' ', regex=True).str.strip()

        # Define additional noise patterns for 'Interaction content'
        noise_1 = [
            r"(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)", r"(january|february|march|april|may|june|july|august|september|october|november|december)",
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", r"\d{2}(:|.)\d{2}",
            r"(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))", r"dear ((customer)|(user))", "dear", r"(hello)|(hallo)|(hi )|(hi there)",
            r"good morning", r"thank you for your patience ((during (our)? investigation)|(and cooperation))?",
            r"thank you for contacting us", r"thank you for your availability", r"thank you for providing us this information",
            r"thank you for contacting", r"thank you for reaching us (back)?", r"thank you for patience", r"thank you for (your)? reply",
            r"thank you for (your)? response", r"thank you for (your)? cooperation", r"thank you for providing us with more information",
            r"thank you very kindly", r"thank you( very much)?", r"i would like to follow up on the case you raised on the date",
            r"i will do my very best to assist you", r"in order to give you the best solution", r"could you please clarify your request with following information:",
            r"in this matter", r"we hope you(( are)|('re)) doing ((fine)|(well))", r"i would like to follow up on the case you raised on",
            r"we apologize for the inconvenience", r"sent from my huawei (cell )?phone", r"original message", "customer support team",
            r"(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland.",
            r"(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
            r"canada, australia, new zealand and other countries", r"\d+", "[^0-9a-zA-Z]+", r"(\s|^).(\s|$)"
        ]

        # Apply noise removal to 'Interaction content' column
        for pattern in noise_1:
            df[self.config.interaction_content] = df[self.config.interaction_content].replace(
                pattern, " ", regex=True)

        # Clean extra spaces and strip
        df[self.config.interaction_content] = df[self.config.interaction_content].replace(
            r'\s+', ' ', regex=True).str.strip()

        # Filter out low-frequency 'y' values
        good_y1 = df.y.value_counts()[df.y.value_counts() > 10].index

        # Check if there are any values to filter
        if not good_y1.empty:
            df = df.loc[df.y.isin(good_y1)]
        else:
            # Optionally log or handle the case where no values meet the condition
            print(
                "Warning: No values in column 'y' have more than 10 occurrences. Skipping filtering.")

        return df


class TranslatorDecorator(DataProcessorDecorator):
    """Concrete decorator for translation"""

    def __str__(self):
        return f"{self._processor}, translation"

    def process(self, df: pd.DataFrame):
        """Translate 'Ticket Summary' from its original language to English."""

        df = self._processor.process(df)
        texts = df[self.config.ticket_summary].to_list()

        model = M2M100ForConditionalGeneration.from_pretrained(
            "facebook/m2m100_418M")
        tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
        nlp_stanza = stanza.Pipeline(
            lang="multilingual", processors="langid", download_method=DownloadMethod.REUSE_RESOURCES)

        text_en_l = []
        for text in tqdm(texts, desc="Translating", unit="text"):
            if text == "":
                text_en_l.append(text)
                continue
            doc = nlp_stanza(text)
            if doc.lang == "en":
                text_en_l.append(text)
            else:
                lang = doc.lang
                if lang == "fro":
                    lang = "fr"
                elif lang == "la":
                    lang = "it"
                elif lang == "nn":
                    lang = "no"
                elif lang == "kmr":
                    lang = "tr"
                tokenizer.src_lang = lang
                encoded_text = tokenizer(text, return_tensors="pt")
                generated_tokens = model.generate(
                    **encoded_text, forced_bos_token_id=tokenizer.get_lang_id("en"))
                text_en = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True)[0]
                text_en_l.append(text_en)

        df[self.config.ticket_summary] = text_en_l
        return df


class UnicodeConversionDecorator(DataProcessorDecorator):
    """Concrete decorator for Unicode conversion"""

    def __str__(self):
        return f"{self._processor}, unicode_conversion"

    def process(self, df: pd.DataFrame):
        """Convert 'Interaction Content' and 'Ticket Summary' to Unicode."""

        df = self._processor.process(df)
        df[self.config.interaction_content] = df[self.config.interaction_content].values.astype(
            'U')
        df[self.config.ticket_summary] = df[self.config.ticket_summary].values.astype(
            'U')
        return df
