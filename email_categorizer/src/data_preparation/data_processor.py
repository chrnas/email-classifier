from abc import ABC, abstractmethod
import pandas as pd
from config import Config
import stanza
from stanza.pipeline.core import DownloadMethod
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm

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

    def __init__(self, processor: DataProcessorBase):
        """Initialize the decorator with a processor instance."""
        self._processor = processor

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
        noise = "(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"
        df["Ticket Summary"] = df["Ticket Summary"].str.lower().replace(
            noise, " ", regex=True).replace(r'\s+', ' ', regex=True).str.strip()

        # Define additional noise patterns for 'Interaction content'
        noise_1 = [
            "(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)", "(january|february|march|april|may|june|july|august|september|october|november|december)",
            "(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", "\d{2}(:|.)\d{2}",
            "(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))", "dear ((customer)|(user))", "dear", "(hello)|(hallo)|(hi )|(hi there)",
            "good morning", "thank you for your patience ((during (our)? investigation)|(and cooperation))?",
            "thank you for contacting us", "thank you for your availability", "thank you for providing us this information",
            "thank you for contacting", "thank you for reaching us (back)?", "thank you for patience", "thank you for (your)? reply",
            "thank you for (your)? response", "thank you for (your)? cooperation", "thank you for providing us with more information",
            "thank you very kindly", "thank you( very much)?", "i would like to follow up on the case you raised on the date",
            "i will do my very best to assist you", "in order to give you the best solution", "could you please clarify your request with following information:",
            "in this matter", "we hope you(( are)|('re)) doing ((fine)|(well))", "i would like to follow up on the case you raised on",
            "we apologize for the inconvenience", "sent from my huawei (cell )?phone", "original message", "customer support team",
            "(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland.",
            "(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
            "canada, australia, new zealand and other countries", "\d+", "[^0-9a-zA-Z]+", "(\s|^).(\s|$)"
        ]

        # Apply noise removal to 'Interaction content' column
        for pattern in noise_1:
            df["Interaction content"] = df["Interaction content"].replace(
                pattern, " ", regex=True)

        # Clean extra spaces and strip
        df["Interaction content"] = df["Interaction content"].replace(
            r'\s+', ' ', regex=True).str.strip()

        # Filter out low-frequency 'y1' values
        good_y1 = df.y1.value_counts()[df.y1.value_counts() > 10].index
        df = df.loc[df.y1.isin(good_y1)]

        return df


class TranslatorDecorator(DataProcessorDecorator):
    """Concrete decorator for translation"""

    def __str__(self):
        return f"{self._processor}, translation"

    def process(self, df: pd.DataFrame):
        """Translate 'Ticket Summary' from its original language to English."""
        df = self._processor.process(df)
        texts = df[Config.TICKET_SUMMARY].to_list()

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

        df[Config.TICKET_SUMMARY] = text_en_l
        return df


class UnicodeConversionDecorator(DataProcessorDecorator):
    """Concrete decorator for Unicode conversion"""

    def __str__(self):
        return f"{self._processor}, unicode_conversion"

    def process(self, df: pd.DataFrame):
        """Convert 'Interaction Content' and 'Ticket Summary' to Unicode."""
        df = self._processor.process(df)
        df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype(
            'U')
        df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype(
            'U')
        return df
