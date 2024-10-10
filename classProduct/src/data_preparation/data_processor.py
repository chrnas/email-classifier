# Methods related to data loading and all pre-processing steps will go here
import pandas as pd
from config import Config
import stanza
from stanza.pipeline.core import DownloadMethod
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm

class DataProcessor():

    def __init__(self,
                 df: pd.DataFrame) -> None:
        self.df: pd.DataFrame = df
        self.X: list = None

    def get_df(self):
        return self.df

    def de_duplication(self):
        df = self.df
        # remove empty y values
        df = df.loc[(df["y"] != '') & (~df["y"].isna()),]
        self.df = df

    def noise_remover(self):
        # Add noise removing code
        temp = self.df
        noise = "(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"
        temp["ts"] = temp["Ticket Summary"].str.lower().replace(noise, " ", regex=True).replace(r'\s+', ' ',
                                                                                                regex=True).str.strip()
        # temp_debug = temp.loc[:, ["Ticket Summary", "ts", "y"]]

        temp["ic"] = temp["Interaction content"].str.lower()
        noise_1 = [
            "(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)",
            "(january|february|march|april|may|june|july|august|september|october|november|december)",
            "(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
            "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            "\d{2}(:|.)\d{2}",
            "(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))",
            "dear ((customer)|(user))",
            "dear",
            "(hello)|(hallo)|(hi )|(hi there)",
            "good morning",
            "thank you for your patience ((during (our)? investigation)|(and cooperation))?",
            "thank you for contacting us",
            "thank you for your availability",
            "thank you for providing us this information",
            "thank you for contacting",
            "thank you for reaching us (back)?",
            "thank you for patience",
            "thank you for (your)? reply",
            "thank you for (your)? response",
            "thank you for (your)? cooperation",
            "thank you for providing us with more information",
            "thank you very kindly",
            "thank you( very much)?",
            "i would like to follow up on the case you raised on the date",
            "i will do my very best to assist you"
            "in order to give you the best solution",
            "could you please clarify your request with following information:"
            "in this matter",
            "we hope you(( are)|('re)) doing ((fine)|(well))",
            "i would like to follow up on the case you raised on",
            "we apologize for the inconvenience",
            "sent from my huawei (cell )?phone",
            "original message",
            "customer support team",
            "(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland.",
            "(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
            "canada, australia, new zealand and other countries",
            "\d+",
            "[^0-9a-zA-Z]+",
            "(\s|^).(\s|$)"]
        for noise in noise_1:
            # print(noise)
            temp["ic"] = temp["ic"].replace(noise, " ", regex=True)
        temp["ic"] = temp["ic"].replace(r'\s+', ' ', regex=True).str.strip()
        # temp_debug = temp.loc[:, ["Interaction content", "ic", "y"]]
        good_y1 = temp.y1.value_counts()[temp.y1.value_counts() > 10].index
        temp = temp.loc[temp.y1.isin(good_y1)]
        self.df = temp

    def translate_to_en(self):
        """Translates the ticket summary to English."""
        df = self.df
        texts = df[Config.TICKET_SUMMARY].to_list()

        t2t_m = "facebook/m2m100_418M"

        model = M2M100ForConditionalGeneration.from_pretrained(t2t_m)
        tokenizer = M2M100Tokenizer.from_pretrained(t2t_m)
        nlp_stanza = stanza.Pipeline(lang="multilingual", processors="langid",
                                     download_method=DownloadMethod.REUSE_RESOURCES)

        text_en_l = []
        for text in tqdm(texts, desc="Translating", unit="text"):
            if text == "":
                text_en_l = text_en_l + [text]
                continue

            doc = nlp_stanza(text)
            # print(doc.lang)
            if doc.lang == "en":
                text_en_l = text_en_l + [text]
            else:
                lang = doc.lang
                if lang == "fro":  # fro = Old French
                    lang = "fr"
                elif lang == "la":  # latin
                    lang = "it"
                elif lang == "nn":  # Norwegian (Nynorsk)
                    lang = "no"
                elif lang == "kmr":  # Kurmanji
                    lang = "tr"

                tokenizer.src_lang = lang
                encoded_hi = tokenizer(text, return_tensors="pt")
                generated_tokens = model.generate(
                    **encoded_hi, forced_bos_token_id=tokenizer.get_lang_id("en"))
                text_en = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True)
                text_en = text_en[0]

                text_en_l = text_en_l + [text_en]

        df[Config.TICKET_SUMMARY] = text_en_l
        self.df = df
    
    def convert_to_unicode(self):
        """Converts the interaction content and ticket summary to unicode."""
        df = self.df
        df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
        df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
        self.df = df
