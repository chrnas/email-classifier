# Methods related to data loading and all pre-processing steps will go here
import pandas as pd
from translate import trans_to_en
from Config import Config
import stanza
from stanza.pipeline.core import DownloadMethod
from transformers import pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


def get_input_data():
    df = pd.read_csv("./data/AppGallery.csv")

    # convert the dtype object to unicode string
    df['Interaction content'] = df['Interaction content'].values.astype('U')
    df['Ticket Summary'] = df['Ticket Summary'].values.astype('U')

    #Optional: rename variable names for remebering easily
    df["y1"] = df["Type 1"]
    df["y2"] = df["Type 2"]
    df["y3"] = df["Type 3"]
    df["y4"] = df["Type 4"]
    df["x"] = df['Interaction content']

    df["y"] = df["y2"]
    # use all the types and not just type 2
    # df["y"] = df[["y1", "y2", "y3", "y4"]].fillna('').apply(lambda row: '|'.join(filter(None, row)), axis=1)

    return df


def de_duplication(df):
    df = df.loc[(df["y"] != '') & (~df["y"].isna()),]  # remove empty y values
    return df


def noise_remover(df):
    # Add noise removing code
    temp = df
    noise = "(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(r\s*:)|(re\s*:)|(\[|\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"
    temp["ts"] = temp["Ticket Summary"].str.lower().replace(noise, " ", regex=True).replace(r'\s+', ' ',
                                                                                            regex=True).str.strip()
    temp_debug = temp.loc[:, ["Ticket Summary", "ts", "y"]]

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
        print(noise)
        temp["ic"] = temp["ic"].replace(noise, " ", regex=True)
    temp["ic"] = temp["ic"].replace(r'\s+', ' ', regex=True).str.strip()
    temp_debug = temp.loc[:, ["Interaction content", "ic", "y"]]
    good_y1 = temp.y1.value_counts()[temp.y1.value_counts() > 10].index
    temp = temp.loc[temp.y1.isin(good_y1)]
    return df


def translate_to_en(df):
    df[Config.TICKET_SUMMARY] = trans_to_en(df[Config.TICKET_SUMMARY].to_list())
    return df

def translate_to_english(texts):
    t2t_m = "facebook/m2m100_418M"
    t2t_pipe = pipeline(task='text2text-generation', model=t2t_m)

    model = M2M100ForConditionalGeneration.from_pretrained(t2t_m)
    tokenizer = M2M100Tokenizer.from_pretrained(t2t_m)
    nlp_stanza = stanza.Pipeline(lang="multilingual", processors="langid",
                                download_method=DownloadMethod.REUSE_RESOURCES)

    text_en_l = []
    for text in texts:
        if text == "":
            text_en_l.append(text)
            continue

        doc = nlp_stanza(text)
        print(doc.lang)
        
        if doc.lang == "en":
            text_en_l.append(text)
        else:
            lang = doc.lang

            # Map special cases to a supported language
            lang_mappings = {
                "fro": "fr",  # Old French
                "la": "it",   # Latin
                "nn": "no",   # Norwegian (Nynorsk)
                "kmr": "tr"   # Kurmanji
            }
            lang = lang_mappings.get(lang, lang)  # Map to the supported language or keep original

            # Translate text to English
            tokenizer.src_lang = lang
            encoded_text = tokenizer(text, return_tensors="pt")
            generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id("en"))
            text_en = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            text_en_l.append(text_en)

            print(text)
            print(text_en)
    return text_en_l
