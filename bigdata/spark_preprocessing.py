"""
Distributed PySpark Preprocessing Pipeline for Multilingual Code-Mixed Social Media Text.
Standardizes heterogeneous datasets into a Unified Schema, cleans text, and detects language/script.
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Optional
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Regular expressions for script & code-mixing detection
RE_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
RE_LATIN = re.compile(r"[a-zA-Z]")
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MENTION = re.compile(r"@\w+")
RE_MULTISPACE = re.compile(r"\s+")

# Lexical markers for romanized Indic code-mixing
HINGLISH_MARKERS = {
    "hai", "hain", "kya", "kyu", "kyun", "nahi", "nahin", "bohot", "bahut", "achha", "accha",
    "bhai", "yaar", "raha", "rahi", "rahe", "gaya", "gayi", "gaye", "karo", "karna", "yeh",
    "woh", "isko", "usko", "apna", "apni", "pe", "se", "aur", "toh", "hi", "bhi", "tha", "thi", "the"
}

MARATHI_ROMAN_MARKERS = {
    "ahe", "ahet", "nahi", "nahiy", "kiti", "khup", "chan", "bhava", "bhau", "kadak",
    "bagha", "baghitle", "chalavla", "ahes", "tula", "mala", "amhi", "yanchi", "tyancha",
    "kela", "keli", "kele", "tras", "gela", "geli", "javun", "punyatil", "shikshan"
}

MARATHI_DEV_MARKERS = {
    "आहे", "नाही", "खूप", "छान", "चित्रपट", "सरकार", "खड्डे", "प्रशासन", "वाटले", "मित्र", "जेवण", "पाहिली", "चाललंय"
}


def clean_social_text(text: str) -> str:
    """Clean social media post text."""
    if not isinstance(text, str):
        return ""
    text = RE_URL.sub("", text)
    text = RE_MULTISPACE.sub(" ", text).strip()
    return text


def detect_script(text: str) -> str:
    """Identify script: devanagari, roman, or mixed."""
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    
    dev_count = len(RE_DEVANAGARI.findall(text))
    lat_count = len(RE_LATIN.findall(text))
    
    if dev_count > 0 and lat_count > 0:
        return "mixed"
    elif dev_count > 0 and lat_count == 0:
        return "devanagari"
    elif lat_count > 0 and dev_count == 0:
        return "roman"
    return "other"


def detect_language(text: str, default_lang: Optional[str] = None) -> str:
    """Identify language: hinglish, marathi, hindi, or english."""
    if not isinstance(text, str) or not text.strip():
        return default_lang or "unknown"
    
    script = detect_script(text)
    lower_words = set(re.findall(r"\b[a-zA-Z\u0900-\u097F]+\b", text.lower()))

    # Check Devanagari Marathi markers
    if script in ["devanagari", "mixed"]:
        if any(marker in text for marker in MARATHI_DEV_MARKERS):
            return "marathi"
        if script == "devanagari":
            # Default devanagari to marathi/hindi context
            return default_lang if default_lang in ["marathi", "hindi"] else "marathi"

    # Check Romanized Marathi markers
    marathi_matches = len(lower_words.intersection(MARATHI_ROMAN_MARKERS))
    hinglish_matches = len(lower_words.intersection(HINGLISH_MARKERS))

    if marathi_matches > 0 and marathi_matches >= hinglish_matches:
        return "marathi"
    elif hinglish_matches > 0:
        return "hinglish"
    
    if script == "roman" and hinglish_matches == 0 and marathi_matches == 0:
        if default_lang in ["hinglish", "marathi"]:
            return default_lang
        return "english"

    return default_lang or "hinglish"


def is_text_code_mixed(text: str, lang: str, script: str) -> bool:
    """Determine if the utterance represents code-mixing."""
    if script == "mixed":
        return True
    if lang in ["hinglish", "marathi"] and script == "roman":
        return True
    return False


class SparkPreprocessor:
    """
    Big Data Preprocessor leveraging PySpark where available,
    with distributed fallback processing.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = {}
        if os.path.exists(config_path):
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

        self.raw_dir = self.config.get("paths", {}).get("raw_data_dir", "data/raw")
        self.processed_dir = self.config.get("paths", {}).get("processed_data_dir", "data/processed")
        self.gold_dir = self.config.get("paths", {}).get("gold_data_dir", "data/gold")

    def load_and_standardize_sentimix(self, file_path: str) -> pd.DataFrame:
        """Standardize SentiMix dataset schema."""
        df = pd.read_csv(file_path)
        standardized = []
        for idx, row in df.iterrows():
            text = str(row.get("tweet_text", ""))
            raw_label = str(row.get("label", "neutral")).lower()
            sentiment = raw_label if raw_label in ["positive", "negative", "neutral"] else "neutral"
            
            cleaned = clean_social_text(text)
            script = detect_script(cleaned)
            lang = detect_language(cleaned, default_lang="hinglish")
            
            standardized.append({
                "id": str(row.get("tweet_id", f"sm_{idx}")),
                "text": cleaned,
                "language": lang,
                "script": script,
                "sentiment": sentiment,
                "toxicity": "non-toxic", # Default unless abusive
                "sarcasm": ("shabaash" in cleaned.lower() or "wah bhai" in cleaned.lower()) and sentiment == "negative",
                "source": "sentimix",
                "is_code_mixed": is_text_code_mixed(cleaned, lang, script)
            })
        return pd.DataFrame(standardized)

    def load_and_standardize_mahasent(self, file_path: str) -> pd.DataFrame:
        """Standardize L3Cube-MahaSent dataset schema."""
        df = pd.read_csv(file_path)
        standardized = []
        for idx, row in df.iterrows():
            text = str(row.get("post_content", ""))
            raw_label = str(row.get("sentiment_tag", "neutral")).lower()
            sentiment = raw_label if raw_label in ["positive", "negative", "neutral"] else "neutral"
            
            cleaned = clean_social_text(text)
            script = detect_script(cleaned)
            lang = detect_language(cleaned, default_lang="marathi")
            
            standardized.append({
                "id": str(row.get("post_id", f"ms_{idx}")),
                "text": cleaned,
                "language": lang,
                "script": script,
                "sentiment": sentiment,
                "toxicity": "non-toxic" if sentiment != "negative" or "aukaat" not in cleaned.lower() else "toxic",
                "sarcasm": ("wah re wah" in cleaned.lower()) and sentiment == "negative",
                "source": "mahasent",
                "is_code_mixed": is_text_code_mixed(cleaned, lang, script)
            })
        return pd.DataFrame(standardized)

    def load_and_standardize_hasoc(self, file_path: str) -> pd.DataFrame:
        """Standardize HASOC dataset schema."""
        df = pd.read_csv(file_path)
        standardized = []
        for idx, row in df.iterrows():
            text = str(row.get("text", ""))
            task_1 = str(row.get("task_1", "NOT")).upper()
            toxicity = "toxic" if task_1 == "HOF" else "non-toxic"
            
            cleaned = clean_social_text(text)
            script = detect_script(cleaned)
            lang = detect_language(cleaned, default_lang="hinglish")
            sentiment = "negative" if toxicity == "toxic" else ("positive" if "congratulations" in cleaned.lower() else "neutral")

            standardized.append({
                "id": str(row.get("id", f"hs_{idx}")),
                "text": cleaned,
                "language": lang,
                "script": script,
                "sentiment": sentiment,
                "toxicity": toxicity,
                "sarcasm": False,
                "source": "hasoc",
                "is_code_mixed": is_text_code_mixed(cleaned, lang, script)
            })
        return pd.DataFrame(standardized)

    def load_and_standardize_gluecos(self, file_path: str) -> pd.DataFrame:
        """Standardize GLUECoS dataset schema."""
        df = pd.read_csv(file_path)
        standardized = []
        for idx, row in df.iterrows():
            text = str(row.get("utterance", ""))
            raw_label = str(row.get("sentiment_gold", "neutral")).lower()
            sentiment = raw_label if raw_label in ["positive", "negative", "neutral"] else "neutral"
            
            cleaned = clean_social_text(text)
            script = detect_script(cleaned)
            lang = detect_language(cleaned, default_lang="hinglish")

            standardized.append({
                "id": str(row.get("uid", f"gc_{idx}")),
                "text": cleaned,
                "language": lang,
                "script": script,
                "sentiment": sentiment,
                "toxicity": "non-toxic",
                "sarcasm": ("wah bhai" in cleaned.lower() and sentiment == "negative"),
                "source": "gluecos",
                "is_code_mixed": is_text_code_mixed(cleaned, lang, script)
            })
        return pd.DataFrame(standardized)

    def run_preprocessing_pipeline(self) -> pd.DataFrame:
        """
        Execute full distributed preprocessing pipeline across all raw files.
        Deduplicates, cleans, normalizes schema, and writes out unified corpus.
        """
        os.makedirs(self.processed_dir, exist_ok=True)
        dfs = []

        # 1. SentiMix
        sentimix_file = os.path.join(self.raw_dir, "sample_sentimix.csv")
        if os.path.exists(sentimix_file):
            logger.info(f"Ingesting & Standardizing {sentimix_file}")
            dfs.append(self.load_and_standardize_sentimix(sentimix_file))

        # 2. MahaSent
        mahasent_file = os.path.join(self.raw_dir, "sample_mahasent.csv")
        if os.path.exists(mahasent_file):
            logger.info(f"Ingesting & Standardizing {mahasent_file}")
            dfs.append(self.load_and_standardize_mahasent(mahasent_file))

        # 3. HASOC
        hasoc_file = os.path.join(self.raw_dir, "sample_hasoc.csv")
        if os.path.exists(hasoc_file):
            logger.info(f"Ingesting & Standardizing {hasoc_file}")
            dfs.append(self.load_and_standardize_hasoc(hasoc_file))

        # 4. GLUECoS
        gluecos_file = os.path.join(self.raw_dir, "sample_gluecos.csv")
        if os.path.exists(gluecos_file):
            logger.info(f"Ingesting & Standardizing {gluecos_file}")
            dfs.append(self.load_and_standardize_gluecos(gluecos_file))

        # 5. Master Gold if present
        gold_file = os.path.join(self.gold_dir, "gold_master.csv")
        if os.path.exists(gold_file):
            logger.info(f"Ingesting & Standardizing Gold Master {gold_file}")
            gold_df = pd.read_csv(gold_file)
            dfs.append(gold_df)

        if not dfs:
            raise FileNotFoundError("No raw data found in data/raw to preprocess.")

        unified_df = pd.concat(dfs, ignore_index=True)
        initial_count = len(unified_df)

        # PySpark / Distributed Processing Demonstration
        from bigdata.spark_session import get_spark_session
        spark = get_spark_session()
        
        if spark is not None:
            try:
                logger.info("Executing PySpark distributed DataFrame transformations...")
                spark_df = spark.createDataFrame(unified_df)
                
                # PySpark Deduplication on text
                spark_dedup = spark_df.dropDuplicates(["text"])
                
                # PySpark filter missing or empty text
                spark_clean = spark_dedup.filter(spark_dedup.text != "").filter(spark_dedup.text.isNotNull())
                
                # Convert back to Pandas for downstream serialization
                unified_df = spark_clean.toPandas()
                logger.info(f"PySpark processing complete. Rows before: {initial_count}, Rows after: {len(unified_df)}")
            except Exception as e:
                logger.warning(f"PySpark execution encountered ({e}), using distributed in-memory pipeline.")
                unified_df = unified_df.drop_duplicates(subset=["text"]).dropna(subset=["text"])
        else:
            unified_df = unified_df.drop_duplicates(subset=["text"]).dropna(subset=["text"])

        # Reset IDs cleanly
        unified_df = unified_df.reset_index(drop=True)
        unified_df["id"] = [f"post_{i:04d}" for i in range(len(unified_df))]

        output_path = os.path.join(self.processed_dir, "unified_social_corpus.csv")
        unified_df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Unified social corpus successfully written to {output_path} ({len(unified_df)} records)")

        # Save metadata summary
        meta = {
            "total_records": len(unified_df),
            "sources": unified_df["source"].value_counts().to_dict(),
            "languages": unified_df["language"].value_counts().to_dict(),
            "scripts": unified_df["script"].value_counts().to_dict(),
            "sentiments": unified_df["sentiment"].value_counts().to_dict(),
            "toxicities": unified_df["toxicity"].value_counts().to_dict(),
            "code_mixed_count": int(unified_df["is_code_mixed"].sum()),
            "sarcastic_count": int(unified_df["sarcasm"].sum()),
        }
        meta_path = os.path.join(self.processed_dir, "corpus_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        return unified_df


def preprocess_all_datasets() -> pd.DataFrame:
    preprocessor = SparkPreprocessor()
    return preprocessor.run_preprocessing_pipeline()


if __name__ == "__main__":
    df = preprocess_all_datasets()
    print(f"Preprocessed unified dataset with {len(df)} posts.")
    print(df.head(5))
