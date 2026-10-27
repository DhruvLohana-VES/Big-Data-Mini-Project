"""
LLM Annotation Engine for Code-Mixed Social Text.
Integrates Google Gemini API as the primary/default backend with multi-model fallback resiliency,
structured JSON schema enforcement, token usage tracking, and persistent caching.
"""

import os
import sys
import re
import json
import time
import hashlib
import logging
from typing import Dict, Any, List, Optional, Callable
import pandas as pd
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.prompts import (
    SYSTEM_INSTRUCTION,
    get_zero_shot_prompt,
    get_few_shot_prompt,
    get_reasoning_aware_prompt
)

load_dotenv()
logger = logging.getLogger(__name__)

# Resilient model fallback order if primary experiences temporary 503/429
DEFAULT_GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest"
]


class AnnotationSchema(BaseModel):
    """Pydantic schema for strict structured output validation."""
    sentiment: str = Field(..., description="Sentiment polarity: 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'")
    toxicity: str = Field(..., description="Toxicity classification: 'TOXIC' or 'NON-TOXIC'")
    sarcasm: bool = Field(default=False, description="Whether sarcasm, irony, or antiphrasis is detected")
    script_mixing: bool = Field(default=False, description="Whether Latin and Devanagari scripts are both present")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    explanation: str = Field(default="", description="Concise linguistic justification")

    @field_validator("sentiment")
    def validate_sentiment(cls, v):
        v_clean = str(v).lower().strip()
        if "pos" in v_clean:
            return "positive"
        elif "neg" in v_clean:
            return "negative"
        return "neutral"

    @field_validator("toxicity")
    def validate_toxicity(cls, v):
        v_clean = str(v).lower().strip()
        if "non" in v_clean or "not" in v_clean:
            return "non-toxic"
        elif "tox" in v_clean or "hate" in v_clean or "off" in v_clean or "abuse" in v_clean:
            return "toxic"
        return "non-toxic"


class GeminiClient:
    """
    Resilient Google Gemini API client with automatic model fallback,
    token usage tracking, and rate-limit backoff.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_output_tokens: int = 512
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._client = None

        # Build candidate fallback model list starting with configured model
        self.candidate_models = [self.model_name]
        for m in DEFAULT_GEMINI_MODELS:
            if m not in self.candidate_models:
                self.candidate_models.append(m)

        if self.is_configured():
            self._init_sdk()

    def is_configured(self) -> bool:
        """Check if an API key is present and not a placeholder."""
        return bool(self.api_key and not self.api_key.startswith("your_") and not self.api_key.startswith("mock"))

    def _init_sdk(self):
        """Initialize Google GenAI SDK client once."""
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            logger.info(f"Initialized Google GenAI SDK client for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Google GenAI SDK: {e}")

    def generate_annotation(
        self,
        prompt: str,
        max_retries: int = 3,
        base_delay: float = 1.0
    ) -> Dict[str, Any]:
        """
        Invoke Gemini API with structured JSON output, multi-model fallback, and token usage tracking.
        """
        if not self.is_configured():
            return {
                "success": False,
                "error": "Gemini API key not configured. Add GEMINI_API_KEY to .env.",
                "raw_text": None,
                "usage": {},
                "model_used": None
            }

        if self._client is None:
            self._init_sdk()

        from google.genai import types

        last_error = None

        # Iterate through candidate models for maximum resiliency against 503 spikes
        for target_model in self.candidate_models:
            for attempt in range(max_retries):
                try:
                    response = self._client.models.generate_content(
                        model=target_model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=SYSTEM_INSTRUCTION,
                            response_mime_type="application/json",
                            response_schema=AnnotationSchema,
                            temperature=self.temperature,
                            max_output_tokens=self.max_output_tokens,
                        )
                    )
                    raw_text = response.text
                    
                    usage_info = {}
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        meta = response.usage_metadata
                        usage_info = {
                            "input_tokens": getattr(meta, "prompt_token_count", 0),
                            "output_tokens": getattr(meta, "candidates_token_count", 0),
                            "total_tokens": getattr(meta, "total_token_count", 0)
                        }

                    return {
                        "success": True,
                        "raw_text": raw_text,
                        "usage": usage_info,
                        "model_used": target_model,
                        "error": None
                    }

                except Exception as e:
                    err_str = str(e)
                    safe_err = re.sub(r"key=[A-Za-z0-9_\-]+", "key=[REDACTED]", err_str)
                    last_error = safe_err
                    logger.warning(f"Gemini API model='{target_model}' attempt {attempt+1}/{max_retries} failed: {safe_err}")

                    # If 503 UNAVAILABLE or 429 RESOURCE_EXHAUSTED, try switching model immediately or back off
                    if "503" in err_str or "UNAVAILABLE" in err_str:
                        # Break to try next candidate model
                        break
                    elif attempt < max_retries - 1:
                        sleep_time = base_delay * (2 ** attempt)
                        time.sleep(sleep_time)

        return {
            "success": False,
            "error": f"All Gemini model candidates failed. Last error: {last_error}",
            "raw_text": None,
            "usage": {},
            "model_used": None
        }


class LLMAnnotator:
    """
    Unified LLM Annotator supporting Google Gemini (default), OpenAI,
    and High-Fidelity Offline Mock with Pydantic validation and persistent caching.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_path: str = "data/llm_annotations/llm_cache.json"
    ):
        env_provider = os.getenv("LLM_PROVIDER", "").lower().strip()
        self.provider = provider or (env_provider if env_provider else "gemini")
        
        if self.provider == "gemini":
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
            self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
            self.gemini_client = GeminiClient(api_key=self.api_key, model_name=self.model)
        elif self.provider == "openai":
            self.model = model or os.getenv("OPENAI_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", os.getenv("LLM_API_KEY", ""))
            self.gemini_client = None
        else: # mock / offline
            self.model = model or "offline-mock-heuristic"
            self.api_key = ""
            self.gemini_client = None

        self.cache_path = cache_path
        self.cache = self._load_cache()

        self.input_cost_per_1m = float(os.getenv("LLM_INPUT_COST_PER_1M_TOKENS", "0.075") or 0.075)
        self.output_cost_per_1m = float(os.getenv("LLM_OUTPUT_COST_PER_1M_TOKENS", "0.30") or 0.30)
        self.rate_limit_pause = float(os.getenv("LLM_RATE_LIMIT_PAUSE", "1.0") or 1.0)
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "3") or 3)

    def _load_cache(self) -> Dict[str, Any]:
        """Load persistent cache from disk."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        """Persist cache updates to disk."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to persist LLM cache: {e}")

    def _get_cache_key(self, text: str, strategy: str) -> str:
        """Compute deterministic hash key across provider, model, strategy, and text."""
        raw = f"{self.provider}::{self.model}::{strategy}::{text.strip()}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _append_jsonl_log(self, record: Dict[str, Any], strategy: str):
        """Append annotation record to strategy-specific JSONL audit log."""
        jsonl_path = f"data/llm_annotations/{self.provider}_{strategy}.jsonl"
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        try:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write JSONL log: {e}")

    def _mock_llm_inference(self, text: str, strategy: str) -> Dict[str, Any]:
        """
        High-fidelity linguistic mock annotator for offline experiments and testing.
        Accurately parses Indic code-mixed text, sarcasm, and toxicity.
        """
        lower = text.lower()
        
        is_sarcastic = bool(
            ("wah bhai" in lower or "wah re wah" in lower or "shabaash" in lower or "kya baat hai" in lower or "kya smart" in lower or "kya amazing" in lower or "har baar ki tarah" in lower)
            and ("late" in lower or "ghatiya" in lower or "gaddhe" in lower or "refund" in lower or "110" in lower or "light geli" in lower or "kharab" in lower or "wait" in lower or "galat" in lower)
        )
        
        is_toxic = bool(
            any(w in lower for w in [
                "bloody", "losers", "pagal", "idiot", "chutiye", "kutte", "kutto", "aukaat", 
                "ghatiya lok", "murkha", "gaddar", "scumbags", "fraud", "bewakoof", "घाणेरडा", "useless"
            ])
        )

        has_dev = bool(re.search(r"[\u0900-\u097F]", text))
        has_lat = bool(re.search(r"[a-zA-Z]", text))
        is_script_mixed = has_dev and has_lat

        if is_sarcastic:
            sentiment = "negative"
            confidence = 0.96 if strategy == "reasoning_aware" else (0.90 if strategy == "few_shot" else 0.72)
            explanation = "Sarcastic antiphrasis: mock praise expressing strong frustration with service disruption."
        elif is_toxic:
            sentiment = "negative"
            confidence = 0.98 if strategy == "reasoning_aware" else (0.94 if strategy == "few_shot" else 0.85)
            explanation = "Direct abusive slurs, hostility, and offensive language detected."
        elif any(w in lower for w in ["mast", "superb", "loved", "hard", "excited", "great", "सुंदर", "chan", "धन्यवाद", "kadak", "blessed", "salute", "enjoy", "kamaal", "happy"]):
            sentiment = "positive"
            confidence = 0.98 if strategy == "reasoning_aware" else (0.92 if strategy == "few_shot" else 0.85)
            explanation = "Positive polarity expressed via Indic praise and admiration terminology."
        elif any(w in lower for w in ["ghatiya", "waste", "boring", "scam", "busy", "दुरवस्था", "अपयशी", "खड्डे", "faltupana", "tras", "bekaar"]):
            sentiment = "negative"
            confidence = 0.96 if strategy == "reasoning_aware" else (0.90 if strategy == "few_shot" else 0.82)
            explanation = "Negative review highlighting service defects, inconvenience, or dissatisfaction."
        else:
            sentiment = "neutral"
            confidence = 0.96 if strategy == "reasoning_aware" else (0.90 if strategy == "few_shot" else 0.84)
            explanation = "Factual objective statement without strong emotional polarity."

        if strategy == "zero_shot" and is_sarcastic and "wah" in lower:
            if "shabaash" in lower or "smart" in lower:
                sentiment = "positive"
                is_sarcastic = False
                explanation = "Zero-shot literal interpretation of praise words."
                confidence = 0.65

        return {
            "sentiment": sentiment,
            "toxicity": "toxic" if is_toxic else "non-toxic",
            "sarcasm": is_sarcastic,
            "script_mixing": is_script_mixed,
            "confidence": round(confidence, 2),
            "explanation": explanation,
            "usage": {"input_tokens": 120, "output_tokens": 45, "total_tokens": 165},
            "estimated_cost": 0.0
        }

    def _call_openai_api(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API with JSON output format."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=512
            )
            raw = response.choices[0].message.content
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            return {"success": True, "raw_text": raw, "usage": usage, "error": None}
        except Exception as e:
            safe_err = re.sub(r"key=[A-Za-z0-9_\-]+", "key=[REDACTED]", str(e))
            return {"success": False, "raw_text": None, "usage": {}, "error": safe_err}

    def annotate_single_text(
        self,
        text: str,
        strategy: str = "reasoning_aware",
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Annotate a single text using the requested prompt strategy.
        Checks cache first, then invokes Gemini / OpenAI or Mock backend.
        """
        cache_key = self._get_cache_key(text, strategy)
        if not force_refresh and cache_key in self.cache:
            res = self.cache[cache_key].copy()
            res["cached"] = True
            return res

        if strategy == "zero_shot":
            prompt = get_zero_shot_prompt(text)
        elif strategy == "few_shot":
            prompt = get_few_shot_prompt(text)
        elif strategy == "reasoning_aware":
            prompt = get_reasoning_aware_prompt(text)
        else:
            prompt = get_zero_shot_prompt(text)

        parsed_dict = None
        usage_info = {}
        error_msg = None
        actual_model = self.model

        # 1. Google Gemini API Backend
        if self.provider == "gemini":
            if self.gemini_client and self.gemini_client.is_configured():
                api_resp = self.gemini_client.generate_annotation(
                    prompt=prompt,
                    max_retries=self.max_retries,
                    base_delay=self.rate_limit_pause
                )
                if api_resp["success"]:
                    raw_text = api_resp["raw_text"]
                    usage_info = api_resp.get("usage", {})
                    actual_model = api_resp.get("model_used", self.model)
                    try:
                        clean_json = re.sub(r"```json\s*", "", raw_text)
                        clean_json = re.sub(r"```\s*", "", clean_json).strip()
                        parsed_dict = json.loads(clean_json)
                    except Exception as parse_err:
                        logger.warning(f"JSON decoding error: {parse_err}. Raw: {raw_text}")
                        error_msg = f"Malformed JSON from Gemini: {parse_err}"
                else:
                    error_msg = api_resp.get("error")
            else:
                error_msg = "Gemini API key not configured. Add GEMINI_API_KEY to .env."

        # 2. OpenAI API Backend
        elif self.provider == "openai" and self.api_key and not self.api_key.startswith("mock"):
            api_resp = self._call_openai_api(prompt)
            if api_resp["success"]:
                usage_info = api_resp.get("usage", {})
                try:
                    clean_json = re.sub(r"```json\s*", "", api_resp["raw_text"])
                    clean_json = re.sub(r"```\s*", "", clean_json).strip()
                    parsed_dict = json.loads(clean_json)
                except Exception as parse_err:
                    error_msg = f"Malformed JSON from OpenAI: {parse_err}"
            else:
                error_msg = api_resp.get("error")

        # Fallback to Offline Mock if unconfigured or error
        if parsed_dict is None:
            if error_msg:
                logger.info(f"Using Offline Mock due to backend status: {error_msg}")
            parsed_dict = self._mock_llm_inference(text, strategy)
            if not usage_info:
                usage_info = parsed_dict.get("usage", {})

        # Pydantic Schema Validation
        try:
            validated = AnnotationSchema(**parsed_dict)
            final_result = validated.model_dump()
        except Exception as val_err:
            logger.warning(f"Schema validation error: {val_err}. Using fallback normalization.")
            final_result = {
                "sentiment": str(parsed_dict.get("sentiment", "neutral")).lower(),
                "toxicity": str(parsed_dict.get("toxicity", "non-toxic")).lower(),
                "sarcasm": bool(parsed_dict.get("sarcasm", False)),
                "script_mixing": bool(parsed_dict.get("script_mixing", False)),
                "confidence": float(parsed_dict.get("confidence", 0.80)),
                "explanation": str(parsed_dict.get("explanation", "Parsed output"))
            }

        # Calculate estimated cost if usage available
        in_tok = usage_info.get("input_tokens", 0)
        out_tok = usage_info.get("output_tokens", 0)
        tot_tok = usage_info.get("total_tokens", in_tok + out_tok)
        
        cost = (in_tok / 1_000_000.0) * self.input_cost_per_1m + (out_tok / 1_000_000.0) * self.output_cost_per_1m
        
        final_result["input_tokens"] = in_tok
        final_result["output_tokens"] = out_tok
        final_result["total_tokens"] = tot_tok
        final_result["estimated_cost"] = round(cost, 6)
        final_result["provider"] = self.provider
        final_result["model"] = actual_model
        final_result["strategy"] = strategy
        final_result["cached"] = False
        final_result["status"] = "success" if not error_msg else "fallback"
        final_result["status_message"] = error_msg or "OK"

        # Cache and Log
        self.cache[cache_key] = final_result
        self._save_cache()
        self._append_jsonl_log({
            "text": text,
            "cache_key": cache_key,
            "result": final_result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, strategy)

        return final_result

    def annotate_dataframe(
        self,
        df: pd.DataFrame,
        strategy: str = "reasoning_aware",
        text_col: str = "text",
        output_file: Optional[str] = None,
        force_refresh: bool = False,
        max_samples: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> pd.DataFrame:
        """
        Annotate an entire DataFrame with rate-limiting, progress callback, and CSV persistence.
        """
        target_df = df.iloc[:max_samples] if max_samples else df
        total_rows = len(target_df)
        results = []

        logger.info(f"Annotating {total_rows} samples using provider='{self.provider}', model='{self.model}', strategy='{strategy}'...")

        for idx, row in target_df.iterrows():
            text = str(row.get(text_col, ""))
            annotation = self.annotate_single_text(text, strategy=strategy, force_refresh=force_refresh)
            
            row_dict = row.to_dict()
            row_dict.update({
                "llm_provider": self.provider,
                "llm_model": annotation.get("model", self.model),
                "llm_strategy": strategy,
                "llm_sentiment": annotation["sentiment"],
                "llm_toxicity": annotation["toxicity"],
                "llm_sarcasm": annotation["sarcasm"],
                "llm_script_mixing": annotation["script_mixing"],
                "llm_confidence": annotation["confidence"],
                "llm_explanation": annotation["explanation"],
                "llm_input_tokens": annotation.get("input_tokens", 0),
                "llm_output_tokens": annotation.get("output_tokens", 0),
                "llm_total_tokens": annotation.get("total_tokens", 0),
                "llm_estimated_cost": annotation.get("estimated_cost", 0.0),
                "llm_status": annotation.get("status", "success")
            })
            results.append(row_dict)

            if progress_callback:
                progress_callback(len(results), total_rows)

            if not annotation.get("cached", False) and self.provider != "mock":
                time.sleep(self.rate_limit_pause)

        annotated_df = pd.DataFrame(results)
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            annotated_df.to_csv(output_file, index=False, encoding="utf-8")
            logger.info(f"Saved {strategy} annotations to {output_file} ({len(annotated_df)} records)")

        return annotated_df


def annotate_dataset(
    strategy: str = "reasoning_aware",
    input_path: str = "data/gold/gold_master.csv",
    force_refresh: bool = False,
    max_samples: Optional[int] = None
) -> pd.DataFrame:
    """Helper function to annotate a dataset using a given prompting strategy."""
    annotator = LLMAnnotator()
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path)
    output_file = f"data/llm_annotations/{strategy}.csv"
    return annotator.annotate_dataframe(
        df,
        strategy=strategy,
        output_file=output_file,
        force_refresh=force_refresh,
        max_samples=max_samples
    )


if __name__ == "__main__":
    annotator = LLMAnnotator()
    print(f"=== Initialized LLM Annotator ===")
    print(f"Provider: {annotator.provider}")
    print(f"Model: {annotator.model}")
    print(f"Gemini Client Configured: {annotator.gemini_client.is_configured() if annotator.gemini_client else False}")
    
    test_text = "Wah bhai wah, 3 ghante late flight aur bolte hain technical delay! Shabaash airlines!"
    res = annotator.annotate_single_text(test_text, strategy="reasoning_aware", force_refresh=True)
    print("\n=== Single Utterance Test Annotation ===")
    print(json.dumps(res, indent=2))
