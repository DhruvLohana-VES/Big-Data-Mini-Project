"""
Expert Prompt Engineering Specifications for Indian Code-Mixed Social Text Annotation.
Comprehensive Multilingual NLP Guidelines covering Hinglish, Marathi (Devanagari & Romanized),
English, Mixed Scripts, Sarcasm Antiphrasis, Toxicity Discrimination, and Confidence Calibration.
"""

from typing import Dict, List, Any

SYSTEM_INSTRUCTION = """You are an expert multilingual social-media text annotator specializing in Indian code-mixed language, especially Hinglish and Marathi.
Your task is to annotate social-media utterances for:
1. Sentiment: POSITIVE | NEGATIVE | NEUTRAL
2. Toxicity: TOXIC | NON-TOXIC
3. Sarcasm: boolean (true | false)
4. Script Mixing: boolean (true | false)
5. Confidence: float score between 0.0 and 1.0 (calibrated on semantic clarity)
6. Explanation: concise summary of linguistic evidence.

IMPORTANT CORE PRINCIPLES:
- Do NOT classify based only on individual words. Classify the OVERALL INTENDED MEANING of the utterance.
- The text may contain Hinglish, Marathi (Devanagari), Romanized Marathi (Marathish), English, Hindi, mixed scripts, slang, informal spellings (kya/kyaaa, bohot/bahut, khup/khoop), abbreviations, emojis, internet expressions, sarcasm, irony, mock praise, indirect insults, and rhetorical statements.
- Do NOT translate the input before classification. Classify directly in its original code-mixed form.

TASK DEFINITIONS:
1. SENTIMENT (POSITIVE | NEGATIVE | NEUTRAL):
   - POSITIVE: Genuine approval, happiness, satisfaction, praise, gratitude, excitement, affection.
   - NEGATIVE: Genuine criticism, anger, disappointment, frustration, dislike, contempt.
   - NEUTRAL: Primarily factual, informational, descriptive, or lacking emotional stance.
   - SARCASTIC SENTIMENT: If a statement uses apparently positive words to communicate a negative attitude (e.g. 'Wah kya amazing service hai, 2 ghante wait karwaya'), classify sentiment according to the intended negative meaning (sentiment = NEGATIVE, sarcasm = true).

2. SARCASM (true | false):
   - true when the speaker intentionally uses wording whose literal meaning contradicts intended meaning (mock praise, irony, indirect criticism, rhetorical praise contradicted by context).
   - Look at the COMPLETE utterance. Do not require an emoji or explicit marker.
   - Do NOT label genuine enthusiastic praise as sarcasm merely because it contains exaggeration.

3. TOXICITY (TOXIC | NON-TOXIC):
   - Toxicity and sentiment are DIFFERENT. A negative post or complaint is NOT toxic unless it contains direct personal insults, harassment, threats, hate speech, severe profanity, or targeted abuse.
   - Negative criticism alone (e.g. 'Service bahut kharab thi') is NON-TOXIC.

4. SCRIPT MIXING (true | false):
   - Refers to using MORE THAN ONE WRITING SYSTEM in the same utterance (e.g. Latin + Devanagari).
   - Language mixing is NOT script mixing. Romanized Hindi/Marathi ('Hinglish mein baat karna') is entirely Latin script -> script_mixing = false.
   - 'भाई kya kar raha hai?' or 'Superb काम केलं आहे' -> script_mixing = true.

5. CONFIDENCE (0.0 to 1.0):
   - Calibrate based on semantic clarity: Clear (0.90-1.00), Moderately ambiguous (0.70-0.90), Highly ambiguous (0.50-0.70). Do not default to 0.98.

INTERNAL DECISION PROCESS (Do NOT output raw reasoning steps; output ONLY structured JSON):
- Understand utterance as a whole -> identify languages/slang/context -> determine literal vs intended meaning -> check for sarcasm/antiphrasis -> assign intended sentiment -> independently check toxicity -> check for multi-script -> calibrate confidence."""


# Comprehensive benchmark exemplars defining the annotation policy
FEW_SHOT_EXEMPLARS = [
    {
        "text": "Yaar aaj ka match kya mast tha, full enjoy kiya.",
        "annotation": {
            "sentiment": "POSITIVE",
            "toxicity": "NON-TOXIC",
            "sarcasm": False,
            "script_mixing": False,
            "confidence": 0.98,
            "explanation": "Genuine positive enthusiasm expressing enjoyment of a cricket match in Hinglish Latin script."
        }
    },
    {
        "text": "Service bahut bekaar thi, pura din waste ho gaya.",
        "annotation": {
            "sentiment": "NEGATIVE",
            "toxicity": "NON-TOXIC",
            "sarcasm": False,
            "script_mixing": False,
            "confidence": 0.96,
            "explanation": "Direct negative complaint about poor service without abusive hostility or script mixing."
        }
    },
    {
        "text": "Wah kya amazing service hai, 2 ghante wait karwaya.",
        "annotation": {
            "sentiment": "NEGATIVE",
            "toxicity": "NON-TOXIC",
            "sarcasm": True,
            "script_mixing": False,
            "confidence": 0.96,
            "explanation": "Sarcastic antiphrasis: uses mock praise ('Wah kya amazing service') to communicate dissatisfaction with 2-hour delay."
        }
    },
    {
        "text": "Bahut badiya bhai, har baar ki tarah kaam kharab kar diya.",
        "annotation": {
            "sentiment": "NEGATIVE",
            "toxicity": "NON-TOXIC",
            "sarcasm": True,
            "script_mixing": False,
            "confidence": 0.95,
            "explanation": "Sarcastic praise ('Bahut badiya') contradicted by the context of ruined work."
        }
    },
    {
        "text": "Tu ek useless idiot hai bloody loser.",
        "annotation": {
            "sentiment": "NEGATIVE",
            "toxicity": "TOXIC",
            "sarcasm": False,
            "script_mixing": False,
            "confidence": 0.99,
            "explanation": "Direct abusive hostility and personal insults targeting the recipient."
        }
    },
    {
        "text": "Bro aaj college mein lecture bahut boring tha.",
        "annotation": {
            "sentiment": "NEGATIVE",
            "toxicity": "NON-TOXIC",
            "sarcasm": False,
            "script_mixing": False,
            "confidence": 0.95,
            "explanation": "Hinglish language mixing in pure Latin script expressing mild boredom."
        }
    },
    {
        "text": "भाई आज college में क्या हुआ?",
        "annotation": {
            "sentiment": "NEUTRAL",
            "toxicity": "NON-TOXIC",
            "sarcasm": False,
            "script_mixing": True,
            "confidence": 0.96,
            "explanation": "Factual question mixing Devanagari script ('भाई आज', 'में क्या हुआ?') and Latin script ('college')."
        }
    },
    {
        "text": "आजचा दिवस खूप छान होता, मस्त वाटलं.",
        "annotation": {
            "sentiment": "POSITIVE",
            "toxicity": "NON-TOXIC",
            "sarcasm": False,
            "script_mixing": False,
            "confidence": 0.99,
            "explanation": "Genuine positive satisfaction expressed in Marathi native Devanagari script."
        }
    },
    {
        "text": "Aajcha divas khup chan hota, khup anand zala.",
        "annotation": {
            "sentiment": "POSITIVE",
            "toxicity": "NON-TOXIC",
            "sarcasm": False,
            "script_mixing": False,
            "confidence": 0.98,
            "explanation": "Romanized Marathi (Marathish) in Latin script expressing genuine happiness."
        }
    },
    {
        "text": "Wah re wah, 2 taas light geli aani he bill pathavtat!",
        "annotation": {
            "sentiment": "NEGATIVE",
            "toxicity": "NON-TOXIC",
            "sarcasm": True,
            "script_mixing": False,
            "confidence": 0.94,
            "explanation": "Romanized Marathi sarcasm ('Wah re wah') mocking electricity outage and billing."
        }
    }
]


def get_zero_shot_prompt(text: str) -> str:
    """
    Generate Zero-Shot prompt.
    Provides task definition, language context, label definitions, and input text (no exemplars).
    """
    return f"""You are an expert multilingual Indic computational linguist. Annotate the following social media utterance in its original code-mixed form.

Target Text: "{text}"

Annotation Guidelines:
1. sentiment: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL' (Based on INTENDED attitude; sarcastic praise = NEGATIVE)
2. toxicity: 'TOXIC' (abuse/threats/slurs) | 'NON-TOXIC' (disappointment/criticism is non-toxic)
3. sarcasm: true (ironic mock praise, indirect criticism) | false
4. script_mixing: true (both Latin and Devanagari alphabets in same text) | false (Romanized Hindi/Marathi alone is false)
5. confidence: float between 0.0 and 1.0 (calibrated on semantic ambiguity)
6. explanation: concise justification of classification evidence

Output strictly valid JSON conforming to the schema."""


def get_few_shot_prompt(text: str) -> str:
    """
    Generate Few-Shot prompt with reference exemplars.
    Ensures target sample is never included in reference examples.
    """
    exemplars_text = ""
    for idx, ex in enumerate(FEW_SHOT_EXEMPLARS, 1):
        if ex["text"].strip().lower() == text.strip().lower():
            continue
        exemplars_text += f"""Example {idx}:
Text: "{ex['text']}"
Output:
{{
  "sentiment": "{ex['annotation']['sentiment']}",
  "toxicity": "{ex['annotation']['toxicity']}",
  "sarcasm": {str(ex['annotation']['sarcasm']).lower()},
  "script_mixing": {str(ex['annotation']['script_mixing']).lower()},
  "confidence": {ex['annotation']['confidence']},
  "explanation": "{ex['annotation']['explanation']}"
}}

"""

    return f"""Reference Exemplars of Indian Code-Mixed Social Text Annotation Policy:

{exemplars_text}Now annotate the following target utterance:

Target Text: "{text}"

Output:"""


def get_reasoning_aware_prompt(text: str) -> str:
    """
    Generate Reasoning-Aware prompt.
    Instructs the model to perform internal 10-step pragmatic and linguistic validation
    and output strictly the calibrated structured JSON.
    """
    return f"""Perform an expert linguistic and pragmatic analysis of the following Indian social media utterance:
1. Examine code-mixing dynamics (Hindi/Marathi in Latin or Devanagari).
2. Check for pragmatic nuances (sarcasm, irony, mock praise masking complaint).
3. Distinguish toxic abuse from ordinary complaints or dissatisfaction.
4. Distinguish language mixing from multi-script mixing.
5. Calibrate confidence based on semantic clarity vs ambiguity.

Target Text: "{text}"

Return strictly the structured JSON object:
{{
  "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
  "toxicity": "TOXIC" | "NON-TOXIC",
  "sarcasm": true | false,
  "script_mixing": true | false,
  "confidence": <float between 0.0 and 1.0>,
  "explanation": "<concise justification>"
}}"""
