"""
Multilingual Coercion Detection Support
=========================================
Extends the English DeBERTaV3 coercion detector to support Indian languages.

Strategy:
1. Language detection on input text
2. For non-English text: translate to English using a lightweight model,
   then run through the English coercion detector
3. Additionally check for language-specific coercion patterns
   (Hindi/Marathi urgency phrases, financial scam terminology)

Supported languages:
- English (native)
- Hindi / Hinglish (via translation + pattern matching)
- Marathi (via translation + pattern matching)
- Tamil, Telugu, Kannada (via translation)
- Bengali, Gujarati (via translation)

For production: fine-tune a multilingual model (XLM-RoBERTa or IndicBERT)
directly on coercion data in all target languages.
"""

import re
from typing import Dict, Tuple, Optional


# Common coercion keywords and phrases in Indian languages
# These are checked BEFORE translation as a fast pre-filter

HINDI_COERCION_PATTERNS = {
    "urgency": [
        r"turant|तुरंत",            # immediately
        r"abhi|अभी",               # right now
        r"jaldi|जल्दी",             # quickly
        r"der mat karo|देर मत करो",  # don't delay
        r"samay khatam|समय खत्म",   # time is up
        r"aakhri mauka|आखरी मौका",  # last chance
    ],
    "financial": [
        r"paisa bhejo|पैसा भेजो",       # send money
        r"transfer karo|ट्रांसफर करो",   # make transfer
        r"OTP batao|OTP बताओ",          # tell OTP
        r"bank account|बैंक अकाउंट",
        r"UPI|upi|यूपीआई",
        r"loan|लोन|कर्ज",               # loan/debt
    ],
    "authority": [
        r"police|पुलिस",
        r"court|कोर्ट|अदालत",
        r"sarkar|सरकार",               # government
        r"tax department|कर विभाग",
        r"CBI|सीबीआई",
        r"income tax|इनकम टैक्स",
    ],
    "threat": [
        r"giraftar|गिरफ्तार",          # arrest
        r"jail|जेल",
        r"FIR",
        r"band ho jayega|बंद हो जाएगा",  # will be closed
        r"block|ब्लॉक",
        r"cancel|कैंसल",
    ],
}

MARATHI_COERCION_PATTERNS = {
    "urgency": [
        r"लगेच|lagech",        # immediately
        r"आत्ता|atta",          # right now
        r"लवकर|lavkar",        # quickly
    ],
    "financial": [
        r"पैसे पाठवा|paise pathva",   # send money
        r"ट्रान्सफर करा|transfer kara",
    ],
    "threat": [
        r"अटक|atak",          # arrest
        r"तुरुंग|turung",       # jail
    ],
}

# Compile patterns for fast matching
def _compile_patterns(pattern_dict):
    compiled = {}
    for category, patterns in pattern_dict.items():
        compiled[category] = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns]
    return compiled

HINDI_COMPILED = _compile_patterns(HINDI_COERCION_PATTERNS)
MARATHI_COMPILED = _compile_patterns(MARATHI_COERCION_PATTERNS)


def detect_language(text: str) -> str:
    """
    Simple language detection based on script analysis.
    Returns: 'en', 'hi', 'mr', 'ta', 'te', 'kn', 'bn', 'gu', or 'hinglish'
    """
    # Count characters by Unicode block
    devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    tamil = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
    telugu = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
    kannada = sum(1 for c in text if '\u0C80' <= c <= '\u0CFF')
    bengali = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
    gujarati = sum(1 for c in text if '\u0A80' <= c <= '\u0AFF')
    latin = sum(1 for c in text if 'a' <= c.lower() <= 'z')

    total = len(text.strip())
    if total == 0:
        return "en"

    # Hinglish: significant mix of Latin + Devanagari
    if latin > total * 0.3 and devanagari > total * 0.1:
        return "hinglish"

    # Pure script detection
    if devanagari > total * 0.3:
        return "hi"  # Could be Hindi or Marathi — check patterns later
    if tamil > total * 0.3:
        return "ta"
    if telugu > total * 0.3:
        return "te"
    if kannada > total * 0.3:
        return "kn"
    if bengali > total * 0.3:
        return "bn"
    if gujarati > total * 0.3:
        return "gu"

    return "en"


def check_coercion_patterns(text: str, language: str) -> Dict:
    """
    Fast pattern-based coercion check for Indian languages.
    This runs BEFORE the ML model as a pre-filter.

    Returns dict with detected categories and matched patterns.
    """
    results = {
        "language_detected": language,
        "patterns_found": [],
        "categories_triggered": [],
        "coercion_score": 0.0,
    }

    # Select pattern set based on language
    if language in ("hi", "hinglish"):
        pattern_sets = HINDI_COMPILED
    elif language == "mr":
        pattern_sets = MARATHI_COMPILED
    else:
        # No patterns for this language yet — rely on translation + ML
        return results

    for category, patterns in pattern_sets.items():
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                results["patterns_found"].extend(matches)
                if category not in results["categories_triggered"]:
                    results["categories_triggered"].append(category)

    # Score based on how many categories triggered
    n_categories = len(results["categories_triggered"])
    if n_categories >= 3:
        results["coercion_score"] = 0.85  # Very likely coercion
    elif n_categories >= 2:
        results["coercion_score"] = 0.65  # Likely coercion
    elif n_categories >= 1:
        results["coercion_score"] = 0.40  # Possible coercion
    else:
        results["coercion_score"] = 0.0

    return results


def translate_to_english(text: str, source_lang: str) -> Optional[str]:
    """
    Translate Indian language text to English for the coercion ML model.

    Uses Helsinki-NLP MarianMT models (small, fast, runs on CPU).
    Falls back to None if translation is not available.

    For production: use IndicTrans2 or Google Cloud Translation API.
    """
    try:
        from transformers import MarianMTModel, MarianTokenizer

        # Language code to MarianMT model mapping
        MODELS = {
            "hi": "Helsinki-NLP/opus-mt-hi-en",
            "hinglish": "Helsinki-NLP/opus-mt-hi-en",  # Best effort
            "mr": "Helsinki-NLP/opus-mt-mr-en",
            "ta": "Helsinki-NLP/opus-mt-ta-en",
            "te": "Helsinki-NLP/opus-mt-te-en",
            "bn": "Helsinki-NLP/opus-mt-bn-en",
            "gu": "Helsinki-NLP/opus-mt-gu-en",
        }

        model_name = MODELS.get(source_lang)
        if not model_name:
            return None

        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)

        return result

    except Exception as e:
        print(f"[Multilingual] Translation failed ({source_lang}->en): {e}")
        return None
