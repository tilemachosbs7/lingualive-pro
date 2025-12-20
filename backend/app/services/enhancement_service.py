"""
Enhancement Service - AAA-Studio Level Optimizations

This module provides advanced enhancement capabilities for the Deepgram+DeepL combo:
- Filler word removal
- Named entity preservation  
- Technical term detection
- Multi-language auto-detection
- Fallback chain management
- Circuit breaker pattern
- Comprehensive metrics
- Quality scoring
- Glossary management
- Idiom detection
- Formality auto-detection
- Semantic coherence
- Greek-specific enhancements
- RTL language support
- Session recovery
- Conversation memory
- Key terms extraction
- A/B testing framework
"""

import asyncio
import logging
import re
import time
import hashlib
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Provider health status for circuit breaker."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ProviderHealth:
    """Track provider health for circuit breaker pattern."""
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    total_requests: int = 0
    total_failures: int = 0
    avg_latency_ms: float = 0.0
    status: ProviderStatus = ProviderStatus.HEALTHY
    
    def record_success(self, latency_ms: float):
        self.consecutive_failures = 0
        self.last_success_time = datetime.now()
        self.total_requests += 1
        # Rolling average
        self.avg_latency_ms = (self.avg_latency_ms * (self.total_requests - 1) + latency_ms) / self.total_requests
        self.status = ProviderStatus.HEALTHY
    
    def record_failure(self):
        self.consecutive_failures += 1
        self.last_failure_time = datetime.now()
        self.total_requests += 1
        self.total_failures += 1
        
        # Update status based on consecutive failures
        if self.consecutive_failures >= 5:
            self.status = ProviderStatus.UNHEALTHY
        elif self.consecutive_failures >= 2:
            self.status = ProviderStatus.DEGRADED
    
    def is_available(self, cooldown_seconds: int = 30) -> bool:
        """Check if provider is available for requests."""
        if self.status == ProviderStatus.HEALTHY:
            return True
        if self.status == ProviderStatus.DEGRADED:
            return True  # Try anyway
        # Unhealthy - check cooldown
        if self.last_failure_time:
            elapsed = (datetime.now() - self.last_failure_time).total_seconds()
            return elapsed > cooldown_seconds
        return True


@dataclass
class TranslationMetrics:
    """Comprehensive metrics tracking for translation quality."""
    # Request metrics
    total_requests: int = 0
    cache_hits: int = 0
    fallback_count: int = 0
    
    # Latency metrics (ms)
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    latency_histogram: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Per-provider metrics
    provider_requests: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    provider_failures: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Per-language pair metrics  
    language_pair_requests: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    language_pair_latency: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    
    # Quality indicators
    short_translations: int = 0  # Suspiciously short
    empty_translations: int = 0
    error_count: int = 0
    
    def record_request(
        self, 
        latency_ms: float, 
        provider: str, 
        source_lang: str,
        target_lang: str,
        was_cache_hit: bool = False,
        was_fallback: bool = False,
        original_len: int = 0,
        translated_len: int = 0,
    ):
        self.total_requests += 1
        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        
        # Histogram buckets
        if latency_ms < 100:
            self.latency_histogram["<100ms"] += 1
        elif latency_ms < 300:
            self.latency_histogram["100-300ms"] += 1
        elif latency_ms < 500:
            self.latency_histogram["300-500ms"] += 1
        elif latency_ms < 1000:
            self.latency_histogram["500ms-1s"] += 1
        else:
            self.latency_histogram[">1s"] += 1
        
        if was_cache_hit:
            self.cache_hits += 1
        if was_fallback:
            self.fallback_count += 1
            
        self.provider_requests[provider] += 1
        
        lang_pair = f"{source_lang or 'auto'}->{target_lang}"
        self.language_pair_requests[lang_pair] += 1
        self.language_pair_latency[lang_pair] += latency_ms
        
        # Quality checks
        if original_len > 10 and translated_len < original_len * 0.3:
            self.short_translations += 1
        if translated_len == 0:
            self.empty_translations += 1
    
    def record_error(self, provider: str):
        self.error_count += 1
        self.provider_failures[provider] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "total_requests": self.total_requests,
            "cache_hit_rate": self.cache_hits / max(1, self.total_requests),
            "fallback_rate": self.fallback_count / max(1, self.total_requests),
            "error_rate": self.error_count / max(1, self.total_requests),
            "avg_latency_ms": self.total_latency_ms / max(1, self.total_requests),
            "min_latency_ms": self.min_latency_ms if self.min_latency_ms != float('inf') else 0,
            "max_latency_ms": self.max_latency_ms,
            "latency_distribution": dict(self.latency_histogram),
            "provider_usage": dict(self.provider_requests),
            "provider_failures": dict(self.provider_failures),
            "language_pairs": dict(self.language_pair_requests),
            "quality_warnings": {
                "short_translations": self.short_translations,
                "empty_translations": self.empty_translations,
            }
        }


class FillerWordRemover:
    """Remove filler words from transcribed text."""
    
    # Common filler words by language
    FILLER_PATTERNS = {
        "en": [
            r'\b(uh+|um+|uhm+|ah+|er+|hmm+)\b',
            r'\b(you know|i mean|like|basically|actually|literally)\b(?=\s*,|\s+)',
            r'\b(so+)\b(?=\s*,)',
        ],
        "el": [
            r'\b(εεε|ααα|χμμ|μμμ)\b',
            r'\b(δηλαδή|ξέρεις|κατάλαβες)\b(?=\s*,|\s+)',
        ],
        "de": [
            r'\b(äh+|ähm+|hmm+)\b',
            r'\b(also|sozusagen|eigentlich)\b(?=\s*,)',
        ],
        "fr": [
            r'\b(euh+|heu+|ben+|hein)\b',
            r'\b(genre|en fait|tu vois)\b(?=\s*,)',
        ],
        "es": [
            r'\b(eh+|este|pues|bueno)\b(?=\s*,)',
        ],
    }
    
    def __init__(self):
        # Pre-compile patterns for performance
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for lang, patterns in self.FILLER_PATTERNS.items():
            self._compiled_patterns[lang] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def remove_fillers(self, text: str, language: str = "en") -> str:
        """Remove filler words from text."""
        if not text:
            return text
        
        patterns = self._compiled_patterns.get(language, self._compiled_patterns.get("en", []))
        
        result = text
        for pattern in patterns:
            result = pattern.sub('', result)
        
        # Clean up extra spaces and punctuation
        result = re.sub(r'\s+', ' ', result)  # Multiple spaces
        result = re.sub(r'\s+,', ',', result)  # Space before comma
        result = re.sub(r',\s*,', ',', result)  # Double commas
        result = result.strip()
        
        return result


class NamedEntityPreserver:
    """Preserve named entities (names, brands, technical terms) during translation."""
    
    # Common patterns for named entities
    ENTITY_PATTERNS = [
        # Capitalized words (potential proper nouns)
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
        # All caps (acronyms, brands)
        r'\b[A-Z]{2,}\b',
        # CamelCase (code, brands)
        r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',
        # Technical terms with special chars
        r'\b\w+(?:\.\w+)+\b',  # domain.com, file.ext
        r'\b\w+(?:-\w+)+\b',   # kebab-case
        r'\b\w+(?:_\w+)+\b',   # snake_case
    ]
    
    # Known entities to always preserve
    KNOWN_ENTITIES = {
        # Tech companies
        "Google", "Microsoft", "Apple", "Amazon", "Meta", "Facebook", "Twitter",
        "OpenAI", "DeepL", "Deepgram", "AssemblyAI",
        # Programming
        "Python", "JavaScript", "TypeScript", "React", "Angular", "Vue",
        "FastAPI", "Django", "Flask", "Node.js", "npm",
        # Common proper nouns
        "API", "SDK", "JSON", "HTTP", "HTTPS", "WebSocket", "REST",
    }
    
    def __init__(self):
        self._patterns = [re.compile(p) for p in self.ENTITY_PATTERNS]
        self._entity_cache: Dict[str, str] = {}
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        entities = set()
        
        # Check known entities
        for entity in self.KNOWN_ENTITIES:
            if entity.lower() in text.lower():
                entities.add(entity)
        
        # Check patterns
        for pattern in self._patterns:
            matches = pattern.findall(text)
            entities.update(matches)
        
        return list(entities)
    
    def create_placeholders(self, text: str, entities: List[str]) -> Tuple[str, Dict[str, str]]:
        """Replace entities with placeholders for translation."""
        placeholder_map = {}
        result = text
        
        for i, entity in enumerate(sorted(entities, key=len, reverse=True)):
            placeholder = f"__ENTITY_{i}__"
            placeholder_map[placeholder] = entity
            # Case-insensitive replacement
            pattern = re.compile(re.escape(entity), re.IGNORECASE)
            result = pattern.sub(placeholder, result)
        
        return result, placeholder_map
    
    def restore_entities(self, text: str, placeholder_map: Dict[str, str]) -> str:
        """Restore entities from placeholders."""
        result = text
        for placeholder, entity in placeholder_map.items():
            result = result.replace(placeholder, entity)
        return result


class TechnicalTermDetector:
    """Detect and preserve technical/code terms."""
    
    TECHNICAL_PATTERNS = [
        # Code patterns
        r'`[^`]+`',  # Backtick code
        r'"[^"]*\(\)"',  # Function calls in quotes
        r'\b\w+\(\)',  # Function calls
        r'\b\w+\[\w*\]',  # Array access
        r'\$\w+',  # Variables
        r'@\w+',  # Decorators/mentions
        r'#\w+',  # Hashtags/preprocessor
        r'\b0x[0-9a-fA-F]+\b',  # Hex numbers
        r'\b\d+\.\d+\.\d+\b',  # Version numbers
        r'https?://\S+',  # URLs
        r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',  # Emails
    ]
    
    def __init__(self):
        self._patterns = [re.compile(p) for p in self.TECHNICAL_PATTERNS]
    
    def extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms that shouldn't be translated."""
        terms = []
        for pattern in self._patterns:
            matches = pattern.findall(text)
            terms.extend(matches)
        return list(set(terms))
    
    def preserve_technical_terms(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Replace technical terms with placeholders."""
        terms = self.extract_technical_terms(text)
        placeholder_map = {}
        result = text
        
        for i, term in enumerate(sorted(terms, key=len, reverse=True)):
            placeholder = f"__TECH_{i}__"
            placeholder_map[placeholder] = term
            result = result.replace(term, placeholder)
        
        return result, placeholder_map


class LanguageDetector:
    """Simple language detection based on character patterns."""
    
    LANGUAGE_PATTERNS = {
        "el": re.compile(r'[\u0370-\u03FF\u1F00-\u1FFF]'),  # Greek
        "zh": re.compile(r'[\u4E00-\u9FFF]'),  # Chinese
        "ja": re.compile(r'[\u3040-\u309F\u30A0-\u30FF]'),  # Japanese
        "ko": re.compile(r'[\uAC00-\uD7AF]'),  # Korean
        "ar": re.compile(r'[\u0600-\u06FF]'),  # Arabic
        "he": re.compile(r'[\u0590-\u05FF]'),  # Hebrew
        "ru": re.compile(r'[\u0400-\u04FF]'),  # Cyrillic (Russian, etc)
        "th": re.compile(r'[\u0E00-\u0E7F]'),  # Thai
    }
    
    def detect(self, text: str) -> Optional[str]:
        """Detect language from text. Returns None for Latin-based languages."""
        if not text:
            return None
        
        # Count characters matching each pattern
        scores = {}
        for lang, pattern in self.LANGUAGE_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                scores[lang] = len(matches)
        
        if scores:
            # Return language with most matches
            return max(scores, key=scores.get)
        
        # Could be English or other Latin-based
        return None


class QualityScorer:
    """Score translation quality based on heuristics."""
    
    def __init__(self):
        self._recent_scores: List[float] = []
        self._max_history = 100
    
    def score(
        self,
        original: str,
        translated: str,
        source_lang: Optional[str],
        target_lang: str,
        latency_ms: float,
    ) -> Dict[str, Any]:
        """Score translation quality."""
        if not original or not translated:
            return {"score": 0.0, "issues": ["empty_text"]}
        
        issues = []
        score = 1.0
        
        # Length ratio check
        len_ratio = len(translated) / len(original)
        if len_ratio < 0.3:
            issues.append("too_short")
            score -= 0.3
        elif len_ratio > 3.0:
            issues.append("too_long")
            score -= 0.2
        
        # Latency penalty
        if latency_ms > 2000:
            issues.append("slow")
            score -= 0.1
        
        # Check for untranslated markers
        if "__ENTITY_" in translated or "__TECH_" in translated:
            issues.append("unrestored_placeholders")
            score -= 0.5
        
        # Check for obvious errors
        if translated.lower() == original.lower() and source_lang != target_lang:
            issues.append("possibly_untranslated")
            score -= 0.4
        
        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))
        
        # Track history
        self._recent_scores.append(score)
        if len(self._recent_scores) > self._max_history:
            self._recent_scores.pop(0)
        
        return {
            "score": round(score, 2),
            "issues": issues,
            "avg_recent_score": round(sum(self._recent_scores) / len(self._recent_scores), 2),
        }


class DynamicRateLimiter:
    """Dynamic rate limiting based on provider performance."""
    
    def __init__(self, base_delay_ms: int = 100):
        self._base_delay_ms = base_delay_ms
        self._current_delay_ms = base_delay_ms
        self._last_request_time: float = 0
        self._consecutive_successes = 0
        self._consecutive_failures = 0
    
    async def wait_if_needed(self):
        """Wait if rate limit requires it."""
        now = time.perf_counter()
        elapsed_ms = (now - self._last_request_time) * 1000
        
        if elapsed_ms < self._current_delay_ms:
            wait_ms = self._current_delay_ms - elapsed_ms
            await asyncio.sleep(wait_ms / 1000.0)
        
        self._last_request_time = time.perf_counter()
    
    def record_success(self):
        """Decrease delay on success."""
        self._consecutive_successes += 1
        self._consecutive_failures = 0
        
        if self._consecutive_successes >= 5:
            # Reduce delay (min base delay)
            self._current_delay_ms = max(
                self._base_delay_ms,
                self._current_delay_ms * 0.8
            )
            self._consecutive_successes = 0
    
    def record_failure(self):
        """Increase delay on failure."""
        self._consecutive_failures += 1
        self._consecutive_successes = 0
        
        # Exponential backoff (max 5 seconds)
        self._current_delay_ms = min(
            5000,
            self._current_delay_ms * (1.5 ** self._consecutive_failures)
        )


# ============================================================================
# ΚΑΤΗΓΟΡΙΑ 2: TRANSLATION QUALITY & SEMANTICS
# ============================================================================

class IdiomDetector:
    """Detect and handle idioms and colloquialisms (2.3)."""
    
    # Common idioms by language (source -> meaning for context)
    IDIOMS = {
        "en": {
            "break a leg": "good luck",
            "piece of cake": "very easy",
            "hit the nail on the head": "exactly right",
            "let the cat out of the bag": "reveal a secret",
            "once in a blue moon": "very rarely",
            "under the weather": "feeling sick",
            "cost an arm and a leg": "very expensive",
            "bite the bullet": "face difficulty bravely",
            "spill the beans": "reveal secret information",
            "kick the bucket": "die",
            "burning the midnight oil": "working late",
            "barking up the wrong tree": "pursuing wrong course",
        },
        "el": {
            "βρέχει καρεκλοπόδαρα": "raining heavily",
            "τα έκανε θάλασσα": "messed everything up",
            "έπεσε στα μαλακά": "got lucky",
            "μου την έσπασε": "annoyed me",
            "τρώω τα μούτρα μου": "working very hard",
        },
    }
    
    def __init__(self):
        self._compiled_patterns: Dict[str, List[Tuple[re.Pattern, str]]] = {}
        for lang, idioms in self.IDIOMS.items():
            self._compiled_patterns[lang] = [
                (re.compile(r'\b' + re.escape(idiom) + r'\b', re.IGNORECASE), meaning)
                for idiom, meaning in idioms.items()
            ]
    
    def detect_idioms(self, text: str, language: str = "en") -> List[Dict[str, str]]:
        """Detect idioms in text."""
        found = []
        patterns = self._compiled_patterns.get(language, [])
        
        for pattern, meaning in patterns:
            match = pattern.search(text)
            if match:
                found.append({
                    "idiom": match.group(),
                    "meaning": meaning,
                    "position": match.start(),
                })
        
        return found
    
    def add_context_hints(self, text: str, language: str = "en") -> str:
        """Add context hints for idioms to help translation."""
        idioms = self.detect_idioms(text, language)
        if not idioms:
            return text
        
        # Add hints in parentheses for translator context
        result = text
        for idiom_info in sorted(idioms, key=lambda x: -x["position"]):
            hint = f" [meaning: {idiom_info['meaning']}]"
            pos = idiom_info["position"] + len(idiom_info["idiom"])
            result = result[:pos] + hint + result[pos:]
        
        return result


class FormalityDetector:
    """Auto-detect formality level of speech (2.4)."""
    
    # Indicators of informal speech
    INFORMAL_PATTERNS = {
        "en": [
            r'\b(gonna|wanna|gotta|kinda|sorta|dunno|lemme|gimme)\b',
            r'\b(yeah|yep|nope|nah|hey|yo|dude|bro|man)\b',
            r'\b(awesome|cool|sick|lit|dope|crazy)\b',
            r"(n't|'s|'re|'ll|'ve|'d)\b",  # Contractions
            r'!{2,}',  # Multiple exclamations
            r'\b(lol|omg|wtf|btw|idk|tbh)\b',
        ],
        "el": [
            r'\b(ρε|μωρέ|φίλε|μάγκα)\b',
            r'\b(τέλειο|φοβερό|τρελό)\b',
        ],
    }
    
    # Indicators of formal speech
    FORMAL_PATTERNS = {
        "en": [
            r'\b(therefore|furthermore|moreover|however|nevertheless)\b',
            r'\b(regarding|concerning|pursuant|hereby|thereof)\b',
            r'\b(shall|would like to|kindly|respectfully)\b',
            r'\b(dear|sincerely|regards|faithfully)\b',
        ],
        "el": [
            r'\b(επομένως|ωστόσο|συνεπώς|εντούτοις)\b',
            r'\b(παρακαλώ|ευχαριστώ πολύ|με εκτίμηση)\b',
        ],
    }
    
    def __init__(self):
        self._informal_compiled: Dict[str, List[re.Pattern]] = {}
        self._formal_compiled: Dict[str, List[re.Pattern]] = {}
        
        for lang, patterns in self.INFORMAL_PATTERNS.items():
            self._informal_compiled[lang] = [re.compile(p, re.IGNORECASE) for p in patterns]
        for lang, patterns in self.FORMAL_PATTERNS.items():
            self._formal_compiled[lang] = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def detect_formality(self, text: str, language: str = "en") -> str:
        """Detect formality level: 'formal', 'informal', or 'default'."""
        if not text:
            return "default"
        
        informal_count = 0
        formal_count = 0
        
        for pattern in self._informal_compiled.get(language, []):
            informal_count += len(pattern.findall(text))
        
        for pattern in self._formal_compiled.get(language, []):
            formal_count += len(pattern.findall(text))
        
        # Normalize by text length
        text_words = len(text.split())
        if text_words < 3:
            return "default"
        
        informal_ratio = informal_count / text_words
        formal_ratio = formal_count / text_words
        
        if informal_ratio > 0.1 and informal_ratio > formal_ratio * 2:
            return "less"  # DeepL formality value
        elif formal_ratio > 0.05 and formal_ratio > informal_ratio * 2:
            return "more"  # DeepL formality value
        
        return "default"


class SemanticCoherenceTracker:
    """Track term consistency across sentences (2.2)."""
    
    def __init__(self, max_terms: int = 50):
        # term -> preferred translation
        self._term_translations: Dict[str, str] = {}
        self._max_terms = max_terms
        self._term_usage_count: Dict[str, int] = defaultdict(int)
    
    def record_translation(self, source_term: str, translated_term: str):
        """Record a term translation for consistency."""
        key = source_term.lower().strip()
        if not key:
            return
        
        self._term_translations[key] = translated_term
        self._term_usage_count[key] += 1
        
        # Evict least used if over capacity
        if len(self._term_translations) > self._max_terms:
            least_used = min(self._term_usage_count, key=self._term_usage_count.get)
            del self._term_translations[least_used]
            del self._term_usage_count[least_used]
    
    def get_consistent_translation(self, source_term: str) -> Optional[str]:
        """Get previously used translation for consistency."""
        key = source_term.lower().strip()
        if key in self._term_translations:
            self._term_usage_count[key] += 1
            return self._term_translations[key]
        return None
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract potential key terms from text."""
        # Simple extraction: capitalized words, repeated words
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        word_counts = defaultdict(int)
        for word in text.lower().split():
            word = re.sub(r'[^\w]', '', word)
            if len(word) > 3:
                word_counts[word] += 1
        
        # Return words appearing multiple times
        repeated = [w for w, c in word_counts.items() if c >= 2]
        return list(set(words + repeated))
    
    def clear(self):
        """Clear all tracked terms."""
        self._term_translations.clear()
        self._term_usage_count.clear()


# ============================================================================
# ΚΑΤΗΓΟΡΙΑ 4: ROBUSTNESS & ERROR HANDLING
# ============================================================================

class SessionRecovery:
    """Session recovery for connection drops (4.5)."""
    
    def __init__(self, max_history: int = 20):
        self._session_history: Dict[str, List[Dict[str, Any]]] = {}
        self._max_history = max_history
    
    def save_state(self, session_id: str, state: Dict[str, Any]):
        """Save session state for recovery."""
        if session_id not in self._session_history:
            self._session_history[session_id] = []
        
        state["timestamp"] = datetime.now().isoformat()
        self._session_history[session_id].append(state)
        
        # Keep only recent history
        if len(self._session_history[session_id]) > self._max_history:
            self._session_history[session_id].pop(0)
    
    def get_last_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get last saved state for session."""
        if session_id in self._session_history and self._session_history[session_id]:
            return self._session_history[session_id][-1]
        return None
    
    def get_history(self, session_id: str, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent history for session."""
        if session_id not in self._session_history:
            return []
        return self._session_history[session_id][-count:]
    
    def clear_session(self, session_id: str):
        """Clear session history."""
        if session_id in self._session_history:
            del self._session_history[session_id]


class GracefulDegradation:
    """Handle failures gracefully (4.3)."""
    
    def __init__(self):
        self._fallback_messages = {
            "timeout": "[Translation timed out]",
            "error": "[Translation unavailable]",
            "rate_limit": "[Rate limit reached, retrying...]",
        }
    
    def get_fallback_text(self, original: str, error_type: str = "error") -> str:
        """Get fallback text when translation fails."""
        # Option 1: Return original text with marker
        if error_type == "timeout":
            return f"{original} ⏳"
        elif error_type == "rate_limit":
            return f"{original} ⏸️"
        else:
            return f"{original} ⚠️"
    
    def should_retry(self, error_type: str, attempt: int) -> bool:
        """Determine if retry is worthwhile."""
        if error_type == "rate_limit" and attempt < 3:
            return True
        if error_type == "timeout" and attempt < 2:
            return True
        if error_type == "network" and attempt < 3:
            return True
        return False


class NetworkResilience:
    """Handle network issues gracefully (4.4)."""
    
    def __init__(self):
        self._last_successful_request: float = 0
        self._connection_quality: str = "good"  # good, degraded, poor
        self._latency_samples: List[float] = []
        self._max_samples = 20
    
    def record_latency(self, latency_ms: float):
        """Record latency sample."""
        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > self._max_samples:
            self._latency_samples.pop(0)
        
        self._update_connection_quality()
        self._last_successful_request = time.perf_counter()
    
    def _update_connection_quality(self):
        """Update connection quality assessment."""
        if not self._latency_samples:
            return
        
        avg_latency = sum(self._latency_samples) / len(self._latency_samples)
        
        if avg_latency < 300:
            self._connection_quality = "good"
        elif avg_latency < 800:
            self._connection_quality = "degraded"
        else:
            self._connection_quality = "poor"
    
    def get_recommended_timeout(self, base_timeout_ms: int) -> int:
        """Get recommended timeout based on connection quality."""
        multipliers = {
            "good": 1.0,
            "degraded": 1.5,
            "poor": 2.5,
        }
        return int(base_timeout_ms * multipliers.get(self._connection_quality, 1.0))
    
    def is_connection_stable(self) -> bool:
        """Check if connection is stable."""
        return self._connection_quality in ("good", "degraded")


# ============================================================================
# ΚΑΤΗΓΟΡΙΑ 5: USER EXPERIENCE (UX)
# ============================================================================

class KeyboardShortcuts:
    """Keyboard shortcut definitions (5.6)."""
    
    SHORTCUTS = {
        "toggle_capture": {"key": "Space", "ctrl": True, "description": "Start/Stop capture"},
        "clear_history": {"key": "Delete", "ctrl": True, "description": "Clear translation history"},
        "copy_translation": {"key": "c", "ctrl": True, "shift": True, "description": "Copy last translation"},
        "toggle_hud": {"key": "h", "ctrl": True, "shift": True, "description": "Show/Hide HUD"},
        "switch_provider": {"key": "p", "ctrl": True, "shift": True, "description": "Switch translation provider"},
    }
    
    @classmethod
    def get_shortcuts_json(cls) -> str:
        """Get shortcuts as JSON for frontend."""
        return json.dumps(cls.SHORTCUTS)


# ============================================================================
# ΚΑΤΗΓΟΡΙΑ 6: DATA & ANALYTICS
# ============================================================================

class ABTestingFramework:
    """A/B testing for translation providers (6.4)."""
    
    def __init__(self):
        self._experiments: Dict[str, Dict[str, Any]] = {}
        self._results: Dict[str, Dict[str, List[float]]] = {}
    
    def create_experiment(self, name: str, variants: List[str], traffic_split: List[float] = None):
        """Create a new A/B test experiment."""
        if traffic_split is None:
            traffic_split = [1.0 / len(variants)] * len(variants)
        
        self._experiments[name] = {
            "variants": variants,
            "traffic_split": traffic_split,
            "created_at": datetime.now().isoformat(),
        }
        self._results[name] = {variant: [] for variant in variants}
    
    def get_variant(self, experiment_name: str, user_id: str = None) -> str:
        """Get variant for user (deterministic based on user_id)."""
        if experiment_name not in self._experiments:
            return None
        
        exp = self._experiments[experiment_name]
        
        # Deterministic assignment based on user_id hash
        if user_id:
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            bucket = (hash_val % 100) / 100.0
        else:
            import random
            bucket = random.random()
        
        cumulative = 0.0
        for variant, split in zip(exp["variants"], exp["traffic_split"]):
            cumulative += split
            if bucket < cumulative:
                return variant
        
        return exp["variants"][-1]
    
    def record_result(self, experiment_name: str, variant: str, metric: float):
        """Record result for variant."""
        if experiment_name in self._results and variant in self._results[experiment_name]:
            self._results[experiment_name][variant].append(metric)
    
    def get_results(self, experiment_name: str) -> Dict[str, Dict[str, float]]:
        """Get experiment results summary."""
        if experiment_name not in self._results:
            return {}
        
        summary = {}
        for variant, metrics in self._results[experiment_name].items():
            if metrics:
                summary[variant] = {
                    "count": len(metrics),
                    "avg": sum(metrics) / len(metrics),
                    "min": min(metrics),
                    "max": max(metrics),
                }
            else:
                summary[variant] = {"count": 0, "avg": 0, "min": 0, "max": 0}
        
        return summary


class UsageAnalytics:
    """Track usage patterns (6.5)."""
    
    def __init__(self):
        self._language_pair_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_latency": 0, "errors": 0}
        )
        self._hourly_usage: Dict[int, int] = defaultdict(int)
        self._daily_usage: Dict[str, int] = defaultdict(int)
    
    def record_usage(self, source_lang: str, target_lang: str, latency_ms: float, success: bool = True):
        """Record a translation usage."""
        pair = f"{source_lang or 'auto'}->{target_lang}"
        self._language_pair_stats[pair]["count"] += 1
        self._language_pair_stats[pair]["total_latency"] += latency_ms
        if not success:
            self._language_pair_stats[pair]["errors"] += 1
        
        # Time-based tracking
        now = datetime.now()
        self._hourly_usage[now.hour] += 1
        self._daily_usage[now.strftime("%Y-%m-%d")] += 1
    
    def get_slowest_pairs(self, limit: int = 5) -> List[Tuple[str, float]]:
        """Get slowest language pairs by average latency."""
        pairs_with_avg = []
        for pair, stats in self._language_pair_stats.items():
            if stats["count"] > 0:
                avg = stats["total_latency"] / stats["count"]
                pairs_with_avg.append((pair, avg))
        
        return sorted(pairs_with_avg, key=lambda x: -x[1])[:limit]
    
    def get_most_used_pairs(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get most used language pairs."""
        pairs_with_count = [(pair, stats["count"]) for pair, stats in self._language_pair_stats.items()]
        return sorted(pairs_with_count, key=lambda x: -x[1])[:limit]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get usage summary."""
        total = sum(s["count"] for s in self._language_pair_stats.values())
        errors = sum(s["errors"] for s in self._language_pair_stats.values())
        
        return {
            "total_translations": total,
            "error_rate": errors / max(1, total),
            "slowest_pairs": self.get_slowest_pairs(),
            "most_used_pairs": self.get_most_used_pairs(),
            "peak_hour": max(self._hourly_usage, key=self._hourly_usage.get) if self._hourly_usage else None,
        }


# ============================================================================
# ΚΑΤΗΓΟΡΙΑ 7: LANGUAGE-SPECIFIC OPTIMIZATIONS
# ============================================================================

class GreekEnhancements:
    """Greek-specific enhancements (7.1)."""
    
    # Greek diacritics normalization
    DIACRITIC_MAP = {
        'ά': 'α', 'έ': 'ε', 'ή': 'η', 'ί': 'ι', 'ό': 'ο', 'ύ': 'υ', 'ώ': 'ω',
        'Ά': 'Α', 'Έ': 'Ε', 'Ή': 'Η', 'Ί': 'Ι', 'Ό': 'Ο', 'Ύ': 'Υ', 'Ώ': 'Ω',
        'ϊ': 'ι', 'ϋ': 'υ', 'ΐ': 'ι', 'ΰ': 'υ',
    }
    
    # Common Greek OCR/transcription errors
    ERROR_CORRECTIONS = {
        "κωι": "και",
        "μέ": "με",
        "νά": "να",
        "θά": "θα",
    }
    
    def normalize_text(self, text: str) -> str:
        """Normalize Greek text."""
        result = text
        for error, correction in self.ERROR_CORRECTIONS.items():
            result = result.replace(error, correction)
        return result
    
    def remove_diacritics(self, text: str) -> str:
        """Remove diacritics for comparison."""
        result = text
        for accented, plain in self.DIACRITIC_MAP.items():
            result = result.replace(accented, plain)
        return result


class RTLSupport:
    """Right-to-left language support (7.3)."""
    
    RTL_LANGUAGES = {"ar", "he", "fa", "ur", "yi"}
    
    # Unicode RTL markers
    RLM = '\u200F'  # Right-to-left mark
    LRM = '\u200E'  # Left-to-right mark
    
    def is_rtl(self, language: str) -> bool:
        """Check if language is RTL."""
        return language.lower() in self.RTL_LANGUAGES
    
    def add_rtl_markers(self, text: str, language: str) -> str:
        """Add RTL markers for proper rendering."""
        if not self.is_rtl(language):
            return text
        return f"{self.RLM}{text}{self.LRM}"
    
    def get_text_direction(self, language: str) -> str:
        """Get CSS direction value."""
        return "rtl" if self.is_rtl(language) else "ltr"


class CJKOptimization:
    """Chinese/Japanese/Korean optimization (7.2)."""
    
    CJK_LANGUAGES = {"zh", "ja", "ko", "zh-CN", "zh-TW"}
    
    def is_cjk(self, language: str) -> bool:
        """Check if language is CJK."""
        return language.lower() in self.CJK_LANGUAGES
    
    def should_split_sentences(self, language: str) -> bool:
        """CJK languages don't use spaces - different sentence splitting."""
        return not self.is_cjk(language)
    
    def estimate_word_count(self, text: str, language: str) -> int:
        """Estimate word count (CJK counts characters differently)."""
        if self.is_cjk(language):
            # CJK: roughly 1.5 characters per "word"
            return len(text) // 2
        return len(text.split())


# ============================================================================
# ΚΑΤΗΓΟΡΙΑ 10: SPECIAL FEATURES
# ============================================================================

class KeyTermsExtractor:
    """Extract and highlight important terms (10.5)."""
    
    def __init__(self):
        self._stop_words = {
            "en": {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                   "have", "has", "had", "do", "does", "did", "will", "would", "could",
                   "should", "may", "might", "must", "shall", "can", "to", "of", "in",
                   "for", "on", "with", "at", "by", "from", "as", "into", "through",
                   "during", "before", "after", "above", "below", "between", "under",
                   "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
                   "not", "only", "own", "same", "than", "too", "very", "just"},
            "el": {"ο", "η", "το", "οι", "τα", "και", "να", "με", "σε", "για", "από",
                   "είναι", "θα", "που", "αυτό", "αυτή", "αυτός"},
        }
    
    def extract_terms(self, text: str, language: str = "en", limit: int = 10) -> List[Dict[str, Any]]:
        """Extract key terms from text."""
        stop_words = self._stop_words.get(language, self._stop_words["en"])
        
        # Tokenize and count
        words = re.findall(r'\b\w+\b', text.lower())
        word_counts = defaultdict(int)
        
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_counts[word] += 1
        
        # Score by frequency and position (earlier = more important)
        scored_terms = []
        for word, count in word_counts.items():
            first_pos = words.index(word) if word in words else len(words)
            position_score = 1 - (first_pos / max(1, len(words)))
            score = count * (1 + position_score)
            scored_terms.append({"term": word, "count": count, "score": round(score, 2)})
        
        # Sort by score and return top terms
        scored_terms.sort(key=lambda x: -x["score"])
        return scored_terms[:limit]


class ConversationMemory:
    """Remember context from previous sessions (10.6)."""
    
    def __init__(self, max_sessions: int = 10, max_items_per_session: int = 50):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._max_sessions = max_sessions
        self._max_items = max_items_per_session
    
    def start_session(self, session_id: str):
        """Start a new conversation session."""
        self._sessions[session_id] = {
            "started_at": datetime.now().isoformat(),
            "messages": [],
            "key_terms": set(),
            "language_pair": None,
        }
        
        # Evict old sessions
        if len(self._sessions) > self._max_sessions:
            oldest = min(self._sessions, key=lambda k: self._sessions[k]["started_at"])
            del self._sessions[oldest]
    
    def add_message(self, session_id: str, original: str, translated: str):
        """Add message to session memory."""
        if session_id not in self._sessions:
            self.start_session(session_id)
        
        session = self._sessions[session_id]
        session["messages"].append({
            "original": original,
            "translated": translated,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Keep only recent messages
        if len(session["messages"]) > self._max_items:
            session["messages"].pop(0)
    
    def get_recent_context(self, session_id: str, count: int = 5) -> List[str]:
        """Get recent messages for context."""
        if session_id not in self._sessions:
            return []
        
        messages = self._sessions[session_id]["messages"][-count:]
        return [m["original"] for m in messages]
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get session summary."""
        if session_id not in self._sessions:
            return {}
        
        session = self._sessions[session_id]
        return {
            "message_count": len(session["messages"]),
            "started_at": session["started_at"],
            "key_terms": list(session.get("key_terms", [])),
        }


class EnhancementService:
    """Main enhancement service combining ALL optimizations."""
    
    def __init__(self):
        # Category 1: STT Quality
        self.filler_remover = FillerWordRemover()
        self.language_detector = LanguageDetector()
        
        # Category 2: Translation Quality
        self.entity_preserver = NamedEntityPreserver()
        self.tech_detector = TechnicalTermDetector()
        self.idiom_detector = IdiomDetector()
        self.formality_detector = FormalityDetector()
        self.semantic_tracker = SemanticCoherenceTracker()
        self.quality_scorer = QualityScorer()
        
        # Category 3: Performance
        self.rate_limiter = DynamicRateLimiter()
        
        # Category 4: Robustness
        self.session_recovery = SessionRecovery()
        self.graceful_degradation = GracefulDegradation()
        self.network_resilience = NetworkResilience()
        
        # Category 6: Analytics
        self.metrics = TranslationMetrics()
        self.ab_testing = ABTestingFramework()
        self.usage_analytics = UsageAnalytics()
        
        # Category 7: Language-specific
        self.greek_enhancer = GreekEnhancements()
        self.rtl_support = RTLSupport()
        self.cjk_optimizer = CJKOptimization()
        
        # Category 10: Special Features
        self.key_terms_extractor = KeyTermsExtractor()
        self.conversation_memory = ConversationMemory()
        
        # Provider health tracking
        self.provider_health: Dict[str, ProviderHealth] = {
            "deepl": ProviderHealth(),
            "openai": ProviderHealth(),
            "google": ProviderHealth(),
        }
        
        # Glossary cache (domain -> terms)
        self._glossary_cache: Dict[str, Dict[str, str]] = {}
        
        # Initialize default A/B test
        self.ab_testing.create_experiment(
            "translation_provider",
            ["deepl", "openai"],
            [0.8, 0.2]  # 80% DeepL, 20% OpenAI
        )
    
    def preprocess_text(
        self,
        text: str,
        source_lang: Optional[str] = None,
        remove_fillers: bool = True,
        preserve_entities: bool = True,
        preserve_technical: bool = True,
        detect_idioms: bool = True,
    ) -> Tuple[str, Dict[str, str]]:
        """Preprocess text before translation.
        
        Returns:
            Tuple of (processed_text, restoration_map)
        """
        if not text:
            return "", {}
        
        result = text
        restoration_map = {}
        
        # Detect language if not provided
        if not source_lang:
            source_lang = self.language_detector.detect(text) or "en"
        
        # Greek-specific normalization
        if source_lang == "el":
            result = self.greek_enhancer.normalize_text(result)
        
        # Remove filler words
        if remove_fillers:
            result = self.filler_remover.remove_fillers(result, source_lang)
        
        # Add idiom context hints
        if detect_idioms:
            result = self.idiom_detector.add_context_hints(result, source_lang)
        
        # Preserve technical terms
        if preserve_technical:
            result, tech_map = self.tech_detector.preserve_technical_terms(result)
            restoration_map.update(tech_map)
        
        # Preserve named entities
        if preserve_entities:
            entities = self.entity_preserver.extract_entities(result)
            result, entity_map = self.entity_preserver.create_placeholders(result, entities)
            restoration_map.update(entity_map)
        
        return result, restoration_map
    
    def postprocess_text(
        self,
        text: str,
        restoration_map: Dict[str, str],
        target_lang: str = "en",
    ) -> str:
        """Postprocess text after translation."""
        if not text:
            return ""
        
        result = text
        
        # Restore all placeholders
        for placeholder, original in restoration_map.items():
            result = result.replace(placeholder, original)
        
        # Add RTL markers if needed
        if self.rtl_support.is_rtl(target_lang):
            result = self.rtl_support.add_rtl_markers(result, target_lang)
        
        # Remove idiom context hints (they shouldn't be in output)
        result = re.sub(r'\s*\[meaning:[^\]]+\]', '', result)
        
        # Clean up any remaining artifacts
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def detect_formality(self, text: str, source_lang: str = "en") -> str:
        """Detect formality level for DeepL."""
        return self.formality_detector.detect_formality(text, source_lang)
    
    def extract_key_terms(self, text: str, language: str = "en") -> List[Dict[str, Any]]:
        """Extract key terms from text."""
        return self.key_terms_extractor.extract_terms(text, language)
    
    def get_fallback_text(self, original: str, error_type: str = "error") -> str:
        """Get graceful degradation text on failure."""
        return self.graceful_degradation.get_fallback_text(original, error_type)
    
    def get_recommended_timeout(self, base_timeout_ms: int) -> int:
        """Get network-aware timeout recommendation."""
        return self.network_resilience.get_recommended_timeout(base_timeout_ms)
    
    def record_network_latency(self, latency_ms: float):
        """Record network latency for resilience calculations."""
        self.network_resilience.record_latency(latency_ms)
    
    def save_session_state(self, session_id: str, state: Dict[str, Any]):
        """Save session state for recovery."""
        self.session_recovery.save_state(session_id, state)
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get session history for recovery."""
        return self.session_recovery.get_history(session_id)
    
    def add_to_conversation(self, session_id: str, original: str, translated: str):
        """Add to conversation memory."""
        self.conversation_memory.add_message(session_id, original, translated)
    
    def get_conversation_context(self, session_id: str, count: int = 5) -> List[str]:
        """Get recent conversation context."""
        return self.conversation_memory.get_recent_context(session_id, count)
    
    def record_term_translation(self, source_term: str, translated_term: str):
        """Record term translation for semantic coherence."""
        self.semantic_tracker.record_translation(source_term, translated_term)
    
    def get_consistent_translation(self, source_term: str) -> Optional[str]:
        """Get consistent translation for term."""
        return self.semantic_tracker.get_consistent_translation(source_term)
    
    def get_best_provider(self, preferred: str = "deepl") -> str:
        """Get best available provider based on health status."""
        # Check if preferred is healthy
        if preferred in self.provider_health:
            if self.provider_health[preferred].is_available():
                return preferred
        
        # Fallback chain: deepl -> openai -> google
        fallback_order = ["deepl", "openai", "google"]
        for provider in fallback_order:
            if provider in self.provider_health:
                if self.provider_health[provider].is_available():
                    return provider
        
        # All unhealthy - return preferred anyway
        return preferred
    
    def record_provider_success(self, provider: str, latency_ms: float):
        """Record successful request to provider."""
        if provider in self.provider_health:
            self.provider_health[provider].record_success(latency_ms)
        self.rate_limiter.record_success()
    
    def record_provider_failure(self, provider: str):
        """Record failed request to provider."""
        if provider in self.provider_health:
            self.provider_health[provider].record_failure()
        self.rate_limiter.record_failure()
    
    def add_glossary_term(self, domain: str, source_term: str, target_term: str):
        """Add a term to domain glossary."""
        if domain not in self._glossary_cache:
            self._glossary_cache[domain] = {}
        self._glossary_cache[domain][source_term.lower()] = target_term
    
    def apply_glossary(self, text: str, domain: str, target_lang: str) -> str:
        """Apply domain-specific glossary to translated text."""
        if domain not in self._glossary_cache:
            return text
        
        result = text
        for source, target in self._glossary_cache[domain].items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(source), re.IGNORECASE)
            result = pattern.sub(target, result)
        
        return result
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return {
            "translation_metrics": self.metrics.get_summary(),
            "provider_health": {
                name: {
                    "status": health.status.value,
                    "consecutive_failures": health.consecutive_failures,
                    "total_requests": health.total_requests,
                    "avg_latency_ms": round(health.avg_latency_ms, 1),
                }
                for name, health in self.provider_health.items()
            },
            "rate_limiter": {
                "current_delay_ms": self.rate_limiter._current_delay_ms,
            },
            "usage_analytics": self.usage_analytics.get_summary(),
            "ab_test_results": self.ab_testing.get_results("translation_provider"),
            "network_quality": self.network_resilience._connection_quality,
            "keyboard_shortcuts": KeyboardShortcuts.get_shortcuts_json(),
        }
    
    def record_usage(self, source_lang: str, target_lang: str, latency_ms: float, success: bool = True):
        """Record usage for analytics."""
        self.usage_analytics.record_usage(source_lang, target_lang, latency_ms, success)
        self.record_network_latency(latency_ms)
    
    def get_ab_test_variant(self, experiment: str, user_id: str = None) -> str:
        """Get A/B test variant for user."""
        return self.ab_testing.get_variant(experiment, user_id)
    
    def record_ab_result(self, experiment: str, variant: str, metric: float):
        """Record A/B test result."""
        self.ab_testing.record_result(experiment, variant, metric)
    
    def is_cjk_language(self, lang: str) -> bool:
        """Check if language is CJK."""
        return self.cjk_optimizer.is_cjk(lang)
    
    def is_rtl_language(self, lang: str) -> bool:
        """Check if language is RTL."""
        return self.rtl_support.is_rtl(lang)
    
    def get_text_direction(self, lang: str) -> str:
        """Get text direction for language."""
        return self.rtl_support.get_text_direction(lang)


# ============================================================================
# REMAINING ENHANCEMENTS (Categories 1, 2, 3, 5, 6, 10)
# ============================================================================

class AccentAwareProcessing:
    """Accent-Aware Processing (1.5)."""
    
    ACCENT_PATTERNS = {
        "british": {
            "patterns": [r'\b(colour|favourite|centre|analyse|behaviour|organise)\b'],
            "region": "UK/AU/NZ",
        },
        "american": {
            "patterns": [r'\b(color|favorite|center|analyze|behavior|organize)\b'],
            "region": "US/CA",
        },
        "indian": {
            "common_phrases": ["kindly", "do the needful", "prepone", "revert back"],
        },
    }
    
    def __init__(self):
        self._detected_accent: Optional[str] = None
        self._accent_scores: Dict[str, int] = defaultdict(int)
    
    def detect_accent(self, text: str) -> str:
        """Detect accent/dialect from text patterns."""
        for accent, info in self.ACCENT_PATTERNS.items():
            patterns = info.get("patterns", [])
            phrases = info.get("common_phrases", [])
            
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    self._accent_scores[accent] += 1
            
            for phrase in phrases:
                if phrase.lower() in text.lower():
                    self._accent_scores[accent] += 1
        
        if self._accent_scores:
            self._detected_accent = max(self._accent_scores, key=self._accent_scores.get)
        
        return self._detected_accent or "neutral"
    
    def get_normalization_map(self) -> Dict[str, str]:
        """Get spelling normalization map."""
        if self._detected_accent == "british":
            return {
                "colour": "color",
                "favourite": "favorite",
                "centre": "center",
            }
        return {}


class PronounResolver:
    """Pronoun Resolution for clarity (2.6)."""
    
    def __init__(self):
        self._entity_stack: List[str] = []
        self._max_stack = 10
    
    def track_entity(self, entity: str):
        """Track a named entity for pronoun resolution."""
        if entity and entity not in self._entity_stack:
            self._entity_stack.append(entity)
            if len(self._entity_stack) > self._max_stack:
                self._entity_stack.pop(0)
    
    def resolve_pronoun(self, pronoun: str, gender_hint: str = None) -> Optional[str]:
        """Try to resolve a pronoun to a tracked entity."""
        if not self._entity_stack:
            return None
        
        pronoun_lower = pronoun.lower()
        
        # Simple heuristic: return most recent entity
        if pronoun_lower in ["he", "she", "it", "they"]:
            return self._entity_stack[-1]
        
        return None
    
    def add_clarity(self, text: str) -> str:
        """Add clarifying references if pronouns are ambiguous."""
        # For translation, this would add context hints
        return text
    
    def clear(self):
        """Clear entity stack."""
        self._entity_stack.clear()


class VADOptimizer:
    """Voice Activity Detection Optimization (1.1)."""
    
    def __init__(self):
        self._speech_start_threshold = 0.6
        self._speech_end_threshold = 0.4
        self._min_speech_duration_ms = 200
        self._max_silence_duration_ms = 1000
        self._current_speech_start: Optional[float] = None
        self._last_speech_end: Optional[float] = None
    
    def process_audio_segment(self, confidence: float, timestamp_ms: float) -> Dict[str, Any]:
        """Process audio segment and detect speech boundaries."""
        result = {
            "is_speech": False,
            "speech_start": False,
            "speech_end": False,
            "should_process": True,
        }
        
        if confidence >= self._speech_start_threshold:
            result["is_speech"] = True
            if self._current_speech_start is None:
                self._current_speech_start = timestamp_ms
                result["speech_start"] = True
            self._last_speech_end = None
        else:
            if self._current_speech_start is not None:
                if self._last_speech_end is None:
                    self._last_speech_end = timestamp_ms
                else:
                    silence_duration = timestamp_ms - self._last_speech_end
                    if silence_duration > self._max_silence_duration_ms:
                        speech_duration = self._last_speech_end - self._current_speech_start
                        if speech_duration >= self._min_speech_duration_ms:
                            result["speech_end"] = True
                        self._current_speech_start = None
                        self._last_speech_end = None
        
        return result
    
    def reset(self):
        """Reset VAD state."""
        self._current_speech_start = None
        self._last_speech_end = None


class NoiseDetector:
    """Real-time Noise Detection (1.4)."""
    
    def __init__(self):
        self._noise_threshold = 0.3
        self._noise_samples: List[float] = []
        self._max_samples = 50
        self._ambient_noise_level = 0.1
    
    def update_noise_level(self, audio_energy: float):
        """Update ambient noise level estimation."""
        self._noise_samples.append(audio_energy)
        if len(self._noise_samples) > self._max_samples:
            self._noise_samples.pop(0)
        
        # Ambient noise is the 25th percentile of recent samples
        if len(self._noise_samples) >= 10:
            sorted_samples = sorted(self._noise_samples)
            idx = len(sorted_samples) // 4
            self._ambient_noise_level = sorted_samples[idx]
    
    def is_noisy(self, confidence: float) -> bool:
        """Check if current segment is too noisy."""
        return confidence < self._noise_threshold
    
    def should_skip_segment(self, confidence: float, text: str) -> Tuple[bool, str]:
        """Determine if segment should be skipped due to noise."""
        if self.is_noisy(confidence):
            return True, "noisy_segment"
        if len(text.strip()) < 2:
            return True, "too_short"
        return False, ""
    
    def get_adaptive_threshold(self) -> float:
        """Get adaptive confidence threshold based on ambient noise."""
        # Higher noise = higher threshold needed
        return min(0.9, self._noise_threshold + self._ambient_noise_level)


class AcousticAdaptation:
    """Acoustic Environment Adaptation (1.3)."""
    
    def __init__(self):
        self._environment_profiles = {
            "quiet": {"confidence_threshold": 0.5, "min_words": 2},
            "normal": {"confidence_threshold": 0.7, "min_words": 3},
            "noisy": {"confidence_threshold": 0.85, "min_words": 4},
            "very_noisy": {"confidence_threshold": 0.95, "min_words": 5},
        }
        self._current_environment = "normal"
        self._confidence_history: List[float] = []
        self._max_history = 100
    
    def record_confidence(self, confidence: float):
        """Record confidence score to adapt environment."""
        self._confidence_history.append(confidence)
        if len(self._confidence_history) > self._max_history:
            self._confidence_history.pop(0)
        
        self._update_environment()
    
    def _update_environment(self):
        """Update environment classification based on recent confidences."""
        if len(self._confidence_history) < 10:
            return
        
        avg_confidence = sum(self._confidence_history[-20:]) / min(20, len(self._confidence_history))
        
        if avg_confidence > 0.9:
            self._current_environment = "quiet"
        elif avg_confidence > 0.75:
            self._current_environment = "normal"
        elif avg_confidence > 0.6:
            self._current_environment = "noisy"
        else:
            self._current_environment = "very_noisy"
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current environment-adapted settings."""
        return self._environment_profiles[self._current_environment]
    
    def get_environment(self) -> str:
        """Get current environment classification."""
        return self._current_environment


class TemporalConsistency:
    """Maintain tense consistency across sentences (2.7)."""
    
    TENSE_PATTERNS = {
        "en": {
            "past": [r'\b(was|were|did|had|went|said|made|came|took|got)\b', r'\b\w+ed\b'],
            "present": [r'\b(is|are|do|does|have|has|go|say|make|come|take|get)\b', r'\b\w+s\b'],
            "future": [r'\b(will|shall|going to|gonna)\b'],
        },
        "el": {
            "past": [r'\b(ήταν|είχε|έκανε|πήγε|είπε)\b'],
            "present": [r'\b(είναι|έχει|κάνει|πάει|λέει)\b'],
            "future": [r'\b(θα)\b'],
        },
    }
    
    def __init__(self):
        self._detected_tense: Optional[str] = None
        self._tense_counts: Dict[str, int] = defaultdict(int)
        self._compiled_patterns: Dict[str, Dict[str, List[re.Pattern]]] = {}
        
        for lang, tenses in self.TENSE_PATTERNS.items():
            self._compiled_patterns[lang] = {}
            for tense, patterns in tenses.items():
                self._compiled_patterns[lang][tense] = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def detect_tense(self, text: str, language: str = "en") -> str:
        """Detect dominant tense in text."""
        if language not in self._compiled_patterns:
            return "unknown"
        
        tense_scores = defaultdict(int)
        
        for tense, patterns in self._compiled_patterns[language].items():
            for pattern in patterns:
                matches = pattern.findall(text)
                tense_scores[tense] += len(matches)
        
        if not tense_scores:
            return "unknown"
        
        dominant = max(tense_scores, key=tense_scores.get)
        self._tense_counts[dominant] += 1
        
        # Update detected tense if clear majority
        total = sum(self._tense_counts.values())
        if total >= 3:
            for tense, count in self._tense_counts.items():
                if count / total > 0.6:
                    self._detected_tense = tense
                    break
        
        return dominant
    
    def get_expected_tense(self) -> Optional[str]:
        """Get expected tense based on conversation history."""
        return self._detected_tense
    
    def is_consistent(self, text: str, language: str = "en") -> bool:
        """Check if text is consistent with detected tense."""
        if not self._detected_tense:
            return True
        
        current_tense = self.detect_tense(text, language)
        return current_tense == self._detected_tense or current_tense == "unknown"
    
    def reset(self):
        """Reset tense tracking."""
        self._detected_tense = None
        self._tense_counts.clear()


class BatchTranslator:
    """Batch Translation for efficiency (3.4)."""
    
    def __init__(self, max_batch_size: int = 5, max_wait_ms: int = 100):
        self._pending_texts: List[Tuple[str, asyncio.Future]] = []
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._batch_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
    
    async def add_text(self, text: str) -> asyncio.Future:
        """Add text to batch and return future for result."""
        future = asyncio.get_event_loop().create_future()
        
        async with self._batch_lock:
            self._pending_texts.append((text, future))
            
            if len(self._pending_texts) >= self._max_batch_size:
                # Batch full - flush immediately
                await self._flush_batch()
            elif len(self._pending_texts) == 1:
                # First item - schedule flush
                self._schedule_flush()
        
        return future
    
    def _schedule_flush(self):
        """Schedule batch flush after max_wait_ms."""
        async def delayed_flush():
            await asyncio.sleep(self._max_wait_ms / 1000.0)
            async with self._batch_lock:
                if self._pending_texts:
                    await self._flush_batch()
        
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        self._flush_task = asyncio.create_task(delayed_flush())
    
    async def _flush_batch(self):
        """Flush pending batch - to be overridden with actual translation."""
        batch = self._pending_texts[:]
        self._pending_texts.clear()
        
        # Return texts as-is (actual translation handled by caller)
        for text, future in batch:
            if not future.done():
                future.set_result(text)
    
    def get_pending_count(self) -> int:
        """Get number of pending texts."""
        return len(self._pending_texts)


class ConfidenceVisualizer:
    """Confidence Visualization (5.1)."""
    
    CONFIDENCE_LEVELS = {
        "high": {"min": 0.9, "color": "#22c55e", "label": "High confidence"},
        "medium": {"min": 0.7, "color": "#eab308", "label": "Medium confidence"},
        "low": {"min": 0.5, "color": "#f97316", "label": "Low confidence"},
        "very_low": {"min": 0.0, "color": "#ef4444", "label": "Very low confidence"},
    }
    
    def get_confidence_info(self, confidence: float) -> Dict[str, Any]:
        """Get visualization info for confidence score."""
        for level, info in self.CONFIDENCE_LEVELS.items():
            if confidence >= info["min"]:
                return {
                    "level": level,
                    "color": info["color"],
                    "label": info["label"],
                    "percentage": round(confidence * 100, 1),
                }
        
        return {
            "level": "unknown",
            "color": "#9ca3af",
            "label": "Unknown",
            "percentage": 0,
        }
    
    def format_for_display(self, text: str, confidence: float) -> str:
        """Format text with confidence indicator."""
        info = self.get_confidence_info(confidence)
        return f"{text} [{info['percentage']}%]"


class AlternativeTranslations:
    """Alternative Translations support (5.2)."""
    
    def __init__(self):
        self._alternatives_cache: Dict[str, List[str]] = {}
        self._max_cache = 100
    
    def store_alternatives(self, text: str, alternatives: List[str]):
        """Store alternative translations."""
        self._alternatives_cache[text.lower()] = alternatives
        
        if len(self._alternatives_cache) > self._max_cache:
            # Remove oldest
            oldest = next(iter(self._alternatives_cache))
            del self._alternatives_cache[oldest]
    
    def get_alternatives(self, text: str) -> List[str]:
        """Get stored alternatives."""
        return self._alternatives_cache.get(text.lower(), [])
    
    async def generate_alternatives(self, text: str, primary: str, target_lang: str) -> List[str]:
        """Generate alternative translations (placeholder - needs API call)."""
        # This would call multiple providers or use temperature sampling
        # For now, return slight variations as placeholder
        alternatives = []
        
        # Simple heuristic alternatives
        if primary:
            # Alternative 1: Different word order (if possible)
            words = primary.split()
            if len(words) > 3:
                alt1 = " ".join(words[1:] + [words[0]])
                alternatives.append(alt1)
        
        return alternatives[:3]


class QualityTrending:
    """Monitor quality improvement/degradation over time (6.2)."""
    
    def __init__(self, window_size: int = 100):
        self._quality_scores: List[Tuple[datetime, float]] = []
        self._window_size = window_size
        self._alerts: List[Dict[str, Any]] = []
    
    def record_quality(self, score: float):
        """Record quality score."""
        self._quality_scores.append((datetime.now(), score))
        
        if len(self._quality_scores) > self._window_size:
            self._quality_scores.pop(0)
        
        self._check_for_degradation()
    
    def _check_for_degradation(self):
        """Check for quality degradation."""
        if len(self._quality_scores) < 20:
            return
        
        recent = [s for _, s in self._quality_scores[-10:]]
        older = [s for _, s in self._quality_scores[-20:-10]]
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        if recent_avg < older_avg * 0.8:  # 20% degradation
            self._alerts.append({
                "type": "quality_degradation",
                "timestamp": datetime.now().isoformat(),
                "message": f"Quality dropped from {older_avg:.2f} to {recent_avg:.2f}",
                "severity": "warning",
            })
    
    def get_trend(self) -> Dict[str, Any]:
        """Get quality trend summary."""
        if len(self._quality_scores) < 10:
            return {"trend": "insufficient_data", "avg": 0}
        
        scores = [s for _, s in self._quality_scores]
        first_half = scores[:len(scores)//2]
        second_half = scores[len(scores)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg * 1.1:
            trend = "improving"
        elif second_avg < first_avg * 0.9:
            trend = "degrading"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "current_avg": round(second_avg, 3),
            "previous_avg": round(first_avg, 3),
            "sample_size": len(self._quality_scores),
        }
    
    def get_alerts(self, clear: bool = True) -> List[Dict[str, Any]]:
        """Get and optionally clear alerts."""
        alerts = self._alerts[:]
        if clear:
            self._alerts.clear()
        return alerts


class EmotionDetector:
    """Detect speaker emotion (10.1)."""
    
    EMOTION_PATTERNS = {
        "en": {
            "happy": [r'\b(happy|joy|excited|great|wonderful|amazing|love|fantastic)\b', r'!{2,}', r'😊|😄|🎉'],
            "sad": [r'\b(sad|sorry|unfortunately|regret|miss|lost|crying)\b', r'😢|😭|💔'],
            "angry": [r'\b(angry|furious|hate|annoyed|frustrated|damn|stupid)\b', r'!{3,}', r'😠|😡|🤬'],
            "surprised": [r'\b(wow|omg|surprised|shocked|unbelievable|incredible)\b', r'\?{2,}', r'😮|😲|🤯'],
            "fearful": [r'\b(scared|afraid|worried|nervous|anxious|terrified)\b', r'😨|😰|😱'],
        },
        "el": {
            "happy": [r'\b(χαρούμενος|χαρά|υπέροχο|φανταστικό|αγαπώ)\b'],
            "sad": [r'\b(λυπημένος|λυπάμαι|δυστυχώς)\b'],
            "angry": [r'\b(θυμωμένος|νευριασμένος|μισώ)\b'],
        },
    }
    
    def __init__(self):
        self._compiled: Dict[str, Dict[str, List[re.Pattern]]] = {}
        for lang, emotions in self.EMOTION_PATTERNS.items():
            self._compiled[lang] = {}
            for emotion, patterns in emotions.items():
                self._compiled[lang][emotion] = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def detect(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Detect emotions in text."""
        if language not in self._compiled:
            language = "en"
        
        emotion_scores = defaultdict(int)
        
        for emotion, patterns in self._compiled.get(language, {}).items():
            for pattern in patterns:
                matches = pattern.findall(text)
                emotion_scores[emotion] += len(matches)
        
        if not emotion_scores:
            return {"primary": "neutral", "confidence": 0.5, "all_emotions": {}}
        
        total = sum(emotion_scores.values())
        primary = max(emotion_scores, key=emotion_scores.get)
        confidence = emotion_scores[primary] / total if total > 0 else 0
        
        return {
            "primary": primary,
            "confidence": round(confidence, 2),
            "all_emotions": dict(emotion_scores),
        }
    
    def get_tone_adjustment(self, emotion: str) -> str:
        """Get tone adjustment hint for translation."""
        adjustments = {
            "happy": "upbeat, enthusiastic",
            "sad": "somber, gentle",
            "angry": "firm, assertive",
            "surprised": "exclamatory",
            "fearful": "cautious, concerned",
            "neutral": "neutral",
        }
        return adjustments.get(emotion, "neutral")


class SummaryGenerator:
    """Auto-generate summary of translated content (10.4)."""
    
    def __init__(self, max_summary_sentences: int = 3):
        self._content_buffer: List[str] = []
        self._max_buffer = 50
        self._max_summary = max_summary_sentences
    
    def add_content(self, text: str):
        """Add content to buffer."""
        self._content_buffer.append(text)
        if len(self._content_buffer) > self._max_buffer:
            self._content_buffer.pop(0)
    
    def generate_summary(self) -> str:
        """Generate summary of buffered content."""
        if not self._content_buffer:
            return ""
        
        # Simple extractive summary: pick sentences with most important words
        all_text = " ".join(self._content_buffer)
        sentences = re.split(r'[.!?]+', all_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= self._max_summary:
            return " ".join(sentences)
        
        # Score sentences by word frequency
        word_freq = defaultdict(int)
        for sentence in sentences:
            for word in sentence.lower().split():
                if len(word) > 3:
                    word_freq[word] += 1
        
        sentence_scores = []
        for sentence in sentences:
            score = sum(word_freq.get(w.lower(), 0) for w in sentence.split())
            sentence_scores.append((score, sentence))
        
        # Get top sentences
        sentence_scores.sort(reverse=True)
        top_sentences = [s for _, s in sentence_scores[:self._max_summary]]
        
        return ". ".join(top_sentences) + "."
    
    def get_key_points(self, limit: int = 5) -> List[str]:
        """Extract key points from content."""
        if not self._content_buffer:
            return []
        
        all_text = " ".join(self._content_buffer)
        
        # Find sentences with key indicators
        key_patterns = [
            r'important[ly]?',
            r'key',
            r'main',
            r'significant',
            r'note that',
            r'remember',
            r'crucial',
            r'essential',
        ]
        
        sentences = re.split(r'[.!?]+', all_text)
        key_points = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            for pattern in key_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    key_points.append(sentence)
                    break
        
        return key_points[:limit]
    
    def clear(self):
        """Clear content buffer."""
        self._content_buffer.clear()


class ErrorReporter:
    """Error reporting and analysis (4.6)."""
    
    def __init__(self, max_errors: int = 100):
        self._errors: List[Dict[str, Any]] = []
        self._max_errors = max_errors
        self._error_patterns: Dict[str, int] = defaultdict(int)
    
    def record_error(self, error_type: str, message: str, context: Dict[str, Any] = None):
        """Record an error."""
        error = {
            "type": error_type,
            "message": message,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
        }
        
        self._errors.append(error)
        self._error_patterns[error_type] += 1
        
        if len(self._errors) > self._max_errors:
            self._errors.pop(0)
    
    def get_suggestions(self) -> List[str]:
        """Get suggestions based on error patterns."""
        suggestions = []
        
        if self._error_patterns.get("timeout", 0) > 5:
            suggestions.append("Consider increasing timeout or switching to faster provider")
        
        if self._error_patterns.get("rate_limit", 0) > 3:
            suggestions.append("Increase debounce delay to reduce API calls")
        
        if self._error_patterns.get("network", 0) > 5:
            suggestions.append("Check network connection; consider offline fallback")
        
        if self._error_patterns.get("low_confidence", 0) > 10:
            suggestions.append("Audio quality may be poor; try noise reduction")
        
        return suggestions
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary."""
        return {
            "total_errors": len(self._errors),
            "error_types": dict(self._error_patterns),
            "recent_errors": self._errors[-5:],
            "suggestions": self.get_suggestions(),
        }
    
    def clear(self):
        """Clear error history."""
        self._errors.clear()
        self._error_patterns.clear()


# Update EnhancementService to include all new components
EnhancementService.vad_optimizer = None
EnhancementService.noise_detector = None
EnhancementService.acoustic_adaptation = None
EnhancementService.temporal_consistency = None
EnhancementService.batch_translator = None
EnhancementService.confidence_visualizer = None
EnhancementService.alternative_translations = None
EnhancementService.quality_trending = None
EnhancementService.emotion_detector = None
EnhancementService.summary_generator = None
EnhancementService.error_reporter = None


def _init_additional_services(self):
    """Initialize additional enhancement services."""
    self.vad_optimizer = VADOptimizer()
    self.noise_detector = NoiseDetector()
    self.acoustic_adaptation = AcousticAdaptation()
    self.temporal_consistency = TemporalConsistency()
    self.batch_translator = BatchTranslator()
    self.confidence_visualizer = ConfidenceVisualizer()
    self.alternative_translations = AlternativeTranslations()
    self.quality_trending = QualityTrending()
    self.emotion_detector = EmotionDetector()
    self.summary_generator = SummaryGenerator()
    self.error_reporter = ErrorReporter()


# Monkey-patch to add initialization
_original_init = EnhancementService.__init__

def _new_init(self):
    _original_init(self)
    _init_additional_services(self)

EnhancementService.__init__ = _new_init


# Global singleton
enhancement_service = EnhancementService()
