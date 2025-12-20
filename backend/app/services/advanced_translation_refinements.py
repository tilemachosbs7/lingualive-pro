"""
Advanced Translation Refinements Module
=====================================

5 Advanced Translation Techniques:
1. Back-Translation Validation
2. Hybrid Translation with Consensus
3. Domain Detection & Adaptation
4. Sentence Complexity Analysis
5. User Correction Learning
"""

import asyncio
import logging
import re
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# 1. BACK-TRANSLATION VALIDATION
# ============================================================================

class BackTranslationValidator:
    """Validate translation quality via back-translation."""
    
    def __init__(self, similarity_threshold: float = 0.75):
        self._threshold = similarity_threshold
        self._validation_cache: Dict[str, Dict[str, Any]] = {}
    
    async def validate(
        self,
        original: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        translate_func,
    ) -> Dict[str, Any]:
        """
        Validate translation by back-translating and comparing.
        
        Returns:
            {
                "is_valid": bool,
                "similarity": float 0-1,
                "back_translation": str,
                "issues": [str],
                "confidence": float,
            }
        """
        try:
            # Back-translate target → source
            back_translation = await translate_func(
                translation, target_lang, source_lang
            )
            
            # Compare original vs back-translation
            similarity = self._calculate_similarity(original, back_translation)
            
            issues = []
            if similarity < 0.6:
                issues.append("High semantic drift detected")
            if similarity < 0.75:
                issues.append("Potential mistranslation")
            
            # Check for lost content
            original_words = set(original.lower().split())
            back_words = set(back_translation.lower().split())
            lost_words = original_words - back_words
            if len(lost_words) > len(original_words) * 0.2:  # >20% lost
                issues.append(f"Lost {len(lost_words)} key words")
            
            is_valid = similarity >= self._threshold
            confidence = min(1.0, similarity * 1.2)  # Boost confidence if valid
            
            return {
                "is_valid": is_valid,
                "similarity": round(similarity, 3),
                "back_translation": back_translation,
                "issues": issues,
                "confidence": round(confidence, 3),
                "original": original,
                "translation": translation,
            }
        
        except Exception as e:
            logger.error(f"Back-translation validation failed: {e}")
            return {
                "is_valid": True,  # Assume valid on error
                "similarity": 0.5,
                "back_translation": None,
                "issues": [f"Validation error: {str(e)}"],
                "confidence": 0.5,
            }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (0-1)."""
        from difflib import SequenceMatcher
        
        # Normalize
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        # Exact match
        if t1 == t2:
            return 1.0
        
        # Word-level similarity
        words1 = set(t1.split())
        words2 = set(t2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        jaccard = intersection / union if union > 0 else 0
        
        # Character-level similarity
        matcher = SequenceMatcher(None, t1, t2)
        char_sim = matcher.ratio()
        
        # Combined (weighted)
        return (jaccard * 0.6 + char_sim * 0.4)


# ============================================================================
# 2. HYBRID TRANSLATION WITH CONSENSUS
# ============================================================================

@dataclass
class TranslationVariant:
    """Single translation variant from a provider."""
    provider: str
    text: str
    confidence: float = 0.7
    latency_ms: float = 0.0
    quality_score: float = 0.0


class HybridTranslator:
    """
    Combine translations from multiple providers using consensus voting.
    
    Strategy:
    - Fast providers (DeepL) for initial translation
    - Quality providers (OpenAI) for refinement
    - Vote on best translation
    """
    
    def __init__(self):
        self._provider_weights = {
            "deepl": 0.6,      # Fast, reliable
            "openai": 0.8,     # Accurate but slower
            "google": 0.5,     # Good but less consistent
        }
        self._cache: Dict[str, List[TranslationVariant]] = {}
    
    async def get_consensus_translation(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        providers: Dict[str, callable],  # {provider_name: translate_func}
        timeout_s: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Get consensus translation from multiple providers.
        
        Returns:
            {
                "translation": best translation,
                "consensus_score": how confident we are,
                "variants": [all variants],
                "method": "exact_match" | "word_vote" | "best_provider",
            }
        """
        variants = []
        
        # Query all providers in parallel (with timeout)
        tasks = [
            self._translate_with_provider(text, source_lang, target_lang, provider, func)
            for provider, func in providers.items()
        ]
        
        try:
            variants = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout_s
            )
            # Filter out exceptions
            variants = [v for v in variants if not isinstance(v, Exception)]
        except asyncio.TimeoutError:
            logger.warning(f"Consensus translation timeout after {timeout_s}s")
        
        if not variants:
            return {
                "translation": None,
                "consensus_score": 0.0,
                "variants": [],
                "method": "failed",
            }
        
        # Analyze variants
        best_translation, consensus_score, method = self._vote_best(variants)
        
        return {
            "translation": best_translation,
            "consensus_score": round(consensus_score, 3),
            "variants": [
                {
                    "provider": v.provider,
                    "text": v.text,
                    "confidence": v.confidence,
                    "latency_ms": v.latency_ms,
                }
                for v in variants
            ],
            "method": method,
        }
    
    async def _translate_with_provider(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        provider: str,
        translate_func,
    ) -> TranslationVariant:
        """Translate with single provider and measure metrics."""
        import time
        start = time.perf_counter()
        
        try:
            translation = await translate_func(text, source_lang, target_lang)
            latency_ms = (time.perf_counter() - start) * 1000
            
            # Weight by provider reliability
            confidence = self._provider_weights.get(provider, 0.6)
            
            return TranslationVariant(
                provider=provider,
                text=translation,
                confidence=confidence,
                latency_ms=latency_ms,
            )
        except Exception as e:
            logger.error(f"Provider {provider} failed: {e}")
            raise
    
    def _vote_best(
        self,
        variants: List[TranslationVariant]
    ) -> Tuple[str, float, str]:
        """Vote for best translation among variants."""
        
        # Check for exact matches
        text_counts = defaultdict(list)
        for v in variants:
            text_counts[v.text].append(v)
        
        # Exact match consensus (2+ providers agree)
        for text, voters in text_counts.items():
            if len(voters) >= 2:
                avg_confidence = sum(v.confidence for v in voters) / len(voters)
                return text, avg_confidence, "exact_match"
        
        # Semantic similarity voting
        best_text = max(variants, key=lambda v: v.confidence).text
        avg_confidence = sum(v.confidence for v in variants) / len(variants)
        
        return best_text, avg_confidence, "best_provider"


# ============================================================================
# 3. DOMAIN DETECTION & ADAPTATION
# ============================================================================

@dataclass
class DomainProfile:
    """Settings for a specific domain."""
    name: str
    keywords: List[str]
    formality: str  # formal, neutral, casual
    terminology_priority: bool  # Preserve technical terms
    glossary_id: Optional[str] = None
    confidence_threshold: float = 0.7


class DomainDetector:
    """
    Detect content domain and apply domain-specific optimization.
    
    Domains: tech, medical, legal, business, casual, academic
    """
    
    DOMAIN_PROFILES = {
        "tech": DomainProfile(
            name="tech",
            keywords=["api", "code", "function", "variable", "database", "server", "client", "python", "javascript", "sql", "framework"],
            formality="neutral",
            terminology_priority=True,
        ),
        "medical": DomainProfile(
            name="medical",
            keywords=["patient", "doctor", "disease", "symptom", "treatment", "medication", "diagnosis", "clinical", "therapy", "health"],
            formality="formal",
            terminology_priority=True,
        ),
        "legal": DomainProfile(
            name="legal",
            keywords=["agreement", "contract", "liability", "defendant", "plaintiff", "judgment", "clause", "herein", "thereof", "legal"],
            formality="formal",
            terminology_priority=True,
        ),
        "business": DomainProfile(
            name="business",
            keywords=["budget", "revenue", "profit", "investment", "stakeholder", "quarterly", "strategy", "market", "client", "contract"],
            formality="formal",
            terminology_priority=False,
        ),
        "academic": DomainProfile(
            name="academic",
            keywords=["research", "study", "analysis", "hypothesis", "conclusion", "methodology", "citation", "abstract", "peer", "publication"],
            formality="formal",
            terminology_priority=True,
        ),
        "casual": DomainProfile(
            name="casual",
            keywords=["hey", "cool", "awesome", "friend", "chat", "fun", "love", "hate", "gonna", "wanna"],
            formality="casual",
            terminology_priority=False,
        ),
    }
    
    def __init__(self):
        self._domain_cache: Dict[str, str] = {}
    
    def detect(self, text: str) -> Tuple[str, float]:
        """
        Detect domain of text.
        
        Returns:
            (domain_name, confidence 0-1)
        """
        text_lower = text.lower()
        scores = defaultdict(int)
        
        # Score each domain
        for domain_name, profile in self.DOMAIN_PROFILES.items():
            for keyword in profile.keywords:
                if keyword in text_lower:
                    scores[domain_name] += 1
        
        if not scores:
            return "casual", 0.3
        
        best_domain = max(scores, key=scores.get)
        max_score = scores[best_domain]
        
        # Confidence based on keyword density
        text_words = len(text_lower.split())
        confidence = min(0.95, (max_score / max(1, text_words / 10)) * 0.5)
        
        return best_domain, round(confidence, 2)
    
    def get_profile(self, domain: str) -> DomainProfile:
        """Get profile for domain."""
        return self.DOMAIN_PROFILES.get(domain, self.DOMAIN_PROFILES["casual"])
    
    def get_formality(self, text: str) -> str:
        """Get recommended formality for text."""
        domain, _ = self.detect(text)
        return self.get_profile(domain).formality


# ============================================================================
# 4. SENTENCE COMPLEXITY ANALYSIS
# ============================================================================

class ComplexityAnalyzer:
    """Analyze and handle complex sentence structures."""
    
    def __init__(self):
        self._complexity_threshold = 25  # Syllable-word ratio
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze sentence complexity."""
        sentences = re.split(r'[.!?]+', text)
        
        scores = []
        for sentence in sentences:
            score = self._calculate_complexity(sentence)
            scores.append(score)
        
        avg_complexity = sum(scores) / len(scores) if scores else 0
        
        return {
            "average_complexity": round(avg_complexity, 2),
            "is_complex": avg_complexity > self._complexity_threshold,
            "max_complexity": max(scores) if scores else 0,
            "sentence_count": len(sentences),
        }
    
    def _calculate_complexity(self, sentence: str) -> float:
        """Calculate complexity score for sentence."""
        sentence = sentence.strip()
        if not sentence:
            return 0.0
        
        words = sentence.split()
        word_count = len(words)
        
        if word_count < 2:
            return 0.0
        
        # Factors
        avg_word_length = sum(len(w) for w in words) / word_count
        clause_count = len(re.findall(r'\b(and|or|but|because|if|when|while)\b', sentence, re.IGNORECASE)) + 1
        nested_depth = sentence.count('(') + sentence.count('[')
        
        # Complexity = word_length * clauses * nesting
        complexity = avg_word_length * clause_count * (1 + nested_depth * 0.2)
        
        return complexity
    
    async def split_complex_sentence(
        self,
        text: str,
        translate_func,
        source_lang: str,
        target_lang: str,
    ) -> str:
        """
        Split complex sentence into clauses and translate separately.
        """
        if not self.analyze(text)["is_complex"]:
            return await translate_func(text, source_lang, target_lang)
        
        # Split by clauses
        clauses = self._extract_clauses(text)
        
        # Translate each
        translations = []
        for clause in clauses:
            try:
                translated = await translate_func(clause, source_lang, target_lang)
                translations.append(translated)
            except Exception as e:
                logger.warning(f"Clause translation failed: {e}")
                translations.append(clause)
        
        # Rejoin with appropriate connectors
        result = self._rejoin_clauses(translations, text)
        return result
    
    def _extract_clauses(self, text: str) -> List[str]:
        """Extract clauses from complex sentence."""
        # Split by common connectors
        parts = re.split(r'\b(and|or|but|because|if|when|while|although|though)\b', text, flags=re.IGNORECASE)
        
        # Pair connectors with following text
        clauses = []
        for i in range(0, len(parts), 2):
            clause = parts[i]
            if i + 1 < len(parts):
                clause += " " + parts[i + 1]
            
            clause = clause.strip()
            if clause:
                clauses.append(clause)
        
        return clauses
    
    def _rejoin_clauses(self, translations: List[str], original: str) -> str:
        """Rejoin translated clauses intelligently."""
        # Extract connectors from original
        connectors = re.findall(r'\b(and|or|but|because|if|when|while|although|though)\b', original, re.IGNORECASE)
        
        # Rebuild with connectors
        result = translations[0] if translations else ""
        for i, connector in enumerate(connectors):
            if i + 1 < len(translations):
                result += f" {connector} {translations[i + 1]}"
        
        return result


# ============================================================================
# 5. USER CORRECTION LEARNING
# ============================================================================

@dataclass
class CorrectionRecord:
    """User correction record."""
    original: str
    auto_translation: str
    user_correction: str
    source_lang: str
    target_lang: str
    timestamp: datetime = field(default_factory=datetime.now)
    confidence_improvement: float = 0.0  # How much better user's version is


class CorrectionLearner:
    """Learn from user corrections to improve future translations."""
    
    def __init__(self, max_corrections: int = 1000):
        self._corrections: List[CorrectionRecord] = []
        self._max_corrections = max_corrections
        self._stats = {
            "total_corrections": 0,
            "frequent_patterns": defaultdict(int),
            "improvement_avg": 0.0,
        }
    
    def record_correction(
        self,
        original: str,
        auto_translation: str,
        user_correction: str,
        source_lang: str,
        target_lang: str,
    ):
        """Record user correction."""
        # Calculate improvement
        auto_length = len(auto_translation)
        corrected_length = len(user_correction)
        improvement = abs(corrected_length - auto_length) / max(1, auto_length)
        
        record = CorrectionRecord(
            original=original,
            auto_translation=auto_translation,
            user_correction=user_correction,
            source_lang=source_lang,
            target_lang=target_lang,
            confidence_improvement=improvement,
        )
        
        self._corrections.append(record)
        self._stats["total_corrections"] += 1
        
        # Extract patterns
        self._extract_patterns(record)
        
        # Trim if over limit
        if len(self._corrections) > self._max_corrections:
            self._corrections.pop(0)
    
    def _extract_patterns(self, record: CorrectionRecord):
        """Extract correction patterns from record."""
        # Example patterns
        if "was" in record.auto_translation and "were" in record.user_correction:
            self._stats["frequent_patterns"]["was_were_correction"] += 1
        
        if "the " in record.auto_translation and "the " not in record.user_correction:
            self._stats["frequent_patterns"]["article_removal"] += 1
    
    def get_similar_corrections(
        self,
        text: str,
        target_lang: str,
        top_n: int = 3,
    ) -> List[CorrectionRecord]:
        """
        Get similar past corrections to guide translation.
        """
        from difflib import SequenceMatcher
        
        similar = []
        for record in self._corrections:
            if record.target_lang != target_lang:
                continue
            
            ratio = SequenceMatcher(None, text.lower(), record.original.lower()).ratio()
            if ratio > 0.6:  # Similar enough
                similar.append((ratio, record))
        
        # Return top N by similarity
        similar.sort(reverse=True)
        return [r for _, r in similar[:top_n]]
    
    def get_suggestions(self, text: str, target_lang: str) -> List[str]:
        """Get translation suggestions based on past corrections."""
        similar = self.get_similar_corrections(text, target_lang)
        
        suggestions = []
        for record in similar:
            suggestions.append(record.user_correction)
        
        return suggestions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get correction learning statistics."""
        avg_improvement = (
            sum(c.confidence_improvement for c in self._corrections) / len(self._corrections)
            if self._corrections else 0.0
        )
        
        return {
            "total_corrections": self._stats["total_corrections"],
            "avg_improvement": round(avg_improvement, 3),
            "frequent_patterns": dict(self._stats["frequent_patterns"]),
            "corrections_stored": len(self._corrections),
        }


# ============================================================================
# MASTER CONTROLLER
# ============================================================================

class AdvancedRefinementController:
    """Central controller for all advanced translation refinements."""
    
    def __init__(self):
        self.back_translator = BackTranslationValidator()
        self.hybrid_translator = HybridTranslator()
        self.domain_detector = DomainDetector()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.correction_learner = CorrectionLearner()
        
        self._stats = {
            "validations_run": 0,
            "hybrid_uses": 0,
            "domain_detections": 0,
            "complexity_splits": 0,
            "corrections_applied": 0,
        }
    
    async def refine_translation(
        self,
        original: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        translate_func,
        enable_features: Dict[str, bool],
    ) -> Dict[str, Any]:
        """
        Run full refinement pipeline on translation.
        
        Returns refined translation with metadata.
        """
        result = {
            "original": original,
            "translation": translation,
            "refinements": {},
            "quality_score": 0.7,
        }
        
        # 1. Domain Detection
        if enable_features.get("domain_detection", True):
            domain, confidence = self.domain_detector.detect(original)
            result["refinements"]["domain"] = {
                "name": domain,
                "confidence": confidence,
                "formality": self.domain_detector.get_profile(domain).formality,
            }
            self._stats["domain_detections"] += 1
        
        # 2. Complexity Analysis
        if enable_features.get("complexity_analysis", True):
            complexity = self.complexity_analyzer.analyze(original)
            result["refinements"]["complexity"] = complexity
            
            if complexity["is_complex"]:
                try:
                    refined = await self.complexity_analyzer.split_complex_sentence(
                        original, translate_func, source_lang, target_lang
                    )
                    if refined and refined != translation:
                        translation = refined
                        self._stats["complexity_splits"] += 1
                except Exception as e:
                    logger.warning(f"Complexity splitting failed: {e}")
        
        # 3. Back-Translation Validation
        if enable_features.get("back_translation", True):
            validation = await self.back_translator.validate(
                original, translation, source_lang, target_lang, translate_func
            )
            result["refinements"]["validation"] = {
                "is_valid": validation["is_valid"],
                "similarity": validation["similarity"],
                "issues": validation["issues"],
            }
            result["quality_score"] = validation["confidence"]
            self._stats["validations_run"] += 1
        
        # 4. User Corrections
        if enable_features.get("user_corrections", True):
            suggestions = self.correction_learner.get_suggestions(original, target_lang)
            if suggestions:
                result["refinements"]["suggestions"] = suggestions
                self._stats["corrections_applied"] += 1
        
        result["translation"] = translation
        return result
    
    def record_user_correction(
        self,
        original: str,
        auto_translation: str,
        user_correction: str,
        source_lang: str,
        target_lang: str,
    ):
        """Record user correction for learning."""
        self.correction_learner.record_correction(
            original, auto_translation, user_correction, source_lang, target_lang
        )
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get stats from all subsystems."""
        return {
            "controller_stats": self._stats,
            "corrections": self.correction_learner.get_stats(),
        }


# Global singleton
advanced_refinement_controller = AdvancedRefinementController()
