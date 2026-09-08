"""Retrievers over the consolidated knowledge base.

All retrievers share one interface -- fit(texts) then retrieve(query, k) --
so they can be swapped at runtime from the UI.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

_PUNCT = re.compile(r"[^a-z0-9\s]")


_VOWELS = "aeiou"


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _contains_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def stem(word: str) -> str:
    """Strip plural and tense endings (Porter step 1).

    Helpdesk questions and ticket titles rarely agree on inflection: users type
    'my wifi keeps dropping', the ticket says 'Wi-Fi drops'. Without stemming
    those share no token and the match is lost. Full Porter is overkill here;
    step 1 covers -s, -ed and -ing, which is where the disagreement lives.
    """
    if len(word) <= 3:
        return word

    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-3] + "i"
    elif word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]

    stripped = False
    if word.endswith("eed"):
        if _contains_vowel(word[:-3]):
            word = word[:-1]
    elif word.endswith("ed") and _contains_vowel(word[:-2]):
        word, stripped = word[:-2], True
    elif word.endswith("ing") and _contains_vowel(word[:-3]):
        word, stripped = word[:-3], True

    if stripped:
        if word.endswith(("at", "bl", "iz")):
            word += "e"
        elif (
            len(word) > 1
            and word[-1] == word[-2]
            and word[-1] not in "lsz"
            and _is_consonant(word, len(word) - 1)
        ):
            word = word[:-1]                      # dropp -> drop
        elif (
            len(word) >= 3
            and _is_consonant(word, len(word) - 1)
            and not _is_consonant(word, len(word) - 2)
            and _is_consonant(word, len(word) - 3)
            and word[-1] not in "wxy"
        ):
            word += "e"                            # wak -> wake
    return word


def normalise(text: str) -> str:
    """Fold the spellings helpdesk users actually type into one form.

    'Wi-Fi', 'wi fi' and 'wifi' must land on the same tokens, otherwise a
    hyphen decides whether a ticket is found. Hyphens and apostrophes are
    removed rather than replaced with a space ('wi-fi' -> 'wifi'), and
    remaining punctuation becomes whitespace.
    """
    text = str(text).lower().replace("-", "").replace("'", "").replace("\u2019", "")
    return " ".join(stem(token) for token in _PUNCT.sub(" ", text).split())


@dataclass
class Hit:
    """One retrieved knowledge base entry."""

    index: int
    score: float

    def as_dict(self) -> dict:
        return {"index": self.index, "score": self.score}


def _top_k(scores: np.ndarray, k: int) -> list[Hit]:
    """Top-k by descending score, without a full sort."""
    k = max(1, min(int(k), scores.shape[0]))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [Hit(index=int(i), score=float(scores[i])) for i in idx]


class TfidfRetriever:
    """Lexical retrieval over word and character n-grams.

    Character n-grams are included because helpdesk questions are typed in a
    hurry: 'wifi', 'wi-fi' and 'wifii' share no word token but plenty of
    character trigrams.
    """

    name = "tfidf"

    def __init__(self, max_features: int = 50_000):
        self.max_features = max_features
        self.vectorizer_ = None
        self.index_ = None

    def fit(self, texts: list[str]) -> TfidfRetriever:
        self.texts_ = [normalise(t) for t in texts]
        self.vectorizer_ = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        max_features=self.max_features,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        sublinear_tf=True,
                        min_df=1,
                        max_features=self.max_features,
                    ),
                ),
            ],
            # Words carry the meaning; character n-grams are a spelling safety
            # net and would otherwise swamp the score with shared substrings.
            transformer_weights={"word": 1.0, "char": 0.35},
        )
        self.index_ = self.vectorizer_.fit_transform(self.texts_)
        return self

    def retrieve(self, query: str, k: int = 4) -> list[Hit]:
        if self.index_ is None:
            raise RuntimeError("Retriever is not fitted.")
        query_vec = self.vectorizer_.transform([normalise(query)])
        scores = cosine_similarity(query_vec, self.index_).ravel()
        return _top_k(scores, k)


class EmbeddingRetriever:
    """Dense retrieval with a sentence-transformer, cached to disk.

    The model and its encoded index are loaded lazily so the app still starts
    (and still answers, lexically) on a machine without torch installed.
    """

    name = "embedding"

    def __init__(self, model_name: str, cache_dir: Path | None = None):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model_ = None
        self.index_ = None

    def _load_model(self):
        if self.model_ is None:
            from sentence_transformers import SentenceTransformer

            self.model_ = SentenceTransformer(self.model_name)
        return self.model_

    def fit(self, texts: list[str], fingerprint: str = "") -> EmbeddingRetriever:
        self.texts_ = [str(t) for t in texts]  # embeddings handle raw text best
        cache_file = None
        if self.cache_dir and fingerprint:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            slug = self.model_name.replace("/", "_")
            cache_file = self.cache_dir / f"emb_{slug}_{fingerprint}.npy"
            if cache_file.exists():
                self.index_ = np.load(cache_file)
                return self

        self.index_ = self._load_model().encode(
            self.texts_,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if cache_file is not None:
            np.save(cache_file, self.index_)
        return self

    def retrieve(self, query: str, k: int = 4) -> list[Hit]:
        if self.index_ is None:
            raise RuntimeError("Retriever is not fitted.")
        query_vec = self._load_model().encode(
            [str(query)], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        return _top_k(self.index_ @ query_vec, k)


class HybridRetriever:
    """Reciprocal rank fusion of a lexical and a dense retriever.

    Lexical retrieval wins on exact strings (error codes, model numbers);
    dense retrieval wins on paraphrase. Fusing ranks rather than scores
    avoids comparing two incompatible similarity scales.
    """

    name = "hybrid"

    def __init__(self, lexical, dense, rrf_k: int = 60):
        self.lexical = lexical
        self.dense = dense
        self.rrf_k = rrf_k

    def fit(self, texts, fingerprint: str = "") -> HybridRetriever:
        self.lexical.fit(texts)
        self.dense.fit(texts, fingerprint=fingerprint)
        return self

    def retrieve(self, query: str, k: int = 4) -> list[Hit]:
        pool = max(k * 3, 10)
        fused: dict[int, float] = {}
        raw: dict[int, float] = {}
        for retriever in (self.lexical, self.dense):
            for rank, hit in enumerate(retriever.retrieve(query, k=pool)):
                fused[hit.index] = fused.get(hit.index, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                raw[hit.index] = max(raw.get(hit.index, 0.0), hit.score)

        best = sorted(fused, key=lambda i: -fused[i])[:k]
        # Report the underlying similarity, not the fusion weight: the score is
        # shown to users and compared against a threshold, so it must stay
        # interpretable as "how similar is this really".
        return [Hit(index=int(i), score=float(raw[i])) for i in best]


def build_retriever(kind: str, settings, texts: list[str], fingerprint: str = ""):
    """Construct and fit a retriever, degrading gracefully to lexical only.

    Returns
    -------
    (retriever, note) : tuple
        `note` is empty on success, or explains the fallback.
    """
    kind = (kind or "hybrid").lower()

    if kind == "tfidf":
        return TfidfRetriever().fit(texts), ""

    dense = EmbeddingRetriever(settings.embedding_model, settings.cache_dir)
    try:
        if kind == "embedding":
            return dense.fit(texts, fingerprint=fingerprint), ""
        hybrid = HybridRetriever(TfidfRetriever(), dense)
        return hybrid.fit(texts, fingerprint=fingerprint), ""
    except Exception as exc:
        note = (
            f"Embeddings unavailable ({type(exc).__name__}), using keyword search only. "
            "Install sentence-transformers for semantic matching."
        )
        return TfidfRetriever().fit(texts), note
