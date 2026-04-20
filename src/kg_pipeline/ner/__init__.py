from .crf_baseline import CRFNamedEntityRecognizer, evaluate_ner

__all__ = [
    "CRFNamedEntityRecognizer",
    "evaluate_ner",
]

try:
    from .bert_crf import BERTCRFConfig, BERTCRFNamedEntityRecognizer

    __all__.extend(
        [
            "BERTCRFConfig",
            "BERTCRFNamedEntityRecognizer",
        ]
    )
except Exception:
    # Keep CRF baseline usable even when torch stack is unavailable.
    BERTCRFConfig = None  # type: ignore[assignment]
    BERTCRFNamedEntityRecognizer = None  # type: ignore[assignment]
