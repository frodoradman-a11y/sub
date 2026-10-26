"""SubASR: subtitle-oriented speech recognition.

Architecture (see docs/architecture-proposal.md):
    audio -> encoder (Gemma 4 audio tower or a small Conformer)
          -> CTC / TDT / audio-event heads            ("fast" mode)
          -> adapter -> LLM (Gemma 4 text model)      ("edit" and "quality" modes)
"""

__version__ = "0.1.0"
