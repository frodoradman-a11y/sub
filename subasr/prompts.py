"""Prompt construction for the LLM modes.

The audio embeddings are spliced into the prompt where ``{audio}`` stands. With
``prompt_template: chat`` the tokenizer's own chat template is used (Gemma 4), so
the fine-tuned model keeps the conversation format it was instruction-tuned on.
"""

from __future__ import annotations

from subasr.config import LLMConfig

AUDIO_MARK = "{audio}"
_CHAT_MARK = "AUDIO"  # private-use marker, never produced by real text

LANGUAGE_NAMES = {"fa": "Persian", "en": "English", "ar": "Arabic", "tr": "Turkish", "fr": "French", "de": "German"}


class PromptBuilder:
    def __init__(self, cfg: LLMConfig, tokenizer):
        self.cfg = cfg
        self.tok = tokenizer

    def instruction(self, language: str, context: str | None = None, glossary: list[str] | None = None) -> str:
        parts = [self.cfg.instruction.format(language=LANGUAGE_NAMES.get(language, language))]
        if context:
            parts.append(self.cfg.context_instruction.format(context=context))
        if glossary:
            parts.append(self.cfg.glossary_instruction.format(glossary=", ".join(glossary)))
        return "\n".join(p for p in parts if p)

    def build(
        self, language: str, context: str | None = None, glossary: list[str] | None = None
    ) -> tuple[list[int], list[int]]:
        """Token ids before and after the audio embeddings."""
        instruction = self.instruction(language, context, glossary)
        if self.cfg.prompt_template == "chat":
            text = self.tok.apply_chat_template([{"role": "user", "content": f"{instruction}\n{_CHAT_MARK}"}])
            before, _, after = text.partition(_CHAT_MARK)
        else:
            if AUDIO_MARK not in self.cfg.prompt_template:
                raise ValueError("prompt_template must contain {audio}")
            before, _, after = self.cfg.prompt_template.partition(AUDIO_MARK)
            before = before.replace("{instruction}", instruction)
            after = after.replace("{instruction}", instruction)
        return self.tok.encode(before), self.tok.encode(after)
