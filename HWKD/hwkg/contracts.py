"""Repeat explicit task boundaries during hardware edits; advice, not enforcement."""
import re


def scope_reminder(task, provider_prompt=None):
    parts = [str(task or '')]
    if isinstance(provider_prompt, str):
        parts.append(provider_prompt)
    elif isinstance(provider_prompt, dict):
        parts.extend(v for k, v in provider_prompt.items()
                     if k in ('system', 'user', 'assistant') and isinstance(v, str))
    text = '\n'.join(parts)
    offline = bool(re.search(r'no network downloads|offline-only|no internet downloads', text, re.I))
    separate = ('train.csv' in text and 'validation.csv' in text and bool(re.search(
        r'never concatenate validation|validation.{0,80}(?:eval-only|not.{0,20}train)|fit.{0,25}train.{0,15}only', text, re.I)))
    if offline and separate:
        return 'Keep offline flags/local weights; fit train.csv only.'
    if offline:
        return 'Keep offline flags; use local weights only.'
    if separate:
        return 'Fit train.csv only; validation is eval-only.'
    return ''
