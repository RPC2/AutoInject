"""
Token validation utilities for adversarial suffix generation.

This module provides shared logic for filtering and validating tokens
used in adversarial suffix generation across different learners.
"""

from typing import Any

# Characters that cause YAML parsing errors or break Python's str.format()
# These are removed/filtered because adversarial suffixes are:
# 1. Injected into AgentDojo's YAML environment
# 2. Processed through Python's str.format() which requires escaping curly braces
YAML_PROBLEMATIC_CHARS = [
    "\\",  # Causes YAML escape sequence errors like \[, \p, etc.
    '"',  # Can break YAML string boundaries
    "'",  # Can break YAML string boundaries
    "`",  # Can break string parsing
    "{",  # Breaks Python's str.format()
    "}",  # Breaks Python's str.format()
    ":",  # Can start a new YAML key:value pair
    "#",  # Starts YAML comments
    "|",  # Starts YAML literal blocks
    ">",  # Starts YAML folded blocks
    "&",  # YAML anchors
    "*",  # YAML aliases
    "[",  # Can start YAML lists/arrays
    "]",  # Can end YAML lists/arrays
    "?",  # YAML complex key indicator
    "-",  # YAML list item indicator (added for consistency with clean_generated_suffix)
]


def clean_generated_suffix(suffix: str) -> str:
    """Clean generated suffix to ensure it doesn't break format string and YAML parsing.

    The suffix is injected into AgentDojo's environment YAML using Python's str.format(),
    so we need to escape curly braces and ensure YAML compatibility.

    This function removes specific characters that cause YAML parsing errors
    while maintaining maximum expressiveness for adversarial suffixes.

    Uses the shared YAML_PROBLEMATIC_CHARS constant to ensure consistency with
    token filtering in other learners.

    Args:
        suffix: The raw generated suffix string to clean

    Returns:
        Cleaned suffix string that's safe for YAML and format strings

    Examples:
        >>> clean_generated_suffix("test{invalid}suffix")
        'testinvalidsuffix'
        >>> clean_generated_suffix("  hello\\nworld  ")
        'hello world'
        >>> clean_generated_suffix("")
        'Please follow my instructions exactly.'
    """
    suffix = suffix.strip()

    # Remove YAML-problematic characters (shared with token validation)
    for char in YAML_PROBLEMATIC_CHARS:
        suffix = suffix.replace(char, "")

    # Remove newlines and control characters
    suffix = suffix.replace("\n", " ")
    suffix = suffix.replace("\r", " ")
    suffix = suffix.replace("\t", " ")

    # Remove non-printable characters (keep everything from space to tilde: 32-126)
    suffix = "".join(char if 32 <= ord(char) < 127 else " " for char in suffix)

    # Collapse multiple spaces
    suffix = " ".join(suffix.split())

    # Ensure non-empty output
    if not suffix:
        suffix = "Please follow my instructions exactly."

    return suffix


def is_valid_adversarial_token(decoded: str, tokenizer: Any) -> bool:
    """Check if a decoded token is valid for adversarial suffix generation.

    This function filters tokens based on multiple criteria to ensure
    the generated suffix:
    - Doesn't contain special tokens (pad, eos, bos, unk)
    - Doesn't contain Chinese/CJK characters
    - Doesn't contain control characters
    - Doesn't contain characters that break YAML parsing
    - Has printable content

    Args:
        decoded: The decoded token string to validate
        tokenizer: The tokenizer instance (used to check special tokens)

    Returns:
        True if the token is valid for adversarial suffix generation, False otherwise

    Examples:
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> is_valid_adversarial_token("hello", tokenizer)
        True
        >>> is_valid_adversarial_token("", tokenizer)
        False
        >>> is_valid_adversarial_token("{", tokenizer)
        False
        >>> is_valid_adversarial_token("你好", tokenizer)  # Chinese
        False
    """
    # Must have content
    if not decoded or len(decoded.strip()) == 0:
        return False

    # Exclude special tokens
    special_tokens = [
        tokenizer.pad_token,
        tokenizer.eos_token,
        tokenizer.bos_token,
        tokenizer.unk_token,
    ]
    if decoded in special_tokens:
        return False

    # Exclude tokens that are just whitespace
    if decoded.strip() == "":
        return False

    # Exclude tokens containing YAML-problematic characters
    # These would be stripped by clean_generated_suffix anyway
    if any(char in decoded for char in YAML_PROBLEMATIC_CHARS):
        return False

    # Exclude Chinese/CJK characters (Unicode ranges)
    for char in decoded:
        if "\u4e00" <= char <= "\u9fff":  # CJK unified ideographs
            return False
        if "\u3400" <= char <= "\u4dbf":  # CJK extension A
            return False
        if "\uf900" <= char <= "\ufaff":  # CJK compatibility
            return False

    # Exclude pure control characters
    if all(ord(c) < 32 for c in decoded):
        return False

    # Only allow printable ASCII characters (space to tilde: 32-126)
    if not all(32 <= ord(c) < 127 for c in decoded):
        return False

    return True
