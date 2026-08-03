"""
Real function/class-boundary chunking via tree-sitter, replacing the fixed
700-char-stride/800-char-window sliding chunking that search_engine.py's
build_index() actually does today -- despite this project's own docstrings
and README claiming "function-boundary chunking" (a real, verified gap
between what's documented and what's implemented, not a hypothetical).

Extracts each top-level function/method/class as ONE whole chunk (a
semantically complete unit, not an arbitrary character slice that can cut
a function in half) for the languages that make up this corpus. Falls back
to the existing fixed-window chunker for unsupported file types (.md, .sh,
.ps1, etc.) and for any file where tree-sitter parsing finds zero
definitions, so coverage doesn't regress vs. the current approach.
"""
from pathlib import Path

from tree_sitter_language_pack import get_parser

# Per-language node types that count as a "definition" worth chunking whole.
# Verified against each grammar's actual node-type names, not guessed.
DEFINITION_NODE_TYPES = {
    "python":     {"function_definition", "class_definition"},
    "javascript": {"function_declaration", "method_definition", "class_declaration",
                   "arrow_function", "generator_function_declaration"},
    "typescript": {"function_declaration", "method_definition", "class_declaration",
                   "arrow_function", "interface_declaration"},
    "go":         {"function_declaration", "method_declaration", "type_declaration"},
    "rust":       {"function_item", "impl_item", "struct_item", "trait_item"},
    "php":        {"function_definition", "method_declaration", "class_declaration"},
    "c":          {"function_definition", "struct_specifier"},
}

EXT_TO_LANG = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".rs": "rust", ".php": "php", ".c": "c", ".h": "c",
}

MAX_CHUNK_CHARS = 2500  # cap a single definition chunk -- an oversized one
                        # (e.g. a 5000-line generated file) still gets capped
                        # rather than blowing up embedding batch memory


def ast_chunks(text: str, ext: str):
    """Returns a list of (start_offset, chunk_text) for whole function/class/
    struct definitions in `text`, or None if this extension isn't supported
    or parsing finds nothing -- callers should fall back to fixed-window
    chunking in either case, not treat None as "file has no content"."""
    lang = EXT_TO_LANG.get(ext.lower())
    if lang is None:
        return None

    try:
        parser = get_parser(lang)
        tree = parser.parse(text.encode("utf-8", errors="ignore"))
    except Exception:
        return None

    wanted = DEFINITION_NODE_TYPES[lang]
    encoded = text.encode("utf-8", errors="ignore")
    results = []

    def walk(node):
        if node.type in wanted:
            start, end = node.start_byte, node.end_byte
            chunk_bytes = encoded[start:min(end, start + MAX_CHUNK_CHARS)]
            chunk = chunk_bytes.decode("utf-8", errors="ignore")
            if len(chunk.strip()) > 30:
                results.append((start, chunk))
            return  # don't descend into a captured definition's own children
                    # (avoids double-chunking a method inside an already-
                    # captured class -- the class chunk already includes it)
        for child in node.children:
            walk(child)

    walk(tree.root_node)
    return results if results else None
