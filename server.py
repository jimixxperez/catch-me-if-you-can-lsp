#!/usr/bin/env python3
import ast
import json
import os
import logging

from pygls.server import LanguageServer
from lsprotocol.types import (
    Diagnostic,
    DiagnosticSeverity,
    DidChangeTextDocumentParams,
    DidOpenTextDocumentParams,
    Position,
    Range,
)

# Configure logging (sent to stdout)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Capture all messages

formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler("lsp_server.log", mode="w")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Path to the exception mapping file generated earlier.
EXCEPTION_MAPPING_FILE = "exceptions_map.json"

# Load exception mapping.
if os.path.exists(EXCEPTION_MAPPING_FILE):
    try:
        with open(EXCEPTION_MAPPING_FILE, "r", encoding="utf-8") as f:
            exception_mapping = json.load(f)
        logging.info("Loaded exception mapping from %s", EXCEPTION_MAPPING_FILE)
    except Exception as e:
        logging.error("Error loading exception mapping: %s", e)
        exception_mapping = {}
else:
    logging.warning(
        "Exception mapping file %s not found. No diagnostics will be produced.",
        EXCEPTION_MAPPING_FILE,
    )
    exception_mapping = {}

# Initialize the language server.
ls = LanguageServer("exception-warning-lsp", "v0.1")

# --- AST Visitor for detecting unguarded function calls ---


class ExceptionCallVisitor(ast.NodeVisitor):
    """
    Walks the AST to find function calls that are not inside a try block.
    For qualified calls (e.g. `json.loads`), it consults the loaded exception mapping.
    """

    def __init__(self, exception_mapping):
        self.exception_mapping = exception_mapping
        self.diagnostics = []
        self.try_depth = 0  # Tracks nesting within try statements

    def visit_Try(self, node):
        self.try_depth += 1
        self.generic_visit(node)
        self.try_depth -= 1

    def visit_Call(self, node):
        # Only consider calls not inside a try block.
        if self.try_depth == 0:
            module_name = None
            function_name = None
            # Check if call is of the form: module.function(...)
            if isinstance(node.func, ast.Attribute):
                # e.g. json.loads(...)
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id != "self"
                ):
                    module_name = node.func.value.id
                    function_name = node.func.attr

            if module_name and function_name:
                # Look up the module in the exception mapping.
                mod_map = self.exception_mapping.get(module_name)
                if mod_map:
                    functions = mod_map.get("functions", {})
                    # Check if the function is recorded in the mapping.
                    func_info = functions.get(function_name)
                    if func_info and func_info.get("raises"):
                        ex_list = func_info["raises"]
                        diag_msg = (
                            f"Call to {module_name}.{function_name} may raise exceptions: "
                            f"{', '.join(ex_list)}"
                        )
                        # Compute diagnostic range from AST node (LSP is 0-indexed)
                        start = Position(
                            line=node.lineno - 1, character=node.col_offset
                        )
                        # This simple approach uses the length of the function name.
                        end = Position(
                            line=node.lineno - 1,
                            character=node.col_offset + len(function_name),
                        )
                        diagnostic = Diagnostic(
                            range=Range(start=start, end=end),
                            message=diag_msg,
                            severity=DiagnosticSeverity.Warning,
                            source="exception-warning-lsp",
                        )
                        self.diagnostics.append(diagnostic)
        # Continue walking the tree.
        self.generic_visit(node)


def analyze_document(text):
    """
    Parses the Python source code and runs ExceptionCallVisitor to collect diagnostics.
    """
    try:
        tree = ast.parse(text)
    except Exception as e:
        logging.error("AST parse error: %s", e)
        return []
    visitor = ExceptionCallVisitor(exception_mapping)
    visitor.visit(tree)
    return visitor.diagnostics


# --- LSP Handlers ---


@ls.feature("textDocument/didOpen")
def did_open(ls, params: DidOpenTextDocumentParams):
    text = params.text_document.text
    diagnostics = analyze_document(text)
    ls.publish_diagnostics(params.text_document.uri, diagnostics)


@ls.feature("textDocument/didChange")
def did_change(ls, params: DidChangeTextDocumentParams):
    # Get the updated document text (assuming full text sync)
    text = ls.workspace.get_document(params.text_document.uri).source
    logger.debug(text)
    diagnostics = analyze_document(text)
    logger.debug(f"{diagnostics}")
    ls.publish_diagnostics(params.text_document.uri, diagnostics)


def main():
    ls.start_io()


if __name__ == "__main__":
    main()
