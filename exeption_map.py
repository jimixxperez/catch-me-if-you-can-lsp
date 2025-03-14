import sys
import os
import ast
import inspect
import importlib
import pkgutil
import json
import argparse
import logging
from collections import defaultdict

# Set up logging to stdout
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def get_stdlib_module_names():
    """
    Returns a set of module names that are considered part of the standard library.
    Uses sys.stdlib_module_names (Python 3.10+) if available; otherwise, it falls back
    to scanning the standard library directory and includes built-in modules.
    """
    stdlib_modules = set()
    if hasattr(sys, "stdlib_module_names"):
        stdlib_modules.update(sys.stdlib_module_names)
        logging.info("Using sys.stdlib_module_names for module discovery.")
    else:
        stdlib_path = os.path.dirname(os.__file__)
        for finder, name, ispkg in pkgutil.iter_modules([stdlib_path]):
            stdlib_modules.add(name)
        logging.info("Scanned the standard library directory for module names.")
    stdlib_modules.update(sys.builtin_module_names)
    logging.info("Added built-in module names.")
    return stdlib_modules


def get_exception_name(expr):
    """
    Attempt to extract the exception type name from an AST node in a raise statement.
    """
    if isinstance(expr, ast.Call):
        # e.g., raise ValueError("message")
        return get_exception_name(expr.func)
    elif isinstance(expr, ast.Name):
        return expr.id
    elif isinstance(expr, ast.Attribute):
        # e.g., raise json.JSONDecodeError(...)
        value = get_exception_name(expr.value)
        return f"{value}.{expr.attr}" if value else expr.attr
    return None


class FunctionRaiseCollector(ast.NodeVisitor):
    """
    AST visitor to collect exceptions raised directly in a function's body.
    Does not descend into nested function definitions.
    """

    def __init__(self):
        self.raises = []

    def visit_Raise(self, node):
        if node.exc:
            name = get_exception_name(node.exc)
            if name:
                self.raises.append(name)
        # Do not call generic_visit to avoid nested function defs.
        # However, if there are nested expressions, you can call it.
        # self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Skip nested function definitions.
        pass

    def visit_AsyncFunctionDef(self, node):
        # Skip nested async function definitions.
        pass


class FunctionCallCollector(ast.NodeVisitor):
    """
    AST visitor to collect function calls within a function body that are not wrapped
    by a try/except block. It attempts to resolve simple calls (e.g. foo() or self.foo())
    to their function names.
    """

    def __init__(self, current_class=None):
        self.calls = []
        self.current_class = current_class
        self.try_depth = 0  # counter to mark if we are inside a try block

    def visit_Try(self, node):
        # Increase try depth for the body.
        self.try_depth += 1
        for stmt in node.body:
            self.visit(stmt)
        self.try_depth -= 1
        # Visit orelse and finalbody as well (they are not protected by except).
        for stmt in node.orelse:
            self.visit(stmt)
        for stmt in node.finalbody:
            self.visit(stmt)

    def visit_Call(self, node):
        if self.try_depth == 0:
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                # For calls like self.foo() inside a class.
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                    and self.current_class
                ):
                    func_name = f"{self.current_class}.{node.func.attr}"
                else:
                    # Fallback: record the attribute name.
                    func_name = node.func.attr
            if func_name:
                self.calls.append(func_name)
        self.generic_visit(node)


class ModuleFunctionCollectorWithCalls(ast.NodeVisitor):
    """
    AST visitor that collects function (and async function) definitions in a module,
    along with:
      - Directly raised exceptions (via a FunctionRaiseCollector)
      - Calls to other functions (via a FunctionCallCollector) that are not in a try block.
    """

    def __init__(self):
        self.functions = {}  # qualified function name -> { "raises": [...], "calls": [...], "lineno": int }
        self.class_stack = []

    def visit_ClassDef(self, node):
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        if self.class_stack:
            qualified_name = ".".join(self.class_stack + [node.name])
            current_class = self.class_stack[-1]
        else:
            qualified_name = node.name
            current_class = None

        # Collect direct raises.
        raise_collector = FunctionRaiseCollector()
        for stmt in node.body:
            raise_collector.visit(stmt)
        raises_list = raise_collector.raises

        # Collect calls.
        call_collector = FunctionCallCollector(current_class=current_class)
        call_collector.visit(node)
        calls_list = call_collector.calls

        self.functions[qualified_name] = {
            "raises": raises_list,
            "calls": calls_list,
            "lineno": node.lineno,
        }

        # Continue visiting nested definitions.
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        if self.class_stack:
            qualified_name = ".".join(self.class_stack + [node.name])
            current_class = self.class_stack[-1]
        else:
            qualified_name = node.name
            current_class = None

        raise_collector = FunctionRaiseCollector()
        for stmt in node.body:
            raise_collector.visit(stmt)
        raises_list = raise_collector.raises

        call_collector = FunctionCallCollector(current_class=current_class)
        call_collector.visit(node)
        calls_list = call_collector.calls

        self.functions[qualified_name] = {
            "raises": raises_list,
            "calls": calls_list,
            "lineno": node.lineno,
        }
        self.generic_visit(node)


def propagate_exceptions(functions):
    """
    Propagate exceptions through the call graph within the module.
    For each function, if it calls another function (by name) that is defined in the module,
    merge the callee's exceptions into the caller's set.
    This iterative process continues until a fixpoint is reached.
    """
    changed = True
    while changed:
        changed = False
        for fname, data in functions.items():
            current_ex = set(data["raises"])
            for callee in data.get("calls", []):
                if callee in functions:
                    callee_ex = set(functions[callee]["raises"])
                    union = current_ex.union(callee_ex)
                    if union != current_ex:
                        data["raises"] = list(union)
                        current_ex = union
                        changed = True
    return functions


def analyze_module(module_name):
    """
    Imports the module, introspects its exception classes, and, if possible,
    analyzes its source code (if available as a .py file) to collect:
      - Exception classes defined in the module.
      - Function definitions with direct raises and calls.
      - Propagates exceptions through the call graph.
    Returns a dictionary with two keys:
      - "exceptions_defined": list of exception class names defined in the module.
      - "functions": mapping of function qualified names to { "raises": [...], "calls": [...], "lineno": int }
    """
    result = {"exceptions_defined": [], "functions": {}}
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        logging.warning("Skipping module %s due to import error: %s", module_name, e)
        return result

    # Introspect exception classes defined in the module.
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if (
            issubclass(obj, BaseException)
            and getattr(obj, "__module__", None) == mod.__name__
        ):
            result["exceptions_defined"].append(name)

    # Attempt AST analysis if the module source is available.
    try:
        source_file = inspect.getsourcefile(mod)
    except TypeError:
        logging.info(
            "Module %s is a built-in module, skipping AST analysis.", module_name
        )
        source_file = None

    if source_file and os.path.exists(source_file) and source_file.endswith(".py"):
        try:
            with open(source_file, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=source_file)
            collector = ModuleFunctionCollectorWithCalls()
            collector.visit(tree)
            functions = collector.functions
            # Propagate exceptions from called functions to callers.
            functions = propagate_exceptions(functions)
            result["functions"] = functions
            logging.info("AST analysis complete for module: %s", module_name)
        except Exception as e:
            logging.warning("AST analysis failed for module %s: %s", module_name, e)
    else:
        logging.info("No accessible .py source for module: %s", module_name)

    return result


def map_stdlib(progress_file):
    """
    Iterates through all stdlib modules, analyzes each module for exception definitions,
    direct raises in functions, and indirect raises via calls. Progress is persisted
    to a JSON file after each module.
    Returns a dictionary mapping module names to their analysis.
    """
    stdlib_modules = sorted(get_stdlib_module_names())
    mapping = {}
    # Load previous progress if available.
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as pf:
                mapping = json.load(pf)
            logging.info("Loaded progress from %s", progress_file)
        except Exception as e:
            logging.error("Could not load progress file %s: %s", progress_file, e)

    processed = set(mapping.keys())
    remaining = [m for m in stdlib_modules if m not in processed]
    logging.info(
        "Resuming progress: %d modules already processed, %d modules remaining.",
        len(processed),
        len(remaining),
    )

    try:
        for modname in remaining:
            logging.info("Processing module: %s", modname)
            mapping[modname] = analyze_module(modname)
            # Persist progress after each module.
            with open(progress_file, "w", encoding="utf-8") as pf:
                json.dump(mapping, pf, indent=2)
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Progress persisted to %s", progress_file)
        sys.exit(1)

    logging.info("Completed mapping for all modules.")
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Map Python stdlib exceptions and function raise statements (including propagation through calls) to a JSON file with persistent progress."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="exceptions_map.json",
        help="Output JSON file name for final exception mapping (default: exceptions_map.json)",
    )
    parser.add_argument(
        "-p",
        "--progress",
        type=str,
        default="exceptions_map_partial.json",
        help="Progress JSON file name for persistent progress (default: exceptions_map_partial.json)",
    )
    args = parser.parse_args()

    mapping = map_stdlib(args.progress)
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)
        logging.info("Final exception mapping written to %s", args.output)
    except Exception as e:
        logging.error("Failed to write final output file: %s", e)


if __name__ == "__main__":
    main()
