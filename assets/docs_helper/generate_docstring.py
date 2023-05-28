"""
A script to generate docstrings for Python files based on a JSON config.

Usage
-----
python generate_docstring.py --param_file parameters.json --py_files file1.py file2.py

Returns
-------
The script will generate a new file for each Python file provided with the suffix "_updated".
User can then compare the original file and the updated file to check if the docstrings are
generated correctly. If the docstrings are correct, the updated file can be used to replace
the original file.
"""

import ast
import json
import re
import argparse
import textwrap
import os
from collections import defaultdict

NO_DEFAULT = "<No Default>"
MAX_WIDTH = 79
INDENT = " " * 4
DOC_FUNC = "-DOC_FUNC>"
DOC_RETURN = "-DOC_RETURN>"
DOC_ATTR = "-DOC_ATTR>"


class FunctionVisitor(ast.NodeVisitor):
    """
    A class for visiting FunctionDef nodes in an abstract syntax tree (AST).

    This class extends the ast.NodeVisitor class and is used to extract all
    functions (not part of any class) in the Python source code.

    Attributes
    ----------
    funcs : list
        A list of FunctionDef nodes that are not part of any class.
    inside_class : bool
        A flag indicating whether the current node being visited is inside a class.

    Methods
    -------
    visit_FunctionDef(node) :
        Extracts the FunctionDef node if it is not part of any class.
    visit_ClassDef(node) :
        Sets the inside_class flag to True when entering a ClassDef node and to False when leaving.
    """

    def __init__(self):
        self.funcs = []
        self.inside_class = False

    def visit_FunctionDef(self, node):
        if not self.inside_class:
            self.funcs.append(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.inside_class = True
        self.generic_visit(node)
        self.inside_class = False


class ClassVisitor(ast.NodeVisitor):
    """
    A class for visiting ClassDef nodes in an abstract syntax tree (AST).

    This class extends the ast.NodeVisitor class and is used to extract all
    classes and their respective methods in the Python source code.

    Attributes
    ----------
    classes : list
        A list of ClassDef nodes in the Python source code.
    class_methods : dict
        A dictionary mapping class names to their respective FunctionDef nodes.

    Methods
    -------
    visit_ClassDef(node) :
        Extracts the ClassDef node and its methods.
    """

    def __init__(self):
        self.classes = []
        self.class_methods = {}

    def visit_ClassDef(self, node):
        self.classes.append(node)
        self.class_methods[node.name] = [
            n for n in ast.iter_child_nodes(node) if isinstance(n, ast.FunctionDef)
        ]
        self.generic_visit(node)


def wrap_text(text):
    return textwrap.fill(
        text, MAX_WIDTH, initial_indent=INDENT, subsequent_indent=INDENT
    )


def add_indent(text, indent):
    text = "\n".join([indent + line for line in text.split("\n")])
    return text


def generate_params_docstring(func_name, params, doc_items):
    """
    Generates the parameters docstring for a function.

    Raises
    ------
    Exception
        If any attribute is not found in the documentation items.
    """
    param_strings = []
    for param, default in params.items():
        if param not in doc_items:
            raise Exception(
                f"Parameter {param} not found in param file for function {func_name}"
            )

        optional = ", optional" if default != NO_DEFAULT else ""
        default_str = f" Default is {default}." if default != NO_DEFAULT else ""
        wrapped_desp = wrap_text(f"{doc_items[param]['desp']}{default_str}")
        param_strings.append(
            f"{param} : {doc_items[param]['type']}{optional}\n{wrapped_desp}"
        )

    return "\nParameters\n----------\n" + "\n".join(param_strings)


def generate_returns_docstring(func_name, doc_items):
    """Generates the returns docstring for a function."""
    return_strings = [
        f"{item['type']}\n{wrap_text(item['desp'])}" for item in doc_items.values()
    ]
    return "\nReturns\n-------\n" + "\n".join(return_strings)


def extract_params(func):
    """Extracts parameters and their default values from a function."""
    params = [arg.arg for arg in func.args.args if arg.arg != "self"]
    default_values = [
        NO_DEFAULT if i < len(func.args.defaults) else ast.literal_eval(node)
        for i, node in enumerate(func.args.defaults)
    ]
    return dict(zip(params, default_values))


def process_func(func, func_name, func_doc_items, content):
    """
    Adds docstrings to a function.

    Raises
    ------
    Exception
        If the docstring marker is not found in the content (source code).
    """

    def replace_marker_if_present(marker_name, new_text):
        marker = f"<{marker_prefix}{marker_name}"
        if marker in content:
            return content.replace(marker, add_indent(new_text, indent))
        raise Exception(f"Cannot find {marker} in {func_name} docstring")

    ori_docstring = ast.get_docstring(func)
    marker_prefix = func_name.split(".", 1)[-1]
    indent = " " * (func.col_offset + 4)

    if DOC_FUNC in ori_docstring:
        func_params = extract_params(func)
        param_docstring = generate_params_docstring(
            func_name, func_params, func_doc_items["params"]
        )
        content = replace_marker_if_present(DOC_FUNC, param_docstring)

    if DOC_RETURN in ori_docstring:
        return_docstring = generate_returns_docstring(
            func_name, func_doc_items["returns"]
        )
        content = replace_marker_if_present(DOC_RETURN, return_docstring)

    return content


def extract_string_between_markers(text, start_marker, end_marker):
    """Extracts a string in the provided text that lies between the start and end markers."""
    pattern = rf"({re.escape(start_marker)}.*?{re.escape(end_marker)})"
    return next(re.finditer(pattern, text, flags=re.DOTALL)).group()


def generate_attrs_docstring(attrs, doc_items):
    """
    Generates docstring for the attributes based on the documentation items provided.

    Raises
    ------
    Exception
        If any attribute is not found in the documentation items.
    """
    attrs_docstring = []
    for attr in attrs:
        if attr in doc_items:
            attrs_docstring.append(
                f"{attr} : {doc_items[attr]['type']}\n{wrap_text(doc_items[attr]['desp'])}\n"
            )
        else:
            raise Exception(
                f"Attribute {attr} not found in param file for class {cls_name}"
            )
    return f"\nAttributes\n----------\n{''.join(attrs_docstring)}"[:-1]


def process_class(cls, cls_name, cls_doc_items, content):
    """
    Processes a given class to replace attribute docstring markers with actual attribute docstrings.

    Raises
    ------
    Exception
        If attr_string is not found in the content.
    """
    attr_string = extract_string_between_markers(
        content, f"<{cls_name.split('.', 1)[-1]}{DOC_ATTR}", ">"
    )
    attrs = [attr.strip() for attr in attr_string.split("\n")[1:-1]]
    if attr_string in content:
        attrs_docstring = generate_attrs_docstring(attrs, cls_doc_items["attrs"])
        indent = " " * (cls.col_offset + 4)
        content = content.replace(attr_string, add_indent(attrs_docstring, indent))
    else:
        raise Exception(f"Cannot find {attr_string} in {cls_name} docstring")

    return content


def need_docstring(node, *markers):
    """Check if the node requires a docstring."""
    docstring = ast.get_docstring(node)
    return any(marker in docstring for marker in markers) if docstring else False


def process_file(file_path, doc_items):
    """Process a Python file to update docstrings."""
    with open(file_path, "r") as file:
        content = file.read()
    file_name = os.path.basename(file_path).rsplit(".", 1)[0]

    module = ast.parse(content)

    func_visitor = FunctionVisitor()
    func_visitor.visit(module)

    class_visitor = ClassVisitor()
    class_visitor.visit(module)

    for func in func_visitor.funcs:
        if need_docstring(func, DOC_FUNC, DOC_RETURN):
            func_name = f"{file_name}.{func.name}"
            content = process_func(func, func_name, doc_items[func_name], content)

    for cls_name, funcs in class_visitor.class_methods.items():
        for func in funcs:
            if need_docstring(func, DOC_FUNC, DOC_RETURN):
                func_name = f"{file_name}.{cls_name}.{func.name}"
                content = process_func(func, func_name, doc_items[func_name], content)

    for cls in class_visitor.classes:
        if need_docstring(cls, DOC_ATTR):
            cls_name = f"{file_name}.{cls.name}"
            content = process_class(cls, cls_name, doc_items[cls_name], content)

    file_dir = os.path.dirname(file_path)
    with open(os.path.join(file_dir, f"{file_name}_updated.py"), "w") as file:
        file.write(content)


def parse_json(json_file):
    """Parses a JSON file into a dictionary of documentation items."""
    with open(json_file, "r") as file:
        raw_items = json.load(file)

    doc_items = defaultdict(lambda: {"params": {}, "returns": {}, "attrs": {}})
    for item in raw_items:
        for key in item["target"]:
            for func_name in item["target"][key]:
                doc_items[func_name][key][item["param"]] = item

    return doc_items


def main(json_file, py_files):
    doc_items = parse_json(json_file)
    for py_file in py_files:
        process_file(py_file, doc_items)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate docstrings for python files based on a JSON config."
    )
    parser.add_argument(
        "--param_file",
        type=str,
        default="parameters.json",
        required=False,
        help="Path to the JSON file that contains docstring config. Default is parameters.json.",
    )
    parser.add_argument(
        "--py_files",
        type=str,
        nargs="+",
        help="Path(s) to the Python file(s) for which to generate docstrings.",
    )
    args = parser.parse_args()

    main(args.param_file, args.py_files)
