import os
import tree_sitter_python
import tree_sitter_java
import tree_sitter_c_sharp
from tree_sitter import Language, Parser, Query, QueryCursor


def parse_code(lines, start_point, end_point):
    if start_point.row == end_point.row:
        return lines[start_point.row][start_point.column:end_point.column]

    data = []
    data.append(lines[start_point.row][start_point.column:])
    for row in range(start_point.row + 1, end_point.row):
        data.append(lines[row])
    data.append(lines[end_point.row][:end_point.column])
    return ('\n').join(data)


def get_functions(path):
    if os.environ['LANG'] == 'python':
        PY_LANGUAGE = Language(tree_sitter_python.language())
        query = Query(PY_LANGUAGE, """
            (function_definition
                name: (identifier) @function.name
                body: (block
                    (expression_statement (string) @function.docstring)
                ) @function.body
            ) @function
        """)
        parser = Parser(PY_LANGUAGE)
    elif os.environ['LANG'] == 'java':
        JAVA_LANGUAGE = Language(tree_sitter_java.language())
        query = Query(JAVA_LANGUAGE, """
            ((block_comment) @function.docstring
            (method_declaration
                name: (identifier) @function.name
                body: (block) @function.body
            ) @function)
        """)
        parser = Parser(JAVA_LANGUAGE)
    elif os.environ['LANG'] == 'csharp':
        CS_LANGUAGE = Language(tree_sitter_c_sharp.language())
        query = Query(CS_LANGUAGE, """
            ((comment)* @function.docstring
            (method_declaration
                name: (identifier) @function.name
                body: (block) @function.body
            ) @function)
        """)
        parser = Parser(CS_LANGUAGE)
    else:
        print('Error tree_sitter no query')
        exit(1)

    query_cursor = QueryCursor(query)

    with open(path, 'rb') as f:
        code = f.read()

    with open(path) as f:
        lines = f.read().split('\n')

    tree = parser.parse(code)
    matches = query_cursor.matches(tree.root_node)
    for match in matches:
        if 'function.docstring' not in match[1]:
            continue

        name = parse_code(lines, match[1]['function.name'][0].start_point, match[1]['function.name'][0].end_point)
        docstring = parse_code(lines, match[1]['function.docstring'][0].start_point, match[1]['function.docstring'][-1].end_point)
        if len(docstring.split()) < 3 or len(docstring.split()) > 256:
            continue
        if 'https://' in docstring or 'http://' in docstring or '<img' in docstring:
            continue

        if os.environ['LANG'] == 'python':
            start_point = match[1]['function'][0].start_point
            end_point = match[1]['function.docstring'][0].end_point
        elif os.environ['LANG'] == 'java':
            start_point = match[1]['function.docstring'][0].start_point
            end_point = match[1]['function.body'][0].start_point
            if match[1]['function'][0].start_point.row - match[1]['function.docstring'][0].end_point.row > 1:
                continue
        elif os.environ['LANG'] == 'csharp':
            start_point = match[1]['function.docstring'][0].start_point
            end_point = match[1]['function.body'][0].start_point
            if match[1]['function'][0].start_point.row - match[1]['function.docstring'][-1].end_point.row > 1:
                continue
        else:
            print('Error tree_sitter parsing')
            exit(1)

        signature = parse_code(lines, start_point, end_point)

        yield name.strip(), signature.strip()


def get_function_calls(function_code):
    if os.environ['LANG'] == 'python':
        PY_LANGUAGE = Language(tree_sitter_python.language())
        query = Query(PY_LANGUAGE, """
            (call
                function: (identifier) @call.name
            )
        """)
        parser = Parser(PY_LANGUAGE)
    elif os.environ['LANG'] == 'java':
        JAVA_LANGUAGE = Language(tree_sitter_java.language())
        query = Query(JAVA_LANGUAGE, """
            (method_invocation
                name: (identifier) @call.name
            )
        """)
        parser = Parser(JAVA_LANGUAGE)
    elif os.environ['LANG'] == 'csharp':
        CS_LANGUAGE = Language(tree_sitter_c_sharp.language())
        query = Query(CS_LANGUAGE, """
            (invocation_expression
                function: (member_access_expression
                name: (identifier) @call.name)
            )
        """)
        parser = Parser(CS_LANGUAGE)
    else:
        print('Error tree_siiter')
        exit(1)

    query_cursor = QueryCursor(query)

    tree = parser.parse('\n'.join(function_code).encode())
    captures = query_cursor.captures(tree.root_node)
    if 'call.name' not in captures:
        return []

    results = [(c.start_point.row, parse_code(function_code, c.start_point, c.end_point)) for c in captures['call.name']]
    return [x[1] for x in sorted(results, key=lambda x: x[0], reverse=True)]
