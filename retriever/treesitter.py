import os
import tree_sitter_python
import tree_sitter_java
import tree_sitter_c_sharp
from tree_sitter import Language, Parser, Query, QueryCursor


def parse_code(f, lines):
    if f.start_point.row == f.end_point.row:
        return lines[f.start_point.row][f.start_point.column:f.end_point.column]

    data = []
    data.append(lines[f.start_point.row][f.start_point.column:])
    for row in range(f.start_point.row + 1, f.end_point.row):
        data.append(lines[row])
    data.append(lines[f.end_point.row][:f.end_point.column])
    return ('\n').join(data)


def get_functions(path):
    if os.environ['LANG'] == 'python':
        PY_LANGUAGE = Language(tree_sitter_python.language())
        query = Query(PY_LANGUAGE, """
            (function_definition
                name: (identifier) @function.name
                body: (block) @function.body
            ) @function
        """)
        parser = Parser(PY_LANGUAGE)
    elif os.environ['LANG'] == 'java':
        JAVA_LANGUAGE = Language(tree_sitter_java.language())
        query = Query(JAVA_LANGUAGE, """
            (method_declaration
                name: (identifier) @function.name
                body: (block) @function.body
            ) @function
        """)
        parser = Parser(JAVA_LANGUAGE)
    elif os.environ['LANG'] == 'csharp':
        CS_LANGUAGE = Language(tree_sitter_c_sharp.language())
        query = Query(CS_LANGUAGE, """
            (method_declaration
                name: (identifier) @function.name
                body: (block) @function.body
            ) @function
        """)
        parser = Parser(CS_LANGUAGE)
    else:
        print('Error tree_siiter')
        exit(1)

    query_cursor = QueryCursor(query)

    with open(path, 'rb') as f:
        code = f.read()

    with open(path) as f:
        lines = f.read().split('\n')

    tree = parser.parse(code)
    matches = query_cursor.matches(tree.root_node)
    for match in matches:
        name = parse_code(match[1]['function.name'][0], lines)
        f = parse_code(match[1]['function'][0], lines)
        body = parse_code(match[1]['function.body'][0], lines)
        signature = f[:-len(body)]

        yield name.strip(), signature.strip(), f.strip()


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
    elif os.environ['LANG'] == 'typescript':
        TYPESCRIPT_LANGUAGE = Language(tree_sitter_typescript.language_typescript())
        query = Query(TYPESCRIPT_LANGUAGE, """
            (call_expression
                function: (identifier) @call.name
            )
        """)
        parser = Parser(TYPESCRIPT_LANGUAGE)
    else:
        print('Error tree_siiter')
        exit(1)

    query_cursor = QueryCursor(query)

    tree = parser.parse('\n'.join(function_code).encode())
    captures = query_cursor.captures(tree.root_node)
    if 'call.name' not in captures:
        return []

    results = [(c.start_point.row, parse_code(c, function_code))  for c in captures['call.name']]
    return [x[1] for x in sorted(results, key=lambda x: x[0], reverse=True)]
