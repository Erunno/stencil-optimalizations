import os

DEAD = False
ALIVE = True

DEAD_VAR = '_'
ALIVE_VAR = 'X'

CPP_START_MARKER = '/* LEXICON_START */'
CPP_END_MARKER = '/* LEXICON_END */'

class Pattern:
    def __init__(self, lines, start_idx):
        idx = start_idx + 1

        while idx < len(lines) and (not lines[idx].startswith(':')):
            idx += 1 
        
        self.lines = lines[start_idx:idx]

    def name(self) -> str:
        desc_line = self.lines[0]
        raw_name = desc_line.split(':')[1]
        return raw_name.strip().replace(' ', '-').lower()
    
    def is_empty(self) -> bool:
        body = self.body()
        return len(body) == 0 or (len(body) == 1 and len(body[0]) == 0)

    def _replace_ws(self, line):
        return line.replace(' ', '-')
    
    def body(self) -> list[list[bool]]:
        body_lines = self.lines

        start_idx = -1
        end_idx = -1

        definition_chars = ['.', '*', ' ', '\n', '\t']

        for i in range(len(body_lines)):
            start_idx = i
            if all(c in definition_chars for c in body_lines[i]):
                break

        for i in range(start_idx, len(body_lines)):
            end_idx = i
            if not all(c in definition_chars for c in body_lines[i]):
                break
        
        body_lines = body_lines[start_idx:end_idx]

        body = []
        for line in body_lines:
            row = []
            for c in line.strip():
                if c == '.':
                    row.append(DEAD)
                elif c == '*':
                    row.append(ALIVE)
            body.append(row)

        return body

    def to_cpp_record(self):
        name = self.name()
        cpp_body_rows = self._load_cpp_body_rows()

        rec_template = '''{"<name>", {
<rows>}},
'''
        
        rec = rec_template.replace('<name>', name)
        rec = rec.replace('<rows>', cpp_body_rows)

        return rec

    def _load_cpp_body_rows(self):
        row_template = '''{<cells>}'''

        body = self.body()
        rows = []

        for row in body:
            cells = []
            for cell in row:
                var = ALIVE_VAR if cell == ALIVE else DEAD_VAR
                cells.append(var)
            
            cells_str = ', '.join(cells)

            row = row_template.replace('<cells>', cells_str)
            rows.append(row)

        return ',\n'.join(rows)


class Lexicon:
    def __init__(self):
        self.patterns: list[Pattern] = []

    def parse(self, input_text: str):
        lines = input_text.split('\n')

        for i in range(len(lines)):
            if lines[i].startswith(':'):
                pattern = Pattern(lines, i)
                self.patterns.append(pattern)

    def to_cpp(self):
        cpp_body = []

        for pattern in self.patterns:
            if not pattern.is_empty():
                cpp_body.append(pattern.to_cpp_record())

        return '\n'.join(cpp_body)


def replace_section(cpp_src, lexicon: Lexicon):
    new_src = ''

    skipping = False

    for line in cpp_src.split('\n'):
        if not skipping:
            new_src += line + '\n'

        if CPP_START_MARKER in line:
            skipping = True
            new_src += lexicon.to_cpp() + '\n'
        
        if CPP_END_MARKER in line:
            skipping = False
            new_src += line + '\n'

    return new_src

#
# MAIN
#

WORKING_DIR = os.path.dirname(os.path.abspath(__file__))

LEXICON_FILE_PATH = os.path.join(WORKING_DIR, 'lexicon.txt')
LEXICON_TEXT = open(LEXICON_FILE_PATH, 'r').read()

CPP_SOURCE_FILE_PATH = os.path.join(WORKING_DIR, 'patterns.cpp')
CPP_SOURCE_CONTENT = open(CPP_SOURCE_FILE_PATH, 'r').read()

lexicon = Lexicon()
lexicon.parse(LEXICON_TEXT)

new_cpp_src = replace_section(CPP_SOURCE_CONTENT, lexicon)

with open(CPP_SOURCE_FILE_PATH, 'w') as f:
    f.write(new_cpp_src)

print ('Done!')
