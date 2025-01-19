lexer grammar language_lexer;

IncludePath: [<] ('\\'. | ~[>]*) [>];
String: ["] ('\\'. | ~["])* ["];
Char: ['] ('\\'. | ~[']) ['];
Float: [0-9]+ '.' [0-9]*;
Int: [0-9]+;
Bool: 'true' | 'false';

IncludeDirective: '#include';

Keyword
    : 'alignas'
    | 'alignof'
    | 'and'
    | 'and_eq'
    | 'asm'
    | 'auto'
    | 'bitand'
    | 'bitor'
    | 'bool'
    | 'break'
    | 'case'
    | 'catch'
    | 'char'
    | 'char16_t'
    | 'char32_t'
    | 'class'
    | 'compl'
    | 'const'
    | 'constexpr'
    | 'const_cast'
    | 'continue'
    | 'decltype'
    | 'default'
    | 'delete'
    | 'do'
    | 'double'
    | 'dynamic_cast'
    | 'else'
    | 'enum'
    | 'explicit'
    | 'export'
    | 'extern'
    | 'float'
    | 'for'
    | 'friend'
    | 'goto'
    | 'if'
    | 'inline'
    | 'int'
    | 'long'
    | 'mutable'
    | 'namespace'
    | 'new'
    | 'noexcept'
    | 'not'
    | 'nullptr'
    | 'operator'
    | 'or'
    | 'private'
    | 'protected'
    | 'public'
    | 'register'
    | 'reinterpret_cast'
    | 'return'
    | 'short'
    | 'signed'
    | 'sizeof'
    | 'static'
    | 'static_assert'
    | 'static_cast'
    | 'struct'
    | 'switch'
    | 'template'
    | 'this'
    | 'thread_local'
    | 'throw'
    | 'try'
    | 'typedef'
    | 'typeid'
    | 'typename'
    | 'union'
    | 'unsigned'
    | 'using'
    | 'virtual'
    | 'void'
    | 'volatile'
    | 'wchar_t'
    | 'while'
    | 'xor'
    ;

Access
    : '->'
    | '.'
    ;

Operator
    // TWO LETTERS:
    // Increment and Decrement
    : '++'
    | '--'
    // Assigment Operators
    | '+='
    | '-='
    | '*='
    | '/='
    | '%='
    // Relational Operators
    | '=='
    | '!='
    | '>='
    | '<='
    // Logical Operators
    | '&&'
    | '||'
    // Bitwise Operators
    | '<<'
    | '>>'
    // ONE LETTER:
    // Bitwise Operators
    | '&'
    | '|'
    | '^'
    | '~'
    // Relational Operators
    | '>'
    | '<'
    // Logical Operator
    | '!'
    // Assigment Operator
    | '='
    // Arithmetic Operators
    | '+'
    | '-'
    | '*'
    | '/'
    | '%'
    ;

Comma: ',';
Semicolon: ';';

NewLine: [\n|\r];
Whitespace: [ \t];

OpeningBracket: '(';
ClosingBracket: ')';

OpeningSquareBracket: '[';
ClosingSquareBracket: ']';

CodeBlockOpeningBracket: '{';
CodeBlockClosingBracket: '}';

NamespaceSeparator: '::';

UppercaseIdentifier: [A-Z_] [A-Z0-9_]*;
Identifier: [a-zA-Z_] [a-zA-Z0-9_]*;
