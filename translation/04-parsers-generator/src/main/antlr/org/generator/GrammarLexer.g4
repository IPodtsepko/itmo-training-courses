lexer grammar GrammarLexer;

@header {
package org.generator;
}

// The keyword for the designation of imports.
Import : 'import' ;

// The keyword for the lexer rules.
Val : 'val';

// The keyword for the parser rules.
Def : 'def';

// A colon marks the start of a lexer or parser rule.
Colon : ':';

// A semicolon marks the end of a lexer or parser rule.
Semicolon : ';';

// A comma that is a separator in lists.
Comma : ',';

// Opening bracket.
OpeningBracket : '(';

// Closing bracket.
ClosingBracket : ')';

// Identifier - a non-empty sequence of letters and numbers
Identifier : [a-zA-Z0-9\\.]+;

// Left arrow to improve the readability of the parser rules.
// Analog of "return" in ANTLR.
Arrow : '->';

// Java regular expression for declaring lexer rules
// (can also be used for parsing strings in quotes)
RegularExpression : '\''(~['])+'\'';

// Alternative separator in the parser rules
Pipe : '|';

// All whitespace characters are skipped when parsing grammar
Whitespaces : [ \r\n\t] -> skip;
