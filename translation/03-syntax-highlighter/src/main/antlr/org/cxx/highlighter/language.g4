grammar language;

import language_lexer;

@header { package org.cxx.highlighter; }

main
    : (include | function | variable)* skip EOF
    ;

include
    : skip includeDirective skip (includePath | string)
    ;

includeDirective
    : IncludeDirective
    ;

includePath
    : IncludePath
    ;

variable
    : skip type skip identifier skip (safeOperator value)?;

function
    : skip functionDeclaration skip functionBody?
    ;

functionDeclaration
    :  skip type skip functionName skip OpeningBracket skip argumentsDeclaration? skip ClosingBracket
    ;

functionName
    :   identifier
    ;

functionBody
    : CodeBlockOpeningBracket (skip line skip)* CodeBlockClosingBracket
    ;

line
    : value
    | variable
    | keyword skip value // return x + 1;
    ;

argumentsDeclaration
    : variable (skip Comma skip variable)*
    ;

argumentList
    : (value (skip Comma skip value)*)?
    ;

value
    : operand (call | index)* access* (safeOperator value)*
    ;

safeOperator
    : skip operator skip
    ;

operator
    : Operator
    ;

access
    : skip Access internal
    ;

internal
    : (member | string) (call | index)*
    ;

member
    : identifier
    ;

operand
    : OpeningBracket skip value skip ClosingBracket
    | Operator skip value
    | identifier
    | literal
    ;

call
    : skip OpeningBracket skip argumentList skip ClosingBracket
    ;

index
    : skip OpeningSquareBracket skip value skip ClosingSquareBracket
    ;

type
    : nonReferenceType (skip Operator)?
    ;

nonReferenceType
    : (keyword skip)* identifier?
    ;

identifier
    : someId (NamespaceSeparator someId)*
    ;

someId
    : uppercase
    | anyId
    ;

uppercase
    : UppercaseIdentifier
    ;

anyId
    : Identifier
    ;

literal
    : string
    | primitive
    | (Operator skip primitive);

string
    : String
    ;

primitive
    : char
    | Int
    | Float
    | Bool
    ;

char
    : Char
    ;

keyword: Keyword;

skip
    : ( NewLine
      | Whitespace
      | Semicolon
      )*
    ;
