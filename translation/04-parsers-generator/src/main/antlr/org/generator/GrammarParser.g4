parser grammar GrammarParser;

options {
    tokenVocab=GrammarLexer;
}

@header {
package org.generator;

import org.generator.util.GrammarItem;
import org.generator.util.Import;
import org.generator.util.LexerRule;
import org.generator.util.ParserRule;

import org.generator.util.Alternative;
import org.generator.util.AlternativeItem;
import org.generator.util.Attribute;
import org.generator.util.CodeInsertion;
import org.generator.util.ParserRuleCall;
import org.generator.util.LexerRuleCall;

import java.util.List;
import java.util.Map;
import java.util.*;
}

/**
 * Description of the grammar in the form of a set of imports and rules for the lexer and parser.
 *
 * @return values list of imports and rules
 */
grammarDescription returns [List<GrammarItem> values] @init { $values = new ArrayList<>(); }
    : rule[$values]+
    ;

/**
 * Description of the grammar in the form of a set of imports and rules for the lexer and parser.
 *
 * @param values the list to which imports and rules are added.
 */
rule [List<GrammarItem> values]
    : Import javaImport [$values] Semicolon
    | Val lexerRuleDeclaration [$values] Semicolon
    | Def parserRuleDeclaration [$values] Semicolon
    ;

/**
 * Import declaration in quotes.
 *
 * @param values the list to which imports and rules are added.
 */
javaImport [List<GrammarItem> values]
    : RegularExpression {
            {
                final var text = $RegularExpression.text;
                $values.add(new Import(text.substring(1, text.length() - 1)));
            }
        }
    ;

/**
 * A rule for a lexer that is actually a regular expression.
 *
 * @param values the list to which imports and rules are added.
 * @return value parsed terminal as Java-object.
 */
lexerRuleDeclaration [List<GrammarItem> values] returns [LexerRule value] @init { $value = new LexerRule(); }
    : Identifier { $value.setName($Identifier.text); } Colon RegularExpression {
            {
                final var text = $RegularExpression.text;
                $value.setRegularExpression(text.substring(1, text.length() - 1));
            }
        } {
            $values.add($value);
        }
    ;

/**
 * Rule for the parser.
 *
 * @param values the list to which imports and rules are added.
 * @return value parsed non-terminal as Java-object.
 */
parserRuleDeclaration [List<GrammarItem> values] returns [ParserRule value] @init { $value = new ParserRule(); }
    : Identifier {
          $value.name = $Identifier.text;
      }
      attributes {
                $value.inherited = $attributes.list;
      }
      Arrow
      attributes {
          $value.synthesized = $attributes.list;
      }
      // List of alternatives:
      Colon alternatives[$value]+ {
          $values.add($value);
      }
    ;

/**
 * List of attributes listed in parentheses.
 *
 * @return list parsed attributes as Java-list of Java-objects.
 */
attributes returns [List<Attribute> list] @init { $list = new ArrayList<>(); }
    : OpeningBracket
      (
            attribute { $list.add($attribute.value); } | attribute { $list.add($attribute.value); } Comma
      )*
      ClosingBracket
    ;

/**
 * An attribute declaration consisting of its type and name.
 *
 * @return value parsed attribute as Java-objects.
 */
attribute returns [Attribute value] @init {$value = new Attribute();}
    : // Just type in quotes, it is convenient when the type consists of several words:
      RegularExpression {
            {
                final var text = $RegularExpression.text;
                $value.setType(text.substring(1, text.length() - 1));
            }
      }
      // Value name:
      Identifier {
            $value.setName($Identifier.text);
      }
    ;

/**
 * Parser rules separated by pipeline symbols.
 *
 * @param value a nonterminal whose alternatives are parsed.
 */
alternatives [ParserRule value]
    : (Pipe? alternative {
            $value.alternatives.add($alternative.value);
        })+
    ;

/**
 * Parser rules separated by pipeline symbols.
 *
 * @return line parsed alternative as Java-object.
 */
alternative returns [Alternative value] @init { $value = new Alternative(); }
    : (unit[$value])+
    ;

/**
 * Alternative element of the parser rule.
 *
 * @param units an alternative whose elements are parsed.
 */
unit [Alternative value]
    : codeInsertion[$value]
    | token[$value]
    ;

/**
 * Code insertion in alternative (specified in the quotes).
 *
 * @param units an alternative whose elements are parsed.
 */
codeInsertion [Alternative value]
    : RegularExpression {
            final var text = $RegularExpression.text;
            final var code = text.substring(1, text.length() - 1);
            value.addItem(new CodeInsertion(code));
        }
    ;

/**
 * Terminal or non-terminal from alternative.
 *
 * @param units an alternative whose elements are parsed.
 * @returns name terminal's or non-terminal's name.
 */
token [Alternative value] returns [String name]
    : Identifier {
            $name = $Identifier.text;
        }
      maybeCall[$name] {
            final var parserRule = $maybeCall.parserRule;
            final var item = parserRule == null ? new LexerRuleCall($name) : parserRule;
            value.addItem(item);
      }
    ;

/**
 * Checks if there are no parentheses indicating a token call.
 * The token after which there are brackets is a non-terminal.
 *
 * @param name terminal's or non-terminal's name.
 * @returns parserRules non-null Java-object containing a non-terminal
 *          if a list of arguments for the call was provided,
 *          else null.
 */
maybeCall [String name] returns [ParserRuleCall parserRule]
    : (OpeningBracket /* is non-terminal */ {
            $parserRule = new ParserRuleCall($name);
        }
      callArguments[$parserRule]?
      ClosingBracket)?
    ;

/**
 * List of non-terminal call arguments.
 *
 * @param parserRules an non-terminal whose arguments are parsed.
 */
callArguments [ParserRuleCall parserRule]
    : (callArgument[$parserRule] Comma)* callArgument[$parserRule]
    ;

/**
 * Single non-terminal call argument.
 *
 * @param parserRules an non-terminal whose arguments are parsed.
 */
callArgument [ParserRuleCall parserRule]
    : Identifier { $parserRule.addArgument($Identifier.text); }
    ;
