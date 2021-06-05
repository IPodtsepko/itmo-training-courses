package expression.parser;

import expression.CommonExpression;
import expression.Const;
import expression.Variable;
import expression.interfaces.Parser;
import expression.operators.binary.*;
import expression.operators.unary.Abs;
import expression.operators.unary.Negate;
import expression.operators.unary.Square;
import expression.parser.exceptions.*;

import java.util.List;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class ExpressionParser extends BaseParser implements Parser {
    private static final List<List<String>> OPERATORS_FOR_PARSING_DEPTH = List.of(
            List.of("+", "-"),
            List.of("*", "/", "mod")
    );
    private static final int MIN_PARSING_DEPTH = 0;
    private static final int MAX_PARSING_DEPTH = OPERATORS_FOR_PARSING_DEPTH.size();

    private static final List<Character> SPECIAL_OPERATION_CHARS = List.of('+', '-', '*', '/', '(', ')', '\0');

    public CommonExpression parse(final String source) {
        return parse(new StringSource(source));
    }

    public CommonExpression parse(final CharSource source) {
        setSource(source);
        final CommonExpression result = parseExpression(MIN_PARSING_DEPTH);
        if (eof()) {
            return result;
        }
        throw new ExtraCharExceptions(getPosition());
    }

    private CommonExpression parseExpression(final int parsingDepth) {
        skipWhitespace();
        if (parsingDepth == MAX_PARSING_DEPTH) {
            return parseOperand();
        }
        CommonExpression parsed = parseExpression(parsingDepth + 1);
        while (true) {
            boolean found = false;
            skipWhitespace();
            for (final String operator : OPERATORS_FOR_PARSING_DEPTH.get(parsingDepth)) {
                if (test(operator)) {
                    if (operator.length() > 1 && !Character.isWhitespace(ch) && ch != '-') {
                        throw new UnexpectedCharException(' ', ch, getPosition());
                    }
                    parsed = buildExpression(operator, parsed, parseExpression(parsingDepth + 1));
                    found = true;
                }
            }
            if (!found) break;
        }
        return parsed;
    }

    private CommonExpression parseOperand() {
        skipWhitespace();
        if (test('-')) {
            if (isDigit(ch)) {
                return parseConst(true);
            } else {
                return new Negate(parseOperand());
            }
        } else if (isDigit(ch)) {
            return parseConst(false);
        } else if (test('(')) {
            final CommonExpression expression = parseExpression(MIN_PARSING_DEPTH);
            expect(')');
            return expression;
        } else {
            final String parsed = parseToken(ch -> !SPECIAL_OPERATION_CHARS.contains(ch) && !Character.isWhitespace(ch));
            switch (parsed) {
                case "abs":
                    return new Abs(parseOperand());
                case "square":
                    return new Square(parseOperand());
                case "x":
                case "y":
                case "z":
                    return new Variable(parsed);
                case "":
                    throw new ArgumentNotFoundException(getPosition());
                default:
                    throw new InvalidVariableException(parsed, getPosition());
            }
        }
    }

    private String parseToken(final CharChecker checker) {
        final StringBuilder sb = new StringBuilder();
        while (checker.isSuitable(ch)) {
            sb.append(ch);
            nextChar();
        }
        return sb.toString();
    }

    private Const parseConst(final boolean isNegative) {
        final String parsed = (isNegative ? "-" : "") + parseToken(ch -> isDigit(ch) || ch == '.');
        final int position = getPosition();
        skipWhitespace();
        if (isDigit(ch)) {
            throw new SpaceInNumberException(position);
        }
        return new Const(parsed);
    }

    private static CommonExpression buildExpression(final String operator, final CommonExpression left, final CommonExpression right) {
        switch (operator) {
            case "+":
                return new Add(left, right);
            case "-":
                return new Subtract(left, right);
            case "*":
                return new Multiple(left, right);
            case "/":
                return new Divide(left, right);
            case "mod":
                return new Mod(left, right);
            default:
                throw new UnsupportedOperationException(operator);
        }
    }

    private boolean isDigit(final char c) {
        return '0' <= c && c <= '9';
    }

    private void skipWhitespace() {
        while (Character.isWhitespace(ch)) {
            nextChar();
        }
    }

    private interface CharChecker {
        boolean isSuitable(char ch);
    }
}