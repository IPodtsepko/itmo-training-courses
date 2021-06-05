package expression.exceptions;

import expression.CommonExpression;
import expression.Const;
import expression.Variable;
import expression.exceptions.expression_exceptions.UnsupportedOperatorException;
import expression.exceptions.parse_exceptions.*;
import expression.parser.BaseParser;
import expression.parser.CharSource;
import expression.parser.StringSource;

import java.util.List;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class ExpressionParser extends BaseParser implements Parser {
    private static final List<List<String>> OPERATORS_FOR_PARSING_DEPTH = List.of(
            List.of("min", "max"),
            List.of("+", "-"),
            List.of("*", "/")
    );
    private static final int MIN_PARSING_DEPTH = 0;
    private static final int MAX_PARSING_DEPTH = OPERATORS_FOR_PARSING_DEPTH.size();

    private static final List<Character> SPECIAL_OPERATION_CHARS = List.of('+', '-', '*', '/', '(', ')', '\0');

    public CommonExpression parse(final String source) {
        return parse(new StringSource(source));
    }

    public CommonExpression parse(final CharSource source) {
        setSource(source);
        CommonExpression result = parseExpression(MIN_PARSING_DEPTH);
        if (eof()) {
            return result;
        }
        if (ch == ')') {
            throw new OpenParenthesisMissedException(getPosition());
        }
        throw new ExtraCharExceptions(getPosition());
    }

    private CommonExpression parseExpression(int parsingDepth) {
        skipWhitespace();
        if (parsingDepth == MAX_PARSING_DEPTH) {
            return parseOperand();
        }
        CommonExpression parsed = parseExpression(parsingDepth + 1);
        while (true) {
            boolean found = false;
            skipWhitespace();
            for (String operator : OPERATORS_FOR_PARSING_DEPTH.get(parsingDepth)) {
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
                return new CheckedNegate(parseOperand());
            }
        } else if (isDigit(ch)) {
            return parseConst(false);
        } else if (test('(')) {
            CommonExpression expression = parseExpression(MIN_PARSING_DEPTH);
            expect(')');
            return expression;
        } else {
            String parsed = parseToken(ch -> !SPECIAL_OPERATION_CHARS.contains(ch) && !Character.isWhitespace(ch));
            switch (parsed) {
                case "abs":
                    return new CheckedAbs(parseOperand());
                case "sqrt":
                    return new CheckedSqrt(parseOperand());
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

    private interface CharChecker {
        boolean isSuitable(char ch);
    }

    private String parseToken(CharChecker checker) {
        StringBuilder sb = new StringBuilder();
        while (checker.isSuitable(ch)) {
            sb.append(ch);
            nextChar();
        }
        return sb.toString();
    }

    private Const parseConst(boolean isNegative) {
        String parsed = (isNegative ? "-" : "") + parseToken(ch -> isDigit(ch));
        int position = getPosition();
        skipWhitespace();
        if (isDigit(ch)) {
            throw new SpaceInNumberException(position);
        }
        try {
            return new Const(Integer.parseInt(parsed));
        } catch (NumberFormatException e) {
            throw new InvalidConstException(parsed, getPosition());
        }
    }

    private CommonExpression buildExpression(String operator, CommonExpression x, CommonExpression y) {
        switch (operator) {
            case "min":
                return new Min(x, y);
            case "max":
                return new Max(x, y);
            case "+":
                return new CheckedAdd(x, y);
            case "-":
                return new CheckedSubtract(x, y);
            case "*":
                return new CheckedMultiply(x, y);
            case "/":
                return new CheckedDivide(x, y);
            default:
                throw new UnsupportedOperatorException(operator);
        }
    }

    private boolean isDigit(char c) {
        return '0' <= c && c <= '9';
    }

    private void skipWhitespace() {
        while (Character.isWhitespace(ch)) {
            nextChar();
        }
    }
}
