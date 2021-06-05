package expression.parser;

import expression.CommonExpression;
import expression.Const;
import expression.Variable;
import expression.binary_operators.Add;
import expression.binary_operators.Divide;
import expression.binary_operators.Multiply;
import expression.binary_operators.Subtract;
import expression.binary_operators.bitwiseOperations.And;
import expression.binary_operators.bitwiseOperations.Or;
import expression.binary_operators.bitwiseOperations.Xor;
import expression.unary_operators.Count;
import expression.unary_operators.Minus;
import expression.unary_operators.Not;

import java.util.Map;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class ExpressionParser extends BaseParser implements Parser {
    private static final int MIN_PARSING_DEPTH = 0;
    private static final int MAX_PARSING_DEPTH = 5;

    private static final Map<String, Integer> OPERATORS_PARSING_DEPTH = Map.of(
            "|", 0,
            "^", 1,
            "&", 2,
            "+", 3,
            "-", 3,
            "*", 4,
            "/", 4
    );
    private static final Map<Character, String> OPERATOR_SIGNS = Map.of(
            '+', "+",
            '-', "-",
            '*', "*",
            '/', "/",
            '|', "|",
            '^', "^",
            '&', "&",
            ')', ")"
    );
    private String lastParsedOperator = "";

    public CommonExpression parse(final String source) {
        return parse(new StringSource(source));
    }

    public CommonExpression parse(final CharSource source) {
        setSource(source);
        skipWhitespace();
        CommonExpression result = parseExpression(MIN_PARSING_DEPTH);
        skipWhitespace();
        if (eof()) {
            return result;
        }
        throw error(String.format("End of expression excepted, actual: %c", ch));
    }

    private CommonExpression parseExpression(int parsingDepth) {
        skipWhitespace();
        if (parsingDepth == MAX_PARSING_DEPTH) {
            return parseOperand();
        }
        CommonExpression parsed = parseExpression(parsingDepth + 1);
        while (OPERATORS_PARSING_DEPTH.getOrDefault(lastParsedOperator, MAX_PARSING_DEPTH) == parsingDepth) {
            parsed = buildExpression(
                    lastParsedOperator, parsed, parseExpression(parsingDepth + 1)
            );
        }
        skipWhitespace();
        return parsed;
    }

    private CommonExpression parseOperand() {
        skipWhitespace();
        if (test('-')) {
            if (isDigit()) {
                return parseConst(true);
            } else {
                return new Minus(parseOperand());
            }
        } else if (test('~')) {
            return new Not(parseOperand());
        } else if (test('c')) {
            expect("ount");
            return new Count(parseOperand());
        } else if (isDigit()) {
            return parseConst(false);
        } else if (test('(')) {
            CommonExpression expression = parseExpression(MIN_PARSING_DEPTH);
            if (!lastParsedOperator.equals(")")) {
                throw error("Excepted ')'");
            }
            updateOperator();
            return expression;

        } else {
            return parseVariable();
        }
    }

    private Const parseConst(boolean isNegative) {
        StringBuilder sb = new StringBuilder();
        if (isNegative) {
            sb.append('-');
        }
        while (isDigit()) {
            sb.append(ch);
            nextChar();
        }
        updateOperator();
        String parsed = sb.toString();
        try {
            return new Const(Integer.parseInt(parsed));
        } catch (NumberFormatException e) {
            throw error(String.format("Invalid const: %s", parsed));
        }
    }

    private Variable parseVariable() {
        skipWhitespace();
        StringBuilder sb = new StringBuilder();
        while (!updateOperator() && !eof()) {
            sb.append(ch);
            nextChar();
        }
        return new Variable(sb.toString());
    }

    private boolean updateOperator() {
        skipWhitespace();
        if (eof()) {
            lastParsedOperator = "";
        }
        if (!OPERATOR_SIGNS.containsKey(ch)) {
            return false;
        }
        String exceptedOperator = OPERATOR_SIGNS.get(ch);
        expect(exceptedOperator);
        lastParsedOperator = exceptedOperator;
        return true;
    }

    private CommonExpression buildExpression(String operator, CommonExpression x, CommonExpression y) {
        switch (operator) {
            case "|":
                return new Or(x, y);
            case "^":
                return new Xor(x, y);
            case "&":
                return new And(x, y);
            case "+":
                return new Add(x, y);
            case "-":
                return new Subtract(x, y);
            case "*":
                return new Multiply(x, y);
            case "/":
                return new Divide(x, y);
            default:
                throw error(String.format("Unsupported operator: %s", operator));
        }
    }

    private boolean isDigit() {
        return between('0', '9');
    }

    private void skipWhitespace() {
        while (Character.isWhitespace(ch)) {
            nextChar();
        }
    }
}
