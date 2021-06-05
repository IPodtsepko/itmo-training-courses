package expression.exceptions.expression_exceptions;

import expression.exceptions.expression_exceptions.ExpressionException;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class UnsupportedOperatorException extends ExpressionException {
    public UnsupportedOperatorException(String operator) {
        super(String.format("unsupported operator encountered: %s", operator));
    }
}
