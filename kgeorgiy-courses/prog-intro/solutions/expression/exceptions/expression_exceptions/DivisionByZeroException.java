package expression.exceptions.expression_exceptions;

import expression.exceptions.expression_exceptions.ExpressionException;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class DivisionByZeroException extends ExpressionException {
    public DivisionByZeroException() {
        super("division by zero");
    }
}
