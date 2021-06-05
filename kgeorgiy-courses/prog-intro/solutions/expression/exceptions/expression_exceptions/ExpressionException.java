package expression.exceptions.expression_exceptions;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
abstract public class ExpressionException extends RuntimeException {
    public ExpressionException(String message) {
        super(message);
    }
}
