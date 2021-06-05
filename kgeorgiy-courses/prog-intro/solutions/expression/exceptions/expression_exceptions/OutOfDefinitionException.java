package expression.exceptions.expression_exceptions;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class OutOfDefinitionException extends ExpressionException {
    public OutOfDefinitionException(int undefinedValue) {
        super(String.format("Out of scope of function definition for %d", undefinedValue));
    }
}
