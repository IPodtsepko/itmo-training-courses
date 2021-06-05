package expression.exceptions;

import expression.CommonExpression;
import expression.PrioritiesPattern;
import expression.exceptions.expression_exceptions.OutOfDefinitionException;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class CheckedSqrt extends CheckedUnaryOperator {
    protected CheckedSqrt(CommonExpression arg) {
        super(arg);
    }

    @Override
    protected boolean overflowWillHappen(int x) {
        return false;
    }

    @Override
    public int applyFor(int x) {
        if (x < 0) {
            throw new OutOfDefinitionException(x);
        }
        return (int) Math.sqrt(x);
    }

    @Override
    public String getOperator() {
        return "sqrt";
    }

    @Override
    public PrioritiesPattern getPriority() {
        return PrioritiesPattern.MINUS;
    }
}
