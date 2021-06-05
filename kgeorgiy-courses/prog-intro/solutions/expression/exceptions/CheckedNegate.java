package expression.exceptions;

import expression.CommonExpression;
import expression.DoubleExpression;
import expression.PrioritiesPattern;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class CheckedNegate extends CheckedUnaryOperator {

    public CheckedNegate(CommonExpression arg) {
        super(arg);
    }

    @Override
    protected boolean overflowWillHappen(int x) {
        return x == Integer.MIN_VALUE;
    }

    @Override
    public int applyFor(int x) {
        return -x;
    }

    @Override
    public String getOperator() {
        return "-";
    }

    @Override
    public PrioritiesPattern getPriority() {
        return PrioritiesPattern.MINUS;
    }
}
