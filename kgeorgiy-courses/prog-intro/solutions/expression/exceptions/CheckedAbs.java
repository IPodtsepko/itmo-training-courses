package expression.exceptions;

import expression.CommonExpression;
import expression.PrioritiesPattern;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class CheckedAbs extends CheckedUnaryOperator {
    protected CheckedAbs(CommonExpression arg) {
        super(arg);
    }

    @Override
    protected boolean overflowWillHappen(int x) {
        return x == Integer.MIN_VALUE;
    }

    @Override
    public int applyFor(int x) {
        return x > 0? x : -x;
    }

    @Override
    public String getOperator() {
        return "abs";
    }

    @Override
    public PrioritiesPattern getPriority() {
        return PrioritiesPattern.MINUS;
    }
}
