package expression.exceptions;

import expression.CommonExpression;
import expression.PrioritiesPattern;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class CheckedMultiply extends CheckedBinaryOperation {
    public CheckedMultiply(CommonExpression x, CommonExpression y) {
        super(x, y);
    }

    @Override
    public PrioritiesPattern getPriority() {
        return PrioritiesPattern.MULTIPLE;
    }

    @Override
    protected String getOperator() {
        return "*";
    }

    @Override
    protected int calculate(int x, int y) {
        return x * y;
    }

    @Override
    protected Boolean needRightBrackets() {
        return false;
    }

    @Override
    protected boolean overflowWillHappen(int x, int y) {
        if (x == 0 || y == 0) {
            return false;
        }
        int multiple = x * y;
        return multiple / x != y || multiple / y != x;
    }
}
