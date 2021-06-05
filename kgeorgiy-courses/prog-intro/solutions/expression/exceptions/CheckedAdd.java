package expression.exceptions;

import expression.CommonExpression;
import expression.PrioritiesPattern;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class CheckedAdd extends CheckedBinaryOperation {
    public CheckedAdd(CommonExpression x, CommonExpression y) {
        super(x, y);
    }

    @Override
    protected int calculate(int x, int y) {
        return x + y;
    }

    @Override
    public PrioritiesPattern getPriority() {
        return PrioritiesPattern.ADD;
    }

    @Override
    protected String getOperator() {
        return "+";
    }

    @Override
    protected Boolean needRightBrackets() {
        return false;
    }

    @Override
    protected boolean overflowWillHappen(int x, int y) {
        return y < 0 && !(Integer.MIN_VALUE - y <= x) || y > 0 && !(x <= Integer.MAX_VALUE - y);
    }
}
