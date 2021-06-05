package expression.exceptions;

import expression.CommonExpression;
import expression.PrioritiesPattern;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class CheckedSubtract extends CheckedBinaryOperation {
    public CheckedSubtract(final CommonExpression x, final CommonExpression y) {
        super(x, y);
    }

    @Override
    public PrioritiesPattern getPriority() {
        return PrioritiesPattern.ADD;
    }

    @Override
    protected String getOperator() {
        return "-";
    }

    @Override
    protected Boolean needRightBrackets() {
        return true;
    }

    @Override
    protected boolean overflowWillHappen(int x, int y) {
        return y > 0 && !(Integer.MIN_VALUE + y <= x) || y < 0 && !(x <= Integer.MAX_VALUE + y);
    }

    @Override
    protected int calculate(int x, int y) {
        return x - y;
    }
}
