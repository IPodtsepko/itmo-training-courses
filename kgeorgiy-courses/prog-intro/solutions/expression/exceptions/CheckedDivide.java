package expression.exceptions;

import expression.CommonExpression;
import expression.PrioritiesPattern;
import expression.exceptions.expression_exceptions.DivisionByZeroException;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class CheckedDivide extends CheckedBinaryOperation {
    public CheckedDivide(CommonExpression x, CommonExpression y) {
        super(x, y);
    }

    @Override
    protected int calculate(int x, int y) {
        if (y == 0) {
            throw new DivisionByZeroException();
        }
        return x / y;
    }

    @Override
    public PrioritiesPattern getPriority() {
        return PrioritiesPattern.MULTIPLE;
    }

    @Override
    protected String getOperator() {
        return "/";
    }

    @Override
    protected Boolean needRightBrackets() {
        return true;
    }

    @Override
    protected boolean overflowWillHappen(int x, int y) {
        return x == Integer.MIN_VALUE && y == -1;
    }
}
