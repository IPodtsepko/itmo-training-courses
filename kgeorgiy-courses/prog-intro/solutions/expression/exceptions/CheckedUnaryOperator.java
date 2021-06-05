package expression.exceptions;

import expression.CommonExpression;
import expression.exceptions.expression_exceptions.OverflowException;
import expression.unary_operators.UnaryOperator;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public abstract class CheckedUnaryOperator extends UnaryOperator {
    protected CheckedUnaryOperator(CommonExpression arg) {
        super(arg);
    }

    abstract protected boolean overflowWillHappen(int x);

    private int checkedApplyFor(int x) {
        if (overflowWillHappen(x)) {
            throw new OverflowException();
        }
        return applyFor(x);
    }

    @Override
    public int evaluate(int x) {
        return checkedApplyFor(arg.evaluate(x));
    }

    @Override
    public int evaluate(int x, int y, int z) {
        return checkedApplyFor(arg.evaluate(x, y, z));
    }

    @Override
    public double applyFor(double x) {
        throw new UnsupportedOperationException();
    }
}
