package expression.binary_operators;

import expression.CommonExpression;
import expression.PrioritiesPattern;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class Subtract extends BinaryOperation {

    public Subtract(final CommonExpression x, final CommonExpression y) {
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
    protected double calculate(double x, double y) {
        return x - y;
    }

    @Override
    protected Boolean needRightBrackets() {
        return true;
    }

    @Override
    protected int calculate(int x, int y) {
        return x - y;
    }
}
