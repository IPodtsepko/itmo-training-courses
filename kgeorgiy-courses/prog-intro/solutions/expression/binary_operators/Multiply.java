package expression.binary_operators;

import expression.CommonExpression;
import expression.PrioritiesPattern;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class Multiply extends BinaryOperation {

    public Multiply(CommonExpression x, CommonExpression y) {
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
    protected double calculate(double x, double y) {
        return x * y;
    }

    @Override
    protected Boolean needRightBrackets() {
        return false;
    }
}
