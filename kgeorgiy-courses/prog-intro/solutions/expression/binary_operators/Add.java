package expression.binary_operators;

import expression.CommonExpression;
import expression.PrioritiesPattern;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class Add extends BinaryOperation {

    public Add(CommonExpression x, CommonExpression y) {
        super(x, y);
    }

    @Override
    protected int calculate(int x, int y) {
        return x + y;
    }

    @Override
    protected double calculate(double x, double y) {
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
}
