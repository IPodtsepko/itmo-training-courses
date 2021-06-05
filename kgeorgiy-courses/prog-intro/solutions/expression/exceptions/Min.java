package expression.exceptions;

import expression.CommonExpression;
import expression.PrioritiesPattern;
import expression.binary_operators.BinaryOperation;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class Min extends BinaryOperation {
    public Min(CommonExpression leftOperand, CommonExpression rightOperand) {
        super(leftOperand, rightOperand);
    }

    @Override
    protected int calculate(int x, int y) {
        return x < y ? x : y;
    }

    @Override
    protected double calculate(double x, double y) {
        return x < y ? x : y;
    }

    @Override
    public PrioritiesPattern getPriority() {
        return PrioritiesPattern.COMPARATOR;
    }

    @Override
    protected String getOperator() {
        return "min";
    }

    @Override
    protected Boolean needRightBrackets() {
        return false;
    }
}
