package expression.exceptions;

import expression.CommonExpression;
import expression.binary_operators.BinaryOperation;
import expression.exceptions.expression_exceptions.OverflowException;

public abstract class CheckedBinaryOperation extends BinaryOperation {
    public CheckedBinaryOperation(final CommonExpression leftOperand, final CommonExpression rightOperand) {
        super(leftOperand, rightOperand);
    }

    @Override
    protected double calculate(double x, double y) {
        throw new UnsupportedOperationException();
    }

    abstract protected boolean overflowWillHappen(int x, int y);

    public int checkedCalculate(int x, int y) {
        if (overflowWillHappen(x, y)) {
            throw new OverflowException();
        }
        return calculate(x, y);
    }

    @Override
    public int evaluate(int value) {
        return checkedCalculate(leftOperand.evaluate(value), leftOperand.evaluate(value));
    }

    @Override
    public int evaluate(int x, int y, int z) {
        return checkedCalculate(leftOperand.evaluate(x, y, z), rightOperand.evaluate(x, y, z));
    }

    @Override
    public double evaluate(double x) {
        throw new UnsupportedOperationException();
    }
}
