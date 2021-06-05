package expression.binary_operators;

import expression.CommonExpression;
import expression.PrioritiesPattern;

import java.util.Objects;

public abstract class BinaryOperation implements CommonExpression {
    protected final CommonExpression leftOperand;
    protected final CommonExpression rightOperand;

    public BinaryOperation(final CommonExpression leftOperand, final CommonExpression rightOperand) {
        this.leftOperand = leftOperand;
        this.rightOperand = rightOperand;
    }

    abstract protected int calculate(int x, int y);

    abstract protected double calculate(double x, double y);

    abstract public PrioritiesPattern getPriority();

    abstract protected String getOperator();

    abstract protected Boolean needRightBrackets();

    @Override
    public int evaluate(int value) {
        return calculate(leftOperand.evaluate(value), rightOperand.evaluate(value));
    }

    @Override
    public double evaluate(double value) {
        return calculate(leftOperand.evaluate(value), rightOperand.evaluate(value));
    }

    @Override
    public int evaluate(int x, int y, int z) {
        return calculate(leftOperand.evaluate(x, y, z), rightOperand.evaluate(x, y, z));
    }

    @Override
    public void putStringTo(StringBuilder dest) {
        dest.append('(');
        leftOperand.putStringTo(dest);
        dest.append(String.format(" %s ", getOperator()));
        rightOperand.putStringTo(dest);
        dest.append(')');
    }

    @Override
    public void putMiniStringTo(StringBuilder dest, boolean inBrackets) {
        if (inBrackets) dest.append('(');

        boolean needLeft = priorityLower(leftOperand);
        leftOperand.putMiniStringTo(dest, needLeft);

        dest.append(String.format(" %s ", getOperator()));

        boolean needRight = priorityLower(rightOperand)
                || needRightBracketsAndPriorityEquals(rightOperand);
        rightOperand.putMiniStringTo(dest, needRight);

        if (inBrackets) dest.append(')');
    }

    @Override
    public String toString() {
        StringBuilder dest = new StringBuilder();
        putStringTo(dest);
        return dest.toString();
    }

    @Override
    public String toMiniString() {
        StringBuilder dest = new StringBuilder();
        putMiniStringTo(dest, false);
        return dest.toString();
    }

    private boolean priorityLower(CommonExpression x) {
        return x.getPriority().compareTo(getPriority()) < 0;
    }

    private boolean needRightBracketsAndPriorityEquals(CommonExpression x) {
        return x instanceof BinaryOperation
                && (((BinaryOperation) x).needRightBrackets() || needRightBrackets())
                && x.getPriority().equals(getPriority());
    }

    @Override
    public boolean equals(final Object other) {
        if (this == other) {
            return true;
        }
        if (other == null || getClass() != other.getClass()) {
            return false;
        }
        BinaryOperation that = (BinaryOperation) other;
        return leftOperand.equals(that.leftOperand) && rightOperand.equals(that.rightOperand);
    }

    @Override
    public int hashCode() {
        return Objects.hash(leftOperand, rightOperand, getClass());
    }
}
