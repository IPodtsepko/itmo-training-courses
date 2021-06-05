package expression.operators.binary;

import expression.CommonExpression;
import expression.solvers.Solver;

import java.util.Objects;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public abstract class BinaryOperator extends CommonExpression {
    protected final CommonExpression left, right;

    public BinaryOperator(final CommonExpression left, final CommonExpression right) {
        this.left = left;
        this.right = right;
    }

    protected abstract String getOperator();

    protected abstract boolean needRightBrackets();

    @Override
    public <T> T evaluate(T x, Solver<T> solver) {
        T evaluatedLeft = this.left.evaluate(x, solver);
        T evaluatedRight = this.right.evaluate(x, solver);
        return solve(evaluatedLeft, evaluatedRight, solver);
    }

    @Override
    public <T> T evaluate(T x, T y, T z, Solver<T> solver) {
        T evaluatedLeft = this.left.evaluate(x, y, z, solver);
        T evaluatedRight = this.right.evaluate(x, y, z, solver);
        return solve(evaluatedLeft, evaluatedRight, solver);
    }

    public abstract <T> T solve(T left, T right, Solver<T> solver);

    @Override
    public void putStringTo(StringBuilder dest) {
        dest.append('(');
        left.putStringTo(dest);
        dest.append(" ").append(getOperator()).append(" ");
        right.putStringTo(dest);
        dest.append(')');
    }

    @Override
    public void putMiniStringTo(StringBuilder dest, boolean inBrackets) {
        if (inBrackets) dest.append('(');

        boolean needLeft = priorityLower(left);
        left.putMiniStringTo(dest, needLeft);

        dest.append(" ").append(getOperator()).append(" ");

        boolean needRight = priorityLower(right) || needRightBracketsAndPriorityEquals(right);
        right.putMiniStringTo(dest, needRight);

        if (inBrackets) dest.append(')');
    }

    private boolean priorityLower(CommonExpression x) {
        return x.getPriority() < getPriority();
    }

    private boolean needRightBracketsAndPriorityEquals(CommonExpression x) {
        return x instanceof BinaryOperator
                && (((BinaryOperator) x).needRightBrackets() || needRightBrackets())
                && x.getPriority() == getPriority();
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (other == null || getClass() != other.getClass()) {
            return false;
        }
        BinaryOperator that = (BinaryOperator) other;
        return this.left.equals(that.left) && this.right.equals(that.right);
    }

    @Override
    public int hashCode() {
        return Objects.hash(left, right, getClass());
    }
}
