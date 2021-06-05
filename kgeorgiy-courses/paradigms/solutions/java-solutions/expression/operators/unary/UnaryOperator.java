package expression.operators.unary;

import expression.CommonExpression;
import expression.solvers.Solver;

import java.util.Objects;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public abstract class UnaryOperator extends CommonExpression {
    protected final CommonExpression arg;

    public UnaryOperator(final CommonExpression arg) {
        this.arg = arg;
    }

    public abstract <T> T solve(T x, Solver<T> solver);

    public abstract String getOperator();

    @Override
    public int getPriority() {
        return 0;
    }

    @Override
    public <T> T evaluate(T x, Solver<T> solver) {
        T evaluated = this.arg.evaluate(x, solver);
        return solve(evaluated, solver);
    }

    @Override
    public <T> T evaluate(T x, T y, T z, Solver<T> solver) {
        T evaluated = this.arg.evaluate(x, y, z, solver);
        return solve(evaluated, solver);
    }

    @Override
    public void putStringTo(StringBuilder dest) {
        dest.append(getOperator()).append('(');
        arg.putStringTo(dest);
        dest.append(')');
    }

    @Override
    public void putMiniStringTo(StringBuilder dest, boolean inBrackets) {
        if (inBrackets) {
            dest.append('(');
        }

        dest.append(getOperator());
        arg.putMiniStringTo(dest, arg.getPriority() < getPriority());

        if (inBrackets) {
            dest.append(')');
        }
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null || getClass() != obj.getClass()) {
            return false;
        }
        UnaryOperator other = (UnaryOperator) obj;
        return arg.equals(other.arg);
    }

    @Override
    public int hashCode() {
        return Objects.hash(arg, getClass());
    }
}
