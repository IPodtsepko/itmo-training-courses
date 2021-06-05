package expression;

import expression.solvers.Solver;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class Const extends CommonExpression {
    private final String value;

    public Const(String value) {
        this.value = value;
    }

    @Override
    public <T> T evaluate(T x, Solver<T> solver) {
        return solver.parse(value);
    }

    @Override
    public <T> T evaluate(T x, T y, T z, Solver<T> solver) {
        return solver.parse(value);
    }

    @Override
    public void putStringTo(StringBuilder dest) {
        dest.append(value);
    }

    @Override
    public void putMiniStringTo(StringBuilder dest, boolean inBrackets) {
        dest.append(value);
    }

    @Override
    public int getPriority() {
        return Integer.MAX_VALUE;
    }
}
