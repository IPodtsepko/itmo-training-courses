package expression;

import expression.solvers.Solver;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class Variable extends CommonExpression {
    private final String name;

    public Variable(final String name) {
        this.name = name;
    }

    @Override
    public <T> T evaluate(T x, Solver<T> solver) {
        return x;
    }

    @Override
    public <T> T evaluate(T x, T y, T z, Solver<T> solver) {
        if (name.equals("x")) {
            return x;
        } else if (name.equals("y")) {
            return y;
        } else {
            return z;
        }
    }

    @Override
    public void putStringTo(StringBuilder dest) {
        dest.append(name);
    }

    @Override
    public void putMiniStringTo(StringBuilder dest, boolean inBrackets) {
        dest.append(name);
    }

    @Override
    public int getPriority() {
        return Integer.MAX_VALUE;
    }
}
