package expression;

import expression.interfaces.DoubleExpression;
import expression.interfaces.Expression;
import expression.interfaces.TripleExpression;
import expression.solvers.DSolver;
import expression.solvers.ISolver;
import expression.solvers.Solver;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public abstract class CommonExpression implements Expression, DoubleExpression, TripleExpression {
    public abstract <T> T evaluate(T x, Solver<T> solver);

    public abstract <T> T evaluate(T x, T y, T z, Solver<T> solver);

    public abstract void putStringTo(StringBuilder dest);

    public abstract void putMiniStringTo(StringBuilder dest, boolean inBrackets);

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

    public abstract int getPriority();

    @Override
    public int evaluate(int x) {
        return evaluate(x, ISolver.INSTANCE);
    }

    @Override
    public double evaluate(double x) {
        return evaluate(x, DSolver.INSTANCE);
    }

    @Override
    public int evaluate(int x, int y, int z) {
        return evaluate(x, y, z, ISolver.INSTANCE);
    }
}
