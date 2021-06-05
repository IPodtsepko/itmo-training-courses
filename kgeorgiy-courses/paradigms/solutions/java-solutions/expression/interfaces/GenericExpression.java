package expression.interfaces;

import expression.solvers.Solver;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public interface GenericExpression<T extends Number> {
    public T evaluate(T x, T y, T z, Solver<T> solver);
}
