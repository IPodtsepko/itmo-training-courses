package expression.solvers;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public abstract class Solver<T> {
    public abstract T add(T x, T y);

    public abstract T subtract(T x, T y);

    public abstract T divide(T x, T y);

    public abstract T multiple(T x, T y);

    public abstract T mod(T x, T y);

    public abstract T negate(T x);

    public abstract T abs(T x);

    public abstract T square(T x);

    public abstract T valueOf(int x);

    public abstract T parse(String value);
}
