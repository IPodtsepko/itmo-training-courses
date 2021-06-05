package expression.solvers;

import expression.solvers.exceptions.DivisionByZeroException;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class USolver extends Solver<Integer> {
    public static final Solver<Integer> INSTANCE = new USolver();

    @Override
    public Integer add(Integer x, Integer y) {
        return x + y;
    }

    @Override
    public Integer subtract(Integer x, Integer y) {
        return x - y;
    }

    @Override
    public Integer divide(Integer x, Integer y) {
        if (y == 0) {
            throw new DivisionByZeroException();
        }
        return x / y;
    }

    @Override
    public Integer multiple(Integer x, Integer y) {
        return x * y;
    }

    @Override
    public Integer mod(Integer x, Integer y) {
        if (y == 0) {
            throw new DivisionByZeroException();
        }
        return x % y;
    }

    @Override
    public Integer negate(Integer x) {
        return -x;
    }

    @Override
    public Integer abs(Integer x) {
        return x > 0 ? x : -x;
    }

    @Override
    public Integer square(Integer x) {
        return x * x;
    }

    @Override
    public Integer valueOf(int x) {
        return x;
    }

    @Override
    public Integer parse(String value) {
        return Integer.parseInt(value);
    }
}
