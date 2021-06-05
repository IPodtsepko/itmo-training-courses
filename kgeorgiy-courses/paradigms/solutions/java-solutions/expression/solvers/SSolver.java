package expression.solvers;

import expression.solvers.exceptions.DivisionByZeroException;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class SSolver extends Solver<Short> {
    public static final Solver<Short> INSTANCE = new SSolver();

    @Override
    public Short add(Short x, Short y) {
        return (short) (x + y);
    }

    @Override
    public Short subtract(Short x, Short y) {
        return (short) (x - y);
    }

    @Override
    public Short divide(Short x, Short y) {
        if (y == 0) {
            throw new DivisionByZeroException();
        }
        return (short) (x / y);
    }

    @Override
    public Short multiple(Short x, Short y) {
        return (short) (x * y);
    }

    @Override
    public Short mod(Short x, Short y) {
        if (y == 0) {
            throw new DivisionByZeroException();
        }
        return (short) (x % y);
    }

    @Override
    public Short negate(Short x) {
        return (short) (-x);
    }

    @Override
    public Short abs(Short x) {
        return (short) (x > 0 ? x : -x);
    }

    @Override
    public Short square(Short x) {
        return (short) (x * x);
    }

    @Override
    public Short valueOf(int x) {
        return (short) x;
    }

    @Override
    public Short parse(String value) {
        return Short.parseShort(value);
    }
}
