package expression.solvers;

import expression.solvers.exceptions.DivisionByZeroException;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class LSolver extends Solver<Long> {
    public static final Solver<Long> INSTANCE = new LSolver();

    @Override
    public Long add(Long x, Long y) {
        return x + y;
    }

    @Override
    public Long subtract(Long x, Long y) {
        return x - y;
    }

    @Override
    public Long divide(Long x, Long y) {
        if (y == 0) {
            throw new DivisionByZeroException();
        }
        return x / y;
    }

    @Override
    public Long multiple(Long x, Long y) {
        return x * y;
    }

    @Override
    public Long mod(Long x, Long y) {
        if (y == 0) {
            throw new DivisionByZeroException();
        }
        return x % y;
    }

    @Override
    public Long negate(Long x) {
        return -x;
    }

    @Override
    public Long abs(Long x) {
        return x > 0 ? x : -x;
    }

    @Override
    public Long square(Long x) {
        return x * x;
    }

    @Override
    public Long valueOf(int x) {
        return (long) x;
    }

    @Override
    public Long parse(String value) {
        return Long.parseLong(value);
    }
}
