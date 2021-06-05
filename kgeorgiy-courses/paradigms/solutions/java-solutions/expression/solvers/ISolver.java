package expression.solvers;

import expression.solvers.exceptions.DivisionByZeroException;
import expression.solvers.exceptions.OverflowException;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class ISolver extends Solver<Integer> {
    public static Solver<Integer> INSTANCE = new ISolver();

    private ISolver() {};

    private static void throwOverflowExceptionIf(boolean condition) {
        if (condition) {
            throw new OverflowException();
        }
    }

    public static boolean addOverflow(Integer x, Integer y) {
        return y < 0 && !(Integer.MIN_VALUE - y <= x) || y > 0 && !(x <= Integer.MAX_VALUE - y);
    }

    public static boolean subtractOverflow(Integer x, Integer y) {
        return y > 0 && !(Integer.MIN_VALUE + y <= x) || y < 0 && !(x <= Integer.MAX_VALUE + y);
    }

    public static boolean multipleOverflow(Integer x, Integer y) {
        if (x > 0) {
            return Integer.MIN_VALUE / x > y || Integer.MAX_VALUE / x < y;
        }
        if (y > 0) {
            return Integer.MIN_VALUE / y > x;
        }
        return x != 0 && Integer.MAX_VALUE / x > y;
    }

    public static boolean divideOverflow(Integer x, Integer y) {
        return x == Integer.MIN_VALUE && y == -1;
    }

    public static boolean negateOverflow(Integer x) {
        return x == Integer.MIN_VALUE;
    }

    @Override
    public Integer add(Integer x, Integer y) {
        throwOverflowExceptionIf(addOverflow(x, y));
        return x + y;
    }

    @Override
    public Integer subtract(Integer x, Integer y) {
        throwOverflowExceptionIf(subtractOverflow(x, y));
        return x - y;
    }

    @Override
    public Integer divide(Integer x, Integer y) {
        if (y == 0) {
            throw new DivisionByZeroException();
        }
        throwOverflowExceptionIf(divideOverflow(x, y));
        return x / y;
    }

    @Override
    public Integer multiple(Integer x, Integer y) {
        throwOverflowExceptionIf(multipleOverflow(x, y));
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
        throwOverflowExceptionIf(negateOverflow(x));
        return -x;
    }

    @Override
    public Integer abs(Integer x) {
        throwOverflowExceptionIf(negateOverflow(x));
        return x > 0 ? x : -x;
    }

    @Override
    public Integer square(Integer x) {
        throwOverflowExceptionIf(multipleOverflow(x, x));
        return x * x;
    }

    @Override
    public Integer parse(String value) {
        return Integer.parseInt(value);
    }

    @Override
    public Integer valueOf(int x) {
        return x;
    }
}
