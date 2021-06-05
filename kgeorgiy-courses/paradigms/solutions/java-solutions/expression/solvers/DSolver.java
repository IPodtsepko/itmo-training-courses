package expression.solvers;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class DSolver extends Solver<Double> {
    public static Solver<Double> INSTANCE = new DSolver();

    @Override
    public Double add(Double x, Double y) {
        return x + y;
    }

    @Override
    public Double subtract(Double x, Double y) {
        return x - y;
    }

    @Override
    public Double divide(Double x, Double y) {
        return x / y;
    }

    @Override
    public Double multiple(Double x, Double y) {
        return x * y;
    }

    @Override
    public Double mod(Double x, Double y) {
        return x % y;
    }

    @Override
    public Double negate(Double x) {
        return -x;
    }

    @Override
    public Double abs(Double x) {
        return x > 0 ? x : -x;
    }

    @Override
    public Double square(Double x) {
        return x * x;
    }

    @Override
    public Double valueOf(int x) {
        return (double) x;
    }

    @Override
    public Double parse(String value) {
        return Double.parseDouble(value);
    }
}
