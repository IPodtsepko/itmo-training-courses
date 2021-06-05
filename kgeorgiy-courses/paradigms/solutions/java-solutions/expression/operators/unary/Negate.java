package expression.operators.unary;

import expression.CommonExpression;
import expression.solvers.Solver;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class Negate extends UnaryOperator {
    public Negate(CommonExpression arg) {
        super(arg);
    }

    @Override
    public <T> T solve(T x, Solver<T> solver) {
        return solver.negate(x);
    }

    @Override
    public String getOperator() {
        return "-";
    }
}
