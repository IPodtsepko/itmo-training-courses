package expression.operators.binary;

import expression.CommonExpression;
import expression.solvers.Solver;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class Divide extends BinaryOperator {


    public Divide(CommonExpression left, CommonExpression right) {
        super(left, right);
    }

    @Override
    public int getPriority() {
        return 2 * (1 << 10);
    }

    @Override
    protected String getOperator() {
        return "/";
    }

    @Override
    protected boolean needRightBrackets() {
        return true;
    }

    @Override
    public <T> T solve(T left, T right, Solver<T> solver) {
        return solver.divide(left, right);
    }
}
