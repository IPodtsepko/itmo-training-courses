package expression.operators.binary;

import expression.CommonExpression;
import expression.solvers.Solver;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class Multiple extends BinaryOperator {


    public Multiple(CommonExpression left, CommonExpression right) {
        super(left, right);
    }

    @Override
    public int getPriority() {
        return 2 * (1 << 10);
    }

    @Override
    protected String getOperator() {
        return "*";
    }

    @Override
    protected boolean needRightBrackets() {
        return false;
    }

    @Override
    public <T> T solve(T left, T right, Solver<T> solver) {
        return solver.multiple(left, right);
    }
}
