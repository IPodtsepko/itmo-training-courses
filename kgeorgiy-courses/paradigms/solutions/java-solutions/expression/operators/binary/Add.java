package expression.operators.binary;

import expression.CommonExpression;
import expression.solvers.Solver;

/**
 * Title task: "Homework 5. Computing in various types: generics"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class Add extends BinaryOperator {
    public Add(CommonExpression left, CommonExpression right) {
        super(left, right);
    }

    @Override
    public int getPriority() {
        return 1 << 10;
    }

    @Override
    protected String getOperator() {
        return "+";
    }

    @Override
    protected boolean needRightBrackets() {
        return false;
    }

    @Override
    public <T> T solve(T left, T right, Solver<T> solver) {
        return solver.add(left, right);
    }
}
