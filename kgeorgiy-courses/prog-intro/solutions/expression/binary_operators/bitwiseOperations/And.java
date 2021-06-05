package expression.binary_operators.bitwiseOperations;

import expression.CommonExpression;
import expression.PrioritiesPattern;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class And extends BitwiseOperation {
    public And(CommonExpression x, CommonExpression y) {
        super(x, y);
    }

    @Override
    protected int calculate(int x, int y) {
        return x & y;
    }

    @Override
    protected String getOperator() {
        return "&";
    }

    @Override
    public PrioritiesPattern getPriority() {
        return PrioritiesPattern.AND;
    }
}
