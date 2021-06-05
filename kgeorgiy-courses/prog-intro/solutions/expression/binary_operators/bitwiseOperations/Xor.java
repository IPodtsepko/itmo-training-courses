package expression.binary_operators.bitwiseOperations;

import expression.CommonExpression;
import expression.PrioritiesPattern;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class Xor extends BitwiseOperation {
    public Xor(CommonExpression x, CommonExpression y) {
        super(x, y);
    }

    @Override
    protected int calculate(int x, int y) {
        return x ^ y;
    }

    @Override
    protected String getOperator() {
        return "^";
    }

    @Override
    public PrioritiesPattern getPriority() {
        return PrioritiesPattern.XOR;
    }
}
