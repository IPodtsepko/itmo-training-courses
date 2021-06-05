package expression.binary_operators.bitwiseOperations;

import expression.CommonExpression;
import expression.binary_operators.BinaryOperation;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public abstract class BitwiseOperation extends BinaryOperation {
    public BitwiseOperation(CommonExpression x, CommonExpression y) {
        super(x, y);
    }

    @Override
    protected double calculate(double x, double y) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected Boolean needRightBrackets() {
        return false;
    }
}
