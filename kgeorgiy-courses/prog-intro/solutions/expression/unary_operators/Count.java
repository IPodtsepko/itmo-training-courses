package expression.unary_operators;

import expression.CommonExpression;
import expression.PrioritiesPattern;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class Count extends UnaryOperator {
    public Count(CommonExpression arg) {
        super(arg);
    }

    @Override
    public int applyFor(int x) {
        return Integer.bitCount(x);
    }

    @Override
    public double applyFor(double x) {
        throw new UnsupportedOperationException();
    }

    @Override
    public String getOperator() {
        return "count";
    }

    @Override
    public PrioritiesPattern getPriority() {
        return PrioritiesPattern.MINUS;
    }
}
