package expression.unary_operators;

import expression.CommonExpression;
import expression.PrioritiesPattern;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class Minus extends UnaryOperator {

    public Minus(CommonExpression arg) {
        super(arg);
    }

    @Override
    public int applyFor(int x) {
        return -x;
    }

    @Override
    public double applyFor(double x) {
        return -x;
    }

    @Override
    public String getOperator() {
        return "-";
    }

    @Override
    public PrioritiesPattern getPriority() {
        return PrioritiesPattern.MINUS;
    }
}
