package expression.unary_operators;

import expression.CommonExpression;

import java.util.Objects;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public abstract class UnaryOperator implements CommonExpression {
    protected final CommonExpression arg;

    protected UnaryOperator(CommonExpression arg) {
        this.arg = arg;
    }

    abstract public int applyFor(int x);

    abstract public double applyFor(double x);

    abstract public String getOperator();

    @Override
    public int evaluate(int x) {
        return applyFor(arg.evaluate(x));
    }

    @Override
    public double evaluate(double x) {
        return applyFor(arg.evaluate(x));
    }

    @Override
    public int evaluate(int x, int y, int z) {
        return applyFor(arg.evaluate(x, y, z));
    }

    @Override
    public void putStringTo(StringBuilder dest) {
        dest.append(getOperator());
        dest.append(String.format("(%s)", arg.toString()));
    }

    @Override
    public void putMiniStringTo(StringBuilder dest, boolean inBrackets) {
        if (inBrackets) dest.append('(');
        if (arg.getPriority().compareTo(getPriority()) < 0) {
            putStringTo(dest);
        } else {
            dest.append(getOperator());
            dest.append(arg.toString());
        }
        dest.append(')');
    }

    @Override
    public String toString() {
        StringBuilder dest = new StringBuilder();
        putStringTo(dest);
        return dest.toString();
    }

    @Override
    public String toMiniString() {
        if (arg.getPriority().compareTo(getPriority()) < 0) {
            return toString();
        }
        return String.format("%s%s", getOperator(), arg.toMiniString());
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null || getClass() != obj.getClass()) {
            return false;
        }
        UnaryOperator other = (UnaryOperator) obj;
        return arg.equals(other.arg);
    }

    @Override
    public int hashCode() {
        return Objects.hash(arg, getClass());
    }
}
