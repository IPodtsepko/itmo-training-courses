package expression;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public abstract class Value implements CommonExpression {
    @Override
    public void putStringTo(StringBuilder dest) {
        dest.append(toString());
    }

    @Override
    public void putMiniStringTo(StringBuilder dest, boolean inBrackets) {
        dest.append(toString());
    }

    @Override
    public String toMiniString() {
        return toString();
    }

    @Override
    public PrioritiesPattern getPriority() {
        return PrioritiesPattern.VALUE;
    }
}
