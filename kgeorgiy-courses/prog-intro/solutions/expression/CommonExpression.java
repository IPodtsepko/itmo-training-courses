package expression;


public interface CommonExpression extends Expression, DoubleExpression, TripleExpression {
    void putStringTo(StringBuilder dest);

    void putMiniStringTo(StringBuilder dest, boolean inBrackets);

    PrioritiesPattern getPriority();
}
