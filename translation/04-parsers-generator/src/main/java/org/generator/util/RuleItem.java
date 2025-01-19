package org.generator.util;

/**
 * Denotes a named alternative element inside the parser rule that is not a code insert.
 *
 * @author Igor Podtsepko
 */
public abstract sealed class RuleItem implements AlternativeItem permits ParserRuleCall, LexerRuleCall {
    public String name = null;

    @Override
    public boolean isRuleItem() {
        return true;
    }

    public abstract String getType();

    public String declaration() {
        return String.format("        %s %s = null;", getType(), name);
    }
}
