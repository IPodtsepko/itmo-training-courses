package org.generator.util;

/**
 * Contains data for "calling" a parser rule from an alternative to another lexer rule.
 *
 * @author Igor Podtsepko
 */
public final class LexerRuleCall extends RuleItem {
    public LexerRuleCall(String name) {
        this.name = name;
    }

    @Override
    public String getType() {
        return "Token";
    }

    @Override
    public String toString() {
        return name;
    }

    @Override
    public String generated() {
        return String.format("""
                                        // Start of processing lexer rule
                                        assertToken(Token.Type.%s);
                                        %s = curToken;
                                        _localctx.children.add(new Node("%s:" + curToken.text));
                                        curToken = lexer.next();
                                        // End of processing lexer rule
                        """,
                name, name, name);
    }

}
