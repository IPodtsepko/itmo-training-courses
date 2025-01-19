package org.generator.util;

/**
 * Contains basic information about the rule for the lexer, actually a named regular expression.
 *
 * @author Igor Podtsepko
 */
public class LexerRule implements GrammarItem {
    private String name;
    private String regularExpression;

    public String getName() {
        return name;
    }

    public void setName(final String name) {
        this.name = name;
    }

    public String getRegularExpression() {
        return regularExpression;
    }

    public void setRegularExpression(final String regularExpression) {
        this.regularExpression = regularExpression;
    }
}
