package org.generator;

import org.generator.util.LexerRule;

import java.util.List;
import java.util.stream.Collectors;

/**
 * The class responsible for generating the <code>Token</code> enumeration to the
 * set of rules for the lexer listed in the grammar description.
 *
 * @author Igor Podtsepko
 */
public class TokenGenerator {
    public static String generate(final String packageName, final List<LexerRule> lexerRules) {
        return String.format(
                """
                package %s;
                
                /**
                 * A class containing the necessary information about the token.
                 * <p>
                 * Note: This file is generated automatically.
                 * </p>
                 */
                public class Token {
                    public String text;
                    public Type token;

                    public Token(Type token, String text) {
                        this.text = text;
                        this.token = token;
                    }

                    enum Type {
                        %s,
                        END,
                        Epsilon
                    }

                    @Override
                    public String toString() {
                        return text;
                    }
                }
                """,
                packageName,
                lexerRules.stream()
                        .filter(rule -> !rule.getName().equals("Skip") && !rule.getName().equals("Epsilon"))
                        .map(LexerRule::getName)
                        .collect(Collectors.joining("," + System.lineSeparator() + "        ")));
    }
}
