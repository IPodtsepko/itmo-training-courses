package org.generator;

import org.generator.util.LexerRule;

import java.util.List;
import java.util.stream.Collectors;

/**
 * The class responsible for generating the lexer according to the
 * set of rules for the lexer listed in the grammar description.
 *
 * @author Igor Podtsepko
 */
public class LexerGenerator {
    public static String generate(final String packageName, final List<LexerRule> lexerRules) {
        return String.format("""
                        package %s;

                        import java.util.regex.*;
                        import java.util.Map;
                        import java.util.Iterator;

                        public class Lexer implements Iterator<Token>, Iterable<Token> {
                            private final Map<Token.Type, Pattern> patterns = Map.ofEntries(
                        %s
                            );

                            private final Pattern skip = Pattern.compile("[%s]+");
                            private final Matcher matcher = skip.matcher("");

                            private int begin = 0;
                            private int end = 0;
                            private boolean hasNext = true;

                            private final String text;

                            public int position() {
                                return end;
                            }

                            @Override
                            public Iterator<Token> iterator() {
                                return this;
                            }

                            public Lexer(String text) {
                                this.text = text;
                            }

                            @Override
                            public boolean hasNext() {
                                return hasNext;
                            }

                            private boolean matchLookingAt() {
                                if (matcher.lookingAt()) {
                                    begin = end;
                                    end = begin + matcher.end();
                                    matcher.reset(text.substring(end));
                                    return true;
                                }
                                return false;
                            }

                            @Override
                            public Token next() {
                                begin = end;
                                matcher.usePattern(skip);
                                matcher.reset(text.substring(begin));
                                matchLookingAt();
                                for (final Token.Type type : Token.Type.values()) {
                                    if (type == Token.Type.END || type == Token.Type.Epsilon) {
                                        continue;
                                    }
                                    matcher.usePattern(patterns.get(type));
                                    if (matchLookingAt()) {
                                        return new Token(type, text.substring(begin, end));
                                    }
                                }
                                if (end != text.length()) {
                                    throw new Error();
                                }
                                hasNext = false;
                                return new Token(Token.Type.END, null);
                            }
                        }""",
                packageName,
                rulesMapping(lexerRules),
                ruleForSkipping(lexerRules));
    }

    private static String rulesMapping(final List<LexerRule> lexerRules) {
        return lexerRules
                .stream()
                .filter(LexerGenerator::isInteresting)
                .map(LexerGenerator::generateMapEntry)
                .collect(Collectors.joining("," + System.lineSeparator()));
    }

    private static String ruleForSkipping(final List<LexerRule> lexerRules) {
        return lexerRules.stream().filter(LexerGenerator::isSkip).findAny().orElseThrow().getRegularExpression();
    }

    private static String generateMapEntry(final LexerRule lexerRule) {
        return String.format(
                "            Map.entry(Token.Type.%s, Pattern.compile(\"%s\"))",
                lexerRule.getName(),
                lexerRule.getRegularExpression()
        );
    }

    /**
     * Checks whether the rule is meaningful. That is, that the token
     * that it sets does not need to be skipped.
     *
     * @param lexerRule rule for checking.
     * @return <code>true</code> if rule is not <code>Skip</code>, otherwise - <code>false</code>.
     */
    private static boolean isInteresting(final LexerRule lexerRule) {
        return !isSkip(lexerRule);
    }

    /**
     * Checks that the token that sets the rule should be skipped.
     *
     * @param lexerRule rule for checking.
     * @return <code>true</code> if rule is <code>Skip</code>, otherwise - <code>false</code>.
     */
    private static boolean isSkip(final LexerRule lexerRule) {
        return lexerRule.getName().equals("Skip");
    }

}
