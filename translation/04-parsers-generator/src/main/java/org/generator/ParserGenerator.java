package org.generator;

import org.generator.util.*;

import javax.swing.text.html.parser.Parser;
import java.util.*;
import java.util.stream.Collectors;

/**
 * The class responsible for generating the parser according to the
 * set of rules for the parser listed in the grammar description.
 *
 * @author Igor Podtsepko
 */
public class ParserGenerator {

    private static class GrammarProperties {

        final List<ParserRule> parserRules;
        Map<String, Set<String>> first = new HashMap<>();
        Map<String, Set<String>> follow = new HashMap<>();

        public GrammarProperties(final List<ParserRule> parserRules) {
            this.parserRules = parserRules;
            updateFirst();
            updateFollow();
        }

        public void updateFirst() {
            first = new HashMap<>();

            for (final ParserRule parserRule : parserRules) {
                first.put(parserRule.name, new HashSet<>());
            }

            boolean anyChanges = true;
            while (anyChanges) {
                anyChanges = false;
                for (final ParserRule parserRule : parserRules) {
                    for (final Alternative alternative : parserRule.alternatives) {
                        final RuleItem cell = (RuleItem) alternative.getItems().get(0);
                        final Set<String> actualSet = first.get(parserRule.name);
                        if (cell instanceof LexerRuleCall) {
                            anyChanges |= actualSet.add(cell.name);
                        } else if (cell instanceof ParserRuleCall) {
                            final Set<String> addition = first.get(cell.name);
                            anyChanges |= actualSet.addAll(addition);
                        }
                    }
                }
            }
        }

        public void updateFollow() {
            follow = new HashMap<>();

            for (final ParserRule parserRule : parserRules) {
                follow.put(parserRule.name, new HashSet<>());
            }

            final String startNonTerminal = parserRules.get(0).name;
            follow.get(startNonTerminal).add("END");

            boolean anyChanges = true;
            while (anyChanges) {
                anyChanges = false;
                for (final ParserRule A : parserRules) {
                    for (final Alternative alpha : A.alternatives) {
                        final List<RuleItem> items = alpha.ruleItems();
                        for (int i = 0; i < items.size(); ++i) {
                            final RuleItem B = items.get(i);
                            if (B instanceof LexerRuleCall) {
                                continue; // For terminals, the set of follow is not defined
                            }
                            if (i + 1 == items.size()) { // B is last element of alpha
                                anyChanges |= follow.get(B.name).addAll(follow.get(A.name));
                                continue;
                            }
                            final RuleItem gamma = items.get(i + 1);
                            if (gamma instanceof LexerRuleCall) {
                                if (gamma.name.equals("Epsilon")) {
                                    anyChanges |= follow.get(B.name).addAll(follow.get(A.name));
                                } else {
                                    anyChanges |= follow.get(B.name).add(gamma.name);
                                }
                            } else { // This is not a terminal, so we use first and follow
                                anyChanges |= follow.get(B.name).addAll(
                                        first.get(gamma.name).stream()
                                                .filter(terminal -> !terminal.equals("Epsilon"))
                                                .toList()
                                );
                                if (first.get(gamma.name).contains("Epsilon")) {
                                    anyChanges |= follow.get(B.name).addAll(follow.get(A.name));
                                }
                            }
                        }
                    }
                }
            }
        }

        public boolean isLL1() {
            for (final ParserRule A : parserRules) {
                for (final Alternative alpha : A.alternatives) {
                    final RuleItem alphaStart = (RuleItem) alpha.getItems().get(0);
                    final Set<String> alphaFirst =
                            alphaStart instanceof LexerRuleCall ?
                                    Set.of(alphaStart.name) : new HashSet<>(first.get(alphaStart.name));
                    for (final Alternative beta : A.alternatives) {
                        if (alpha.equals(beta)) {
                            continue;
                        }
                        final RuleItem betaStart = (RuleItem) beta.getItems().get(0);
                        final Set<String> betaFirst = betaStart instanceof LexerRuleCall ?
                                Set.of(betaStart.name) : new HashSet<>(first.get(betaStart.name));
                        if (alphaFirst.stream().anyMatch(betaFirst::contains) || betaFirst.stream().anyMatch(alphaFirst::contains)) {
                            return false;
                        }
                        if (!alphaFirst.contains("Epsilon")) {
                            continue;
                        }
                        final Set<String> followA = follow.get(A.name);
                        if (followA.stream().anyMatch(betaFirst::contains) || betaFirst.stream().anyMatch(followA::contains)) {
                            return false;
                        }
                    }
                }
            }
            return true;
        }


        private void updateFirst1() {
            for (final ParserRule A : parserRules) {
                for (final Alternative alpha : A.alternatives) {
                    // A -> alpha
                    final RuleItem alphaStart = (RuleItem) alpha.getItems().get(0);
                    final Set<String> alphaFirst = new HashSet<>(
                            alphaStart instanceof LexerRuleCall ?
                                    Set.of(alphaStart.name) : first.get(alphaStart.name)
                    );
                    alpha.setFollow1(new HashSet<>(alphaFirst));
                    alpha.getFollow1().remove("Epsilon");
                    if (alphaFirst.contains("Epsilon")) {
                        alpha.getFollow1().addAll(follow.get(A.name));
                    }
                }
            }
        }
    }

    public static String generate(final String packageName,
                                  final List<Import> imports,
                                  final List<ParserRule> parserRules
    ) {
        final GrammarProperties properties = new GrammarProperties(parserRules);
        if (!properties.isLL1()) {
            throw new RuntimeException("Expected LL(1) grammar");
        }
        properties.updateFirst1();
        return String.format("""
                        package %s;

                        %s

                        import java.text.ParseException;
                        import org.generator.util.Tree;

                        public class Parser {
                            public Token curToken;
                            public final Lexer lexer;

                            public Parser(String text) {
                                lexer = new Lexer(text);
                                curToken = lexer.next();
                            }

                            public void assertToken(final Token.Type expected) throws ParseException {
                                if (expected != curToken.token) {
                                    throw new ParseException("Unexpected token at %%s", lexer.position());
                                }
                            }

                        %s

                            public static class Node extends Tree {
                                public Node(String name) {
                                    super(name);
                                }
                            }
                        }""",
                packageName,
                imports.stream()
                        .map(Import::generated)
                        .map(target -> String.format("import %s;", target))
                        .collect(Collectors.joining(System.lineSeparator())),
                parserRules.stream()
                        .map(GrammarItem::generated)
                        .collect(Collectors.joining(System.lineSeparator())));
    }
}
