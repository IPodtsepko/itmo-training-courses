package org.generator.util;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Contains basic information about the rule for the parser.
 *
 * @author Igor Podtsepko
 */
public class ParserRule implements GrammarItem {
    public String name;

    public List<Attribute> inherited = new ArrayList<>();
    public List<Attribute> synthesized = new ArrayList<>();

    public List<Alternative> alternatives = new ArrayList<>();

    @Override
    public String generated() {
        return String.format("""
                %s

                %s
                """, container(), parsingFunction());
    }

    private String container() {
        final String className = CommonUtils.classFrom(name);
        return String.format("""
                    /**
                     * A class containing information for the rule "%s".
                     */
                    public static class %s extends Node {
                        public %s(String name) {
                            super(name);
                        }

                        /// Synthesized attributes:
                %s

                        /// Inherited attributes:
                %s
                    }
                """, name, className, className, declareFields(synthesized), declareFields(inherited));
    }

    private String declareFields(final List<Attribute> attributes) {
        return attributes
                .stream()
                .map(attribute -> String.format("        public %s;", attribute))
                .collect(Collectors.joining("\n"));
    }

    private String parsingFunction() {
        final String arguments = inherited.stream().map(Attribute::toString).collect(Collectors.joining(",\n"));
        return String.format("""
                            /**
                             * A function that performs parsing according to the rule "%s".
                             */
                            public %s %s(%s) throws ParseException {
                        %s
                            }
                        """,
                name, CommonUtils.classFrom(name), name, arguments, body()
        );
    }

    private String body() {
        return String.format("""
                                // Attributes declarations
                        %s
                                // Variables from rules
                        %s
                                var _localctx = new %s("%s");

                                switch (curToken.token) {
                        %s
                                }
                                // Wrapping inherited attributes
                        %s
                                // Wrapping synthesized attributes
                        %s

                                return _localctx;
                        """,
                synthesizedVariables(),
                allItems(),
                CommonUtils.classFrom(name), name,
                rules(),
                wrapAttributes(inherited),
                wrapAttributes(synthesized));
    }

    private String synthesizedVariables() {
        return synthesized
                .stream()
                .map(attribute -> String.format("        %s = null;", attribute))
                .collect(Collectors.joining(System.lineSeparator()));
    }

    private String allItems() {
        return alternatives
                .stream()
                .map(Alternative::getItems)
                .flatMap(Collection::stream)
                .filter(AlternativeItem::isRuleItem)
                .map(AlternativeItem::asRuleItem)
                .map(RuleItem::declaration)
                .collect(Collectors.toSet())
                .stream()
                .collect(Collectors.joining(System.lineSeparator()));
    }

    private String rules() {
        StringBuilder rules = new StringBuilder();
        final String ident = "            ";
        for (final Alternative alternative : alternatives) {
            if (alternative.getFollow1().isEmpty()) {
                continue;
            }
            final String expected = String.join(", ", alternative.getFollow1());
            rules.append(ident).append("case ").append(expected).append(" -> {%n");
            alternative.getItems().forEach(item -> {
                if (item instanceof LexerRuleCall && ((LexerRuleCall) item).name.equals("Epsilon")) {
                    rules.append(ident).append("    _localctx.children.add(new Node(\"Epsilon\"));%n");
                    return;
                }
                rules.append(item.generated()).append("%n");
            });
            rules.append(ident).append("}%n");
        }
        rules.append(ident).append("default -> throw new ParseException(\"Unexpected token\", lexer.position());");
        return String.format(rules.toString());
    }

    private String wrapAttributes(List<Attribute> attributes) {
        return attributes
                .stream()
                .map(Attribute::getName)
                .map(name -> String.format("        _localctx.%s = %s;", name, name))
                .collect(Collectors.joining(System.lineSeparator()));
    }
}
