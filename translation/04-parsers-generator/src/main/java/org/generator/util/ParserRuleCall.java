package org.generator.util;

import java.util.ArrayList;
import java.util.List;

/**
 * Contains data for "calling" a parser rule from an alternative to another parser rule.
 *
 * @author Igor Podtsepko
 */
public final class ParserRuleCall extends RuleItem {
    private final List<String> arguments = new ArrayList<>();

    @Override
    public String getType() {
        return CommonUtils.classFrom(name);
    }

    @Override
    public String generated() {
        return String.format("""
                                        // Start of processing parsing rule
                                        %s = %s(%s);
                                        _localctx.children.add(%s);
                                        // End of processing parsing rule
                        """,
                name, name, String.join(", ", arguments), name
        );
    }

    @Override
    public String toString() {
        return name;
    }

    public ParserRuleCall(final String name) {
        this.name = name;
    }

    public void addArgument(final String arg) {
        arguments.add(arg);
    }
}
