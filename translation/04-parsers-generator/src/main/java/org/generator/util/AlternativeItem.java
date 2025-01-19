package org.generator.util;

/**
 * Represents an alternative element from a parser rule:
 * another parser rule, a lexer rule, or a code insertion.
 *
 * @author Igor Podtsepko
 */
public sealed interface AlternativeItem permits CodeInsertion, RuleItem {
    /**
     * @return Generated code to generateTranslator a parser of this alternative element.
     */
    default String generated() {
        return this.toString();
    }

    /**
     * Performs the up conversion from {@link AlternativeItem} to {@link CodeInsertion}.
     *
     * @param item Alternative item created as {@link CodeInsertion}.
     * @return <code>item</code> as {@link CodeInsertion}.
     */
    static CodeInsertion asCodeInsertion(final AlternativeItem item) {
        return (CodeInsertion) item;
    }

    /**
     * Performs the up conversion from {@link AlternativeItem} to {@link CodeInsertion}.
     *
     * @param item Alternative item created as {@link RuleItem}.
     * @return <code>item</code> as {@link RuleItem}
     */
    static RuleItem asRuleItem(final AlternativeItem item) {
        return (RuleItem) item;
    }

    /**
     * @return <code>true</code> if this alternative item created as {@link CodeInsertion}.
     */
    default boolean isCodeInsertion() {
        return false;
    }

    /**
     * @return <code>true</code> if this alternative item created as {@link RuleItem}.
     */
    default boolean isRuleItem() {
        return false;
    }

}
