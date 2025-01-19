package org.generator.util;

/**
 * Denotes any grammar element.
 *
 * @author Igor Podtsepko
 */
public interface GrammarItem {
    default String generated() {
        return toString();
    }
}
