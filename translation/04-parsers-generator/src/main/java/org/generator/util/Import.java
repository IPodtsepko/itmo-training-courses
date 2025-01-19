package org.generator.util;

/**
 * Contains the data of the import directive from the grammar.
 *
 * @author Igor Podtsepko
 */
public class Import implements GrammarItem {
    public String target;

    public Import(final String target) {
        this.target = target;
    }

    @Override
    public String toString() {
        return target;
    }
}
