package org.generator.util;

/**
 * Contains data for working with code insertions.
 *
 * @author Igor Podtsepko
 */
public final class CodeInsertion implements AlternativeItem {
    public String code;

    public CodeInsertion(String code) {
        this.code = code;
    }

    @Override
    public boolean isCodeInsertion() {
        return true;
    }

    @Override
    public String toString() {
        return String.format("""
                                        // Start of code insertion
                                        %s;
                                        // End of code insertion
                        """,
                this.code
        );
    }

}
