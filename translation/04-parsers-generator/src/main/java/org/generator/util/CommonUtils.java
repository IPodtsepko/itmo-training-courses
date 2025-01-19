package org.generator.util;

/**
 * Provides general generation utilities.
 *
 * @author Igor Podtsepko
 */
public class CommonUtils {

    /**
     * @param raw name of lexer or parser rule.
     * @return name of class for this rule.
     */
    public static String classFrom(String raw) {
        return Character.toUpperCase(raw.charAt(0)) + raw.substring(1) + "Context";
    }
}
