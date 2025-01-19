package org.cxx.highlighter;

import java.util.List;

public interface Theme {
    public static class Style {
        String fontFamily = null;
        String fontStile = null;
        String fontSize = null;
        String color = null;
        String backgroundColor = null;

        public Style setFontFamily(final String fontFamily) {
            this.fontFamily = fontFamily;
            return this;
        }

        public Style setFontStile(final String fontStile) {
            this.fontStile = fontStile;
            return this;
        }

        public Style setFontSize(final String fontSize) {
            this.fontSize = fontSize;
            return this;
        }

        public Style setColor(final String color) {
            this.color = color;
            return this;
        }

        public Style setBackgroundColor(final String backgroundColor) {
            this.backgroundColor = backgroundColor;
            return this;
        }

        public String build() {

            StringBuilder builder = new StringBuilder();
            if (fontFamily != null) {
                builder.append("font-family: ").append(fontFamily).append(';');
            }
            if (fontStile != null) {
                builder.append("font-style: ").append(fontStile).append(';');
            }
            if (fontSize != null) {
                builder.append("font-size: ").append(fontSize).append(';');
            }
            if (color != null) {
                builder.append("color: ").append(color).append(';');
            }
            if (backgroundColor != null) {
                builder.append("background-color: ").append(backgroundColor).append(';');
            }
            return builder.toString();
        }

    }

    /**
     * @return URL for imports.
     */
    List<String> imports();

    /**
     * @return CSS description for common style ("body" tag).
     */
    String commonStyle();

    /**
     * @return CSS description for keywords style (like "return" or "nullptr").
     */
    String keywordsStyle();

    /**
     * @return CSS description for strings style (all in quotes).
     */
    String stringsStyle();

    /**
     * @return CSS description for number literals style.
     */
    String literalsStyle();

    /**
     * @return CSS description for members style.
     */
    String membersStyle();

    /**
     * @return CSS description for function name in function definitions style.
     */
    String functionDefinitionStyle();

    /**
     * @return CSS description for operators style.
     */
    String operatorsStyle();

    /**
     * @return CSS description for numeration style.
     */
    String numerationStyle();

    /**
     * @return CSS description for marco rules style.
     */
    String macroStyle();

    /**
     * @return CSS description for uppercase identifiers style.
     */
    String constantStyle();
}
