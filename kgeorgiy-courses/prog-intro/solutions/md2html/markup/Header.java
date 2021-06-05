package md2html.markup;

import java.util.List;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class Header extends FormattedContainer implements BlockFormattedElement {
    private final int level;

    public Header(int level, List<InlineFormattedElement> elements) {
        super(elements);
        this.level = level;
    }

    @Override
    public void toMarkdown(StringBuilder dest) {
        dest.append("#".repeat(level));
        toMarkdown(dest, "");
        dest.append('\n');
    }

    @Override
    public void toHtml(StringBuilder dest) {
        toHtml(dest, "<h" + level + ">", "</h" + level + ">\n");
    }
}
