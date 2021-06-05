package md2html.markup;

import java.util.List;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class Code extends FormattedContainer implements InlineFormattedElement {
    public Code() {
        super();
    }

    public Code(List<InlineFormattedElement> elements) {
        super(elements);
    }

    @Override
    public void toMarkdown(StringBuilder dest) {
        toMarkdown(dest, "`");
    }

    @Override
    public void toHtml(StringBuilder dest) {
        toHtml(dest, "<code>", "</code>");
    }
}
