package md2html.markup;

import java.util.List;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class Emphasis extends FormattedContainer implements InlineFormattedElement {
    public Emphasis() {
        super();
    }

    public Emphasis(List<InlineFormattedElement> elements) {
        super(elements);
    }

    @Override
    public void toMarkdown(StringBuilder dest) {
        toMarkdown(dest, "*");
    }

    @Override
    public void toHtml(StringBuilder dest) {
        toHtml(dest, "<em>", "</em>");
    }
}
