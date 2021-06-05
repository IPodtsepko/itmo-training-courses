package md2html.markup;

import java.util.List;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class Strong extends FormattedContainer implements InlineFormattedElement {
    public Strong() {
        super();
    }

    public Strong(List<InlineFormattedElement> elements) {
        super(elements);
    }

    @Override
    public void toMarkdown(StringBuilder dest) {
        toMarkdown(dest, "__");
    }

    @Override
    public void toHtml(StringBuilder dest) {
        toHtml(dest, "<strong>", "</strong>");
    }
}
