package md2html.markup;

import java.util.List;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class Strikeout extends FormattedContainer implements InlineFormattedElement {
    public Strikeout() {
        super();
    }
    public Strikeout(List<InlineFormattedElement> elements) {
        super(elements);
    }

    @Override
    public void toMarkdown(StringBuilder dest) {
        toMarkdown(dest, "~");
    }

    @Override
    public void toHtml(StringBuilder dest) {
        toHtml(dest, "<s>", "</s>");
    }
}
