package md2html.markup;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public abstract class FormattedContainer extends FormattedElement {
    private final List<InlineFormattedElement> elements;

    public FormattedContainer() {
        elements = new ArrayList<>();
    }

    public FormattedContainer(List<InlineFormattedElement> elements) {
        this.elements = elements;
    }

    public void add(InlineFormattedElement newElement) {
        elements.add(newElement);
    }

    public void add(FormattedContainer newElement) {
        elements.add((InlineFormattedElement) newElement);
    }

    public void toMarkdown(StringBuilder dest, String tag) {
        toMarkdown(dest, tag, tag);
    }

    public void toMarkdown(StringBuilder dest, String openingTag, String closingTag) {
        dest.append(openingTag);
        for (Formatted element : elements) {
            element.toMarkdown(dest);
        }
        dest.append(closingTag);
    }

    public void toHtml(StringBuilder dest, String openingTag, String closingTag) {
        dest.append(openingTag);
        for (Formatted element : elements) {
            element.toHtml(dest);
        }
        dest.append(closingTag);
    }
}
