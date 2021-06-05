package md2html.markup;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class Text extends FormattedElement implements InlineFormattedElement {
    private final String text;

    public Text(String text) {
        this.text = text;
    }

    @Override
    public void toMarkdown(StringBuilder dest) {
        dest.append(text);
    }

    @Override
    public void toHtml(StringBuilder dest) {
        dest.append(text);
    }
}
