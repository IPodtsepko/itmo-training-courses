package md2html.markup;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class Image extends FormattedElement implements InlineFormattedElement {
    private String address;
    private String name;

    public Image(String name, String address) {
        this.name = name;
        this.address = address;
    }


    public void toMarkdown(StringBuilder dest) {
        dest.append("[");
        dest.append(name);
        dest.append("](");
        dest.append(address);
        dest.append(")");
    }

    public void toHtml(StringBuilder dest) {
        dest.append("<img alt='");
        dest.append(name);
        dest.append("' src='");
        dest.append(address);
        dest.append("'>");
    }
}
