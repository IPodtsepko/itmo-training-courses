package md2html.markup;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public interface Formatted {
    void toMarkdown(StringBuilder dest);

    void toHtml(StringBuilder dest);
}
