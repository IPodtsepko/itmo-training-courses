package md2html.markup;

import java.util.List;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class Page extends FormattedElement {
    private final List<BlockFormattedElement> blocks;

    public Page(List<BlockFormattedElement> blocks) {
        this.blocks = blocks;
    }

    @Override
    public void toMarkdown(StringBuilder dest) {
        for (Formatted block : blocks) {
            block.toMarkdown(dest);
        }
    }

    @Override
    public void toHtml(StringBuilder dest) {
        for (Formatted block : blocks) {
            block.toHtml(dest);
        }
    }
}
