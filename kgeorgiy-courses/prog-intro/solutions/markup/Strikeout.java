package markup;

import java.util.List;

public class Strikeout extends FormattedContainer implements InlineFormattedElement {

    public Strikeout(List<InlineFormattedElement> elements) {
        super(elements);
    }

    @Override
    public void toTex(StringBuilder dest) {
        toTex(dest, "\\textst{", "}");
    }

    @Override
    public void toMarkdown(StringBuilder dest) {
        toMarkdown(dest, "~");
    }
}
