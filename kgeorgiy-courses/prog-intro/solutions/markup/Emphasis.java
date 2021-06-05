package markup;

import java.util.List;

public class Emphasis extends FormattedContainer implements InlineFormattedElement {

    public Emphasis(List<InlineFormattedElement> elements) {
        super(elements);
    }

    @Override
    public void toTex(StringBuilder dest) {
        toTex(dest, "\\emph{", "}");
    }

    @Override
    public void toMarkdown(StringBuilder dest) {
        toMarkdown(dest, "*");
    }
}
