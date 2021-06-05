package markup;

import java.util.List;

public class Paragraph extends FormattedContainer implements BlockFormattedElement {

    public Paragraph(List<InlineFormattedElement> elements) {
        super(elements);
    }

    @Override
    public void toTex(StringBuilder dest) {
        toTex(dest, "", "");
    }

    @Override
    public void toMarkdown(StringBuilder dest) {
        toMarkdown(dest, "");
    }
}
