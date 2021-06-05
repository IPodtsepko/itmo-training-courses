package markup;

import java.util.List;

public class Strong extends FormattedContainer implements InlineFormattedElement {

    public Strong(List<InlineFormattedElement> elements) {
        super(elements);
    }

    @Override
    public void toTex(StringBuilder dest) {
        toTex(dest, "\\textbf{", "}");
    }

    @Override
    public void toMarkdown(StringBuilder dest) {
        toMarkdown(dest, "__");
    }
}
