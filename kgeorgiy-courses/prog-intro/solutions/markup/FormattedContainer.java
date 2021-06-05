package markup;

import java.util.List;

public abstract class FormattedContainer implements Formatted {
    private final List<? extends Formatted> elements;

    public FormattedContainer(List<? extends Formatted> elements) {
        this.elements = elements;
    }

    public void toTex(StringBuilder dest, String beginningTag, String endingTag) {
        dest.append(beginningTag);
        for (Formatted element : elements) {
            element.toTex(dest);
        }
        dest.append(endingTag);
    }

    public void toMarkdown(StringBuilder dest, String tag) {
        dest.append(tag);
        for (Formatted element : elements) {
            element.toMarkdown(dest);
        }
        dest.append(tag);
    }
}
