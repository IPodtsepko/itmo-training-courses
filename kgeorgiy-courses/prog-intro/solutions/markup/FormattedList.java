package markup;

import java.util.List;

public abstract class FormattedList extends FormattedContainer implements BlockFormattedElement {
    public FormattedList(List<ListItem> items) {
        super(items);
    }

    @Override
    public void toMarkdown(StringBuilder dest) {
        throw new UnsupportedOperationException();
    }
}
