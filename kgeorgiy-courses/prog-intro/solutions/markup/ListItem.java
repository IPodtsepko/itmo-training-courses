package markup;

import java.util.List;

public class ListItem extends FormattedContainer {

    public ListItem(List<BlockFormattedElement> blockElements) {
        super(blockElements);
    }

    @Override
    public void toTex(StringBuilder dest) {
        toTex(dest, "\\item ", "");
    }

    @Override
    public void toMarkdown(StringBuilder dest) {
        throw new UnsupportedOperationException();
    }
}
