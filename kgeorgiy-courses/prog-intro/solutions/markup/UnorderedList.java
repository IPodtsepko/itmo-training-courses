package markup;

import java.util.List;

public class UnorderedList extends FormattedList {

    public UnorderedList(List<ListItem> items) {
        super(items);
    }

    @Override
    public void toTex(StringBuilder dest) {
        toTex(dest, "\\begin{itemize}", "\\end{itemize}");
    }
}
