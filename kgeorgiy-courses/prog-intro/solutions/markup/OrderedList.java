package markup;

import java.util.List;

public class OrderedList extends FormattedList {
    public OrderedList(List<ListItem> items) {
        super(items);
    }

    @Override
    public void toTex(StringBuilder dest) {
        toTex(dest, "\\begin{enumerate}", "\\end{enumerate}");
    }
}
