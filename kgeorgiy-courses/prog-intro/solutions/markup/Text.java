package markup;

public class Text implements InlineFormattedElement {
    private String text;

    public Text(String text) {
        this.text = text;
    }

    @Override
    public void toTex(StringBuilder dest) {
        dest.append(text);
    }

    @Override
    public void toMarkdown(StringBuilder dest) {
        dest.append(text);
    }
}
