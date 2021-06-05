package markup;

public interface Formatted {
    void toTex(StringBuilder dest);

    void toMarkdown(StringBuilder dest);
}
