package md2html.parser.exceptions;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class TagNotFoundException extends ParseException {
    public TagNotFoundException(String tag, int position) {
        super("Expected tag \"" + tag + "\" in position " + position);
    }

    public TagNotFoundException(String tag) {
        super("Expected tag \"" + tag + "\" before EOF");
    }
}
