package md2html.parser;

import md2html.markup.*;
import md2html.parser.exceptions.TagNotFoundException;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * @author Igor Podtsepko (i.podtsepko@niuitmo.com)
 */
public class InlineMarkupParser extends BaseParser {
    private static final List<String> SUPPORTED_TAGS = List.of("**", "*", "__", "_", "--", "`", "![");
    private static final Map<String, Integer> TAGS_INDEXES =
            SUPPORTED_TAGS.stream().collect(Collectors.toMap(Function.identity(), SUPPORTED_TAGS::indexOf));
    private static final Set<Character> RESERVED =
            SUPPORTED_TAGS.stream().map(tag -> tag.charAt(0)).collect(Collectors.toSet());
    private static final Map<Character, String> CODES_FOR_HTML = Map.of(
            '<', "&lt;",
            '>', "&gt;",
            '&', "&amp;"
    );

    private List<Boolean> isFound;
    private List<InlineFormattedElement> parsedMarks;
    private StringBuilder parsedText;
    private Deque<String> foundedTags;
    private Deque<FormattedContainer> foundedMarks;

    public List<InlineFormattedElement> parse(final StringBuilder source) {
        return parse(source.toString());
    }

    public List<InlineFormattedElement> parse(final String source) {
        setSource(new StringSource(source));

        isFound = new ArrayList<>(Collections.nCopies(SUPPORTED_TAGS.size(), false));
        parsedMarks = new LinkedList<>();
        parsedText = new StringBuilder();
        foundedTags = new ArrayDeque<>();
        foundedMarks = new ArrayDeque<>();

        while (!eof()) {
            if (test('\\')) {
                parseEscape();
            } else if (CODES_FOR_HTML.containsKey(ch)) {
                parsedText.append(CODES_FOR_HTML.get(ch));
                nextChar();
            } else if (!RESERVED.contains(ch)) {
                parsedText.append(ch);
                nextChar();
            } else {
                parseTag();
            }
        }
        pushText();
        if (!foundedMarks.isEmpty()) {
            throw new TagNotFoundException(foundedTags.getFirst());
        }
        return List.copyOf(parsedMarks);
    }

    private void parseTag() {
        char previous = getPosition() > 1 ? previous() : ' ';

        String tag = getTag();
        if (tag == null) {
            parsedText.append(ch);
            nextChar();
            return;
        }

        int iTag = TAGS_INDEXES.get(tag);

        if (Character.isWhitespace(previous) && Character.isWhitespace(ch)) {
            parsedText.append(tag);
            return;
        }

        pushText();
        if (!foundedTags.isEmpty() && isFound.get(iTag)) {
            if (!foundedTags.getFirst().equals(tag)) {
                throw new TagNotFoundException(foundedTags.getFirst(), getPosition());
            }
            InlineFormattedElement founded = (InlineFormattedElement) foundedMarks.pop();
            foundedTags.pop();
            isFound.set(iTag, false);
            pushMark(founded);
        } else if (tag.equals("![")) {
                pushMark(parseImage());
        } else {
            isFound.set(iTag, true);
            FormattedContainer newElement = newElement(tag);
            foundedMarks.push(newElement);
            foundedTags.push(tag);
        }
    }

    private InlineFormattedElement parseImage() {
        String name = tokenToTag("](");
        String address = tokenToTag(")");
        return new Image(name, address);
    }

    private String tokenToTag(String tag) {
        StringBuilder token = new StringBuilder();
        while (!test(tag)) {
            if (eof()) {
                throw new TagNotFoundException(tag);
            }
            token.append(ch);
            nextChar();
        }
        return token.toString();
    }

    private void pushText() {
        if (parsedText.length() > 0) {
            pushMark(new Text(parsedText.toString()));
        }
        parsedText = new StringBuilder();
    }

    public void pushMark(InlineFormattedElement mark) {
        if (foundedMarks.isEmpty()) {
            parsedMarks.add(mark);
        } else {
            foundedMarks.getFirst().add(mark);
        }
    }

    private void parseEscape() {
        for (Character c : RESERVED) {
            if (test(c)) {
                parsedText.append(c);
                return;
            }
        }
        parsedText.append('\\');
    }

    public String getTag() {
        for (String tag : SUPPORTED_TAGS) {
            if (test(tag)) {
                return tag;
            }
        }
        return null;
    }

    private FormattedContainer newElement(String tag) {
        switch (tag) {
            case "*":
            case "_":
                return new Emphasis();
            case "**":
            case "__":
                return new Strong();
            case "--":
                return new Strikeout();
            case "`":
                return new Code();
            default:
                return null;
        }
    }
}
