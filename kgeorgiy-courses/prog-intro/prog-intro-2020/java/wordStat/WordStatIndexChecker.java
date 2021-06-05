package wordStat;

import base.Asserts;
import base.Pair;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * @author Georgiy Korneev (kgeorgiy@kgeorgiy.info)
 */
public class WordStatIndexChecker extends WordStatChecker {
    public WordStatIndexChecker(final String className) {
        super(className);
    }

    public final void testPP(final String[] text, final List<Pair<String, String>> expected) {
        final List<String> expectedList = expected.stream()
                .map(p -> p.first + " " + p.second)
                .collect(Collectors.toList());
        Asserts.assertEquals("output", expectedList, runFiles(Arrays.asList(text)));
        counter.passed();
    }
}
