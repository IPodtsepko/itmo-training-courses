package org.generator.util;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Contains the necessary data to work with the alternative inside the rule for the parser.
 *
 * @author Igor Podtsepko
 */
public class Alternative {
    private Set<String> follow1 = new HashSet<>();
    private final List<AlternativeItem> items = new ArrayList<>();

    /**
     * Adds a new item to the end of the alternative.
     *
     * @param item item to insert
     */
    public void addItem(AlternativeItem item) {
        getItems().add(item);
    }

    @Override
    public String toString() {
        return getItems().stream().map(Object::toString).collect(Collectors.joining("\n"));
    }

    /**
     * @return The set of FOLLOW<sub>1</sub> for this alternative.
     */
    public Set<String> getFollow1() {
        return follow1;
    }

    /**
     * Updates the FOLLOW<sub>1</sub> set for this alternative.
     *
     * @param follow1 new FOLLOW<sub>1</sub>.
     */
    public void setFollow1(Set<String> follow1) {
        this.follow1 = follow1;
    }

    /**
     * @return List of items in this alternative
     */
    public List<AlternativeItem> getItems() {
        return items;
    }

    /**
     * @return List of items of the {@link RuleItem} type.
     */
    public List<RuleItem> ruleItems() {
        return items.stream().filter(AlternativeItem::isRuleItem).map(AlternativeItem::asRuleItem).toList();
    }
}
