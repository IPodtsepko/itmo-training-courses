package org.generator.util;

/**
 * Contains the necessary data to work with the typed attribute.
 *
 * @author Igor Podtsepko
 */
public class Attribute {
    private String type;
    private String name;

    @Override
    public String toString() {
        return String.format("%s %s", getType(), getName());
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
