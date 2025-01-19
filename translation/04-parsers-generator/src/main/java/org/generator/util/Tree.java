package org.generator.util;

import java.util.ArrayList;
import java.util.List;

public abstract class Tree {
    public List<Tree> children;
    public String name;

    public Tree(String name) {
        this.name = name;
        children = new ArrayList<>();
    }
}
