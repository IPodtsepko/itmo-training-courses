package queue;

import java.util.Objects;

/**
 * Title task: "Homework 2, 3. Queue: Programming Aproaches, OOP"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

/**
 * Model:
 *       [a1, a2, a3, ..., an]
 *       n is size of queue
 *
 * Inv:
 *       n >= 0
 *       for all i in [1 : n] a[i] != null
 *
 * Immutable:
 *       n = n' && for all i in [1 : n] a[i] = a'[i]
 */

public class ArrayQueue extends AbstractQueue {

    private static final int INITIAL_CAPACITY = 16;
    private Object[] elements;
    private int tail;

    public ArrayQueue() {
        elements = new Object[INITIAL_CAPACITY];
    }

    @Override
    protected void enqueueImplementation(Object element) {
        ensureCapacity(size);
        elements[head()] = element;
    }

    @Override
    protected Object elementImplementation() {
        return elements[tail];
    }

    @Override
    protected void dequeueImplementation() {
        elements[tail] = null;
        tail = (tail + 1) % elements.length;
    }

    @Override
    protected void clearImplementation() {
        elements = new Object[INITIAL_CAPACITY];
        tail = 0;
    }

    // pre: element != null
    public void push(Object element) {
        Objects.requireNonNull(element);
        ensureCapacity(++size);
        tail = (tail + elements.length - 1) % elements.length;
        elements[tail] = element;
    }
    // post: n = n' + 1 && a[1] == e && for all i in [2 : n] a[i] == a'[i - 1]

    // pre: n > 0
    public Object peek() {
        assert size > 0;
        return elements[head()];
    }
    // post: R = a[n] && Immutable

    // pre: n > 0
    public Object remove() {
        assert size > 0;
        Object element = elements[head()];
        elements[head()] = null;
        size--;
        return element;
    }
    // post: R = a'[n'], n = n' - 1 && for all i in [1 : n] a[i] = a'[i]

    private int head() {
        return (tail + size - 1) % elements.length;
    }

    private void ensureCapacity(int newSize) {
        int capacity = elements.length;
        if (newSize >= capacity) {
            Object[] resized = new Object[2 * capacity];
            System.arraycopy(elements, tail, resized, 0, capacity - tail);
            System.arraycopy(elements, 0, resized, capacity - tail, head() + 1);
            elements = resized;
            tail = 0;
        }
    }
}
