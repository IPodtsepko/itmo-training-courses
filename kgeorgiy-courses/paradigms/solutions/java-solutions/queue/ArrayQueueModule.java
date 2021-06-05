package queue;

import java.util.Objects;

/**
 * Title task: "Homework 2. Queue: Programming Aproaches"
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

public class ArrayQueueModule {
    private static final int INITIAL_CAPACITY = 16;
    private static Object[] elements = new Object[INITIAL_CAPACITY];
    private static int tail;
    private static int size;

    // pre: element != null
    public static void enqueue(Object element) {
        Objects.requireNonNull(element);
        ensureCapacity(++size);
        elements[head()] = element;
    }
    // post: n = n' + 1 && for all i in [1 : n'] a[i] == a'[i] && a[n] = e

    // pre: element != null
    public static void push(Object element) {
        Objects.requireNonNull(element);
        ensureCapacity(++size);
        tail = (tail + elements.length - 1) % elements.length;
        elements[tail] = element;
    }
    // post: n = n' + 1 && a[1] == e && for all i in [2 : n] a[i] == a'[i - 1]

    // pre: n > 0
    public static Object peek() {
        assert size > 0;
        return elements[head()];
    }
    // post: R = a[n] && Immutable

    // pre: n > 0
    public static Object element() {
        assert size > 0;
        return elements[tail];
    }
    // post: R = a[1] && Immutable

    // pre: n > 0
    public static Object remove() {
        assert size > 0;
        Object element = elements[head()];
        elements[head()] = null;
        size--;
        return element;
    }
    // post: R = a'[n'], n = n' - 1 && for all i in [1 : n] a[i] = a'[i]

    // pre: n > 0
    public static Object dequeue() {
        assert  size > 0;
        size--;
        Object element = elements[tail];
        elements[tail] = null;
        tail = (tail + 1) % elements.length;
        return element;
    }
    // post: R = a[1] && n = n' - 1 && for all i in [1 : n] a[i] = a'[i + 1]

    // pre: true
    public static int size() {
        return size;
    }
    // post: R == n && Immutable

    // pre: true
    public static boolean isEmpty() {
        return size() == 0;
    }
    // post: post: R == (n == 0) && Immutable

    // pre: true
    public static void clear() {
        elements = new Object[INITIAL_CAPACITY];
        tail = 0;
        size = 0;
    }
    // post: n == 0

    private static int head() {
        return (tail + size - 1) % elements.length;
    }

    private static void ensureCapacity(int newSize) {
        int capacity = elements.length;
        if (newSize >= capacity) {
            Object[] resized = new Object[2 * capacity];
            if (tail + size < capacity) {
                System.arraycopy(elements, tail, resized, 0, size);
            } else {
                int cntNotLooped = capacity - tail;
                System.arraycopy(elements, tail, resized, 0, cntNotLooped);
                System.arraycopy(elements, 0, resized, cntNotLooped, head() + 1);
            }
            elements = resized;
            tail = 0;
        }
    }
}
