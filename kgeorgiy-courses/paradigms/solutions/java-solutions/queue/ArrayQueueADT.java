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

public class ArrayQueueADT { // Abstract Data Type
    private static final int INITIAL_CAPACITY = 16;
    private Object[] elements;
    private int tail;
    private int size;

    // pre: true
    public ArrayQueueADT() {
        elements = new Object[INITIAL_CAPACITY];
    }
    // post: n == 0

    // pre: element != null, q != null
    public static void enqueue(ArrayQueueADT q, Object element) {
        Objects.requireNonNull(element);
        Objects.requireNonNull(q);
        ensureCapacity(q, ++q.size);
        q.elements[head(q)] = element;
    }
    // post: n = n' + 1 && for all i in [1 : n'] a[i] == a'[i] && a[n] = e

    // pre: element != null, q != null
    public static void push(ArrayQueueADT q, Object element) {
        Objects.requireNonNull(element);
        Objects.requireNonNull(q);
        ensureCapacity(q, ++q.size);
        q.tail = (q.tail + q.elements.length - 1) % q.elements.length;
        q.elements[q.tail] = element;
    }
    // post: n = n' + 1 && a[1] == e && for all i in [2 : n] a[i] == a'[i - 1]

    // pre: n > 0, q != null
    public static Object peek(ArrayQueueADT q) {
        assert q.size > 0;
        Objects.requireNonNull(q);
        return q.elements[head(q)];
    }
    // post: R = a[n] && Immutable

    // pre: n > 0, q != null
    public static Object element(ArrayQueueADT q) {
        assert q.size > 0;
        Objects.requireNonNull(q);
        return q.elements[q.tail];
    }
    // post: R = a[1] && Immutable

    // pre: n > 0, q != null
    public static Object remove(ArrayQueueADT q) {
        assert q.size > 0;
        Objects.requireNonNull(q);
        Object element = q.elements[head(q)];
        q.elements[head(q)] = null;
        q.size--;
        return element;
    }
    // post: R = a'[n'], n = n' - 1 && for all i in [1 : n] a[i] = a'[i]

    // pre: n > 0, q != null
    public static Object dequeue(ArrayQueueADT q) {
        assert q.size > 0;
        Objects.requireNonNull(q);
        q.size--;
        Object element = q.elements[q.tail];
        q.elements[q.tail] = null;
        q.tail = (q.tail + 1) % q.elements.length;
        return element;
    }
    // post: R = a[1] && n = n' - 1 && for all i in [1 : n] a[i] = a'[i + 1]

    // pre: q != null
    public static int size(ArrayQueueADT q) {
        Objects.requireNonNull(q);
        return q.size;
    }
    // post: R == n && Immutable

    // pre: q != null
    public static boolean isEmpty(ArrayQueueADT q) {
        Objects.requireNonNull(q);
        return q.size == 0;
    }
    // post: post: R == (n == 0) && Immutable

    // pre: q != null
    public static void clear(ArrayQueueADT q) {
        Objects.requireNonNull(q);
        q.elements = new Object[INITIAL_CAPACITY];
        q.tail = 0;
        q.size = 0;
    }
    // post: n == 0

    private static int head(ArrayQueueADT q) {
        return (q.tail + q.size - 1) % q.elements.length;
    }

    private static void ensureCapacity(ArrayQueueADT q, int newSize) {
        int capacity = q.elements.length;
        if (newSize >= capacity) {
            Object[] resized = new Object[2 * capacity];
            if (q.tail + q.size < capacity) {
                System.arraycopy(q.elements, q.tail, resized, 0, q.size);
            } else {
                int cntNotLooped = capacity - q.tail;
                System.arraycopy(q.elements, q.tail, resized, 0, cntNotLooped);
                System.arraycopy(q.elements, 0, resized, cntNotLooped, head(q) + 1);
            }
            q.elements = resized;
            q.tail = 0;
        }
    }
}
