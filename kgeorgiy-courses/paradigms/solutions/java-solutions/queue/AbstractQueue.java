package queue;

import java.util.Objects;

/**
 * Title task: "Homework 3. Queue: OOP"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public abstract class AbstractQueue implements Queue {
    protected int size;

    @Override
    public void enqueue(Object element) {
        Objects.requireNonNull(element);
        size++;
        enqueueImplementation(element);
    }

    protected abstract void enqueueImplementation(Object element);

    @Override
    public Object element() {
        assert size > 0;
        return elementImplementation();
    }

    protected abstract Object elementImplementation();

    @Override
    public Object dequeue() {
        assert size > 0;
        Object element = element();
        dequeueImplementation();
        size--;
        return element;
    }

    protected abstract void dequeueImplementation();

    @Override
    public void clear() {
        size = 0;
        clearImplementation();
    }

    protected abstract void clearImplementation();

    @Override
    public boolean contains(Object element) {
        return contains(element, false);
    }

    @Override
    public boolean removeFirstOccurrence(Object element) {
        return contains(element, true);
    }

    private boolean contains(Object element, boolean needRemove) {
        Objects.requireNonNull(element);
        boolean contains = false;
        int originalSize = this.size;
        for (int i = 0; i < originalSize; i++) {
            Object e = this.dequeue();
            if (!contains && e.equals(element)) {
                contains = true;
                if (needRemove) {
                    continue;
                }
            }
            this.enqueue(e);
        }
        return contains;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public boolean isEmpty() {
        return size == 0;
    }
}
