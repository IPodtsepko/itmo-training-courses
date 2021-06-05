package queue;

/**
 * Title task: "Homework 3. Queue: OOP"
 * @author Igor Podtsepko (i.podtsepko@outlook.com)
 */

public class LinkedQueue extends AbstractQueue {
    private static final Node FICTIVE_NODE = new Node(null, null);

    private Node head;
    private Node tail;

    public LinkedQueue() {
        tail = FICTIVE_NODE;
        head = tail;
    }

    @Override
    protected void enqueueImplementation(Object element) {
        head.next = new Node(element, null);
        head = head.next;
    }

    @Override
    protected Object elementImplementation() {
        return tail.next.data;
    }

    @Override
    protected void dequeueImplementation() {
        tail = tail.next;
    }

    @Override
    protected void clearImplementation() {
        head = tail;
        tail.next = null;
    }

    private static class Node {
        private Object data;
        private Node next;

        public Node(Object data, Node next) {
            this.data = data;
            this.next = next;
        }
    }
}
