#include <assert.h>
#include <iostream>

template <typename T> class List
{
public:
    List(): _head(nullptr), _tail(nullptr) {}
    ~List();

    void pushFront(T);
    void pushBack(T);
    T popFront();
    T popBack();

    int size() const;
    bool empty() const { return _head == nullptr; }
    void print() const;

private:
    struct Node {
        T val;
        Node *prev;
        Node *next;
        Node(T val, Node *prev, Node *next): val(val), prev(prev), next(next){}
    };

    Node *_head;
    Node *_tail;
};

template <typename T> List<T>::~List()
{
    while (_head) {
        Node *current(_head);
        _head = _head->next;
        delete current;
    }
}

template <typename T> void List<T>::pushFront(T val)
{
    Node *front_node = new Node(val, nullptr, _head);
    // corner case
    if (_tail == nullptr) {
        _tail = front_node;
        _head = front_node;
    } else {
        _head->prev = front_node;
        _head = front_node;
    }
}

template <typename T> void List<T>::pushBack(T val)
{
    Node *back_node = new Node(val, _tail, nullptr);
    // corner case
    if (_head == nullptr) {
        _head = back_node;
        _tail = back_node;
    } else {
        _tail->next = back_node;
        _tail = back_node;
    }
}

template <typename T> T List<T>::popFront()
{
    if (this->empty())
        throw("Cannot popFront when list is empty");

    Node *temp(_head);
    T val = _head->val;
    _head = _head->next;
    if (_head) 
        _head->prev = nullptr;
    else 
        _tail = nullptr;
    delete temp;
    return val;
}

template <typename T> T List<T>::popBack()
{
    if (this->empty())
        throw("Cannot popBack when list is empty");
    
    Node *temp(_tail);
    T val = _tail->val;
    _tail = _tail->prev;
    if (_tail)
        _tail->next = nullptr;
    else
        _head = nullptr;

    delete temp;
    return val;
}

template <typename T> int List<T>::size() const 
{
    int size = 0;
    Node *current(_head);
    while (current) {
        size++;
        current = current->next;
    }
    return size;
}

template <typename T> void List<T>::print() const
{
    Node *current(_head);
    while (current) {
        std::cout << "list current val = " << current->val << std::endl;
        current = current->next;
    }
}

int main()
{
    List<int> list1;
    list1.pushBack(9);
    assert(list1.size() == 1);

    list1.pushFront(10);
    assert(list1.size() == 2);
    assert(list1.popBack() == 9);
    assert(list1.popFront() == 10);

    return 0;
}

