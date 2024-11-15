/*
 * File:   DLinkedList.h
 */

#ifndef DLINKEDLIST_H
#define DLINKEDLIST_H

#include "list/IList.h"

#include <sstream>
#include <iostream>
#include <type_traits>
using namespace std;

template <class T>
class DLinkedList : public IList<T>
{
public:
    class Node;        // Forward declaration
    class Iterator;    // Forward declaration
    class BWDIterator; // Forward declaration

protected:
    Node *head; // this node does not contain user's data
    Node *tail; // this node does not contain user's data
    int count;
    bool (*itemEqual)(T &lhs, T &rhs);        // function pointer: test if two items (type: T&) are equal or not
    void (*deleteUserData)(DLinkedList<T> *); // function pointer: be called to remove items (if they are pointer type)

public:
    DLinkedList(
        void (*deleteUserData)(DLinkedList<T> *) = 0,
        bool (*itemEqual)(T &, T &) = 0);
    DLinkedList(const DLinkedList<T> &list);
    DLinkedList<T> &operator=(const DLinkedList<T> &list);
    ~DLinkedList();

    // Inherit from IList: BEGIN
    void add(T e);
    void add(int index, T e);
    T removeAt(int index);
    bool removeItem(T item, void (*removeItemData)(T) = 0);
    bool empty();
    int size();
    void clear();
    T &get(int index);
    int indexOf(T item);
    bool contains(T item);
    string toString(string (*item2str)(T &) = 0);
    // Inherit from IList: END

    void println(string (*item2str)(T &) = 0)
    {
        cout << toString(item2str) << endl;
    }
    void setDeleteUserDataPtr(void (*deleteUserData)(DLinkedList<T> *) = 0)
    {
        this->deleteUserData = deleteUserData;
    }

    bool contains(T array[], int size)
    {
        int idx = 0;
        for (DLinkedList<T>::Iterator it = begin(); it != end(); it++)
        {
            if (!equals(*it, array[idx++], this->itemEqual))
                return false;
        }
        return true;
    }

    /*
     * free(DLinkedList<T> *list):
     *  + to remove user's data (type T, must be a pointer type, e.g.: int*, Point*)
     *  + if users want a DLinkedList removing their data,
     *      he/she must pass "free" to constructor of DLinkedList
     *      Example:
     *      DLinkedList<T> list(&DLinkedList<T>::free);
     */
    static void free(DLinkedList<T> *list)
    {
        typename DLinkedList<T>::Iterator it = list->begin();
        while (it != list->end())
        {
            delete *it;
            it++;
        }
    }

    /* begin, end and Iterator helps user to traverse a list forwardly
     * Example: assume "list" is object of DLinkedList

     DLinkedList<char>::Iterator it;
     for(it = list.begin(); it != list.end(); it++){
            char item = *it;
            std::cout << item; //print the item
     }
     */
    Iterator begin()
    {
        return Iterator(this, true);
    }
    Iterator end()
    {
        return Iterator(this, false);
    }

    /* last, beforeFirst and BWDIterator helps user to traverse a list backwardly
     * Example: assume "list" is object of DLinkedList

     DLinkedList<char>::BWDIterator it;
     for(it = list.last(); it != list.beforeFirst(); it--){
            char item = *it;
            std::cout << item; //print the item
     }
     */
    BWDIterator bbegin()
    {
        return BWDIterator(this, true);
    }
    BWDIterator bend()
    {
        return BWDIterator(this, false);
    }

protected:
    static bool equals(T &lhs, T &rhs, bool (*itemEqual)(T &, T &))
    {
        if (itemEqual == 0)
            return lhs == rhs;
        else
            return itemEqual(lhs, rhs);
    }
    void copyFrom(const DLinkedList<T> &list);
    void removeInternalData();
    Node *getPreviousNodeOf(int index);

    //////////////////////////////////////////////////////////////////////
    ////////////////////////  INNER CLASSES DEFNITION ////////////////////
    //////////////////////////////////////////////////////////////////////
public:
    class Node
    {
    public:
        T data;
        Node *next;
        Node *prev;
        friend class DLinkedList<T>;

    public:
        Node(Node *next = 0, Node *prev = 0)
        {
            this->next = next;
            this->prev = prev;
        }
        Node(T data, Node *next = 0, Node *prev = 0)
        {
            this->data = data;
            this->next = next;
            this->prev = prev;
        }
    };

    //////////////////////////////////////////////////////////////////////
    class Iterator
    {
    private:
        DLinkedList<T> *pList;
        Node *pNode;

    public:
        Iterator(DLinkedList<T> *pList = 0, bool begin = true)
        {
            if (begin)
            {
                if (pList != 0)
                    this->pNode = pList->head->next;
                else
                    pNode = 0;
            }
            else
            {
                if (pList != 0)
                    this->pNode = pList->tail;
                else
                    pNode = 0;
            }
            this->pList = pList;
        }

        Iterator &operator=(const Iterator &iterator)
        {
            this->pNode = iterator.pNode;
            this->pList = iterator.pList;
            return *this;
        }
        void remove(void (*removeItemData)(T) = 0)
        {
            pNode->prev->next = pNode->next;
            pNode->next->prev = pNode->prev;
            Node *pNext = pNode->prev; // MUST prev, so iterator++ will go to end
            if (removeItemData != 0)
                removeItemData(pNode->data);
            delete pNode;
            pNode = pNext;
            pList->count -= 1;
        }

        T &operator*()
        {
            return pNode->data;
        }
        bool operator!=(const Iterator &iterator)
        {
            return pNode != iterator.pNode;
        }
        // Prefix ++ overload
        Iterator &operator++()
        {
            pNode = pNode->next;
            return *this;
        }
        // Postfix ++ overload
        Iterator operator++(int)
        {
            Iterator iterator = *this;
            ++*this;
            return iterator;
        }
    };

        //////////////////////////////////////////////////////////////////////
    class BWDIterator
    {
    private:
        DLinkedList<T> *pList;
        Node *pNode;

    public:
        BWDIterator(DLinkedList<T> *pList = 0, bool begin = true)
        {
            if (begin)
            {
                if (pList != 0)
                    this->pNode = pList->tail->prev;
                else
                    pNode = 0;
            }
            else
            {
                if (pList != 0)
                    this->pNode = pList->head;
                else
                    pNode = 0;
            }
            this->pList = pList;
        }

        BWDIterator &operator=(const Iterator &iterator)
        {
            this->pNode = iterator.pNode;
            this->pList = iterator.pList;
            return *this;
        }
        void remove(void (*removeItemData)(T) = 0)
        {
            pNode->next->prev = pNode->prev;
            pNode->prev->next = pNode->next;
            Node * pPrev = pNode->next; // MUST next, so iterator++ will go to head
            if (removeItemData != 0)
                removeItemData(pNode->data);
            delete pNode;
            pNode = pPrev;
            pList->count -= 1;
        }

        T &operator*()
        {
            return pNode->data;
        }
        bool operator!=(const BWDIterator &iterator)
        {
            return pNode != iterator.pNode;
        }
        // Prefix ++ overload
        BWDIterator &operator++()
        {
            pNode = pNode->prev;
            return *this;
        }
        // Postfix ++ overload
        BWDIterator operator++(int)
        {
            BWDIterator iterator = *this;
            ++*this;
            return iterator;
        }
    };
};
//////////////////////////////////////////////////////////////////////
// Define a shorter name for DLinkedList:

template <class T>
using List = DLinkedList<T>;

//////////////////////////////////////////////////////////////////////
////////////////////////     METHOD DEFNITION      ///////////////////
//////////////////////////////////////////////////////////////////////

template <class T>
DLinkedList<T>::DLinkedList(
    void (*deleteUserData)(DLinkedList<T> *),
    bool (*itemEqual)(T &, T &))
{
    // TODO
    count = 0;
    setDeleteUserDataPtr(deleteUserData);
    this->itemEqual = itemEqual;
    head = new Node();
    tail = new Node();
    head->next = tail;
    tail->prev = head;
}

template <class T>
DLinkedList<T>::DLinkedList(const DLinkedList<T> &list)
{
    // TODO
    copyFrom(list);
}

template <class T>
DLinkedList<T> &DLinkedList<T>::operator=(const DLinkedList<T> &list)
{
    // TODO
    if (this != &list){
        copyFrom(list);
    }
    return *this;
}

template <class T>
DLinkedList<T>::~DLinkedList()
{
    // TODO
    removeInternalData();
}

template <class T>
void DLinkedList<T>::add(T e)
{
    // TODO
    if (empty()){
        head->next = new Node(e, tail, head);
        tail->prev = head->next;
    }
    else{
        Node *tmp = new Node(e, tail, tail->prev);
        tail->prev->next = tmp;
        tail->prev = tmp;
    }
    count++;
}
template <class T>
void DLinkedList<T>::add(int index, T e)
{
    // TODO
    if (index < 0 or index > count) throw std::out_of_range("the input index is out of range!");
    if (empty() or index == count) {
        add(e);
        return;
    }
    Node *current = head->next;
    for (int i = 0; i < index; i++){
        current = current->next;
    }
    Node *tmp = new Node(e, current, current->prev);
    if (index != 0) current->prev->next = tmp;
    else head->next = tmp;
    current->prev = tmp;
    count++;
}

template <class T>
typename DLinkedList<T>::Node *DLinkedList<T>::getPreviousNodeOf(int index)
{
    /**
     * Returns the node preceding the specified index in the doubly linked list.
     * If the index is in the first half of the list, it traverses from the head; otherwise, it traverses from the tail.
     * Efficiently navigates to the node by choosing the shorter path based on the index's position.
     */
    // TODO
    if (index < count/2){
        Node *current = head->next;
        for (int i = 0; i < index; i++){
            current = current->next;
        }
        return current;
    }
    else {
        Node *current = tail->prev;
        for (int i = count-1; i > index; i--){
            current = current->prev;
        }
        return current;
    }
}

template <class T>
T DLinkedList<T>::removeAt(int index)
{
    // TODO
    if (index < 0 or index >= count) throw std::out_of_range("the input index is out of range!");
    Node *current = head->next;
    T res = current->data;
    if (index == 0) {  // Removing the first element
        if (count == 1) {  // If it's the only element in the list
            head->next = tail;
            tail->prev = head;
        } else {
            head->next = current->next;
            head->next->prev = head;
        }
    } 
    else if (index == count-1){
        current = tail->prev;
        res = current->data;
        // Node *prev = current->prev;
        current->prev->next = tail;
        tail->prev = current->prev;
    }
    else{
        for (int i = 0; i < index; i++){
            current = current->next;
            if (current->next == tail) break;
        }
        res = current->data;
        // Node *prev = current->prev;
        // Node *next = current->next;
        current->prev->next = current->next;
        current->next->prev = current->prev;
    }
    count--;
    delete current;
    return res;
}

template <class T>
bool DLinkedList<T>::empty()
{
    // TODO
    if (count != 0) return false;
    if (head->next != tail) return false;
    if (tail->prev != head) return false;
    return true;
}

template <class T>
int DLinkedList<T>::size()
{
    // TODO
    return count;
}

template <class T>
void DLinkedList<T>::clear()
{
    // TODO
    removeInternalData();
    head = new Node();
    tail = new Node();
    head->next = tail;
    tail->prev = head;
    count = 0;
}

template <class T>
T &DLinkedList<T>::get(int index)
{
    // TODO
    if (index < 0 or index >= count) throw std::out_of_range("the input index is out of range!");
    Node *current = head->next;
    for (int i = 0; i < index; i++){
        current = current->next;
    }
    return current->data;
}

template <class T>
int DLinkedList<T>::indexOf(T item)
{
    // TODO
    Node *current = head->next;
    if (itemEqual != NULL){
        for (int i = 0; i < count; i++){
            if(itemEqual(current->data, item)) return i;
            current = current->next;
        }
        return -1;
    }
    for (int i = 0; i < count; i++){
        if(current->data == item) return i;
        current = current->next;
    }
    return -1;
}

template <class T>
bool DLinkedList<T>::removeItem(T item, void (*removeItemData)(T))
{
    // TODO
    int item_index = indexOf(item);
    if (item_index == -1) return false;
    T item_removing = removeAt(item_index);
    if (removeItemData != NULL){
        removeItemData(item_removing);
    }
    return true;
}

template <class T>
bool DLinkedList<T>::contains(T item)
{
    // TODO
    if (itemEqual != NULL){
        Node *current = head->next;
        while(current != tail) {
            if(itemEqual((current->data), item)) return true;
            current = current->next;
        }
        return false;
    }
    else {
        Node *current = head->next;
        while(current != tail) {
            if(current->data == item) return true;
            current = current->next;
        }
        return false;
    }
}

template <class T>
string DLinkedList<T>::toString(string (*item2str)(T &))
{
    /**
     * Converts the list into a string representation, where each element is formatted using a user-provided function.
     * If no custom function is provided, it directly uses the element's default string representation.
     * Example: If the list contains {1, 2, 3} and the provided function formats integers, calling toString would return "[1, 2, 3]".
     *
     * @param item2str A function that converts an item of type T to a string. If null, default to string conversion of T.
     * @return A string representation of the list with elements separated by commas and enclosed in square brackets.
     */
    // TODO
    string result = "[";
    Node *current = head->next;
    if (*item2str != NULL){
        for (int i = 0; i < count-1; i++){
            result += (*item2str)(current->data);
            result += ", ";
            current = current->next;
        }
        result += (*item2str)(current->data);
        result += "]";
        return result;
    }
    for (int i = 0; i < count-1; i++){
            stringstream data_str;
            data_str << current->data;
            result += data_str.str();
            result += ", ";
            current = current->next;
        }
    stringstream data_str;
    data_str << current->data;
    result += data_str.str();
    result += "]";
    return result;
}

template <class T>
void DLinkedList<T>::copyFrom(const DLinkedList<T> &list)
{
    /**
     * Copies the contents of another doubly linked list into this list.
     * Initializes the current list to an empty state and then duplicates all data and pointers from the source list.
     * Iterates through the source list and adds each element, preserving the order of the nodes.
     */
    // TODO
    clear();
    if (list.deleteUserData != NULL) 
        setDeleteUserDataPtr(list.deleteUserData);
    if (list.itemEqual != NULL) 
        itemEqual = list.itemEqual;
    Node* current = list.head->next;
    while (current != list.tail) {
        add(current->data);
        current = current->next;
    }
}

template <class T>
void DLinkedList<T>::removeInternalData()
{
    /**
     * Clears the internal data of the list by deleting all nodes and user-defined data.
     * If a custom deletion function is provided, it is used to free the user's data stored in the nodes.
     * Traverses and deletes each node between the head and tail to release memory.
     */
    // TODO
    if (deleteUserData != NULL and !empty()){
        deleteUserData(this);
    }
    Node *tmp;
    Node *removing_item = head->next;
    while (removing_item != tail){
        tmp = removing_item;
        removing_item = removing_item->next;
        if (tmp != tail) removeItem(tmp->data);
    }
    delete tail;
    delete head;
    count = 0;
    itemEqual = NULL;
    deleteUserData = NULL;
}

#endif /* DLINKEDLIST_H */
