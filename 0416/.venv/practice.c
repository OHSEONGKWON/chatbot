#include<stdio.h>
#include<stdlib.h>

typedef struct Node{
    int data;
    struct node *next;
} Node;

Node* insert(Node* head, int data){
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode -> data = data;
    newNode -> next = head;
    return newNode;
}

Node* asdf(Node* head, int data){
    if (head == Null || head -> data==data) return head;
    Data *prev = NULL, *curr=head;
    while(curr!=NULL&&curr->value!=value){
        prev = curr;
        curr = curr -> next;
    }
    if (curr != NULL && prev != NULL){
        prev -> next = curr -> next;
    }
    return head;
}

int main(){
    Node *head = NULL, *curr;
    for(int i=1;i<=5;i++){
        head = insert(head, i);
    }
    head = asdf(head, 3);
    for(curr = head; curr != NULL; curr = curr -> next){
        printf("%d ", curr -> data);
    }
    return 0;
}