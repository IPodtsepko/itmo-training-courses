% Title task: "Homework 13. Tree-map on Prolog"
% Author:     Igor Podtsepko (i.podtsepko@outlook.com)

node(Data, LC, RC, node(Data, H, LC, RC)) :-
    height(LC, LH), height(RC, RH),
    (LH > RH, !, H is LH + 1; H is RH + 1).

height(null, 0) :- !.
height(node(_, H, _, _), H).

factor(null, 0) :- !.
factor(node(_, _, LC, RC), F) :-
    height(LC, LH), height(RC, RH), F is RH - LH.

r_rotate(node(YData, _, node(XData, _, A, B), C), Result) :-
    node(YData, B, C, Y), node(XData, A, Y, Result).
l_rotate(node(XData, _, A, node(YData, _, B, C)), Result) :-
    node(XData, A, B, X), node(YData, X, C, Result).

r_child_rotate(Child, Result) :- 
    factor(Child, F), F < 0, !, r_rotate(Child, Result).
r_child_rotate(Child, Child).

l_child_rotate(Child, Result) :-
    factor(Child, F), F > 0, !, l_rotate(Child, Result).
l_child_rotate(Child, Child).

balance(null, null) :- !.
balance(Tree, Tree) :- factor(Tree, F), -2 < F, F < 2, !.
balance(Tree, Result) :-
    node(Data, _, LC, RC) = Tree, factor(Tree, F),
    (F is 2, !,
        r_child_rotate(RC, RC1),
        node(Data, LC, RC1, Tmp),
        l_rotate(Tmp, Result);
    % else:
        l_child_rotate(LC, LC1),
        node(Data, LC1, RC, Tmp),
        r_rotate(Tmp, Result)).

map_put(null, K, V, Result) :- node((K, V), null, null, Result).
map_put(node((K, _), _, LC, RC), K, V, Result) :-
    !, node((K, V), LC, RC, Result).
map_put(node(Data, _, LC, RC), Key, Value, Result) :-
    (K, _) = Data,
    (Key < K, !,
        map_put(LC, Key, Value, LC1),
        node(Data, LC1, RC, Tmp);
    % else:
        map_put(RC, Key, Value, RC1),
        node(Data, LC, RC1, Tmp)),
    balance(Tmp, Result).

map_get(node((K, V), _, _, _), K, V) :- !.
map_get(node((K, _), _, LC, _), Key, V) :- Key < K, !, map_get(LC, Key, V).
map_get(node((K, _), _, _, RC), Key, V) :- Key > K, map_get(RC, Key, V).

find_min(node(Data, _, null, _), Data) :- !.
find_min(node(_, _, LC, _), Result) :- find_min(LC, Result).

map_getLast(node(Data, _, _, null), Data) :- !.
map_getLast(node(_, _, _, RC), Result) :- map_getLast(RC, Result).

remove_min(node(_, _, null, RC), RC) :- !.
remove_min(node(Data, _, LC, RC), Result) :-
    remove_min(LC, LC1),
    node(Data, LC1, RC, Tmp),
    balance(Tmp, Result).

map_removeLast(null, null).
map_removeLast(node(_, _, LC, null), LC) :- !.
map_removeLast(node(Data, _, LC, RC), Result) :-
	map_removeLast(RC, RC1),
	node(Data, LC, RC1, Tmp),
	balance(Tmp, Result).

cut_root(node(_, _, LC, null), LC) :- !.
cut_root(node(_, _, LC, RC), Result) :-
    find_min(RC, Data),
    remove_min(RC, RC1),
    node(Data, LC, RC1, Result).

map_remove(null, _, null) :- !.
map_remove(Tree, Key, Result) :-
    node(Data, _, LC, RC) = Tree, (K, _) = Data,
    (Key < K, !,
        map_remove(LC, Key, LC1),
        node(Data, LC1, RC, Tmp);
    Key > K, !,
        map_remove(RC, Key, RC1),
        node(Data, LC, RC1, Tmp);
    % else:
        cut_root(Tree, Tmp)),
    balance(Tmp, Result).

sorted([]).
sorted([_]) :- !.
sorted([(X, _), B | T]) :- (Y, _) = B, X =< Y, sorted([B | T]).

by_sorted([H | T], 1, T, Result) :- !, node(H, null, null, Result).
by_sorted([X, Y | Tail], 2, Tail, Result) :-
    !, node(X, null, null, LC),
    node(Y, LC, null, Result).
by_sorted(List, N, Rest, Result) :-
    LN is N // 2, RN is N - LN - 1,
    by_sorted(List, LN, [Root | Tail], LC),
    by_sorted(Tail, RN, Rest, RC),
    node(Root, LC, RC, Result).

map_build([], null) :- !.
map_build(List, Result) :-
    sorted(List), !, length(List, N), by_sorted(List, N, [], Result).
map_build([(K, V) | Tail], Result) :-
    map_build(Tail, Tmp), map_put(Tmp, K, V, Result).
