% Title task: "Homework 14. Parsing Prolog Expressions"
% Author:     Igor Podtsepko (i.podtsepko@outlook.com)

% helpers
lookup(K, [(K, V) | _], V) :- !.
lookup(K, [_ | T], V) :- lookup(K, T, V).

all_member([], _) :- !.
all_member([H | T], List) :-
    member(H, List), all_member(T, List).

nonvar(V, _) :- var(V).
nonvar(V, T) :- nonvar(V), call(T).

is_var_char(C) :-
		member(C, ['x', 'y', 'z', 'X', 'Y', 'Z']).
is_digit(X) :-
    member(X, ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']).

% < EXPRESSIONS IMLPEMENTATION >
const(Value, const(Value)).
variable(Name, variable(Name)).

op_add(X, Y, operation(op_add, X, Y)).
op_subtract(X, Y, operation(op_subtract, X, Y)).
op_multiply(X, Y, operation(op_multiply, X, Y)).
op_divide(X, Y, operation(op_divide, X, Y)).
op_negate(X, operation(op_negate, X)).

% Mode
op_sinh(X, operation(op_sinh, X)).
op_cosh(X, operation(op_cosh, X)).

operation(op_add, X, Y, R) :- R is X + Y.
operation(op_subtract, X, Y, R) :- R is X - Y.
operation(op_multiply, X, Y, R) :- R is X * Y.
operation(op_divide, X, Y, R) :- R is X / Y.
operation(op_negate, X, R) :- R is -X.

% Mode
operation(op_sinh, X, R) :- R is (exp(X) - exp(-X)) / 2.
operation(op_cosh, X, R) :- R is (exp(X) + exp(-X)) / 2.

evaluate(const(Value), _, Value).
evaluate(variable(Name), Vars, Result) :-
		atom_chars(Name, [RealName | _]),
    lookup(RealName, Vars, Result).
evaluate(operation(F, X), Vars, R) :-
    evaluate(X, Vars, RX),
    operation(F, RX, R).
evaluate(operation(F, X, Y), Vars, R) :-
    evaluate(X, Vars, RX),
    evaluate(Y, Vars, RY),
    operation(F, RX, RY, R).

% < GRAMMAR >
:- load_library('alice.tuprolog.lib.DCGLibrary').

valid_name([H]) --> { is_var_char(H) }, [H].
valid_name([H | T]) --> { is_var_char(H) }, [H], valid_name(T).
% Values
infix(variable(Name)) -->
		{ nonvar(Name, atom_chars(Name, Chars))},
		valid_name(Chars),
		{Chars = [_ | _], atom_chars(Name, Chars) }.
infix(const(Value)) --> 
    { nonvar(Value, number_chars(Value, Chars)) },
    signed(Chars),
    { Chars = [_ | _], number_chars(Value, Chars) }.
    
% binary operations
infix(operation(F, X, Y)) -->
    {var(F)},
	  ['('], in_spaces(X), operator(F), in_spaces(Y), [')'].
infix(operation(F, X, Y)) -->
	  {nonvar(F)},
	  ['('], infix(X), [' '], operator(F), [' '], infix(Y), [')'].
	  
% unary operations
infix(operation(F, X)) --> {var(F)}, operator(F), spaces, in_brackets(X).
infix(operation(F, X)) --> {nonvar(F)}, operator(F), in_brackets(X).
infix(E) --> {var(E)}, in_brackets(E).

% helpers
in_brackets(E) --> ['('], in_spaces(E), [')'].

spaces --> [].
spaces --> [' '], spaces.

in_spaces(E) --> { nonvar(E) }, infix(E).
in_spaces(E) --> { var(E) }, spaces, infix(E), spaces.

signed(V)       --> unsigned(V).
signed([H | T]) --> {member(H, ['+', '-'])}, [H], unsigned(T).

unsigned([V]) --> digits([V]).
unsigned(['.' | T]) --> ['.'], digits(T).
unsigned([H | T]) --> digits([H]), unsigned(T).

digits([H])     --> { is_digit(H) }, [H].
digits([H | T]) --> { is_digit(H) }, [H], digits(T).

operator(op_add)      --> ['+'].
operator(op_subtract) --> ['-'].
operator(op_multiply) --> ['*'].
operator(op_divide)   --> ['/'].
operator(op_negate)   --> ['n', 'e', 'g', 'a', 't', 'e'].

% mode
operator(op_sinh) 		--> ['s', 'i', 'n', 'h'].
operator(op_cosh) 		--> ['c', 'o', 's', 'h'].

infix_str(E, S) :-
    ground(E), phrase(infix(E), C), atom_chars(S, C).
infix_str(E, S) :-
    atom(S), atom_chars(S, C), phrase(in_spaces(E), C).
