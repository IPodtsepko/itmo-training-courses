% Title task: "Homework 12. Primes in Prolog"
% Author:     Igor Podtsepko (i.podtsepko@outlook.com)

% < SEARCH FOR PRIMES AND COMPOSITES >

% sieve of Eratosthenes
init(Max) :- loop(2, Max).

:- dynamic(composite/1).
composite(1).

prime(X) :- \+ composite(X).

loop(Prime, Max) :-
    prime(Prime), Max >= Prime,
    NextComposit is 2 * Prime,
    sift(NextComposit, Max, Prime), !.
loop(Any, Max) :- Max >= Any, X is Any + 1, loop(X, Max).

sift(Composit, Max, Prime) :-
    Max >= Composit, assert(composite(Composit)),
    NextComposit is Composit + Prime,
    sift(NextComposit, Max, Prime).


% < DIVISORS >

% divisors to number helper
product_primes([], _, 1).
product_primes([CurrentDivisor | Divisors], LowerBound, Product) :-
  CurrentDivisor >= LowerBound, prime(CurrentDivisor),
  product_primes(Divisors, CurrentDivisor, ProductOther),
  Product is ProductOther * CurrentDivisor.

% number -> divisors helper
split_to_divisors(Number, Curr, [Number]) :- Number < Curr * Curr, !.
split_to_divisors(Number, Curr, [Divisor | OtherDivisors]) :-
 	mod(Number, Curr) > 0, Next is Curr + 1,
  split_to_divisors(Number, Next, [Divisor | OtherDivisors]).
split_to_divisors(Number, Curr, [Divisor | OtherDivisors]) :-
  Divisor is Curr, 0 is mod(Number, Curr),
  Factor is Number // Curr,
  split_to_divisors(Factor, Curr, OtherDivisors).

prime_divisors(1, []) :- !.
prime_divisors(Number, Divisors) :- var(Number), !, product_primes(Divisors, 2, Number).
prime_divisors(Number, Divisors) :- number(Number), split_to_divisors(Number, 2, Divisors).


% < MODIFICATION (GCD) >

gcd(A, B, GCD) :-
  number(A), number(B),
	prime_divisors(A, AD),
	prime_divisors(B, BD),
	gcd(AD, BD, GCD), !.

% merge divisors halpers
gcd([], _, 1) :- !.
gcd(_, [], 1).
gcd([AH | AT], [BH | BT], GCD) :- AH is BH, gcd(AT, BT, P), GCD is P * AH.
gcd([AH | AT], [BH | BT], GCD) :- AH < BH, gcd(AT, [BH | BT], GCD).
gcd([AH | AT], [BH | BT], GCD) :- AH > BH, gcd([AH | AT], BT, GCD).
