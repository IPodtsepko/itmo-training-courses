// Title task: "Homework 5. Functional expressions in JavaScript"
// Author:     Igor Podtsepko (i.podtsepko@outlook.com)

"use strict"

const variable = (name) => (...vars) => vars[SUPPORTED_VARIABLES_NAMES.indexOf(name)];

const operation = f => (...args) => (...vars) => f(...args.map(g => g(...vars)))

const cnst = value => operation(() => value)();

const one = cnst(1)
const two = cnst(2)

const unary = f => x => operation(f)(x)
const negate = unary(x => -x)

const binary = f => (x, y) => operation(f)(x, y)
const add = binary((x, y) => x + y);
const subtract = binary((x, y) => x - y)
const multiply = binary((x, y) => x * y)
const divide = binary((x, y) => x / y)

const min5 = (a, b, c, d, e) => operation(Math.min)(a, b, c, d, e)
const max3 = (a, b, c) => operation(Math.max)(a, b, c)


const SUPPORTED_OPERATORS = {
    '+': add,
    '-': subtract,
    '*': multiply,
    '/': divide,
    'min5': min5,
    'max3': max3,
    'negate': negate
}

const SPECIAL_CONSTANTS = {
    'one': one,
    'two': two
}

const SUPPORTED_VARIABLES_NAMES = ['x', 'y', 'z']

const parse = expression => {
    const tokens = expression.trim().split(/\s+/)
    let operandStack = []
    for (const token of tokens) {
        if (token in SUPPORTED_OPERATORS) {
            const f = SUPPORTED_OPERATORS[token]
            operandStack.push(f(...operandStack.splice(operandStack.length - f.length, f.length)))
        } else if (token in SPECIAL_CONSTANTS) {
            operandStack.push(SPECIAL_CONSTANTS[token])
        } else {
            operandStack.push(SUPPORTED_VARIABLES_NAMES.indexOf(token) !== -1 ? variable(token) : cnst(+token))
        }
    }
    return operandStack.pop()
}