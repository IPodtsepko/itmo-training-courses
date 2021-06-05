// Title task: "Homework 6, 7. Object expressions and error processing in JavaScript"
// Author:     Igor Podtsepko (i.podtsepko@outlook.com)

"use strict";

const SUPPORTED_VARIABLES_NAMES = ['x', 'y', 'z']
const SUPPORTED_OPERATORS = new Map()

function makeAbstractValueType() {
    const AbstractValueType = function(value) {
        this.value = value
    }
    AbstractValueType.prototype = {
        evaluate(...vars) {
            return this.calculate(this.value, ...vars)
        },
        diff(variable) {
            return this.derivative(this.value, variable)
        },
        toString() {
            return this.value.toString()
        }
    }
    AbstractValueType.prototype.prefix = AbstractValueType.prototype.postfix = AbstractValueType.prototype.toString
    return AbstractValueType
}

const AbstractValueType = makeAbstractValueType()

function TypeFactory(calculate, derivative) {
    const valueType = function(value) {
        AbstractValueType.call(this, value)
    }

    valueType.prototype = Object.create(AbstractValueType.prototype);
    valueType.prototype.calculate = calculate
    valueType.prototype.derivative = derivative
    valueType.prototype.constructor = valueType

    return valueType
}

const Const = TypeFactory((value, ...vars) => value, value => Const.ZERO)
Const.ZERO = new Const(0)
Const.ONE = new Const(1)
Const.TWO = new Const(2)
Const.THREE = new Const(3)

const Variable = TypeFactory(
    (name, ...vars) => vars[SUPPORTED_VARIABLES_NAMES.indexOf(name)],
    (v, name) => v === name ? Const.ONE : Const.ZERO
)
for (const name of SUPPORTED_VARIABLES_NAMES) {
    Variable[name] = new Variable(name)
}

function AbstractOperationFactory() {
    const AbstractOperation = function(...operands) {
        this.operands = operands
    }
    AbstractOperation.prototype = {
        DELIMITER: " ",
        evaluate(...vars) {
            return this.operation(...this.operands.map(g => g.evaluate(...vars)))
        },
        toString() {
            return this.operands.concat(this.operator).join(this.DELIMITER)
        },
        prefix() {
            return '(' + this.operator + this.DELIMITER + this.operands.map(g => g.prefix()).join(this.DELIMITER) + ')'
        },
        postfix() {
            return '(' + this.operands.map(g => g.postfix()).join(this.DELIMITER) + this.DELIMITER + this.operator + ')'
        },
        diff(variable) {
            return this.derivative(...this.operands.map(f => [f, f.diff(variable)]))
        },
        constructor: AbstractOperation
    }
    return AbstractOperation;
}

const AbstractOperation = AbstractOperationFactory()

function defineOperation(operation, derivative, operator) {
    const Operation = function(...operands) {
        AbstractOperation.call(this, ...operands)
    }

    Operation.prototype = Object.create(AbstractOperation.prototype);
    Operation.prototype.operation = operation
    Operation.prototype.derivative = derivative
    Operation.prototype.operator = operator
    Operation.prototype.constructor = Operation

    SUPPORTED_OPERATORS.set(operator, Operation)

    return Operation
}

const Add = defineOperation(
    (f, g) => f + g,
    ([f, df], [g, dg]) => new Add(df, dg),
    '+'
)

const Subtract = defineOperation(
    (f, g) => f - g,
    ([f, df], [g, dg]) => new Subtract(df, dg),
    '-'
)
const Multiply = defineOperation(
    (f, g) => f * g,
    ([f, g], [df, dg]) => new Add(new Multiply(df, g), new Multiply(f, dg)),
    '*'
)
const getSquare = f => new Multiply(f, f)
const Divide = defineOperation(
    (f, g) => f / g,
    ([f, df], [g, dg]) => new Divide(new Subtract(
        new Multiply(df, g),
        new Multiply(f, dg)), getSquare(g)),
    '/'
)

const Negate = defineOperation(
    f => -f,
    ([f, df]) => new Negate(df),
    'negate'
)

const Cube = defineOperation(
    f => f ** 3,
    ([f, df]) => new Multiply(new Multiply(Const.THREE, getSquare(f)), df),
    'cube'
)

const Cbrt = defineOperation(
    f => Math.cbrt(f),
    ([f, df]) => new Divide(df, new Multiply(Const.THREE, getSquare(new Cbrt(f)))),
    'cbrt'
)

const sumSquares = (...operands) => operands.map(arg => arg ** 2).reduce((acc, x) => acc + x, 0);

const sumsqDerivative = (...args) => {
    return args.map(([f, df]) => new Multiply(Const.TWO, new Multiply(f, df))).reduce(
        (acc, value) => new Add(acc, value), Const.ZERO
    )
}

const Sumsq = defineOperation(
    (...operands) => sumSquares(...operands),
    sumsqDerivative,
    "sumsq"
)


const Length = defineOperation(
    (...operands) => Math.sqrt(sumSquares(...operands)),
    (...args) => args.length === 0 ? Const.ZERO : new Divide(
        sumsqDerivative(...args),
        new Multiply(Const.TWO, new Length(...args.map(f => f[0])))
    ),
    "length"
)

function parse(expression) {
    const tokens = expression.trim().split(/\s+/)
    let stack = []
    for (const token of tokens) {
        let operand
        if (SUPPORTED_OPERATORS.has(token)) {
            const operator = SUPPORTED_OPERATORS.get(token)
            const arity = operator.prototype.operation.length
            operand = new operator(...stack.splice(stack.length - arity, arity))
        } else if (SUPPORTED_VARIABLES_NAMES.indexOf(token) !== -1) {
            operand = new Variable(token)
        } else {
            operand = new Const(parseFloat(token))
        }
        stack.push(operand)
    }
    return stack.pop()
}


/* Homework 7. Exceptions */

function AbstractParseErrorFactory() {
    const ParseError = function(message) {
        this.message = message
    }
    ParseError.prototype = Error.prototype
    ParseError.prototype.name = "ParseError"
    ParseError.prototype.constructor = ParseError
    return ParseError
}

const AbstractParseError = AbstractParseErrorFactory()

function ParseErrorFactory(name, createMessage) {
    const parseError = function(...args) {
        AbstractParseError.call(this, createMessage(...args))
    }
    parseError.prototype = AbstractParseError.prototype
    parseError.prototype.createMessage = createMessage
    parseError.prototype.name = name
    return parseError
}

const UnexpectedCharError = ParseErrorFactory("UnexpectedCharError",
    (c, ch, i) => "Expected '" + c + "', found '" + ch + "' in position " + i
)
const EmptyInputError = ParseErrorFactory("EmptyInputError", () => "Expression expected")
const EofExpectError = ParseErrorFactory("EofExpectError", () => "EOF expected")
const MissedTokenError = ParseErrorFactory("MissedTokenError", () => "Token expected")
const UnknownOperatorError = ParseErrorFactory("UnknownOperatorError",
    (operation) => "Unknown operation '" + operation + "'")
const UnexpectedTokenError = ParseErrorFactory("UnexpectedTokenError",
    (token) => "Unexpected token '" + token + "'")
const ArityError = ParseErrorFactory("ArityError",
    (sign, n, m) => "Operator '" + sign + "' requires " + n + " arguments, found " + m)
const InvalidParseModeError = ParseErrorFactory(
    "InvalidParseModeError", (mode) => "Invalid mode " + mode)

function isSpace(c) {
    return /\s/.test(c)
}

function isBracket(c) {
    return /[()]/.test(c)
}

function StringSourceFactory() {
    const stringSource = function(source) {
        this.source = String(source)
        this.i = 0
    }
    stringSource.prototype = {
        hasNext: function() {
            return this.i < this.source.length
        },
        next: function() {
            return this.source[this.i++]
        },
        constructor: stringSource,
        name: "StringSource"
    }
    return stringSource;
}

const StringSource = StringSourceFactory()

function BaseParserFactory() {
    const BaseParser = function(source) {
        this.source = new StringSource(String(source))
        this.ch = 0xffff
    }
    BaseParser.EOF_CHARACTER = '\0'
    BaseParser.prototype = {
        nextChar() {
            this.ch = this.source.hasNext() ? this.source.next() : BaseParser.EOF_CHARACTER;
        },
        nextToken(isSuitable) {
            this.skipWhitespaces()
            const token = Array()
            while (isSuitable(this.ch) && !this.eof()) {
                token.push(this.ch)
                this.nextChar()
            }
            return token.join('')
        },
        test(c) {
            if (this.ch === c) {
                this.nextChar()
                return true
            }
            return false
        },
        expect(c) {
            if (!this.test(c)) {
                throw new UnexpectedCharError(c, this.ch, this.source.i)
            }
        },
        skipWhitespaces() {
            while (isSpace(this.ch)) {
                this.nextChar()
            }
        },
        eof() {
            return this.ch === BaseParser.EOF_CHARACTER
        },
        constructor: BaseParser,
        name: "BaseParser"
    }
    return BaseParser;
}

const BaseParser = BaseParserFactory()

function ParserFactory(mode) {
    const parser = function(source) {
        BaseParser.call(this, source)
        this.nextChar()
    }
    if ([ParserFactory.PREFIX_MODE, ParserFactory.POSTFIX_MODE].indexOf(mode) === -1) {
        throw new InvalidParseModeError(mode);
    }
    parser.prototype.mode = mode
    parser.prototype = Object.create(BaseParser.prototype)
    parser.prototype.parse = function() {
        const f = this.parseExpression()
        if (f === null) {
            throw new EmptyInputError();
        }
        this.skipWhitespaces()
        if (!this.eof()) {
            throw new EofExpectError()
        }
        return f
    }
    parser.prototype.getNextToken = function() {
        return this.nextToken(c => !isSpace(c) && !isBracket(c));
    }
    parser.prototype.parseExpression = function() {
        this.skipWhitespaces()
        if (this.test('(')) {
            const tokens = []
            let expression
            while ((expression = this.parseExpression()) !== null) {
                tokens.push(expression);
            }
            this.expect(')')
            return this.buildOperation(tokens)
        }
        const token = this.getNextToken();
        if (token.length === 0) {
            return null
        } else {
            return this.processToken(token)
        }
    }
    parser.prototype.processToken = function(token) {
        if (token === '') {
            throw new MissedTokenError()
        }
        if (SUPPORTED_OPERATORS.has(token)) {
            return SUPPORTED_OPERATORS.get(token)
        } else if (!isNaN(token)) {
            return new Const(parseFloat(token))
        } else if (SUPPORTED_VARIABLES_NAMES.indexOf(token) > -1) {
            return new Variable(token)
        } else {
            throw new UnknownOperatorError(token)
        }
    }
    parser.prototype.buildOperation = function(tokens) {
        if (tokens.length === 0) {
            throw new MissedTokenError();
        }

        const f = this.mode === ParserFactory.PREFIX_MODE ? tokens.shift() : tokens.pop()
            // is constructor ?
        if (typeof f !== "function") {
            throw new UnknownOperatorError(f);
        }

        // other isn't constructor ?
        if (tokens.some(f => typeof f === "function")) {
            throw new UnexpectedTokenError()
        }

        const arity = f.prototype.operation.length
        if (arity > 0 && arity !== tokens.length) {
            throw new ArityError(f.prototype.operator, arity, tokens.length)
        }

        return new f(...tokens)
    }
    parser.prototype.constructor = parser
    parser.prototype.name = mode + "Parser"

    return parser
}
ParserFactory.PREFIX_MODE = 0;
ParserFactory.POSTFIX_MODE = 1;

const PrefixParser = ParserFactory(ParserFactory.PREFIX_MODE)
const PostfixParser = ParserFactory(ParserFactory.POSTFIX_MODE)

function parsePrefix(prefixFormExpression) {
    const parser = new PrefixParser(prefixFormExpression)
    return parser.parse()
}

function parsePostfix(postfixFormExpression) {
    const parser = new PostfixParser(postfixFormExpression)
    return parser.parse()
}