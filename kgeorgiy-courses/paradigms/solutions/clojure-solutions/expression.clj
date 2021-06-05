; Title task: "Homework 9, 10, 11. Functional and object expressions, combinator parser"
; Author:     Igor Podtsepko (i.podtsepko@outlook.com)

; < FUNCTIONAL EXPRESSIONS >
(def constant constantly)
(defn variable [name] (fn [variable-map] (variable-map name)))

(defn abstract-operation [f]
  (fn [& operands]
    (fn [variable-map] (apply f (map (fn [operand] (operand variable-map)) operands)))))

(defn divide-implementation
  ([x] (divide-implementation 1 x))
  ([x & xs] (/ x (double (apply * xs)))))

(defn avg-implementation [& operands] (/ (apply + operands) (count operands)))

(def add (abstract-operation +))
(def subtract (abstract-operation -))
(def multiply (abstract-operation *))
(def divide (abstract-operation divide-implementation))
(def negate (abstract-operation #(- %)))
(def cos (abstract-operation #(Math/cos %)))
(def sin (abstract-operation #(Math/sin %)))
(def avg (abstract-operation avg-implementation))
(def sum add)

(def functional-operations-map
  {'sin    sin
   'cos    cos
   'negate negate
   '+      add
   '-      subtract
   '*      multiply
   '/      divide
   'sum    sum
   'avg    avg})

(defn parse-implementation [operation-map constant-type variable-type]
  (letfn [(process-token [token]
            (cond
              (number? token) (constant-type token)
              (symbol? token) (variable-type (str token))
              (seq? token) (apply (operation-map (first token)) (map process-token (pop token)))))]
    (comp process-token read-string)))

(def parseFunction (parse-implementation functional-operations-map constant variable))

; < OBJECT EXPRESSIONS >
(load-file "proto.clj")

(def evaluate (method :evaluate))
(def toString (method :toString))
(def diff (method :diff))
(def toStringInfix (method :toStringInfix))

; <----CONSTANT---->
(def _value (field :value))

(declare ZERO)

(def Constant
  (constructor
    (fn [this value] (assoc this :value value))
    {:evaluate      (fn [this _] (_value this))
     :toString      (fn [this] (format "%.1f" (double (_value this))))
     :diff          (fn [_ _] ZERO)
     :toStringInfix (fn [this] (toString this))}))

(def ZERO (Constant 0))
(def ONE (Constant 1))
(def TWO (Constant 2))

; <----VARIABLE---->
(def _name (field :name))

(def Variable
  (constructor
    (fn [this name] (assoc this :name name))
    {:evaluate      (fn [this variable-map] (variable-map (clojure.string/lower-case (str (first (_name this))))))
     :toString      (fn [this] (_name this))
     :diff          (fn [this variable-name] (if (= (_name this) variable-name) ONE ZERO))
     :toStringInfix (fn [this] (toString this))}))

; <----OPERATIONS---->
(def _function (field :function))
(def _operator (field :operators))
(def _derivative (field :derivative))
(def _operands (field :operands))

(def abstract-operation
  (constructor
    ; constructor-implementation
    (fn [this operator function derivative]
      (assoc this :function function :operators operator :derivative derivative))
    ; prototype
    {:evaluate      (fn [this variable-map] (apply (_function this) (mapv #(evaluate % variable-map) (_operands this))))
     :toString      (fn [this] (str "(" (_operator this) " " (clojure.string/join " " (mapv toString (_operands this))) ")"))
     :diff          (fn [this variable] ((_derivative this) (_operands this) (mapv #(diff % variable) (_operands this))))
     :toStringInfix (fn [this]
                      (let [[x y] (mapv toStringInfix (_operands this))
                            op (_operator this)]
                        (if (boolean y)
                          (format "(%s %s %s)" x op y)
                          (format "%s(%s)" op x))))}))

(defn operation-factory [function, operator, derivative]
  (constructor
    (fn [this & operands] (assoc this :operands operands))
    (abstract-operation function operator derivative)))

(def Add (operation-factory '+ + (fn [_ df] (apply Add df))))
(def Subtract (operation-factory '- - (fn [_ df] (apply Subtract df))))
(def Negate (operation-factory 'negate - (fn [_ [df]] (Negate df))))


(declare Multiply)

(defn multiply'-helper [wrap [f & fs] [df & dfs]]
  (if (empty? fs)
    (wrap df)
    (recur #(wrap (Add (apply Multiply df fs) (Multiply f %))) fs dfs)))

(def multiply' (partial multiply'-helper identity))

(def Multiply (operation-factory '* * multiply'))

(def IPow (operation-factory '** #(Math/pow %1 %2) identity))
(def ILog (operation-factory (symbol "//") #(/ (Math/log (Math/abs %2)) (Math/log (Math/abs %1))) identity))

(def Square #(Multiply % %))

(declare Divide)

(defn divide' [[f & fs] [df & dfs]]
  (if (empty? fs)
    (Divide (Negate df) (Square f))
    (let [g (apply Multiply fs)
          dg (multiply' fs dfs)]
      (Divide (Subtract (Multiply df g) (Multiply f dg)) (Square g)))))

(def Divide (operation-factory '/ divide-implementation divide'))

(def Sum (operation-factory 'sum + (fn [_ df] (apply Sum df))))

(def Avg (operation-factory 'avg avg-implementation (fn [_ df] (Divide (apply Add df) (Constant (count df))))))

(def object-operations-map
  {'+            Add
   '-            Subtract
   'negate       Negate
   '*            Multiply
   '/            Divide
   'sum          Sum
   'avg          Avg
   '**           IPow
   (symbol "//") ILog})

(def parseObject (parse-implementation object-operations-map Constant Variable))

; < PARSER >
(load-file "parser.clj")
(defn +string [s] (+str (apply +seq (map (comp +char str) (char-array s)))))

(def *digit (+char "0123456789"))

(def *space (+char " \t\n\r"))
(def *ws (+ignore (+star *space)))
(defn *in-ws [p] (+seqn 0 *ws p *ws))

(def *all-chars (mapv char (range 32 128)))
(def *letter (+char (apply str (filter #(Character/isLetter %) *all-chars))))

(def *uint (+str (+star *digit)))
(def *cnst
  (+seqf (comp Constant read-string str) (+opt (+char "+-")) *digit *uint (+opt (+char ".")) *uint))
(def *var (+map Variable (+str (+plus (+char "xyzXYZ")))))

(def unary-operators ["negate"])

(def l-assoc true)
(def r-assoc false)

(defn depth [operators assoc] {:operators operators :assoc assoc})
(def depths
  [(depth ["+" "-"] l-assoc)
   (depth ["*" "/"] l-assoc)
   (depth ["**" "//"] r-assoc)])
(def max-depth (count depths))

(defn *operation [ops] (+map #(object-operations-map (symbol %)) (apply +or (map +string ops))))

(declare *expr)
(def *brackets (+seqn 1 (+char "(") #'*expr (+char ")")))

(declare *operand)
(def *unary (+seqf #(%1 %2) (*operation unary-operators) #'*operand))

(def *operand (*in-ws (+or *cnst *var *brackets *unary)))

(defn build [l-assoc?]
  (fn [tokens] (let [[first-token & other-tokens] (if l-assoc? tokens (reverse tokens))]
                 (reduce (fn [x [op y]] (apply op (if l-assoc? [x y] [y x])))
                         first-token (vec (partition 2 other-tokens))))))

(defn *depth [i]
  (if (< i max-depth)
    (let [*arg (*depth (inc i))
          cur (nth depths i)
          fst *arg
          other (+map (partial apply concat)
                      (+star (+seq (*operation (cur :operators)) *arg)))]
      (+map (build (cur :assoc)) (+seqf cons fst other)))
    *operand))

(def *expr (*depth 0))
(defn parseObjectInfix [str] (-value (*expr str)))