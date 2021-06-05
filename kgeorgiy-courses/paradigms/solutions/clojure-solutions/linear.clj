; Title task: "Homework 8. Linear algebra on Clojure"
; Author:     Igor Podtsepko (i.podtsepko@outlook.com)

(defn valid-vector? [vector] (and (vector? vector) (every? number? vector)))
(defn one-length-vectors? [& vectors] (apply == (mapv count vectors)))
(defn valid-vector-set? [vectors] (and (every? valid-vector? vectors) (apply one-length-vectors? vectors)))

(defn valid-matrix? [matrix] (and (vector? matrix) (valid-vector-set? matrix)))
(defn one-size-matrices? [& matrices] (and (apply = (mapv count matrices)) (one-length-vectors? (mapv first matrices))))
(defn valid-matrix-set? [matrices] (and (every? valid-matrix? matrices) (apply one-size-matrices? matrices)))

(defn vector-coordinate [f]
  (fn [& vectors] {:pre  [(valid-vector-set? vectors)]
                   :post [(one-length-vectors? (vector (first vectors) %)) (valid-vector? %)]}
    (apply (partial mapv f) vectors)))

(def v+ (vector-coordinate +))
(def v- (vector-coordinate -))
(def v* (vector-coordinate *))
(def vd (vector-coordinate /))

(defn scalar [& vectors] {:pre  [(valid-vector-set? vectors)]
                          :post [(number? %)]}
  (apply + (apply v* vectors)))

(defn vect [& vectors] {:pre  [(valid-vector-set? vectors) (= 3 (count (first vectors)))]
                        :post [(valid-vector? %) (= 3 (count %))]}
  (let [multiple-coordinates (fn [x y i j] (* (nth x i) (nth y j)))
        get-component-helper (fn [x y i j] (- (multiple-coordinates x y i j) (multiple-coordinates x y j i)))
        component (fn [x y i] (get-component-helper x y (rem (+ i 1) 3) (rem (+ i 2) 3)))]
    (reduce (fn [x y] (vector (component x y 0) (component x y 1) (component x y 2))) vectors)))

(defn v*s [vector & scalars] {:pre  [(valid-vector? vector) (every? number? scalars)]
                              :post [(valid-vector? %) (one-length-vectors? vector %)]}
  (mapv (partial * (apply * scalars)) vector))

(defn matrix-coordinate [f]
  (fn [& matrices] {:pre  [(valid-matrix-set? matrices)]
                    :post [(one-size-matrices? (vector (first matrices) %)) (valid-matrix? %)]}
    (apply (partial mapv f) matrices)))

(def m+ (matrix-coordinate v+))
(def m- (matrix-coordinate v-))
(def m* (matrix-coordinate v*))
(def md (matrix-coordinate vd))

(defn transpose [matrix] {:pre [(valid-matrix? matrix)]}
  (apply mapv vector matrix))

(defn m*s [matrix & scalars] {:pre  [(valid-matrix? matrix) (every? number? scalars)]
                              :post [(valid-matrix? %) (one-size-matrices? matrix %)]}
  (let [product (apply * scalars)
        component #(v*s % product)]
    (mapv component matrix)))

(defn m*v [matrix & vectors] {:pre [(valid-matrix? matrix)
                                    (valid-vector-set? vectors)
                                    (one-length-vectors? (first matrix) (first vectors))]}
  (let [product (apply v* vectors)
        component #(scalar % product)]
    (mapv component matrix)))

(defn m*m [& matrices] {:pre [(every? valid-matrix? matrices)
                              (every? true? (map (fn [A, B] (= (count (first A)) (count B)))
                                              matrices (subvec (vector matrices) 1)))]}
  (reduce (fn [A B] (mapv (fn [u] (mapv (fn [v] (apply + (v* u v))) (transpose B))) A)) matrices))

(defn valid-simplex? [simplex]
  (and (vector? simplex)
       (or (valid-vector? simplex)
           (and
             (== (count (first simplex)) (count simplex))
             (valid-simplex? (first simplex))
             (valid-simplex? (apply vector (rest simplex)))))))

(defn valid-simplex-set? [simplexes]
  (or (valid-vector-set? simplexes)
      (and (every? valid-simplex? simplexes)
           (apply = (mapv count simplexes)))))

(defn simplex-coordinate [f]
  (fn x [& simplexes] {:pre [(valid-simplex-set? simplexes)]}
    (if (valid-vector? (first simplexes))
      (apply (vector-coordinate f) simplexes)
      (apply mapv x simplexes))))

(def x+ (simplex-coordinate +))
(def x- (simplex-coordinate -))
(def x* (simplex-coordinate *))
(def xd (simplex-coordinate /))